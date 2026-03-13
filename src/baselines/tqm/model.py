"""TQM 2024 baseline — LTR fusion of lexical + semantic signals.

Reproduces the winning approach from COLIEE 2024 Task 1 (arXiv 2404.00947):
    1. Fine-tune bi-encoder (all-MiniLM-L6-v2) with hard negatives
    2. Extract 10 features per (query, candidate) pair
    3. Train LightGBM LambdaMART ranker
"""
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np

from baselines.common.base_model import BaselineModel
from baselines.tqm.bi_encoder import TQMBiEncoder
from baselines.tqm.features import (
    FEATURE_NAMES,
    build_tfidf,
    extract_tqm_features,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("output/baselines/models")


class TQMBaseline(BaselineModel):
    """TQM 2024 (LTR Fusion): BM25 + bi-encoder features -> LambdaMART."""

    def __init__(self, top_k: int = 200):
        self._top_k = top_k
        self._bi_encoder = TQMBiEncoder()
        self._embeddings: dict[str, np.ndarray] = {}
        self._tfidf_vectors: dict = {}
        self._ranker: lgb.Booster | None = None
        self._ranker_path = MODEL_DIR / "tqm_lambdamart.pkl"

    def name(self) -> str:
        return "TQM 2024 (LTR Fusion)"

    def train(
        self,
        corpus: dict[str, str],
        train_queries: list[str],
        labels: dict[str, list[str]],
        bm25_candidates: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        """Train TQM pipeline: bi-encoder fine-tune, feature extraction, LambdaMART."""
        if bm25_candidates is None:
            raise ValueError("TQM baseline requires bm25_candidates")

        # 1. Fine-tune bi-encoder (or load cached)
        logger.info("Step 1/4: Fine-tuning bi-encoder")
        self._bi_encoder.finetune(corpus, train_queries, labels, bm25_candidates)

        # 2. Encode corpus embeddings
        logger.info("Step 2/4: Encoding corpus embeddings")
        self._embeddings = self._bi_encoder.encode_corpus(corpus)

        # 3. Build TF-IDF vectors
        logger.info("Step 3/4: Building TF-IDF vectors")
        _, self._tfidf_vectors = build_tfidf(corpus)

        # 4. Train LambdaMART
        logger.info("Step 4/4: Training LambdaMART ranker")
        self._train_lambdamart(corpus, train_queries, labels, bm25_candidates)

    def _train_lambdamart(
        self,
        corpus: dict[str, str],
        train_queries: list[str],
        labels: dict[str, list[str]],
        bm25_candidates: dict[str, list[tuple[str, float]]],
    ) -> None:
        """Train LightGBM LambdaMART on extracted features.

        Uses pickle for model serialization since LightGBM Booster objects
        cannot be serialized with JSON. The model file is only loaded from
        local trusted output — never from untrusted sources.
        """
        # Build feature matrix and labels (with negative subsampling per group)
        rng = np.random.RandomState(42)
        max_neg_ratio = 10
        max_neg_per_query = 50

        all_features = []
        all_labels = []
        group_sizes = []

        for qid in train_queries:
            candidates = bm25_candidates.get(qid, [])
            if not candidates:
                continue

            positives = set(labels.get(qid, []))

            pos_items = []
            neg_items = []
            for cid, _ in candidates[:self._top_k]:
                if cid == qid:
                    continue
                feats = extract_tqm_features(
                    qid,
                    cid,
                    corpus,
                    bm25_candidates,
                    self._embeddings,
                    self._tfidf_vectors,
                )
                feat_vec = [feats[fn] for fn in FEATURE_NAMES]
                lbl = 1 if cid in positives else 0
                if lbl == 1:
                    pos_items.append((feat_vec, lbl))
                else:
                    neg_items.append((feat_vec, lbl))

            # Subsample negatives
            n_neg = min(len(neg_items), max(len(pos_items) * max_neg_ratio, 1), max_neg_per_query)
            if len(neg_items) > n_neg:
                idx = rng.choice(len(neg_items), n_neg, replace=False)
                neg_items = [neg_items[i] for i in idx]

            group_items = pos_items + neg_items
            if group_items:
                all_features.extend([f for f, _ in group_items])
                all_labels.extend([l for _, l in group_items])
                group_sizes.append(len(group_items))

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.float32)

        logger.info(
            "LambdaMART training data: %d samples, %d groups, %d positives (%.1f%%)",
            len(y),
            len(group_sizes),
            int(y.sum()),
            100 * y.mean(),
        )

        train_data = lgb.Dataset(
            X,
            label=y,
            group=group_sizes,
            feature_name=FEATURE_NAMES,
            free_raw_data=False,
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [5, 10],
            "num_leaves": 63,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "verbose": -1,
        }

        self._ranker = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data],
            callbacks=[
                lgb.log_evaluation(period=100),
                lgb.early_stopping(stopping_rounds=50, verbose=True),
            ],
        )

        # Save model
        self._ranker_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._ranker_path, "wb") as f:
            pickle.dump(self._ranker, f)
        logger.info("Saved LambdaMART to %s", self._ranker_path)

    def predict(
        self,
        query_id: str,
        corpus: dict[str, str],
        bm25_candidates: list[tuple[str, float]] | None = None,
    ) -> list[tuple[str, float]]:
        """Predict relevance scores for candidates using LambdaMART.

        Returns sorted list of (candidate_id, score) descending by score.
        """
        if self._ranker is None:
            raise RuntimeError("Model not trained — call train() first")

        if bm25_candidates is None:
            return []

        # Build a full bm25_candidates dict with just this query for feature extraction
        bm25_dict = {query_id: bm25_candidates}

        features = []
        candidate_ids = []

        for cid, _ in bm25_candidates[:self._top_k]:
            if cid == query_id:
                continue
            feats = extract_tqm_features(
                query_id,
                cid,
                corpus,
                bm25_dict,
                self._embeddings,
                self._tfidf_vectors,
            )
            feat_vec = [feats[fn] for fn in FEATURE_NAMES]
            features.append(feat_vec)
            candidate_ids.append(cid)

        if not features:
            return []

        X = np.array(features, dtype=np.float32)
        scores = self._ranker.predict(X)

        results = list(zip(candidate_ids, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def predict_batch(
        self,
        query_ids: list[str],
        corpus: dict[str, str],
        bm25_candidates: dict[str, list[tuple[str, float]]] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Batch prediction — uses shared bm25_candidates for shared_top_k feature."""
        if self._ranker is None:
            raise RuntimeError("Model not trained — call train() first")

        if bm25_candidates is None:
            return {qid: [] for qid in query_ids}

        results = {}
        for qid in query_ids:
            cands = bm25_candidates.get(qid, [])
            if not cands:
                results[qid] = []
                continue

            features = []
            candidate_ids = []

            for cid, _ in cands[: self._top_k]:
                if cid == qid:
                    continue
                feats = extract_tqm_features(
                    qid,
                    cid,
                    corpus,
                    bm25_candidates,
                    self._embeddings,
                    self._tfidf_vectors,
                )
                feat_vec = [feats[fn] for fn in FEATURE_NAMES]
                features.append(feat_vec)
                candidate_ids.append(cid)

            if not features:
                results[qid] = []
                continue

            X = np.array(features, dtype=np.float32)
            scores = self._ranker.predict(X)

            ranked = list(zip(candidate_ids, scores.tolist()))
            ranked.sort(key=lambda x: x[1], reverse=True)
            results[qid] = ranked

        return results
