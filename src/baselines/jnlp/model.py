"""JNLP 2025 baseline: BM25 + SAILER + LightGBM."""
import logging

import lightgbm as lgb
import numpy as np

from baselines.common.base_model import BaselineModel
from baselines.jnlp.features import (
    build_corpus_freq,
    extract_jnlp_features,
)
from baselines.jnlp.sailer_encoder import SAILEREncoder

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "bm25_full",
    "bm25_para_max",
    "qld_score",
    "sailer_sim",
    "bm25_rank",
    "bm25_ratio",
    "query_len",
    "doc_len",
]


class JNLPBaseline(BaselineModel):
    def __init__(self):
        self._lgb_model: lgb.Booster | None = None
        self._sailer: SAILEREncoder | None = None
        self._sailer_embeddings: dict[str, np.ndarray] = {}
        self._corpus_freq: dict[str, int] = {}
        self._total_terms: int = 0

    def name(self) -> str:
        return "JNLP 2025 (BM25+SAILER+LightGBM)"

    def train(self, corpus, train_queries, labels, bm25_candidates=None):
        if bm25_candidates is None:
            raise ValueError("JNLP baseline requires pre-computed BM25 candidates")

        # 1. Encode corpus with SAILER
        self._sailer = SAILEREncoder()
        self._sailer_embeddings = self._sailer.encode_corpus(corpus)

        # 2. Build corpus frequencies for QLD
        self._corpus_freq, self._total_terms = build_corpus_freq(corpus)

        # 3. Build training data from BM25 candidates (with negative subsampling)
        logger.info("Extracting training features for %d queries...", len(train_queries))
        rng = np.random.RandomState(42)
        max_neg_ratio = 10  # keep at most 10x negatives per positive
        max_neg_per_query = 50

        X_rows = []
        y_rows = []
        for qi, qid in enumerate(train_queries):
            positives = set(labels.get(qid, []))
            cands = bm25_candidates.get(qid, [])

            pos_cands = [(c, s) for c, s in cands if c in positives]
            neg_cands = [(c, s) for c, s in cands if c not in positives]

            # Subsample negatives
            n_neg = min(len(neg_cands), max(len(pos_cands) * max_neg_ratio, 1), max_neg_per_query)
            if len(neg_cands) > n_neg:
                idx = rng.choice(len(neg_cands), n_neg, replace=False)
                neg_cands = [neg_cands[i] for i in idx]

            for cand_id, _ in pos_cands + neg_cands:
                feats = extract_jnlp_features(
                    qid, cand_id, corpus, bm25_candidates,
                    self._sailer_embeddings, self._corpus_freq, self._total_terms,
                )
                X_rows.append([feats[f] for f in FEATURE_NAMES])
                y_rows.append(1 if cand_id in positives else 0)
            if (qi + 1) % 500 == 0:
                logger.info("  Features: %d/%d queries", qi + 1, len(train_queries))

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int32)
        logger.info("Training set: %d samples, %d positive (%.2f%%)",
                     len(y), y.sum(), 100 * y.mean())

        # 4. Train LightGBM
        train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
        params = {
            "objective": "binary",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self._lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
        )
        logger.info("LightGBM trained: %d trees", self._lgb_model.num_trees())

    def predict(self, query_id, corpus, bm25_candidates=None):
        if self._lgb_model is None:
            raise RuntimeError("Model not trained")
        if bm25_candidates is None:
            return []

        cands = bm25_candidates if isinstance(bm25_candidates, list) else []
        if not cands:
            return []

        # Build all-candidates dict for feature extraction
        all_cands = {query_id: cands}
        X_rows = []
        cand_ids = []
        for cand_id, _ in cands:
            feats = extract_jnlp_features(
                query_id, cand_id, corpus, all_cands,
                self._sailer_embeddings, self._corpus_freq, self._total_terms,
            )
            X_rows.append([feats[f] for f in FEATURE_NAMES])
            cand_ids.append(cand_id)

        X = np.array(X_rows, dtype=np.float32)
        scores = self._lgb_model.predict(X)

        results = list(zip(cand_ids, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
