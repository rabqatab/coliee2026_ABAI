"""GraphRAG Option C adapter — wraps our pipeline as a BaselineModel.

Loads pre-trained meta-learner fold models from output/models_v2/meta_learner/
and uses them for prediction. Falls back to BM25-only if models aren't available.
"""
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np

from baselines.common.base_model import BaselineModel
from coliee_task1.stages.bm25 import BM25Index
from coliee_task1.stages.graphrag import GraphRAGLite

logger = logging.getLogger(__name__)

MODELS_DIR = Path("output/models_v2/meta_learner")


class GraphRAGAdapter(BaselineModel):
    """Wraps the Option C pipeline for baseline comparison."""

    def __init__(self, top_k: int = 200):
        self._top_k = top_k
        self._bm25: BM25Index | None = None
        self._graphrag: GraphRAGLite | None = None
        self._fold_models: list[lgb.Booster] = []
        self._threshold: float = 0.5
        self._corpus = None

    def name(self) -> str:
        return "GraphRAG Option C (ours)"

    def train(self, corpus, train_queries, labels, bm25_candidates=None):
        self._corpus = corpus

        # Build BM25 index
        doc_ids = sorted(corpus.keys())
        self._bm25 = BM25Index()
        self._bm25.fit(doc_ids, [corpus[d] for d in doc_ids])

        # Build GraphRAG Lite entity index
        self._graphrag = GraphRAGLite()
        self._graphrag.fit(corpus)

        # Try loading pre-trained meta-learner fold models
        if MODELS_DIR.exists():
            config_path = MODELS_DIR / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text())
                self._threshold = config.get("threshold", 0.5)

            fold_files = sorted(MODELS_DIR.glob("fold_*.txt"))
            for fpath in fold_files:
                try:
                    booster = lgb.Booster(model_file=str(fpath))
                    self._fold_models.append(booster)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", fpath, e)

            if self._fold_models:
                # Check feature count matches FEATURE_COLS
                from coliee_task1.stages.meta_learner import FEATURE_COLS
                expected = self._fold_models[0].num_feature()
                if expected != len(FEATURE_COLS):
                    logger.warning(
                        "Meta-learner expects %d features but FEATURE_COLS has %d; "
                        "using BM25+GraphRAG fallback", expected, len(FEATURE_COLS))
                    self._fold_models = []
                else:
                    logger.info("Loaded %d fold models (threshold=%.3f)",
                                len(self._fold_models), self._threshold)
                    return

        logger.info("Using BM25+GraphRAG entity overlap fallback")

    def predict(self, query_id, corpus, bm25_candidates=None):
        # Get BM25 candidates
        if bm25_candidates:
            candidates = bm25_candidates
        else:
            results = self._bm25.query(corpus[query_id], top_k=self._top_k + 1)
            candidates = [(did, s) for did, s in results if did != query_id]

        if not candidates:
            return []

        # If we have fold models, use the full feature set
        if self._fold_models:
            return self._predict_with_meta_learner(query_id, candidates, corpus)

        # Fallback: BM25 score + GraphRAG entity overlap
        return self._predict_fallback(query_id, candidates, corpus)

    def _predict_with_meta_learner(self, query_id, candidates, corpus):
        """Predict using pre-trained LightGBM fold ensemble."""
        from coliee_task1.stages.meta_learner import FEATURE_COLS

        features = []
        cand_ids = []
        for cid, bm25_score in candidates:
            if cid == query_id:
                continue
            feats = self._build_simple_features(query_id, cid, bm25_score, corpus)
            features.append([feats.get(c, 0.0) for c in FEATURE_COLS])
            cand_ids.append(cid)

        if not features:
            return []

        X = np.array(features)
        # Average predictions across folds
        preds = np.zeros(len(X))
        for booster in self._fold_models:
            preds += booster.predict(X)
        preds /= len(self._fold_models)

        results = list(zip(cand_ids, preds.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self._top_k]

    def _predict_fallback(self, query_id, candidates, corpus):
        """Fallback: combine BM25 score with GraphRAG entity overlap."""
        results = []
        for cid, bm25_score in candidates:
            if cid == query_id:
                continue
            grag = self._graphrag.get_pair_features(query_id, cid) if self._graphrag else {}
            entity_score = grag.get("entity_overlap_score", 0.0)
            fused = 0.7 * bm25_score + 0.3 * entity_score
            results.append((cid, fused))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self._top_k]

    def _build_simple_features(self, query_id, candidate_id, bm25_score, corpus):
        """Build a simplified feature dict for meta-learner prediction.

        Only fills BM25 and GraphRAG features since we don't have
        bi-encoder/cross-encoder scores in the baseline comparison context.
        """
        feats = {
            "bm25_score": bm25_score,
            "bm25_rrf_score": bm25_score,  # approximate
            "biencoder_score": 0.0,
            "biencoder_rank": 999.0,
            "crossencoder_score": 0.0,
            "crossencoder_rank": 999.0,
            "tfidf_cosine": 0.0,
            "jaccard": 0.0,
            "shared_bigrams": 0.0,
            "length_ratio": 0.0,
            "shared_legal_terms": 0.0,
            "n_context_matches": 0.0,
            "max_context_bm25": 0.0,
        }

        # GraphRAG features
        if self._graphrag:
            grag = self._graphrag.get_pair_features(query_id, candidate_id)
            for k, v in grag.items():
                feats[k] = v

        return feats
