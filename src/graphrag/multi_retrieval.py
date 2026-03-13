"""BGE-M3 multi-signal retrieval: dense + sparse + ColBERT from a single model.

Produces 4 features per (query, candidate) pair:
  - m3_dense_score: dense embedding cosine similarity
  - m3_sparse_score: learned sparse (lexical match) score
  - m3_colbert_score: late-interaction (token-level) MaxSim score
  - m3_fused_score: weighted combination of the three signals

Reference: Chen et al., "M3-Embedding" (ACL Findings 2024, arXiv:2402.03216)
"""
import logging
import time
from pathlib import Path

# Compatibility shim: FlagEmbedding 1.3.5 uses is_torch_fx_available which was
# removed in transformers 5.x. Patch it before any FlagEmbedding import.
import transformers.utils.import_utils as _tutils
if not hasattr(_tutils, "is_torch_fx_available"):
    _tutils.is_torch_fx_available = lambda: False

import numpy as np

from graphrag.config import (
    BGE_M3_MODEL,
    BGE_M3_BATCH_SIZE,
    BGE_M3_MAX_LENGTH,
    BGE_M3_WEIGHTS,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def fuse_multi_scores(
    dense: dict[str, float],
    sparse: dict[str, float],
    colbert: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fuse dense, sparse, and ColBERT scores with weighted combination.

    Missing scores for a document in any signal are treated as 0.
    """
    if weights is None:
        weights = BGE_M3_WEIGHTS

    all_keys = set(dense) | set(sparse) | set(colbert)
    fused = {}
    for key in all_keys:
        fused[key] = (
            weights["dense"] * dense.get(key, 0.0)
            + weights["sparse"] * sparse.get(key, 0.0)
            + weights["colbert"] * colbert.get(key, 0.0)
        )
    return fused


def extract_multi_features(
    query_id: str,
    candidate_id: str,
    multi_scores: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    """Extract per-pair features from pre-computed multi-signal scores.

    Args:
        query_id: Query document ID.
        candidate_id: Candidate document ID.
        multi_scores: {query_id: {signal_name: {candidate_id: score}}}.

    Returns:
        Dict with m3_dense_score, m3_sparse_score, m3_colbert_score, m3_fused_score.
    """
    q_scores = multi_scores.get(query_id, {})
    return {
        "m3_dense_score": q_scores.get("dense", {}).get(candidate_id, 0.0),
        "m3_sparse_score": q_scores.get("sparse", {}).get(candidate_id, 0.0),
        "m3_colbert_score": q_scores.get("colbert", {}).get(candidate_id, 0.0),
        "m3_fused_score": q_scores.get("fused", {}).get(candidate_id, 0.0),
    }


def encode_corpus_m3(
    corpus_texts: dict[str, str],
    model_name: str = BGE_M3_MODEL,
    batch_size: int = BGE_M3_BATCH_SIZE,
    max_length: int = BGE_M3_MAX_LENGTH,
) -> dict[str, dict]:
    """Encode entire corpus with BGE-M3, returning dense, sparse, ColBERT representations.

    Returns:
        {doc_id: {"dense": np.ndarray, "sparse": dict, "colbert": np.ndarray}}
    """
    from FlagEmbedding import BGEM3FlagModel

    logger.info("Loading BGE-M3 model: %s", model_name)
    model = BGEM3FlagModel(model_name, use_fp16=True)

    doc_ids = sorted(corpus_texts.keys())
    texts = [corpus_texts[d][:max_length * 4] for d in doc_ids]  # rough char limit

    logger.info("Encoding %d documents with BGE-M3 (batch_size=%d) ...", len(texts), batch_size)
    t0 = time.time()

    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    logger.info("BGE-M3 encoding complete in %.1f seconds", time.time() - t0)

    corpus_repr = {}
    for i, doc_id in enumerate(doc_ids):
        corpus_repr[doc_id] = {
            "dense": output["dense_vecs"][i],
            "sparse": output["lexical_weights"][i],
            "colbert": output["colbert_vecs"][i],
        }

    return corpus_repr


def score_candidates_m3(
    query_ids: list[str],
    candidate_lists: dict[str, list[str]],
    corpus_repr: dict[str, dict],
    weights: dict[str, float] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Score all (query, candidate) pairs using pre-computed BGE-M3 representations.

    Returns:
        {query_id: {"dense": {cid: score}, "sparse": {cid: score},
                     "colbert": {cid: score}, "fused": {cid: score}}}
    """
    if weights is None:
        weights = BGE_M3_WEIGHTS

    logger.info("Scoring candidates with BGE-M3 multi-signal ...")
    t0 = time.time()
    results = {}

    for qid in query_ids:
        if qid not in corpus_repr:
            continue

        q_repr = corpus_repr[qid]
        candidates = candidate_lists.get(qid, [])
        dense_scores = {}
        sparse_scores = {}
        colbert_scores = {}

        for cid in candidates:
            if cid not in corpus_repr:
                continue
            c_repr = corpus_repr[cid]

            # Dense: cosine similarity (vectors already normalized by BGE-M3)
            dense_scores[cid] = float(np.dot(q_repr["dense"], c_repr["dense"]))

            # Sparse: dot product of lexical weight dicts
            q_sparse = q_repr["sparse"]
            c_sparse = c_repr["sparse"]
            shared_tokens = set(q_sparse.keys()) & set(c_sparse.keys())
            sparse_scores[cid] = sum(
                q_sparse[t] * c_sparse[t] for t in shared_tokens
            )

            # ColBERT: MaxSim (max over candidate tokens per query token, then sum)
            q_colbert = q_repr["colbert"]  # (n_q_tokens, dim)
            c_colbert = c_repr["colbert"]  # (n_c_tokens, dim)
            if len(q_colbert) > 0 and len(c_colbert) > 0:
                sim_matrix = np.dot(q_colbert, c_colbert.T)
                max_sim = sim_matrix.max(axis=1)
                colbert_scores[cid] = float(max_sim.sum())
            else:
                colbert_scores[cid] = 0.0

        # Normalize scores to [0, 1] within each signal per query
        for scores_dict in [dense_scores, sparse_scores, colbert_scores]:
            if scores_dict:
                max_s = max(scores_dict.values())
                min_s = min(scores_dict.values())
                rng = max_s - min_s
                if rng > 0:
                    for k in scores_dict:
                        scores_dict[k] = (scores_dict[k] - min_s) / rng

        fused = fuse_multi_scores(dense_scores, sparse_scores, colbert_scores, weights)

        results[qid] = {
            "dense": dense_scores,
            "sparse": sparse_scores,
            "colbert": colbert_scores,
            "fused": fused,
        }

    logger.info("BGE-M3 scoring complete in %.1f seconds", time.time() - t0)
    return results
