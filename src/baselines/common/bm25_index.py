"""Shared BM25 index for all baselines.

Wraps graphrag.bm25.BM25Index, built once over the full corpus.
Caches top-K results to disk to avoid redundant 45-min computation.
"""
import json
import logging
import time
from pathlib import Path

from coliee_task1.stages.bm25 import BM25Index

logger = logging.getLogger(__name__)

CACHE_PATH = Path("output/baselines/bm25_candidates.json")


def build_shared_bm25(corpus: dict[str, str], top_k: int = 200) -> tuple[
    BM25Index, dict[str, list[tuple[str, float]]]
]:
    """Build BM25 index and compute top-K candidates for all documents.

    Caches candidates to disk. Loads from cache if corpus size matches.

    Returns:
        (bm25_index, candidates_cache)
        candidates_cache: {doc_id: [(cand_id, score), ...]} top_k results
    """
    doc_ids = sorted(corpus.keys())
    texts = [corpus[d] for d in doc_ids]

    bm25 = BM25Index()
    bm25.fit(doc_ids, texts)

    # Try loading from disk cache
    if CACHE_PATH.exists():
        try:
            raw = json.loads(CACHE_PATH.read_text())
            if raw.get("n_docs") == len(doc_ids) and raw.get("top_k") == top_k:
                candidates = {
                    k: [(cid, score) for cid, score in v]
                    for k, v in raw["candidates"].items()
                }
                logger.info("Loaded BM25 candidates from cache (%d docs)", len(candidates))
                return bm25, candidates
            else:
                logger.info("Cache stale (n_docs=%s vs %d), recomputing",
                            raw.get("n_docs"), len(doc_ids))
        except Exception as e:
            logger.warning("Failed to load BM25 cache: %s", e)

    # Compute from scratch
    logger.info("Pre-computing BM25 top-%d for %d documents...", top_k, len(doc_ids))
    t0 = time.time()
    candidates = {}
    for i, qid in enumerate(doc_ids):
        results = bm25.query(corpus[qid], top_k=top_k + 1)
        candidates[qid] = [(did, s) for did, s in results if did != qid][:top_k]
        if (i + 1) % 2000 == 0:
            logger.info("  BM25 candidates: %d/%d (%.1fs)",
                        i + 1, len(doc_ids), time.time() - t0)

    logger.info("BM25 candidates computed in %.1fs", time.time() - t0)

    # Save to disk cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "n_docs": len(doc_ids),
        "top_k": top_k,
        "candidates": {k: v for k, v in candidates.items()},
    }
    CACHE_PATH.write_text(json.dumps(cache_data))
    logger.info("Saved BM25 candidates cache to %s", CACHE_PATH)

    return bm25, candidates
