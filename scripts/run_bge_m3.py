"""Run BGE-M3 encoding + scoring standalone using cached pipeline data.

Memory-efficient two-phase approach:
  Phase 1: Encode all docs → keep dense (1024-dim) + sparse (small dicts)
  Phase 2: For each query, encode query+candidates with ColBERT, score, discard vectors

This avoids storing 9,556 × 8192 × 1024 ColBERT matrices in RAM simultaneously.

Note: pickle is used intentionally here for compatibility with the
existing pipeline cache format (internal data only, not untrusted).
"""
import gc
import logging
import pickle  # noqa: S403 — internal cache only, not untrusted data
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("output/pipeline_cache")

# Reduced max_length for ColBERT phase to control memory
COLBERT_MAX_LENGTH = 1024
COLBERT_BATCH_SIZE = 32


def main():
    total_t0 = time.time()

    # Load cached data (internal pipeline cache, safe to unpickle)
    logger.info("Loading stage1 cache (clean_corpus) ...")
    with open(CACHE_DIR / "stage1.pkl", "rb") as f:
        raw_corpus, clean_corpus, contexts = pickle.load(f)  # noqa: S301
    logger.info("Loaded %d documents", len(clean_corpus))

    logger.info("Loading stage2 cache (rrf_results) ...")
    with open(CACHE_DIR / "stage2.pkl", "rb") as f:
        stage2 = pickle.load(f)  # noqa: S301
    rrf_results = stage2[0]
    query_ids = sorted(rrf_results.keys())
    logger.info("Loaded %d queries with RRF candidates", len(query_ids))

    del raw_corpus, contexts, stage2
    gc.collect()

    # Build candidate lists
    candidate_lists = {
        qid: [cid for cid, _ in rrf_results.get(qid, [])]
        for qid in query_ids
    }

    # ── Phase 1: Dense + Sparse encoding (small memory footprint) ──
    import transformers.utils.import_utils as _tutils
    if not hasattr(_tutils, "is_torch_fx_available"):
        _tutils.is_torch_fx_available = lambda: False

    from FlagEmbedding import BGEM3FlagModel

    logger.info("Loading BGE-M3 model: BAAI/bge-m3")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    doc_ids = sorted(clean_corpus.keys())
    texts = [clean_corpus[d][:8192 * 4] for d in doc_ids]

    logger.info("=== Phase 1: Dense + Sparse encoding (%d docs) ===", len(texts))
    t0 = time.time()

    output = model.encode(
        texts,
        batch_size=8,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,  # Skip ColBERT to save memory
    )

    logger.info("Phase 1 complete in %.1f seconds", time.time() - t0)

    # Store dense + sparse per doc
    dense_vecs = {}
    sparse_vecs = {}
    for i, doc_id in enumerate(doc_ids):
        dense_vecs[doc_id] = output["dense_vecs"][i]
        sparse_vecs[doc_id] = output["lexical_weights"][i]

    del output
    gc.collect()
    logger.info("Stored dense (%d) + sparse (%d) representations",
                len(dense_vecs), len(sparse_vecs))

    # ── Phase 2: ColBERT scoring per query (encode on-the-fly) ──
    logger.info("=== Phase 2: ColBERT per-query scoring ===")
    t0 = time.time()

    # Collect all unique doc IDs needed for ColBERT
    needed_docs = set()
    for qid in query_ids:
        needed_docs.add(qid)
        for cid in candidate_lists.get(qid, []):
            needed_docs.add(cid)
    needed_docs = sorted(needed_docs)
    logger.info("Need ColBERT vectors for %d unique documents", len(needed_docs))

    # Encode in chunks to get ColBERT vectors
    colbert_vecs = {}
    chunk_size = 200  # encode 200 docs at a time
    for chunk_start in range(0, len(needed_docs), chunk_size):
        chunk_ids = needed_docs[chunk_start:chunk_start + chunk_size]
        chunk_texts = [clean_corpus[d][:COLBERT_MAX_LENGTH * 4] for d in chunk_ids]

        chunk_output = model.encode(
            chunk_texts,
            batch_size=COLBERT_BATCH_SIZE,
            max_length=COLBERT_MAX_LENGTH,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )

        for i, doc_id in enumerate(chunk_ids):
            colbert_vecs[doc_id] = chunk_output["colbert_vecs"][i]

        del chunk_output
        gc.collect()

        done = min(chunk_start + chunk_size, len(needed_docs))
        logger.info("  ColBERT encoded %d/%d docs", done, len(needed_docs))

    logger.info("Phase 2 encoding complete in %.1f seconds", time.time() - t0)

    # ── Phase 3: Score all queries ──
    logger.info("=== Phase 3: Scoring all query-candidate pairs ===")
    t0 = time.time()

    from graphrag.multi_retrieval import fuse_multi_scores
    from graphrag.config import BGE_M3_WEIGHTS

    results = {}
    for qi, qid in enumerate(query_ids):
        if qid not in dense_vecs:
            continue

        candidates = candidate_lists.get(qid, [])
        dense_scores = {}
        sparse_scores = {}
        colbert_scores = {}

        for cid in candidates:
            if cid not in dense_vecs:
                continue

            # Dense cosine similarity
            dense_scores[cid] = float(np.dot(dense_vecs[qid], dense_vecs[cid]))

            # Sparse dot product
            q_sp = sparse_vecs[qid]
            c_sp = sparse_vecs[cid]
            shared = set(q_sp.keys()) & set(c_sp.keys())
            sparse_scores[cid] = sum(q_sp[t] * c_sp[t] for t in shared)

            # ColBERT MaxSim
            if qid in colbert_vecs and cid in colbert_vecs:
                q_cb = colbert_vecs[qid]
                c_cb = colbert_vecs[cid]
                if len(q_cb) > 0 and len(c_cb) > 0:
                    sim_matrix = np.dot(q_cb, c_cb.T)
                    colbert_scores[cid] = float(sim_matrix.max(axis=1).sum())
                else:
                    colbert_scores[cid] = 0.0
            else:
                colbert_scores[cid] = 0.0

        # Normalize to [0, 1] per signal
        for scores_dict in [dense_scores, sparse_scores, colbert_scores]:
            if scores_dict:
                max_s = max(scores_dict.values())
                min_s = min(scores_dict.values())
                rng = max_s - min_s
                if rng > 0:
                    for k in scores_dict:
                        scores_dict[k] = (scores_dict[k] - min_s) / rng

        fused = fuse_multi_scores(dense_scores, sparse_scores, colbert_scores, BGE_M3_WEIGHTS)
        results[qid] = {
            "dense": dense_scores,
            "sparse": sparse_scores,
            "colbert": colbert_scores,
            "fused": fused,
        }

        if (qi + 1) % 200 == 0:
            logger.info("  Scored %d/%d queries", qi + 1, len(query_ids))

    logger.info("Scoring complete in %.1f seconds", time.time() - t0)

    # Save cache
    out_path = CACHE_DIR / "stage3_m3.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)  # noqa: S301
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info("Saved stage3_m3.pkl (%.1f MB)", size_mb)

    total_time = time.time() - total_t0
    logger.info("BGE-M3 complete in %.1f minutes", total_time / 60)
    logger.info("Queries scored: %d", len(results))
    if results:
        sample_qid = next(iter(results))
        sample_signals = {k: len(v) for k, v in results[sample_qid].items()}
        logger.info("Sample query %s signals: %s", sample_qid, sample_signals)


if __name__ == "__main__":
    main()
