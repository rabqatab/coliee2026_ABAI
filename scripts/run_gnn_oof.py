"""Run GNN reranker OOF (Stage 5.5) standalone using cached pipeline data.

Trains 5 GNN models with GroupKFold cross-validation, producing proper OOF scores
where each query is scored only by a model that never saw its labels.

Fold assignments are deterministic (sorted query_ids + GroupKFold) and match
the meta-learner folds exactly.

Loads stages 1-4 from cache, builds corpus graph, trains 5 GAT models,
scores each validation fold, and saves result as stage5_5_oof.pkl.

Pickle is used intentionally for pipeline cache compatibility (internal data only).
"""
import gc
import json
import logging
import pickle  # noqa: S403 — internal pipeline cache
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("output/pipeline_cache")
PROJECT_ROOT = Path(__file__).parent.parent


def main():
    total_t0 = time.time()

    # Load cached data
    logger.info("Loading stage1 cache ...")
    with open(CACHE_DIR / "stage1.pkl", "rb") as f:
        _, clean_corpus, _ = pickle.load(f)  # noqa: S301
    logger.info("Loaded %d documents", len(clean_corpus))

    logger.info("Loading stage2 cache ...")
    with open(CACHE_DIR / "stage2.pkl", "rb") as f:
        stage2 = pickle.load(f)  # noqa: S301
    rrf_results = stage2[0]
    bm25_rrf = stage2[2]
    del stage2

    logger.info("Loading stage3 cache (biencoder) ...")
    with open(CACHE_DIR / "stage3.pkl", "rb") as f:
        biencoder_scores = pickle.load(f)  # noqa: S301

    logger.info("Loading stage4 cache (crossencoder) ...")
    with open(CACHE_DIR / "stage4.pkl", "rb") as f:
        crossencoder_scores = pickle.load(f)  # noqa: S301

    # Load labels
    labels_path = PROJECT_ROOT / "data" / "task1" / "task1_train_labels_2026.json"
    with open(labels_path) as f:
        labels = json.load(f)
    query_ids = sorted(labels.keys())
    logger.info("Loaded %d query labels", len(labels))

    gc.collect()

    # Import GNN components
    from graphrag.gnn_reranker import (
        build_corpus_graph,
        build_node_features,
        train_gnn_oof,
    )
    from graphrag.config import GNN_K_NEIGHBORS, BIENCODER_MODEL, N_FOLDS
    from graphrag.config import MODELS_DIR as MDIR
    from graphrag.finetune_biencoder import encode_corpus
    from sentence_transformers import SentenceTransformer

    logger.info("=== Stage 5.5: GNN Score Refinement (OOF, %d folds) ===", N_FOLDS)

    # Load bi-encoder for embeddings
    model_path = MDIR / "biencoder" / "final"
    if model_path.exists():
        logger.info("Loading fine-tuned bi-encoder from %s", model_path)
        bi_model = SentenceTransformer(str(model_path))
    else:
        logger.info("Loading base bi-encoder: %s", BIENCODER_MODEL)
        bi_model = SentenceTransformer(BIENCODER_MODEL)

    logger.info("Encoding corpus for GNN ...")
    enc_doc_ids, embeddings = encode_corpus(bi_model, clean_corpus)
    id_to_idx = {did: i for i, did in enumerate(enc_doc_ids)}
    logger.info("Encoded %d docs, embedding dim=%d", len(enc_doc_ids), embeddings.shape[1])

    del bi_model
    gc.collect()

    # Build corpus graph
    adj = build_corpus_graph(embeddings, enc_doc_ids, k=GNN_K_NEIGHBORS)

    # Build node features
    retrieval_scores: dict[int, dict[str, float]] = {}
    for qid in query_ids:
        for cid, _ in rrf_results.get(qid, []):
            if cid in id_to_idx:
                idx = id_to_idx[cid]
                if idx not in retrieval_scores:
                    retrieval_scores[idx] = {"bm25": 0.0, "biencoder": 0.0, "crossencoder": 0.0}
                retrieval_scores[idx]["bm25"] = max(
                    retrieval_scores[idx]["bm25"],
                    bm25_rrf.get(qid, {}).get(cid, 0.0),
                )
                retrieval_scores[idx]["biencoder"] = max(
                    retrieval_scores[idx]["biencoder"],
                    biencoder_scores.get(qid, {}).get(cid, 0.0),
                )
                retrieval_scores[idx]["crossencoder"] = max(
                    retrieval_scores[idx]["crossencoder"],
                    crossencoder_scores.get(qid, {}).get(cid, 0.0),
                )

    node_feats = build_node_features(embeddings, 0, retrieval_scores)
    logger.info("Node features: shape=%s", node_feats.shape)

    # Prepare ALL training queries
    all_queries = []
    for qid in query_ids:
        if qid not in id_to_idx:
            continue
        q_idx = id_to_idx[qid]
        gold = set(labels.get(qid, []))
        candidates = rrf_results.get(qid, [])
        c_idxs = [id_to_idx[cid] for cid, _ in candidates if cid in id_to_idx]
        c_labels = [1 if cid in gold else 0 for cid, _ in candidates if cid in id_to_idx]
        if c_idxs:
            all_queries.append({
                "query_idx": q_idx,
                "candidate_idxs": c_idxs,
                "labels": c_labels,
            })
    logger.info("Prepared %d training queries for GNN OOF", len(all_queries))

    # Train GNN with OOF (5 fold models, proper cross-validation)
    gnn_scores = train_gnn_oof(
        adj, node_feats, all_queries, enc_doc_ids, query_ids,
        rrf_results, n_folds=N_FOLDS,
    )
    logger.info("GNN OOF scored %d queries", len(gnn_scores))

    # Save cache (pickle used for pipeline cache compatibility — internal data only)
    out_path = CACHE_DIR / "stage5_5_oof.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(gnn_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info("Saved stage5_5_oof.pkl (%.1f MB)", size_mb)

    total_time = time.time() - total_t0
    logger.info("GNN OOF reranker complete in %.1f minutes", total_time / 60)


if __name__ == "__main__":
    main()
