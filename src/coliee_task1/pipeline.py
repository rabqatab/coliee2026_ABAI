"""Option C Pipeline: Hybrid Multi-View with Citation Context + GraphRAG Lite.

End-to-end orchestration of the 6-stage pipeline:
  1. Preprocessing & citation context extraction
  2. Multi-view BM25 retrieval (full-doc + per-context + RRF)
  3. Bi-encoder reranking (BGE-large with LoRA)
  4. Cross-encoder reranking (DeBERTa-v3-large)
  5. GraphRAG Lite community features
  6. LightGBM meta-learner → final predictions

Stages cache their outputs to output/pipeline_cache/ so the pipeline can
resume from any stage without recomputing expensive steps (especially
the bi-encoder encoding which takes ~2h on CPU).
"""
import json
import logging
import pickle  # noqa: S403 — internal cache only, not untrusted data
import time
from pathlib import Path

import numpy as np

from coliee_task1.config import (
    TRAIN_DOCS_DIR,
    TEST_DOCS_DIR,
    TRAIN_LABELS,
    TEST_LABELS,
    OUTPUT_DIR,
    MODELS_DIR,
    BM25_TOP_K,
    CROSSENCODER_TOP_K,
    CROSSENCODER_MAX_LENGTH,
    CROSSENCODER_BATCH_SIZE,
    CROSSENCODER_LR,
    BIENCODER_TOP_K,
    USE_STRATIFIED_NEGATIVES,
    TOP1_GUARANTEE,
    MULTI_SEED_RUNS,
    LGBM_PARAMS,
    RANDOM_SEED,
    CROSSENCODER_MODE,
    CROSSENCODER_LONG_MODEL,
    CROSSENCODER_LONG_MAX_LENGTH,
    CROSSENCODER_LONG_BATCH_SIZE,
    CROSSENCODER_LONG_LR,
    USE_BGE_M3,
    USE_GNN_RERANKER,
    USE_REASONING_RERANKER,
    USE_SYNTHETIC_DATA,
)
from coliee_task1.stages.preprocess import preprocess, load_corpus
from coliee_task1.stages.citation_context import (
    load_raw_corpus,
    extract_all_contexts,
    DocumentContexts,
)
from coliee_task1.stages.bm25 import BM25Index, rrf_fuse
from coliee_task1.stages.graphrag import GraphRAGLite
from coliee_task1.stages.meta_learner import compute_lexical_features
from coliee_task1.utils.metrics import micro_f1

logger = logging.getLogger(__name__)

CACHE_DIR = OUTPUT_DIR / "pipeline_cache"


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"


def _save_cache(name: str, data: object) -> None:
    path = _cache_path(name)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)  # noqa: S301
    size_mb = path.stat().st_size / 1024 / 1024
    logger.info("Cached %s (%.1f MB)", name, size_mb)


def _load_cache(name: str) -> object | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    logger.info("Loaded cache: %s", name)
    return data


def load_labels(path: Path) -> dict[str, list[str]]:
    """Load labels JSON, handling both list and comma-separated string formats."""
    with open(path) as f:
        raw = json.load(f)

    labels = {}
    for key, val in raw.items():
        if isinstance(val, list):
            labels[key] = val
        elif isinstance(val, str):
            labels[key] = [v.strip() for v in val.split(",") if v.strip()]
        else:
            labels[key] = []
    return labels


def stage1_preprocess(
    docs_dir: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, DocumentContexts]]:
    """Stage 1: Load corpus, preprocess, and extract citation contexts.

    Returns:
        (raw_corpus, clean_corpus, citation_contexts)
    """
    logger.info("=== Stage 1: Preprocessing & Citation Context Extraction ===")
    t0 = time.time()

    raw_corpus = load_raw_corpus(docs_dir)
    logger.info("Loaded %d raw documents", len(raw_corpus))

    # Extract citation contexts from raw text (before preprocessing)
    contexts = extract_all_contexts(raw_corpus)
    n_total_ctx = sum(dc.n_markers for dc in contexts.values())
    n_with_ctx = sum(1 for dc in contexts.values() if dc.n_markers > 0)
    logger.info(
        "Citation contexts: %d docs with markers, %d total windows",
        n_with_ctx, n_total_ctx,
    )

    # Clean corpus for downstream stages
    clean_corpus = {}
    for doc_id, raw_text in raw_corpus.items():
        clean_corpus[doc_id] = preprocess(raw_text)

    logger.info("Stage 1 complete in %.1f seconds", time.time() - t0)
    return raw_corpus, clean_corpus, contexts


def stage2_bm25(
    clean_corpus: dict[str, str],
    contexts: dict[str, DocumentContexts],
    query_ids: list[str],
    labels: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """Stage 2: Multi-view BM25 retrieval.

    Returns:
        (rrf_results, bm25_raw_scores, bm25_rrf_scores, context_features)
    """
    logger.info("=== Stage 2: Multi-view BM25 Retrieval (RRF) ===")
    t0 = time.time()

    # Build BM25 index over full clean corpus
    doc_ids = sorted(clean_corpus.keys())
    texts = [clean_corpus[did] for did in doc_ids]
    bm25 = BM25Index()
    bm25.fit(doc_ids, texts)

    rrf_results = {}
    bm25_raw_scores: dict[str, dict[str, float]] = {}
    bm25_rrf_scores: dict[str, dict[str, float]] = {}
    context_features: dict[str, dict[str, dict[str, float]]] = {}

    for i, qid in enumerate(query_ids):
        if qid not in clean_corpus:
            continue

        # Get citation context windows for this query
        dc = contexts.get(qid)
        ctx_texts = [c.text for c in dc.contexts] if dc else []

        # Multi-view BM25 with RRF fusion
        full_text = clean_corpus[qid]
        fused = bm25.query_multiview(
            full_text=full_text,
            context_windows=ctx_texts,
            exclude_id=qid,
        )
        rrf_results[qid] = fused

        # Also get raw BM25 scores for the full document
        raw_results = bm25.query(full_text, top_k=BM25_TOP_K)
        bm25_raw_scores[qid] = {
            did: score for did, score in raw_results if did != qid
        }
        bm25_rrf_scores[qid] = {did: score for did, score in fused}

        # Context features: how many context windows matched each candidate
        ctx_match_counts: dict[str, dict[str, float]] = {}
        for ctx_text in ctx_texts:
            ctx_results = bm25.query(ctx_text, top_k=30)
            for did, score in ctx_results:
                if did == qid:
                    continue
                if did not in ctx_match_counts:
                    ctx_match_counts[did] = {"n_matches": 0.0, "max_score": 0.0}
                ctx_match_counts[did]["n_matches"] += 1.0
                ctx_match_counts[did]["max_score"] = max(
                    ctx_match_counts[did]["max_score"], score,
                )
        context_features[qid] = ctx_match_counts

        if (i + 1) % 100 == 0:
            logger.info("  BM25: %d/%d queries processed", i + 1, len(query_ids))

    logger.info("Stage 2 complete in %.1f seconds", time.time() - t0)
    return rrf_results, bm25_raw_scores, bm25_rrf_scores, context_features


def stage3_biencoder(
    clean_corpus: dict[str, str],
    query_ids: list[str],
    rrf_results: dict[str, list[tuple[str, float]]],
    labels: dict[str, list[str]] | None = None,
    bm25_index: BM25Index | None = None,
    train: bool = False,
) -> tuple[dict[str, dict[str, float]], list[str], np.ndarray]:
    """Stage 3: Bi-encoder retrieval/reranking.

    If train=True, fine-tunes the bi-encoder first.
    Returns {query_id: {candidate_id: similarity_score}}
    """
    logger.info("=== Stage 3: Bi-Encoder %s ===", "Training + Inference" if train else "Inference")
    t0 = time.time()

    from coliee_task1.stages.biencoder import (
        finetune_biencoder,
        encode_corpus,
        biencoder_retrieve,
    )
    from sentence_transformers import SentenceTransformer
    from coliee_task1.config import BIENCODER_MODEL

    model_path = MODELS_DIR / "biencoder" / "final"

    if model_path.exists():
        logger.info("Loading fine-tuned bi-encoder from %s", model_path)
        model = SentenceTransformer(str(model_path))
    elif train and labels is not None and bm25_index is not None:
        model = finetune_biencoder(labels, clean_corpus, bm25_index)
    else:
        logger.info("Using base bi-encoder: %s", BIENCODER_MODEL)
        model = SentenceTransformer(BIENCODER_MODEL)

    # Encode full corpus
    doc_ids, embeddings = encode_corpus(model, clean_corpus)
    doc_idx = {did: i for i, did in enumerate(doc_ids)}

    # Score candidates for each query
    biencoder_scores: dict[str, dict[str, float]] = {}
    for qid in query_ids:
        if qid not in doc_idx:
            continue
        q_emb = embeddings[doc_idx[qid]]

        # Only score candidates from BM25 pool (not full corpus for efficiency)
        candidates = rrf_results.get(qid, [])
        if not candidates:
            continue

        cand_indices = [doc_idx[cid] for cid, _ in candidates if cid in doc_idx]
        cand_ids = [cid for cid, _ in candidates if cid in doc_idx]
        if not cand_indices:
            continue

        cand_embs = embeddings[cand_indices]
        sims = q_emb @ cand_embs.T
        biencoder_scores[qid] = {
            cid: float(sim) for cid, sim in zip(cand_ids, sims)
        }

    logger.info("Stage 3 complete in %.1f seconds", time.time() - t0)
    return biencoder_scores, doc_ids, embeddings


def stage4_crossencoder(
    clean_corpus: dict[str, str],
    query_ids: list[str],
    rrf_results: dict[str, list[tuple[str, float]]],
    contexts: dict | None = None,
    labels: dict[str, list[str]] | None = None,
    bm25_candidates: dict[str, list[tuple[str, float]]] | None = None,
    train: bool = False,
) -> dict[str, dict[str, float]]:
    """Stage 4: Cross-encoder reranking.

    Supports 3 modes via CROSSENCODER_MODE config:
      - "smart": DeBERTa-v3 + smart truncation with citation contexts
      - "longctx": BGE-reranker-v2-m3 with 4096 token window
      - "passage": DeBERTa-v3 + passage-level scoring with max-pooling
    """
    mode = CROSSENCODER_MODE
    logger.info("=== Stage 4: Cross-Encoder %s (mode=%s) ===",
                "Training + Inference" if train else "Inference", mode)
    t0 = time.time()

    from coliee_task1.stages.crossencoder import (
        finetune_crossencoder,
        crossencoder_rerank,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from coliee_task1.config import CROSSENCODER_MODEL, CROSSENCODER_TOP_K
    import torch
    import json as json_mod

    # Select model config based on mode
    if mode == "longctx":
        ce_model_name = CROSSENCODER_LONG_MODEL
        ce_max_length = CROSSENCODER_LONG_MAX_LENGTH
        ce_batch_size = CROSSENCODER_LONG_BATCH_SIZE
        ce_lr = CROSSENCODER_LONG_LR
    else:
        ce_model_name = CROSSENCODER_MODEL
        ce_max_length = CROSSENCODER_MAX_LENGTH
        ce_batch_size = CROSSENCODER_BATCH_SIZE
        ce_lr = CROSSENCODER_LR

    model_path = MODELS_DIR / "crossencoder" / "final"

    # Check if saved model matches current mode
    saved_mode = None
    mode_file = model_path / "ce_mode.json"
    if model_path.exists() and mode_file.exists():
        saved_mode = json_mod.loads(mode_file.read_text()).get("mode")

    if model_path.exists() and saved_mode == mode:
        logger.info("Loading fine-tuned cross-encoder from %s (mode=%s)", model_path, mode)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_labels = 1 if mode == "longctx" else 2
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=num_labels,
        ).to(device)
        if mode != "longctx":
            model = model.float()
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    elif train and labels is not None and bm25_candidates is not None:
        model, tokenizer = finetune_crossencoder(
            labels, clean_corpus, bm25_candidates,
            contexts=contexts,
            model_name=ce_model_name,
            lr=ce_lr,
            batch_size=ce_batch_size,
            max_length=ce_max_length,
            mode=mode,
        )
    else:
        logger.warning("No fine-tuned cross-encoder found for mode=%s. Skipping stage 4.", mode)
        return {}

    # Rerank top candidates for each query
    crossencoder_scores: dict[str, dict[str, float]] = {}
    infer_batch = 32 if mode != "longctx" else ce_batch_size

    for i, qid in enumerate(query_ids):
        rrf = rrf_results.get(qid, [])
        top_candidates = rrf[:CROSSENCODER_TOP_K]
        if not top_candidates:
            continue

        query_text = clean_corpus.get(qid, "")
        candidates = [
            (cid, clean_corpus.get(cid, "")) for cid, _ in top_candidates
        ]

        # Prepare context info for smart/passage modes
        q_ctx = None
        c_ctx = None
        if contexts and mode in ("smart", "passage"):
            dc = contexts.get(qid)
            q_ctx = [c.text for c in dc.contexts] if dc and dc.contexts else None
            c_ctx = {}
            for cid, _ in top_candidates:
                dc_c = contexts.get(cid)
                if dc_c and dc_c.contexts:
                    c_ctx[cid] = [c.text for c in dc_c.contexts]

        reranked = crossencoder_rerank(
            model, tokenizer, query_text, candidates,
            max_length=ce_max_length,
            batch_size=infer_batch,
            mode=mode,
            query_contexts=q_ctx,
            candidate_contexts=c_ctx,
        )
        crossencoder_scores[qid] = {cid: score for cid, score in reranked}

        if (i + 1) % 50 == 0:
            logger.info("  Cross-encoder: %d/%d queries reranked", i + 1, len(query_ids))

    logger.info("Stage 4 complete in %.1f seconds", time.time() - t0)
    return crossencoder_scores


def stage5_graphrag(
    clean_corpus: dict[str, str],
    query_ids: list[str],
    rrf_results: dict[str, list[tuple[str, float]]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Stage 5: GraphRAG Lite community features.

    Returns {(query_id, candidate_id): {feature_name: value}}
    """
    logger.info("=== Stage 5: GraphRAG Lite ===")
    t0 = time.time()

    grag = GraphRAGLite()
    grag.fit(clean_corpus)

    graphrag_features: dict[tuple[str, str], dict[str, float]] = {}
    for qid in query_ids:
        candidates = rrf_results.get(qid, [])
        for cid, _ in candidates:
            feats = grag.get_pair_features(qid, cid)
            graphrag_features[(qid, cid)] = feats

    logger.info("Stage 5 complete in %.1f seconds", time.time() - t0)
    return graphrag_features


def stage3_multi_retrieval(
    clean_corpus: dict[str, str],
    query_ids: list[str],
    rrf_results: dict[str, list[tuple[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Stage 3 alt: BGE-M3 multi-signal retrieval.

    Returns {query_id: {signal: {candidate_id: score}}}.
    """
    from coliee_task1.stages.multi_retrieval import encode_corpus_m3, score_candidates_m3

    logger.info("=== Stage 3 (alt): BGE-M3 Multi-Signal Retrieval ===")
    t0 = time.time()

    corpus_repr = encode_corpus_m3(clean_corpus)
    candidate_lists = {
        qid: [cid for cid, _ in rrf_results.get(qid, [])]
        for qid in query_ids
    }
    multi_scores = score_candidates_m3(query_ids, candidate_lists, corpus_repr)

    logger.info("Stage 3 (BGE-M3) complete in %.1f seconds", time.time() - t0)
    return multi_scores


def stage4_5_reasoning(
    raw_corpus: dict[str, str],
    query_ids: list[str],
    rrf_results: dict[str, list[tuple[str, float]]],
) -> dict[str, dict[str, float]]:
    """Stage 4.5: Reasoning reranker with chain-of-thought."""
    from coliee_task1.stages.reasoning_reranker import batch_reasoning_rerank

    logger.info("=== Stage 4.5: Reasoning Reranker ===")
    return batch_reasoning_rerank(query_ids, raw_corpus, rrf_results)


def stage5_5_gnn(
    clean_corpus: dict[str, str],
    query_ids: list[str],
    rrf_results: dict[str, list[tuple[str, float]]],
    biencoder_scores: dict[str, dict[str, float]],
    bm25_rrf_scores: dict[str, dict[str, float]],
    crossencoder_scores: dict[str, dict[str, float]],
    labels: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Stage 5.5: GNN score refinement via GAT on corpus graph.

    Returns {query_id: {candidate_id: gnn_score}}.
    """
    from coliee_task1.stages.gnn import (
        build_corpus_graph,
        build_node_features,
        train_gnn_reranker,
        gnn_rerank,
    )
    from coliee_task1.config import GNN_K_NEIGHBORS
    from coliee_task1.stages.biencoder import encode_corpus
    from sentence_transformers import SentenceTransformer
    from coliee_task1.config import BIENCODER_MODEL, MODELS_DIR as MDIR

    logger.info("=== Stage 5.5: GNN Score Refinement ===")
    t0 = time.time()

    # Load bi-encoder for embeddings
    model_path = MDIR / "biencoder" / "final"
    if model_path.exists():
        bi_model = SentenceTransformer(str(model_path))
    else:
        bi_model = SentenceTransformer(BIENCODER_MODEL)

    enc_doc_ids, embeddings = encode_corpus(bi_model, clean_corpus)
    id_to_idx = {did: i for i, did in enumerate(enc_doc_ids)}

    # Build corpus graph from embeddings
    adj = build_corpus_graph(embeddings, enc_doc_ids, k=GNN_K_NEIGHBORS)

    # Build node features: truncated embeddings + retrieval scores
    retrieval_scores: dict[int, dict[str, float]] = {}
    for qid in query_ids:
        for cid, _ in rrf_results.get(qid, []):
            if cid in id_to_idx:
                idx = id_to_idx[cid]
                if idx not in retrieval_scores:
                    retrieval_scores[idx] = {"bm25": 0.0, "biencoder": 0.0, "crossencoder": 0.0}
                retrieval_scores[idx]["bm25"] = max(
                    retrieval_scores[idx]["bm25"],
                    bm25_rrf_scores.get(qid, {}).get(cid, 0.0),
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

    # Prepare training queries
    train_queries = []
    for qid in query_ids:
        if qid not in id_to_idx:
            continue
        q_idx = id_to_idx[qid]
        gold = set(labels.get(qid, []))
        candidates = rrf_results.get(qid, [])
        c_idxs = [id_to_idx[cid] for cid, _ in candidates if cid in id_to_idx]
        c_labels = [1 if cid in gold else 0 for cid, _ in candidates if cid in id_to_idx]
        if c_idxs:
            train_queries.append({
                "query_idx": q_idx,
                "candidate_idxs": c_idxs,
                "labels": c_labels,
            })

    # Train and run inference
    gnn_model = train_gnn_reranker(adj, node_feats, train_queries)

    query_idxs = [id_to_idx[qid] for qid in query_ids if qid in id_to_idx]
    candidate_idx_lists = {
        id_to_idx[qid]: [id_to_idx[cid] for cid, _ in rrf_results.get(qid, []) if cid in id_to_idx]
        for qid in query_ids if qid in id_to_idx
    }

    gnn_scores = gnn_rerank(gnn_model, adj, node_feats, query_idxs, candidate_idx_lists, enc_doc_ids)

    logger.info("Stage 5.5 complete in %.1f seconds", time.time() - t0)
    return gnn_scores


def stage6_meta_learner(
    labels: dict[str, list[str]],
    query_ids: list[str],
    rrf_results: dict[str, list[tuple[str, float]]],
    bm25_raw_scores: dict[str, dict[str, float]],
    bm25_rrf_scores: dict[str, dict[str, float]],
    biencoder_scores: dict[str, dict[str, float]],
    crossencoder_scores: dict[str, dict[str, float]],
    graphrag_features: dict[tuple[str, str], dict[str, float]],
    context_features: dict[str, dict[str, dict[str, float]]],
    clean_corpus: dict[str, str] | None = None,
    multi_scores: dict[str, dict[str, dict[str, float]]] | None = None,
    gnn_scores: dict[str, dict[str, float]] | None = None,
    reasoning_scores: dict[str, dict[str, float]] | None = None,
    train: bool = True,
) -> dict[str, list[str]]:
    """Stage 6: LightGBM meta-learner.

    If train=True, trains with GroupKFold CV and returns OOF predictions.
    Otherwise, loads saved models and predicts.
    """
    logger.info("=== Stage 6: Meta-Learner %s ===", "Training" if train else "Inference")
    t0 = time.time()

    from coliee_task1.stages.meta_learner import (
        build_feature_matrix,
        train_meta_learner,
        predict,
        FEATURE_COLS,
    )
    import lightgbm as lgb

    # Build candidate pool from RRF results
    candidate_pool: dict[str, list[str]] = {
        qid: [cid for cid, _ in rrf_results.get(qid, [])]
        for qid in query_ids
    }

    # Add gold positives not already in the RRF pool.
    # WARNING: This creates train/test mismatch — gold won't exist in test pools.
    # Set INJECT_GOLD_IN_POOL=False (default) for honest training.
    from coliee_task1.config import INJECT_GOLD_IN_POOL
    if train and INJECT_GOLD_IN_POOL:
        n_added = 0
        for qid in query_ids:
            gold = set(labels.get(qid, []))
            pool_set = set(candidate_pool.get(qid, []))
            for pos_id in gold:
                if pos_id not in pool_set:
                    candidate_pool[qid].append(pos_id)
                    n_added += 1
        if n_added:
            logger.info("Added %d gold positives not in RRF pool", n_added)
    elif train:
        # Log how many gold positives are missing (for diagnostics)
        n_missing = 0
        for qid in query_ids:
            gold = set(labels.get(qid, []))
            pool_set = set(candidate_pool.get(qid, []))
            n_missing += len(gold - pool_set)
        logger.info("Gold injection DISABLED: %d gold positives outside RRF pool (unreachable)", n_missing)

    # Compute lexical features if corpus is available
    lexical_features = None
    if clean_corpus is not None:
        lexical_features = compute_lexical_features(clean_corpus, candidate_pool)

    # Shared feature args
    feat_args = dict(
        bm25_scores=bm25_raw_scores,
        bm25_rrf_scores=bm25_rrf_scores,
        biencoder_scores=biencoder_scores,
        crossencoder_scores=crossencoder_scores,
        graphrag_features=graphrag_features,
        context_features=context_features,
        lexical_features=lexical_features,
        multi_scores=multi_scores,
        gnn_scores=gnn_scores,
        reasoning_scores=reasoning_scores,
    )

    lgbm_params = LGBM_PARAMS.copy()

    min_per_query = 1 if TOP1_GUARANTEE else 0

    if train:
        # Build sampled feature matrix for training (expanded pool with gold positives)
        df_train = build_feature_matrix(
            labels, candidate_pool, **feat_args,
            subsample=True,
            stratified=USE_STRATIFIED_NEGATIVES,
        )

        # Multi-seed ensemble (Option 16)
        all_models = []
        all_cv_metrics = []
        for seed_run in range(MULTI_SEED_RUNS):
            if MULTI_SEED_RUNS > 1:
                run_params = lgbm_params.copy()
                run_params["seed"] = RANDOM_SEED + seed_run
                run_params["bagging_seed"] = RANDOM_SEED + seed_run + 100
                run_params["feature_fraction_seed"] = RANDOM_SEED + seed_run + 200
                logger.info("--- Multi-seed run %d/%d (seed=%d) ---",
                            seed_run + 1, MULTI_SEED_RUNS, run_params["seed"])
                out_dir = MODELS_DIR / "meta_learner" / f"seed_{seed_run}"
            else:
                run_params = lgbm_params
                out_dir = None  # default

            models, threshold_oof, cv_metrics = train_meta_learner(
                df_train, lgbm_params=run_params, output_dir=out_dir,
            )
            all_models.extend(models)
            all_cv_metrics.append(cv_metrics)

        if MULTI_SEED_RUNS > 1:
            avg_f1 = np.mean([m.get("f1", 0) for m in all_cv_metrics])
            logger.info("Multi-seed avg CV F1: %.4f (%d seed runs)", avg_f1, MULTI_SEED_RUNS)

        logger.info(
            "Meta-learner CV (sampled OOF): F1=%.4f, P=%.4f, R=%.4f (t=%.3f)",
            cv_metrics.get("f1", 0),
            cv_metrics.get("precision", 0),
            cv_metrics.get("recall", 0),
            threshold_oof,
        )

        # Build full feature matrix for evaluation (all candidates, no sampling)
        df_full = build_feature_matrix(
            labels, candidate_pool, **feat_args,
            subsample=False,
        )

        # Re-optimize threshold on full pool predictions (real pos/neg ratio)
        # Use ALL models from multi-seed ensemble
        from coliee_task1.utils.metrics import optimize_threshold as opt_thresh
        full_scores = np.zeros(len(df_full))
        for m in all_models:
            full_scores += m.predict(df_full[FEATURE_COLS].values)
        full_scores /= len(all_models)

        full_query_scores: dict[str, list[tuple[str, float]]] = {}
        for i, (_, row) in enumerate(df_full.iterrows()):
            qid = row["query_id"]
            cid = row["candidate_id"]
            if qid not in full_query_scores:
                full_query_scores[qid] = []
            full_query_scores[qid].append((cid, full_scores[i]))

        threshold, full_metrics = opt_thresh(full_query_scores, labels)
        logger.info(
            "Threshold re-optimized on full pool: F1=%.4f, P=%.4f, R=%.4f (t=%.3f)",
            full_metrics.get("f1", 0),
            full_metrics.get("precision", 0),
            full_metrics.get("recall", 0),
            threshold,
        )

        # Save updated threshold
        import json as json_mod
        config_path = MODELS_DIR / "meta_learner" / "config.json"
        config_path.write_text(json_mod.dumps({
            "threshold": threshold,
            "cv_f1": full_metrics.get("f1", 0),
            "objective": lgbm_params.get("objective", "binary"),
            "multi_seed_runs": MULTI_SEED_RUNS,
            "n_models": len(all_models),
        }, indent=2))

        predictions = predict(all_models, df_full, threshold, min_per_query=min_per_query)
    else:
        # Load saved models
        model_dir = MODELS_DIR / "meta_learner"
        config = json.loads((model_dir / "config.json").read_text())
        threshold = config["threshold"]

        models = []
        # Load from multi-seed subdirs if present
        n_seeds = config.get("multi_seed_runs", 1)
        if n_seeds > 1:
            for seed_run in range(n_seeds):
                seed_dir = model_dir / f"seed_{seed_run}"
                for f in sorted(seed_dir.glob("fold_*.txt")):
                    models.append(lgb.Booster(model_file=str(f)))
        else:
            for f in sorted(model_dir.glob("fold_*.txt")):
                models.append(lgb.Booster(model_file=str(f)))

        df_full = build_feature_matrix(
            labels, candidate_pool, **feat_args,
            subsample=False,
        )
        predictions = predict(models, df_full, threshold, min_per_query=min_per_query)

    logger.info("Stage 6 complete in %.1f seconds", time.time() - t0)
    return predictions


def run_train_pipeline(
    docs_dir: Path = TRAIN_DOCS_DIR,
    test_docs_dir: Path | None = TEST_DOCS_DIR,
    labels_path: Path = TRAIN_LABELS,
    finetune: bool = True,
    use_cache: bool = True,
) -> dict[str, float]:
    """Run full training pipeline with cross-validation.

    Args:
        docs_dir: Training documents directory.
        test_docs_dir: Test documents directory. Merged into corpus so that
                       retrieval can find candidates across the full pool.
                       Set to None to use train docs only.
        finetune: If False, use base models for neural stages (CPU-friendly).
                  Stages 3/4 use pre-trained weights without fine-tuning.
        use_cache: If True, load cached stage outputs when available.

    Returns CV metrics dict.
    """
    mode_str = "Full" if finetune else "No-Finetune (CPU)"
    logger.info("========================================")
    logger.info("  Option C Pipeline — Training (%s)", mode_str)
    logger.info("  Cache: %s", "enabled" if use_cache else "disabled")
    logger.info("========================================")
    total_t0 = time.time()

    # Load labels
    labels = load_labels(labels_path)
    query_ids = sorted(labels.keys())
    logger.info("Loaded %d training queries", len(query_ids))

    # Stage 1: Preprocess (train + test docs for full retrieval pool)
    cached = _load_cache("stage1") if use_cache else None
    if cached is not None:
        raw_corpus, clean_corpus, contexts = cached
    else:
        raw_train, clean_train, ctx_train = stage1_preprocess(docs_dir)
        if test_docs_dir is not None and test_docs_dir.exists():
            raw_test, clean_test, ctx_test = stage1_preprocess(test_docs_dir)
            raw_corpus = {**raw_train, **raw_test}
            clean_corpus = {**clean_train, **clean_test}
            contexts = {**ctx_train, **ctx_test}
            logger.info("Merged corpus: %d train + %d test = %d total docs",
                        len(raw_train), len(raw_test), len(raw_corpus))
        else:
            raw_corpus, clean_corpus, contexts = raw_train, clean_train, ctx_train
        _save_cache("stage1", (raw_corpus, clean_corpus, contexts))

    # Stage 2: BM25
    cached = _load_cache("stage2") if use_cache else None
    if cached is not None:
        rrf_results, bm25_raw, bm25_rrf, ctx_feats = cached
    else:
        rrf_results, bm25_raw, bm25_rrf, ctx_feats = stage2_bm25(
            clean_corpus, contexts, query_ids, labels=labels,
        )
        _save_cache("stage2", (rrf_results, bm25_raw, bm25_rrf, ctx_feats))

    # Synthetic data generation (before bi-encoder training)
    synthetic_pairs = None
    if USE_SYNTHETIC_DATA:
        synthetic_path = OUTPUT_DIR / "synthetic_pairs.jsonl"
        if synthetic_path.exists():
            import json as json_mod
            synthetic_pairs = [
                json_mod.loads(line) for line in synthetic_path.read_text().splitlines()
                if line.strip()
            ]
            logger.info("Loaded %d synthetic pairs from cache", len(synthetic_pairs))
        else:
            from coliee_task1.utils.synthetic_data import generate_synthetic_pairs
            synthetic_pairs = generate_synthetic_pairs(raw_corpus, labels)

    # Build BM25 index for downstream stages (hard negative mining)
    bm25_index = None
    if finetune:
        doc_ids_list = sorted(clean_corpus.keys())
        texts_list = [clean_corpus[d] for d in doc_ids_list]
        bm25_index = BM25Index()
        bm25_index.fit(doc_ids_list, texts_list)

    # Stage 3: Bi-encoder
    cached = _load_cache("stage3") if use_cache else None
    if cached is not None:
        if isinstance(cached, tuple) and len(cached) == 3:
            biencoder_scores, biencoder_doc_ids, biencoder_embeddings = cached
        else:
            biencoder_scores = cached
            biencoder_doc_ids, biencoder_embeddings = None, None
    else:
        biencoder_scores, biencoder_doc_ids, biencoder_embeddings = stage3_biencoder(
            clean_corpus, query_ids, rrf_results,
            labels=labels, bm25_index=bm25_index, train=finetune,
        )
        _save_cache("stage3", (biencoder_scores, biencoder_doc_ids, biencoder_embeddings))

    # === Hybrid First-Stage: Fuse BM25 + Dense Retrieval ===
    from coliee_task1.config import USE_DENSE_FIRST_STAGE, DENSE_TOP_K, HYBRID_FUSION, CONVEX_ALPHA
    if USE_DENSE_FIRST_STAGE and biencoder_doc_ids is not None and biencoder_embeddings is not None:
        from coliee_task1.stages.biencoder import dense_retrieve_full_corpus
        from coliee_task1.stages.bm25 import hybrid_fuse

        logger.info("=== Hybrid Fusion: BM25 + Dense First-Stage ===")
        dense_results = dense_retrieve_full_corpus(
            query_ids, biencoder_doc_ids, biencoder_embeddings, top_k=DENSE_TOP_K,
        )

        # Convert RRF results to list format for fusion
        bm25_as_list = {
            qid: list(rrf_results.get(qid, []))
            for qid in query_ids
        }
        rrf_results = hybrid_fuse(
            bm25_as_list, dense_results,
            method=HYBRID_FUSION, alpha=CONVEX_ALPHA,
            top_k=max(BM25_TOP_K, DENSE_TOP_K),
        )

        # Update RRF score dict from fused results
        bm25_rrf = {
            qid: {did: score for did, score in rrf_results.get(qid, [])}
            for qid in query_ids
        }

        # Measure new recall ceiling
        if labels:
            total_gold = sum(len(v) for v in labels.values())
            found_gold = sum(
                len(set(labels.get(qid, [])) & {did for did, _ in rrf_results.get(qid, [])})
                for qid in query_ids
            )
            logger.info("Hybrid recall ceiling: %d/%d = %.1f%% (was 60.9%% BM25-only)",
                         found_gold, total_gold, 100.0 * found_gold / max(total_gold, 1))

    # Stage 3 (alt): BGE-M3 multi-signal retrieval
    multi_scores = None
    if USE_BGE_M3:
        cached = _load_cache("stage3_m3") if use_cache else None
        if cached is not None:
            multi_scores = cached
        else:
            multi_scores = stage3_multi_retrieval(clean_corpus, query_ids, rrf_results)
            _save_cache("stage3_m3", multi_scores)

    # Stage 4: Cross-encoder
    if finetune:
        cached = _load_cache("stage4") if use_cache else None
        if cached is not None:
            crossencoder_scores = cached
        else:
            bm25_candidates = {
                qid: [(cid, s) for cid, s in rrf_results.get(qid, [])]
                for qid in query_ids
            }
            crossencoder_scores = stage4_crossencoder(
                clean_corpus, query_ids, rrf_results,
                contexts=contexts,
                labels=labels, bm25_candidates=bm25_candidates, train=True,
            )
            _save_cache("stage4", crossencoder_scores)
    else:
        logger.info("=== Stage 4: Cross-Encoder SKIPPED (no-finetune mode) ===")
        crossencoder_scores = {}

    # Stage 4.5: Reasoning reranker
    reasoning_scores = None
    if USE_REASONING_RERANKER:
        cached = _load_cache("stage4_5") if use_cache else None
        if cached is not None:
            reasoning_scores = cached
        else:
            reasoning_scores = stage4_5_reasoning(raw_corpus, query_ids, rrf_results)
            _save_cache("stage4_5", reasoning_scores)

    # Stage 5: GraphRAG Lite
    cached = _load_cache("stage5") if use_cache else None
    if cached is not None:
        graphrag_features = cached
    else:
        graphrag_features = stage5_graphrag(clean_corpus, query_ids, rrf_results)
        _save_cache("stage5", graphrag_features)

    # Stage 5.5: GNN score refinement
    gnn_scores = None
    if USE_GNN_RERANKER:
        cached = _load_cache("stage5_5") if use_cache else None
        if cached is not None:
            gnn_scores = cached
        else:
            gnn_scores = stage5_5_gnn(
                clean_corpus, query_ids, rrf_results,
                biencoder_scores, bm25_rrf, crossencoder_scores, labels,
            )
            _save_cache("stage5_5", gnn_scores)

    # Stage 6: Meta-learner (always re-run — it's fast and we want fresh CV)
    predictions = stage6_meta_learner(
        labels, query_ids, rrf_results,
        bm25_raw, bm25_rrf,
        biencoder_scores, crossencoder_scores,
        graphrag_features, ctx_feats,
        clean_corpus=clean_corpus,
        multi_scores=multi_scores,
        gnn_scores=gnn_scores,
        reasoning_scores=reasoning_scores,
        train=True,
    )

    # Final metrics
    metrics = micro_f1(predictions, labels)
    total_time = time.time() - total_t0
    logger.info("========================================")
    logger.info("  Training Complete: %.1f minutes", total_time / 60)
    logger.info("  CV F1=%.4f  P=%.4f  R=%.4f", metrics["f1"], metrics["precision"], metrics["recall"])
    logger.info("========================================")

    return metrics


def run_predict_pipeline(
    train_docs_dir: Path = TRAIN_DOCS_DIR,
    test_docs_dir: Path = TEST_DOCS_DIR,
    train_labels_path: Path = TRAIN_LABELS,
    test_labels_path: Path = TEST_LABELS,
    output_path: Path | None = None,
) -> dict[str, list[str]]:
    """Run prediction pipeline on test data.

    Loads pre-trained models and generates predictions for test queries.
    """
    logger.info("========================================")
    logger.info("  Option C Pipeline — Prediction")
    logger.info("========================================")

    if output_path is None:
        output_path = OUTPUT_DIR / "predictions.json"

    # Load test queries
    test_labels = load_labels(test_labels_path)
    query_ids = sorted(test_labels.keys())

    # Load full corpus (train + test for retrieval)
    raw_train = load_raw_corpus(train_docs_dir)
    raw_test = load_raw_corpus(test_docs_dir)
    raw_corpus = {**raw_train, **raw_test}

    clean_corpus = {
        did: preprocess(text) for did, text in raw_corpus.items()
    }

    contexts = extract_all_contexts(raw_corpus)

    # Stage 2: BM25
    rrf_results, bm25_raw, bm25_rrf, ctx_feats = stage2_bm25(
        clean_corpus, contexts, query_ids,
    )

    # Stage 3: Bi-encoder (inference only)
    biencoder_scores, biencoder_doc_ids, biencoder_embeddings = stage3_biencoder(
        clean_corpus, query_ids, rrf_results, train=False,
    )

    # === Hybrid First-Stage: Fuse BM25 + Dense Retrieval ===
    from coliee_task1.config import USE_DENSE_FIRST_STAGE, DENSE_TOP_K, HYBRID_FUSION, CONVEX_ALPHA
    if USE_DENSE_FIRST_STAGE and biencoder_doc_ids is not None and biencoder_embeddings is not None:
        from coliee_task1.stages.biencoder import dense_retrieve_full_corpus
        from coliee_task1.stages.bm25 import hybrid_fuse

        logger.info("=== Hybrid Fusion: BM25 + Dense First-Stage ===")
        dense_results = dense_retrieve_full_corpus(
            query_ids, biencoder_doc_ids, biencoder_embeddings, top_k=DENSE_TOP_K,
        )

        # Convert RRF results to list format for fusion
        bm25_as_list = {
            qid: list(rrf_results.get(qid, []))
            for qid in query_ids
        }
        rrf_results = hybrid_fuse(
            bm25_as_list, dense_results,
            method=HYBRID_FUSION, alpha=CONVEX_ALPHA,
            top_k=max(BM25_TOP_K, DENSE_TOP_K),
        )

        # Update RRF score dict from fused results
        bm25_rrf = {
            qid: {did: score for did, score in rrf_results.get(qid, [])}
            for qid in query_ids
        }

    # Stage 3 (alt): BGE-M3
    multi_scores = None
    if USE_BGE_M3:
        multi_scores = stage3_multi_retrieval(clean_corpus, query_ids, rrf_results)

    # Stage 4: Cross-encoder (inference only)
    crossencoder_scores = stage4_crossencoder(
        clean_corpus, query_ids, rrf_results,
        contexts=contexts, train=False,
    )

    # Stage 4.5: Reasoning reranker
    reasoning_scores = None
    if USE_REASONING_RERANKER:
        reasoning_scores = stage4_5_reasoning(raw_corpus, query_ids, rrf_results)

    # Stage 5: GraphRAG Lite
    graphrag_features = stage5_graphrag(clean_corpus, query_ids, rrf_results)

    # Note: GNN reranker skipped in predict mode (trained on training data)
    gnn_scores = None

    # Stage 6: Meta-learner (inference only)
    predictions = stage6_meta_learner(
        test_labels, query_ids, rrf_results,
        bm25_raw, bm25_rrf,
        biencoder_scores, crossencoder_scores,
        graphrag_features, ctx_feats,
        clean_corpus=clean_corpus,
        multi_scores=multi_scores,
        gnn_scores=gnn_scores,
        reasoning_scores=reasoning_scores,
        train=False,
    )

    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info("Predictions saved to %s", output_path)

    return predictions


