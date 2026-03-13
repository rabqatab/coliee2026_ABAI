"""Baseline models for COLIEE 2026 Tasks 1 & 2.

Implements simple-but-strong baselines to establish performance floors:
  Task 1: BM25-only, TF-IDF cosine, BM25+Lexical→LightGBM
  Task 2: BM25-only, TF-IDF cosine
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from graphrag.bm25 import BM25Index, tokenize
from graphrag.config import (
    DATA_DIR,
    TASK1_DIR,
    TRAIN_DOCS_DIR,
    TRAIN_LABELS,
    N_FOLDS,
    RANDOM_SEED,
    OUTPUT_DIR,
)
from graphrag.metrics import micro_f1, optimize_threshold
from graphrag.preprocess import preprocess, load_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _bm25_batch_scores(bm25_index: BM25Index, query_ids: list[str],
                        corpus: dict[str, str], id_to_idx: dict[str, int],
                        top_k: int = 200) -> dict[str, list[tuple[str, float]]]:
    """Batch BM25 scoring with progress tracking.

    Returns dict[query_id -> [(doc_id, score), ...]] sorted desc, top_k results.
    """
    doc_ids = bm25_index._doc_ids
    scores_dict: dict[str, list[tuple[str, float]]] = {}
    n = len(query_ids)
    t0 = time.time()

    for i, qid in enumerate(query_ids):
        if qid not in corpus:
            continue
        tokens = tokenize(corpus[qid])
        raw_scores = bm25_index._score_tokens(tokens)
        # Use argpartition for O(n) top-k instead of O(n log n) full sort
        k = min(top_k + 1, len(raw_scores))
        top_indices = np.argpartition(-raw_scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-raw_scores[top_indices])]
        results = []
        for idx in top_indices:
            did = doc_ids[idx]
            if did != qid:
                results.append((did, float(raw_scores[idx])))
            if len(results) >= top_k:
                break
        scores_dict[qid] = results

        if (i + 1) % 200 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            logger.info("  BM25 query %d/%d (%.1f q/s, ETA %.0fs)", i + 1, n, rate, eta)

    return scores_dict

# ─────────────────────────────────────────────
# Task 1 Baselines
# ─────────────────────────────────────────────


def _load_task1_data():
    """Load Task 1 corpus + labels."""
    logger.info("Loading Task 1 corpus from %s ...", TRAIN_DOCS_DIR)
    raw_corpus = load_corpus(TRAIN_DOCS_DIR)
    corpus = {k: preprocess(v) for k, v in raw_corpus.items()}
    labels = json.loads(TRAIN_LABELS.read_text())
    logger.info("Loaded %d documents, %d queries", len(corpus), len(labels))
    return corpus, labels


def task1_bm25_baseline(corpus: dict[str, str], labels: dict[str, list[str]]) -> dict:
    """Baseline 1: Pure BM25 retrieval with threshold optimization.

    For each query, retrieve top-K by BM25 score, then optimize a score
    threshold on the full training set to maximize micro-F1.
    """
    logger.info("=" * 60)
    logger.info("Task 1 Baseline: BM25-only")
    logger.info("=" * 60)

    t0 = time.time()

    doc_ids = sorted(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]

    # Build index
    bm25 = BM25Index()
    bm25.fit(doc_ids, doc_texts)
    logger.info("BM25 index built in %.1fs", time.time() - t0)

    # Query all labeled queries
    query_ids = sorted(labels.keys())
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    scores = _bm25_batch_scores(bm25, query_ids, corpus, id_to_idx, top_k=200)
    logger.info("BM25 retrieval done in %.1fs", time.time() - t0)

    # Optimize threshold
    best_thresh, best_metrics = optimize_threshold(scores, labels)

    elapsed = time.time() - t0
    result = {
        "model": "BM25-only",
        "task": "Task 1",
        "threshold": best_thresh,
        **best_metrics,
        "time_seconds": elapsed,
    }
    _print_result(result)
    return result


def task1_tfidf_baseline(corpus: dict[str, str], labels: dict[str, list[str]]) -> dict:
    """Baseline 2: TF-IDF cosine similarity with threshold optimization."""
    logger.info("=" * 60)
    logger.info("Task 1 Baseline: TF-IDF Cosine")
    logger.info("=" * 60)

    t0 = time.time()

    doc_ids = sorted(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]

    # Build TF-IDF matrix
    logger.info("Building TF-IDF matrix ...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        sublinear_tf=True,
        stop_words="english",
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    logger.info("TF-IDF matrix: %s, built in %.1fs", tfidf_matrix.shape, time.time() - t0)

    # Batch: compute cosine similarity for all queries at once
    query_ids = sorted(labels.keys())
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    query_indices = [id_to_idx[qid] for qid in query_ids if qid in id_to_idx]
    valid_query_ids = [qid for qid in query_ids if qid in id_to_idx]

    # Batch cosine sim: queries × all docs (sparse matrix multiplication is fast)
    logger.info("Computing cosine similarities (batch) ...")
    query_matrix = tfidf_matrix[query_indices]
    sim_matrix = (query_matrix @ tfidf_matrix.T).toarray()  # (n_queries, n_docs)
    logger.info("Similarity matrix computed in %.1fs", time.time() - t0)

    scores: dict[str, list[tuple[str, float]]] = {}
    for i, qid in enumerate(valid_query_ids):
        q_idx = id_to_idx[qid]
        sim_row = sim_matrix[i]
        sim_row[q_idx] = -1  # Exclude self
        top_indices = np.argsort(-sim_row)[:200]
        scores[qid] = [(doc_ids[idx], float(sim_row[idx])) for idx in top_indices]

    logger.info("TF-IDF retrieval done in %.1fs", time.time() - t0)

    best_thresh, best_metrics = optimize_threshold(scores, labels)

    elapsed = time.time() - t0
    result = {
        "model": "TF-IDF Cosine",
        "task": "Task 1",
        "threshold": best_thresh,
        **best_metrics,
        "time_seconds": elapsed,
    }
    _print_result(result)
    return result


def task1_lgbm_baseline(corpus: dict[str, str], labels: dict[str, list[str]]) -> dict:
    """Baseline 3: BM25 + lexical features → LightGBM with GroupKFold CV.

    Features per (query, candidate) pair:
      - BM25 score
      - TF-IDF cosine similarity
      - Jaccard word overlap
      - Shared bigrams Jaccard
      - Length ratio
      - Shared legal terms count

    This mirrors the signal validation approach and the JNLP 2025 winner.
    """
    import lightgbm as lgb

    logger.info("=" * 60)
    logger.info("Task 1 Baseline: BM25 + Lexical → LightGBM (5-fold CV)")
    logger.info("=" * 60)

    t0 = time.time()

    doc_ids = sorted(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]

    # Pre-compute BM25 index
    logger.info("Building BM25 index ...")
    bm25 = BM25Index()
    bm25.fit(doc_ids, doc_texts)

    # Pre-compute TF-IDF matrix
    logger.info("Building TF-IDF matrix ...")
    vectorizer = TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words="english", norm="l2")
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    # Pre-compute token sets and bigram sets for Jaccard
    logger.info("Pre-computing token/bigram sets ...")
    token_sets = {}
    bigram_sets = {}
    word_counts = {}
    for did in doc_ids:
        tokens = tokenize(corpus[did])
        token_sets[did] = set(tokens)
        bigram_sets[did] = set(zip(tokens[:-1], tokens[1:])) if len(tokens) > 1 else set()
        word_counts[did] = len(tokens)

    # Legal terms for shared legal terms feature
    LEGAL_TERMS = {
        "judicial", "review", "reasonable", "standard", "evidence", "burden",
        "proof", "procedural", "fairness", "immigration", "refugee", "patent",
        "charter", "rights", "freedoms", "appeal", "dismissed", "allowed",
        "applicant", "respondent", "minister", "officer", "tribunal", "board",
        "decision", "finding", "conclusion", "analysis", "statute", "section",
        "subsection", "paragraph", "precedent", "principle", "test", "factors",
        "consideration", "discretion", "jurisdiction", "natural", "justice",
        "credibility", "assessment",
    }

    # Pre-compute BM25 scores for all queries (the slow part — do it once)
    logger.info("Pre-computing BM25 scores for all queries ...")
    query_ids = sorted(labels.keys())
    query_ids = [q for q in query_ids if q in id_to_idx]

    # Cache BM25 scores and top-200 results per query
    bm25_all_scores = {}  # qid -> full score array (7708 floats)
    bm25_top200 = {}  # qid -> [(doc_id, score), ...]
    for i, qid in enumerate(query_ids):
        q_tokens = tokenize(corpus[qid])
        raw = bm25._score_tokens(q_tokens)
        bm25_all_scores[qid] = raw
        top_idx = np.argsort(-raw)[:201]
        bm25_top200[qid] = [(doc_ids[j], float(raw[j])) for j in top_idx if doc_ids[j] != qid][:200]
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info("  BM25 precompute %d/%d (%.1f q/s, ETA %.0fs)",
                        i + 1, len(query_ids), rate, (len(query_ids) - i - 1) / rate)

    logger.info("BM25 precomputation done in %.1fs", time.time() - t0)

    def compute_features(qid: str, cid: str) -> list[float]:
        """Compute 6 features for a (query, candidate) pair."""
        c_idx = id_to_idx[cid]

        # BM25 score (from cache)
        bm25_score = float(bm25_all_scores[qid][c_idx])

        # TF-IDF cosine (sparse dot product — fast)
        q_vec = tfidf_matrix[id_to_idx[qid]]
        c_vec = tfidf_matrix[c_idx]
        tfidf_cos = float((q_vec @ c_vec.T).toarray()[0, 0])

        # Jaccard
        q_set = token_sets[qid]
        c_set = token_sets[cid]
        union_size = len(q_set | c_set)
        jaccard = len(q_set & c_set) / union_size if union_size > 0 else 0

        # Shared bigrams
        q_bi = bigram_sets[qid]
        c_bi = bigram_sets[cid]
        bi_union = len(q_bi | c_bi)
        bi_jaccard = len(q_bi & c_bi) / bi_union if bi_union > 0 else 0

        # Length ratio
        wc_q = word_counts[qid]
        wc_c = word_counts[cid]
        length_ratio = min(wc_q, wc_c) / max(wc_q, wc_c) if max(wc_q, wc_c) > 0 else 0

        # Shared legal terms
        q_legal = token_sets[qid] & LEGAL_TERMS
        c_legal = token_sets[cid] & LEGAL_TERMS
        shared_legal = len(q_legal & c_legal)

        return [bm25_score, tfidf_cos, jaccard, bi_jaccard, length_ratio, shared_legal]

    # Build training pairs: for each query, positives + sampled negatives
    logger.info("Building training pairs ...")

    all_X = []
    all_y = []
    all_groups = []  # query index for GroupKFold
    all_qids = []
    all_cids = []

    np.random.seed(RANDOM_SEED)
    for qi, qid in enumerate(query_ids):
        positives = [c for c in labels[qid] if c in id_to_idx]
        if not positives:
            continue

        # Use cached BM25 top-200 as candidate pool
        candidate_pool = set(did for did, _ in bm25_top200[qid])
        candidate_pool.update(positives)

        # Negatives: candidates that are NOT in the gold set
        positive_set = set(positives)
        negatives = [c for c in candidate_pool if c not in positive_set]

        # Sample negatives: up to 10x positives, capped at 50
        n_neg = min(len(positives) * 10, 50, len(negatives))
        if negatives and n_neg > 0:
            sampled_negs = list(np.random.choice(negatives, size=n_neg, replace=False))
        else:
            sampled_negs = []

        # Compute features for positives
        for cid in positives:
            all_X.append(compute_features(qid, cid))
            all_y.append(1)
            all_groups.append(qi)
            all_qids.append(qid)
            all_cids.append(cid)

        # Compute features for negatives
        for cid in sampled_negs:
            all_X.append(compute_features(qid, cid))
            all_y.append(0)
            all_groups.append(qi)
            all_qids.append(qid)
            all_cids.append(cid)

        if (qi + 1) % 500 == 0:
            logger.info("  Feature extraction: %d / %d queries", qi + 1, len(query_ids))

    X = np.array(all_X)
    y = np.array(all_y)
    groups = np.array(all_groups)

    logger.info(
        "Training data: %d pairs (%d positive, %d negative) from %d queries",
        len(y), sum(y), len(y) - sum(y), len(set(groups)),
    )
    logger.info("Feature extraction done in %.1fs", time.time() - t0)

    # GroupKFold Cross-Validation
    feature_names = ["bm25_score", "tfidf_cosine", "jaccard", "shared_bigrams", "length_ratio", "shared_legal_terms"]
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_metrics = []
    fold_thresholds = []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info("--- Fold %d ---", fold_i)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train LightGBM
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "is_unbalance": True,
        }
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        # Predict on validation
        val_probs = model.predict(X_val)

        # Build per-query score dicts for threshold optimization
        val_scores: dict[str, list[tuple[str, float]]] = {}
        val_labels_fold: dict[str, list[str]] = {}

        for idx_in_val, orig_idx in enumerate(val_idx):
            qid = all_qids[orig_idx]
            cid = all_cids[orig_idx]
            prob = float(val_probs[idx_in_val])

            if qid not in val_scores:
                val_scores[qid] = []
            val_scores[qid].append((cid, prob))

            if qid not in val_labels_fold:
                val_labels_fold[qid] = labels.get(qid, [])

        best_thresh, best_m = optimize_threshold(val_scores, val_labels_fold)
        fold_metrics.append(best_m)
        fold_thresholds.append(best_thresh)

        logger.info(
            "  Fold %d: F1=%.4f P=%.4f R=%.4f thresh=%.3f",
            fold_i, best_m["f1"], best_m["precision"], best_m["recall"], best_thresh,
        )

    # Aggregate across folds
    avg_f1 = np.mean([m["f1"] for m in fold_metrics])
    avg_p = np.mean([m["precision"] for m in fold_metrics])
    avg_r = np.mean([m["recall"] for m in fold_metrics])
    std_f1 = np.std([m["f1"] for m in fold_metrics])
    avg_thresh = np.mean(fold_thresholds)

    elapsed = time.time() - t0
    result = {
        "model": "BM25+Lexical→LightGBM (5-fold CV)",
        "task": "Task 1",
        "cv_f1_mean": float(avg_f1),
        "cv_f1_std": float(std_f1),
        "cv_precision_mean": float(avg_p),
        "cv_recall_mean": float(avg_r),
        "cv_threshold_mean": float(avg_thresh),
        "fold_f1s": [float(m["f1"]) for m in fold_metrics],
        "time_seconds": elapsed,
    }
    _print_result(result)
    return result


# ─────────────────────────────────────────────
# Task 2 Baselines
# ─────────────────────────────────────────────

TASK2_DIR = DATA_DIR / "task2"


def _load_task2_data():
    """Load Task 2 training data."""
    train_dir = TASK2_DIR / "task2_train_files_2026"
    labels_path = TASK2_DIR / "task2_train_labels_2026.json"

    labels_raw = json.loads(labels_path.read_text())

    # Normalize labels: handle both list and comma-separated string formats
    labels = {}
    for case_id, val in labels_raw.items():
        if isinstance(val, list):
            labels[case_id] = val
        elif isinstance(val, str):
            labels[case_id] = [v.strip() for v in val.split(",") if v.strip()]
        else:
            labels[case_id] = []

    # Load case data
    case_ids = sorted(labels.keys())
    logger.info("Task 2: %d labeled cases from %s", len(case_ids), train_dir)

    cases = {}
    for cid in case_ids:
        case_dir = train_dir / cid
        if not case_dir.is_dir():
            continue

        fragment_path = case_dir / "entailed_fragment.txt"
        para_dir = case_dir / "paragraphs"

        if not fragment_path.exists() or not para_dir.is_dir():
            continue

        fragment = fragment_path.read_text(encoding="utf-8", errors="replace").strip()
        paragraphs = {}
        for p in sorted(para_dir.glob("*.txt")):
            paragraphs[p.name] = p.read_text(encoding="utf-8", errors="replace").strip()

        cases[cid] = {
            "fragment": fragment,
            "paragraphs": paragraphs,
            "labels": labels[cid],
        }

    logger.info("Loaded %d valid cases", len(cases))
    return cases


def task2_bm25_baseline(cases: dict) -> dict:
    """Baseline 1: BM25 score of fragment against each paragraph."""
    logger.info("=" * 60)
    logger.info("Task 2 Baseline: BM25-only")
    logger.info("=" * 60)

    t0 = time.time()

    # Build labels dict and scores dict
    labels_dict: dict[str, list[str]] = {}
    all_scores: dict[str, list[tuple[str, float]]] = {}

    for cid, case in cases.items():
        fragment = case["fragment"]
        paragraphs = case["paragraphs"]
        labels_dict[cid] = case["labels"]

        if not paragraphs:
            all_scores[cid] = []
            continue

        # Build a per-case BM25 index over paragraphs
        para_ids = sorted(paragraphs.keys())
        para_texts = [paragraphs[pid] for pid in para_ids]

        bm25 = BM25Index()
        bm25.fit(para_ids, para_texts)

        # Query with fragment
        results = bm25.query(fragment, top_k=len(para_ids))
        all_scores[cid] = results

    # Optimize threshold
    best_thresh, best_metrics = optimize_threshold(all_scores, labels_dict)

    elapsed = time.time() - t0
    result = {
        "model": "BM25-only",
        "task": "Task 2",
        "threshold": best_thresh,
        **best_metrics,
        "time_seconds": elapsed,
    }
    _print_result(result)
    return result


def task2_tfidf_baseline(cases: dict) -> dict:
    """Baseline 2: TF-IDF cosine similarity of fragment vs each paragraph."""
    logger.info("=" * 60)
    logger.info("Task 2 Baseline: TF-IDF Cosine")
    logger.info("=" * 60)

    t0 = time.time()

    labels_dict: dict[str, list[str]] = {}
    all_scores: dict[str, list[tuple[str, float]]] = {}

    for cid, case in cases.items():
        fragment = case["fragment"]
        paragraphs = case["paragraphs"]
        labels_dict[cid] = case["labels"]

        if not paragraphs:
            all_scores[cid] = []
            continue

        para_ids = sorted(paragraphs.keys())
        para_texts = [paragraphs[pid] for pid in para_ids]

        # Fit TF-IDF on [fragment] + paragraphs
        all_texts = [fragment] + para_texts
        vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english", norm="l2")
        tfidf = vectorizer.fit_transform(all_texts)

        # Cosine similarity of fragment (row 0) vs all paragraphs (rows 1+)
        sim = sklearn_cosine(tfidf[0:1], tfidf[1:]).flatten()
        results = [(para_ids[j], float(sim[j])) for j in np.argsort(-sim)]
        all_scores[cid] = results

    best_thresh, best_metrics = optimize_threshold(all_scores, labels_dict)

    elapsed = time.time() - t0
    result = {
        "model": "TF-IDF Cosine",
        "task": "Task 2",
        "threshold": best_thresh,
        **best_metrics,
        "time_seconds": elapsed,
    }
    _print_result(result)
    return result


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────


def _print_result(result: dict):
    """Print a formatted result summary."""
    logger.info("-" * 60)
    logger.info("RESULT: %s — %s", result["task"], result["model"])
    logger.info("-" * 60)

    if "cv_f1_mean" in result:
        logger.info("  CV F1:    %.4f ± %.4f", result["cv_f1_mean"], result["cv_f1_std"])
        logger.info("  CV P:     %.4f", result["cv_precision_mean"])
        logger.info("  CV R:     %.4f", result["cv_recall_mean"])
        logger.info("  Threshold: %.3f (avg)", result["cv_threshold_mean"])
        logger.info("  Per-fold: %s", [f"{f:.4f}" for f in result["fold_f1s"]])
    else:
        logger.info("  F1:       %.4f", result["f1"])
        logger.info("  P:        %.4f", result["precision"])
        logger.info("  R:        %.4f", result["recall"])
        logger.info("  Threshold: %.3f", result["threshold"])

    logger.info("  Time:     %.1fs", result["time_seconds"])
    logger.info("-" * 60)


def run_all_baselines():
    """Run all baselines and save results."""
    results = []

    # ── Task 1 ──
    corpus, labels = _load_task1_data()

    r1 = task1_bm25_baseline(corpus, labels)
    results.append(r1)

    r2 = task1_tfidf_baseline(corpus, labels)
    results.append(r2)

    r3 = task1_lgbm_baseline(corpus, labels)
    results.append(r3)

    # ── Task 2 ──
    cases = _load_task2_data()

    r4 = task2_bm25_baseline(cases)
    results.append(r4)

    r5 = task2_tfidf_baseline(cases)
    results.append(r5)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("BASELINE SUMMARY")
    logger.info("=" * 60)
    for r in results:
        f1 = r.get("cv_f1_mean", r.get("f1", 0))
        logger.info("  %-40s F1=%.4f  (%.0fs)", f"{r['task']}: {r['model']}", f1, r["time_seconds"])
    logger.info("=" * 60)

    # Save results
    out_path = OUTPUT_DIR / "baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialization
    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.int64)):
                sr[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                sr[k] = float(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            else:
                sr[k] = v
        serializable.append(sr)
    out_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Results saved to %s", out_path)

    return results


if __name__ == "__main__":
    run_all_baselines()
