"""Sweep thresholds on predict-mode intermediate cache to generate multiple submissions.

Loads cached intermediate scores from a completed predict pipeline run,
rebuilds feature matrix, runs LGBM inference once, then applies multiple
thresholds to produce submission files.

Pickle is used intentionally for pipeline cache compatibility (internal data only).
"""
import json
import logging
import pickle  # noqa: S403 — internal pipeline cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "output" / "pipeline_cache"
MODELS_DIR = PROJECT_ROOT / "output" / "models_v2"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    # Load predict intermediate cache
    cache_path = CACHE_DIR / "predict_intermediate.pkl"
    if not cache_path.exists():
        logger.error("No predict intermediate cache found at %s", cache_path)
        logger.error("Run predict pipeline first: python -m graphrag.run_pipeline_v2 predict")
        return

    logger.info("Loading predict intermediate cache ...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)  # noqa: S301 — internal pipeline data
    logger.info("Cache loaded (keys: %s)", list(cache.keys()))

    # Extract data
    test_labels = cache["test_labels"]
    query_ids = cache["query_ids"]
    rrf_results = cache["rrf_results"]
    bm25_raw = cache["bm25_raw"]
    bm25_rrf = cache["bm25_rrf"]
    biencoder_scores = cache["biencoder_scores"]
    crossencoder_scores = cache["crossencoder_scores"]
    graphrag_features = cache["graphrag_features"]
    ctx_feats = cache["ctx_feats"]
    multi_scores = cache["multi_scores"]
    gnn_scores = cache["gnn_scores"]
    reasoning_scores = cache["reasoning_scores"]
    clean_corpus = cache["clean_corpus"]

    # Build feature matrix (same as stage 6 predict mode)
    from graphrag.meta_learner import build_feature_matrix, FEATURE_COLS
    from graphrag.metrics import scores_to_predictions

    labels = test_labels
    candidate_pool = {
        qid: [cid for cid, _ in rrf_results.get(qid, [])]
        for qid in query_ids
    }

    # Compute lexical features (same as stage 6)
    from graphrag.run_pipeline_v2 import compute_lexical_features
    lexical_features = compute_lexical_features(clean_corpus, candidate_pool)

    logger.info("Building feature matrix ...")
    df_full = build_feature_matrix(
        labels, candidate_pool,
        bm25_scores=bm25_raw,
        bm25_rrf_scores=bm25_rrf,
        biencoder_scores=biencoder_scores,
        crossencoder_scores=crossencoder_scores,
        graphrag_features=graphrag_features,
        context_features=ctx_feats,
        lexical_features=lexical_features,
        multi_scores=multi_scores,
        gnn_scores=gnn_scores,
        reasoning_scores=reasoning_scores,
        subsample=False,
    )
    logger.info("Feature matrix: %d rows, %d features", len(df_full), len(FEATURE_COLS))

    # Load saved models
    model_dir = MODELS_DIR / "meta_learner"
    config = json.loads((model_dir / "config.json").read_text())
    base_threshold = config["threshold"]

    models = []
    n_seeds = config.get("multi_seed_runs", 1)
    if n_seeds > 1:
        for seed_run in range(n_seeds):
            seed_dir = model_dir / f"seed_{seed_run}"
            for f in sorted(seed_dir.glob("fold_*.txt")):
                models.append(lgb.Booster(model_file=str(f)))
    else:
        for f in sorted(model_dir.glob("fold_*.txt")):
            models.append(lgb.Booster(model_file=str(f)))
    logger.info("Loaded %d models (seeds=%d)", len(models), n_seeds)

    # Run inference once
    X = df_full[FEATURE_COLS].values
    raw_scores = np.zeros(len(df_full))
    for model in models:
        raw_scores += model.predict(X)
    raw_scores /= len(models)

    # Build query-level scores
    query_scores: dict[str, list[tuple[str, float]]] = {}
    for i, row in df_full.iterrows():
        qid = row["query_id"]
        cid = row["candidate_id"]
        if qid not in query_scores:
            query_scores[qid] = []
        query_scores[qid].append((cid, raw_scores[i]))

    logger.info("Raw scores: min=%.4f, max=%.4f, mean=%.4f, median=%.4f",
                raw_scores.min(), raw_scores.max(), raw_scores.mean(), np.median(raw_scores))

    # Score distribution analysis
    p90 = np.percentile(raw_scores, 90)
    p95 = np.percentile(raw_scores, 95)
    p99 = np.percentile(raw_scores, 99)
    logger.info("Score percentiles: p90=%.4f, p95=%.4f, p99=%.4f", p90, p95, p99)

    # Sweep thresholds
    thresholds = [
        0.30, 0.35, 0.40, 0.45, 0.50,
        base_threshold,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    ]
    thresholds = sorted(set(thresholds))

    logger.info("\n=== Threshold Sweep ===")
    results = []
    for t in thresholds:
        preds = scores_to_predictions(query_scores, t, min_per_query=1)
        counts = [len(v) for v in preds.values()]
        total = sum(counts)
        avg = total / len(preds) if preds else 0
        marker = " <-- base" if abs(t - base_threshold) < 0.001 else ""
        logger.info("  t=%.3f: %d predictions, avg=%.1f/query, min=%d, max=%d%s",
                     t, total, avg, min(counts), max(counts), marker)
        results.append({
            "threshold": t,
            "total_preds": total,
            "avg_per_query": avg,
            "min_per_query": min(counts),
            "max_per_query": max(counts),
            "predictions": preds,
        })

    # Select top 3 runs: base threshold + one lower (recall) + one higher (precision)
    # Strategy: maximize diversity for the 3 allowed submissions
    base_idx = next(i for i, r in enumerate(results) if abs(r["threshold"] - base_threshold) < 0.001)
    base_result = results[base_idx]

    # Pick lower threshold: target avg ~5-6 predictions (higher recall)
    lower_candidates = [r for r in results if r["avg_per_query"] > base_result["avg_per_query"] + 0.5
                        and r["avg_per_query"] < 8]
    if lower_candidates:
        recall_result = min(lower_candidates, key=lambda r: abs(r["avg_per_query"] - 5.5))
    else:
        recall_result = results[max(0, base_idx - 2)]

    # Pick higher threshold: target avg ~2 predictions (higher precision)
    higher_candidates = [r for r in results if r["avg_per_query"] < base_result["avg_per_query"] - 0.5
                         and r["avg_per_query"] > 1.2]
    if higher_candidates:
        precision_result = min(higher_candidates, key=lambda r: abs(r["avg_per_query"] - 2.0))
    else:
        precision_result = results[min(len(results) - 1, base_idx + 2)]

    # Ensure all 3 are different
    selected = [
        ("run1_balanced", base_result),
        ("run2_recall", recall_result),
        ("run3_precision", precision_result),
    ]

    logger.info("\n=== Selected Runs ===")
    for name, result in selected:
        t = result["threshold"]
        out_path = OUTPUT_DIR / f"submission_{name}.json"
        with open(out_path, "w") as f:
            json.dump(result["predictions"], f, indent=2)
        logger.info("  %s: t=%.3f, %d predictions (avg %.1f/q) -> %s",
                     name, t, result["total_preds"], result["avg_per_query"], out_path)


if __name__ == "__main__":
    main()
