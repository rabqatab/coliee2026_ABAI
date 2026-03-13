"""Ablation study: measure each pipeline component's contribution.

Loads the saved feature matrix from training and retrains the LightGBM
meta-learner with incrementally more feature groups. This takes seconds
since only the lightweight meta-learner is retrained, not the neural models.

Usage:
    uv run python scripts/run_ablation.py
    # Or inside Docker:
    PYTHONPATH=/workspace/coliee2026/src python /workspace/coliee2026/scripts/run_ablation.py
"""
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ablation")

# --- Feature groups ---
FEATURE_GROUPS = {
    "bm25": ["bm25_score", "bm25_rrf_score"],
    "context": ["n_context_matches", "max_context_bm25"],
    "biencoder": ["biencoder_score", "biencoder_rank"],
    "crossencoder": ["crossencoder_score", "crossencoder_rank"],
    "graphrag": [
        "same_community_0.5", "same_community_1.0", "same_community_2.0",
        "community_jaccard", "shared_statutes", "shared_judges",
        "same_domain", "same_outcome", "entity_overlap_score",
    ],
}

# --- Ablation configurations (incremental) ---
ABLATION_CONFIGS = {
    "A: BM25 only":             ["bm25"],
    "B: + Citation Context":    ["bm25", "context"],
    "C: + Bi-encoder":          ["bm25", "context", "biencoder"],
    "D: + Cross-encoder":       ["bm25", "context", "biencoder", "crossencoder"],
    "E: + GraphRAG Lite":       ["bm25", "context", "biencoder", "crossencoder", "graphrag"],
    # Single-component ablations (drop one at a time from full)
    "F: Full - GraphRAG":       ["bm25", "context", "biencoder", "crossencoder"],
    "G: Full - Cross-encoder":  ["bm25", "context", "biencoder", "graphrag"],
    "H: Full - Bi-encoder":     ["bm25", "context", "crossencoder", "graphrag"],
    "I: Full - Context":        ["bm25", "biencoder", "crossencoder", "graphrag"],
}

# LightGBM parameters (same as main pipeline)
LGBM_PARAMS = {
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
N_ESTIMATORS = 500
EARLY_STOPPING = 50
N_FOLDS = 5
RANDOM_SEED = 42


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    """Train LightGBM with GroupKFold CV on given features, return metrics."""
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["query_id"].values
    query_ids = df["query_id"].values
    candidate_ids = df["candidate_id"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_scores = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_train, y_train)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING),
                lgb.log_evaluation(period=0),  # suppress per-round logging
            ],
        )
        oof_scores[val_idx] = model.predict(X_val)

    # Convert to query-level scores and optimize threshold
    query_scores: dict[str, list[tuple[str, float]]] = {}
    labels_dict: dict[str, list[str]] = {}

    for qid, cid, score, label in zip(query_ids, candidate_ids, oof_scores, y):
        query_scores.setdefault(qid, []).append((cid, score))
        if label == 1:
            labels_dict.setdefault(qid, []).append(cid)
    for qid in query_scores:
        labels_dict.setdefault(qid, [])

    # Threshold sweep
    best_f1, best_threshold = 0.0, 0.5
    best_metrics = {}
    for t in np.arange(0.01, 1.0, 0.01):
        preds = {}
        for qid, candidates in query_scores.items():
            preds[qid] = [cid for cid, s in candidates if s >= t]

        tp = fp = fn = 0
        for qid in labels_dict:
            true_set = set(labels_dict[qid])
            pred_set = set(preds.get(qid, []))
            tp += len(true_set & pred_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_metrics = {"f1": f1, "precision": p, "recall": r,
                            "tp": tp, "fp": fp, "fn": fn,
                            "threshold": best_threshold}

    return best_metrics


def main():
    # Locate feature matrix
    project_root = Path(__file__).parent.parent
    feature_path = project_root / "output" / "feature_matrix.parquet"

    if not feature_path.exists():
        logger.error("Feature matrix not found at %s", feature_path)
        logger.error("Run the training pipeline first to generate it.")
        sys.exit(1)

    df = pd.read_parquet(feature_path)
    n_pos = df["label"].sum()
    n_neg = len(df) - n_pos
    logger.info("Loaded feature matrix: %d rows, %d pos / %d neg", len(df), n_pos, n_neg)

    # Run ablation
    results = {}
    logger.info("=" * 70)
    logger.info("  ABLATION STUDY")
    logger.info("=" * 70)

    for config_name, groups in ABLATION_CONFIGS.items():
        feature_cols = []
        for g in groups:
            feature_cols.extend(FEATURE_GROUPS[g])

        # Verify all features exist in DataFrame
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            logger.warning("  %s: SKIPPED (missing features: %s)", config_name, missing)
            continue

        logger.info("  Running: %s (%d features)", config_name, len(feature_cols))
        metrics = train_and_evaluate(df, feature_cols)
        results[config_name] = metrics

        logger.info(
            "    F1=%.4f  P=%.4f  R=%.4f  (threshold=%.2f)",
            metrics["f1"], metrics["precision"], metrics["recall"],
            metrics["threshold"],
        )

    # Summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("  ABLATION RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("  %-30s  %8s  %8s  %8s  %5s", "Configuration", "F1", "Prec", "Recall", "Thr")
    logger.info("  " + "-" * 67)

    prev_f1 = 0.0
    for config_name, metrics in results.items():
        f1 = metrics["f1"]
        delta = f1 - prev_f1 if not config_name.startswith("F:") else 0.0
        delta_str = f"(+{delta:.4f})" if delta > 0 and not config_name.startswith(("F:", "G:", "H:", "I:")) else ""
        logger.info(
            "  %-30s  %8.4f  %8.4f  %8.4f  %5.2f  %s",
            config_name, f1, metrics["precision"], metrics["recall"],
            metrics["threshold"], delta_str,
        )
        if not config_name.startswith(("F:", "G:", "H:", "I:")):
            prev_f1 = f1

    # Drop-one analysis
    if "E: + GraphRAG Lite" in results:
        full_f1 = results["E: + GraphRAG Lite"]["f1"]
        logger.info("")
        logger.info("  Drop-one analysis (impact of removing each component):")
        drop_configs = {
            "GraphRAG": "F: Full - GraphRAG",
            "Cross-encoder": "G: Full - Cross-encoder",
            "Bi-encoder": "H: Full - Bi-encoder",
            "Context": "I: Full - Context",
        }
        for component, config_key in drop_configs.items():
            if config_key in results:
                drop_f1 = results[config_key]["f1"]
                logger.info("    Drop %-15s: F1=%.4f (delta=%.4f)", component, drop_f1, drop_f1 - full_f1)

    logger.info("=" * 70)

    # Save results
    output_path = project_root / "output" / "ablation_results.json"
    output_path.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
