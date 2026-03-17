"""Run harness: runs baselines, computes metrics, outputs comparison."""
import json
import logging
import time
from pathlib import Path

import pandas as pd

from baselines.common.base_model import BaselineModel
from baselines.common.data_loader import Dataset
from coliee_task1.utils.metrics import micro_f1, optimize_threshold, scores_to_predictions

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output" / "baselines"


def assess_baseline(
    model: BaselineModel,
    dataset: Dataset,
    bm25_candidates: dict[str, list[tuple[str, float]]] | None = None,
) -> dict:
    """Train and assess a single baseline model. Returns metrics dict."""
    name = model.name()
    logger.info("=" * 60)
    logger.info("  Running: %s", name)
    logger.info("=" * 60)

    # Train
    t0 = time.time()
    model.train(
        corpus=dataset.corpus,
        train_queries=dataset.train_queries,
        labels=dataset.labels,
        bm25_candidates=bm25_candidates,
    )
    train_time = time.time() - t0
    logger.info("  Training: %.1fs", train_time)

    # Optimize threshold on training queries (filter labels to train set only)
    train_scores = model.predict_batch(
        dataset.train_queries, dataset.corpus, bm25_candidates,
    )
    train_labels = {q: dataset.labels[q] for q in dataset.train_queries if q in dataset.labels}
    train_threshold, train_metrics = optimize_threshold(train_scores, train_labels)
    logger.info("  Train threshold: %.3f (F1=%.4f)", train_threshold, train_metrics["f1"])

    # Predict on validation queries
    t0 = time.time()
    val_scores = model.predict_batch(
        dataset.val_queries, dataset.corpus, bm25_candidates,
    )
    infer_time = time.time() - t0
    infer_per_query = infer_time / max(len(dataset.val_queries), 1)

    # Apply threshold (filter labels to val set only)
    val_preds = scores_to_predictions(val_scores, train_threshold)
    val_labels = {q: dataset.labels[q] for q in dataset.val_queries if q in dataset.labels}
    val_metrics = micro_f1(val_preds, val_labels)

    logger.info(
        "  Val: F1=%.4f  P=%.4f  R=%.4f  (threshold=%.3f)",
        val_metrics["f1"], val_metrics["precision"], val_metrics["recall"],
        train_threshold,
    )
    logger.info("  Inference: %.1fs total (%.3fs/query)", infer_time, infer_per_query)

    return {
        "name": name,
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "threshold": train_threshold,
        "train_f1": train_metrics["f1"],
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "infer_per_query_s": infer_per_query,
        "val_tp": val_metrics["tp"],
        "val_fp": val_metrics["fp"],
        "val_fn": val_metrics["fn"],
    }


def run_comparison(
    baselines: list[BaselineModel],
    dataset: Dataset,
    bm25_candidates: dict[str, list[tuple[str, float]]] | None = None,
) -> pd.DataFrame:
    """Run all baselines and produce comparison table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for model in baselines:
        try:
            result = assess_baseline(model, dataset, bm25_candidates)
            results.append(result)
        except Exception as e:
            logger.error("  FAILED: %s — %s", model.name(), e, exc_info=True)
            results.append({"name": model.name(), "val_f1": 0.0, "error": str(e)})

    df = pd.DataFrame(results)

    # Print comparison table
    logger.info("")
    logger.info("=" * 80)
    logger.info("  BASELINE COMPARISON — COLIEE 2026 Task 1 (Val Set)")
    logger.info("=" * 80)
    logger.info("  %-35s  %7s  %7s  %7s  %6s  %8s",
                "Baseline", "F1", "Prec", "Recall", "Thr", "Train(s)")
    logger.info("  " + "-" * 77)
    for _, row in df.iterrows():
        logger.info(
            "  %-35s  %7.4f  %7.4f  %7.4f  %6.3f  %8.1f",
            row.get("name", "?"),
            row.get("val_f1", 0),
            row.get("val_precision", 0),
            row.get("val_recall", 0),
            row.get("threshold", 0),
            row.get("train_time_s", 0),
        )
    logger.info("=" * 80)

    # Save outputs
    df.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(results, indent=2, default=float)
    )
    logger.info("Results saved to %s", OUTPUT_DIR)

    return df
