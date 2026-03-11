"""Scoring metrics and threshold optimization for COLIEE Task 1."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def micro_f1(
    predictions: dict[str, list[str]],
    labels: dict[str, list[str]],
) -> dict[str, float]:
    """Compute micro-averaged F1, precision, and recall.

    This matches the official COLIEE evaluation metric.
    """
    tp = 0
    fp = 0
    fn = 0

    for query_id in labels:
        true_set = set(labels[query_id])
        pred_set = set(predictions.get(query_id, []))

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def scores_to_predictions(
    scores: dict[str, list[tuple[str, float]]],
    threshold: float,
) -> dict[str, list[str]]:
    """Convert scored candidates to binary predictions using a threshold.

    Args:
        scores: dict mapping query_id -> [(candidate_id, score), ...]
        threshold: minimum score to predict as positive

    Returns:
        dict mapping query_id -> [predicted_candidate_ids]
    """
    predictions = {}
    for query_id, candidates in scores.items():
        predictions[query_id] = [
            cid for cid, score in candidates if score >= threshold
        ]
    return predictions


def optimize_threshold(
    scores: dict[str, list[tuple[str, float]]],
    labels: dict[str, list[str]],
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the threshold that maximizes micro-F1 on the given data.

    Returns:
        (best_threshold, best_metrics_dict)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}

    for t in thresholds:
        preds = scores_to_predictions(scores, float(t))
        metrics = micro_f1(preds, labels)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(t)
            best_metrics = metrics

    logger.info(
        "Best threshold=%.3f: F1=%.4f P=%.4f R=%.4f",
        best_threshold,
        best_metrics.get("f1", 0),
        best_metrics.get("precision", 0),
        best_metrics.get("recall", 0),
    )
    return best_threshold, best_metrics
