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
    min_per_query: int = 0,
) -> dict[str, list[str]]:
    """Convert scored candidates to binary predictions using a threshold.

    Args:
        scores: dict mapping query_id -> [(candidate_id, score), ...]
        threshold: minimum score to predict as positive
        min_per_query: minimum predictions per query (0 = no guarantee,
                       1 = always include top-1 even if below threshold)

    Returns:
        dict mapping query_id -> [predicted_candidate_ids]
    """
    predictions = {}
    for query_id, candidates in scores.items():
        above = [cid for cid, score in candidates if score >= threshold]
        if len(above) < min_per_query and candidates:
            # Add top-K candidates by score to meet minimum
            sorted_cands = sorted(candidates, key=lambda x: -x[1])
            top_ids = [cid for cid, _ in sorted_cands[:min_per_query]]
            above = list(dict.fromkeys(above + top_ids))  # deduplicate, preserve order
        predictions[query_id] = above
    return predictions


def adaptive_threshold_predictions(
    scores: dict[str, list[tuple[str, float]]],
    global_threshold: float,
    method: str = "score_gap",
    min_per_query: int = 1,
) -> dict[str, list[str]]:
    """Per-query adaptive thresholding.

    Methods:
        "score_gap": Find largest gap in sorted scores, predict above the gap.
                     Falls back to global threshold if gap is ambiguous.
        "relative":  Predict candidates within top_ratio of the query's max score.
        "hybrid":    Use global threshold as floor, but also include candidates
                     within 80% of the top score.

    Args:
        scores: {query_id: [(candidate_id, score), ...]}
        global_threshold: fallback threshold
        method: one of "score_gap", "relative", "hybrid"
        min_per_query: minimum predictions per query

    Returns:
        {query_id: [predicted_candidate_ids]}
    """
    predictions = {}

    for query_id, candidates in scores.items():
        if not candidates:
            predictions[query_id] = []
            continue

        sorted_cands = sorted(candidates, key=lambda x: -x[1])
        cand_scores = [s for _, s in sorted_cands]

        if method == "score_gap":
            # Find the largest score gap among consecutive candidates
            if len(cand_scores) < 2:
                cut_idx = 1
            else:
                gaps = [cand_scores[i] - cand_scores[i + 1] for i in range(len(cand_scores) - 1)]
                max_gap_idx = int(np.argmax(gaps))
                # Only use gap if it's meaningful (> 0.05) and position is reasonable
                if gaps[max_gap_idx] > 0.05 and max_gap_idx < len(cand_scores) // 2:
                    cut_idx = max_gap_idx + 1
                else:
                    # Fall back to global threshold
                    cut_idx = sum(1 for s in cand_scores if s >= global_threshold)

        elif method == "relative":
            top_score = cand_scores[0]
            relative_thresh = top_score * 0.8
            cut_idx = sum(1 for s in cand_scores if s >= relative_thresh)

        elif method == "hybrid":
            top_score = cand_scores[0]
            relative_thresh = top_score * 0.8
            cut_idx = sum(
                1 for s in cand_scores
                if s >= global_threshold or s >= relative_thresh
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        cut_idx = max(cut_idx, min_per_query)
        predictions[query_id] = [cid for cid, _ in sorted_cands[:cut_idx]]

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
        # Derive sweep range from actual score distribution
        all_scores = [s for cands in scores.values() for _, s in cands]
        if all_scores:
            lo = min(all_scores)
            hi = max(all_scores)
            thresholds = np.linspace(lo, hi, 200)
        else:
            thresholds = np.arange(0.01, 1.0, 0.01)

    best_f1 = -1.0
    best_threshold = 0.5
    best_metrics: dict[str, float] = {"f1": 0, "precision": 0, "recall": 0, "tp": 0, "fp": 0, "fn": 0}

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
