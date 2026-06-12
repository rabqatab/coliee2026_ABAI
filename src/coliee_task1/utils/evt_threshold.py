"""EVT-based adaptive list truncation.

Uses Generalized Pareto Distribution (GPD) to model the tail of the
score distribution per query. Candidates with scores above the GPD
threshold (p < alpha) are predicted as relevant.

Ref: Bahri et al. (SIGIR 2023) — "Surprise: Result List Truncation
via Extreme Value Theory"
"""
import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def evt_truncate(
    scores: dict[str, list[tuple[str, float]]],
    alpha: float = 0.05,
    min_per_query: int = 1,
    min_tail_size: int = 10,
    fallback_threshold: float = 0.5,
) -> dict[str, list[str]]:
    """Per-query adaptive truncation using Extreme Value Theory.

    For each query:
    1. Sort candidates by score (descending)
    2. Fit a GPD to the bottom half of scores (the "null" distribution)
    3. Compute survival probability for each candidate
    4. Predict as relevant if p(score | null) < alpha (i.e., score is surprising)

    Args:
        scores: {query_id: [(candidate_id, score), ...]}
        alpha: Significance level for "surprising" scores (lower = more selective)
        min_per_query: Minimum predictions per query
        min_tail_size: Minimum candidates needed to fit GPD (fall back otherwise)
        fallback_threshold: Used when GPD fitting fails

    Returns:
        {query_id: [predicted_candidate_ids]}
    """
    predictions = {}

    for query_id, candidates in scores.items():
        if not candidates:
            predictions[query_id] = []
            continue

        sorted_cands = sorted(candidates, key=lambda x: -x[1])
        cand_scores = np.array([s for _, s in sorted_cands])

        if len(cand_scores) < min_tail_size:
            above = [cid for cid, s in sorted_cands if s >= fallback_threshold]
            if len(above) < min_per_query:
                above = [cid for cid, _ in sorted_cands[:min_per_query]]
            predictions[query_id] = above
            continue

        # Use bottom half as null distribution
        n_tail = len(cand_scores) // 2
        tail_scores = cand_scores[-n_tail:]

        try:
            threshold_val = tail_scores[0]
            exceedances = tail_scores - tail_scores[-1]
            exceedances = exceedances[exceedances > 0]

            if len(exceedances) < 5:
                raise ValueError("Not enough exceedances")

            shape, loc, scale = stats.genpareto.fit(exceedances)

            shifted_scores = cand_scores - tail_scores[-1]
            p_values = 1 - stats.genpareto.cdf(shifted_scores, shape, loc=loc, scale=scale)

            selected = [
                cid for (cid, _), p in zip(sorted_cands, p_values)
                if p < alpha
            ]
        except (ValueError, RuntimeError):
            gaps = np.diff(-cand_scores)
            if len(gaps) > 0:
                max_gap_idx = int(np.argmax(gaps))
                if gaps[max_gap_idx] > 0.03 and max_gap_idx < len(cand_scores) // 3:
                    selected = [cid for cid, _ in sorted_cands[:max_gap_idx + 1]]
                else:
                    selected = [cid for cid, s in sorted_cands if s >= fallback_threshold]
            else:
                selected = [cid for cid, s in sorted_cands if s >= fallback_threshold]

        if len(selected) < min_per_query:
            selected = [cid for cid, _ in sorted_cands[:min_per_query]]

        predictions[query_id] = selected

    return predictions
