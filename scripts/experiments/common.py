"""Shared building blocks for the COLIEE 2026 journal experiments.

Factored out of the three authoritative ad-hoc scripts (w1_c1_full.py,
w1_e_stage_metrics.py, w2_b_calibration.py) so that the clean modules under
`scripts/experiments/` share one definition of the feature list, the feature
matrix loader, the ranking/scoring metrics, the per-query cutoff rules, and the
result writer.

Nothing here calls wall-clock time or random in a way that affects results.
`write_result` stamps each output with the git short SHA and the calling script
name only.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = scripts/experiments/common.py -> parents[2].
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from coliee_task1.config import OUTPUT_DIR  # noqa: E402
from coliee_task1.utils.metrics import (  # noqa: E402,F401  (re-exported)
    mean_average_precision,
    ndcg_at_k,
    recall_at_k,
)

# --- Paths -----------------------------------------------------------------
FEATURE_MATRIX = OUTPUT_DIR / "feature_matrix.parquet"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
PIPELINE_CACHE = OUTPUT_DIR / "pipeline_cache"
LABELS_PATH = REPO / "data" / "task1" / "task1_train_labels_2026.json"

# --- The 34 features reported in the paper (verbatim from w2_b_calibration) --
FEATURES_34 = [
    "bm25_score", "bm25_rrf_score", "n_context_matches", "max_context_bm25",
    "tfidf_cosine", "jaccard", "shared_bigrams", "length_ratio", "shared_legal_terms",
    "biencoder_score", "biencoder_rank", "m3_dense_score", "m3_sparse_score",
    "m3_colbert_score", "m3_fused_score", "crossencoder_score", "crossencoder_rank",
    "same_community_0.5", "same_community_1.0", "same_community_2.0", "community_jaccard",
    "shared_statutes", "shared_judges", "same_domain", "same_outcome",
    "entity_overlap_score", "gnn_score", "gnn_rank",
    "bm25_rrf_rank_norm", "bm25_rrf_score_gap", "biencoder_score_gap",
    "crossencoder_score_gap", "top_score_ratio", "score_above_median",
]

# LightGBM params for the calibration meta-learner. Kept inline (NOT config's
# LGBM_PARAMS) to preserve the authoritative w2_b_calibration.py numbers:
# num_leaves=63, lr=0.05, is_unbalance=True, no early stopping.
CALIB_LGBM_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "is_unbalance": True,
}


# --- Feature matrix loader -------------------------------------------------
def load_feature_matrix() -> pd.DataFrame:
    """Read the honest RRF-only feature matrix from output/feature_matrix.parquet."""
    return pd.read_parquet(FEATURE_MATRIX)


def select_features(df: pd.DataFrame) -> list[str]:
    """Return the FEATURES_34 columns present in df (asserts all 34 exist)."""
    feats = [f for f in FEATURES_34 if f in df.columns]
    assert len(feats) == 34, f"got {len(feats)} features, expected 34"
    return feats


# --- Ranking / scoring metrics ---------------------------------------------
# recall_at_k, ndcg_at_k, mean_average_precision are re-exported from
# coliee_task1.utils.metrics above. prf is the micro P/R/F1 used by the
# calibration experiment (set-based, over the per-query prediction dicts).
def prf(preds: dict, lab: dict) -> dict:
    """Micro precision/recall/F1 over {qid: [ids]} predictions vs labels."""
    tp = fp = fn = 0
    for qid in lab:
        t = set(lab[qid])
        p = set(preds.get(qid, []))
        tp += len(t & p)
        fp += len(p - t)
        fn += len(t - p)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"f1": F, "precision": P, "recall": R}


# --- Per-query cutoff rules (from w2_b_calibration) ------------------------
def pred_global(qs: dict, t: float) -> dict:
    """Global threshold: keep candidates with score >= t."""
    return {q: [c for c, s in cs if s >= t] for q, cs in qs.items()}


def pred_score_gap(qs: dict, min_k: int = 1) -> dict:
    """Per-query: cut at the largest adjacent score gap (threshold-free), >=min_k."""
    out = {}
    for q, cs in qs.items():
        s = [x[1] for x in cs]
        if len(s) < 2:
            cut = 1
        else:
            gaps = [s[i] - s[i + 1] for i in range(len(s) - 1)]
            gi = int(np.argmax(gaps))
            cut = gi + 1 if (gaps[gi] > 0.05 and gi < len(s) // 2) else min_k
        cut = max(cut, min_k)
        out[q] = [c for c, _ in cs[:cut]]
    return out


def pred_top_ratio(qs: dict, r: float, min_k: int = 1) -> dict:
    """Per-query: keep candidates with score >= r * (query top score)."""
    out = {}
    for q, cs in qs.items():
        if not cs:
            out[q] = []
            continue
        top = cs[0][1]
        keep = [c for c, s in cs if s >= r * top]
        if len(keep) < min_k:
            keep = [c for c, _ in cs[:min_k]]
        out[q] = keep
    return out


def pred_fixed_k(qs: dict, k: int) -> dict:
    """Per-query: take the top-k candidates."""
    return {q: [c for c, _ in cs[:max(1, k)]] for q, cs in qs.items()}


# --- Result writer ---------------------------------------------------------
def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_result(name: str, obj: dict, script: str | None = None) -> Path:
    """Write obj to output/experiments/<name>.json with metadata.

    Metadata injected: git short SHA and the calling script name (no
    wall-clock / random). Returns the written path.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "git_sha": _git_short_sha(),
            "script": script or Path(sys.argv[0]).name,
        },
        **obj,
    }
    path = EXPERIMENTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path
