"""Analysis A — EVT/GPD adaptive per-query cutoff vs a transferred global threshold.

Same clean 3-way chronological split (train_core 70% / calib 10% / test 20% by
query id) and ONE LightGBM meta-learner (34 features, CALIB_LGBM_PARAMS) as
scripts/experiments/calibration.py. We then apply the EVT (Generalized Pareto
tail-fit) per-query cutoff from src/coliee_task1/utils/evt_threshold.py to the
test-slice scores and compare test micro-F1 to:
  (i)  global threshold swept on calib, transferred to test (~0.345)
  (ii) the cutoff baselines already in output/w2/b_calibration.json
       (score_gap, top_ratio, fixed_k, count_regressor)

Question: does EVT beat the global threshold? We also sweep EVT's alpha on calib
(no test peeking) and report the calib-selected alpha result, plus the best EVT
over a small alpha grid for reference.

Run:    uv run python scripts/experiments/evt_cutoff.py
Writes: output/experiments/evt_cutoff.json
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy import stats

from common import (
    CALIB_LGBM_PARAMS,
    REPO,
    load_feature_matrix,
    pred_global,
    prf,
    select_features,
    write_result,
)

sys.path.insert(0, str(REPO / "src"))
from coliee_task1.utils.evt_threshold import evt_truncate  # noqa: E402


def evt_truncate_topfit(scores, alpha=0.05, q_null=0.75, min_per_query=1):
    """Corrected EVT cutoff: fit GPD to the bulk (lower scores) and keep only
    the top candidates whose score is a significant upper-tail outlier above the
    bulk (p < alpha). This is the precision-oriented "Surprise" truncation:
    threshold = q_null quantile of the per-query scores defines the null bulk,
    GPD is fit to exceedances of that threshold, and a candidate is kept while
    its survival prob under the fitted tail is below alpha. Selection stops at the
    first candidate that is not surprising (contiguous top block).
    """
    preds = {}
    for qid, cands in scores.items():
        if not cands:
            preds[qid] = []
            continue
        sc = sorted(cands, key=lambda x: -x[1])
        a = np.array([s for _, s in sc], dtype=float)
        if len(a) < 10:
            preds[qid] = [c for c, _ in sc[:min_per_query]]
            continue
        u = np.quantile(a, q_null)
        exc = a[a > u] - u
        keep = min_per_query
        try:
            if len(exc) < 5:
                raise ValueError("few exceedances")
            shape, loc, scale = stats.genpareto.fit(exc, floc=0.0)
            # walk down the ranked list; keep while score is an upper-tail outlier
            keep = 0
            for c, s in sc:
                if s <= u:
                    break
                p = 1.0 - stats.genpareto.cdf(s - u, shape, loc=loc, scale=scale)
                if p < alpha:
                    keep += 1
                else:
                    break
        except (ValueError, RuntimeError):
            keep = min_per_query
        keep = max(keep, min_per_query)
        preds[qid] = [c for c, _ in sc[:keep]]
    return preds


def qs_lab(df, scores):
    qs, lab = {}, {}
    for qid, cid, sc, y in zip(
        df.query_id.values, df.candidate_id.values, scores, df.label.values
    ):
        qs.setdefault(qid, []).append((cid, float(sc)))
        if y == 1:
            lab.setdefault(qid, []).append(cid)
    for qid in qs:
        lab.setdefault(qid, [])
        qs[qid].sort(key=lambda x: -x[1])
    return qs, lab


def sweep_global(qs, lab):
    best = (-1.0, 0.5)
    for t in np.arange(0.01, 1.0, 0.01):
        f = prf(pred_global(qs, float(t)), lab)["f1"]
        if f > best[0]:
            best = (f, float(t))
    return best[1]


def main():
    df = load_feature_matrix()
    feats = select_features(df)
    qids = sorted(df.query_id.unique())
    n = len(qids)
    i70, i80 = int(n * 0.70), int(n * 0.80)
    core, calib, test = set(qids[:i70]), set(qids[i70:i80]), set(qids[i80:])
    print(f"split: core={len(core)} calib={len(calib)} test={len(test)}", flush=True)

    core_df = df[df.query_id.isin(core)]
    model = lgb.train(
        CALIB_LGBM_PARAMS,
        lgb.Dataset(core_df[feats].values, core_df.label.values),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )

    calib_df = df[df.query_id.isin(calib)].copy()
    test_df = df[df.query_id.isin(test)].copy()
    calib_qs, calib_lab = qs_lab(calib_df, model.predict(calib_df[feats].values))
    test_qs, test_lab = qs_lab(test_df, model.predict(test_df[feats].values))

    # --- (i) global threshold swept on calib, transferred to test ---
    t_global = sweep_global(calib_qs, calib_lab)
    global_res = {
        **prf(pred_global(test_qs, t_global), test_lab),
        "param": f"t={t_global:.2f}",
    }

    # --- EVT: sweep alpha on calib, transfer to test (no test peeking) ---
    alpha_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    calib_evt = {}
    for a in alpha_grid:
        preds = evt_truncate(calib_qs, alpha=a, min_per_query=1)
        calib_evt[a] = prf(preds, calib_lab)["f1"]
    best_alpha = max(alpha_grid, key=lambda a: calib_evt[a])

    evt_calib_sel = {
        **prf(evt_truncate(test_qs, alpha=best_alpha, min_per_query=1), test_lab),
        "param": f"alpha={best_alpha} (selected on calib)",
    }

    # EVT alpha grid evaluated directly on test (reference / sensitivity only).
    evt_test_grid = {}
    for a in alpha_grid:
        evt_test_grid[a] = prf(
            evt_truncate(test_qs, alpha=a, min_per_query=1), test_lab
        )
    best_alpha_test = max(alpha_grid, key=lambda a: evt_test_grid[a]["f1"])
    evt_oracle = {
        **evt_test_grid[best_alpha_test],
        "param": f"alpha={best_alpha_test} (peeked on test)",
    }

    # --- corrected top-tail EVT (precision-oriented), alpha swept on calib ---
    calib_evt2 = {}
    for a in alpha_grid:
        calib_evt2[a] = prf(
            evt_truncate_topfit(calib_qs, alpha=a, min_per_query=1), calib_lab
        )["f1"]
    best_alpha2 = max(alpha_grid, key=lambda a: calib_evt2[a])
    evt2_calib_sel = {
        **prf(
            evt_truncate_topfit(test_qs, alpha=best_alpha2, min_per_query=1),
            test_lab,
        ),
        "param": f"alpha={best_alpha2} (selected on calib, top-tail fit)",
    }
    evt2_test_grid = {
        str(a): prf(
            evt_truncate_topfit(test_qs, alpha=a, min_per_query=1), test_lab
        )
        for a in alpha_grid
    }

    # --- baselines already measured (from output/w2/b_calibration.json) ---
    legacy = json.loads((REPO / "output" / "w2" / "b_calibration.json").read_text())
    legacy_baselines = {
        k: legacy["results"][k]
        for k in ("score_gap", "top_ratio", "fixed_k", "count_regressor")
        if k in legacy["results"]
    }

    # best EVT across BOTH module variants (calib-selected) vs global
    best_evt = max(
        [evt_calib_sel, evt2_calib_sel], key=lambda r: r["f1"]
    )
    helps = best_evt["f1"] > global_res["f1"]
    delta = best_evt["f1"] - global_res["f1"]

    payload = {
        "split": {"core": len(core), "calib": len(calib), "test": len(test)},
        "global_threshold": global_res,
        "evt_module_calib_selected": evt_calib_sel,
        "evt_module_oracle_alpha": evt_oracle,
        "evt_topfit_calib_selected": evt2_calib_sel,
        "evt_module_calib_alpha_f1": {str(a): calib_evt[a] for a in alpha_grid},
        "evt_module_test_alpha_grid": {str(a): evt_test_grid[a] for a in alpha_grid},
        "evt_topfit_calib_alpha_f1": {str(a): calib_evt2[a] for a in alpha_grid},
        "evt_topfit_test_alpha_grid": evt2_test_grid,
        "prior_cutoff_baselines": legacy_baselines,
        "best_evt_variant": best_evt,
        "evt_helps_vs_global": bool(helps),
        "evt_minus_global_f1": delta,
        "verdict": (
            f"Best EVT variant {'beats' if helps else 'does NOT beat'} the global "
            f"threshold (delta_F1={delta:+.4f})."
        ),
    }

    path = write_result("evt_cutoff", payload, script="experiments/evt_cutoff.py")
    print(json.dumps(payload, indent=2, default=float), flush=True)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
