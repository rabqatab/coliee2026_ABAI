"""Calibration-robust per-query decision rules vs a transferred global threshold.

(Authoritative temporal/calibration experiment, from w2_b_calibration.py.)

Clean 3-way chronological split (train_core / calib / test, by query id) with
ONE model:
  - train_core (oldest 70%): train the LightGBM meta-learner
  - calib       (next 10%):  fit ALL decision-rule parameters (no test peeking)
  - test        (newest 20%): evaluate
This fixes the D-refine caveat (threshold fit on calib scores from the SAME
model), whose 0.188 "calibration penalty" was a score-scale artifact.

Decision rules evaluated on test (params fit on calib only):
  global_train   - single global threshold swept on calib, applied to test
  score_gap      - per-query: cut at the largest adjacent score gap (>=1)
  top_ratio      - per-query: keep within r of the query top score (r fit on calib)
  fixed_k        - top-K per query, K = round(mean pool-gold per calib query)
  count_regressor- LightGBM regressor predicts pool-gold count per query; top-round(pred)
Reference upper bounds (peek at test labels -- NOT deployable):
  oracle_global  - global threshold swept on test
  oracle_k       - per query, take top-(true #gold-in-pool)

Run:  uv run python scripts/experiments/calibration.py
Writes: output/experiments/calibration.json  (+ legacy output/w2/b_calibration.json)
"""
import json

import lightgbm as lgb
import numpy as np

from common import (
    CALIB_LGBM_PARAMS,
    REPO,
    load_feature_matrix,
    pred_fixed_k,
    pred_global,
    pred_score_gap,
    pred_top_ratio,
    prf,
    select_features,
    write_result,
)

LEGACY_OUT = REPO / "output" / "w2" / "b_calibration.json"


def qs_lab(df, scores):
    qs, lab = {}, {}
    for qid, cid, sc, y in zip(df.query_id.values, df.candidate_id.values, scores, df.label.values):
        qs.setdefault(qid, []).append((cid, float(sc)))
        if y == 1:
            lab.setdefault(qid, []).append(cid)
    for qid in qs:
        lab.setdefault(qid, [])
        qs[qid].sort(key=lambda x: -x[1])
    return qs, lab


def sweep_global(qs, lab):
    best = (-1, 0.5)
    for t in np.arange(0.01, 1.0, 0.01):
        f = prf(pred_global(qs, float(t)), lab)["f1"]
        if f > best[0]:
            best = (f, float(t))
    return best[1]


def qdist_feats(cs):
    a = np.array([s for _, s in cs], dtype=float)
    n = len(a)
    if n == 0:
        return [0] * 10
    return [n, float(a.max()), float(a.mean()), float(a.std()),
            float(a[0] - a[1]) if n > 1 else 0.0,
            float(a[:3].mean()), float(a[:5].mean()),
            float((a > 0.5).sum()), float((a > 0.3).sum()), float((a > 0.1).sum())]


def main():
    df = load_feature_matrix()
    feats = select_features(df)
    qids = sorted(df.query_id.unique())
    n = len(qids)
    i70, i80 = int(n * 0.70), int(n * 0.80)
    core, calib, test = set(qids[:i70]), set(qids[i70:i80]), set(qids[i80:])
    print(f"split: core={len(core)} calib={len(calib)} test={len(test)}", flush=True)

    core_df = df[df.query_id.isin(core)]
    model = lgb.train(CALIB_LGBM_PARAMS, lgb.Dataset(core_df[feats].values, core_df.label.values),
                      num_boost_round=300, callbacks=[lgb.log_evaluation(0)])

    calib_df = df[df.query_id.isin(calib)].copy()
    test_df = df[df.query_id.isin(test)].copy()
    calib_qs, calib_lab = qs_lab(calib_df, model.predict(calib_df[feats].values))
    test_qs, test_lab = qs_lab(test_df, model.predict(test_df[feats].values))

    # --- fit rule params on calib ---
    t_global = sweep_global(calib_qs, calib_lab)
    best_r = max(np.arange(0.5, 0.99, 0.02),
                 key=lambda r: prf(pred_top_ratio(calib_qs, float(r)), calib_lab)["f1"])
    fixed_k = int(round(np.mean([len(v) for v in calib_lab.values()])))
    Xc = np.array([qdist_feats(calib_qs[q]) for q in calib_qs])
    yc = np.array([len(calib_lab[q]) for q in calib_qs], dtype=float)
    reg = lgb.train({"objective": "regression_l2", "num_leaves": 15, "learning_rate": 0.05,
                     "verbose": -1}, lgb.Dataset(Xc, yc), num_boost_round=200)

    def pred_count_reg(qs):
        out = {}
        for q, cs in qs.items():
            k = int(round(float(reg.predict(np.array([qdist_feats(cs)]))[0])))
            out[q] = [c for c, _ in cs[:max(1, k)]]
        return out

    # --- evaluate on test ---
    res = {
        "global_train": {**prf(pred_global(test_qs, t_global), test_lab), "param": f"t={t_global:.2f}"},
        "score_gap": prf(pred_score_gap(test_qs), test_lab),
        "top_ratio": {**prf(pred_top_ratio(test_qs, best_r), test_lab), "param": f"r={best_r:.2f}"},
        "fixed_k": {**prf(pred_fixed_k(test_qs, fixed_k), test_lab), "param": f"k={fixed_k}"},
        "count_regressor": prf(pred_count_reg(test_qs), test_lab),
        "oracle_global": {**prf(pred_global(test_qs, sweep_global(test_qs, test_lab)), test_lab),
                          "param": f"t={sweep_global(test_qs, test_lab):.2f}"},
    }
    ok = {q: [c for c, _ in test_qs[q][:len(test_lab[q])]] for q in test_qs}
    res["oracle_k"] = prf(ok, test_lab)

    base = res["global_train"]["f1"]
    orac = res["oracle_global"]["f1"]
    span = max(orac - base, 1e-9)
    for k, v in res.items():
        v["recovery_frac"] = (v["f1"] - base) / span
    payload = {"split": {"core": len(core), "calib": len(calib), "test": len(test)},
               "results": res, "baseline_f1": base, "oracle_global_f1": orac,
               "calibration_penalty": orac - base}

    path = write_result("calibration", payload, script="experiments/calibration.py")
    LEGACY_OUT.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_OUT.write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps(payload, indent=2, default=float), flush=True)
    print(f"\nSaved {path}\nSaved {LEGACY_OUT}")


if __name__ == "__main__":
    main()
