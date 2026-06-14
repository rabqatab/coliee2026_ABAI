"""W2 Remedy B: calibration-robust per-query decision rules vs a transferred global threshold.

Clean 3-way chronological split (train_core / calib / test, by query id) with ONE model:
  - train_core (oldest 70%): train the LightGBM meta-learner
  - calib       (next 10%):  fit ALL decision-rule parameters (no test peeking)
  - test        (newest 20%): evaluate
This also fixes the D-refine caveat (threshold fit on calib scores from the SAME model).

Decision rules evaluated on test (params fit on calib only):
  global_train   - single global threshold swept on calib, applied to test  (the broken baseline)
  score_gap      - per-query: cut at the largest adjacent score gap (threshold-free), >=1
  top_ratio      - per-query: keep candidates within `r` of the query's top score (r fit on calib)
  fixed_k        - take top-K per query, K = round(mean pool-gold per calib query)
  count_regressor- LightGBM regressor predicts pool-gold count per query from score-dist features; top-round(pred)
Reference upper bounds (peek at test labels — NOT deployable):
  oracle_global  - global threshold swept on test
  oracle_k       - per query, take top-(true #gold-in-pool)
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FM = REPO / "output" / "feature_matrix.parquet"
OUT = REPO / "output" / "w2" / "b_calibration.json"

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
LGBM = {"objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "is_unbalance": True}


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


def prf(preds, lab):
    tp = fp = fn = 0
    for qid in lab:
        t = set(lab[qid]); p = set(preds.get(qid, []))
        tp += len(t & p); fp += len(p - t); fn += len(t - p)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"f1": F, "precision": P, "recall": R}


def pred_global(qs, t):
    return {q: [c for c, s in cs if s >= t] for q, cs in qs.items()}


def sweep_global(qs, lab):
    best = (-1, 0.5)
    for t in np.arange(0.01, 1.0, 0.01):
        f = prf(pred_global(qs, float(t)), lab)["f1"]
        if f > best[0]:
            best = (f, float(t))
    return best[1]


def pred_score_gap(qs, min_k=1):
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


def pred_top_ratio(qs, r, min_k=1):
    out = {}
    for q, cs in qs.items():
        if not cs:
            out[q] = []; continue
        top = cs[0][1]
        keep = [c for c, s in cs if s >= r * top]
        if len(keep) < min_k:
            keep = [c for c, _ in cs[:min_k]]
        out[q] = keep
    return out


def pred_fixed_k(qs, k):
    return {q: [c for c, _ in cs[:max(1, k)]] for q, cs in qs.items()}


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
    df = pd.read_parquet(FM)
    feats = [f for f in FEATURES_34 if f in df.columns]
    assert len(feats) == 34, f"got {len(feats)} features"
    qids = sorted(df.query_id.unique())
    n = len(qids)
    i70, i80 = int(n * 0.70), int(n * 0.80)
    core, calib, test = set(qids[:i70]), set(qids[i70:i80]), set(qids[i80:])
    print(f"split: core={len(core)} calib={len(calib)} test={len(test)}", flush=True)

    core_df = df[df.query_id.isin(core)]
    model = lgb.train(LGBM, lgb.Dataset(core_df[feats].values, core_df.label.values),
                      num_boost_round=300, callbacks=[lgb.log_evaluation(0)])

    calib_df = df[df.query_id.isin(calib)].copy()
    test_df = df[df.query_id.isin(test)].copy()
    calib_qs, calib_lab = qs_lab(calib_df, model.predict(calib_df[feats].values))
    test_qs, test_lab = qs_lab(test_df, model.predict(test_df[feats].values))

    # --- fit rule params on calib ---
    t_global = sweep_global(calib_qs, calib_lab)
    # best top-ratio r on calib
    best_r = max(np.arange(0.5, 0.99, 0.02),
                 key=lambda r: prf(pred_top_ratio(calib_qs, float(r)), calib_lab)["f1"])
    fixed_k = int(round(np.mean([len(v) for v in calib_lab.values()])))
    # count regressor: fit on calib queries (model didn't train on them)
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
    # oracle per-query k
    ok = {q: [c for c, _ in test_qs[q][:len(test_lab[q])]] for q in test_qs}
    res["oracle_k"] = prf(ok, test_lab)

    base = res["global_train"]["f1"]
    orac = res["oracle_global"]["f1"]
    span = max(orac - base, 1e-9)
    for k, v in res.items():
        v["recovery_frac"] = (v["f1"] - base) / span
    out = {"split": {"core": len(core), "calib": len(calib), "test": len(test)},
           "results": res, "baseline_f1": base, "oracle_global_f1": orac,
           "calibration_penalty": orac - base}
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    sys.exit(main())
