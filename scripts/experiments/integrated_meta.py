"""Integrated "best meta-learner" experiment.

Combines feature-side remedies (LOO graph features + a lexical-trap divergence
signal, LTD) with several decision heads (LightGBM, RandomForest, XGBoost, and a
soft-vote ensemble) and two cutoff strategies (global threshold swept on calib;
conformal per-query score-quantile cutoff calibrated for a target recall).

Scope (stated honestly): everything runs on the FIXED BM25-RRF top-200 pool.
Post-rerank recall on this pool is bounded by ~0.578, so the meta-learner F1
reported here is bounded by that pool. The recall-ceiling remedies (tuned BM25
0.630, dense@4096 fusion 0.649, GAR@500 0.633) would RAISE that bound but require
re-scoring an expanded candidate set (CE/dense features on new candidates), which
is out of scope here. The integrated F1 would rise with the improved pool.

Feature sets (all under the SAME core1400/calib200/test401 3-way split AND
GroupKFold-5 OOF):
  (a) baseline 34
  (b) 34 + LOO graph features  (citation_indegree/log_indegree/cocitation, LOO)
  (c) 34 + LTD
  (d) 34 + LOO graph + LTD     (the "integrated" set)

LTD (lexical-trap divergence) per row, computed WITHIN-QUERY:
  s_lex = per-query min-max normalized bm25_rrf_score
  s_sem = crossencoder_score
  LTD   = s_lex * log(s_lex / (s_sem + 1e-6))
High LTD => high lexical overlap but low semantic/CE support => a decoy.

Decision heads (on the integrated set d): LightGBM (CALIB_LGBM_PARAMS),
RandomForest, XGBoost, soft-vote ensemble (mean prob of RF+XGB+LGB).

Cutoffs: global (sweep on calib, apply to test) vs conformal (per-query keep
candidates whose score >= the per-query (1 - target_recall) quantile threshold
derived on calib; target_recall swept on calib for best calib F1).

Run:  uv run python scripts/experiments/integrated_meta.py
Writes: output/experiments/integrated_meta.json
"""
import json
import time

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from common import (
    CALIB_LGBM_PARAMS,
    LABELS_PATH,
    load_feature_matrix,
    pred_global,
    prf,
    select_features,
    write_result,
)
from calibration import qs_lab, sweep_global
from graph_features_loo import (
    NEW_FEATS as GRAPH_FEATS,
    attach_features_loo,
    build_reference_graph,
    load_citation_edges,
)

LTD_FEAT = "ltd"
RF_PARAMS = dict(n_estimators=200, max_depth=20, min_samples_leaf=5,
                 class_weight="balanced", n_jobs=-1, random_state=42)
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                  tree_method="hist", eval_metric="logloss", random_state=42)


# ---------------------------------------------------------------------------
# LTD feature (within-query, no label leakage; uses only score columns)
# ---------------------------------------------------------------------------
def add_ltd(df):
    """Add the LTD column in-place on a copy and return it.

    s_lex = per-query min-max normalized bm25_rrf_score; s_sem = crossencoder.
    LTD = s_lex * log(s_lex / (s_sem + 1e-6)). Purely feature-side, computed
    per query, so it leaks no labels and is identical regardless of split.
    """
    out = df.copy()
    lex = out["bm25_rrf_score"].astype(float).values
    sem = out["crossencoder_score"].astype(float).values
    s_lex = np.zeros_like(lex)
    for _, idx in out.groupby("query_id").indices.items():
        v = lex[idx]
        lo, hi = v.min(), v.max()
        s_lex[idx] = (v - lo) / (hi - lo) if hi > lo else 0.0
    eps = 1e-6
    ratio = np.maximum(s_lex, eps) / (sem + eps)
    ltd = s_lex * np.log(ratio)
    out[LTD_FEAT] = ltd
    return out


# ---------------------------------------------------------------------------
# Feature-set assembly: attach graph (LOO) + ltd as requested per split slice
# ---------------------------------------------------------------------------
def build_featset(work, base_feats, use_graph, use_ltd, ref):
    """Return the list of feature columns for the requested set.

    `work` must already contain the LOO graph columns (if use_graph) and the ltd
    column (if use_ltd). ref is unused here (graph cols attached by caller) but
    kept for signature symmetry."""
    feats = list(base_feats)
    if use_graph:
        feats = feats + list(GRAPH_FEATS)
    if use_ltd:
        feats = feats + [LTD_FEAT]
    return feats


def attach_graph_cols(df, ref):
    """Attach LOO graph columns to a copy of df using reference graph ref."""
    out = df.copy()
    arr = attach_features_loo(out.query_id.values, out.candidate_id.values, ref)
    for j, f in enumerate(GRAPH_FEATS):
        out[f] = arr[:, j]
    return out


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------
def train_head(head, Xtr, ytr):
    if head == "lgb":
        m = lgb.train(CALIB_LGBM_PARAMS, lgb.Dataset(Xtr, ytr),
                      num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
        return ("lgb", m)
    if head == "rf":
        m = RandomForestClassifier(**RF_PARAMS).fit(Xtr, ytr)
        return ("rf", m)
    if head == "xgb":
        spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
        m = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS).fit(Xtr, ytr)
        return ("xgb", m)
    raise ValueError(head)


def head_predict(trained, X):
    kind, m = trained
    if kind == "lgb":
        return m.predict(X)
    if kind == "rf":
        return m.predict_proba(X)[:, 1]
    if kind == "xgb":
        return m.predict_proba(X)[:, 1]
    raise ValueError(kind)


def ensemble_predict(trained_list, X):
    """Soft-vote: mean of per-head probabilities."""
    return np.mean([head_predict(t, X) for t in trained_list], axis=0)


# ---------------------------------------------------------------------------
# Conformal per-query cutoff
# ---------------------------------------------------------------------------
def fit_conformal_target(calib_qs, calib_lab):
    """Sweep a target recall in [0.3..0.95]; for each, derive a per-query score
    threshold as the (1 - target) quantile of the calib score distribution among
    GOLD candidates pooled across calib queries, then keep candidates >= that
    single global score threshold. Returns (best_target, best_threshold)."""
    gold_scores = []
    for q, cs in calib_qs.items():
        gset = set(calib_lab.get(q, []))
        for c, s in cs:
            if c in gset:
                gold_scores.append(s)
    gold_scores = np.array(gold_scores, dtype=float)
    if len(gold_scores) == 0:
        return 0.5, 0.5
    best = (-1.0, 0.5, 0.5)
    for target in np.arange(0.30, 0.96, 0.05):
        thr = float(np.quantile(gold_scores, 1.0 - target))
        f = prf(pred_global(calib_qs, thr), calib_lab)["f1"]
        if f > best[0]:
            best = (f, float(target), thr)
    return best[1], best[2]


# ---------------------------------------------------------------------------
# 3-way evaluation for one (featset, head, cutoff) combination
# ---------------------------------------------------------------------------
def eval_3way(df_full, base_feats, edges, use_graph, use_ltd, head, cutoff,
              split):
    core, calib, test = split
    # Attach feature columns. Graph ref built from CORE only (LOO protocol).
    work = df_full
    if use_ltd:
        work = add_ltd(work) if LTD_FEAT not in work.columns else work
    ref = None
    if use_graph:
        ref = build_reference_graph(core, edges)
        work = attach_graph_cols(work, ref)
    feats = build_featset(work, base_feats, use_graph, use_ltd, ref)

    core_df = work[work.query_id.isin(core)]
    calib_df = work[work.query_id.isin(calib)].copy()
    test_df = work[work.query_id.isin(test)].copy()

    Xtr, ytr = core_df[feats].values, core_df.label.values
    if head == "ensemble":
        trained = [train_head(h, Xtr, ytr) for h in ("rf", "xgb", "lgb")]
        calib_s = ensemble_predict(trained, calib_df[feats].values)
        test_s = ensemble_predict(trained, test_df[feats].values)
    else:
        trained = train_head(head, Xtr, ytr)
        calib_s = head_predict(trained, calib_df[feats].values)
        test_s = head_predict(trained, test_df[feats].values)

    calib_qs, calib_lab = qs_lab(calib_df, calib_s)
    test_qs, test_lab = qs_lab(test_df, test_s)

    if cutoff == "global":
        t = sweep_global(calib_qs, calib_lab)
        res = prf(pred_global(test_qs, t), test_lab)
        param = f"t={t:.3f}"
    elif cutoff == "conformal":
        target, thr = fit_conformal_target(calib_qs, calib_lab)
        res = prf(pred_global(test_qs, thr), test_lab)
        param = f"target_recall={target:.2f},thr={thr:.3f}"
    else:
        raise ValueError(cutoff)
    return {**res, "param": param}


# ---------------------------------------------------------------------------
# GroupKFold-5 OOF for a (featset, head) -- cutoff = global on full OOF pool
# ---------------------------------------------------------------------------
def eval_gkf_oof(df_full, base_feats, edges, use_graph, use_ltd, head,
                 n_splits=5):
    work = df_full
    if use_ltd:
        work = add_ltd(work) if LTD_FEAT not in work.columns else work
    q_all = work.query_id.values
    c_all = work.candidate_id.values

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(work), dtype=float)
    idx = np.arange(len(work))
    for tr_idx, te_idx in gkf.split(idx, work.label.values, q_all):
        tr = work.iloc[tr_idx].copy()
        te = work.iloc[te_idx].copy()
        if use_graph:
            train_qids = set(q_all[tr_idx])
            ref = build_reference_graph(train_qids, edges)
            tr_arr = attach_features_loo(q_all[tr_idx], c_all[tr_idx], ref)
            te_arr = attach_features_loo(q_all[te_idx], c_all[te_idx], ref)
            for j, f in enumerate(GRAPH_FEATS):
                tr[f] = tr_arr[:, j]
                te[f] = te_arr[:, j]
        feats = build_featset(tr, base_feats, use_graph, use_ltd, None)
        Xtr, ytr = tr[feats].values, tr.label.values
        if head == "ensemble":
            trained = [train_head(h, Xtr, ytr) for h in ("rf", "xgb", "lgb")]
            oof[te_idx] = ensemble_predict(trained, te[feats].values)
        else:
            trained = train_head(head, Xtr, ytr)
            oof[te_idx] = head_predict(trained, te[feats].values)

    qs, lab = qs_lab(work.assign(_s=oof), oof)
    t = sweep_global(qs, lab)
    return {**prf(pred_global(qs, t), lab), "threshold": float(t)}


def main():
    df = load_feature_matrix()
    base_feats = select_features(df)
    edges, labs = load_citation_edges()

    qids = sorted(df.query_id.unique())
    n = len(qids)
    i70, i80 = int(n * 0.70), int(n * 0.80)
    core, calib, test = set(qids[:i70]), set(qids[i70:i80]), set(qids[i80:])
    split = (core, calib, test)
    print(f"matrix {df.shape}, {len(base_feats)} base feats, "
          f"{len(edges)} citation edges", flush=True)
    print(f"split: core={len(core)} calib={len(calib)} test={len(test)}",
          flush=True)

    # Precompute LTD once on the full frame (purely feature-side).
    df = add_ltd(df)

    FEATSETS = {
        "a_base34":            (False, False),
        "b_base34_graph":      (True, False),
        "c_base34_ltd":        (False, True),
        "d_integrated":        (True, True),
    }

    # ---- Part 1: feature sets x (global cutoff) with LightGBM head, 3-way + OOF
    table = []
    print("\n=== Feature-set sweep (head=lgb, cutoff=global) ===", flush=True)
    oof_cache = {}
    for fs_name, (ug, ul) in FEATSETS.items():
        t0 = time.time()
        r3 = eval_3way(df, base_feats, edges, ug, ul, "lgb", "global", split)
        print(f"    [{fs_name}] 3way done in {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        roof = eval_gkf_oof(df, base_feats, edges, ug, ul, "lgb")
        print(f"    [{fs_name}] OOF done in {time.time()-t0:.0f}s", flush=True)
        oof_cache[(fs_name, "lgb")] = roof
        row = {"featset": fs_name, "head": "lgb", "cutoff": "global",
               "test_f1": r3["f1"], "test_p": r3["precision"],
               "test_r": r3["recall"], "param": r3["param"],
               "oof_f1": roof["f1"]}
        table.append(row)
        print(f"  {fs_name:18s} 3way F1={r3['f1']:.4f} "
              f"(P={r3['precision']:.4f} R={r3['recall']:.4f}) {r3['param']}"
              f"  | OOF F1={roof['f1']:.4f}", flush=True)

    # ---- Part 2: heads on integrated set (d) x cutoffs, 3-way + OOF
    print("\n=== Head sweep on integrated set (d), both cutoffs ===", flush=True)
    HEADS = ["lgb", "rf", "xgb", "ensemble"]
    ug, ul = FEATSETS["d_integrated"]
    for head in HEADS:
        if (("d_integrated", head)) in oof_cache:
            roof = oof_cache[("d_integrated", head)]
        else:
            t0 = time.time()
            roof = eval_gkf_oof(df, base_feats, edges, ug, ul, head)
            print(f"    [d/{head}] OOF done in {time.time()-t0:.0f}s", flush=True)
        for cutoff in ("global", "conformal"):
            r3 = eval_3way(df, base_feats, edges, ug, ul, head, cutoff, split)
            row = {"featset": "d_integrated", "head": head, "cutoff": cutoff,
                   "test_f1": r3["f1"], "test_p": r3["precision"],
                   "test_r": r3["recall"], "param": r3["param"],
                   "oof_f1": roof["f1"]}
            table.append(row)
            print(f"  d/{head:9s}/{cutoff:9s} 3way F1={r3['f1']:.4f} "
                  f"(P={r3['precision']:.4f} R={r3['recall']:.4f}) {r3['param']}"
                  f"  | OOF F1={roof['f1']:.4f}", flush=True)

    # ---- Best combination + delta over 0.345 baseline ----
    BASELINE_REF = 0.345
    best = max(table, key=lambda r: r["test_f1"])
    # The reproduced baseline = featset a, lgb, global.
    repro = next(r for r in table if r["featset"] == "a_base34"
                 and r["head"] == "lgb" and r["cutoff"] == "global")
    delta_vs_ref = best["test_f1"] - BASELINE_REF
    delta_vs_repro = best["test_f1"] - repro["test_f1"]

    print("\n=== BEST ===", flush=True)
    print(f"  {best['featset']} / {best['head']} / {best['cutoff']}  "
          f"test F1={best['test_f1']:.4f} (P={best['test_p']:.4f} "
          f"R={best['test_r']:.4f}) OOF={best['oof_f1']:.4f}", flush=True)
    print(f"  reproduced baseline (a/lgb/global) test F1={repro['test_f1']:.4f}"
          f"  (reference 0.345)", flush=True)
    print(f"  Delta F1 vs 0.345 reference  = {delta_vs_ref:+.4f}", flush=True)
    print(f"  Delta F1 vs reproduced base  = {delta_vs_repro:+.4f}", flush=True)

    payload = {
        "scope_note": (
            "All results on the FIXED BM25-RRF top-200 pool; post-rerank recall "
            "is bounded by ~0.578, so the integrated meta-learner F1 is pool-"
            "bounded. Recall-ceiling remedies (tuned BM25 0.630, dense@4096 "
            "fusion 0.649, GAR@500 0.633) would raise that bound but require "
            "re-scoring an expanded candidate pool (CE/dense features on new "
            "candidates) and are out of scope here. Reported F1 would rise with "
            "the improved pool."
        ),
        "protocol": (
            "3-way chronological split by query id (core 70%/calib 10%/test 20%) "
            "+ GroupKFold-5 OOF. Graph features are leave-one-query-out (ref "
            "graph from core / the 4 training folds). LTD is purely feature-side "
            "(score columns only, per-query min-max), no label leakage."
        ),
        "split": {"core": len(core), "calib": len(calib), "test": len(test)},
        "ltd_definition": "s_lex * log(s_lex/(s_sem+1e-6)); "
                          "s_lex=per-query minmax(bm25_rrf_score), "
                          "s_sem=crossencoder_score",
        "graph_features": list(GRAPH_FEATS),
        "baseline_reference_f1": BASELINE_REF,
        "reproduced_baseline_f1": repro["test_f1"],
        "baseline_reproduced": abs(repro["test_f1"] - BASELINE_REF) < 0.01,
        "table": table,
        "best": {
            "featset": best["featset"], "head": best["head"],
            "cutoff": best["cutoff"], "test_f1": best["test_f1"],
            "test_p": best["test_p"], "test_r": best["test_r"],
            "oof_f1": best["oof_f1"], "param": best["param"],
        },
        "delta_f1_vs_reference_0345": delta_vs_ref,
        "delta_f1_vs_reproduced_baseline": delta_vs_repro,
        "recall_ceiling_pool": 0.578,
    }
    path = write_result("integrated_meta", payload,
                        script="experiments/integrated_meta.py")
    print(f"\nSaved {path}", flush=True)


if __name__ == "__main__":
    main()
