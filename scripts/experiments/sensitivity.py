"""Reviewer-requested hyperparameter sensitivity sweeps on the cached feature matrix.

All sweeps use the SAME clean 3-way chronological split (train_core 70% / calib 10%
/ test 20%, by query id) as scripts/experiments/calibration.py, with one LightGBM
meta-learner per configuration. F1 is the micro set-based F1 on TEST, using a single
global threshold swept on CALIB (no test peeking).

Three feasible sweeps (no stage re-runs):
  1. Leiden-resolution contribution: ablate same_community_{0.5,1.0,2.0} individually
     and all-three together; report delta-F1 vs baseline.
  2. Feature-group ablation refresh: drop lexical / graph / BGE-M3 / GNN groups.
  3. LightGBM hyperparameter sensitivity: sweep num_leaves, learning_rate,
     min_child_samples, feature_fraction (one axis at a time around the default).

Run: uv run python scripts/experiments/sensitivity.py
Writes: output/experiments/sensitivity.json
"""
import copy

import lightgbm as lgb
import numpy as np

from common import (
    CALIB_LGBM_PARAMS,
    FEATURES_34,
    load_feature_matrix,
    pred_global,
    prf,
    select_features,
    write_result,
)

# Feature groups (exactly as specified by the reviewer request).
LEXICAL = ["tfidf_cosine", "jaccard", "shared_bigrams", "length_ratio", "shared_legal_terms"]
GRAPH = ["same_community_0.5", "same_community_1.0", "same_community_2.0",
         "community_jaccard", "shared_statutes", "shared_judges",
         "same_domain", "same_outcome", "entity_overlap_score"]
BGE_M3 = ["m3_dense_score", "m3_sparse_score", "m3_colbert_score", "m3_fused_score"]
GNN = ["gnn_score", "gnn_rank"]
LEIDEN = ["same_community_0.5", "same_community_1.0", "same_community_2.0"]


def qs_lab(df, scores):
    qs, lab = {}, {}
    for qid, cid, sc, y in zip(df.query_id.values, df.candidate_id.values,
                               scores, df.label.values):
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


def run_config(df, feats, params, splits):
    """Train on core, sweep threshold on calib, evaluate test F1. Returns (f1, t)."""
    core, calib, test = splits
    core_df = df[df.query_id.isin(core)]
    model = lgb.train(
        params,
        lgb.Dataset(core_df[feats].values, core_df.label.values),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )
    calib_df = df[df.query_id.isin(calib)]
    test_df = df[df.query_id.isin(test)]
    calib_qs, calib_lab = qs_lab(calib_df, model.predict(calib_df[feats].values))
    test_qs, test_lab = qs_lab(test_df, model.predict(test_df[feats].values))
    t = sweep_global(calib_qs, calib_lab)
    f1 = prf(pred_global(test_qs, t), test_lab)["f1"]
    return float(f1), float(t)


def main():
    df = load_feature_matrix()
    feats_all = select_features(df)  # 34 features, asserts presence
    qids = sorted(df.query_id.unique())
    n = len(qids)
    i70, i80 = int(n * 0.70), int(n * 0.80)
    core, calib, test = set(qids[:i70]), set(qids[i70:i80]), set(qids[i80:])
    splits = (core, calib, test)
    print(f"split: core={len(core)} calib={len(calib)} test={len(test)}", flush=True)

    # --- Baseline: all 34, default CALIB params -------------------------------
    base_f1, base_t = run_config(df, feats_all, CALIB_LGBM_PARAMS, splits)
    print(f"baseline F1={base_f1:.4f} (t={base_t:.2f})", flush=True)

    # --- Sweep 1: Leiden-resolution contribution -----------------------------
    leiden = {}
    for res in LEIDEN:
        feats = [f for f in feats_all if f != res]
        f1, t = run_config(df, feats, CALIB_LGBM_PARAMS, splits)
        leiden[f"drop_{res}"] = {"f1": f1, "delta_f1": f1 - base_f1, "t": t,
                                 "n_features": len(feats)}
        print(f"  leiden drop {res}: F1={f1:.4f} dF1={f1-base_f1:+.4f}", flush=True)
    feats = [f for f in feats_all if f not in LEIDEN]
    f1, t = run_config(df, feats, CALIB_LGBM_PARAMS, splits)
    leiden["drop_all_three"] = {"f1": f1, "delta_f1": f1 - base_f1, "t": t,
                                "n_features": len(feats)}
    print(f"  leiden drop ALL: F1={f1:.4f} dF1={f1-base_f1:+.4f}", flush=True)

    # --- Sweep 2: feature-group ablation refresh -----------------------------
    groups = {"lexical": LEXICAL, "graph": GRAPH, "bge_m3": BGE_M3, "gnn": GNN}
    group_abl = {}
    for gname, gfeats in groups.items():
        feats = [f for f in feats_all if f not in gfeats]
        f1, t = run_config(df, feats, CALIB_LGBM_PARAMS, splits)
        group_abl[f"drop_{gname}"] = {"f1": f1, "delta_f1": f1 - base_f1, "t": t,
                                      "n_dropped": len(gfeats), "n_features": len(feats)}
        print(f"  group drop {gname} ({len(gfeats)}): F1={f1:.4f} dF1={f1-base_f1:+.4f}",
              flush=True)

    # --- Sweep 3: LightGBM hyperparameter sensitivity ------------------------
    # One axis at a time, varied around the default CALIB params, all 34 features.
    axes = {
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.02, 0.05, 0.1],
        "min_child_samples": [10, 20, 50],
        "feature_fraction": [0.6, 0.8, 1.0],
    }
    hp = {}
    all_f1s = []
    for axis, values in axes.items():
        axis_res = {}
        for v in values:
            params = copy.deepcopy(CALIB_LGBM_PARAMS)
            params[axis] = v
            f1, t = run_config(df, feats_all, params, splits)
            axis_res[str(v)] = {"f1": f1, "t": t}
            all_f1s.append(f1)
            print(f"  hp {axis}={v}: F1={f1:.4f}", flush=True)
        f1vals = [axis_res[str(v)]["f1"] for v in values]
        axis_res["_summary"] = {
            "min": float(min(f1vals)), "max": float(max(f1vals)),
            "range": float(max(f1vals) - min(f1vals)), "std": float(np.std(f1vals)),
            "default_value": CALIB_LGBM_PARAMS.get(axis),
        }
        hp[axis] = axis_res
    hp["_overall"] = {
        "min": float(min(all_f1s)), "max": float(max(all_f1s)),
        "range": float(max(all_f1s) - min(all_f1s)), "std": float(np.std(all_f1s)),
        "n_configs": len(all_f1s),
    }
    print(f"  hp overall: min={min(all_f1s):.4f} max={max(all_f1s):.4f} "
          f"range={max(all_f1s)-min(all_f1s):.4f} std={np.std(all_f1s):.4f}", flush=True)

    # --- Sensitivities that CANNOT be done from cached features ---------------
    requires_rerun = [
        {"name": "citation-context window size (+/-150 words)",
         "current_value": "150 words around <FRAGMENT_SUPPRESSED>",
         "requires": "re-run Stage 1 (citation_context.py) + Stage 2 (bm25.py)",
         "reason": "context windows are extracted then BM25-scored upstream of the "
                   "cached matrix; sweeping window size changes n_context_matches, "
                   "max_context_bm25, and the BM25/RRF features, none recomputable here."},
        {"name": "entity type weights (statute 0.50 / judge 0.15 / domain 0.10 / outcome 0.05)",
         "current_value": "statute=0.50, judge=0.15, domain=0.10, outcome=0.05",
         "requires": "re-run Stage 5 (graphrag_lite.py) graph-feature recompute",
         "reason": "entity_overlap_score and the shared_* graph features are weighted "
                   "sums fixed at compute time; re-weighting needs the raw entity sets, "
                   "not present in the cached matrix."},
        {"name": "number of hard negatives (7) in bi-encoder training",
         "current_value": "7 hard negatives per positive",
         "requires": "re-run Stage 3 (finetune_biencoder.py) retrain",
         "reason": "biencoder_score / biencoder_rank / biencoder_score_gap come from a "
                   "trained model; changing the negative count requires retraining the "
                   "bi-encoder and re-scoring, not possible from cached scores."},
    ]

    payload = {
        "split": {"core": len(core), "calib": len(calib), "test": len(test)},
        "baseline": {"f1": base_f1, "t": base_t, "n_features": len(feats_all),
                     "params": "CALIB_LGBM_PARAMS (num_leaves=63, lr=0.05, ff=0.8)"},
        "sweep1_leiden_resolution": leiden,
        "sweep2_feature_group_ablation": group_abl,
        "sweep3_lgbm_hyperparameters": hp,
        "requires_stage_reruns": requires_rerun,
    }
    path = write_result("sensitivity", payload, script="experiments/sensitivity.py")
    print(f"\nSaved {path}", flush=True)


if __name__ == "__main__":
    main()
