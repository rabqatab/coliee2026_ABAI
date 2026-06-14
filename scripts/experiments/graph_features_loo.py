"""Journal experiment #2 (CORRECTED): Graph-as-features with leave-one-query-out.

Convert the citation-graph / structure signal into LightGBM meta-learner
features (citation indegree, log-indegree, co-citation) and measure Delta F1.

----------------------------------------------------------------------------
THE BUG (in graph_features.py) and THE FIX
----------------------------------------------------------------------------
The original `attach_features` assigns each candidate its GLOBAL citation
indegree from the reference (core / training-fold) graph, regardless of which
query it is paired with. So for a TRAINING query q in the reference graph and
its gold candidate c, indegree[c] INCLUDES q's own q->c edge. Every core gold
therefore has indegree >= 1, while held-out test golds (whose query is not in
the reference graph) only sometimes have indegree > 0. The model learns this
inflated core-gold signal and collapses at test: a train/test feature
distribution mismatch on the positive class.

THE FIX -- leave-one-query-out (LOO) features. For each (query q, candidate c):

    loo_indegree(q, c) = ref_indegree[c] - (1 if q in ref_graph and q cites c
                                            else 0)

For reference (training) queries this subtracts q's own edge; for held-out
queries (not in the reference graph) it subtracts nothing (correct: q never
contributed). Same LOO for cocitation_count. log_indegree = log1p(loo_indegree).

`pagerank` and `community_indegree` cannot be cheaply LOO'd, so they are DROPPED
from this corrected run (stated explicitly).

Reference graph per protocol:
  - 3-way split: built from CORE queries only. LOO subtracts the own-edge for
    core rows; calib/test rows (queries not in core) are untouched.
  - GroupKFold(5) OOF: for each held-out fold, reference = the 4 training folds.
    LOO subtracts the own-edge for training rows; held-out rows untouched.

Baseline  = FEATURES_34 meta-learner (calibration.py protocol).
Treatment = FEATURES_34 + {citation_indegree (LOO), log_indegree (LOO),
            cocitation_count (LOO)}.

Run:  uv run python scripts/experiments/graph_features_loo.py
Writes: output/experiments/graph_features_loo.json  (does NOT overwrite the old
        graph_features.json).
"""
import json

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

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

# LOO-able graph features only (pagerank / community_indegree dropped).
NEW_FEATS = [
    "citation_indegree",
    "log_indegree",
    "cocitation_count",
]


def load_citation_edges():
    """Return list of (query_id, cited_candidate_id) directed edges from labels."""
    labs = json.loads(LABELS_PATH.read_text())
    edges = []
    for q, cited in labs.items():
        for c in cited:
            edges.append((q, c))
    return edges, labs


def build_reference_graph(ref_query_ids, edges):
    """Build the reference citation structure from edges whose SOURCE query is in
    `ref_query_ids`.

    Returns:
      ref_indeg  : cand_id -> # DISTINCT reference queries citing it.
      ref_cocite : cand_id -> co-citation incidences (sum over its citers of
                   (out_degree(citer) - 1)).
      q_cites    : query_id -> set(cand_ids it cites)   (reference queries only).
      q_outdeg   : query_id -> # DISTINCT cands it cites (reference queries only).
    """
    ref_set = set(ref_query_ids)

    q_cites = {}
    for q, c in edges:
        if q in ref_set:
            q_cites.setdefault(q, set()).add(c)
    q_outdeg = {q: len(cs) for q, cs in q_cites.items()}

    # indegree = number of DISTINCT reference queries citing the candidate.
    ref_indeg = {}
    for q, cs in q_cites.items():
        for c in cs:
            ref_indeg[c] = ref_indeg.get(c, 0) + 1

    # cocitation: for candidate c, sum over its citers q of (out_degree(q) - 1).
    ref_cocite = {}
    for q, cs in q_cites.items():
        contrib = q_outdeg[q] - 1
        for c in cs:
            ref_cocite[c] = ref_cocite.get(c, 0) + contrib

    return ref_indeg, ref_cocite, q_cites, q_outdeg


def attach_features_loo(query_ids, candidate_ids, ref):
    """Per-(query, candidate) LOO graph features.

    ref = (ref_indeg, ref_cocite, q_cites, q_outdeg).
    For each row, subtract the contribution of q's OWN edge when q is a
    reference query that cites c.
    """
    ref_indeg, ref_cocite, q_cites, q_outdeg = ref
    n = len(query_ids)
    arr = np.zeros((n, len(NEW_FEATS)), dtype=float)
    idx_ind = NEW_FEATS.index("citation_indegree")
    idx_log = NEW_FEATS.index("log_indegree")
    idx_coc = NEW_FEATS.index("cocitation_count")

    for i, (q, c) in enumerate(zip(query_ids, candidate_ids)):
        d = ref_indeg.get(c, 0)
        coc = ref_cocite.get(c, 0)
        cited = q_cites.get(q)  # None if q not a reference query
        if cited is not None and c in cited:
            # q is a reference query and q->c is in the reference graph: remove it.
            d -= 1                       # one fewer distinct citer
            coc -= (q_outdeg[q] - 1)     # remove q's co-citation contribution for c
        d = max(d, 0)
        coc = max(coc, 0)
        arr[i, idx_ind] = float(d)
        arr[i, idx_log] = float(np.log1p(d))
        arr[i, idx_coc] = float(coc)
    return arr


def gold_indegree_distribution(query_ids, candidate_ids, labels, ref):
    """For the gold (label==1) rows in the given slice, report the distribution
    of the LOO citation_indegree feature."""
    arr = attach_features_loo(query_ids, candidate_ids, ref)
    idx_ind = NEW_FEATS.index("citation_indegree")
    mask = labels == 1
    vals = arr[mask, idx_ind]
    if len(vals) == 0:
        return {"n": 0, "frac_gt0": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "n": int(len(vals)),
        "frac_gt0": float((vals > 0).mean()),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
    }


def three_way(df, feats, edges, treatment):
    """calibration.py protocol: core 70 / calib 10 / test 20 by query id.

    Returns test result, trained model, feats used, split sizes, and (when
    treatment) the train-vs-test gold indegree verification distributions.
    """
    qids = sorted(df.query_id.unique())
    n = len(qids)
    i70, i80 = int(n * 0.70), int(n * 0.80)
    core, calib, test = set(qids[:i70]), set(qids[i70:i80]), set(qids[i80:])

    work = df.copy()
    verify = None
    use_feats = feats
    if treatment:
        ref = build_reference_graph(core, edges)
        arr = attach_features_loo(work.query_id.values, work.candidate_id.values, ref)
        for j, f in enumerate(NEW_FEATS):
            work[f] = arr[:, j]
        use_feats = feats + NEW_FEATS

        core_mask = work.query_id.isin(core).values
        test_mask = work.query_id.isin(test).values
        verify = {
            "core_train_golds": gold_indegree_distribution(
                work.query_id.values[core_mask], work.candidate_id.values[core_mask],
                work.label.values[core_mask], ref),
            "test_golds": gold_indegree_distribution(
                work.query_id.values[test_mask], work.candidate_id.values[test_mask],
                work.label.values[test_mask], ref),
        }

    core_df = work[work.query_id.isin(core)]
    calib_df = work[work.query_id.isin(calib)].copy()
    test_df = work[work.query_id.isin(test)].copy()

    model = lgb.train(
        CALIB_LGBM_PARAMS,
        lgb.Dataset(core_df[use_feats].values, core_df.label.values),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )
    calib_qs, calib_lab = qs_lab(calib_df, model.predict(calib_df[use_feats].values))
    test_qs, test_lab = qs_lab(test_df, model.predict(test_df[use_feats].values))

    t = sweep_global(calib_qs, calib_lab)
    test_res = prf(pred_global(test_qs, t), test_lab)
    return ({**test_res, "threshold": float(t)}, model, use_feats,
            len(core), len(calib), len(test), verify)


def groupkfold_oof(df, feats, edges, treatment, n_splits=5):
    """GroupKFold(5) OOF F1. Reference graph (treatment) built from the 4 training
    folds, LOO applied so a query's own gold edges never enter its candidates'
    indegree. One global threshold swept on the full OOF prediction pool."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df.query_id.values
    oof = np.zeros(len(df), dtype=float)

    q_all = df.query_id.values
    c_all = df.candidate_id.values

    X_idx = np.arange(len(df))
    for tr_idx, te_idx in gkf.split(X_idx, df.label.values, groups):
        tr = df.iloc[tr_idx].copy()
        te = df.iloc[te_idx].copy()
        use_feats = feats
        if treatment:
            train_qids = set(q_all[tr_idx])
            ref = build_reference_graph(train_qids, edges)
            tr_arr = attach_features_loo(q_all[tr_idx], c_all[tr_idx], ref)
            te_arr = attach_features_loo(q_all[te_idx], c_all[te_idx], ref)
            for j, f in enumerate(NEW_FEATS):
                tr[f] = tr_arr[:, j]
                te[f] = te_arr[:, j]
            use_feats = feats + NEW_FEATS
        model = lgb.train(
            CALIB_LGBM_PARAMS,
            lgb.Dataset(tr[use_feats].values, tr.label.values),
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)],
        )
        oof[te_idx] = model.predict(te[use_feats].values)

    qs, lab = qs_lab(df.assign(_s=oof), oof)
    t = sweep_global(qs, lab)
    return {**prf(pred_global(qs, t), lab), "threshold": float(t)}


def main():
    df = load_feature_matrix()
    feats = select_features(df)
    edges, labs = load_citation_edges()
    print(f"loaded matrix {df.shape}, {len(feats)} base feats, "
          f"{len(edges)} citation edges over {len(labs)} queries", flush=True)
    print(f"NEW (LOO) features: {NEW_FEATS}  "
          f"(pagerank & community_indegree DROPPED -- not cheaply LOO'able)", flush=True)

    # --- 3-way split (calibration.py protocol) ---
    base_test, _, _, ncore, ncalib, ntest, _ = three_way(df, feats, edges, treatment=False)
    treat_test, treat_model, use_feats, _, _, _, verify = three_way(
        df, feats, edges, treatment=True)
    print(f"split: core={ncore} calib={ncalib} test={ntest}", flush=True)
    print(f"3way baseline  test F1={base_test['f1']:.4f} "
          f"(P={base_test['precision']:.4f} R={base_test['recall']:.4f} t={base_test['threshold']:.2f})",
          flush=True)
    print(f"3way treatment test F1={treat_test['f1']:.4f} "
          f"(P={treat_test['precision']:.4f} R={treat_test['recall']:.4f} t={treat_test['threshold']:.2f})",
          flush=True)
    delta = treat_test["f1"] - base_test["f1"]
    print(f"3way Delta F1 = {delta:+.4f}", flush=True)

    # --- VERIFICATION: gold indegree distribution after LOO ---
    print("\n--- LOO verification: gold citation_indegree distribution ---", flush=True)
    ctr = verify["core_train_golds"]
    tst = verify["test_golds"]
    print(f"  core(train) golds: n={ctr['n']:5d}  frac>0={ctr['frac_gt0']:.3f}  "
          f"mean={ctr['mean']:.3f}  max={ctr['max']:.0f}", flush=True)
    print(f"  test golds       : n={tst['n']:5d}  frac>0={tst['frac_gt0']:.3f}  "
          f"mean={tst['mean']:.3f}  max={tst['max']:.0f}", flush=True)
    print("  (before fix these were ~1.00/1.92 vs ~0.37/0.80 -- should now be comparable)",
          flush=True)

    # --- GroupKFold-5 OOF ---
    base_oof = groupkfold_oof(df, feats, edges, treatment=False)
    treat_oof = groupkfold_oof(df, feats, edges, treatment=True)
    print(f"\nGKF5 baseline  OOF F1={base_oof['f1']:.4f}  treatment OOF F1={treat_oof['f1']:.4f}  "
          f"Delta={treat_oof['f1'] - base_oof['f1']:+.4f}", flush=True)

    # --- feature importance (treatment 3-way model, gain) ---
    gains = treat_model.feature_importance(importance_type="gain")
    names = use_feats
    order = np.argsort(gains)[::-1]
    ranked = [(names[i], float(gains[i])) for i in order]
    rank_of = {name: r + 1 for r, (name, _) in enumerate(ranked)}
    n_total = len(names)
    new_importance = {}
    for f in NEW_FEATS:
        new_importance[f] = {
            "gain": float(gains[names.index(f)]),
            "rank": rank_of[f],
            "of": n_total,
        }
    print(f"\nFeature importance (gain) ranks out of {n_total}:", flush=True)
    for f in NEW_FEATS:
        ni = new_importance[f]
        print(f"  {f:22s} rank {ni['rank']:2d}/{ni['of']}  gain={ni['gain']:.1f}", flush=True)
    print("\nTop 10 features by gain:", flush=True)
    for name, g in ranked[:10]:
        tag = " <-- NEW" if name in NEW_FEATS else ""
        print(f"  {name:24s} {g:12.1f}{tag}", flush=True)

    payload = {
        "protocol": "calibration.py 3-way (core70/calib10/test20 by query id) + GroupKFold5 OOF",
        "fix": (
            "Leave-one-query-out (LOO) graph features: for each (q, c), subtract "
            "q's own citation edge from the reference-graph indegree/cocitation "
            "when q is in the reference graph. Removes the train/test "
            "positive-class feature distribution mismatch that made every "
            "core gold have indegree>=1. pagerank & community_indegree DROPPED "
            "(not cheaply LOO'able)."
        ),
        "split": {"core": ncore, "calib": ncalib, "test": ntest},
        "n_citation_edges": len(edges),
        "new_features": NEW_FEATS,
        "dropped_features": ["pagerank", "community_indegree"],
        "loo_verification_gold_indegree": verify,
        "three_way": {
            "baseline_f1": base_test["f1"],
            "treatment_f1": treat_test["f1"],
            "delta_f1": delta,
            "baseline": base_test,
            "treatment": treat_test,
        },
        "groupkfold5_oof": {
            "baseline_f1": base_oof["f1"],
            "treatment_f1": treat_oof["f1"],
            "delta_f1": treat_oof["f1"] - base_oof["f1"],
            "baseline": base_oof,
            "treatment": treat_oof,
        },
        "new_feature_importance": new_importance,
        "top10_features_by_gain": ranked[:10],
    }
    path = write_result("graph_features_loo", payload,
                        script="experiments/graph_features_loo.py")
    print(f"\nSaved {path}", flush=True)


if __name__ == "__main__":
    main()
