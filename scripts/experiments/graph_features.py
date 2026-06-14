"""Journal experiment #2: Graph-as-features.

Convert the citation-graph / structure signal into LightGBM meta-learner
features (citation indegree, pagerank, co-citation) instead of feeding the GNN
in as a separate score, and measure the resulting Delta F1.

Citation graph: directed edge query -> cited_candidate for every gold pair in
the TRAINING labels (2001 queries, 8251 edges). Node ids are the .txt document
ids shared by the feature matrix and the label file.

Leakage control (critical):
  citation_indegree / pagerank / cocitation for a (query, candidate) pair are
  computed OUT-OF-FOLD: the edges contributed by the query's own evaluation
  fold are EXCLUDED, so a gold edge can never inform the indegree of its own
  target. Concretely:
    - 3-way split: graph built from `core` query edges only; applied to
      core/calib/test (calib & test queries never contribute edges, and core
      rows are scored with a graph that does include their own edges -- but the
      reported metric is the held-out `test` fold, which is fully OOF).
    - GroupKFold(5) OOF: for each held-out fold f, build the graph from the
      other 4 folds' query edges only, then score fold f's rows.

Baseline  = FEATURES_34 meta-learner (calibration.py protocol).
Treatment = FEATURES_34 + {citation_indegree, log_indegree, pagerank,
            cocitation_count, community_indegree}.

Run:  uv run python scripts/experiments/graph_features.py
Writes: output/experiments/graph_features.json
"""
import json

import lightgbm as lgb
import numpy as np
import networkx as nx
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

NEW_FEATS = [
    "citation_indegree",
    "log_indegree",
    "pagerank",
    "cocitation_count",
    "community_indegree",
]


def load_citation_edges():
    """Return list of (query_id, cited_candidate_id) directed edges from labels."""
    labs = json.loads(LABELS_PATH.read_text())
    edges = []
    for q, cited in labs.items():
        for c in cited:
            edges.append((q, c))
    return edges, labs


def build_graph_features(train_query_ids, edges, candidate_community=None):
    """Build per-candidate graph features from edges whose SOURCE query is in
    `train_query_ids` (the training side of a split).

    Returns dict cand_id -> {feature_name: value}. Candidates absent from the
    sub-graph implicitly get 0 (handled by the caller via .get).
    """
    train_set = set(train_query_ids)
    sub = [(q, c) for (q, c) in edges if q in train_set]

    # Directed citation graph query -> cited.
    G = nx.DiGraph()
    G.add_edges_from(sub)

    # citation_indegree = number of DISTINCT training queries citing this cand.
    # (DiGraph collapses duplicate edges, so in_degree == distinct citers.)
    indeg = dict(G.in_degree())

    # pagerank on the directed graph (authority of cited docs).
    if G.number_of_nodes() > 0:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-08)
    else:
        pr = {}

    # cocitation_count: for a candidate, how many (citer, other-cand) co-citation
    # incidences it participates in = sum over its citers of (out_degree(citer)-1).
    # i.e. total number of times this candidate is co-cited alongside some other
    # candidate by a shared citer.
    out_deg = dict(G.out_degree())  # citer -> #candidates it cites
    cocite = {}
    for q, c in sub:
        cocite[c] = cocite.get(c, 0) + (out_deg.get(q, 0) - 1)

    # community_indegree: indegree restricted to citers in the same graphrag
    # community as the candidate is not available here (community is a per-pair
    # feature, not a node attribute). We instead expose the candidate's total
    # community size proxy via indegree weighted by distinct citers already in
    # indeg; to keep it meaningful and leakage-safe we define community_indegree
    # as the indegree among citers that themselves are cited (hub-of-hubs), i.e.
    # in_degree over the subgraph induced on nodes with out_degree>0. This is a
    # cheap structural authority-within-the-active-graph signal.
    active = {n for n in G.nodes() if out_deg.get(n, 0) > 0}
    comm_indeg = {}
    for q, c in sub:
        if q in active:
            comm_indeg[c] = comm_indeg.get(c, 0) + 1

    feats = {}
    for c in set(indeg) | set(pr) | set(cocite):
        d = indeg.get(c, 0)
        feats[c] = {
            "citation_indegree": float(d),
            "log_indegree": float(np.log1p(d)),
            "pagerank": float(pr.get(c, 0.0)),
            "cocitation_count": float(cocite.get(c, 0)),
            "community_indegree": float(comm_indeg.get(c, 0)),
        }
    return feats


def attach_features(df_idx_cands, gf):
    """Map candidate ids -> array of NEW_FEATS columns (0 for unseen cands)."""
    n = len(df_idx_cands)
    arr = np.zeros((n, len(NEW_FEATS)), dtype=float)
    zero = {f: 0.0 for f in NEW_FEATS}
    for i, c in enumerate(df_idx_cands):
        row = gf.get(c, zero)
        for j, f in enumerate(NEW_FEATS):
            arr[i, j] = row[f]
    return arr


def train_predict(train_df, eval_df, feats):
    model = lgb.train(
        CALIB_LGBM_PARAMS,
        lgb.Dataset(train_df[feats].values, train_df.label.values),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model, model.predict(eval_df[feats].values)


def three_way(df, feats, edges, treatment, extra_cols=None):
    """calibration.py protocol: core 70 / calib 10 / test 20 by query id.

    Returns test-F1 (global_train threshold swept on calib), the trained model,
    and the test query set.
    """
    qids = sorted(df.query_id.unique())
    n = len(qids)
    i70, i80 = int(n * 0.70), int(n * 0.80)
    core, calib, test = set(qids[:i70]), set(qids[i70:i80]), set(qids[i80:])

    work = df.copy()
    if treatment:
        # Graph built from CORE queries only -> calib & test are fully OOF.
        gf = build_graph_features(core, edges)
        arr = attach_features(work.candidate_id.values, gf)
        for j, f in enumerate(NEW_FEATS):
            work[f] = arr[:, j]
        use_feats = feats + NEW_FEATS
    else:
        use_feats = feats

    core_df = work[work.query_id.isin(core)]
    calib_df = work[work.query_id.isin(calib)].copy()
    test_df = work[work.query_id.isin(test)].copy()

    model, _ = train_predict(core_df, core_df.head(1), use_feats)
    calib_qs, calib_lab = qs_lab(calib_df, model.predict(calib_df[use_feats].values))
    test_qs, test_lab = qs_lab(test_df, model.predict(test_df[use_feats].values))

    t = sweep_global(calib_qs, calib_lab)
    test_res = prf(pred_global(test_qs, t), test_lab)
    return {**test_res, "threshold": float(t)}, model, use_feats, len(core), len(calib), len(test)


def groupkfold_oof(df, feats, edges, treatment, n_splits=5):
    """GroupKFold(5) OOF F1. Graph (treatment) built from the 4 training folds
    only, applied to the held-out fold. One global threshold swept on the full
    OOF prediction pool (sanity ~0.26 baseline)."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df.query_id.values
    oof = np.zeros(len(df), dtype=float)

    work = df.copy()
    if treatment:
        for f in NEW_FEATS:
            work[f] = 0.0

    X_idx = np.arange(len(df))
    for tr_idx, te_idx in gkf.split(X_idx, df.label.values, groups):
        tr = work.iloc[tr_idx]
        te = work.iloc[te_idx]
        use_feats = feats
        if treatment:
            train_qids = set(df.iloc[tr_idx].query_id.unique())
            gf = build_graph_features(train_qids, edges)
            te_arr = attach_features(df.iloc[te_idx].candidate_id.values, gf)
            tr_arr = attach_features(df.iloc[tr_idx].candidate_id.values, gf)
            tr = tr.copy()
            te = te.copy()
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

    # --- 3-way split (calibration.py protocol) ---
    base_test, _, _, ncore, ncalib, ntest = three_way(df, feats, edges, treatment=False)
    treat_test, treat_model, use_feats, _, _, _ = three_way(df, feats, edges, treatment=True)
    print(f"split: core={ncore} calib={ncalib} test={ntest}", flush=True)
    print(f"3way baseline  test F1={base_test['f1']:.4f} "
          f"(P={base_test['precision']:.4f} R={base_test['recall']:.4f} t={base_test['threshold']:.2f})",
          flush=True)
    print(f"3way treatment test F1={treat_test['f1']:.4f} "
          f"(P={treat_test['precision']:.4f} R={treat_test['recall']:.4f} t={treat_test['threshold']:.2f})",
          flush=True)
    delta = treat_test["f1"] - base_test["f1"]
    print(f"3way Delta F1 = {delta:+.4f}", flush=True)

    # --- GroupKFold-5 OOF ---
    base_oof = groupkfold_oof(df, feats, edges, treatment=False)
    treat_oof = groupkfold_oof(df, feats, edges, treatment=True)
    print(f"GKF5 baseline  OOF F1={base_oof['f1']:.4f}  treatment OOF F1={treat_oof['f1']:.4f}  "
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
        "leakage_control": (
            "Graph features computed out-of-fold: 3-way uses CORE queries only "
            "(test fully held out); GroupKFold5 builds the citation graph from "
            "the 4 training folds and scores the held-out fold, so a query's own "
            "gold edges never enter its candidates' indegree/pagerank."
        ),
        "split": {"core": ncore, "calib": ncalib, "test": ntest},
        "n_citation_edges": len(edges),
        "new_features": NEW_FEATS,
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
    path = write_result("graph_features", payload, script="experiments/graph_features.py")
    print(f"\nSaved {path}", flush=True)


if __name__ == "__main__":
    main()
