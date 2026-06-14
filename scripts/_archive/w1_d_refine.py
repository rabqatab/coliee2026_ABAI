"""W1 D-refine: clean temporal-vs-CV with the paper's 34 features + threshold-TRANSFER.

Loads the regenerated output/feature_matrix.parquet (RRF-only honest eval pool),
restricts to the 34 features reported in the paper, and reports:
  1. random GroupKFold CV F1 (OOF, threshold optimized on OOF)
  2. temporal 80/20: threshold FIT ON TRAIN (via train-internal OOF) and APPLIED to the
     newest-20% hold-out  -> the realistic threshold-transfer F1 (calibration test)
  3. temporal oracle: threshold optimized directly on the hold-out
The calibration penalty = oracle - transfer isolates threshold-miscalibration
(diagnosis #3) from temporal generalization (diagnosis #2).
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent
FM = REPO / "output" / "feature_matrix.parquet"
OUT = REPO / "output" / "w1" / "d_refine.json"

# The 34 features reported in the camera-ready (Table 1): 9 + 8 + 11 + 6.
FEATURES_34 = [
    # Stage 1 (9)
    "bm25_score", "bm25_rrf_score", "n_context_matches", "max_context_bm25",
    "tfidf_cosine", "jaccard", "shared_bigrams", "length_ratio", "shared_legal_terms",
    # Stage 2 (8)
    "biencoder_score", "biencoder_rank", "m3_dense_score", "m3_sparse_score",
    "m3_colbert_score", "m3_fused_score", "crossencoder_score", "crossencoder_rank",
    # Stage 3 (11)
    "same_community_0.5", "same_community_1.0", "same_community_2.0", "community_jaccard",
    "shared_statutes", "shared_judges", "same_domain", "same_outcome",
    "entity_overlap_score", "gnn_score", "gnn_rank",
    # Stage 4 / score-distribution (6)
    "bm25_rrf_rank_norm", "bm25_rrf_score_gap", "biencoder_score_gap",
    "crossencoder_score_gap", "top_score_ratio", "score_above_median",
]

LGBM = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "is_unbalance": True,
}
N_EST = 500
ES = 50


def _qs_lab(df, scores):
    qs, lab = {}, {}
    for qid, cid, sc, y in zip(df.query_id.values, df.candidate_id.values, scores, df.label.values):
        qs.setdefault(qid, []).append((cid, float(sc)))
        if y == 1:
            lab.setdefault(qid, []).append(cid)
    for qid in qs:
        lab.setdefault(qid, [])
    return qs, lab


def _f1_at(qs, lab, t):
    tp = fp = fn = 0
    for qid in lab:
        true = set(lab[qid])
        pred = {c for c, s in qs.get(qid, []) if s >= t}
        tp += len(true & pred); fp += len(pred - true); fn += len(true - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"f1": f1, "precision": p, "recall": r, "threshold": float(t)}


def _sweep(qs, lab):
    best = {"f1": -1.0}
    for t in np.arange(0.01, 1.0, 0.01):
        m = _f1_at(qs, lab, float(t))
        if m["f1"] > best["f1"]:
            best = m
    return best


def _oof(df, feats):
    X = df[feats].values; y = df.label.values; g = df.query_id.values
    oof = np.zeros(len(df))
    for tr, va in GroupKFold(5).split(X, y, g):
        dtr = lgb.Dataset(X[tr], y[tr]); dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(LGBM, dtr, num_boost_round=N_EST, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(ES), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va])
    return oof


def main():
    df = pd.read_parquet(FM)
    feats = [f for f in FEATURES_34 if f in df.columns]
    missing = [f for f in FEATURES_34 if f not in df.columns]
    print(f"features present: {len(feats)}/34  missing={missing}", flush=True)
    assert len(feats) == 34, f"expected 34 features, got {len(feats)}"

    # 1) random GroupKFold CV
    oof = _oof(df, feats)
    cv = _sweep(*_qs_lab(df, oof))
    print(f"[CV] {cv}", flush=True)

    # 2) temporal 80/20 by query id (chronological proxy)
    qids = sorted(df.query_id.unique())
    n_tr = int(len(qids) * 0.8)
    train_q, test_q = set(qids[:n_tr]), set(qids[n_tr:])
    tr_df = df[df.query_id.isin(train_q)].reset_index(drop=True)
    te_df = df[df.query_id.isin(test_q)].reset_index(drop=True)

    # train-fit threshold via train-internal OOF (no peeking at test)
    tr_oof = _oof(tr_df, feats)
    train_thr = _sweep(*_qs_lab(tr_df, tr_oof))["threshold"]

    # model on full train (fixed rounds), predict test
    model = lgb.train(LGBM, lgb.Dataset(tr_df[feats].values, tr_df.label.values),
                      num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
    te_pred = model.predict(te_df[feats].values)
    te_qs, te_lab = _qs_lab(te_df, te_pred)
    transfer = _f1_at(te_qs, te_lab, train_thr)   # realistic: train threshold applied to test
    oracle = _sweep(te_qs, te_lab)                # oracle: threshold tuned on test

    out = {
        "n_features": len(feats),
        "random_cv": cv,
        "temporal_train_fit_threshold": train_thr,
        "temporal_transfer_train_thr": transfer,
        "temporal_oracle": oracle,
        "calibration_penalty_f1": oracle["f1"] - transfer["f1"],
        "n_train_queries": len(train_q),
        "n_test_queries": len(test_q),
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    sys.exit(main())
