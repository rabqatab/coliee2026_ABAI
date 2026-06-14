"""W1 Experiment D: Chronological (temporal) split evaluation of the meta-learner.

Reviewer flagged that the workshop text describes an 8:2 chronological split but
reports random 5-fold GroupKFold CV. This script reports BOTH and quantifies the
temporal-generalization gap.

Pipeline:
  1. Load cached stage outputs (no neural recomputation).
  2. Assemble the full feature matrix via build_feature_matrix (subsample=False,
     RRF-only candidate pool, NO gold injection -- honest evaluation).
  3. Save it to output/feature_matrix.parquet (artifact the ablation script expects).
  4. Evaluate meta-learner micro-F1 under:
       (a) random 5-fold GroupKFold CV  -> sanity check, should reproduce ~0.311
       (b) chronological 80/20 split    -> oldest 80% train, newest 20% test
  5. Print F1/P/R/threshold for both, save output/w1/d_temporal_split.json.

Run:
    uv run python scripts/w1_d_temporal_split.py
"""
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# --- Make the cached stage1 DocumentContexts unpicklable under its old module path ---
import coliee_task1.stages.citation_context as _cc
_graphrag_mod = type(sys)("graphrag")
sys.modules.setdefault("graphrag", _graphrag_mod)
sys.modules.setdefault("graphrag.citation_context", _cc)

from coliee_task1.config import (
    LGBM_PARAMS,
    N_FOLDS,
    OUTPUT_DIR,
    RANDOM_SEED,
    TRAIN_LABELS,
)
from coliee_task1.pipeline import load_labels, _load_cache
from coliee_task1.stages.meta_learner import (
    build_feature_matrix,
    compute_lexical_features,
    FEATURE_COLS,
)
from coliee_task1.utils.metrics import optimize_threshold

# Production "honest CV" recipe (reproduces ablation honest_f1=0.3111):
#   - train on a SUBSAMPLED + GOLD-INJECTED + STRATIFIED matrix
#   - multi-seed-3 ensemble (3 seeds x 5 folds = 15 models)
#   - evaluate on the FULL RRF-only pool, threshold via optimize_threshold (linspace-200)
MULTI_SEED = 3
USE_STRATIFIED = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("w1_d")


def _qnum(qid):
    """Numeric case id from '008447.txt' (chronological proxy: ascending = oldest)."""
    return int(qid.split(".")[0])


def _seed_params(seed):
    """LGBM params for one seed run (drops n_estimators/early_stopping_rounds)."""
    p = {k: v for k, v in LGBM_PARAMS.items()
         if k not in ("n_estimators", "early_stopping_rounds")}
    p["seed"] = RANDOM_SEED + seed
    p["bagging_seed"] = RANDOM_SEED + seed + 100
    p["feature_fraction_seed"] = RANDOM_SEED + seed + 200
    return p


def _train_one(X_train, y_train, X_val, y_val, params):
    """Train a single LightGBM booster with early stopping on the val set."""
    n_estimators = LGBM_PARAMS.get("n_estimators", 800)
    early_stopping = LGBM_PARAMS.get("early_stopping_rounds", 80)
    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    return lgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval], valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping),
            lgb.log_evaluation(period=0),
        ],
    )


def _full_pool_scores(models, df_full, feature_cols):
    """Average ensemble predictions over the full (eval) pool."""
    X = df_full[feature_cols].values
    scores = np.zeros(len(df_full))
    for m in models:
        scores += m.predict(X)
    scores /= len(models)
    return scores


def _query_scores_and_labels(df_full, scores):
    """{qid: [(cid, score)]} for optimize_threshold; gold labels from df rows."""
    query_scores = {}
    labels_dict = {}
    for qid, cid, s, lab in zip(
        df_full["query_id"].values, df_full["candidate_id"].values,
        scores, df_full["label"].values,
    ):
        query_scores.setdefault(qid, []).append((cid, s))
        if lab == 1:
            labels_dict.setdefault(qid, []).append(cid)
    for qid in query_scores:
        labels_dict.setdefault(qid, [])
    return query_scores, labels_dict


def _train_seed_ensemble(df_tr_rows, feature_cols, es_frac=0.15):
    """Train a MULTI_SEED ensemble on a set of TRAIN rows.

    Per seed, a single query-grouped early-stopping slice (es_frac of the train
    queries) provides the validation set; it never touches the outer eval rows.
    Returns the list of MULTI_SEED boosters. (One slice per seed keeps this fast:
    MULTI_SEED models per call rather than MULTI_SEED x n_inner_folds.)
    """
    X = df_tr_rows[feature_cols].values
    y = df_tr_rows["label"].values
    uniq_q = df_tr_rows["query_id"].unique()
    models = []
    for seed in range(MULTI_SEED):
        rng = np.random.default_rng(RANDOM_SEED + seed)
        perm = rng.permutation(uniq_q)
        n_val_q = max(1, int(len(perm) * es_frac))
        val_q = set(perm[:n_val_q])
        va_mask = df_tr_rows["query_id"].isin(val_q).values
        tr_mask = ~va_mask
        params = _seed_params(seed)
        models.append(_train_one(X[tr_mask], y[tr_mask], X[va_mask], y[va_mask], params))
    return models


def eval_groupkfold(df_train, df_full, feature_cols, n_folds=N_FOLDS):
    """Honest random GroupKFold CV with proper OUT-OF-FOLD discipline.

    Reproduces the production honest-CV recipe (ablation honest_f1=0.3111):
      - outer GroupKFold over queries
      - for each outer fold: train a MULTI_SEED ensemble on the gold-injected TRAIN
        rows of the OTHER queries, then score the held-out fold's queries from the
        full RRF-only EVAL pool (no in-sample contamination)
      - accumulate OOF eval scores, then sweep the threshold via optimize_threshold.
    """
    eval_queries = np.array(sorted(df_full["query_id"].unique(), key=_qnum))
    # Outer query-level folds.
    gkf = GroupKFold(n_splits=n_folds)
    dummy = np.zeros(len(eval_queries))
    oof_scores = {}  # row index in df_full -> score
    df_full = df_full.reset_index(drop=True)
    full_idx_by_q = df_full.groupby("query_id").indices

    for fold, (tr_q_idx, va_q_idx) in enumerate(gkf.split(dummy, dummy, eval_queries)):
        train_q = set(eval_queries[tr_q_idx])
        val_q = set(eval_queries[va_q_idx])
        df_tr_rows = df_train[df_train["query_id"].isin(train_q)].reset_index(drop=True)
        logger.info("  OOF fold %d/%d: train on %d queries (%d rows), eval %d queries",
                    fold + 1, n_folds, len(train_q), len(df_tr_rows), len(val_q))
        models = _train_seed_ensemble(df_tr_rows, feature_cols)

        # Score held-out eval queries from the full RRF-only pool.
        rows = np.concatenate([full_idx_by_q[q] for q in val_q if q in full_idx_by_q])
        Xev = df_full.loc[rows, feature_cols].values
        s = np.zeros(len(rows))
        for m in models:
            s += m.predict(Xev)
        s /= len(models)
        for r, sc in zip(rows, s):
            oof_scores[r] = sc

    scores = np.array([oof_scores.get(i, 0.0) for i in range(len(df_full))])
    qs, ld = _query_scores_and_labels(df_full, scores)
    t, metrics = optimize_threshold(qs, ld)
    metrics["threshold"] = t
    metrics["n_models_per_fold"] = MULTI_SEED * N_FOLDS
    return metrics


def eval_temporal(df_train_full, df_full, feature_cols, val_fraction=0.2):
    """Chronological 80/20 split with the SAME honest recipe.

    Case ids look like '008447.txt'; sorting by numeric id ascending = oldest->newest.
    The newest `val_fraction` of queries are held out. The MULTI_SEED ensemble is
    trained ONLY on the oldest 80% of queries (gold-injected TRAIN rows; inner folds
    for early stopping), then evaluated on the newest 20% of queries from the full
    RRF-only pool. Train and val queries are disjoint -> already OOF-clean.
    """
    all_q = sorted(df_full["query_id"].unique(), key=_qnum)
    n_val = max(1, int(len(all_q) * val_fraction))
    val_q = set(all_q[-n_val:])
    train_q = set(all_q[:-n_val])

    lo_tr, hi_tr = _qnum(min(train_q, key=_qnum)), _qnum(max(train_q, key=_qnum))
    lo_va, hi_va = _qnum(min(val_q, key=_qnum)), _qnum(max(val_q, key=_qnum))
    logger.info(
        "  Temporal split: %d train queries (id %06d-%06d), %d val queries (id %06d-%06d, newest)",
        len(train_q), lo_tr, hi_tr, len(val_q), lo_va, hi_va,
    )

    df_tr = df_train_full[df_train_full["query_id"].isin(train_q)].reset_index(drop=True)
    df_va = df_full[df_full["query_id"].isin(val_q)].reset_index(drop=True)

    logger.info("  Temporal: training %d-seed ensemble on %d train rows", MULTI_SEED, len(df_tr))
    models = _train_seed_ensemble(df_tr, feature_cols)

    scores = _full_pool_scores(models, df_va, feature_cols)
    qs, ld = _query_scores_and_labels(df_va, scores)
    t, metrics = optimize_threshold(qs, ld)
    metrics["threshold"] = t
    metrics["n_models"] = len(models)
    metrics["n_train_queries"] = len(train_q)
    metrics["n_val_queries"] = len(val_q)
    return metrics


def main():
    logger.info("=" * 70)
    logger.info("  W1-D: Chronological vs GroupKFold split evaluation")
    logger.info("=" * 70)

    # --- Load labels and cached stage outputs ---
    labels = load_labels(TRAIN_LABELS)
    query_ids = sorted(labels.keys())
    logger.info("Loaded %d training queries", len(query_ids))

    raw_corpus, clean_corpus, contexts = _load_cache("stage1")
    rrf_results, bm25_raw, bm25_rrf, ctx_feats = _load_cache("stage2")
    stage3 = _load_cache("stage3")
    biencoder_scores = stage3[0] if isinstance(stage3, tuple) else stage3
    multi_scores = _load_cache("stage3_m3")
    crossencoder_scores = _load_cache("stage4")
    graphrag_features = _load_cache("stage5")
    gnn_scores = _load_cache("stage5_5_oof")  # OOF GNN scores for honest CV

    # --- EVAL pool (HONEST): RRF only, NO gold injection ---
    eval_pool = {
        qid: [cid for cid, _ in rrf_results.get(qid, [])]
        for qid in query_ids
    }
    n_missing = sum(
        len(set(labels.get(qid, [])) - set(eval_pool.get(qid, [])))
        for qid in query_ids
    )
    logger.info("Honest eval pool (RRF-only): %d gold positives unreachable (outside pool)", n_missing)

    # --- TRAIN pool: RRF + gold injection (training-only trick) ---
    train_pool = {qid: list(eval_pool.get(qid, [])) for qid in query_ids}
    n_injected = 0
    for qid in query_ids:
        gold = set(labels.get(qid, []))
        pool_set = set(train_pool[qid])
        for g in gold:
            if g not in pool_set:
                train_pool[qid].append(g)
                n_injected += 1
    logger.info("Train pool: injected %d gold positives not in RRF pool", n_injected)

    # --- Lexical features over the union (train pool superset; required for ~0.311) ---
    lexical_features = compute_lexical_features(clean_corpus, train_pool)

    feat_args = dict(
        bm25_scores=bm25_raw,
        bm25_rrf_scores=bm25_rrf,
        biencoder_scores=biencoder_scores,
        crossencoder_scores=crossencoder_scores,
        graphrag_features=graphrag_features,
        context_features=ctx_feats,
        lexical_features=lexical_features,
        multi_scores=multi_scores,
        gnn_scores=gnn_scores,
        reasoning_scores=None,
        raw_corpus=raw_corpus,
    )

    # --- TRAIN matrix: subsampled + gold-injected + stratified negatives ---
    logger.info("Building TRAIN feature matrix (subsample=True, stratified=%s, gold-injected) ...",
                USE_STRATIFIED)
    df_train = build_feature_matrix(
        labels, train_pool, **feat_args, subsample=True, stratified=USE_STRATIFIED,
    )

    # --- EVAL matrix: full RRF-only pool, no sampling (honest) ---
    logger.info("Building EVAL feature matrix (subsample=False, RRF-only) ...")
    df_full = build_feature_matrix(labels, eval_pool, **feat_args, subsample=False)
    logger.info("Eval matrix: %d rows, %d pos / %d neg",
                len(df_full), int(df_full["label"].sum()), int(len(df_full) - df_full["label"].sum()))

    # --- Save artifact for the ablation script (the full RRF-only eval matrix) ---
    out_parquet = OUTPUT_DIR / "feature_matrix.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(out_parquet)
    logger.info("Saved feature matrix -> %s", out_parquet)

    feature_cols = FEATURE_COLS

    # --- (a) Random GroupKFold CV (sanity check ~0.311) ---
    logger.info("-" * 70)
    logger.info("  (a) Random 5-fold GroupKFold CV (honest production recipe)")
    logger.info("-" * 70)
    cv = eval_groupkfold(df_train, df_full, feature_cols)
    logger.info("  GroupKFold CV: F1=%.4f P=%.4f R=%.4f (t=%.3f)",
                cv["f1"], cv["precision"], cv["recall"], cv["threshold"])

    # --- (b) Chronological 80/20 split ---
    logger.info("-" * 70)
    logger.info("  (b) Chronological 80/20 split (newest 20%% held out)")
    logger.info("-" * 70)
    temporal = eval_temporal(df_train, df_full, feature_cols, val_fraction=0.2)
    logger.info("  Temporal split: F1=%.4f P=%.4f R=%.4f (t=%.3f)",
                temporal["f1"], temporal["precision"], temporal["recall"], temporal["threshold"])

    gap = cv["f1"] - temporal["f1"]

    # --- Sanity check ---
    sanity_ok = abs(cv["f1"] - 0.311) <= 0.02
    logger.info("=" * 70)
    logger.info("  SANITY CHECK: GroupKFold CV F1=%.4f vs expected 0.311 -> %s",
                cv["f1"], "PASS" if sanity_ok else "OFF (check assembly!)")
    logger.info("  Temporal-generalization gap (CV - temporal) = %.4f", gap)
    logger.info("=" * 70)

    results = {
        "groupkfold_cv": cv,
        "temporal_80_20": temporal,
        "gap_cv_minus_temporal": gap,
        "sanity_check_cv_f1_near_0311": sanity_ok,
        "recipe": {
            "multi_seed": MULTI_SEED,
            "stratified_negatives": USE_STRATIFIED,
            "gold_injected_train_pool": True,
            "eval_pool": "rrf_only_no_gold",
            "n_folds": N_FOLDS,
        },
        "n_eval_rows": int(len(df_full)),
        "n_eval_pos": int(df_full["label"].sum()),
        "n_eval_neg": int(len(df_full) - df_full["label"].sum()),
        "n_train_rows": int(len(df_train)),
        "n_queries": int(df_full["query_id"].nunique()),
        "n_features": len(feature_cols),
        "honest_pool_unreachable_golds": int(n_missing),
    }

    out_json = OUTPUT_DIR / "w1" / "d_temporal_split.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Results saved -> %s", out_json)

    return results


if __name__ == "__main__":
    main()
