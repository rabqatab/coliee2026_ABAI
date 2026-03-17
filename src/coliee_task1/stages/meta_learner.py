"""LightGBM meta-learner: fuses all pipeline stage scores into final predictions.

Assembles features from BM25, bi-encoder, cross-encoder, and GraphRAG Lite,
trains a LightGBM binary classifier with GroupKFold cross-validation,
and optimizes the prediction threshold for micro-averaged F1.
"""
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from coliee_task1.config import (
    LGBM_PARAMS,
    N_FOLDS,
    RANDOM_SEED,
    MODELS_DIR,
)
from coliee_task1.utils.metrics import micro_f1, scores_to_predictions, optimize_threshold

logger = logging.getLogger(__name__)

# Feature columns produced by assemble_features()
BASE_FEATURE_COLS = [
    # BM25 features (2)
    "bm25_score",
    "bm25_rrf_score",
    # Bi-encoder features (2)
    "biencoder_score",
    "biencoder_rank",
    # Cross-encoder features (2)
    "crossencoder_score",
    "crossencoder_rank",
    # Lexical features (5) — ported from baseline (strongest signal)
    "tfidf_cosine",
    "jaccard",
    "shared_bigrams",
    "length_ratio",
    "shared_legal_terms",
    # GraphRAG Lite features (9)
    "same_community_0.5",
    "same_community_1.0",
    "same_community_2.0",
    "community_jaccard",
    "shared_statutes",
    "shared_judges",
    "same_domain",
    "same_outcome",
    "entity_overlap_score",
    # Citation context features (2)
    "n_context_matches",
    "max_context_bm25",
    # BGE-M3 multi-signal features (4) -- when USE_BGE_M3=True
    "m3_dense_score",
    "m3_sparse_score",
    "m3_colbert_score",
    "m3_fused_score",
    # GNN reranker features (2) -- when USE_GNN_RERANKER=True
    "gnn_score",
    "gnn_rank",
    # Reasoning reranker features (2) -- when USE_REASONING_RERANKER=True
    "reasoning_score",
    "reasoning_rank",
]

# Score distribution features (Option 14) — computed per-query in build_feature_matrix
SCORE_DIST_COLS = [
    "bm25_rrf_rank_norm",
    "bm25_rrf_score_gap",
    "biencoder_score_gap",
    "crossencoder_score_gap",
    "top_score_ratio",
    "score_above_median",
]

# PPR features (Option 1) — from graphrag_lite when USE_PPR_FEATURES=True
PPR_COLS = [
    "ppr_score",
    "ppr_rank",
]

# Full feature set (always includes all columns; unused ones default to 0)
FEATURE_COLS = BASE_FEATURE_COLS + SCORE_DIST_COLS + PPR_COLS


def assemble_features(
    query_id: str,
    candidate_id: str,
    bm25_scores: dict[str, dict[str, float]],
    bm25_rrf_scores: dict[str, dict[str, float]],
    biencoder_scores: dict[str, dict[str, float]],
    crossencoder_scores: dict[str, dict[str, float]],
    graphrag_features: dict[tuple[str, str], dict[str, float]],
    context_features: dict[str, dict[str, dict[str, float]]],
    lexical_features: dict[tuple[str, str], dict[str, float]] | None = None,
    multi_scores: dict[str, dict[str, dict[str, float]]] | None = None,
    gnn_scores: dict[str, dict[str, float]] | None = None,
    reasoning_scores: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Assemble all features for a single (query, candidate) pair.

    Args:
        query_id: Query document ID
        candidate_id: Candidate document ID
        bm25_scores: {query_id: {candidate_id: raw_bm25_score}}
        bm25_rrf_scores: {query_id: {candidate_id: rrf_score}}
        biencoder_scores: {query_id: {candidate_id: cosine_similarity}}
        crossencoder_scores: {query_id: {candidate_id: probability}}
        graphrag_features: {(query_id, candidate_id): {feature_name: value}}
        context_features: {query_id: {candidate_id: {n_matches, max_score}}}
        lexical_features: {(query_id, candidate_id): {tfidf_cosine, jaccard, ...}}

    Returns:
        Dict of feature_name -> value
    """
    feats = {}

    # BM25
    q_bm25 = bm25_scores.get(query_id, {})
    feats["bm25_score"] = q_bm25.get(candidate_id, 0.0)
    q_rrf = bm25_rrf_scores.get(query_id, {})
    feats["bm25_rrf_score"] = q_rrf.get(candidate_id, 0.0)

    # Bi-encoder
    q_bi = biencoder_scores.get(query_id, {})
    feats["biencoder_score"] = q_bi.get(candidate_id, 0.0)
    # Rank (lower is better)
    if q_bi:
        sorted_candidates = sorted(q_bi.items(), key=lambda x: -x[1])
        rank_map = {cid: i + 1 for i, (cid, _) in enumerate(sorted_candidates)}
        feats["biencoder_rank"] = float(rank_map.get(candidate_id, len(q_bi) + 1))
    else:
        feats["biencoder_rank"] = 999.0

    # Cross-encoder
    q_ce = crossencoder_scores.get(query_id, {})
    feats["crossencoder_score"] = q_ce.get(candidate_id, 0.0)
    if q_ce:
        sorted_candidates = sorted(q_ce.items(), key=lambda x: -x[1])
        rank_map = {cid: i + 1 for i, (cid, _) in enumerate(sorted_candidates)}
        feats["crossencoder_rank"] = float(rank_map.get(candidate_id, len(q_ce) + 1))
    else:
        feats["crossencoder_rank"] = 999.0

    # Lexical features (from baseline)
    lex = (lexical_features or {}).get((query_id, candidate_id), {})
    feats["tfidf_cosine"] = lex.get("tfidf_cosine", 0.0)
    feats["jaccard"] = lex.get("jaccard", 0.0)
    feats["shared_bigrams"] = lex.get("shared_bigrams", 0.0)
    feats["length_ratio"] = lex.get("length_ratio", 0.0)
    feats["shared_legal_terms"] = lex.get("shared_legal_terms", 0.0)

    # GraphRAG Lite
    grag = graphrag_features.get((query_id, candidate_id), {})
    for col in FEATURE_COLS:
        if col.startswith(("same_community", "community_jaccard",
                          "same_domain", "same_outcome", "entity_overlap")):
            feats[col] = grag.get(col, 0.0)
    # shared_statutes and shared_judges come from GraphRAG, not lexical
    feats["shared_statutes"] = grag.get("shared_statutes", 0.0)
    feats["shared_judges"] = grag.get("shared_judges", 0.0)

    # Citation context
    q_ctx = context_features.get(query_id, {})
    c_ctx = q_ctx.get(candidate_id, {})
    feats["n_context_matches"] = c_ctx.get("n_matches", 0.0)
    feats["max_context_bm25"] = c_ctx.get("max_score", 0.0)

    # BGE-M3 multi-signal features
    if multi_scores is not None:
        from coliee_task1.stages.multi_retrieval import extract_multi_features
        m3_feats = extract_multi_features(query_id, candidate_id, multi_scores)
        feats.update(m3_feats)

    # GNN reranker features
    if gnn_scores is not None:
        gnn_q = gnn_scores.get(query_id, {})
        feats["gnn_score"] = gnn_q.get(candidate_id, 0.0)
        # Compute rank within query
        if gnn_q:
            sorted_cands = sorted(gnn_q.items(), key=lambda x: -x[1])
            rank_map = {cid: i + 1 for i, (cid, _) in enumerate(sorted_cands)}
            feats["gnn_rank"] = float(rank_map.get(candidate_id, len(gnn_q) + 1))
        else:
            feats["gnn_rank"] = 999.0

    # Reasoning reranker features
    if reasoning_scores is not None:
        reas_q = reasoning_scores.get(query_id, {})
        feats["reasoning_score"] = reas_q.get(candidate_id, 0.0)
        if reas_q:
            sorted_cands = sorted(reas_q.items(), key=lambda x: -x[1])
            rank_map = {cid: i + 1 for i, (cid, _) in enumerate(sorted_cands)}
            feats["reasoning_rank"] = float(rank_map.get(candidate_id, len(reas_q) + 1))
        else:
            feats["reasoning_rank"] = 999.0

    return feats


def add_score_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add query-relative score distribution features (Option 14).

    These features capture where a candidate's scores fall relative to
    other candidates for the same query — critical for ranking.
    """
    # Score gaps (candidate score minus query mean)
    for base_col, gap_col in [
        ("bm25_rrf_score", "bm25_rrf_score_gap"),
        ("biencoder_score", "biencoder_score_gap"),
        ("crossencoder_score", "crossencoder_score_gap"),
    ]:
        if base_col in df.columns:
            group_mean = df.groupby("query_id")[base_col].transform("mean")
            df[gap_col] = df[base_col] - group_mean
        else:
            df[gap_col] = 0.0

    # Normalized rank within query (0 = best, 1 = worst)
    rank_col = df.groupby("query_id")["bm25_rrf_score"].rank(
        ascending=False, method="min",
    )
    group_size = df.groupby("query_id")["bm25_rrf_score"].transform("count")
    df["bm25_rrf_rank_norm"] = rank_col / group_size.clip(lower=1)

    # Top score ratio (candidate / query max)
    group_max = df.groupby("query_id")["bm25_rrf_score"].transform("max")
    df["top_score_ratio"] = df["bm25_rrf_score"] / group_max.clip(lower=1e-8)

    # Above median flag
    group_median = df.groupby("query_id")["bm25_rrf_score"].transform("median")
    df["score_above_median"] = (df["bm25_rrf_score"] > group_median).astype(float)

    return df


def build_feature_matrix(
    labels: dict[str, list[str]],
    candidate_pool: dict[str, list[str]],
    bm25_scores: dict[str, dict[str, float]],
    bm25_rrf_scores: dict[str, dict[str, float]],
    biencoder_scores: dict[str, dict[str, float]],
    crossencoder_scores: dict[str, dict[str, float]],
    graphrag_features: dict[tuple[str, str], dict[str, float]],
    context_features: dict[str, dict[str, dict[str, float]]],
    lexical_features: dict[tuple[str, str], dict[str, float]] | None = None,
    multi_scores: dict[str, dict[str, dict[str, float]]] | None = None,
    gnn_scores: dict[str, dict[str, float]] | None = None,
    reasoning_scores: dict[str, dict[str, float]] | None = None,
    subsample: bool = True,
    max_neg_ratio: int = 10,
    max_neg_per_query: int = 50,
    stratified: bool = False,
) -> pd.DataFrame:
    """Build the feature matrix for training or evaluation.

    Args:
        subsample: If True, subsample negatives per query for balanced training.
                   If False, include all candidates (for evaluation/prediction).
        max_neg_ratio: Max negatives per positive when subsampling.
        max_neg_per_query: Absolute cap on negatives per query.
        stratified: If True, stratify negatives by RRF rank (Option 19):
                    50% hard (top RRF rank), 30% medium, 20% random.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for query_id, candidates in candidate_pool.items():
        positive_set = set(labels.get(query_id, []))

        if subsample:
            positives = [c for c in candidates if c in positive_set]
            negatives = [c for c in candidates if c not in positive_set]
            n_neg = min(len(positives) * max_neg_ratio, max_neg_per_query, len(negatives))

            if stratified and negatives and n_neg > 0:
                # Sort negatives by RRF score (descending) for difficulty stratification
                q_rrf = bm25_rrf_scores.get(query_id, {})
                neg_with_score = [(c, q_rrf.get(c, 0.0)) for c in negatives]
                neg_with_score.sort(key=lambda x: -x[1])

                n_hard = max(1, int(n_neg * 0.5))
                n_medium = max(1, int(n_neg * 0.3))
                n_easy = max(1, n_neg - n_hard - n_medium)

                # Hard: top-ranked negatives (high RRF score, model-confusing)
                hard_pool = [c for c, _ in neg_with_score[:len(neg_with_score) // 3]]
                # Medium: mid-ranked
                medium_pool = [c for c, _ in neg_with_score[len(neg_with_score) // 3: 2 * len(neg_with_score) // 3]]
                # Easy: low-ranked
                easy_pool = [c for c, _ in neg_with_score[2 * len(neg_with_score) // 3:]]

                def _sample(pool: list, n: int) -> list:
                    if not pool:
                        return []
                    n = min(n, len(pool))
                    return list(rng.choice(pool, size=n, replace=False))

                sampled_negs = _sample(hard_pool, n_hard) + _sample(medium_pool, n_medium) + _sample(easy_pool, n_easy)
            elif negatives and n_neg > 0 and n_neg < len(negatives):
                sampled_negs = list(rng.choice(negatives, size=n_neg, replace=False))
            else:
                sampled_negs = negatives
            selected = positives + sampled_negs
        else:
            selected = candidates

        for cand_id in selected:
            feats = assemble_features(
                query_id, cand_id,
                bm25_scores, bm25_rrf_scores,
                biencoder_scores, crossencoder_scores,
                graphrag_features, context_features,
                lexical_features,
                multi_scores=multi_scores,
                gnn_scores=gnn_scores,
                reasoning_scores=reasoning_scores,
            )
            feats["query_id"] = query_id
            feats["candidate_id"] = cand_id
            feats["label"] = 1 if cand_id in positive_set else 0
            rows.append(feats)

    df = pd.DataFrame(rows)

    # Fill missing PPR/score-dist columns with defaults
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Add score distribution features (Option 14)
    df = add_score_distribution_features(df)

    n_pos = df["label"].sum()
    n_neg = len(df) - n_pos
    logger.info(
        "Feature matrix: %d rows, %d features, %d pos / %d neg (ratio 1:%.0f)%s",
        len(df), len(FEATURE_COLS), n_pos, n_neg, n_neg / max(n_pos, 1),
        " [sampled]" if subsample else " [full]",
    )
    return df


def train_meta_learner(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    lgbm_params: dict | None = None,
    output_dir: Path | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[list[lgb.Booster], float, dict[str, float]]:
    """Train LightGBM with GroupKFold cross-validation.

    Groups by query_id so all candidates for a query stay in the same fold.
    Supports both binary classification and LambdaRank (ranking) objectives.

    Returns:
        (list_of_fold_models, best_threshold, cv_metrics)
    """
    if lgbm_params is None:
        lgbm_params = LGBM_PARAMS.copy()
    if output_dir is None:
        output_dir = MODELS_DIR / "meta_learner"
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    output_dir.mkdir(parents=True, exist_ok=True)

    is_ranking = lgbm_params.get("objective", "binary") in ("lambdarank", "rank_xendcg")

    X = df[feature_cols].values
    y = df["label"].values
    groups = df["query_id"].values
    query_ids_arr = df["query_id"].values
    candidate_ids_arr = df["candidate_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    models = []
    oof_scores = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info("=== Fold %d/%d ===", fold + 1, n_folds)

        # Extract n_estimators and early_stopping_rounds from params
        params = {k: v for k, v in lgbm_params.items()
                  if k not in ("n_estimators", "early_stopping_rounds")}
        n_estimators = lgbm_params.get("n_estimators", 500)
        early_stopping = lgbm_params.get("early_stopping_rounds", 50)

        if is_ranking:
            # LambdaRank requires data sorted by query, with group sizes
            train_df = df.iloc[train_idx].copy()
            train_df["_orig_idx"] = train_idx
            train_df = train_df.sort_values("query_id")
            train_groups = train_df.groupby("query_id").size().tolist()

            val_df = df.iloc[val_idx].copy()
            val_df["_orig_idx"] = val_idx
            val_df = val_df.sort_values("query_id")
            val_groups = val_df.groupby("query_id").size().tolist()

            X_train = train_df[feature_cols].values
            y_train = train_df["label"].values
            X_val = val_df[feature_cols].values
            y_val = val_df["label"].values

            dtrain = lgb.Dataset(X_train, y_train, group=train_groups)
            dval = lgb.Dataset(X_val, y_val, group=val_groups, reference=dtrain)
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            dtrain = lgb.Dataset(X_train, y_train)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain)

        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=early_stopping),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        # OOF predictions — map back to original indices
        val_preds = model.predict(X_val)
        if is_ranking:
            for orig_i, score in zip(val_df["_orig_idx"].values, val_preds):
                oof_scores[orig_i] = score
        else:
            oof_scores[val_idx] = val_preds

        models.append(model)

        # Save fold model
        model.save_model(str(output_dir / f"fold_{fold}.txt"))

    # Convert OOF scores to query-level format for threshold optimization
    oof_query_scores: dict[str, list[tuple[str, float]]] = {}
    labels_dict: dict[str, list[str]] = {}

    for i, (qid, cid, score, label) in enumerate(
        zip(query_ids_arr, candidate_ids_arr, oof_scores, y)
    ):
        if qid not in oof_query_scores:
            oof_query_scores[qid] = []
        oof_query_scores[qid].append((cid, score))

        if label == 1:
            if qid not in labels_dict:
                labels_dict[qid] = []
            labels_dict[qid].append(cid)

    # Ensure all queries with labels are present
    for qid in oof_query_scores:
        if qid not in labels_dict:
            labels_dict[qid] = []

    # Optimize threshold on OOF predictions
    best_threshold, cv_metrics = optimize_threshold(oof_query_scores, labels_dict)

    logger.info(
        "CV Results: F1=%.4f, P=%.4f, R=%.4f (threshold=%.3f)",
        cv_metrics.get("f1", 0),
        cv_metrics.get("precision", 0),
        cv_metrics.get("recall", 0),
        best_threshold,
    )

    # Save threshold and config
    config = {
        "threshold": best_threshold,
        "cv_f1": cv_metrics.get("f1", 0),
        "objective": lgbm_params.get("objective", "binary"),
        "feature_cols": feature_cols,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Feature importance
    importance = np.zeros(len(feature_cols))
    for m in models:
        importance += m.feature_importance(importance_type="gain")
    importance /= len(models)

    fi = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    logger.info("Feature importance (gain):")
    for name, imp in fi:
        logger.info("  %s: %.1f", name, imp)

    return models, best_threshold, cv_metrics


def predict(
    models: list[lgb.Booster],
    df: pd.DataFrame,
    threshold: float,
    feature_cols: list[str] | None = None,
    min_per_query: int = 0,
) -> dict[str, list[str]]:
    """Generate predictions using ensemble of fold models.

    Averages predictions across all fold models and applies threshold.

    Args:
        min_per_query: minimum predictions per query (Option 18: top-1 guarantee)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    X = df[feature_cols].values
    scores = np.zeros(len(df))
    for model in models:
        scores += model.predict(X)
    scores /= len(models)

    # Convert to query-level predictions
    query_scores: dict[str, list[tuple[str, float]]] = {}
    for i, row in df.iterrows():
        qid = row["query_id"]
        cid = row["candidate_id"]
        if qid not in query_scores:
            query_scores[qid] = []
        query_scores[qid].append((cid, scores[i]))

    return scores_to_predictions(query_scores, threshold, min_per_query=min_per_query)
