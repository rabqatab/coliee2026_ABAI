"""
Task 1 Label Signal Validation

Validates that golden truth labels carry real signal by comparing noticed (cited)
cases against random negatives across multiple text-based features.

Outputs:
  - Statistical test results (stdout)
  - Classifier metrics (stdout)
  - Plots saved to plots/
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "task1"
CORPUS_DIR = DATA_DIR / "task1_train_files_2025"
LABELS_PATH = DATA_DIR / "task1_train_labels_2025.json"
PLOTS_DIR = BASE_DIR / "docs" / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NEG_RATIO = 10  # negatives per positive
MAX_NEG_PER_QUERY = 50
TFIDF_MAX_FEATURES = 50_000
RANDOM_SEED = 42

# Legal terms frequently found in Federal Court of Canada case law
LEGAL_TERMS = {
    "applicant", "respondent", "appellant", "plaintiff", "defendant",
    "court", "judge", "tribunal", "minister", "officer",
    "act", "regulation", "statute", "section", "subsection", "paragraph",
    "immigration", "refugee", "citizenship", "patent", "trademark", "copyright",
    "judicial", "review", "appeal", "motion", "order", "decision",
    "reasonable", "correctness", "procedural", "fairness", "jurisdiction",
    "evidence", "credibility", "burden", "proof", "standard",
    "charter", "constitution", "federal", "canada", "canadian",
    "damages", "injunction", "relief", "remedy", "costs",
    "affidavit", "testimony", "witness", "hearing", "trial",
}


# ---------------------------------------------------------------------------
# Step 1: Load + preprocess corpus
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = text.replace("<FRAGMENT_SUPPRESSED>", "")
    text = text.replace("[End of document]", "")
    # Rejoin broken statute names (lines ending with an open word continued on next line)
    text = re.sub(r"\n(?=[a-z])", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_simple(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{2,}", text.lower())


def load_corpus() -> tuple[list[str], list[str], dict[str, str]]:
    """Load all corpus files. Returns (filenames, texts, name->text mapping)."""
    print("Loading corpus...")
    t0 = time.time()
    filenames = sorted(p.name for p in CORPUS_DIR.glob("*.txt"))
    texts = []
    name_to_text = {}
    for fn in filenames:
        raw = (CORPUS_DIR / fn).read_text(errors="replace")
        cleaned = preprocess(raw)
        texts.append(cleaned)
        name_to_text[fn] = cleaned
    elapsed = time.time() - t0
    print(f"  Loaded {len(filenames)} documents in {elapsed:.1f}s")
    return filenames, texts, name_to_text


# ---------------------------------------------------------------------------
# Step 2: Build retrieval indexes
# ---------------------------------------------------------------------------
class SparseBM25:
    """BM25Okapi implemented via sparse matrices for fast per-pair scoring."""

    def __init__(self, tf_matrix: csr_matrix, k1: float = 1.5, b: float = 0.75):
        n_docs, n_terms = tf_matrix.shape
        # Document lengths and average
        doc_lens = np.array(tf_matrix.sum(axis=1)).ravel().astype(np.float64)
        avgdl = doc_lens.mean()
        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df = np.array((tf_matrix > 0).sum(axis=0)).ravel().astype(np.float64)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        # BM25 TF component per entry: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
        # We build this as a sparse matrix with same sparsity as tf_matrix
        tf_csc = tf_matrix.tocsc().astype(np.float64)
        # Denominator needs per-row (doc) length, applied per nonzero
        tf_csr = tf_matrix.tocsr().astype(np.float64)
        bm25_data = np.empty(tf_csr.nnz, dtype=np.float64)
        for i in range(n_docs):
            start, end = tf_csr.indptr[i], tf_csr.indptr[i + 1]
            tf_vals = tf_csr.data[start:end]
            denom = tf_vals + k1 * (1.0 - b + b * doc_lens[i] / avgdl)
            bm25_data[start:end] = tf_vals * (k1 + 1.0) / denom
        # bm25_tf_matrix[doc, term] = BM25 TF component
        self.bm25_tf = csr_matrix(
            (bm25_data, tf_csr.indices, tf_csr.indptr), shape=tf_csr.shape
        )
        self.idf = idf  # shape (n_terms,)

    def score_pairs(self, q_indices: np.ndarray, c_indices: np.ndarray,
                    q_tf_matrix: csr_matrix) -> np.ndarray:
        """Compute BM25 scores for pairs. q_tf_matrix is the query term-frequency matrix."""
        # For each pair: sum over terms in query of (query_tf_binary * idf[t] * bm25_tf[candidate, t])
        # We use query presence (binary) not raw TF for query-side (standard BM25Okapi)
        n_pairs = len(q_indices)
        scores = np.zeros(n_pairs, dtype=np.float64)
        q_binary = (q_tf_matrix > 0).astype(np.float64)  # binary query term presence

        # Batch: for each pair, score = sum(q_binary[q] * idf * bm25_tf[c])
        # Process in chunks for memory efficiency
        chunk_size = 5000
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            qi = q_indices[start:end]
            ci = c_indices[start:end]
            # q_terms[i, j] = 1 if query i has term j
            q_terms = q_binary[qi]  # sparse (chunk, n_terms)
            # c_bm25[i, j] = BM25 TF of candidate i for term j
            c_bm25 = self.bm25_tf[ci]  # sparse (chunk, n_terms)
            # Element-wise: q_terms * c_bm25 * idf, then sum per row
            # idf is broadcast as dense diag
            weighted = q_terms.multiply(c_bm25).multiply(self.idf[np.newaxis, :])
            scores[start:end] = np.array(weighted.sum(axis=1)).ravel()
        return scores.astype(np.float32)


def build_bm25_index(texts: list[str]) -> tuple[SparseBM25, csr_matrix]:
    print("Building BM25 index (sparse)...")
    t0 = time.time()
    cv = CountVectorizer(token_pattern=r"[a-zA-Z]{2,}", lowercase=True)
    tf_matrix = cv.fit_transform(texts)
    bm25 = SparseBM25(tf_matrix)
    print(f"  BM25 sparse index built in {time.time() - t0:.1f}s "
          f"(vocab={tf_matrix.shape[1]}, nnz={tf_matrix.nnz})")
    return bm25, tf_matrix


def build_tfidf(texts: list[str]) -> tuple[TfidfVectorizer, object]:
    print("Building TF-IDF matrix...")
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    mat = vec.fit_transform(texts)
    print(f"  TF-IDF matrix shape: {mat.shape}, built in {time.time() - t0:.1f}s")
    return vec, mat


# ---------------------------------------------------------------------------
# Step 3: Sample pairs
# ---------------------------------------------------------------------------
def sample_pairs(
    labels: dict[str, list[str]],
    filenames: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    print("Sampling pairs...")
    fn_set = set(filenames)
    rows = []
    for query, positives in labels.items():
        if query not in fn_set:
            continue
        pos_set = set(positives) & fn_set
        if not pos_set:
            continue
        # Positives
        for p in pos_set:
            rows.append((query, p, 1))
        # Negatives: sample from corpus minus query and positives
        neg_pool = list(fn_set - pos_set - {query})
        n_neg = min(len(pos_set) * NEG_RATIO, MAX_NEG_PER_QUERY, len(neg_pool))
        negs = rng.choice(neg_pool, size=n_neg, replace=False)
        for n in negs:
            rows.append((query, n, 0))

    df = pd.DataFrame(rows, columns=["query", "candidate", "label"])
    print(f"  {len(df)} pairs: {df['label'].sum()} positive, {(df['label'] == 0).sum()} negative")
    return df


# ---------------------------------------------------------------------------
# Step 4: Compute features
# ---------------------------------------------------------------------------
def compute_features(
    df: pd.DataFrame,
    filenames: list[str],
    texts: list[str],
    name_to_text: dict[str, str],
    bm25: SparseBM25,
    bm25_tf_matrix: csr_matrix,
    tfidf_mat,
) -> pd.DataFrame:
    print("Computing features...")
    t0 = time.time()
    fn_to_idx = {fn: i for i, fn in enumerate(filenames)}
    n_docs = len(filenames)

    # --- Precompute per-doc data structures ---
    print("  Precomputing token sets, bigrams, legal terms, lengths...")
    t_pre = time.time()
    all_token_sets: list[set[str]] = [set() for _ in range(n_docs)]
    all_legal_sets: list[set[str]] = [set() for _ in range(n_docs)]
    all_bigram_sets: list[set[tuple[str, str]]] = [set() for _ in range(n_docs)]
    all_lengths = np.zeros(n_docs, dtype=np.float64)

    for i, fn in enumerate(filenames):
        text = name_to_text[fn]
        toks = tokenize_simple(text)
        tok_set = set(toks)
        all_token_sets[i] = tok_set
        all_legal_sets[i] = tok_set & LEGAL_TERMS
        all_bigram_sets[i] = set(zip(toks, toks[1:])) if len(toks) > 1 else set()
        all_lengths[i] = len(text)
    print(f"  Precomputation done in {time.time() - t_pre:.1f}s")

    # --- Map df rows to corpus indices ---
    q_indices = np.array([fn_to_idx[q] for q in df["query"]])
    c_indices = np.array([fn_to_idx[c] for c in df["candidate"]])

    # --- BM25: vectorized sparse scoring ---
    print("  Computing BM25 scores (sparse vectorized)...")
    t_bm25 = time.time()
    bm25_scores = bm25.score_pairs(q_indices, c_indices, bm25_tf_matrix)
    print(f"  BM25 done in {time.time() - t_bm25:.1f}s")

    # --- TF-IDF cosine: batch multiply ---
    print("  Computing TF-IDF cosine similarities...")
    t_tfidf = time.time()
    # For each pair, cosine = row_dot / (norm_q * norm_c), but TfidfVectorizer
    # with default norm='l2' already normalizes rows, so cosine = dot product.
    # We compute per-pair dot products via element-wise multiply + sum.
    q_vecs = tfidf_mat[q_indices]
    c_vecs = tfidf_mat[c_indices]
    tfidf_cosines = np.array(q_vecs.multiply(c_vecs).sum(axis=1), dtype=np.float32).ravel()
    print(f"  TF-IDF done in {time.time() - t_tfidf:.1f}s")

    # --- Jaccard, legal terms, length ratio, bigrams: vectorized via precomputed ---
    print("  Computing Jaccard, legal terms, length ratio, bigrams...")
    t_other = time.time()
    jaccard = np.zeros(len(df), dtype=np.float32)
    shared_legal = np.zeros(len(df), dtype=np.float32)
    length_ratio = np.zeros(len(df), dtype=np.float32)
    shared_bigrams = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df)):
        qi, ci = q_indices[i], c_indices[i]

        # Jaccard
        q_set, c_set = all_token_sets[qi], all_token_sets[ci]
        inter = len(q_set & c_set)
        union = len(q_set | c_set)
        jaccard[i] = inter / union if union > 0 else 0.0

        # Shared legal terms
        shared_legal[i] = len(all_legal_sets[qi] & all_legal_sets[ci])

        # Length ratio
        ql, cl = all_lengths[qi], all_lengths[ci]
        length_ratio[i] = min(ql, cl) / max(ql, cl) if max(ql, cl) > 0 else 0.0

        # Shared bigrams
        q_bi, c_bi = all_bigram_sets[qi], all_bigram_sets[ci]
        bi_union = len(q_bi | c_bi)
        shared_bigrams[i] = len(q_bi & c_bi) / bi_union if bi_union > 0 else 0.0

        if (i + 1) % 20000 == 0:
            print(f"    ... {i + 1}/{len(df)} pairs")

    print(f"  Other features done in {time.time() - t_other:.1f}s")

    df["bm25_score"] = bm25_scores
    df["tfidf_cosine"] = tfidf_cosines
    df["jaccard"] = jaccard
    df["shared_legal_terms"] = shared_legal
    df["length_ratio"] = length_ratio
    df["shared_bigrams"] = shared_bigrams

    elapsed = time.time() - t0
    print(f"  All features computed in {elapsed:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Step 5: Statistical tests
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "bm25_score", "tfidf_cosine", "jaccard",
    "shared_legal_terms", "length_ratio", "shared_bigrams",
]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (positive vs. negative pairs)")
    print("=" * 70)
    results = []
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    for feat in FEATURE_COLS:
        a = pos[feat].values.astype(np.float64)
        b = neg[feat].values.astype(np.float64)
        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        ks_stat, ks_p = stats.ks_2samp(a, b)
        d = cohens_d(a, b)
        results.append({
            "Feature": feat,
            "Pos_mean": f"{a.mean():.4f}",
            "Neg_mean": f"{b.mean():.4f}",
            "Welch_t": f"{t_stat:.2f}",
            "t_pval": f"{t_p:.2e}",
            "KS_stat": f"{ks_stat:.4f}",
            "KS_pval": f"{ks_p:.2e}",
            "Cohens_d": f"{d:.3f}",
        })
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return res_df


# ---------------------------------------------------------------------------
# Step 6: ML classification (5-fold GroupKFold)
# ---------------------------------------------------------------------------
def run_classification(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("ML CLASSIFICATION (5-fold GroupKFold, grouped by query)")
    print("=" * 70)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values
    groups = df["query"].values

    gkf = GroupKFold(n_splits=5)

    classifiers = {
        "LogisticRegression": lambda: LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        ),
    }

    all_results = {}
    best_model_name = None
    best_auc = 0.0

    for name, make_clf in classifiers.items():
        print(f"\n--- {name} ---")
        fold_aucs, fold_aps = [], []
        y_true_all, y_prob_all, group_all = [], [], []
        per_query_aucs = []
        feature_importances = np.zeros(len(FEATURE_COLS))

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            clf = make_clf()
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]

            auc = roc_auc_score(y[test_idx], probs)
            ap = average_precision_score(y[test_idx], probs)
            fold_aucs.append(auc)
            fold_aps.append(ap)

            y_true_all.extend(y[test_idx])
            y_prob_all.extend(probs)
            group_all.extend(groups[test_idx])

            # Feature importances
            if hasattr(clf, "feature_importances_"):
                feature_importances += clf.feature_importances_
            elif hasattr(clf, "coef_"):
                feature_importances += np.abs(clf.coef_[0])

            # Per-query AUC
            test_df = pd.DataFrame({
                "query": groups[test_idx], "y": y[test_idx], "prob": probs
            })
            for _, qdf in test_df.groupby("query"):
                if qdf["y"].nunique() == 2:
                    per_query_aucs.append(roc_auc_score(qdf["y"], qdf["prob"]))

            print(f"  Fold {fold}: AUC={auc:.4f}  AP={ap:.4f}")

        feature_importances /= 5
        mean_auc = np.mean(fold_aucs)
        mean_ap = np.mean(fold_aps)
        print(f"  Mean AUC: {mean_auc:.4f} (+/- {np.std(fold_aucs):.4f})")
        print(f"  Mean AP:  {mean_ap:.4f} (+/- {np.std(fold_aps):.4f})")

        y_true_all = np.array(y_true_all)
        y_prob_all = np.array(y_prob_all)

        # Classification report at 0.5 threshold
        y_pred = (y_prob_all >= 0.5).astype(int)
        print(classification_report(y_true_all, y_pred, target_names=["Negative", "Positive"]))

        all_results[name] = {
            "mean_auc": mean_auc,
            "mean_ap": mean_ap,
            "fold_aucs": fold_aucs,
            "fold_aps": fold_aps,
            "y_true": y_true_all,
            "y_prob": y_prob_all,
            "y_pred": y_pred,
            "feature_importances": feature_importances,
            "per_query_aucs": per_query_aucs,
        }

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model_name = name

    return all_results, best_model_name


# ---------------------------------------------------------------------------
# Step 7: Visualizations
# ---------------------------------------------------------------------------
def plot_feature_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, feat in zip(axes.flat, FEATURE_COLS):
        parts = ax.violinplot(
            [df.loc[df["label"] == 0, feat].values, df.loc[df["label"] == 1, feat].values],
            positions=[0, 1],
            showmeans=True,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_title(feat.replace("_", " ").title())
    fig.suptitle("Feature Distributions: Positive vs. Negative Pairs", fontsize=14)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "12_signal_feature_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved: 12_signal_feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df[FEATURE_COLS + ["label"]].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap (including label)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "13_signal_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved: 13_signal_correlation_heatmap.png")


def plot_roc_pr(all_results: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for name, res in all_results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax1.plot(fpr, tpr, label=f'{name} (AUC={res["mean_auc"]:.3f})')

        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        ax2.plot(rec, prec, label=f'{name} (AP={res["mean_ap"]:.3f})')

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curves")
    ax1.legend()

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "14_signal_roc_pr.png", dpi=150)
    plt.close(fig)
    print("  Saved: 14_signal_roc_pr.png")


def plot_feature_importance(all_results: dict):
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, all_results.items()):
        imp = res["feature_importances"]
        order = np.argsort(imp)
        ax.barh([FEATURE_COLS[i] for i in order], imp[order])
        ax.set_title(f"Feature Importance: {name}")
        ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "15_signal_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  Saved: 15_signal_feature_importance.png")


def plot_per_query_auc(all_results: dict, best_model_name: str):
    per_q = all_results[best_model_name]["per_query_aucs"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_q, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(np.median(per_q), color="red", linestyle="--", label=f"Median={np.median(per_q):.3f}")
    ax.set_xlabel("Per-Query AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Query AUC Distribution ({best_model_name})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "16_signal_per_query_auc.png", dpi=150)
    plt.close(fig)
    print("  Saved: 16_signal_per_query_auc.png")


def plot_confusion_matrix(all_results: dict, best_model_name: str):
    res = all_results[best_model_name]
    cm = confusion_matrix(res["y_true"], res["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({best_model_name})")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "17_signal_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("  Saved: 17_signal_confusion_matrix.png")


# ---------------------------------------------------------------------------
# Step 8: Summary verdict
# ---------------------------------------------------------------------------
def print_verdict(all_results: dict, best_model_name: str):
    best_auc = all_results[best_model_name]["mean_auc"]
    best_ap = all_results[best_model_name]["mean_ap"]

    print("\n" + "=" * 70)
    print("SUMMARY VERDICT")
    print("=" * 70)
    print(f"Best model: {best_model_name}")
    print(f"  AUC:  {best_auc:.4f}")
    print(f"  AP:   {best_ap:.4f}")

    if best_auc >= 0.85:
        level = "STRONG"
        msg = (
            "Labels carry strong discriminative signal even with simple features.\n"
            "  Strategy: BM25 + TF-IDF baseline already separates well.\n"
            "  Focus competition effort on neural reranking to push recall on hard cases."
        )
    elif best_auc >= 0.75:
        level = "MODERATE"
        msg = (
            "Labels carry moderate signal from lexical features.\n"
            "  Strategy: BM25/TF-IDF first stage is viable, but semantic features needed.\n"
            "  Invest in dense retrieval (legal-domain embeddings) for second stage."
        )
    elif best_auc >= 0.60:
        level = "WEAK"
        msg = (
            "Lexical features alone provide weak signal.\n"
            "  Strategy: Lexical retrieval is insufficient as sole method.\n"
            "  Must use semantic models (fine-tuned legal transformers) from the start."
        )
    else:
        level = "NO SIGNAL"
        msg = (
            "No meaningful signal from lexical features.\n"
            "  Strategy: Task may require structural/metadata features or\n"
            "  fundamentally different approach (citation graph, etc.)."
        )

    print(f"\n  Signal level: ** {level} **")
    print(f"  {msg}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Step 1: Load corpus
    filenames, texts, name_to_text = load_corpus()

    # Load labels
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    print(f"Labels: {len(labels)} queries, {sum(len(v) for v in labels.values())} total citations")

    # Step 2: Build indexes
    bm25, bm25_tf_matrix = build_bm25_index(texts)
    tfidf_vec, tfidf_mat = build_tfidf(texts)

    # Step 3: Sample pairs
    df = sample_pairs(labels, filenames, rng)

    # Step 4: Compute features
    df = compute_features(df, filenames, texts, name_to_text, bm25, bm25_tf_matrix, tfidf_mat)

    # Step 5: Statistical tests
    stat_df = run_statistical_tests(df)

    # Step 6: ML classification
    all_results, best_model_name = run_classification(df)

    # Step 7: Visualizations
    print("\nGenerating plots...")
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_roc_pr(all_results)
    plot_feature_importance(all_results)
    plot_per_query_auc(all_results, best_model_name)
    plot_confusion_matrix(all_results, best_model_name)

    # Step 8: Verdict
    print_verdict(all_results, best_model_name)


if __name__ == "__main__":
    main()
