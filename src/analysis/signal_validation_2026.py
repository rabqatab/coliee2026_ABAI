"""
Task 1 Label Signal Validation — 2026 Dataset

Replicates the 2025 signal validation on the 2026 corpus and produces
side-by-side comparison plots and statistics.

Outputs:
  - Statistical test results (stdout)
  - Classifier metrics (stdout)
  - Comparison plots saved to docs/analysis/plots/
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
PLOTS_DIR = BASE_DIR / "docs" / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Run on both years
DATASETS = {
    "2025": {
        "corpus_dir": DATA_DIR / "task1_train_files_2025",
        "labels_path": DATA_DIR / "task1_train_labels_2025.json",
    },
    "2026": {
        "corpus_dir": DATA_DIR / "task1_train_files_2026",
        "labels_path": DATA_DIR / "task1_train_labels_2026.json",
    },
}

NEG_RATIO = 10
MAX_NEG_PER_QUERY = 50
TFIDF_MAX_FEATURES = 50_000
RANDOM_SEED = 42

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

FEATURE_COLS = [
    "bm25_score", "tfidf_cosine", "jaccard",
    "shared_legal_terms", "length_ratio", "shared_bigrams",
]


# ---------------------------------------------------------------------------
# Reuse functions from original (unchanged)
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = text.replace("<FRAGMENT_SUPPRESSED>", "")
    text = text.replace("[End of document]", "")
    text = re.sub(r"\n(?=[a-z])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_simple(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{2,}", text.lower())


def load_corpus(corpus_dir: Path) -> tuple[list[str], list[str], dict[str, str]]:
    print(f"Loading corpus from {corpus_dir.name}...")
    t0 = time.time()
    filenames = sorted(p.name for p in corpus_dir.glob("*.txt"))
    texts = []
    name_to_text = {}
    for fn in filenames:
        raw = (corpus_dir / fn).read_text(errors="replace")
        cleaned = preprocess(raw)
        texts.append(cleaned)
        name_to_text[fn] = cleaned
    print(f"  Loaded {len(filenames)} documents in {time.time() - t0:.1f}s")
    return filenames, texts, name_to_text


class SparseBM25:
    def __init__(self, tf_matrix: csr_matrix, k1: float = 1.5, b: float = 0.75):
        n_docs, n_terms = tf_matrix.shape
        doc_lens = np.array(tf_matrix.sum(axis=1)).ravel().astype(np.float64)
        avgdl = doc_lens.mean()
        df = np.array((tf_matrix > 0).sum(axis=0)).ravel().astype(np.float64)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf_csr = tf_matrix.tocsr().astype(np.float64)
        bm25_data = np.empty(tf_csr.nnz, dtype=np.float64)
        for i in range(n_docs):
            start, end = tf_csr.indptr[i], tf_csr.indptr[i + 1]
            tf_vals = tf_csr.data[start:end]
            denom = tf_vals + k1 * (1.0 - b + b * doc_lens[i] / avgdl)
            bm25_data[start:end] = tf_vals * (k1 + 1.0) / denom
        self.bm25_tf = csr_matrix(
            (bm25_data, tf_csr.indices, tf_csr.indptr), shape=tf_csr.shape
        )
        self.idf = idf

    def score_pairs(self, q_indices, c_indices, q_tf_matrix):
        n_pairs = len(q_indices)
        scores = np.zeros(n_pairs, dtype=np.float64)
        q_binary = (q_tf_matrix > 0).astype(np.float64)
        chunk_size = 5000
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            qi = q_indices[start:end]
            ci = c_indices[start:end]
            q_terms = q_binary[qi]
            c_bm25 = self.bm25_tf[ci]
            weighted = q_terms.multiply(c_bm25).multiply(self.idf[np.newaxis, :])
            scores[start:end] = np.array(weighted.sum(axis=1)).ravel()
        return scores.astype(np.float32)


def build_bm25_index(texts):
    print("Building BM25 index...")
    t0 = time.time()
    cv = CountVectorizer(token_pattern=r"[a-zA-Z]{2,}", lowercase=True)
    tf_matrix = cv.fit_transform(texts)
    bm25 = SparseBM25(tf_matrix)
    print(f"  BM25 built in {time.time() - t0:.1f}s (vocab={tf_matrix.shape[1]})")
    return bm25, tf_matrix


def build_tfidf(texts):
    print("Building TF-IDF matrix...")
    t0 = time.time()
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True, dtype=np.float32)
    mat = vec.fit_transform(texts)
    print(f"  TF-IDF shape: {mat.shape}, built in {time.time() - t0:.1f}s")
    return vec, mat


def sample_pairs(labels, filenames, rng):
    print("Sampling pairs...")
    fn_set = set(filenames)
    rows = []
    for query, positives in labels.items():
        if query not in fn_set:
            continue
        pos_set = set(positives) & fn_set
        if not pos_set:
            continue
        for p in pos_set:
            rows.append((query, p, 1))
        neg_pool = list(fn_set - pos_set - {query})
        n_neg = min(len(pos_set) * NEG_RATIO, MAX_NEG_PER_QUERY, len(neg_pool))
        negs = rng.choice(neg_pool, size=n_neg, replace=False)
        for n in negs:
            rows.append((query, n, 0))
    df = pd.DataFrame(rows, columns=["query", "candidate", "label"])
    print(f"  {len(df)} pairs: {df['label'].sum()} positive, {(df['label'] == 0).sum()} negative")
    return df


def compute_features(df, filenames, texts, name_to_text, bm25, bm25_tf_matrix, tfidf_mat):
    print("Computing features...")
    t0 = time.time()
    fn_to_idx = {fn: i for i, fn in enumerate(filenames)}
    n_docs = len(filenames)

    all_token_sets = [set() for _ in range(n_docs)]
    all_legal_sets = [set() for _ in range(n_docs)]
    all_bigram_sets = [set() for _ in range(n_docs)]
    all_lengths = np.zeros(n_docs, dtype=np.float64)

    for i, fn in enumerate(filenames):
        toks = tokenize_simple(name_to_text[fn])
        tok_set = set(toks)
        all_token_sets[i] = tok_set
        all_legal_sets[i] = tok_set & LEGAL_TERMS
        all_bigram_sets[i] = set(zip(toks, toks[1:])) if len(toks) > 1 else set()
        all_lengths[i] = len(name_to_text[fn])

    q_indices = np.array([fn_to_idx[q] for q in df["query"]])
    c_indices = np.array([fn_to_idx[c] for c in df["candidate"]])

    bm25_scores = bm25.score_pairs(q_indices, c_indices, bm25_tf_matrix)

    q_vecs = tfidf_mat[q_indices]
    c_vecs = tfidf_mat[c_indices]
    tfidf_cosines = np.array(q_vecs.multiply(c_vecs).sum(axis=1), dtype=np.float32).ravel()

    jaccard = np.zeros(len(df), dtype=np.float32)
    shared_legal = np.zeros(len(df), dtype=np.float32)
    length_ratio = np.zeros(len(df), dtype=np.float32)
    shared_bigrams = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df)):
        qi, ci = q_indices[i], c_indices[i]
        q_set, c_set = all_token_sets[qi], all_token_sets[ci]
        inter = len(q_set & c_set)
        union = len(q_set | c_set)
        jaccard[i] = inter / union if union > 0 else 0.0
        shared_legal[i] = len(all_legal_sets[qi] & all_legal_sets[ci])
        ql, cl = all_lengths[qi], all_lengths[ci]
        length_ratio[i] = min(ql, cl) / max(ql, cl) if max(ql, cl) > 0 else 0.0
        q_bi, c_bi = all_bigram_sets[qi], all_bigram_sets[ci]
        bi_union = len(q_bi | c_bi)
        shared_bigrams[i] = len(q_bi & c_bi) / bi_union if bi_union > 0 else 0.0

        if (i + 1) % 20000 == 0:
            print(f"    ... {i + 1}/{len(df)} pairs")

    df["bm25_score"] = bm25_scores
    df["tfidf_cosine"] = tfidf_cosines
    df["jaccard"] = jaccard
    df["shared_legal_terms"] = shared_legal
    df["length_ratio"] = length_ratio
    df["shared_bigrams"] = shared_bigrams

    print(f"  Features computed in {time.time() - t0:.1f}s")
    return df


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def run_statistical_tests(df):
    print(f"\n{'='*70}")
    print("STATISTICAL TESTS")
    print(f"{'='*70}")
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
            "Pos_mean": a.mean(),
            "Neg_mean": b.mean(),
            "Welch_t": t_stat,
            "KS_stat": ks_stat,
            "Cohens_d": d,
        })
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return res_df


def run_classification(df):
    print(f"\n{'='*70}")
    print("ML CLASSIFICATION (5-fold GroupKFold)")
    print(f"{'='*70}")

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
            n_estimators=300, learning_rate=0.05, is_unbalance=True,
            random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
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

            if hasattr(clf, "feature_importances_"):
                feature_importances += clf.feature_importances_
            elif hasattr(clf, "coef_"):
                feature_importances += np.abs(clf.coef_[0])

            test_df = pd.DataFrame({"query": groups[test_idx], "y": y[test_idx], "prob": probs})
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
        y_pred = (y_prob_all >= 0.5).astype(int)
        print(classification_report(y_true_all, y_pred, target_names=["Negative", "Positive"]))

        all_results[name] = {
            "mean_auc": mean_auc,
            "mean_ap": mean_ap,
            "fold_aucs": fold_aucs,
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
# Pipeline for one year
# ---------------------------------------------------------------------------
def run_pipeline(year: str, corpus_dir: Path, labels_path: Path):
    """Run full signal validation for one year. Returns (df, stat_df, all_results, best_model)."""
    print(f"\n{'#'*70}")
    print(f"# SIGNAL VALIDATION — {year}")
    print(f"{'#'*70}")

    rng = np.random.default_rng(RANDOM_SEED)
    filenames, texts, name_to_text = load_corpus(corpus_dir)

    with open(labels_path) as f:
        labels = json.load(f)
    # Filter to queries with non-empty labels
    labels = {k: v for k, v in labels.items() if v}
    print(f"Labels: {len(labels)} queries, {sum(len(v) for v in labels.values())} citations")

    bm25, bm25_tf_matrix = build_bm25_index(texts)
    _, tfidf_mat = build_tfidf(texts)
    df = sample_pairs(labels, filenames, rng)
    df = compute_features(df, filenames, texts, name_to_text, bm25, bm25_tf_matrix, tfidf_mat)
    stat_df = run_statistical_tests(df)
    all_results, best_model = run_classification(df)

    return df, stat_df, all_results, best_model


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------
def plot_comparison(results_by_year: dict):
    """Generate comparison plots between years."""
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*70}")

    sns.set_style("whitegrid")
    c25 = "#4C72B0"
    c26 = "#DD8452"

    # --- Plot 1: Feature distributions side-by-side ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, feat in zip(axes.flat, FEATURE_COLS):
        data_to_plot = []
        labels_plot = []
        for year in ["2025", "2026"]:
            df = results_by_year[year]["df"]
            data_to_plot.append(df.loc[df["label"] == 1, feat].values)
            labels_plot.append(f"{year}\nPos")
            data_to_plot.append(df.loc[df["label"] == 0, feat].values)
            labels_plot.append(f"{year}\nNeg")

        parts = ax.violinplot(data_to_plot, positions=[0, 1, 2.5, 3.5],
                              showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(c25 if i < 2 else c26)
            pc.set_alpha(0.6)
        ax.set_xticks([0, 1, 2.5, 3.5])
        ax.set_xticklabels(labels_plot, fontsize=8)
        ax.set_title(feat.replace("_", " ").title())
    fig.suptitle("Feature Distributions: 2025 vs 2026", fontsize=14)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "23_signal_2025v2026_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved: 23_signal_2025v2026_distributions.png")

    # --- Plot 2: ROC/PR comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    styles = {"2025": (c25, "-"), "2026": (c26, "--")}
    for year in ["2025", "2026"]:
        color, ls = styles[year]
        best = results_by_year[year]["best_model"]
        res = results_by_year[year]["all_results"][best]
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax1.plot(fpr, tpr, color=color, linestyle=ls,
                 label=f'{year} {best} (AUC={res["mean_auc"]:.3f})')
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        ax2.plot(rec, prec, color=color, linestyle=ls,
                 label=f'{year} {best} (AP={res["mean_ap"]:.3f})')

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curves (Best Model)"); ax1.legend()
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves (Best Model)"); ax2.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "24_signal_2025v2026_roc_pr.png", dpi=150)
    plt.close(fig)
    print("  Saved: 24_signal_2025v2026_roc_pr.png")

    # --- Plot 3: Per-query AUC comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, year, color in zip(axes, ["2025", "2026"], [c25, c26]):
        best = results_by_year[year]["best_model"]
        per_q = results_by_year[year]["all_results"][best]["per_query_aucs"]
        ax.hist(per_q, bins=30, edgecolor="black", alpha=0.7, color=color)
        ax.axvline(np.median(per_q), color="red", linestyle="--",
                   label=f"Median={np.median(per_q):.3f}")
        ax.set_xlabel("Per-Query AUC")
        ax.set_ylabel("Count")
        ax.set_title(f"{year}: Per-Query AUC ({best})")
        ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "25_signal_2025v2026_per_query_auc.png", dpi=150)
    plt.close(fig)
    print("  Saved: 25_signal_2025v2026_per_query_auc.png")

    # --- Plot 4: Cohen's d comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(FEATURE_COLS))
    width = 0.35
    for i, (year, color) in enumerate(zip(["2025", "2026"], [c25, c26])):
        stat_df = results_by_year[year]["stat_df"]
        ds = stat_df["Cohens_d"].values
        ax.bar(x + (i - 0.5) * width, ds, width, label=year, color=color, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in FEATURE_COLS], fontsize=8)
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Sizes: 2025 vs 2026")
    ax.legend()
    ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.5, label="Large effect (0.8)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "26_signal_2025v2026_effect_sizes.png", dpi=150)
    plt.close(fig)
    print("  Saved: 26_signal_2025v2026_effect_sizes.png")


def print_comparison(results_by_year: dict):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print("SIGNAL VALIDATION COMPARISON: 2025 vs 2026")
    print(f"{'='*70}")

    # Statistical tests
    print(f"\n{'Feature':<20} {'Cohen_d 2025':>14} {'Cohen_d 2026':>14} {'Change':>10}")
    print("-" * 62)
    for i, feat in enumerate(FEATURE_COLS):
        d25 = results_by_year["2025"]["stat_df"].iloc[i]["Cohens_d"]
        d26 = results_by_year["2026"]["stat_df"].iloc[i]["Cohens_d"]
        change = f"{(d26 - d25) / abs(d25) * 100:+.1f}%" if d25 != 0 else "N/A"
        print(f"{feat:<20} {d25:>14.3f} {d26:>14.3f} {change:>10}")

    # Classifier performance
    print(f"\n{'Model':<20} {'AUC 2025':>10} {'AUC 2026':>10} {'AP 2025':>10} {'AP 2026':>10}")
    print("-" * 65)
    for model in ["LogisticRegression", "RandomForest", "LightGBM"]:
        auc25 = results_by_year["2025"]["all_results"][model]["mean_auc"]
        auc26 = results_by_year["2026"]["all_results"][model]["mean_auc"]
        ap25 = results_by_year["2025"]["all_results"][model]["mean_ap"]
        ap26 = results_by_year["2026"]["all_results"][model]["mean_ap"]
        print(f"{model:<20} {auc25:>10.4f} {auc26:>10.4f} {ap25:>10.4f} {ap26:>10.4f}")

    # Per-query AUC
    print(f"\n{'Metric':<30} {'2025':>10} {'2026':>10}")
    print("-" * 55)
    for year in ["2025", "2026"]:
        pass
    best_25 = results_by_year["2025"]["best_model"]
    best_26 = results_by_year["2026"]["best_model"]
    pq25 = results_by_year["2025"]["all_results"][best_25]["per_query_aucs"]
    pq26 = results_by_year["2026"]["all_results"][best_26]["per_query_aucs"]
    print(f"{'Median per-query AUC':<30} {np.median(pq25):>10.3f} {np.median(pq26):>10.3f}")
    print(f"{'Queries with AUC > 0.95':<30} {np.mean(np.array(pq25) > 0.95)*100:>9.1f}% {np.mean(np.array(pq26) > 0.95)*100:>9.1f}%")
    print(f"{'Queries with AUC < 0.6':<30} {np.sum(np.array(pq25) < 0.6):>10} {np.sum(np.array(pq26) < 0.6):>10}")

    # Verdict
    best_auc_26 = results_by_year["2026"]["all_results"][best_26]["mean_auc"]
    best_auc_25 = results_by_year["2025"]["all_results"][best_25]["mean_auc"]
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if best_auc_26 >= 0.85:
        print(f"  2026 signal: STRONG (AUC={best_auc_26:.4f})")
    elif best_auc_26 >= 0.75:
        print(f"  2026 signal: MODERATE (AUC={best_auc_26:.4f})")
    else:
        print(f"  2026 signal: WEAK (AUC={best_auc_26:.4f})")

    diff = best_auc_26 - best_auc_25
    if abs(diff) < 0.01:
        print(f"  Signal stability: CONSISTENT (delta={diff:+.4f})")
    elif diff > 0:
        print(f"  Signal stability: IMPROVED (delta={diff:+.4f})")
    else:
        print(f"  Signal stability: DEGRADED (delta={diff:+.4f})")

    print(f"  Implication: {'Pipeline from 2025 should transfer well to 2026' if abs(diff) < 0.02 else 'May need pipeline adjustments for 2026'}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    results_by_year = {}

    for year, cfg in DATASETS.items():
        df, stat_df, all_results, best_model = run_pipeline(
            year, cfg["corpus_dir"], cfg["labels_path"]
        )
        results_by_year[year] = {
            "df": df,
            "stat_df": stat_df,
            "all_results": all_results,
            "best_model": best_model,
        }

    print_comparison(results_by_year)
    plot_comparison(results_by_year)

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
