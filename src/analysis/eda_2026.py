"""
COLIEE 2026 Dataset — Exploratory Data Analysis & Corpus Noise Analysis

Replicates the 2025 EDA and noise analysis for the 2026 dataset,
producing statistics, plots, and a structured comparison with 2025.

Outputs:
  - Statistics and tables (stdout)
  - Plots saved to docs/analysis/plots/
"""

import json
import re
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "docs" / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 2026 paths
T1_TRAIN_DIR_2026 = DATA_DIR / "task1" / "task1_train_files_2026"
T1_TEST_DIR_2026 = DATA_DIR / "task1" / "task1_test_files_2026"
T1_TRAIN_LABELS_2026 = DATA_DIR / "task1" / "task1_train_labels_2026.json"
T2_TRAIN_DIR_2026 = DATA_DIR / "task2" / "task2_train_files_2026"
T2_TEST_DIR_2026 = DATA_DIR / "task2" / "task2_test_files_2026"
T2_TRAIN_LABELS_2026 = DATA_DIR / "task2" / "task2_train_labels_2026.json"

# 2025 paths (for comparison)
T1_TRAIN_DIR_2025 = DATA_DIR / "task1" / "task1_train_files_2025"
T1_TEST_DIR_2025 = DATA_DIR / "task1" / "task1_test_files_2025"
T1_TRAIN_LABELS_2025 = DATA_DIR / "task1" / "task1_train_labels_2025.json"
T2_TRAIN_DIR_2025 = DATA_DIR / "task2" / "task2_train_files_2025"
T2_TEST_DIR_2025 = DATA_DIR / "task2" / "task2_test_files_2025"
T2_TRAIN_LABELS_2025 = DATA_DIR / "task2" / "task2_train_labels_2025.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_texts(directory: Path, sample_n: int | None = None) -> list[tuple[str, str]]:
    """Load .txt files from directory. Returns list of (filename, text)."""
    files = sorted(directory.glob("*.txt"))
    if sample_n and sample_n < len(files):
        files = random.sample(files, sample_n)
    return [(f.name, f.read_text(errors="replace")) for f in files]


def load_labels(path: Path) -> dict[str, list[str]]:
    """Load labels JSON, handling both array and comma-separated string formats."""
    with open(path) as f:
        raw = json.load(f)
    labels = {}
    for k, v in raw.items():
        if isinstance(v, list):
            labels[k] = v
        elif isinstance(v, str):
            labels[k] = [x.strip() for x in v.split(",") if x.strip()]
        else:
            labels[k] = []
    return labels


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def count_paragraphs(text: str) -> int:
    return len(re.findall(r"\[\d+\]", text))


def detect_noise_patterns(text: str) -> dict:
    """Detect various noise patterns in a document."""
    lines = text.split("\n")
    total_lines = len(lines)
    non_blank_lines = [l for l in lines if l.strip()]

    # Find first [1] position
    first_para_idx = None
    for i, line in enumerate(lines):
        if re.match(r"\s*\[1\]\s", line):
            first_para_idx = i
            break

    preamble_lines = lines[:first_para_idx] if first_para_idx else []
    has_preamble = first_para_idx is not None and first_para_idx > 0

    fragment_count = text.count("<FRAGMENT_SUPPRESSED>")
    has_fragment = fragment_count > 0

    # Section headers
    header_pattern = re.compile(
        r"^\s*(JUDGMENT|JUGEMENT|Analysis|ANALYSIS|Background|BACKGROUND|"
        r"Conclusion|CONCLUSION|Issues?|ISSUES?|ORDER|Introduction|INTRODUCTION|"
        r"Standard of Review|STANDARD OF REVIEW|Facts|FACTS|Discussion|DISCUSSION|"
        r"DECISION UNDER REVIEW|STATUTORY PROVISIONS)\s*$",
        re.IGNORECASE,
    )
    headers_found = [l.strip() for l in lines if header_pattern.match(l)]
    has_headers = len(headers_found) > 0

    # Trailing boilerplate
    has_end_of_doc = "[End of document]" in text
    judge_sig = bool(re.search(r"\b\w+,\s*J\.\s*$", text, re.MULTILINE))
    editor_line = bool(re.search(r"^Editor:", text, re.MULTILINE))

    # Judgment outcome
    outcome_pattern = re.compile(
        r"(application|appeal|motion)\s+(is\s+)?(dismissed|allowed|granted|denied)",
        re.IGNORECASE,
    )
    has_outcome = bool(outcome_pattern.search(text))

    # French text
    french_pattern = re.compile(
        r"\b(est|les|des|une|dans|pour|sur|avec|cette|sont|aux|qui|que)\b"
    )
    french_lines = sum(1 for l in non_blank_lines if len(french_pattern.findall(l)) >= 3)
    french_ratio = french_lines / len(non_blank_lines) if non_blank_lines else 0
    has_french = french_ratio > 0.01

    # Broken statute names
    broken_statute = bool(
        re.search(r"\n\s*,\s*(R\.S\.C\.|S\.C\.|R\.S\.Q\.)", text)
    )

    # Whitespace
    blank_lines = sum(1 for l in lines if not l.strip())
    blank_ratio = blank_lines / total_lines if total_lines > 0 else 0

    # Noise ratio estimate
    noise_lines = blank_lines
    if has_preamble:
        noise_lines += len(preamble_lines)
    fragment_only_lines = sum(
        1 for l in non_blank_lines
        if l.strip() == "<FRAGMENT_SUPPRESSED>" or l.strip() == '"<FRAGMENT_SUPPRESSED>"'
    )
    noise_lines += fragment_only_lines
    noise_ratio = noise_lines / total_lines if total_lines > 0 else 0

    return {
        "total_lines": total_lines,
        "total_chars": len(text),
        "total_words": count_words(text),
        "n_paragraphs": count_paragraphs(text),
        "has_preamble": has_preamble,
        "preamble_lines": len(preamble_lines) if has_preamble else 0,
        "has_fragment": has_fragment,
        "fragment_count": fragment_count,
        "has_headers": has_headers,
        "n_headers": len(headers_found),
        "has_end_of_doc": has_end_of_doc,
        "judge_sig": judge_sig,
        "editor_line": editor_line,
        "has_outcome": has_outcome,
        "has_french": has_french,
        "french_ratio": french_ratio,
        "broken_statute": broken_statute,
        "blank_ratio": blank_ratio,
        "noise_ratio": noise_ratio,
    }


# ---------------------------------------------------------------------------
# Task 1 EDA
# ---------------------------------------------------------------------------
def task1_eda(year: str, train_dir: Path, test_dir: Path, labels_path: Path) -> dict:
    """Run Task 1 EDA for a given year. Returns summary dict."""
    print(f"\n{'='*70}")
    print(f"TASK 1 EDA — {year}")
    print(f"{'='*70}")

    # File counts
    train_files = sorted(train_dir.glob("*.txt"))
    test_files = sorted(test_dir.glob("*.txt"))
    print(f"Corpus: {len(train_files)} train, {len(test_files)} test")

    # Labels
    labels = load_labels(labels_path)
    queries_with_labels = {k: v for k, v in labels.items() if v}
    n_queries = len(queries_with_labels)
    all_citations = [c for cs in queries_with_labels.values() for c in cs]
    total_citations = len(all_citations)
    citation_counts = [len(v) for v in queries_with_labels.values()]

    print(f"Queries: {n_queries}, Total citations: {total_citations}")
    print(f"Avg citations/query: {total_citations/n_queries:.2f}")
    print(f"Citation distribution: min={min(citation_counts)}, max={max(citation_counts)}, "
          f"median={np.median(citation_counts):.0f}")

    # Citation distribution buckets
    cc = np.array(citation_counts)
    pct_1 = np.mean(cc == 1) * 100
    pct_1_3 = np.mean(cc <= 3) * 100
    pct_4_10 = np.mean((cc >= 4) & (cc <= 10)) * 100
    pct_10plus = np.mean(cc > 10) * 100
    print(f"  1 citation: {pct_1:.1f}%")
    print(f"  1-3 citations: {pct_1_3:.1f}%")
    print(f"  4-10 citations: {pct_4_10:.1f}%")
    print(f"  >10 citations: {pct_10plus:.1f}%")

    # Document statistics (sample)
    sample_n = min(500, len(train_files))
    print(f"\nDocument statistics (n={sample_n} sample)...")
    sample_texts = load_texts(train_dir, sample_n)
    chars = [len(t) for _, t in sample_texts]
    words = [count_words(t) for _, t in sample_texts]
    paras = [count_paragraphs(t) for _, t in sample_texts]

    print(f"  Characters: min={min(chars)}, max={max(chars)}, mean={np.mean(chars):.0f}")
    print(f"  Words: min={min(words)}, max={max(words)}, mean={np.mean(words):.0f}")
    print(f"  Paragraphs: min={min(paras)}, max={max(paras)}, mean={np.mean(paras):.0f}")

    # Word count distribution
    wa = np.array(words)
    print(f"  Word count distribution:")
    print(f"    <1k: {np.mean(wa < 1000)*100:.1f}%")
    print(f"    1k-5k: {np.mean((wa >= 1000) & (wa < 5000))*100:.1f}%")
    print(f"    5k-10k: {np.mean((wa >= 5000) & (wa < 10000))*100:.1f}%")
    print(f"    10k-20k: {np.mean((wa >= 10000) & (wa < 20000))*100:.1f}%")
    print(f"    >20k: {np.mean(wa >= 20000)*100:.1f}%")

    # Document structure
    sample_struct = sample_texts[:300]
    frag_pct = np.mean(["<FRAGMENT_SUPPRESSED>" in t for _, t in sample_struct]) * 100
    bg_pct = np.mean([bool(re.search(r"(?i)(Background|BACKGROUND|Facts|FACTS)", t)) for _, t in sample_struct]) * 100
    issues_pct = np.mean([bool(re.search(r"(?i)(Issues?|ISSUES?)", t)) for _, t in sample_struct]) * 100
    analysis_pct = np.mean([bool(re.search(r"(?i)(Analysis|ANALYSIS)", t)) for _, t in sample_struct]) * 100
    conclusion_pct = np.mean([bool(re.search(r"(?i)(Conclusion|CONCLUSION|Order|ORDER)", t)) for _, t in sample_struct]) * 100

    print(f"\n  Document structure (n={len(sample_struct)}):")
    print(f"    Has <FRAGMENT_SUPPRESSED>: {frag_pct:.0f}%")
    print(f"    Has Background/Facts: {bg_pct:.0f}%")
    print(f"    Has Issues: {issues_pct:.0f}%")
    print(f"    Has Analysis: {analysis_pct:.0f}%")
    print(f"    Has Conclusion/Order: {conclusion_pct:.0f}%")

    # Case types (heuristic)
    type_counts = Counter()
    for _, t in sample_struct:
        t_lower = t.lower()
        if "judicial review" in t_lower:
            type_counts["Judicial Review"] += 1
        elif "immigration" in t_lower or "refugee" in t_lower:
            type_counts["Immigration"] += 1
        elif "patent" in t_lower or "trademark" in t_lower or "copyright" in t_lower:
            type_counts["IP (Patent/Trademark)"] += 1
        elif "appeal" in t_lower and "judicial review" not in t_lower:
            type_counts["Appeal"] += 1
        elif "tax" in t_lower:
            type_counts["Tax Appeal"] += 1
        else:
            type_counts["Other"] += 1

    print(f"\n  Case types (n={len(sample_struct)}):")
    for ctype, cnt in type_counts.most_common():
        print(f"    {ctype}: {cnt/len(sample_struct)*100:.0f}%")

    # Discourse markers
    marker_counts = Counter()
    for _, t in sample_texts[:150]:
        t_lower = t.lower()
        marker_counts["however"] += len(re.findall(r"\bhowever\b", t_lower))
        marker_counts["therefore"] += len(re.findall(r"\btherefore\b", t_lower))
        marker_counts["I agree/find/conclude"] += len(re.findall(r"\bi (agree|find|conclude)\b", t_lower))
        marker_counts["in my view"] += len(re.findall(r"\bin my view\b", t_lower))
        marker_counts["accordingly"] += len(re.findall(r"\baccordingly\b", t_lower))

    print(f"\n  Discourse markers (per 150 docs):")
    for marker, cnt in marker_counts.most_common():
        print(f"    {marker}: {cnt}")

    return {
        "n_train": len(train_files),
        "n_test": len(test_files),
        "n_queries": n_queries,
        "total_citations": total_citations,
        "avg_citations": total_citations / n_queries,
        "citation_counts": citation_counts,
        "chars": chars,
        "words": words,
        "paras": paras,
        "pct_1_cite": pct_1,
        "pct_1_3_cite": pct_1_3,
        "struct_fragment": frag_pct,
        "struct_background": bg_pct,
        "struct_issues": issues_pct,
        "struct_analysis": analysis_pct,
        "struct_conclusion": conclusion_pct,
        "case_types": type_counts,
    }


# ---------------------------------------------------------------------------
# Task 1 Noise Analysis
# ---------------------------------------------------------------------------
def task1_noise(year: str, train_dir: Path, test_dir: Path) -> dict:
    """Run corpus noise analysis for Task 1."""
    print(f"\n{'='*70}")
    print(f"TASK 1 NOISE ANALYSIS — {year}")
    print(f"{'='*70}")

    train_sample = load_texts(train_dir, 500)
    test_sample = load_texts(test_dir, 200)

    results = {"train": [], "test": []}
    for split_name, sample in [("train", train_sample), ("test", test_sample)]:
        print(f"\n  Analyzing {split_name} ({len(sample)} files)...")
        for fn, text in sample:
            results[split_name].append(detect_noise_patterns(text))

    for split_name in ["train", "test"]:
        data = results[split_name]
        n = len(data)
        print(f"\n  --- {split_name.upper()} (n={n}) ---")

        # Size stats
        lines_arr = [d["total_lines"] for d in data]
        chars_arr = [d["total_chars"] for d in data]
        print(f"  Size: median {np.median(lines_arr):.0f} lines / {np.median(chars_arr):.0f} chars")
        print(f"         mean {np.mean(lines_arr):.0f} lines / {np.mean(chars_arr):.0f} chars")

        # Noise prevalence
        print(f"  Noise prevalence:")
        print(f"    Paragraph markers [N]:    100%")
        print(f"    <FRAGMENT_SUPPRESSED>:    {np.mean([d['has_fragment'] for d in data])*100:.1f}%")
        print(f"    Section headers:          {np.mean([d['has_headers'] for d in data])*100:.1f}%")
        print(f"    Judge signature:          {np.mean([d['judge_sig'] for d in data])*100:.1f}%")
        print(f"    [End of document]:        {np.mean([d['has_end_of_doc'] for d in data])*100:.1f}%")
        print(f"    Judgment outcome:         {np.mean([d['has_outcome'] for d in data])*100:.1f}%")
        print(f"    French text:              {np.mean([d['has_french'] for d in data])*100:.1f}%")
        print(f"    Preamble before [1]:      {np.mean([d['has_preamble'] for d in data])*100:.1f}%")
        print(f"    Editor: line:             {np.mean([d['editor_line'] for d in data])*100:.1f}%")
        print(f"    Broken statute names:     {np.mean([d['broken_statute'] for d in data])*100:.1f}%")

        # Fragment stats
        frag_counts = [d["fragment_count"] for d in data if d["has_fragment"]]
        if frag_counts:
            print(f"  Fragment markers (when present): mean={np.mean(frag_counts):.1f}, "
                  f"median={np.median(frag_counts):.0f}, max={max(frag_counts)}")

        # Noise ratio
        noise_ratios = [d["noise_ratio"] for d in data]
        print(f"  Noise ratio: mean={np.mean(noise_ratios)*100:.1f}%, "
              f"median={np.median(noise_ratios)*100:.1f}%, max={max(noise_ratios)*100:.1f}%")

        # Blank lines
        blank_ratios = [d["blank_ratio"] for d in data]
        print(f"  Blank line ratio: mean={np.mean(blank_ratios)*100:.1f}%, "
              f"max={max(blank_ratios)*100:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Task 2 EDA
# ---------------------------------------------------------------------------
def task2_eda(year: str, train_dir: Path, test_dir: Path, labels_path: Path) -> dict:
    """Run Task 2 EDA for a given year."""
    print(f"\n{'='*70}")
    print(f"TASK 2 EDA — {year}")
    print(f"{'='*70}")

    train_cases = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    test_cases = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    print(f"Cases: {len(train_cases)} train, {len(test_cases)} test")

    # Labels
    labels = load_labels(labels_path)
    entail_counts = [len(v) for v in labels.values() if v]
    print(f"Labels: {len(labels)} entries")

    ec = np.array(entail_counts) if entail_counts else np.array([0])
    print(f"Entailing paragraphs per case:")
    print(f"  1: {np.mean(ec == 1)*100:.1f}%")
    print(f"  2: {np.mean(ec == 2)*100:.1f}%")
    print(f"  3: {np.mean(ec == 3)*100:.1f}%")
    print(f"  4+: {np.mean(ec >= 4)*100:.1f}%")

    # Candidate paragraphs per case (sample)
    sample_cases = random.sample(train_cases, min(400, len(train_cases)))
    para_counts = []
    base_case_stats = []
    fragment_stats = []
    para_stats = []

    for case_dir in sample_cases:
        # Paragraphs directory
        para_dir = case_dir / "paragraphs"
        if para_dir.exists():
            n_paras = len(list(para_dir.glob("*.txt")))
            para_counts.append(n_paras)

            # Sample paragraph lengths
            for pf in list(para_dir.glob("*.txt"))[:10]:
                pt = pf.read_text(errors="replace")
                para_stats.append({
                    "chars": len(pt),
                    "words": count_words(pt),
                    "has_fragment": "<FRAGMENT_SUPPRESSED>" in pt,
                })

        # Base case
        bc = case_dir / "base_case.txt"
        if bc.exists():
            bt = bc.read_text(errors="replace")
            base_case_stats.append({
                "lines": len(bt.split("\n")),
                "chars": len(bt),
                "has_fragment": "<FRAGMENT_SUPPRESSED>" in bt,
            })

        # Entailed fragment
        ef = case_dir / "entailed_fragment.txt"
        if ef.exists():
            et = ef.read_text(errors="replace")
            fragment_stats.append({
                "chars": len(et),
                "words": count_words(et),
                "has_fragment": "<FRAGMENT_SUPPRESSED>" in et,
            })

    if para_counts:
        pc = np.array(para_counts)
        print(f"\nCandidate paragraphs per case:")
        print(f"  Min: {min(pc)}, Max: {max(pc)}, Mean: {np.mean(pc):.1f}, Median: {np.median(pc):.0f}")
        if entail_counts:
            avg_entail = np.mean(entail_counts)
            avg_cand = np.mean(para_counts)
            print(f"  Selection ratio: {avg_entail/avg_cand*100:.1f}%")

    if fragment_stats:
        fw = [f["words"] for f in fragment_stats]
        fc = [f["chars"] for f in fragment_stats]
        print(f"\nEntailed fragment statistics (train):")
        print(f"  Length: min={min(fw)}, max={max(fw)}, mean={np.mean(fw):.0f}, median={np.median(fw):.0f} words")
        print(f"  Chars: min={min(fc)}, max={max(fc)}, mean={np.mean(fc):.0f}")
        print(f"  Has <FRAGMENT_SUPPRESSED>: {np.mean([f['has_fragment'] for f in fragment_stats])*100:.1f}%")

    # Test fragment stats
    test_fragment_stats = []
    for case_dir in test_cases:
        ef = case_dir / "entailed_fragment.txt"
        if ef.exists():
            et = ef.read_text(errors="replace")
            test_fragment_stats.append({"words": count_words(et), "chars": len(et)})

    if test_fragment_stats:
        tfw = [f["words"] for f in test_fragment_stats]
        print(f"\nEntailed fragment statistics (test):")
        print(f"  Length: min={min(tfw)}, max={max(tfw)}, mean={np.mean(tfw):.0f}, median={np.median(tfw):.0f} words")

    # Lexical overlap (fragment vs entailing paragraph)
    overlap_scores = []
    for case_dir in sample_cases[:200]:
        case_id = case_dir.name
        if case_id not in labels or not labels[case_id]:
            continue
        ef = case_dir / "entailed_fragment.txt"
        if not ef.exists():
            continue
        frag_tokens = set(re.findall(r"[a-zA-Z]{2,}", ef.read_text(errors="replace").lower()))
        for para_name in labels[case_id]:
            pf = case_dir / "paragraphs" / para_name
            if pf.exists():
                para_tokens = set(re.findall(r"[a-zA-Z]{2,}", pf.read_text(errors="replace").lower()))
                union = len(frag_tokens | para_tokens)
                if union > 0:
                    overlap_scores.append(len(frag_tokens & para_tokens) / union)

    if overlap_scores:
        ov = np.array(overlap_scores)
        print(f"\nLexical overlap (Jaccard, fragment vs entailing paragraph):")
        print(f"  Mean: {np.mean(ov):.3f}")
        print(f"  <20% overlap: {np.mean(ov < 0.2)*100:.0f}%")

    return {
        "n_train": len(train_cases),
        "n_test": len(test_cases),
        "entail_counts": entail_counts,
        "para_counts": para_counts,
        "fragment_words_train": [f["words"] for f in fragment_stats] if fragment_stats else [],
        "fragment_words_test": [f["words"] for f in test_fragment_stats] if test_fragment_stats else [],
        "overlap_scores": overlap_scores,
        "base_case_chars": [b["chars"] for b in base_case_stats],
    }


# ---------------------------------------------------------------------------
# Comparison & Visualization
# ---------------------------------------------------------------------------
def plot_comparison(t1_2025: dict, t1_2026: dict, t2_2025: dict, t2_2026: dict):
    """Generate comparison plots between 2025 and 2026."""
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*70}")

    sns.set_style("whitegrid")
    colors_2025 = "#4C72B0"
    colors_2026 = "#DD8452"

    # --- Plot 1: Citation distribution comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins = np.arange(0, max(max(t1_2025["citation_counts"]), max(t1_2026["citation_counts"])) + 2) - 0.5
    ax.hist(t1_2025["citation_counts"], bins=bins, alpha=0.6, label="2025", color=colors_2025, density=True)
    ax.hist(t1_2026["citation_counts"], bins=bins, alpha=0.6, label="2026", color=colors_2026, density=True)
    ax.set_xlabel("Citations per Query")
    ax.set_ylabel("Density")
    ax.set_title("Task 1: Citation Distribution")
    ax.legend()
    ax.set_xlim(-0.5, 20.5)

    ax = axes[1]
    bins_w = np.linspace(0, max(max(t1_2025["words"]), max(t1_2026["words"])), 40)
    ax.hist(t1_2025["words"], bins=bins_w, alpha=0.6, label="2025", color=colors_2025, density=True)
    ax.hist(t1_2026["words"], bins=bins_w, alpha=0.6, label="2026", color=colors_2026, density=True)
    ax.set_xlabel("Words per Document")
    ax.set_ylabel("Density")
    ax.set_title("Task 1: Document Length Distribution")
    ax.legend()

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "20_2025v2026_task1_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved: 20_2025v2026_task1_distributions.png")

    # --- Plot 2: Task 2 comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Entailing paragraph counts
    ax = axes[0]
    for year, data, color in [("2025", t2_2025, colors_2025), ("2026", t2_2026, colors_2026)]:
        if data["entail_counts"]:
            ec = np.array(data["entail_counts"])
            counts = [np.mean(ec == i) * 100 for i in [1, 2, 3]] + [np.mean(ec >= 4) * 100]
            ax.bar(np.arange(4) + (0.2 if year == "2026" else -0.2), counts,
                   width=0.35, label=year, color=color, alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["1", "2", "3", "4+"])
    ax.set_xlabel("Entailing Paragraphs")
    ax.set_ylabel("Percentage")
    ax.set_title("Task 2: Entailing Paragraphs per Case")
    ax.legend()

    # Fragment length comparison
    ax = axes[1]
    box_data = []
    labels_box = []
    for year, data in [("2025", t2_2025), ("2026", t2_2026)]:
        if data["fragment_words_train"]:
            box_data.append(data["fragment_words_train"])
            labels_box.append(f"{year}\ntrain")
        if data["fragment_words_test"]:
            box_data.append(data["fragment_words_test"])
            labels_box.append(f"{year}\ntest")

    if box_data:
        bp = ax.boxplot(box_data, labels=labels_box, patch_artist=True)
        palette = [colors_2025, colors_2025, colors_2026, colors_2026]
        for patch, color in zip(bp["boxes"], palette[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Words")
    ax.set_title("Task 2: Entailed Fragment Length")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "21_2025v2026_task2_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved: 21_2025v2026_task2_distributions.png")

    # --- Plot 3: Summary dashboard ---
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Dataset size comparison (bar chart)
    ax = fig.add_subplot(gs[0, 0])
    categories = ["T1 Train", "T1 Test", "T1 Queries", "T2 Train", "T2 Test"]
    vals_2025 = [t1_2025["n_train"], t1_2025["n_test"], t1_2025["n_queries"],
                 t2_2025["n_train"], t2_2025["n_test"]]
    vals_2026 = [t1_2026["n_train"], t1_2026["n_test"], t1_2026["n_queries"],
                 t2_2026["n_train"], t2_2026["n_test"]]
    x = np.arange(len(categories))
    ax.bar(x - 0.2, vals_2025, 0.35, label="2025", color=colors_2025, alpha=0.8)
    ax.bar(x + 0.2, vals_2026, 0.35, label="2026", color=colors_2026, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_title("Dataset Sizes")
    ax.legend(fontsize=8)

    # Avg citations per query
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(["2025", "2026"],
           [t1_2025["avg_citations"], t1_2026["avg_citations"]],
           color=[colors_2025, colors_2026], alpha=0.8)
    ax.set_title("Avg Citations/Query")
    ax.set_ylabel("Count")

    # Document structure comparison
    ax = fig.add_subplot(gs[0, 2])
    struct_labels = ["FRAG_SUP", "Background", "Issues", "Analysis", "Conclusion"]
    s25 = [t1_2025["struct_fragment"], t1_2025["struct_background"],
           t1_2025["struct_issues"], t1_2025["struct_analysis"], t1_2025["struct_conclusion"]]
    s26 = [t1_2026["struct_fragment"], t1_2026["struct_background"],
           t1_2026["struct_issues"], t1_2026["struct_analysis"], t1_2026["struct_conclusion"]]
    x = np.arange(len(struct_labels))
    ax.bar(x - 0.2, s25, 0.35, label="2025", color=colors_2025, alpha=0.8)
    ax.bar(x + 0.2, s26, 0.35, label="2026", color=colors_2026, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(struct_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Percentage")
    ax.set_title("Document Structure")
    ax.legend(fontsize=8)

    # Word count distribution comparison
    ax = fig.add_subplot(gs[1, 0])
    word_bins = ["<1k", "1k-5k", "5k-10k", "10k-20k", ">20k"]
    for year, data, color in [("2025", t1_2025, colors_2025), ("2026", t1_2026, colors_2026)]:
        wa = np.array(data["words"])
        pcts = [
            np.mean(wa < 1000) * 100,
            np.mean((wa >= 1000) & (wa < 5000)) * 100,
            np.mean((wa >= 5000) & (wa < 10000)) * 100,
            np.mean((wa >= 10000) & (wa < 20000)) * 100,
            np.mean(wa >= 20000) * 100,
        ]
        offset = -0.2 if year == "2025" else 0.2
        ax.bar(np.arange(len(word_bins)) + offset, pcts, 0.35, label=year, color=color, alpha=0.8)
    ax.set_xticks(range(len(word_bins)))
    ax.set_xticklabels(word_bins, fontsize=8)
    ax.set_ylabel("Percentage")
    ax.set_title("Word Count Distribution")
    ax.legend(fontsize=8)

    # Candidate paragraphs (Task 2)
    ax = fig.add_subplot(gs[1, 1])
    for year, data, color in [("2025", t2_2025, colors_2025), ("2026", t2_2026, colors_2026)]:
        if data["para_counts"]:
            ax.hist(data["para_counts"], bins=30, alpha=0.5, label=year, color=color, density=True)
    ax.set_xlabel("Paragraphs per Case")
    ax.set_ylabel("Density")
    ax.set_title("Task 2: Candidate Paragraphs")
    ax.legend(fontsize=8)

    # Lexical overlap (Task 2)
    ax = fig.add_subplot(gs[1, 2])
    for year, data, color in [("2025", t2_2025, colors_2025), ("2026", t2_2026, colors_2026)]:
        if data["overlap_scores"]:
            ax.hist(data["overlap_scores"], bins=30, alpha=0.5, label=year, color=color, density=True)
    ax.set_xlabel("Jaccard Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Task 2: Lexical Overlap")
    ax.legend(fontsize=8)

    fig.suptitle("COLIEE 2025 vs 2026: Dataset Comparison", fontsize=14, fontweight="bold")
    fig.savefig(PLOTS_DIR / "22_2025v2026_summary_dashboard.png", dpi=150)
    plt.close(fig)
    print("  Saved: 22_2025v2026_summary_dashboard.png")


def print_comparison_table(t1_2025, t1_2026, t2_2025, t2_2026,
                           noise_2025, noise_2026):
    """Print a structured comparison table."""
    print(f"\n{'='*70}")
    print("2025 vs 2026 COMPARISON SUMMARY")
    print(f"{'='*70}")

    print("\n--- TASK 1: Legal Case Retrieval ---")
    print(f"{'Metric':<35} {'2025':>10} {'2026':>10} {'Change':>10}")
    print("-" * 70)
    rows = [
        ("Corpus (train)", t1_2025["n_train"], t1_2026["n_train"]),
        ("Corpus (test)", t1_2025["n_test"], t1_2026["n_test"]),
        ("Queries with labels", t1_2025["n_queries"], t1_2026["n_queries"]),
        ("Total citations", t1_2025["total_citations"], t1_2026["total_citations"]),
        ("Avg citations/query", t1_2025["avg_citations"], t1_2026["avg_citations"]),
        ("Mean doc words", np.mean(t1_2025["words"]), np.mean(t1_2026["words"])),
        ("Median doc words", np.median(t1_2025["words"]), np.median(t1_2026["words"])),
        ("% single-citation queries", t1_2025["pct_1_cite"], t1_2026["pct_1_cite"]),
    ]
    for label, v25, v26 in rows:
        if isinstance(v25, float):
            change = f"{(v26 - v25) / v25 * 100:+.1f}%" if v25 != 0 else "N/A"
            print(f"{label:<35} {v25:>10.1f} {v26:>10.1f} {change:>10}")
        else:
            change = f"{(v26 - v25) / v25 * 100:+.1f}%" if v25 != 0 else "N/A"
            print(f"{label:<35} {v25:>10,} {v26:>10,} {change:>10}")

    # Noise comparison
    print(f"\n--- TASK 1: Noise Patterns (Train) ---")
    print(f"{'Pattern':<35} {'2025':>10} {'2026':>10}")
    print("-" * 60)
    if noise_2025 and noise_2026:
        train_25 = noise_2025["train"]
        train_26 = noise_2026["train"]
        patterns = [
            ("FRAGMENT_SUPPRESSED", "has_fragment"),
            ("Section headers", "has_headers"),
            ("Preamble before [1]", "has_preamble"),
            ("[End of document]", "has_end_of_doc"),
            ("Judge signature", "judge_sig"),
            ("French text", "has_french"),
            ("Broken statute names", "broken_statute"),
        ]
        for label, key in patterns:
            v25 = np.mean([d[key] for d in train_25]) * 100
            v26 = np.mean([d[key] for d in train_26]) * 100
            print(f"{label:<35} {v25:>9.1f}% {v26:>9.1f}%")

        # Noise ratio
        nr25 = np.mean([d["noise_ratio"] for d in train_25]) * 100
        nr26 = np.mean([d["noise_ratio"] for d in train_26]) * 100
        print(f"{'Mean noise ratio':<35} {nr25:>9.1f}% {nr26:>9.1f}%")

    print(f"\n--- TASK 2: Legal Case Entailment ---")
    print(f"{'Metric':<35} {'2025':>10} {'2026':>10} {'Change':>10}")
    print("-" * 70)
    t2_rows = [
        ("Train cases", t2_2025["n_train"], t2_2026["n_train"]),
        ("Test cases", t2_2025["n_test"], t2_2026["n_test"]),
    ]
    for label, v25, v26 in t2_rows:
        change = f"{(v26 - v25) / v25 * 100:+.1f}%" if v25 != 0 else "N/A"
        print(f"{label:<35} {v25:>10,} {v26:>10,} {change:>10}")

    if t2_2025["fragment_words_train"] and t2_2026["fragment_words_train"]:
        fw25 = np.mean(t2_2025["fragment_words_train"])
        fw26 = np.mean(t2_2026["fragment_words_train"])
        change = f"{(fw26 - fw25) / fw25 * 100:+.1f}%"
        print(f"{'Mean fragment words (train)':<35} {fw25:>10.1f} {fw26:>10.1f} {change:>10}")

    if t2_2025["fragment_words_test"] and t2_2026["fragment_words_test"]:
        fw25 = np.mean(t2_2025["fragment_words_test"])
        fw26 = np.mean(t2_2026["fragment_words_test"])
        change = f"{(fw26 - fw25) / fw25 * 100:+.1f}%"
        print(f"{'Mean fragment words (test)':<35} {fw25:>10.1f} {fw26:>10.1f} {change:>10}")

    if t2_2025["para_counts"] and t2_2026["para_counts"]:
        pc25 = np.mean(t2_2025["para_counts"])
        pc26 = np.mean(t2_2026["para_counts"])
        change = f"{(pc26 - pc25) / pc25 * 100:+.1f}%"
        print(f"{'Mean candidate paragraphs':<35} {pc25:>10.1f} {pc26:>10.1f} {change:>10}")

    if t2_2025["overlap_scores"] and t2_2026["overlap_scores"]:
        ov25 = np.mean(t2_2025["overlap_scores"])
        ov26 = np.mean(t2_2026["overlap_scores"])
        change = f"{(ov26 - ov25) / ov25 * 100:+.1f}%"
        print(f"{'Mean Jaccard overlap':<35} {ov25:>10.3f} {ov26:>10.3f} {change:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    # Task 1 EDA
    t1_2025 = task1_eda("2025", T1_TRAIN_DIR_2025, T1_TEST_DIR_2025, T1_TRAIN_LABELS_2025)
    t1_2026 = task1_eda("2026", T1_TRAIN_DIR_2026, T1_TEST_DIR_2026, T1_TRAIN_LABELS_2026)

    # Task 1 Noise
    noise_2025 = task1_noise("2025", T1_TRAIN_DIR_2025, T1_TEST_DIR_2025)
    noise_2026 = task1_noise("2026", T1_TRAIN_DIR_2026, T1_TEST_DIR_2026)

    # Task 2 EDA
    t2_2025 = task2_eda("2025", T2_TRAIN_DIR_2025, T2_TEST_DIR_2025, T2_TRAIN_LABELS_2025)
    t2_2026 = task2_eda("2026", T2_TRAIN_DIR_2026, T2_TEST_DIR_2026, T2_TRAIN_LABELS_2026)

    # Comparison
    print_comparison_table(t1_2025, t1_2026, t2_2025, t2_2026, noise_2025, noise_2026)
    plot_comparison(t1_2025, t1_2026, t2_2025, t2_2026)

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
