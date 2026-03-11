# COLIEE 2025 vs 2026 Dataset Comparison

Comprehensive comparison of the 2025 and 2026 competition datasets across EDA, corpus noise patterns, and label signal validation.

---

## Executive Summary

The 2026 dataset is a **modest expansion** of 2025 with nearly identical characteristics. The lexical signal is **equally strong** (LightGBM AUC: 0.932 → 0.936), meaning the 2025 pipeline strategy transfers directly. Key changes: +19% more labeled queries, +5% more corpus documents, and slightly improved feature separability.

---

## Task 1: Legal Case Retrieval

### Dataset Size

| Metric | 2025 | 2026 | Change |
|--------|------|------|--------|
| Corpus (train) | 7,350 | 7,708 | +4.9% |
| Corpus (test) | 2,159 | 1,848 | -14.4% |
| Queries with labels | 1,678 | 2,001 | +19.2% |
| Total citations | 6,881 | 8,251 | +19.9% |
| Avg citations/query | 4.10 | 4.12 | +0.6% |

**Interpretation:** The corpus grew modestly (+358 documents), but labeled queries increased significantly (+323 queries). The test set shrank by 311 cases. The average citation density is virtually unchanged.

### Citation Distribution

| Bucket | 2025 | 2026 |
|--------|------|------|
| 1 citation | 21.0% | 19.8% |
| 1-3 citations | 57.6% | 56.5% |
| 4-10 citations | 36.5% | 37.9% |
| >10 citations | 5.9% | 5.6% |

Distribution is nearly identical — still a long-tail with the majority having 1-3 citations.

### Document Statistics (n=500 sample)

| Metric | 2025 | 2026 |
|--------|------|------|
| Mean words | 5,006 | 4,834 |
| Median words | 3,680 | 3,516 |
| Mean characters | 30,926 | 29,852 |
| Mean paragraphs | 51 | 49 |

Documents are slightly shorter on average (-3.4% words), but the distribution shape is unchanged.

### Document Structure (n=300)

| Pattern | 2025 | 2026 |
|---------|------|------|
| `<FRAGMENT_SUPPRESSED>` | 91% | 89% |
| Background/Facts | 87% | 87% |
| Issues | 96% | 95% |
| Analysis | 78% | 78% |
| Conclusion/Order | 97% | 99% |

Virtually identical structural patterns.

### Case Types (n=300)

| Type | 2025 | 2026 |
|------|------|------|
| Judicial Review | 84% | 79% |
| IP | 7% | 7% |
| Appeal | 5% | 5% |
| Immigration | 4% | 6% |

Slight decrease in Judicial Review proportion (-5pp), slight increase in Immigration (+2pp).

---

## Task 1: Corpus Noise Patterns

### Noise Prevalence (Train)

| Pattern | 2025 | 2026 |
|---------|------|------|
| `<FRAGMENT_SUPPRESSED>` | 90.0% | 89.6% |
| Section headers | 75.0% | 75.2% |
| Judge signature | 92.8% | 92.6% |
| `[End of document]` | 84.2% | 82.2% |
| Judgment outcome | 63.4% | 64.0% |
| French text | 35.4% | 36.4% |
| Preamble before `[1]` | 25.2% | 23.2% |
| Broken statute names | 37.2% | 37.4% |
| **Mean noise ratio** | **15.0%** | **13.3%** |

### Noise Prevalence (Test)

| Pattern | 2025 | 2026 |
|---------|------|------|
| `<FRAGMENT_SUPPRESSED>` | 98.5% | 97.5% |
| Preamble before `[1]` | 0.0% | 0.5% |
| Blank line ratio | 0.0% | 0.0% |
| **Mean noise ratio** | **0.4%** | **0.3%** |

**Key finding:** Noise patterns are nearly identical between years. The train/test asymmetry persists — test files remain much cleaner (0.3% noise) than train (13.3%). The same preprocessing pipeline from 2025 applies directly.

### Fragment Marker Statistics

| Metric | 2025 Train | 2026 Train |
|--------|-----------|-----------|
| Mean per file | 29.9 | 32.6 |
| Median per file | 13 | 15 |
| Max per file | 216 | 237 |

Slightly more fragment markers per document in 2026 — consistent with marginally longer citation lists.

---

## Task 2: Legal Case Entailment

### Dataset Size

| Metric | 2025 | 2026 | Change |
|--------|------|------|--------|
| Train cases | 825 | 925 | +12.1% |
| Test cases | 100 | 100 | 0% |

### Entailing Paragraphs per Case

| Count | 2025 | 2026 |
|-------|------|------|
| 1 | 82.7% | 78.8% |
| 2 | 14.3% | 17.0% |
| 3 | 2.2% | 2.8% |
| 4+ | 0.8% | 1.4% |

Slight shift toward multi-paragraph entailment in 2026: -3.9pp for single-paragraph, +2.7pp for 2 paragraphs.

### Fragment Length

| Metric | 2025 | 2026 | Change |
|--------|------|------|--------|
| Mean words (train) | 35.9 | 37.4 | +4.3% |
| Mean words (test) | 62.5 | 70.4 | +12.8% |
| Median words (train) | 32 | 32 | 0% |
| Median words (test) | 63 | 52 | -17.5% |

**The train/test fragment length gap persists and has widened slightly.** Test fragments are ~1.9x longer than train in 2026 (vs 1.7x in 2025). This distribution shift remains a concern.

### Candidate Paragraphs & Overlap

| Metric | 2025 | 2026 |
|--------|------|------|
| Mean candidates/case | 37.7 | 36.8 |
| Selection ratio | 3.2% | 3.5% |
| Mean Jaccard overlap | 0.175 | 0.170 |
| <20% overlap | 72% | 75% |

Needle-in-haystack difficulty is comparable. Lexical overlap is marginally lower — may be slightly harder for keyword-based approaches.

---

## Label Signal Validation

### Statistical Tests (Cohen's d — Effect Sizes)

| Feature | 2025 | 2026 | Change |
|---------|------|------|--------|
| TF-IDF cosine | 2.266 | 2.305 | +1.7% |
| Shared bigrams | 1.697 | 1.836 | **+8.2%** |
| Jaccard | 1.301 | 1.322 | +1.6% |
| BM25 score | 1.153 | 1.170 | +1.5% |
| Shared legal terms | 0.620 | 0.634 | +2.2% |
| Length ratio | 0.070 | 0.088 | +26.1% |

**All effect sizes improved slightly.** Shared bigrams showed the most improvement (+8.2%), suggesting citation relationships in 2026 may have somewhat stronger phrasal overlap.

### Classifier Performance

| Model | AUC 2025 | AUC 2026 | AP 2025 | AP 2026 |
|-------|----------|----------|---------|---------|
| Logistic Regression | 0.9147 | 0.9181 | 0.7021 | 0.7110 |
| Random Forest | 0.9276 | 0.9303 | 0.7389 | 0.7429 |
| **LightGBM** | **0.9321** | **0.9362** | **0.7504** | **0.7590** |

All models improved marginally on 2026 data. LightGBM remains the best performer.

### Per-Query Separability

| Metric | 2025 | 2026 |
|--------|------|------|
| Median per-query AUC | 0.988 | 0.989 |
| Queries with AUC > 0.95 | 65.4% | 68.7% |
| Queries with AUC < 0.6 | 30 | 31 |

The "easy" majority is slightly easier in 2026, and the "hard tail" is proportionally comparable (~30 hard queries in both years).

### Verdict

**Signal: STRONG** (AUC = 0.9362)
**Stability: CONSISTENT** (delta = +0.0041)

The 2025 pipeline strategy transfers directly to 2026.

---

## Implications for Competition Strategy

1. **Preprocessing:** Same 3-tier pipeline from 2025 applies unchanged. Train/test noise asymmetry persists.

2. **First-stage retrieval:** BM25 + TF-IDF cosine remain strong. No change needed.

3. **Hard cases:** ~31 queries (out of ~2,001) have per-query AUC < 0.6 — these need semantic reranking. The hard tail is proportionally smaller in 2026 (1.5% vs 1.8%).

4. **Task 2 caution:** Fragment length distribution shift (train→test) has grown. Models must handle longer fragments at test time. The slight decrease in lexical overlap (-2.5%) may require slightly stronger semantic features.

5. **More training data:** 19% more labeled queries gives more room for neural model fine-tuning.

---

## Generated Plots

| File | Description |
|------|-------------|
| `../plots/20_2025v2026_task1_distributions.png` | Citation and document length distributions |
| `../plots/21_2025v2026_task2_distributions.png` | Entailing paragraph counts and fragment lengths |
| `../plots/22_2025v2026_summary_dashboard.png` | 6-panel summary dashboard |
| `../plots/23_signal_2025v2026_distributions.png` | Feature violin plots (pos/neg × year) |
| `../plots/24_signal_2025v2026_roc_pr.png` | ROC and PR curves comparison |
| `../plots/25_signal_2025v2026_per_query_auc.png` | Per-query AUC histograms |
| `../plots/26_signal_2025v2026_effect_sizes.png` | Cohen's d comparison bar chart |

## Reproducibility

```bash
uv sync
uv run python src/analysis/eda_2026.py              # ~4s
uv run python src/analysis/signal_validation_2026.py  # ~108s
```

Random seed: 42 (deterministic results)

---

*Generated: 2026-03-10*
