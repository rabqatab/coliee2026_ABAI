# COLIEE 2026 Corpus Noise Analysis

Statistical analysis of noise patterns across both Task 1 and Task 2 corpora.
Based on a sample of 500 train + 200 test files (Task 1) and 400 train + 100 test cases (Task 2), sampled with `random.seed(42)`.

---

## Task 1: Legal Case Retrieval

### 1. Document Size

Documents are highly variable in length -- median is ~190 lines but can exceed 2,300.

|           | Lines     | Characters  |
|-----------|-----------|-------------|
| Min       | 19        | 2,739       |
| P25       | 111       | 13,759      |
| **Median**| **194**   | **22,906**  |
| Mean      | 258.7     | 30,926      |
| P75       | 312       | 37,004      |
| Max       | 2,309     | 305,873     |
| Std Dev   | 245.2     | 30,514      |

Train and test distributions are similar (test is slightly smaller: median 151 lines, 21K chars).

---

### 2. Noise Pattern Prevalence

| Pattern | Train (n=500) | Test (n=200) | Combined |
|---------|---------------|--------------|----------|
| Paragraph markers `[N]` | 100% | 100% | **100%** |
| `<FRAGMENT_SUPPRESSED>` | 91.8% | 96.5% | **93.1%** |
| Section headers | 84.6% | 84.5% | **84.6%** |
| Judge signature lines | 84.0% | 84.5% | **84.1%** |
| `[End of document]` | 84.6% | 90.0% | **86.1%** |
| Judgment outcome lines | 81.4% | 79.0% | **80.7%** |
| French text present | 73.6% | 75.5% | **74.1%** |
| Preamble before `[1]` | 69.2% | 56.0% | **65.4%** |
| `Editor:` line | 54.8% | 62.5% | **57.0%** |
| Broken statute names | 35.6% | 42.5% | **37.6%** |

---

### 3. Preamble / Metadata (before first `[1]`)

**65% of files** have content before the first paragraph marker. When present, the preamble averages **15.5 non-blank lines** (train) / **9.7 lines** (test), reaching up to 79 lines.

Most common preamble elements (among files that have a preamble):

| Pattern | Frequency |
|---------|-----------|
| `Counsel:` / `Solicitors of Record:` | ~81% |
| `Summary:` | ~80% |
| `Federal Court` header | ~79% |
| Date lines | ~77% |
| `<FRAGMENT_SUPPRESSED>` in preamble | ~75% |
| Judge name line | ~71% |
| `MLB headnote/unedited` | ~30% |
| Case citation header `(FC)` / `(FCA)` | ~24% |
| `refd to.` references | ~12% |
| Topic classifications (`Aliens - Topic`, `Administrative Law - Topic`, etc.) | ~10% |

**Key observation**: Test files have a simpler preamble structure (no Topic classifications appeared in test sample, fewer metadata patterns). Train preambles are noisier.

---

### 4. `<FRAGMENT_SUPPRESSED>` Markers

These represent **redacted citations** -- the core of what systems must predict.

| Metric | Train | Test |
|--------|-------|------|
| Files with markers | 91.8% | 96.5% |
| Mean count per file | 26.8 | 18.3 |
| Median count | 13 | 12 |
| Max count | 229 | 163 |

**Positioning** (where do fragments appear?):

| Location | Train | Test |
|----------|-------|------|
| In preamble only | 1.1% | 1.6% |
| In body only | 43.8% | 61.1% |
| In both preamble + body | 55.1% | 37.3% |
| Overall: preamble vs body ratio | 10% / 90% | 4% / 96% |

**~90% of fragment markers are in the document body**, not the preamble. This means removing preambles does NOT eliminate most citation holes.

**5% of train files** (0% of test) have >50% of non-blank lines containing `<FRAGMENT_SUPPRESSED>` -- these are extremely noisy documents.

---

### 5. Section Headers

**84.6% of files** contain section headers. The top headers found (across 700 files):

| Header | Count | Header | Count |
|--------|-------|--------|-------|
| JUDGMENT / JUGEMENT | 404 | Introduction / INTRODUCTION | 122 |
| Analysis / ANALYSIS | 322 | Standard of Review / STANDARD OF REVIEW | 141 |
| Background / BACKGROUND | 223 | DECISION UNDER REVIEW | 48 |
| Conclusion / CONCLUSION | 191 | STATUTORY PROVISIONS | 18 |
| Issues / ISSUES / Issue | 264 | Facts / FACTS | 99 |
| ORDER | 124 | Discussion / DISCUSSION | 8+ |

**Total unique headers found**: 369 (train) + 219 (test).

Note: Many "headers" are actually **statute abbreviations** appearing on standalone lines (e.g., `IRPA`, `PSLRA`, `SARA`, `LAA`, `CCRA`). These are very common (IRPA alone appears 651 times). These are statute name abbreviations defined earlier in the document, not true section headers.

---

### 6. French Text

**74% of files** contain French text, but it's a small fraction of content:

- **Overall French line ratio**: ~5%
- Among files with French: mean 15 lines, median 5-6 lines, max 170 lines
- Typically appears in bilingual statutory provisions (English followed by French translation of the same section)

---

### 7. Trailing Boilerplate

| Pattern | Train | Test |
|---------|-------|------|
| `[End of document]` | 84.6% | 90.0% |
| Judge signature (`Name, J.`) | 84.0% | 84.5% |
| Judgment outcome lines | 81.4% | 79.0% |
| `Editor:` line | 54.8% | 62.5% |

Judgment outcome breakdown (train):

| Outcome | Count |
|---------|-------|
| Application dismissed | 149 |
| JR Application dismissed | 129 |
| Application allowed | 108 |
| JR Application allowed | 78 |
| Appeal dismissed | 23 |
| Appeal allowed | 20 |

---

### 8. Broken Statute Names

**37.6% of files** have statute names split across lines. Pattern: a statute name on one line, followed by `, R.S.C. 1985, c. ...` or `, S.C. 2001, c. ...` on the next line.

Examples:
```
"Immigration and Refugee Protection Act"
", S.C. 2001, c. 27 ("

"Federal Court Act"
", R.S.C. 1985, c. F-7."

"Criminal Code"
", R.S.C. 1985, c. C-46, fraud over $5,000..."
```

These break tokenization and BM25 term matching for statute references.

---

### 9. Whitespace Patterns

- **Train**: Mean 8.2% blank lines, 11.6% of files have >20% blank lines, max 73.5%
- **Test**: 0% blank lines (test files have been cleaned of blank lines)

This is a notable **train/test asymmetry**: test files have no blank lines while train files can be very whitespace-heavy.

---

### 10. Overall Noise Ratio

Estimated percentage of each file that is "noise" (blank lines, preamble, fragment-only lines):

| Metric | Train | Test |
|--------|-------|------|
| Mean | 16.1% | 5.6% |
| Median | 7.3% | 1.9% |
| P25 | 1.3% | 0.6% |
| P75 | 19.7% | 9.3% |
| Max | 79.9% | 28.8% |

**Test files are substantially cleaner than train files** -- less preamble, no blank lines, fewer fragment markers. This matters for preprocessing: a pipeline tuned for noisy train data may over-clean test data.

---

## Task 2: Legal Case Entailment

### 1. Base Case (`base_case.txt`)

Size statistics (similar to Task 1 individual documents):

|           | Lines (train) | Lines (test) | Chars (train) | Chars (test) |
|-----------|--------------|--------------|---------------|--------------|
| Min       | 32           | 60           | 1,850         | 4,310        |
| Median    | 203          | 184.5        | 21,178        | 20,720       |
| Mean      | 312.3        | 212.7        | 25,531        | 25,538       |
| Max       | 1,311        | 861          | 157,498       | 95,631       |

Noise patterns in base cases:

| Pattern | Train (n=400) | Test (n=100) |
|---------|---------------|--------------|
| `<FRAGMENT_SUPPRESSED>` | 96.8% | 95.0% |
| Section headers | 92.8% | 91.0% |
| Trailing boilerplate | 79.0% | 92.0% |
| Broken statute names | 70.5% | 66.0% |
| Preamble before `[1]` | 61.8% | 38.0% |
| French text (>5%) | 10.2% | 7.0% |

Note: `<FRAGMENT_SUPPRESSED>` counts are very low in base cases (mean ~1.1 per case, max 3-5). This is expected since base cases have most citations already removed -- only a few reference markers remain.

---

### 2. Entailed Fragment (`entailed_fragment.txt`)

These are **remarkably clean** -- 98% of fragments have zero noise.

| Metric | Train | Test |
|--------|-------|------|
| Mean length | 213 chars / 35 words | 377 chars / 63 words |
| Median length | 188 chars / 32 words | 364 chars / 63 words |
| Has `<FRAGMENT_SUPPRESSED>` | 0% | 0% |
| Has section headers | 0% | 0% |
| Has French text | 0% | 0% |
| Has paragraph numbers | 0.8% | 1.0% |
| Has broken statute names | 1.8% | 4.0% |
| **Clean (no noise)** | **98.2%** | **96.0%** |

**Important**: Test fragments are nearly **2x longer** than train fragments (63 vs 35 words median). This distribution shift could affect model performance.

---

### 3. Paragraphs (`paragraphs/`)

Paragraph count per case:

| Bucket | Train | Test |
|--------|-------|------|
| 6-10 | 5.8% | 2.0% |
| 11-20 | 22.0% | 29.0% |
| 21-50 | **56.2%** | **59.0%** |
| 51-100 | 13.5% | 7.0% |
| 100+ | 2.5% | 3.0% |

Individual paragraph statistics:

| Metric | Train | Test |
|--------|-------|------|
| Median length | 487 chars / 80 words | 515 chars / 86 words |
| Mean length | 634 chars / 105 words | 702 chars / 117 words |
| Max length | 21,321 chars / 3,503 words | 13,317 chars / 2,127 words |
| Empty paragraphs | 0% | 0% |
| Near-empty (<20 chars) | 0.02% | 0% |

Noise in paragraphs:

| Pattern | Train | Test |
|---------|-------|------|
| `<FRAGMENT_SUPPRESSED>` | **0%** | **0%** |
| Internal `[N]` numbering | 100% | 100% |
| Broken statute names | 5.2% | 4.3% |
| Section headers | 3.5% | 4.5% |
| French text | 1.5% | 1.8% |

**Key finding**: Paragraphs contain **zero** `<FRAGMENT_SUPPRESSED>` markers. This is significant -- the precedent case paragraphs are clean of redacted citations, unlike the base cases in Task 1.

All paragraphs have internal `[N]` numbering (e.g., `[24]`, `[25]`), which is the paragraph's own position marker from the original judgment.

---

### 4. Train vs Test Consistency

| Metric | Train (mean) | Test (mean) | Difference |
|--------|-------------|-------------|------------|
| Base case chars | 25,531 | 25,538 | ~0% |
| Paragraph count | 36.1 | 32.8 | -9% |
| Fragment length (words) | 35.4 | **62.5** | **+77%** |
| Noise score | 2.5 | 2.2 | -12% |
| Preamble prevalence | 61.8% | 38.0% | -38% |

The most notable **distribution shift** is the entailed fragment length: test fragments are substantially longer than train fragments. Models should be tested for robustness to this.

---

## Preprocessing Recommendations

### Priority 1: High Impact, Safe for All Models

| Step | Affects | Rationale |
|------|---------|-----------|
| **Remove preamble** (everything before first `[1]`) | 65% of Task 1 files, 62% of Task 2 base cases | Redundant metadata, counsel info, topic classifications. Avg 15 lines of noise. |
| **Remove trailing boilerplate** (`Editor:`, `[End of document]`, judge signature) | 84-90% of files | Zero semantic value for retrieval/entailment |
| **Strip `<FRAGMENT_SUPPRESSED>` markers** | 93% of Task 1 files | ~27 per file average. Replace with empty string or single space to avoid concatenation artifacts |
| **Normalize whitespace** | 12% of train have >20% blank lines | Collapse multiple blank lines, strip trailing whitespace. Note: test already has 0% blank lines |

### Priority 2: Moderate Impact

| Step | Affects | Rationale |
|------|---------|-----------|
| **Strip paragraph markers `[N]`** | 100% of files | Noise for BM25 term frequency and embedding models. However, they serve as structural anchors for paragraph-level chunking. Consider keeping as metadata. |
| **Rejoin broken statute names** | 38% of Task 1, 71% of Task 2 | Improves tokenization. Regex: detect line ending with statute name + next line starting with `, R.S.C.` / `, S.C.` |
| **Remove duplicate French statutory text** | 74% of files, ~5% of content | French translations of English provisions are noise for English-only retrieval. Detect paired French blocks following English ones. |

### Priority 3: Model-Dependent

| Step | When to Apply | Rationale |
|------|---------------|-----------|
| **Keep section headers** | Dense retrieval, paragraph chunking | Headers like "Analysis", "Issues", "Standard of Review" are natural boundaries for paragraph-level encoding. Remove only for pure BM25. |
| **Remove judgment outcome lines** | If they cause false positives | "Application dismissed" etc. could cause spurious matches between unrelated cases with same outcome. |
| **Statute abbreviation removal** (`IRPA`, `PSLRA`, etc.) | BM25 | Common abbreviations like IRPA appear 651 times and are poor discriminators. Keep for neural models that can learn to weight them. |

### Task-Specific Notes

**Task 1 (Retrieval)**:
- Train is noisier than test (16% vs 6% noise ratio). Don't over-clean.
- Test has zero blank lines while train can have up to 74% blank -- normalize to avoid train/test mismatch.
- Fragment markers are 90% in the body, not preamble. Removing preamble alone won't suffice.

**Task 2 (Entailment)**:
- `entailed_fragment.txt` is 98% clean -- **no preprocessing needed** for fragments.
- `paragraphs/` have **zero** `<FRAGMENT_SUPPRESSED>` markers -- they are clean of redacted citations.
- Main noise in paragraphs: internal `[N]` numbering (100%), broken statute names (5%), section headers (4%).
- **Distribution shift warning**: test entailed fragments are ~2x longer than train (63 vs 35 words). Validate model robustness.
- `base_case.txt` has the same noise as Task 1 files but fewer `<FRAGMENT_SUPPRESSED>` markers (mean ~1 vs ~27).

---

## Appendix: Analysis Scripts

The analysis was performed by two Python scripts:
- `analyze_noise.py` -- Task 1 analysis (500 train + 200 test files)
- `analyze_task2_noise.py` -- Task 2 analysis (400 train + 100 test cases)

Both use `random.seed(42)` for reproducibility.
