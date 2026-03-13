# COLIEE 2026 Data Structure & Corpus

## Directory Layout

```
data/
├── task1/
│   ├── task1_train_files_2026/          # 7,708 case law text files
│   │   ├── 000002.txt
│   │   ├── 000028.txt
│   │   └── ...
│   ├── task1_test_files_2026/           # 1,848 case law text files
│   │   ├── 000055.txt
│   │   ├── 000091.txt
│   │   └── ...
│   ├── task1_train_labels_2026.json     # Gold labels: query -> noticed cases
│   ├── task1_test_no_labels_2026.json   # Test queries with empty lists (submission template)
│   │
│   ├── task1_train_files_2025/          # Previous year (7,350 files)
│   ├── task1_test_files_2025/           # Previous year (2,159 files)
│   ├── task1_train_labels_2025.json
│   ├── task1_test_labels_2025.json      # 2025 test gold labels (available for validation)
│   └── task1_test_no_labels_2025.json
│
└── task2/
    ├── task2_train_files_2026/          # 925 case folders (001–925)
    │   ├── 001/
    │   │   ├── base_case.txt
    │   │   ├── entailed_fragment.txt
    │   │   └── paragraphs/
    │   │       ├── 001.txt
    │   │       └── ...
    │   └── ...
    ├── task2_test_files_2026/           # 100 case folders (926–1025)
    │   ├── 926/
    │   │   ├── base_case.txt
    │   │   ├── entailed_fragment.txt
    │   │   └── paragraphs/
    │   └── ...
    ├── task2_train_labels_2026.json     # Gold labels: case ID -> paragraph list
    │
    ├── task2_train_files_2025/          # Previous year (825 cases)
    ├── task2_test_files_2025/           # Previous year (100 cases)
    ├── task2_train_labels_2025.json
    └── task2_test_labels_2025.json      # 2025 test gold labels
```

---

## Dataset Statistics

### 2026 vs 2025 Comparison

| Metric | 2025 | 2026 | Change |
|--------|------|------|--------|
| **Task 1 train files** | 7,350 | 7,708 | +4.9% |
| **Task 1 test files** | 2,159 | 1,848 | -14.4% |
| **Task 1 train queries** | 1,678 | 2,001 | +19.3% |
| **Task 1 total positives** | — | 8,251 | — |
| **Task 1 avg positives/query** | — | 4.1 | — |
| **Task 1 test queries** | — | 400 | — |
| **Task 2 train cases** | 825 | 925 | +12.1% |
| **Task 2 test cases** | 100 | 100 | same |
| **Task 2 test ID range** | 826–925 | 926–1025 | — |

---

## Task 1: Legal Case Retrieval

### Corpus Description

Federal Court of Canada case law documents stored as plain text files. Both query cases and candidate noticed cases live in the same flat directory. References within query cases have been intentionally removed (replaced with `<FRAGMENT_SUPPRESSED>` markers) so systems cannot simply pattern-match citations.

### File Format

Each `.txt` file is a single legal case document containing numbered paragraphs (e.g., `[1]`, `[2]`, ...) with the full judgment text. Files are named with zero-padded 6-digit IDs (e.g., `000002.txt`).

**Example content** (`000002.txt`):
```
[1]
Bédard, J.
[Translation]: This is an application for judicial review under subsection 72(1) of the
Immigration and Refugee Protection Act ...
[2]
The applicant is a Canadian citizen from Cameroon. On October 16, 2006, ...
```

### Label Files

#### `task1_train_labels_2026.json` — Training gold labels

Maps each query case filename to a list of noticed case filenames:

```json
{
    "008447.txt": ["072495.txt", "082291.txt", "004851.txt", "049315.txt"],
    "067501.txt": ["038025.txt", "072553.txt"],
    "007627.txt": ["003575.txt", "043211.txt"]
}
```

- 2,001 queries, 8,251 total positive pairs (avg 4.1 per query)

#### `task1_test_no_labels_2026.json` — Submission template

Same structure but with empty lists. Participants must populate the noticed case lists:

```json
{
    "078507.txt": [],
    "023478.txt": [],
    "067520.txt": []
}
```

- 400 test queries

### Key Characteristics (from EDA)

- 93.1% of documents contain `<FRAGMENT_SUPPRESSED>` markers (citation removal points)
- Median document length: ~3,599 words (~4,700 tokens)
- 0% of documents fit within 512 tokens — truncation strategy is critical
- Train and test documents share the same pool — a training query may cite a test document

---

## Task 2: Legal Case Entailment

### Corpus Description

Each case is organized as a folder containing three components: the base case (query), the entailed fragment (the decision to be supported), and the paragraphs from the noticed (precedent) case.

### File Structure Per Case

```
{case_id}/
├── base_case.txt           # Full text of the query case (citations removed)
├── entailed_fragment.txt   # The specific decision fragment to entail
└── paragraphs/             # Paragraphs from the precedent/noticed case
    ├── 001.txt
    ├── 002.txt
    └── {N}.txt             # Variable count per case
```

### Label Files

#### `task2_train_labels_2026.json` — Training gold labels

Maps each case ID (string) to a list of paragraph filenames:

```json
{
    "001": ["027.txt"],
    "002": ["014.txt"],
    "003": ["003.txt", "004.txt"]
}
```

- 925 training cases

#### Test submission

No test labels file for 2026 (labels withheld). 100 test cases (926–1025).

---

## Key Differences Between Tasks

| Aspect | Task 1 (Retrieval) | Task 2 (Entailment) |
|--------|-------------------|---------------------|
| Goal | Find which cases are noticed by a query case | Find which paragraphs support a decision |
| Input | Flat text files of case law | Structured folders with base case, fragment, and paragraphs |
| Granularity | Document-level | Paragraph-level |
| Corpus | Federal Court of Canada | Federal Court of Canada |
| Train Size | 7,708 files, 2,001 queries | 925 cases |
| Test Size | 1,848 files, 400 queries | 100 cases (926–1025) |
| Label Format | `{"query.txt": ["noticed1.txt", ...]}` | `{"case_id": ["para.txt", ...]}` |
| Evaluation | Micro-averaged F1 | Micro-averaged F1 |

---

## Important Notes

1. **All case files are plain text** with no special markup beyond paragraph numbering (`[1]`, `[2]`, etc.)
2. **Citations have been removed** from query cases and replaced with `<FRAGMENT_SUPPRESSED>` markers
3. **Paragraph counts vary** significantly between cases in Task 2
4. **All files share a common pool** in Task 1: train and test files may reference each other
5. **2025 data is also available** in the same directory structure — can be used for additional training or validation
6. **Task 2 test label format caveat (2025):** The 2025 test labels use comma-separated strings instead of arrays (e.g., `"827": "025.txt, 027.txt"`) — handle both formats when parsing
