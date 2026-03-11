# COLIEE 2026 Data Structure & Corpus Instructions

## Directory Layout

```
data/
├── task1/
│   ├── task1_train_files_2025/          # 7,350 case law text files
│   │   ├── 000002.txt
│   │   ├── 000028.txt
│   │   └── ...
│   ├── task1_test_files_2025/           # 2,159 case law text files
│   │   ├── 000055.txt
│   │   ├── 000091.txt
│   │   └── ...
│   ├── task1_train_labels_2025.json     # Gold labels: query -> noticed cases
│   ├── task1_test_labels_2025.json      # Gold labels (for evaluation after submission)
│   └── task1_test_no_labels_2025.json   # Test queries with empty lists (submission template)
│
└── task2/
    ├── task2_train_files_2025/          # 825 case folders
    │   ├── 001/
    │   │   ├── base_case.txt            # The query case (with citations removed)
    │   │   ├── entailed_fragment.txt     # The decision fragment to be entailed
    │   │   └── paragraphs/              # Paragraphs from the noticed (precedent) case
    │   │       ├── 001.txt
    │   │       ├── 002.txt
    │   │       └── ... (variable count per case)
    │   ├── 002/
    │   └── ...
    ├── task2_test_files_2025/           # 100 case folders (826-925)
    │   ├── 826/
    │   │   ├── base_case.txt
    │   │   ├── entailed_fragment.txt
    │   │   └── paragraphs/
    │   └── ...
    ├── task2_train_labels_2025.json     # Gold labels: case ID -> paragraph list
    └── task2_test_labels_2025.json      # Gold labels (for evaluation after submission)
```

---

## Task 1: Legal Case Retrieval

### Corpus Description

The corpus consists of Federal Court of Canada case law documents stored as plain text files. Both query cases and candidate noticed cases live in the same flat directory. References within query cases have been intentionally removed so systems cannot simply pattern-match citations.

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

#### `task1_train_labels_2025.json` -- Training gold labels

Maps each query case filename to a list of noticed case filenames:

```json
{
    "008447.txt": ["072495.txt", "082291.txt", "004851.txt", "049315.txt"],
    "067501.txt": ["038025.txt", "072553.txt"],
    "007627.txt": ["003575.txt", "043211.txt"]
}
```

#### `task1_test_no_labels_2025.json` -- Submission template

Same structure but with empty lists. Participants must populate the noticed case lists:

```json
{
    "078507.txt": [],
    "023478.txt": [],
    "067520.txt": []
}
```

#### `task1_test_labels_2025.json` -- Test gold labels (for evaluation)

Same format as training labels, with the correct answers:

```json
{
    "078507.txt": ["081644.txt", "044740.txt", "024615.txt"],
    "023478.txt": ["022954.txt"]
}
```

### Dataset Statistics

| Split | Case Files | Query Cases |
|-------|-----------|-------------|
| Train | 7,350     | Keys in `task1_train_labels_2025.json` |
| Test  | 2,159     | Keys in `task1_test_no_labels_2025.json` |

### Submission Format

Populate the `task1_test_no_labels_2025.json` template by filling in predicted noticed case filenames for each query.

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
    ├── ...
    └── {N}.txt             # Variable count (e.g., 32 for case 001)
```

#### `base_case.txt`

The full text of the query case with certain citation fragments suppressed. This is the case whose decision needs to be supported by precedent.

#### `entailed_fragment.txt`

A specific decision fragment from the query case. This represents "a decision for a part of the case" -- NOT the final verdict, but a specific judicial conclusion that should be logically supported by paragraphs from the precedent case.

**Example** (case 001):
```
Given that the Respondent remains a security risk whom the Minister has not had
sufficient opportunity to investigate to determine the extent of his involvement
with the LTTE, his release into the community will cause real and non-speculative
irreparable harm. ...
```

#### `paragraphs/`

Individual paragraph files from the noticed (precedent) case, numbered sequentially (`001.txt`, `002.txt`, ...). The number of paragraphs varies per case.

### Label Files

#### `task2_train_labels_2025.json` -- Training gold labels

Maps each case ID (string) to a list of paragraph filenames that entail the decision fragment:

```json
{
    "001": ["027.txt"],
    "002": ["014.txt"],
    "003": ["003.txt", "004.txt"],
    "004": ["030.txt"],
    "005": ["011.txt"]
}
```

A case may have one or multiple supporting paragraphs.

#### `task2_test_labels_2025.json` -- Test gold labels (for evaluation)

Maps case ID to paragraph filenames (comma-separated string format):

```json
{
    "826": "009.txt",
    "827": "025.txt, 027.txt",
    "833": "018.txt, 020.txt, 021.txt, 022.txt"
}
```

**Note**: The test labels file uses a slightly different format -- values are comma-separated strings rather than arrays. Be aware of this when parsing.

### Dataset Statistics

| Split | Cases | Case ID Range |
|-------|-------|---------------|
| Train | 825   | 001 - 825     |
| Test  | 100   | 826 - 925     |

### Submission Format

For each test case, predict which paragraph(s) from the `paragraphs/` directory entail the `entailed_fragment.txt`.

---

## Key Differences Between Tasks

| Aspect | Task 1 (Retrieval) | Task 2 (Entailment) |
|--------|-------------------|---------------------|
| Goal | Find which cases are noticed by a query case | Find which paragraphs support a decision |
| Input | Flat text files of case law | Structured folders with base case, fragment, and paragraphs |
| Granularity | Document-level | Paragraph-level |
| Corpus | Federal Court of Canada | Federal Court of Canada |
| Train Size | 7,350 files | 825 cases |
| Test Size | 2,159 files | 100 cases |
| Label Format | `{"query.txt": ["noticed1.txt", ...]}` | `{"case_id": ["para.txt", ...]}` |

---

## Important Notes

1. **All case files are plain text** with no special markup beyond paragraph numbering (`[1]`, `[2]`, etc.)
2. **Citations have been removed** from query cases to prevent trivial lookups
3. **Paragraph counts vary** significantly between cases in Task 2 (from ~30 to 200+)
4. **Test label format inconsistency** in Task 2: training labels use JSON arrays, test labels use comma-separated strings -- handle both formats in parsing code
5. **All files share a common pool** in Task 1: train and test files may reference each other
