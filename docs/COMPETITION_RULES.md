# COLIEE 2026 Competition Rules

> **Note:** Based on COLIEE 2024 rules. Official 2026 page not yet published.
> Check https://sites.ualberta.ca/~rabelo/COLIEE2026/ when available.

---

## 📌 Quick Reference

| Item | Details |
|------|---------|
| **Organizer** | University of Alberta (Rabelo et al.) |
| **Data Source** | Federal Court of Canada (Compass Law) |
| **Max Submissions** | 3 runs per team per task |
| **Evaluation** | Micro-averaged F1 (Tasks 1 & 2) |
| **Publication** | Springer LNAI proceedings |
| **Workshop** | JURISIN (Japan) |

---

## 1. Tasks Overview

### Task 1: Legal Case Retrieval
- **Input:** Query case Q (with citations redacted)
- **Output:** List of "noticed cases" (supporting cases) from corpus
- **Goal:** Predict which cases were cited by the query case

### Task 2: Legal Case Entailment
- **Input:** 
  - `base_case.txt` (query case with fragments suppressed)
  - `entailed_fragment.txt` (conclusion to find support for)
  - `paragraphs/` (candidate paragraphs from a noticed case)
- **Output:** Paragraph ID(s) that entail the fragment
- **Goal:** Identify which paragraph logically supports the conclusion

---

## 2. Evaluation Metrics

### Tasks 1 & 2: Micro-averaged F1

```
Precision = (correctly retrieved) / (total retrieved)
Recall = (correctly retrieved) / (total relevant)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Important:** Micro-average across ALL queries (not per-query average)

---

## 3. Submission Format

### Task 1 Format
```
<query_file> <retrieved_case> <run_tag>
```

Example:
```
000001 000018 myteam_run1
000001 000045 myteam_run1
000001 000130 myteam_run1
000002 000433 myteam_run1
```

### Task 2 Format
```
<query_id> <paragraph_number> <run_tag>
```

Example:
```
001 013 myteam_run1
002 037 myteam_run1
002 002 myteam_run1
003 008 myteam_run1
```

**Run tag rules:**
- Max 12 characters
- Letters and numbers only
- No punctuation

---

## 4. Data Structure

### Task 1
```
task1_train_files_2025/
├── 000001.txt
├── 000002.txt
├── ...
└── (all cases in flat directory)

task1_train_labels_2025.json:
{
    "query_case.txt": ["noticed_case1.txt", "noticed_case2.txt", ...]
}
```

### Task 2
```
task2_train_files_2025/
├── 001/
│   ├── base_case.txt
│   ├── entailed_fragment.txt
│   └── paragraphs/
│       ├── 001.txt
│       ├── 002.txt
│       └── ...
├── 002/
│   └── ...

task2_train_labels_2025.json:
{
    "001": ["013.txt"],
    "002": ["003.txt", "045.txt"]
}
```

---

## 5. Rules & Constraints

### ✅ Allowed
- Any external data for pretraining
- Any model architecture
- Ensemble methods
- External knowledge bases
- LLM APIs (GPT, Claude, etc.)

### ❌ Not Allowed
- Using test labels before submission
- Human intervention during test phase
- Modifying system after seeing test queries
- Data that directly contains test answers

### ⚠️ Must Document
- All datasets used (with dates)
- Pretrained models and their sources
- Training procedure for reproducibility

---

## 6. Typical Schedule (Based on 2024)

| Date | Event |
|------|-------|
| Nov 1 | Training data release |
| Dec 13 | Test data release |
| Dec 13 | Registration deadline |
| Jan 17-18 | Submission deadline (Tasks 1, 2) |
| Jan 20 | Results announced |
| Jan 31 | Paper submission deadline |
| Feb 29 | Acceptance notification |
| Mar 25 | Camera-ready deadline |
| May/Jun | Workshop at JURISIN |

**Deadline timezone:** 23:59 AoE (Anywhere on Earth)

---

## 7. Paper Requirements

- At least one author must present at workshop
- Follow JURISIN format guidelines
- Clearly document:
  - Dataset sources
  - Model architectures
  - Training procedures
  - Reproducibility details

### Publication Tiers
1. **Springer LNAI** - High-quality papers
2. **Local Proceedings (IsAI-JSAI)** - Accepted but not LNAI-qualified

---

## 8. Application Process

1. Download application form from official site
2. Sign memorandum (for data access)
3. Submit to: `coliee_participation@nii.ac.jp`

**For students:** Supervisor signature required on memorandum

---

## 9. Contact

- **General:** coliee_participation@nii.ac.jp
- **Data requests:** rabelo@ualberta.ca

---

## 10. Your Dataset (2025 Version)

Based on your local files:

| | Train | Test |
|---|---|---|
| **Task 1** | 1,678 queries, 7,350 cases | 400 queries |
| **Task 2** | 825 cases | 100 cases |

---

## 11. Tips for Competition

### Do's ✅
1. Start with BM25 baseline (surprisingly strong)
2. Use paragraph-level encoding for long docs
3. Ensemble multiple approaches
4. Tune threshold on dev set carefully
5. Document everything for paper

### Don'ts ❌
1. Don't ignore lexical features
2. Don't use single model only
3. Don't skip error analysis
4. Don't submit without validation

---

## 12. Resources

- **Official site:** https://sites.ualberta.ca/~rabelo/COLIEE2024/
- **Previous proceedings:** Available on official site
- **trec_eval tool:** https://trec.nist.gov/trec_eval/

---

*Last updated: 2026-01-27*
*Based on COLIEE 2024 rules*
