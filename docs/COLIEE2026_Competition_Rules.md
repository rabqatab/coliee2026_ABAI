# COLIEE 2026 Competition Rules

## Overview

COLIEE (Competition on Legal Information Extraction/Entailment) is a legal AI competition focused on case law retrieval and entailment tasks.

---

## General Rules

### LLM Usage Requirements

1. **Open Models Only**: Participants may only use Large Language Models with publicly available training data and/or models (e.g., Hugging Face)
   - **Prohibited**: Closed-source LLMs such as GPT-4o, Gemini, etc.

2. **Model Release Date Restrictions** (Tasks 3 & 4 only):
   - Only LLMs released before **July 15, 2025 (JST)** may be used
   - This restriction prevents test data contamination

3. **Reproducibility Requirement**:
   - Participants must document how to obtain the model in their paper

### Bilingual Data Restrictions

1. **Question Text Processing**:
   - Systems **cannot** simultaneously process both English and Japanese question texts (not realistic)

2. **Law Text Processing**:
   - Using both English and Japanese versions of civil law texts **is permitted** (English translations are publicly available)

3. **Machine Translation Approach**:
   - If using bilingual systems, use machine translation to generate Japanese/English question texts from English/Japanese ones for retrieving results from law texts in both languages

### Automation Requirements

- **No human intervention** at any stage
- Processing must be **entirely automatic** from query execution through result generation
- No modifications to systems based on inspection of test queries

---

## Task 1: Legal Case Retrieval (CL-IR)

### Description

Search a collection of case laws to identify which existing cases support a given query case. Cases are considered "noticed" if they are referenced by the query case.

### Dataset

- **Corpus**: Federal Court of Canada case laws
- **Query Approach**: References within query cases are intentionally removed to simulate realistic scenarios

### Data Format

| Data Type | Contents |
|-----------|----------|
| Training Data | Query cases paired with their noticed (relevant) cases |
| Test Data | Query cases only; participants must predict relevant cases |

### Key Requirements

1. There should be **no human intervention** at any stage
2. No modifications to retrieval system motivated by inspection of test queries
3. Processing must be entirely automated
4. Test case labels remain undisclosed until submission

### Evaluation

Systems are evaluated on their genuine retrieval capabilities without manual adjustments based on test set analysis.

---

## Task 2: Legal Case Entailment (CL-DE)

### Description

Predict the decision of a new case by entailment from previous relevant cases. The system must identify which paragraph from a precedent case supports the decision of a new case.

### Input Components

1. **Query**: New case decision
2. **Noticed Case**: Precedent case
3. **Goal**: Identify the relevant paragraph that logically supports the decision

### Data Format

| Data Type | Contents |
|-----------|----------|
| Training Data | Triples containing: query, noticed case, and paragraph number demonstrating entailment |
| Test Data | Queries and noticed cases only (no paragraph numbers provided) |
| Output | Paragraph(s) from noticed cases that entail the query decision |

### Important Notes

1. **"Decision" Definition**: Refers to judge conclusions entailed by specific case paragraphs, NOT final verdicts
2. Relevant information is packaged in `entailed_fragment.txt` files
3. All execution must occur without manual adjustments or human review

### Key Requirements

1. Processing must be **entirely automatic**
2. No human intervention allowed
3. No test query modifications based on inspection

---

## Summary Table

| Aspect | Requirement |
|--------|-------------|
| LLM Models | Open-source only (no GPT-4o, Gemini, etc.) |
| Model Release (Tasks 3&4) | Before July 15, 2025 (JST) |
| Bilingual Questions | Cannot use both languages simultaneously |
| Bilingual Law Texts | Permitted |
| Automation | 100% automatic, no human intervention |
| Reproducibility | Must document model acquisition in paper |

---

*Document created for COLIEE 2026 participation reference.*
