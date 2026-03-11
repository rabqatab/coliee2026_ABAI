# COLIEE 2026 Dataset - Exploratory Data Analysis

## Task 1: Legal Case Retrieval

### Dataset Size
| Split | Queries | Total Citations | Avg Citations/Query |
|-------|---------|-----------------|---------------------|
| Train | 1,678 | 6,881 | 4.10 |
| Test | 400 | - | - |
| Corpus | 7,350 documents | | |

### Citation Distribution
- Min: 1, Max: 34
- 21% queries have only 1 relevant case
- 57% queries have 1-3 relevant cases
- Long-tail distribution

### Document Statistics (n=500 sample)
- **Characters**: min=1,373, max=255,481, mean=30,787
- **Words**: min=233, max=43,317, mean=4,994
- **Paragraphs**: min=4, max=384, mean=50

### Word Count Distribution
| Range | Percentage |
|-------|------------|
| <1k | 4.4% |
| 1k-5k | 61.2% |
| 5k-10k | 27.2% |
| 10k-20k | 4.8% |
| >20k | 2.4% |

### Document Structure (n=300)
- Has `<FRAGMENT_SUPPRESSED>`: 94%
- Has Background/Facts section: 69%
- Has Issues section: 59%
- Has Analysis section: 79%
- Has Conclusion/Order: 88%

### Case Types
| Type | Percentage |
|------|------------|
| Judicial Review | 67% |
| Immigration | 9% |
| IP (Patent/Trademark) | 8% |
| Appeal | 7% |
| Other | 8% |
| Tax Appeal | 1% |

### Top Acts Referenced
1. Refugee Protection Act
2. Immigration Act
3. Patent Act
4. Constitution Act
5. Indian Act

---

## Task 2: Legal Case Entailment

### Dataset Size
| Split | Cases |
|-------|-------|
| Train | 825 |
| Test | 100 |

### Entailing Paragraphs per Case
| Count | Percentage |
|-------|------------|
| 1 | 82.7% |
| 2 | 14.3% |
| 3 | 2.2% |
| 4+ | 0.8% |

### Candidate Paragraphs per Case
- Min: 3, Max: 315, Mean: 35.7, Median: 29
- **Selection ratio: 3.4%** (needle in haystack)

### Entailed Fragment Statistics
- **Length**: min=6, max=115, mean=38 words
- **Sentences**: 84% are single-sentence fragments

### Fragment Content Patterns
| Pattern | Percentage |
|---------|------------|
| Evidence | 23% |
| Must/Shall | 17% |
| Reasonable/Unreasonable | 12% |
| Test/Elements | 8% |
| Burden of Proof | 7% |
| Standard of Review | 5% |

### Lexical Overlap (Fragment ↔ Entailing Paragraph)
- Mean Jaccard: 0.178
- 70% pairs have <20% word overlap
- **Implication**: Requires semantic understanding, not keyword matching

---

## Linguistic Features

### Discourse Markers (per 150 docs)
- however: 657
- therefore: 453
- "I agree/find/conclude": 258
- "in my view": 182
- accordingly: 153

### Citation Patterns
- "at para X": 712 occurrences
- supra: 205
- "at page X": 200

### Sentence Structure
- Avg words per sentence: 23.4
- Max sentence length: 156 words
- Heavy subordination and relative clauses

---

## Korean vs Canadian Legal System Gaps

| Aspect | Korean (Statutory) | Canadian (Common Law) |
|--------|-------------------|----------------------|
| Primary Source | Written statutes (법률) | Case precedents |
| Reasoning Style | Deductive | Analogical |
| Document Length | Shorter | Verbose (5k+ words) |
| Voice | Impersonal | Personal ("I find...") |
| Citation Focus | Statute articles | Prior cases |

### Transfer Potential
- ✅ Entity recognition (parties, judges, dates)
- ✅ Structural parsing (sections, paragraphs)  
- ✅ Domain vocabulary adaptation
- ❌ Precedent reasoning patterns
- ❌ Judge opinion voice modeling

---

## Key Challenges

### Task 1
1. Long documents → need efficient encoding (chunking, hierarchical)
2. Sparse labels → 21% have only 1 relevant case
3. Must capture legal reasoning similarity, not just keywords

### Task 2
1. Low lexical overlap → pure BM25 won't work
2. Needle in haystack → 3.4% selection ratio
3. Requires understanding entailment, not retrieval

---

*Generated: 2026-01-27*
