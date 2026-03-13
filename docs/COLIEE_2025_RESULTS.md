# COLIEE 2025 Competition Results and Participating Team Papers

**Competition:** 12th Competition on Legal Information Extraction/Entailment (COLIEE 2025)
**Workshop:** Workshop at the 20th International Conference on Artificial Intelligence and Law (ICAIL 2025)
**Location:** Northwestern University, Chicago, USA (hybrid), June 20, 2025
**Proceedings:** https://coliee.org/documents/Proceedings/2025-Proceedings.pdf
**Participation:** 24 teams from 13 countries, ~73 individual researchers

---

## Overview Papers

### 1. ACM ICAIL Overview

**Title:** An Overview of the COLIEE 2025 Competition: Legal Case Law and Statute Law Information Retrieval and Entailment
**Authors:** Randy Goebel, Yoshinobu Kano, Mi-Young Kim, Juliano Rabelo, Ken Satoh, Masaharu Yoshioka (+ Hiroaki Yamada)
**URL:** https://dl.acm.org/doi/10.1145/3769126.3785016
**Venue:** Proceedings of the Twentieth International Conference on Artificial Intelligence and Law (ICAIL 2025)

### 2. Extended Journal Overview (Springer)

**Title:** The COLIEE 2025 Competition on Legal Information Extraction and Entailment: Overview, Discussion, and Dataset Expansion
**URL:** https://www.researchgate.net/publication/400986766
**Status:** Received October 31, 2025; Accepted January 20, 2026 (likely for Review of Socionetwork Strategies journal)

### Participation Summary

| Task | Description | Teams | Runs |
|------|------------|-------|------|
| Task 1 | Case Law Retrieval | 8 | 21 |
| Task 2 | Case Law Entailment | 6 | 18 |
| Task 3 | Statute Law Retrieval | 8 | 22 |
| Task 4 | Statute Law Entailment | 10 | 29 |
| Pilot | Tort Law (Prediction + Rationale) | 4 | 10 |

**Note:** COLIEE 2025 introduced two new pilot tasks: Tort Prediction (TP) and Rationale Extraction (RE) for Japanese tort cases (Civil Code Art. 709).

---

## Task 1 - Case Law Retrieval (Results)

**Dataset:** ~7,350 case law files; 2,159 test queries
**Metric:** Micro-averaged F1
**Key observation:** Top F1 dropped from 0.4432 (2024) to 0.3353 (2025), though direct comparison is difficult due to different test sets.

### Task 1 Leaderboard (Best Run per Team, Ranked by F1)

| Rank | Team | Best Run | F1 | Precision | Recall |
|------|------|----------|-----|-----------|--------|
| 1 | **JNLP** | JNLP_001 | **0.3353** | 0.3042 | 0.3735 |
| 1* | JNLP | JNLP_002 | 0.3267 | 0.2945 | 0.3667 |
| 2 | **UQLegalAI** | CaseLink_3 | **0.2962** | 0.2908 | -- |
| 2* | UQLegalAI | CaseLink_2 | 0.2950 | -- | -- |
| 2* | UQLegalAI | CaseLink_1 | 0.2940 | -- | -- |
| 3 | **AIIR Lab** | AIIR_bi | **0.2171** | 0.2327 | 0.2398 |
| 4 | **UA** | (post-deadline) | **0.2073** | 0.1892 | 0.2291 |
| 5 | **NOWJ** | NOWJ_001 | **0.1984** | 0.1670 | 0.2445 |
| 6 | **OVGU** | -- | lower | -- | -- |
| 7 | **UB_2025** | UB3 | ~0.1363 | low | 0.5910 (highest recall) |
| 8 | **SIL** | -- | **0.0058** | -- | -- |

*Note: Some exact scores are from search result snippets and may not perfectly match final published tables. UA's result was submitted post-deadline with corrections.*

---

## Task 2 - Case Law Entailment (Results)

**Dataset:** 100 test cases (826-925), each with base case, entailed fragment, and candidate paragraphs
**Metric:** Micro-averaged F1
**Key observation:** 6 teams, 18 runs. NOWJ dominated with all three runs in top positions.

### Task 2 Leaderboard (Best Run per Team, Ranked by F1)

| Rank | Team | Best Run | F1 |
|------|------|----------|-----|
| 1 | **NOWJ** | NOWJ_003 | **0.3195** |
| 1* | NOWJ | NOWJ_002 | 0.2865 |
| 1* | NOWJ | NOWJ_001 | 0.2782 |
| 2 | **OVGU** | OVGU_2 | **0.2454** |
| 3 | **JNLP** | JNLP_002 | **0.2412** |
| 3* | JNLP | JNLP_003 | 0.2400 |
| 4 | **AIIR Lab** | AIIRLab_cross | **0.2368** |
| 5 | **CAPTAIN** | qwen2572bm | **0.1882** |
| 6 | **UA** | UA_3 | **0.1778** |

---

## Task 3 - Statute Law Retrieval (Results)

**Dataset:** Japanese Civil Code articles; statute retrieval for bar exam-style questions
**Metric:** F2 score (weighted toward recall)
**Teams:** 8 teams, 22 runs

| Rank | Team | Best Score |
|------|------|-----------|
| 1 | **JNLP** | F2 = 0.836 |
| -- | Other teams | -- |

*JNLP's approach: three-stage pipeline with instruction-based LLM pre-retrieval, cross-encoder classification, and ensemble of multiple LLM outputs.*

---

## Task 4 - Statute Law Entailment (Results)

**Dataset:** Bar exam questions requiring yes/no entailment judgment based on Japanese Civil Code
**Metric:** Accuracy
**Teams:** 10 teams, 29 runs

| Rank | Team | Best Score |
|------|------|-----------|
| 1 | **KIS** | Accuracy = **0.9041** |
| -- | Other teams | -- |

*KIS used balanced few-shot prompting with a LLaMA 3.1-based LLM.*

---

## Pilot Task - Legal Judgment Prediction for Japanese Tort Law (Results)

**New in 2025.** Two subtasks: Tort Prediction (TP) and Rationale Extraction (RE).
**Teams:** 4 teams, 10 runs

| Subtask | Winner | Run | Score |
|---------|--------|-----|-------|
| Tort Prediction | **CAPTAIN** | JAIST-LJPJT25 | Accuracy = **0.765** |
| Rationale Extraction | **KIS** | KIS5 | F1 = **0.712** |

*CAPTAIN used fine-tuned Linkbricks-Horizon-AI-Japanese-Pro-V5-70B (70B params). KIS used fine-tuned modernbert-ja-130m (130M params) -- notably, the smaller model outperformed the larger one for rationale extraction.*

---

## Team Papers and Methods

### 1. JNLP (JAIST) -- Task 1 Winner, Task 3 Winner, Pilot (Tort Prediction co-winner)

**Paper:** "Hybrid Large Language Model-based Framework for Legal Information Retrieval and Entailment"
**Authors:** Hai Nguyen, Hiep Nguyen, Trang Pham, Minh Nguyen, An Trieu, Dinh-Truong Do, Nguyen-Khang Le, Le-Minh Nguyen
**Affiliation:** Japan Advanced Institute of Science and Technology (JAIST), Nguyen Lab
**Tasks:** 1, 2, 3, 4, Pilot
**URL (JAIST award):** https://www.jaist.ac.jp/english/whatsnew/awards/2025/08/20-1.html
**URL (Proceedings):** https://coliee.org/documents/Proceedings/2025-Proceedings.pdf

**Task 1 Method (F1 = 0.3353):**
- Built on UMNLP framework from COLIEE 2024 (pairwise similarity ranking with feed-forward neural network for binary classification)
- Extended feature set with BM25 scores and SAILER scores (capturing both lexical-matching and semantic/structural information)
- Stage 1: BM25 filtering to retrieve top 100-200 candidates (achieves 76-85% recall), processing entire corpus in <30 seconds
- Stage 2: Feature extraction for query-candidate pairs including BM25/QLD scores (calculated dynamically based on top-k retrieved docs)
- Stage 3: LightGBM (Gradient Boosting Decision Tree) re-ranking model trained on relevance labels
- LightGBM selected for strong tabular data performance, computational efficiency (gradient-based one-side sampling, exclusive feature bundling)

**Task 2 Method (F1 = 0.2412):**
- Two-stage pipeline: fine-tuning re-rankers with hard-negative sampling + refining predictions using few-shot prompted LLMs

**Task 3 Method (F2 = 0.836):**
- Three-stage pipeline: (1) instruction-based LLM + reranker for high-recall pre-retrieval, (2) cross-encoder LLM for article relevance classification, (3) ensemble of multiple LLM outputs

**Task 4 Method:**
- Zero-shot, few-shot, and reasoning ensemble prompting using models like Qwen2-72B

---

### 2. UQLegalAI -- Task 1 Runner-Up

**Paper:** "UQLegalAI@COLIEE2025: Advancing Legal Case Retrieval with Large Language Models and Graph Neural Networks"
**Authors:** Yanran Tang, Ruihong Qiu, et al.
**Affiliation:** University of Queensland, Australia
**Tasks:** 1
**arXiv:** https://arxiv.org/abs/2505.20743
**GitHub:** https://github.com/yanran-tang/CaseLink

**Task 1 Method (F1 = 0.2962):**
- **CaseLink** model: inductive graph learning for legal case retrieval
- **Graph construction:** Training and test sets transferred into Global Case Graphs (GCG) exploiting Case-Case, Case-Charge, and Charge-Charge relationships
- **Node features:** LLM text embeddings transform legal texts into dense representations as node features
- **GNN module:** Graph neural network generates case representations by aggregating neighborhood information
- **Training:** InfoNCE contrastive loss + degree regularization
- Three submitted runs (F1: 0.2940, 0.2950, 0.2962) -- all among top submissions
- Key strength: captures intrinsic case connectivity that text-only methods miss

---

### 3. NOWJ -- Task 2 Winner

**Paper:** "NOWJ@COLIEE 2025: A Multi-stage Framework Integrating Embedding Models and Large Language Models for Legal Retrieval and Entailment"
**Authors:** Hoang-Trung Nguyen, Tan-Minh Nguyen, Xuan-Bach Le, Tuan-Kiet Le, Khanh-Huyen Nguyen, Ha-Thanh Nguyen, Thi-Hai-Yen Vuong, Le-Minh Nguyen
**Affiliation:** VNU University of Engineering and Technology (Hanoi), JAIST, NII (Tokyo)
**Tasks:** 1, 2, 3, 4, Pilot (all five)
**arXiv:** https://arxiv.org/abs/2509.08025

**Task 1 Method (F1 = 0.1984, 4th place by team):**
- Four-stage framework: pre-processing, LLM-based summarization, retrieval, re-ranking
- Pre-ranking: BM25, BERT, monoT5
- Embeddings: BGE-m3, LLM2Vec
- LLMs: Qwen-2, QwQ-32B, DeepSeek-V3 for summarization, relevance scoring, contextual re-ranking

**Task 2 Method (F1 = 0.3195, 1st place):**
- Two-stage system combining lexical-semantic filtering with contextualized LLM analysis
- Best run (NOWJ_003): BM25 retrieved top-35 candidates, reranked using DeepSeek-V3 and Qwen/QwQ-32B
- Final predictions by LLM-based entailment classification + majority voting
- Key insight: ensemble voting across multiple LLMs significantly boosted performance

---

### 4. CAPTAIN (JAIST) -- Pilot Task (Tort Prediction) Winner

**Paper:** "Enhancing Legal Text Processing and Structural Analysis with Large Language Models"
**Authors:** Dat Nguyen, Minh-Phuong Nguyen, Quang-Huy Chu, Son T. Luu, Nguyen-Hoang Chu, Trung Vo, Le-Minh Nguyen
**Affiliation:** Japan Advanced Institute of Science and Technology (JAIST), Nguyen Lab
**Tasks:** 2, 3, 4, Pilot
**URL (JAIST award):** https://www.jaist.ac.jp/english/whatsnew/awards/2025/08/20-1.html

**Task 2 Method (F1 = 0.1882):**
- Two-stage: fine-tuning re-rankers with hard-negative sampling + few-shot prompted LLMs for prediction refinement

**Task 3 Method:**
- Three-stage: embedding-based pre-retrieval, LoRA/QLoRA-based fine-tuning, model ensembling

**Task 4 Method:**
- Zero-shot, few-shot, and reasoning ensemble prompting using Qwen2-72B

**Pilot Task (Tort Prediction, Acc = 0.765):**
- Fine-tuned Linkbricks-Horizon-AI-Japanese-Pro-V5-70B

---

### 5. OVGU (Otto von Guericke University Magdeburg)

**Paper:** "LLMs, Knowledge Graphs, and Hybrid Search: Task-Specific Approaches to Legal AI in COLIEE"
**Authors:** (Magdeburg team, including Sabine Wehnert's group)
**Affiliation:** Otto von Guericke University Magdeburg, Germany
**Tasks:** 1, 2, 3, 4, Pilot
**URL:** https://www.researchgate.net/publication/394930419

**Task 1 Method:**
- Ensemble-based system using multiple open-source LLMs via Ollama: gemma3:12b, wizardlm2:7b, phi4, gemma2, deepseek-r1:32b
- OVGU_1: top-5 paragraphs via BM25Plus + case summaries with phi4
- OVGU_2: filtered candidates to those labeled as entailed by LLM ensemble
- OVGU_3: ranked entailed candidates based on BM25 score differences with predefined threshold

**Task 2 Method (F1 = 0.2454, 2nd place):**
- Created "silver dataset" by prompting nine LLMs on training set using problem type definitions
- Models asked to identify problem types and predict entailment labels with reasoning
- Fine-tuned Phi-3-medium-4k-instruct and gemma-1.1-7b-it on silver data
- OVGU1: majority voting across model predictions

**Key techniques:** Proposition-based reformulation, chunked summarization, judge-aware reranking, silver data fine-tuning. Competed with limited resources.

---

### 6. AIIR Lab (University of Southern Maine)

**Paper:** "AIIR Lab at COLIEE 2025: Exploring Applications of Large Language Models for Legal Text Retrieval and Entailment"
**Authors:** Deiby Wu, Sarah Lawrence, Behrooz Mansouri
**Affiliation:** University of Southern Maine (USM), Portland, USA
**Tasks:** 1, 2, 3, 4
**URL:** https://www.researchgate.net/publication/393924641

**Task 1 Method (F1 = 0.2171):**
- LLMs (Mistral-7B, LLaMA-3) for case summarization
- Fine-tuned bi-encoder for ranking

**Task 2 Method (F1 = 0.2368):**
- Fine-tuned cross-encoder for entailment assessment between case paragraphs
- Fallback strategy for edge cases

**Task 3 Method:**
- Augmented training data with LLMs
- Fine-tuned bi-encoder for statute search

**Task 4 Method:**
- Three prompting techniques: zero-shot, few-shot, chain-of-thought (CoT)
- Majority voting for final answer

---

### 7. KIS -- Task 4 Winner, Pilot (Rationale Extraction) Winner

**Paper:** "KIS: COLIEE 2025 Task 4 Solver Using Japanese LLM"
**Authors:** (KIS team)
**Tasks:** 4, Pilot

**Task 4 Method (Accuracy = 0.9041):**
- Balanced few-shot prompting with LLaMA 3.1-based LLM
- Straightforward prompting approach proved highly effective

**Pilot (Rationale Extraction, F1 = 0.712):**
- Fine-tuned modernbert-ja-130m (130M parameters)
- Notably beat CAPTAIN's 70B model, demonstrating smaller masked language models can be competitive against much larger autoregressive models

---

### 8. SIL

**Paper:** "SIL@COLIEE 2025: A Cascading Framework for Finding Relevant Case Laws"
**Authors:** Bhavya Jain, Pooja Harde, Taha Sadikot, Eric Namit Kujur, Sarika Jain
**Tasks:** 1

**Task 1 Method (F1 = 0.0058):**
- Stage 1: MPNet to encode documents into dense embeddings, indexed with FAISS for inner product similarity search (top 100 candidates)
- Stage 2: LightGBM re-ranking model trained on 9 features (BM25/QLD scores/ranks, document/query lengths, citation counts, Doc2Vec similarity)
- Relevance threshold filtering
- Despite multi-stage design, achieved limited performance

---

### 9. UB_2025

**Paper:** (In proceedings)
**Tasks:** 1

**Task 1 Method (F1 = ~0.1363, highest recall = 0.5910):**
- Rhetorical role-based summarization
- Pre-processed documents to remove non-informative content
- MTLD (Measure of Textual Lexical Diversity)-based filtering
- Gradient boosting classifier to label rhetorical roles (facts, arguments, etc.)
- Three strategies tested:
  - Original queries to summarized candidates (highest recall = 0.6379)
  - Summarized queries to original documents (higher precision)
  - Summarized queries to summarized documents
- Despite low F1, demonstrated potential of structure-informed retrieval through very high recall

---

### 10. UA (University of Alberta)

**Paper:** (In proceedings)
**Affiliation:** Department of Computing Science, University of Alberta
**Tasks:** 1, 2, 3, 4

**Task 1 Method (F1 = 0.2073, post-deadline):**
- TF-IDF vectors (1-3 word n-grams) with cosine similarity for initial retrieval
- Date filtering and dynamic thresholding (retain top 50% when >10 candidates)
- 200-word summaries generated using Qwen2-7B

**Task 4 Method:**
- Pre-trained on legal corpus, fine-tuned with LoRA
- 15 sampling runs per query, majority voting for final Yes/No predictions

*Note: Official submissions suffered from implementation bug; post-evaluation showed corrected F1 of 0.2073.*

---

### 11. IRNLPUI

**Paper:** "IRNLPUI at COLIEE 2025: Utilization of LLMs for Statute Law Retrieval and Legal Entailment Task"
**Authors:** Bryan Tjandra, Made Swastika Nata Negara, Alfan Farizki Wicaksono
**Tasks:** 3, 4

**Method:**
- Initial retrieval using TF-IDF vectors (1-3 word n-grams) with cosine similarity
- Date filtering and dynamic thresholding
- Qwen2-7B for 200-word summary generation
- Official submissions suffered from implementation bug; post-evaluation F1 = 0.2073

---

### 12. IIT Bhilai Team (Citation-Neighbourhood)

**Paper:** "Knowledge-Based Legal Case Retrieval"
**Authors:** Chetana et al.
**Affiliation:** IIT Bhilai
**Tasks:** 1
**GitHub:** https://github.com/chetaniitbhilai/Knowledge-Based-Legal-Case-Retrieval

**Method:**
- Citation-neighbourhood retrieval framework
- Represents each case using compact textual segments surrounding citation markers
- Computes similarity-based features on filtered set
- Ensemble classifier combining MLP + Random Forest for relevance scoring

---

## Key Trends and Takeaways for COLIEE 2026

### 1. Multi-Stage Pipelines Dominate
Every top-performing system uses multi-stage pipelines: fast first-stage retrieval (BM25) followed by neural re-ranking. No single-stage system performed competitively.

### 2. BM25 Remains Essential
BM25 is the universal first-stage retriever. JNLP showed BM25 top-100/200 filtering achieves 76-85% recall while processing the entire corpus in <30 seconds. All top systems build on BM25 as a foundation.

### 3. LLM Integration is Now Standard
Almost every team used LLMs (DeepSeek-V3, QwQ-32B, Qwen2-72B, LLaMA-3, Mistral-7B) for summarization, feature extraction, entailment classification, or re-ranking. The best results came from using LLMs in specific pipeline stages, not end-to-end.

### 4. Graph-Based Methods Show Promise (Task 1)
UQLegalAI's CaseLink (GNN + case connectivity graphs) achieved consistent second place, suggesting that modeling inter-case relationships provides signal that text-only methods miss. This is directly relevant to our GraphRAG approach.

### 5. Ensemble/Voting is Critical (Task 2)
NOWJ's winning Task 2 approach used majority voting across multiple LLMs (DeepSeek-V3, QwQ-32B). Ensemble strategies consistently outperformed single-model approaches.

### 6. Feature Engineering Still Matters (Task 1)
JNLP's winning approach used handcrafted features (BM25, QLD, SAILER scores) fed into LightGBM -- not a pure neural approach. The combination of lexical, semantic, and structural features proved most effective.

### 7. Document Structure Emerging as Signal
Teams like UB_2025 (rhetorical roles) and OVGU (judge-aware reranking) explored document structure, achieving high recall despite precision challenges. This suggests structure-aware processing is an underexploited signal.

### 8. Scores Remain Modest
Task 1 top F1 of 0.3353 and Task 2 top F1 of 0.3195 reflect the inherent difficulty of legal case retrieval and entailment. There is substantial room for improvement.

### 9. Open-Source Models Competitive
All systems used open-source models (required by competition rules since 2025). DeepSeek-V3, QwQ-32B, Qwen2-72B, LLaMA-3, and various BERT variants were the most popular choices.

### 10. Smaller Models Can Win
KIS's 130M parameter modernbert-ja beat CAPTAIN's 70B parameter model for rationale extraction, showing task-specific fine-tuning of small models can outperform much larger ones.

---

## Summary of Winners by Task

| Task | Winner | F1/Accuracy | Key Method |
|------|--------|-------------|------------|
| Task 1 (Case Retrieval) | **JNLP** | F1 = 0.3353 | BM25 filtering + SAILER + LightGBM re-ranking |
| Task 2 (Case Entailment) | **NOWJ** | F1 = 0.3195 | BM25 + DeepSeek-V3/QwQ-32B + majority voting |
| Task 3 (Statute Retrieval) | **JNLP** | F2 = 0.836 | LLM pre-retrieval + cross-encoder + ensemble |
| Task 4 (Statute Entailment) | **KIS** | Acc = 0.9041 | Few-shot prompting with LLaMA 3.1 |
| Pilot (Tort Prediction) | **CAPTAIN** | Acc = 0.765 | Fine-tuned 70B Japanese LLM |
| Pilot (Rationale Extraction) | **KIS** | F1 = 0.712 | Fine-tuned modernbert-ja-130m |

---

## Available Papers / URLs

| Team | arXiv | Proceedings | Other |
|------|-------|-------------|-------|
| Overview (ACM) | -- | [ACM](https://dl.acm.org/doi/10.1145/3769126.3785016) | -- |
| Overview (Springer) | -- | -- | [ResearchGate](https://www.researchgate.net/publication/400986766) |
| NOWJ | [2509.08025](https://arxiv.org/abs/2509.08025) | Yes | -- |
| UQLegalAI | [2505.20743](https://arxiv.org/abs/2505.20743) | Yes | [GitHub](https://github.com/yanran-tang/CaseLink) |
| JNLP | -- | Yes | [JAIST](https://www.jaist.ac.jp/english/whatsnew/awards/2025/08/20-1.html) |
| CAPTAIN | -- | Yes | [JAIST](https://www.jaist.ac.jp/english/whatsnew/awards/2025/08/20-1.html) |
| OVGU | -- | Yes | [ResearchGate](https://www.researchgate.net/publication/394930419) |
| AIIR Lab | -- | Yes | [ResearchGate](https://www.researchgate.net/publication/393924641) |
| SIL | -- | Yes | -- |
| KIS | -- | Yes | -- |
| IIT Bhilai | -- | -- | [GitHub](https://github.com/chetaniitbhilai/Knowledge-Based-Legal-Case-Retrieval) |
| Full Proceedings | -- | [PDF](https://coliee.org/documents/Proceedings/2025-Proceedings.pdf) | -- |

---

*Compiled March 2026 from web searches of published papers, proceedings, and competition announcements. Some exact numerical scores may have minor discrepancies from final published tables -- consult the overview papers and proceedings PDF for authoritative numbers.*
