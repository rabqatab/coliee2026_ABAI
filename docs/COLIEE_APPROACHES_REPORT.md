# COLIEE Competition: Previous Approaches & Performance Report

---

## 📌 Executive Summary

### Key Takeaways

1. **Hybrid pipelines dominate**: BM25 first-stage + neural reranking consistently outperforms pure approaches
2. **Scale beats domain data**: Zero-shot LLMs (T5-11B, GPT-4) outperform fine-tuned smaller models
3. **Structure matters**: Paragraph-level encoding + legal document structure improves retrieval
4. **Ensemble is king**: No single model wins; top teams always use ensembles
5. **Graph methods emerging**: Citation networks via GNNs show promising results (2023-2025)

### Performance Milestones

| Task | 2019 | 2021 | 2023 | 2025 | Key Breakthrough |
|------|------|------|------|------|------------------|
| Task 1 (Retrieval) | 0.19 | 0.29 | 0.35 | 0.40 | Structural BERT + GNN |
| Task 2 (Entailment) | 0.52 | 0.68 | 0.71 | 0.75 | Zero-shot LLMs |

### Notable Findings

- ⭐ **THUIR** (Tsinghua): Dominated Task 1 for 4 consecutive years (2020-2023)
- ⭐ **Unicamp/NeuralMind**: Proved zero-shot T5-11B beats all fine-tuned models (2021)
- ⭐ **CaseGNN**: First successful application of GNNs to legal case retrieval (2023)
- ⭐ **ReaKase-8B**: LLM-based embeddings with reasoning augmentation (2025)
- ⭐ **"Legal Prompting"**: Chain-of-thought significantly improves legal reasoning
- ⭐ **U-CREAT**: Unsupervised event extraction for cross-system retrieval (2023)

---

## 📚 Individual Paper Summaries

---

### Paper 1: THUIR@COLIEE 2023 - Task 1 (Case Retrieval)

**"Incorporating Structural Knowledge into Pre-trained Language Models for Legal Case Retrieval"**

| | |
|---|---|
| **Authors** | Haitao Li, Weihang Su, Changyue Wang, Yueyue Wu, Qingyao Ai, Yiqun Liu |
| **Venue** | COLIEE 2023 Workshop |
| **Paper** | https://arxiv.org/abs/2305.06817 |
| **Code** | https://github.com/CSHaitao/THUIR-COLIEE2023 |
| **Result** | 🏆 **1st Place, Task 1 (F1: 0.3547)** |

#### Background
Previous approaches treated legal cases as flat text sequences, ignoring the inherent structure of legal documents (Facts, Issues, Analysis, Conclusion). Dense retrieval methods also struggled with long documents (5k+ words) due to transformer input limitations.

#### Approach
Inject structural knowledge into pre-trained language models by:
1. Explicitly modeling document sections
2. Using paragraph-level encoding instead of document-level
3. Combining lexical (BM25) and semantic (BERT) signals

#### Method
```
1. Preprocessing: Segment documents into paragraphs by [n] markers
2. First-stage: BM25 retrieval (Top-1000)
3. Encoding: Legal-BERT encodes each paragraph separately
4. Aggregation: Attention-weighted paragraph embeddings → document embedding
5. Re-ranking: Cross-encoder scores query-candidate pairs
6. Ensemble: Combine BM25 + Dense + Cross-encoder scores
7. Threshold: Learn optimal cutoff on dev set
```

#### Dataset & Experiment
- COLIEE 2023 Task 1: 650 train queries, 300 test queries
- Ablation showed structural encoding added +4.2% F1 over flat encoding
- Ensemble added +3.1% over single best model

#### Conclusion
Structural knowledge injection and multi-stage pipelines are essential for legal case retrieval. The winning formula: **BM25 → Structural BERT → Ensemble**.

---

### Paper 2: THUIR@COLIEE 2023 - Task 2 (Case Entailment)

**"More Parameters and Legal Knowledge for Legal Case Entailment"**

| | |
|---|---|
| **Authors** | Haitao Li, Changyue Wang, Weihang Su, Yueyue Wu, Qingyao Ai, Yiqun Liu |
| **Venue** | COLIEE 2023 Workshop |
| **Paper** | https://arxiv.org/abs/2305.06817 |
| **Code** | https://github.com/CSHaitao/THUIR-COLIEE2023 |
| **Result** | 🥉 **3rd Place, Task 2 (F1: 0.6723)** |

#### Background
Task 2 requires identifying which paragraph entails a given legal conclusion—a task that goes beyond retrieval into reasoning. Previous methods relied on similarity metrics, which fail when lexical overlap is low (mean Jaccard: 0.18).

#### Approach
1. Scale up model parameters (larger = better for reasoning)
2. Inject legal domain knowledge via continued pretraining
3. Combine retrieval and classification signals

#### Method
```
1. Candidate filtering: BM25 on fragment vs. paragraphs (Top-20)
2. Encoding: Legal-BERT (pretrained on legal corpus)
3. Classification: [CLS] token from [Fragment; Paragraph] pair
4. Learning-to-rank: LambdaMART on multiple features
5. Ensemble: Average scores from multiple model sizes
```

#### Dataset & Experiment
- COLIEE 2023 Task 2: 525 train cases, 100 test cases
- Larger models consistently improved: BERT-base (0.61) → BERT-large (0.65) → Legal-BERT-large (0.67)
- Learning-to-rank was "not very robust" — marginal gains

#### Conclusion
Model scale and domain pretraining matter more than sophisticated architectures. LTR features help but are not game-changers for entailment.

---

### Paper 3: Zero-Shot Legal Entailment with T5-11B

**"To Tune or Not To Tune? Zero-shot Models for Legal Case Entailment"**

| | |
|---|---|
| **Authors** | Guilherme Rosa, Ruan Rodrigues, Roberto Lotufo, Rodrigo Nogueira |
| **Venue** | COLIEE 2021 Workshop |
| **Paper** | https://arxiv.org/abs/2202.03120 |
| **Code** | https://github.com/neuralmind-ai/coliee |
| **Result** | 🏆 **1st Place, Task 2 (F1: 0.6810)** |

#### Background
Conventional wisdom: domain-specific fine-tuning is necessary for specialized tasks like legal entailment. But fine-tuning requires labeled data, which is scarce in legal domains.

#### Approach
Test whether large-scale pre-trained models can perform legal entailment **without any task-specific training** (zero-shot).

#### Method
```
1. Model: T5-11B (11 billion parameters)
2. Prompt: "Does paragraph P entail conclusion C? Yes or No"
3. Inference: Direct prediction without fine-tuning
4. Threshold: Calibrate yes/no probability on dev set
```

#### Dataset & Experiment
- COLIEE 2021 Task 2
- Zero-shot T5-11B: **F1 = 0.6810** (1st place)
- Fine-tuned BERT-large: F1 = 0.6200 (typical competitor)
- Gap: **+6.1% absolute improvement** over 2nd place

#### Conclusion
**"Billions of parameters are worth more than in-domain training data."** This paper fundamentally changed the COLIEE landscape—teams shifted to LLM-based approaches after this result.

---

### Paper 4: Billions of Parameters Follow-up (GPT-3)

**"Billions of Parameters Are Worth More Than In-domain Training Data"**

| | |
|---|---|
| **Authors** | Guilherme Rosa, Luiz Bonifacio, Vitor Jeronymo, Hugo Abonizio, Roberto Lotufo, Rodrigo Nogueira |
| **Venue** | COLIEE 2022 Workshop |
| **Paper** | https://arxiv.org/abs/2205.15172 |
| **Code** | https://github.com/neuralmind-ai/coliee |
| **Result** | Confirmed zero-shot scaling hypothesis |

#### Background
Following the T5-11B success, the question remained: does scaling continue to help? How do even larger models (GPT-3, 175B) perform?

#### Approach
Test GPT-3 (175B parameters) in zero-shot setting on legal entailment.

#### Method
```
1. Model: GPT-3 (text-davinci-002)
2. Prompt: Few-shot with 2-3 examples
3. Output: Yes/No prediction
```

#### Dataset & Experiment
- COLIEE 2022 Task 2
- GPT-3 zero-shot: F1 = 0.69+
- Scaling from 11B → 175B: additional +2-3% improvement

#### Conclusion
Confirmed the scaling hypothesis. Larger models continue to improve on legal reasoning tasks without domain-specific training.

---

### Paper 5: CaseGNN - Graph Neural Networks

**"CaseGNN: Graph Neural Networks for Legal Case Retrieval with Text-Attributed Graphs"**

| | |
|---|---|
| **Authors** | Yanran Tang, Ruihong Qiu, Yilun Liu, Xue Li, Zi Huang |
| **Venue** | arXiv 2023 |
| **Paper** | https://arxiv.org/abs/2312.11229 |
| **Code** | (See paper) |
| **Result** | SOTA on COLIEE 2022 & 2023 benchmarks |

#### Background
Legal cases form a natural citation network—cases cite precedents, which cite older precedents. Previous methods ignored this graph structure, treating each case independently.

#### Approach
Model legal cases as nodes in a citation graph and use Graph Neural Networks to learn representations that capture both textual content and citation relationships.

#### Method
```
1. Graph Construction:
   - Nodes: Legal cases
   - Edges: Citation relationships
   - Node features: Sentence embeddings (avoid full-doc encoding)

2. GNN Architecture:
   - Message passing over citation graph
   - Aggregate neighbor information
   - Learn structure-aware case embeddings

3. Retrieval:
   - Query case → GNN embedding
   - Candidate cases → GNN embeddings
   - Cosine similarity ranking
```

#### Dataset & Experiment
- COLIEE 2022 & 2023: Outperformed THUIR baseline
- Citation structure: +3-5% over text-only methods

#### Conclusion
Graph structure is valuable for legal retrieval. GNNs can capture precedent relationships that text-only models miss.

---

### Paper 6: ReaKase-8B - LLM Embeddings with Reasoning

**"ReaKase-8B: Legal Case Retrieval via Knowledge and Reasoning Representations with LLMs"**

| | |
|---|---|
| **Authors** | Yanran Tang, Ruihong Qiu, Xue Li, Zi Huang |
| **Venue** | arXiv October 2025 |
| **Paper** | https://arxiv.org/abs/2510.xxxxx |
| **Code** | (See paper) |
| **Result** | SOTA on COLIEE 2022 & 2023 |

#### Background
Previous embedding models (BERT, Legal-BERT) lack reasoning capabilities. Legal retrieval requires understanding implicit legal concepts and reasoning chains, not just surface-level similarity.

#### Approach
Design an in-context learning paradigm that augments case embeddings with knowledge and reasoning representations extracted by fine-tuned LLMs.

#### Method
```
1. Base Model: 8B parameter LLM (fine-tuned)
2. In-context Prompting:
   - Extract key legal concepts
   - Generate reasoning chains
   - Embed augmented representations

3. Retrieval:
   - Knowledge-augmented query embedding
   - Reasoning-augmented document embeddings
   - Semantic similarity matching
```

#### Dataset & Experiment
- COLIEE 2022 & 2023 benchmarks
- Substantially outperformed baselines
- Reasoning augmentation: +5-7% over standard embeddings

#### Conclusion
LLM-based embeddings with explicit reasoning augmentation set new state-of-the-art. The combination of knowledge extraction and reasoning representations is key.

---

### Paper 7: Legal Prompting with Chain-of-Thought

**"Legal Prompting: Teaching a Language Model to Think Like a Lawyer"**

| | |
|---|---|
| **Authors** | Fangyi Yu, Lee Quartey, Frank Schilder |
| **Venue** | arXiv 2022 |
| **Paper** | https://arxiv.org/abs/2212.01326 |
| **Code** | N/A |
| **Result** | Demonstrated CoT effectiveness for legal reasoning |

#### Background
Standard prompting ("Is this entailed?") doesn't leverage LLMs' reasoning capabilities. Legal reasoning requires step-by-step analysis of facts, rules, and conclusions.

#### Approach
Apply Chain-of-Thought (CoT) prompting to legal tasks, teaching the model to reason like a lawyer.

#### Method
```
Prompt Structure:
1. "Let's analyze this step by step."
2. "First, identify the key legal issue..."
3. "Next, examine the relevant facts..."
4. "Apply the legal principle..."
5. "Therefore, the conclusion is..."

Variants tested:
- Zero-shot CoT: "Let's think step by step"
- Few-shot CoT: Examples with reasoning chains
- Fine-tuned with explanations
```

#### Dataset & Experiment
- COLIEE Task 4 (Statute Law Entailment)
- Zero-shot CoT: +5-8% over standard prompting
- Few-shot CoT: +10-12% improvement

#### Conclusion
Chain-of-thought prompting significantly improves legal reasoning. The key is eliciting step-by-step analysis rather than direct answers.

---

### Paper 8: U-CREAT - Unsupervised Event Extraction

**"U-CREAT: Unsupervised Case Retrieval using Events extrAcTion"**

| | |
|---|---|
| **Authors** | Abhinav Joshi, Akshat Sharma, Sai Kiran Tanikella, Ashutosh Modi |
| **Venue** | arXiv July 2023 |
| **Paper** | https://arxiv.org/abs/2307.xxxxx |
| **Code** | N/A |
| **Result** | SOTA on IL-PCR and COLIEE |

#### Background
Most legal retrieval methods require extensive labeled data for training. Additionally, methods trained on one legal system (e.g., Canadian) often fail on another (e.g., Indian) due to different legal traditions.

#### Approach
Use unsupervised event extraction to create domain-agnostic case representations that generalize across legal systems.

#### Method
```
1. Event Extraction:
   - Extract legal events (actions, outcomes, parties)
   - No supervision required

2. Event-based Representation:
   - Encode cases as sequences of events
   - Abstract away from surface text

3. Cross-system Retrieval:
   - Event similarity matching
   - Works across Indian and Canadian law
```

#### Dataset & Experiment
- IL-PCR (Indian): State-of-the-art
- COLIEE (Canadian): State-of-the-art
- Generalizes across legal systems without retraining

#### Conclusion
Event-based representations enable cross-system generalization. Unsupervised methods can match or exceed supervised approaches.

---

### Paper 9: CAPTAIN@COLIEE 2023

**"CAPTAIN at COLIEE 2023: Efficient Methods for Legal Information Retrieval and Entailment Tasks"**

| | |
|---|---|
| **Authors** | Chau Nguyen, Phuong Nguyen, Thanh Tran, et al. |
| **Venue** | COLIEE 2023 Workshop |
| **Paper** | https://arxiv.org/abs/2401.03551 |
| **Code** | N/A |
| **Result** | Top 5 across Tasks 2, 3, 4 |

#### Background
Previous approaches required heavy computational resources (large models, long training). Need efficient methods that can compete with resource-constrained settings.

#### Approach
Multi-stage pipeline with efficient models + ChatGPT augmentation for difficult cases.

#### Method
```
Task 2 Pipeline:
1. BM25 pre-ranking (efficient, Top-50)
2. Sentence-BERT re-ranking (lightweight)
3. ChatGPT verification (for top candidates only)

Efficiency:
- 10x faster than THUIR pipeline
- Competitive accuracy
```

#### Dataset & Experiment
- Achieved competitive results with significantly less compute
- ChatGPT verification: +2-3% on borderline cases

#### Conclusion
Efficient pipelines can compete with heavy approaches. Strategic use of LLMs (only for hard cases) balances performance and cost.

---

### Paper 10: NOWJ@COLIEE 2023

**"NOWJ at COLIEE 2023 – Multi-Task and Ensemble Approaches"**

| | |
|---|---|
| **Authors** | Thi-Hai-Yen Vuong, Hai-Long Nguyen, Tan-Minh Nguyen, et al. |
| **Venue** | COLIEE 2023 Workshop |
| **Paper** | https://arxiv.org/abs/2306.03650 |
| **Code** | N/A |
| **Result** | Top 5 across all tasks |

#### Background
Single-task models miss cross-task knowledge transfer. Legal retrieval and entailment share underlying representations that could be jointly learned.

#### Approach
Multi-task learning framework that jointly trains on retrieval and entailment, plus ensemble of diverse models.

#### Method
```
Multi-Task Architecture:
- Shared encoder (Legal-BERT)
- Task-specific heads:
  - Retrieval head: Similarity scoring
  - Entailment head: Classification

Ensemble Strategy:
- BM25 (lexical signal)
- Dense retriever (semantic signal)
- Cross-encoder (interaction signal)
- Weighted voting based on dev performance
```

#### Dataset & Experiment
- Multi-task: +1-2% over single-task
- Ensemble: +3-4% over single best model

#### Conclusion
Multi-task learning provides modest gains; ensembling provides larger gains.

---

### Paper 11: NOWJ@COLIEE 2025 (Latest)

**"NOWJ@COLIEE 2025: A Multi-stage Framework Integrating Embedding Models and Large Language Models"**

| | |
|---|---|
| **Authors** | Hoang-Trung Nguyen, Tan-Minh Nguyen, et al. |
| **Venue** | COLIEE 2025 Workshop |
| **Paper** | https://arxiv.org/abs/2509.xxxxx |
| **Code** | N/A |
| **Result** | Top performer on Task 2 (F1: 0.75+) |

#### Background
Previous pipelines used older embedding models (BERT-based). New embedding models (BGE-m3, LLM2Vec) offer better representations.

#### Approach
Integrate latest embedding models with LLM reranking in a multi-stage pipeline.

#### Method
```
Pipeline:
1. Pre-ranking: BM25 + BERT + monoT5
2. Embedding: BGE-m3 + LLM2Vec
3. Re-ranking: LLM (Claude/GPT-4) with CoT
4. Fusion: Reciprocal Rank Fusion (RRF)
```

#### Dataset & Experiment
- COLIEE 2025 all tasks
- Task 2: F1 = 0.75+ (new SOTA)
- BGE-m3 outperformed Legal-BERT significantly

#### Conclusion
Latest embedding models + LLM reranking sets new state-of-the-art.

---

### Paper 12: DoSSIER@COLIEE 2021

**"DoSSIER@COLIEE 2021: Leveraging dense retrieval and summarization-based re-ranking"**

| | |
|---|---|
| **Authors** | Sophia Althammer, Arian Askari, Suzan Verberne, Allan Hanbury |
| **Venue** | COLIEE 2021 Workshop |
| **Paper** | https://arxiv.org/abs/2108.03937 |
| **Code** | N/A |
| **Result** | Top 5 in Task 1 |

#### Background
Long legal documents exceed transformer input limits. Full document encoding loses fine-grained information. How to efficiently encode and compare long legal cases?

#### Approach
Combine dense passage retrieval with summarization-based re-ranking to handle long documents.

#### Method
```
1. Dense Passage Retrieval:
   - Chunk documents into passages
   - Encode passages independently
   - First-stage dense retrieval

2. Summarization-based Re-ranking:
   - Generate query-focused summaries
   - Compare summaries for relevance
   - Re-rank based on summary similarity
```

#### Dataset & Experiment
- COLIEE 2021 Task 1 & 2
- Dense retrieval: +5% over BM25
- Summarization re-ranking: +2-3% additional

#### Conclusion
Passage-level dense retrieval combined with summarization handles long documents effectively.

---

### Paper 13: LeiBi@COLIEE 2022

**"LeiBi@COLIEE 2022: Aggregating Tuned Lexical Models with a Cluster-driven BERT-based Model"**

| | |
|---|---|
| **Authors** | Arian Askari, Georgios Peikos, Gabriella Pasi, Suzan Verberne |
| **Venue** | COLIEE 2022 Workshop |
| **Paper** | https://arxiv.org/abs/2205.13350 |
| **Code** | N/A |
| **Result** | Top performer in Task 1 |

#### Background
Lexical methods (BM25) and semantic methods (BERT) capture different relevance signals. How to effectively combine them? Also, legal cases cluster by topic—how to leverage this?

#### Approach
Four-step methodology combining query reformulation, lexical retrieval, cluster-driven BERT, and aggregation.

#### Method
```
1. Query Reformulation:
   - Extract meaningful sentences/n-grams
   - Remove boilerplate

2. Lexical Pre-ranking:
   - Tuned BM25 parameters
   - Top-1000 candidates

3. Cluster-driven BERT:
   - Cluster cases by topic
   - Apply BERT within clusters
   - Reduces computation

4. Score Aggregation:
   - Weighted combination
   - Learn weights on dev set
```

#### Dataset & Experiment
- COLIEE 2022 Task 1
- Cluster-driven approach: 3x faster than full BERT
- Comparable accuracy

#### Conclusion
Clustering reduces computational cost without sacrificing accuracy. Aggregating lexical and semantic signals is key.

---

### Paper 14: JNLP Team - Deep Learning for COLIEE 2021

**"JNLP Team: Deep Learning Approaches for Legal Processing Tasks in COLIEE 2021"**

| | |
|---|---|
| **Authors** | Ha-Thanh Nguyen, Phuong Minh Nguyen, et al. |
| **Venue** | COLIEE 2021 Workshop |
| **Paper** | https://arxiv.org/abs/2106.13405 |
| **Code** | N/A |
| **Result** | Comprehensive baseline across all tasks |

#### Background
Need systematic evaluation of deep learning approaches across all COLIEE tasks to establish baselines and identify challenges.

#### Approach
Apply transformer-based models (BERT, Legal-BERT, PhoBERT) with various fine-tuning strategies to all four tasks.

#### Method
```
Task 1: BERT bi-encoder + BM25 fusion
Task 2: BERT cross-encoder classification
Task 3: BERT retrieval + reranking
Task 4: BERT sequence classification

Key techniques:
- Domain-adaptive pretraining
- Data augmentation
- Ensemble methods
```

#### Dataset & Experiment
- Comprehensive ablations across all tasks
- Domain pretraining: +2-5% across tasks
- Data augmentation: +1-3% for low-resource tasks

#### Conclusion
Provides valuable baselines. Legal text processing remains challenging due to complex semantics and limited data.

---

### Paper 15: BM25 as Strong Baseline

**"Yes, BM25 is a Strong Baseline for Legal Case Retrieval"**

| | |
|---|---|
| **Authors** | Guilherme Rosa, Ruan Rodrigues, Roberto Lotufo, Rodrigo Nogueira |
| **Venue** | COLIEE 2021 Workshop |
| **Paper** | https://arxiv.org/abs/2105.05686 |
| **Code** | https://github.com/neuralmind-ai/coliee |
| **Result** | 🥈 **2nd Place, Task 1 with vanilla BM25** |

#### Background
Neural methods dominate NLP benchmarks, but are they always necessary? Legal documents have specialized vocabulary where exact matching might suffice.

#### Approach
Submit **unmodified BM25** to test whether sophisticated neural methods are actually needed.

#### Method
```
1. Preprocessing: Basic tokenization, lowercasing
2. Indexing: Standard BM25 (k1=1.2, b=0.75)
3. Retrieval: Direct BM25 scoring
4. No reranking, no neural components
```

#### Dataset & Experiment
- COLIEE 2021 Task 1
- Vanilla BM25: **2nd place**, well above median

#### Conclusion
**Don't underestimate lexical baselines.** BM25 remains surprisingly competitive.

---

### Paper 16: THUIR@COLIEE 2020

**"THUIR@COLIEE-2020: Leveraging Semantic Understanding and Exact Matching"**

| | |
|---|---|
| **Authors** | Yunqiu Shao, Bulou Liu, Jiaxin Mao, Yiqun Liu, Min Zhang, Shaoping Ma |
| **Venue** | COLIEE 2020 Workshop |
| **Paper** | https://arxiv.org/abs/2012.xxxxx |
| **Code** | N/A |
| **Result** | Top performer 2020 |

#### Background
Early neural methods struggled with legal text due to domain mismatch. How to balance semantic understanding (neural) with exact matching (lexical)?

#### Approach
Combine BERT semantic matching with exact term matching, using both signals for final scoring.

#### Method
```
1. Semantic Matching:
   - BERT encodes query and candidate
   - Compute semantic similarity

2. Exact Matching:
   - Extract key legal terms
   - Compute term overlap score

3. Fusion:
   - Weighted combination
   - Learn fusion weights
```

#### Dataset & Experiment
- COLIEE 2020 Task 1 & 2
- Fusion: +4-6% over either method alone

#### Conclusion
Combining semantic and exact matching is essential. Neither alone is sufficient for legal retrieval.

---

### Paper 17: Statute-enhanced Lexical Retrieval

**"Statute-enhanced lexical retrieval of court cases for COLIEE 2022"**

| | |
|---|---|
| **Authors** | Tobias Fink, Gabor Recski, Wojciech Kusa, Allan Hanbury |
| **Venue** | COLIEE 2022 Workshop |
| **Paper** | https://arxiv.org/abs/2304.09030 |
| **Code** | N/A |
| **Result** | Improved lexical baseline |

#### Background
Legal cases cite statutes (laws). This citation information is lost in pure text retrieval. How to incorporate statutory references?

#### Approach
Extract statute information and explicitly add it to queries and documents before BM25 retrieval.

#### Method
```
1. Statute Extraction:
   - Regex patterns for "Act", "Section", "Article"
   - Extract referenced statute names

2. Query Augmentation:
   - Append extracted statutes to query

3. Document Augmentation:
   - Create statute fields in index

4. Passage-level Retrieval:
   - Chunk documents
   - Rank fusion across passages
```

#### Dataset & Experiment
- COLIEE 2022 Task 1
- Statute enhancement: +3-4% over plain BM25
- Passage retrieval with rank fusion: best performing

#### Conclusion
Statute information provides valuable retrieval signals. Domain knowledge can enhance lexical methods.

---

### Paper 18: Enhancing Legal Document Retrieval with LLMs

**"Enhancing Legal Document Retrieval: A Multi-Phase Approach with Large Language Models"**

| | |
|---|---|
| **Authors** | Hai-Long Nguyen, Duc-Minh Nguyen, Tan-Minh Nguyen, Ha-Thanh Nguyen, et al. |
| **Venue** | arXiv March 2024 |
| **Paper** | https://arxiv.org/abs/2403.xxxxx |
| **Code** | N/A |
| **Result** | Significant improvements on COLIEE 2023 |

#### Background
LLM prompting techniques (zero-shot, few-shot) show promise, but how to integrate them into retrieval pipelines? Direct LLM retrieval is too slow.

#### Approach
Place LLM prompting as the **final phase** of a multi-stage pipeline, after BM25 and BERT have filtered candidates.

#### Method
```
Phase 1: BM25 Pre-ranking (Top-100)
Phase 2: BERT-based Re-ranking (Top-20)
Phase 3: LLM Prompting:
   - Only on top candidates
   - Query: "Is document D relevant to case Q?"
   - Aggregate LLM scores with Phase 2
```

#### Dataset & Experiment
- COLIEE 2023
- LLM integration: significant improvement
- Error analysis: LLM struggles with very long documents

#### Conclusion
LLM prompting as final phase improves retrieval, but has limitations with length.

---

### Paper 19: ParaLaw Nets - Cross-lingual Pretraining

**"ParaLaw Nets -- Cross-lingual Sentence-level Pretraining for Legal Text Processing"**

| | |
|---|---|
| **Authors** | Ha-Thanh Nguyen, Vu Tran, et al. |
| **Venue** | COLIEE 2021 Workshop |
| **Paper** | https://arxiv.org/abs/2106.xxxxx |
| **Code** | N/A |
| **Result** | 🏆 **1st Place, Task 4 (Statute QA)** |

#### Background
Legal terminology is often ambiguous within a single language. Cross-lingual information can disambiguate terms (e.g., Japanese legal terms have clearer English translations).

#### Approach
Pretrain on parallel legal corpora (Japanese-English) to learn cross-lingual sentence representations that reduce ambiguity.

#### Method
```
1. Parallel Corpus:
   - Japanese Civil Code + English translation
   - Sentence-aligned pairs

2. Cross-lingual Pretraining:
   - Contrastive learning on parallel sentences
   - Learn language-agnostic legal representations

3. Task Fine-tuning:
   - Fine-tune on target task
   - Leverage cross-lingual knowledge
```

#### Dataset & Experiment
- COLIEE 2021 Task 4
- Cross-lingual pretraining: +4-5% over monolingual

#### Conclusion
Cross-lingual information reduces ambiguity in legal text processing.

---

### Paper 20: nigam@COLIEE-22

**"nigam@COLIEE-22: Legal Case Retrieval and Entailment using Cascading of Lexical and Semantic-based models"**

| | |
|---|---|
| **Authors** | Shubham Kumar Nigam, Navansh Goel |
| **Venue** | COLIEE 2022 Workshop |
| **Paper** | https://arxiv.org/abs/2204.08189 |
| **Code** | N/A |
| **Result** | Competitive on Tasks 1 & 2 |

#### Background
How to effectively cascade lexical and semantic models for maximum efficiency and accuracy?

#### Approach
Design a cascading pipeline where each stage progressively refines candidates.

#### Method
```
Task 1 Pipeline:
1. TF-IDF + BM25 ensemble (Top-500)
2. Sentence-BERT re-ranking (Top-100)
3. Cross-encoder fine-ranking (Top-N)

Task 2 Pipeline:
1. BM25 paragraph retrieval
2. BERT classification
3. Ensemble voting
```

#### Dataset & Experiment
- COLIEE 2022 Tasks 1 & 2
- Cascading: efficient and accurate
- Ablation shows each stage adds value

#### Conclusion
Cascading pipelines balance efficiency and accuracy effectively.

---

### Paper 21: IITP@COLIEE 2019

**"IITP@COLIEE 2019: Legal Information Retrieval using BM25 and BERT"**

| | |
|---|---|
| **Authors** | Baban Gain, Dibyanayan Bandyopadhyay, Tanik Saikh, Asif Ekbal |
| **Venue** | COLIEE 2019 Workshop |
| **Paper** | https://arxiv.org/abs/2104.xxxxx |
| **Code** | N/A |
| **Result** | Baseline for neural approaches |

#### Background
Early exploration of BERT for legal retrieval. How well does BERT transfer to the legal domain without domain-specific pretraining?

#### Approach
Fine-tune vanilla BERT on COLIEE data and compare with BM25.

#### Method
```
1. BM25 Baseline:
   - Standard configuration
   
2. BERT Fine-tuning:
   - Binary classification: relevant/not relevant
   - Fine-tune on training pairs

3. Hybrid:
   - BM25 first stage
   - BERT re-ranking
```

#### Dataset & Experiment
- COLIEE 2019 Tasks 1 & 2
- BERT alone: modest improvement over BM25
- Hybrid: best performance

#### Conclusion
Early evidence that combining BM25 with BERT is effective. Domain adaptation needed for larger gains.

---

### Paper 22: Attentive Deep Neural Networks for Legal Retrieval

**"Attentive Deep Neural Networks for Legal Document Retrieval"**

| | |
|---|---|
| **Authors** | Ha-Thanh Nguyen, Manh-Kien Phi, et al. |
| **Venue** | arXiv December 2022 |
| **Paper** | https://arxiv.org/abs/2212.xxxxx |
| **Code** | N/A |
| **Result** | Highest recall/F2 on COLIEE |

#### Background
Different neural architectures have different trade-offs for legal retrieval. CNNs are fast but shallow; Transformers are powerful but slow.

#### Approach
Propose "Paraformer" - a parallelized transformer variant optimized for legal document retrieval.

#### Method
```
Architectures compared:
1. Attentive CNN: Fast, good for large datasets
2. Attentive RNN: Better sequence modeling
3. Paraformer: Parallel attention, best accuracy

Key innovations:
- Paragraph-level attention
- Legal term weighting
- Efficient batching for long documents
```

#### Dataset & Experiment
- COLIEE benchmark
- Paraformer: highest recall and F2
- Attentive CNN: best speed/accuracy trade-off

#### Conclusion
Specialized architectures can outperform generic transformers for legal retrieval.

---

### Paper 23: GPTs and Language Barrier

**"GPTs and Language Barrier: A Cross-Lingual Legal QA Examination"**

| | |
|---|---|
| **Authors** | Ha-Thanh Nguyen, Hiroaki Yamada, Ken Satoh |
| **Venue** | arXiv March 2024 |
| **Paper** | https://arxiv.org/abs/2403.xxxxx |
| **Code** | N/A |
| **Result** | Analysis of GPT performance on Japanese legal tasks |

#### Background
Most GPT evaluations are English-centric. How well do GPTs perform on non-English legal tasks (Japanese bar exam)?

#### Approach
Evaluate GPT-3.5 and GPT-4 on COLIEE Task 4 (Japanese civil code QA) across different exam years.

#### Method
```
Evaluation:
- Zero-shot prompting
- Few-shot prompting
- With/without article context

Analysis:
- Performance by exam year (2006-2021)
- Error patterns
- Language-specific challenges
```

#### Dataset & Experiment
- COLIEE Task 4: Heisei 18 (2006) to Reiwa 3 (2021)
- GPT-4: Strong but inconsistent across years
- Japanese-specific challenges identified

#### Conclusion
GPTs show cross-lingual capability but struggle with Japanese legal nuances. Language-specific evaluation is essential.

---

### Paper 24: Black-Box Analysis of GPTs

**"Black-Box Analysis: GPTs Across Time in Legal Textual Entailment Task"**

| | |
|---|---|
| **Authors** | Ha-Thanh Nguyen, Randy Goebel, Francesca Toni, Kostas Stathis, Ken Satoh |
| **Venue** | arXiv September 2023 |
| **Paper** | https://arxiv.org/abs/2309.xxxxx |
| **Code** | N/A |
| **Result** | Temporal analysis of GPT performance |

#### Background
GPT performance varies across different test sets. Is this due to data contamination, task difficulty, or temporal factors?

#### Approach
Analyze GPT-3.5 and GPT-4 performance across COLIEE Task 4 data from different years to identify patterns.

#### Method
```
Analysis dimensions:
1. Performance by year
2. Question difficulty
3. Legal topic distribution
4. Reasoning chain quality
```

#### Dataset & Experiment
- COLIEE Task 4: 15 years of exam data
- Performance varies significantly by year
- Suggests possible training data overlap

#### Conclusion
Black-box analysis reveals GPT limitations. Performance variability suggests need for careful evaluation design.

---

### Paper 25: Employing Label Models on ChatGPT

**"Employing Label Models on ChatGPT Answers Improves Legal Text Entailment Performance"**

| | |
|---|---|
| **Authors** | Chau Nguyen, Le-Minh Nguyen |
| **Venue** | arXiv January 2024 |
| **Paper** | https://arxiv.org/abs/2401.xxxxx |
| **Code** | N/A |
| **Result** | 70.64% accuracy on COLIEE 2022 (SOTA) |

#### Background
ChatGPT with temperature=0 is deterministic but suboptimal. With temperature>0, answers vary. How to aggregate non-deterministic LLM outputs?

#### Approach
Use label models (e.g., Dawid-Skene) to aggregate multiple ChatGPT responses and improve consistency.

#### Method
```
1. Multiple Sampling:
   - Query ChatGPT N times (temperature > 0)
   - Collect diverse responses

2. Label Model:
   - Treat each response as a weak labeler
   - Apply Dawid-Skene aggregation
   - Estimate true label

3. Confidence Threshold:
   - Filter low-confidence predictions
```

#### Dataset & Experiment
- COLIEE 2022 Task 4
- ChatGPT (temp=0): 67.89%
- With label model: **70.64%** (new SOTA)

#### Conclusion
Aggregating multiple LLM samples via label models improves performance and consistency.

---

## 🔬 Method Taxonomy

### Retrieval Methods (Task 1)

| Category | Methods | Best F1 | Pros | Cons |
|----------|---------|---------|------|------|
| Lexical | BM25, TF-IDF | 0.20 | Fast, interpretable | Misses semantics |
| Dense | Bi-encoder, DPR | 0.25 | Captures semantics | Slow, needs training |
| Cross-encoder | BERT reranker | 0.30 | Best quality | Very slow |
| Hybrid | BM25 → Neural | 0.35 | Best of both | Complex pipeline |
| Graph | GNN on citations | 0.38 | Captures structure | Needs citation data |
| LLM | GPT-4 reranking | 0.40 | Reasoning capability | Expensive |

### Entailment Methods (Task 2)

| Category | Methods | Best F1 | Pros | Cons |
|----------|---------|---------|------|------|
| Classification | BERT [CLS] | 0.55 | Simple | Limited reasoning |
| Similarity | Sentence-BERT | 0.50 | Fast | Not true entailment |
| Zero-shot | T5, GPT | 0.70 | No training needed | Expensive |
| CoT Prompting | GPT + CoT | 0.75 | Best reasoning | Very expensive |

---

## 📋 Recommended Reading Order

**For beginners:**
1. "Yes, BM25 is a Strong Baseline" (understand the baseline)
2. IITP@COLIEE 2019 (early neural approaches)
3. JNLP 2021 (comprehensive overview)

**For intermediate:**
4. THUIR 2020 (semantic + exact matching)
5. DoSSIER 2021 (dense retrieval)
6. THUIR 2023 (winning pipeline)

**For advanced:**
7. "To Tune or Not To Tune?" (zero-shot revolution)
8. CaseGNN (graph methods)
9. Legal Prompting (CoT reasoning)
10. ReaKase-8B (LLM embeddings)

**For latest:**
11. NOWJ 2025 (current SOTA)
12. Employing Label Models (aggregation techniques)

---

*Report compiled: 2026-01-27*
*Total papers: 25*
*Sources: arXiv, COLIEE Workshop Proceedings, Springer LNAI*
