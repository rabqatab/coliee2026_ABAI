# COLIEE Competition: Previous Approaches & Performance Report

---

## 📌 Executive Summary

### Key Takeaways

1. **Hybrid pipelines dominate**: BM25 first-stage + neural/feature-based reranking consistently outperforms pure approaches
2. **Feature engineering > end-to-end DL** (Task 1): JNLP won 2025 with BM25+SAILER+LightGBM, TQM won 2024 with LTR fusion — hand-crafted features + gradient boosting beat pure neural
3. **Structure matters**: SAILER (structure-aware pre-training), paragraph-level encoding, and citation contexts all provide strong signal
4. **Ensemble is king**: No single model wins; top teams use ensembles (NOWJ's LLM voting, TQM's LTR fusion, OVGU's model chaining)
5. **Graph methods emerging**: CaseLink GNN achieved consistent 2nd place Task 1 in 2025 (F1=0.2962)
6. **Citation context is a signal**: IIT Bhilai (2025) and UMNLP's "propositions" (2024) independently showed that text around citations captures *why* cases are cited
7. **Open-source models competitive**: DeepSeek-V3, QwQ-32B, Qwen2-72B, LLaMA-3 power all top 2025 systems (closed-source banned since 2025)

### Performance Milestones

| Task | 2019 | 2021 | 2023 | 2024 | 2025 | Key Breakthrough |
|------|------|------|------|------|------|------------------|
| Task 1 (Retrieval) | 0.19 | 0.29 | 0.35 | **0.44** | 0.34 | LTR fusion (2024), SAILER+LightGBM (2025) |
| Task 2 (Entailment) | 0.52 | 0.68 | 0.71 | **0.65** | 0.32 | monoT5+hard negatives (2024), LLM voting (2025) |

*Note: 2025 scores dropped significantly vs. 2024 — likely due to harder/larger test sets, not method regression. The 2025 corpus expanded to ~7,708 documents with 2,159 test queries (vs. 400 in 2024).*

### Notable Findings

- ⭐ **THUIR** (Tsinghua): Dominated Task 1 for 4 consecutive years (2020-2023)
- ⭐ **Unicamp/NeuralMind**: Proved zero-shot T5-11B beats all fine-tuned models (2021)
- ⭐ **CaseGNN/CaseLink**: GNNs for legal case retrieval — 2nd place Task 1 in 2025 (F1=0.2962)
- ⭐ **TQM** (Tsinghua): LTR fusion of lexical+semantic+simple features won Task 1 in 2024 (F1=0.4432)
- ⭐ **JNLP**: BM25+SAILER+LightGBM won Task 1 in 2025 (F1=0.3353) — feature engineering beats deep learning
- ⭐ **UMNLP**: Novel "propositions" features + judge matching — 2nd place 2024 (F1=0.4134)
- ⭐ **IIT Bhilai**: Citation-neighbourhood retrieval — text surrounding citation markers as features (2025)
- ⭐ **ReaKase-8B**: LLM-based embeddings with reasoning augmentation (2025)
- ⭐ **"Legal Prompting"**: Chain-of-thought significantly improves legal reasoning
- ⭐ **U-CREAT**: Unsupervised event extraction for cross-system retrieval (2023)
- ⭐ **KIS**: 130M parameter model beat 70B model for rationale extraction (2025)

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

### Paper 11: NOWJ@COLIEE 2025 — Task 2 Winner

**"NOWJ@COLIEE 2025: A Multi-stage Framework Integrating Embedding Models and Large Language Models for Legal Retrieval and Entailment"**

| | |
|---|---|
| **Authors** | Hoang-Trung Nguyen, Tan-Minh Nguyen, Xuan-Bach Le, Tuan-Kiet Le, Khanh-Huyen Nguyen, Ha-Thanh Nguyen, Thi-Hai-Yen Vuong, Le-Minh Nguyen |
| **Venue** | COLIEE 2025 Workshop (ICAIL 2025) |
| **Paper** | https://arxiv.org/abs/2509.08025 |
| **Code** | N/A |
| **Result** | 🏆 **1st Place, Task 2 (F1: 0.3195)** |

#### Background
Previous entailment approaches relied on single models (monoT5). COLIEE 2025 shifted to open-source-only models, requiring new strategies.

#### Approach
Two-stage system combining lexical-semantic filtering with contextualized LLM analysis and multi-model voting.

#### Method
```
Task 2 Pipeline (winning run):
1. BM25: Retrieve top-35 candidate paragraphs
2. LLM Reranking: DeepSeek-V3 and Qwen/QwQ-32B score candidates
3. Entailment Classification: LLM-based binary prediction
4. Majority Voting: Ensemble across multiple LLM outputs

Task 1 Pipeline (4th place, F1=0.1984):
1. Pre-processing + LLM-based summarization
2. Pre-ranking: BM25, BERT, monoT5
3. Embeddings: BGE-m3, LLM2Vec
4. Re-ranking: Qwen-2, QwQ-32B, DeepSeek-V3
```

#### Dataset & Experiment
- COLIEE 2025 all tasks
- Task 2: F1 = 0.3195 (1st place) — all three NOWJ runs in top positions
- Task 1: F1 = 0.1984 (5th place by team)
- Key insight: majority voting across diverse LLMs significantly boosts performance

#### Conclusion
LLM ensemble voting outperforms single-model approaches for entailment. Open-source LLMs (DeepSeek-V3, QwQ-32B) are competitive for legal reasoning when combined via voting.

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

### Paper 26: TQM@COLIEE 2024 — Task 1 Winner

**"Towards an In-Depth Comprehension of Case Relevance for Better Legal Retrieval"**

| | |
|---|---|
| **Authors** | Haitao Li, You Chen, Zhekai Ge, Qingyao Ai, Yiqun Liu, Quan Zhou, Shuai Huo |
| **Venue** | COLIEE 2024 Workshop (JSAI-isAI 2024), Springer LNCS 14741 |
| **Paper** | https://arxiv.org/abs/2404.00947 |
| **Code** | N/A |
| **Result** | 🏆 **1st Place, Task 1 (F1: 0.4432)** |

#### Background
Prior THUIR work focused on structural encoding (paragraph-level BERT). The TQM team extended this with deeper relevance features and learning-to-rank fusion.

#### Approach
Combine lexical matching, semantic retrieval, and simple case-level features via a learning-to-rank framework with heuristic post-processing.

#### Method
```
1. Pre-processing: Noise elimination from raw case texts
2. Lexical matching: BM25, TF-IDF with enhanced relevance features
3. Semantic retrieval: Dense vector models
4. Feature fusion: Learning-to-rank (LTR) combining all signals + simple features (case length, etc.)
5. Post-processing: Heuristic strategies based on common properties of relevant cases
```

#### Dataset & Experiment
- COLIEE 2024 Task 1: 400 test queries, 7,350 corpus documents
- F1 = 0.4432 (1st), Precision = 0.5057, Recall = 0.3944
- Notably high precision — post-processing heuristics helped reduce false positives

#### Conclusion
LTR fusion of diverse feature types sets new SOTA. Post-processing heuristics based on domain knowledge provide meaningful gains. **This validates the meta-learner approach in our pipeline design.**

---

### Paper 27: UMNLP@COLIEE 2024 — "Propositions" Features

**"Similarity Ranking of Case Law Using Propositions as Features"**

| | |
|---|---|
| **Authors** | Damian Curran, Mike Conway |
| **Venue** | COLIEE 2024 Workshop, Springer LNCS 14741 |
| **Paper** | https://dl.acm.org/doi/10.1007/978-981-97-3076-6_11 |
| **Code** | https://github.com/dc435/COLIEE_2024_Task1 |
| **Result** | 🥈 **2nd Place, Task 1 (F1: 0.4134)** |

#### Background
Standard retrieval features (BM25, dense similarity) miss nuanced legal relationships. What if we could capture *why* a case was cited?

#### Approach
Introduce "propositions" — short summaries of the basis on which a case was noticed — as novel retrieval features, combined with judge matching and quotation extraction.

#### Method
```
1. Feature Extraction:
   - Propositions: short summaries of citation basis
   - Judge name matching (same judge → higher relevance)
   - Verbatim quotation extraction
   - Paragraph/sentence-level similarity features
2. Classification: Feed-forward neural network for binary query-candidate classification
3. Threshold: Optimized cutoff on dev set
```

#### Dataset & Experiment
- COLIEE 2024 Task 1
- F1 = 0.4134 (2nd), Precision = 0.4000, Recall = 0.4277
- "Propositions" feature contributed significantly; judge matching added modest gains
- Code available on GitHub for reproducibility

#### Conclusion
Novel domain-specific features (propositions, judge matching) provide strong signal for legal retrieval. **The "propositions" concept is closely related to our citation context approach — text around `<FRAGMENT_SUPPRESSED>` markers captures similar information.**

---

### Paper 28: AMHR@COLIEE 2024 — Task 2 Winner (monoT5 + Hard Negatives)

**"AMHR COLIEE 2024 Entry: Legal Entailment and Retrieval"**

| | |
|---|---|
| **Authors** | Animesh Nighojkar, Kenneth Jiang, Logan Fields, Onur Bilgin, Stephen Steinle, Yernar Sadybekov, Zaid Marji, John Licato |
| **Venue** | COLIEE 2024 Workshop, Springer LNCS 14741 |
| **Paper** | https://link.springer.com/chapter/10.1007/978-981-97-3076-6_14 |
| **Code** | N/A |
| **Result** | 🏆 **1st Place, Task 2 (F1: 0.6512)** |

#### Background
Task 2 requires identifying which paragraph entails a legal conclusion. Previous methods used standard fine-tuning without careful negative sampling.

#### Approach
Fine-tune monoT5 with hard negative mining (BM25-retrieved non-relevant paragraphs) and a score-ratio threshold for prediction count.

#### Method
```
1. Base Model: monoT5 pre-trained on MS-MARCO
2. Hard Negative Mining:
   - Use BM25 to retrieve similar-but-irrelevant paragraphs
   - Use another monoT5 version to find hard negatives
   - Train on these challenging examples
3. Prediction:
   - Score all candidate paragraphs
   - If score ratio (top1/top2) < 6.619: predict top-2
   - Otherwise: predict top-1 only
   - Threshold = 6.619 determined via grid search
```

#### Dataset & Experiment
- COLIEE 2024 Task 2: 725 train cases, 100 test cases
- F1 = 0.6512, Precision = 0.6364, Recall = 0.6667
- Hard negative mining: significant improvement over standard training
- Score-ratio threshold: elegant solution for variable prediction count

#### Conclusion
Hard negative mining is critical for entailment tasks. The score-ratio threshold is a simple but effective heuristic for deciding prediction count.

---

### Paper 29: JNLP@COLIEE 2025 — Task 1 Winner (BM25 + SAILER + LightGBM)

**"Hybrid Large Language Model-based Framework for Legal Information Retrieval and Entailment"**

| | |
|---|---|
| **Authors** | Hai Nguyen, Hiep Nguyen, Trang Pham, Minh Nguyen, An Trieu, Dinh-Truong Do, Nguyen-Khang Le, Le-Minh Nguyen |
| **Venue** | COLIEE 2025 Workshop (ICAIL 2025) |
| **Paper** | https://coliee.org/documents/Proceedings/2025-Proceedings.pdf |
| **Code** | N/A |
| **Result** | 🏆 **1st Place, Task 1 (F1: 0.3353)**, 🏆 **1st Place, Task 3 (F2: 0.836)** |

#### Background
JNLP built on the UMNLP 2024 framework (propositions/features + NN classifier), extending it with BM25 and SAILER scores fed into LightGBM.

#### Approach
Feature-based gradient boosting (LightGBM) combining lexical (BM25), semantic (SAILER), and structural features — a machine learning approach rather than end-to-end deep learning.

#### Method
```
Task 1 Pipeline:
1. Stage 1 - BM25 Filtering:
   - Retrieve top 100-200 candidates per query
   - Achieves 76-85% recall in <30 seconds over full corpus
2. Stage 2 - Feature Extraction:
   - BM25 scores (full document + paragraph-level)
   - QLD (Query Likelihood with Dirichlet) scores
   - SAILER scores (structure-aware pre-trained model)
   - Dynamic scoring based on top-k retrieved docs
3. Stage 3 - LightGBM Re-ranking:
   - Gradient Boosting Decision Tree trained on relevance labels
   - Gradient-based one-side sampling for efficiency
   - Exclusive feature bundling for sparse features
```

#### Dataset & Experiment
- COLIEE 2025 Task 1: 2,159 test queries, ~7,350 corpus
- F1 = 0.3353 (1st), Precision = 0.3042, Recall = 0.3735
- SAILER scores contributed the most novel signal
- LightGBM chosen for speed and tabular data performance

#### Conclusion
**Feature engineering + gradient boosting beats end-to-end deep learning for legal retrieval.** SAILER (structure-aware pre-training) provides valuable semantic features. This directly validates our Option C pipeline design (BM25 → feature extraction → LightGBM meta-learner).

---

### Paper 30: UQLegalAI/CaseLink@COLIEE 2025 — GNN for Legal Retrieval

**"UQLegalAI@COLIEE2025: Advancing Legal Case Retrieval with Large Language Models and Graph Neural Networks"**

| | |
|---|---|
| **Authors** | Yanran Tang, Ruihong Qiu, et al. |
| **Venue** | COLIEE 2025 Workshop (ICAIL 2025) |
| **Paper** | https://arxiv.org/abs/2505.20743 |
| **Code** | https://github.com/yanran-tang/CaseLink |
| **Result** | 🥈 **2nd Place, Task 1 (F1: 0.2962)** |

#### Background
CaseLink extends CaseGNN with inductive graph learning, enabling predictions on unseen cases without retraining the graph.

#### Approach
Build Global Case Graphs (GCG) exploiting Case-Case, Case-Charge, and Charge-Charge relationships, then use GNNs to learn structure-aware representations.

#### Method
```
1. Graph Construction:
   - Global Case Graphs from training + test data
   - Case-Case edges (citation relationships)
   - Case-Charge edges (legal charge associations)
   - Charge-Charge edges (charge co-occurrence)
2. Node Features:
   - LLM text embeddings as dense node features
3. GNN Training:
   - InfoNCE contrastive loss
   - Degree regularization to handle hub nodes
4. Inference:
   - Inductive: handles unseen test cases
   - Three runs: F1 = 0.2940, 0.2950, 0.2962
```

#### Dataset & Experiment
- COLIEE 2025 Task 1
- Consistent 2nd place across all three submitted runs
- Captures inter-case connectivity that text-only methods miss

#### Conclusion
GNN-based retrieval captures structural signals absent from text-only methods. CaseLink's inductive approach is practical for competition settings. **Relevant to our GraphRAG Lite design — community structure provides similar inter-case signals.**

---

### Paper 31: OVGU@COLIEE 2025 — Silver Data Fine-Tuning

**"LLMs, Knowledge Graphs, and Hybrid Search: Task-Specific Approaches to Legal AI in COLIEE"**

| | |
|---|---|
| **Authors** | Sabine Wehnert et al. |
| **Venue** | COLIEE 2025 Workshop (ICAIL 2025) |
| **Paper** | https://www.researchgate.net/publication/394930419 |
| **Code** | N/A |
| **Result** | 🥈 **2nd Place, Task 2 (F1: 0.2454)** |

#### Background
Limited labeled data constrains fine-tuning. Can LLM-generated labels ("silver data") substitute for human annotations?

#### Approach
Create silver datasets by prompting nine LLMs on training data, then fine-tune smaller models on the aggregated pseudo-labels.

#### Method
```
Task 2:
1. Silver Dataset Creation:
   - Prompt 9 LLMs with problem type definitions
   - Models predict entailment labels + reasoning
   - Aggregate into silver training data
2. Fine-Tuning:
   - Phi-3-medium-4k-instruct
   - gemma-1.1-7b-it
3. Ensemble:
   - Majority voting across model predictions

Task 1:
- BM25Plus + case summaries (phi4)
- LLM ensemble filtering (gemma3:12b, wizardlm2:7b, phi4, deepseek-r1:32b)
```

#### Dataset & Experiment
- COLIEE 2025 Task 2: F1 = 0.2454 (2nd place)
- Silver data fine-tuning outperformed zero-shot approaches
- Resource-efficient: competed with limited compute

#### Conclusion
LLM-generated silver data is a practical strategy for data augmentation when labeled data is scarce.

---

### Paper 32: IIT Bhilai — Citation-Neighbourhood Retrieval

**"Knowledge-Based Legal Case Retrieval"**

| | |
|---|---|
| **Authors** | Chetana et al. |
| **Venue** | COLIEE 2025 |
| **Paper** | N/A |
| **Code** | https://github.com/chetaniitbhilai/Knowledge-Based-Legal-Case-Retrieval |
| **Result** | Participated in Task 1 |

#### Background
Most retrieval methods use full document text. But citation markers in legal cases indicate *where* citations occur — the surrounding text reveals *why* a case was cited.

#### Approach
Represent cases using compact textual segments surrounding citation markers, compute similarity-based features on filtered sets.

#### Method
```
1. Citation Neighbourhood Extraction:
   - Identify citation markers in case text
   - Extract compact text segments around each marker
   - Build case representation from citation contexts
2. Feature Computation:
   - Similarity-based features on filtered candidate set
3. Classification:
   - Ensemble: MLP + Random Forest
   - Combined relevance scoring
```

#### Dataset & Experiment
- COLIEE 2025 Task 1
- Exact scores not published in proceedings overview
- Demonstrated the viability of citation-neighbourhood as a retrieval signal

#### Conclusion
**Citation context is an independent signal for legal retrieval.** This directly validates our citation context window approach. The IIT Bhilai team independently discovered that text surrounding citation markers contains valuable retrieval information.

---

### Paper 33: KIS@COLIEE 2025 — Small Model Wins

**"KIS: COLIEE 2025 Task 4 Solver Using Japanese LLM"**

| | |
|---|---|
| **Authors** | Masaki Fujita et al. |
| **Venue** | COLIEE 2025 Workshop (ICAIL 2025) |
| **Paper** | N/A |
| **Code** | N/A |
| **Result** | 🏆 **1st Place, Task 4 (Acc: 0.9041)**, 🏆 **1st Place, Pilot RE (F1: 0.712)** |

#### Background
Large models (70B+) are assumed necessary for legal reasoning. Is this always true?

#### Approach
Balanced few-shot prompting with a LLaMA 3.1-based LLM for Task 4; fine-tuned modernbert-ja-130m (130M params) for rationale extraction.

#### Method
```
Task 4:
- LLaMA 3.1-based LLM
- Balanced few-shot prompting (equal yes/no examples)
- Straightforward but carefully designed prompts

Pilot (Rationale Extraction):
- modernbert-ja-130m (130M parameters)
- Fine-tuned on rationale extraction task
- Beat CAPTAIN's 70B model (0.712 vs lower)
```

#### Dataset & Experiment
- Task 4: 90/109 correct = 0.9041 accuracy (1st place)
- Pilot RE: F1 = 0.712 (1st place)
- The 130M model beat the 70B model for rationale extraction

#### Conclusion
**Task-specific fine-tuning of small models can outperform much larger models.** Careful prompt design and balanced sampling matter more than raw parameter count for structured legal tasks.

---

## 🔬 Method Taxonomy

### Retrieval Methods (Task 1)

| Category | Methods | Best F1 | Year | Team | Pros | Cons |
|----------|---------|---------|------|------|------|------|
| Lexical | BM25, TF-IDF | 0.19 | 2021 | NeuralMind | Fast, interpretable | Misses semantics |
| Dense | Bi-encoder, DPR | 0.22 | 2025 | AIIR Lab | Captures semantics | Slow, needs training |
| Structural | Paragraph BERT | 0.35 | 2023 | THUIR | Handles long docs | Training-intensive |
| GNN | CaseLink, CaseGNN | 0.30 | 2025 | UQLegalAI | Captures graph structure | Needs citation data |
| LTR Fusion | BM25+Dense+Features | **0.44** | 2024 | TQM | Best of all worlds | Complex pipeline |
| Feature+GBDT | BM25+SAILER+LightGBM | 0.34 | 2025 | JNLP | Efficient, interpretable | Feature engineering effort |
| Propositions | Features+NN | 0.41 | 2024 | UMNLP | Novel signals | Domain-specific features |
| LLM Ensemble | BM25+Multi-LLM | 0.20 | 2025 | NOWJ | Reasoning capability | Slow, expensive |

### Entailment Methods (Task 2)

| Category | Methods | Best F1 | Year | Team | Pros | Cons |
|----------|---------|---------|------|------|------|------|
| Classification | BERT [CLS] | 0.55 | 2023 | Various | Simple | Limited reasoning |
| monoT5 | T5 reranker + hard neg | **0.65** | 2024 | AMHR | Strong performance | Requires MS-MARCO pretraining |
| Chained Models | Legal-BERT chain | 0.60 | 2024 | OVGU | Robust | Complex setup |
| Zero-shot | T5-11B | 0.68 | 2021 | NeuralMind | No training needed | Large model |
| LLM Voting | Multi-LLM ensemble | 0.32 | 2025 | NOWJ | Best open-source | High compute |
| Silver Data | LLM pseudo-labels | 0.25 | 2025 | OVGU | Data-efficient | Noisy labels |

*Note: 2024 and 2025 Task 2 scores are not directly comparable due to different test sets. The 2025 test set was significantly harder.*

---

## 📋 Recommended Reading Order

**For beginners:**
1. "Yes, BM25 is a Strong Baseline" (Paper 15 — understand the baseline)
2. IITP@COLIEE 2019 (Paper 21 — early neural approaches)
3. JNLP 2021 (Paper 14 — comprehensive overview)

**For intermediate:**
4. THUIR 2023 (Paper 1 — winning structural pipeline)
5. TQM 2024 (Paper 26 — LTR fusion, Task 1 winner)
6. AMHR 2024 (Paper 28 — monoT5 + hard negatives, Task 2 winner)

**For advanced:**
7. "To Tune or Not To Tune?" (Paper 3 — zero-shot revolution)
8. CaseLink/CaseGNN (Papers 5, 30 — graph methods evolution)
9. JNLP 2025 (Paper 29 — **current Task 1 SOTA**, BM25+SAILER+LightGBM)
10. UMNLP 2024 (Paper 27 — novel "propositions" features)

**For our pipeline design:**
11. IIT Bhilai 2025 (Paper 32 — citation-neighbourhood, validates our approach)
12. NOWJ 2025 (Paper 11 — LLM ensemble voting)
13. KIS 2025 (Paper 33 — small model can beat large)
14. OVGU 2025 (Paper 31 — silver data fine-tuning)

---

## 📊 Cross-Reference: Detailed Competition Results

For complete team-by-team results tables and all run scores, see:
- `docs/COLIEE_2024_RESULTS.md` — COLIEE 2024 (10 teams Task 1, 6 teams Task 2)
- `docs/COLIEE_2025_RESULTS.md` — COLIEE 2025 (8 teams Task 1, 6 teams Task 2)

---

*Report compiled: 2026-01-27, updated: 2026-03-11*
*Total papers: 33*
*Sources: arXiv, COLIEE Workshop Proceedings (ICAIL 2025, JSAI-isAI 2024), Springer LNAI, ResearchGate*
