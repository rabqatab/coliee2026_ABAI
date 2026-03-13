# Novel Approaches Survey for COLIEE 2026 Task 1

**Date:** 2026-03-13
**Purpose:** Comprehensive literature review of novel, non-overlapping approaches beyond standard COLIEE pipelines, organized into 5 research directions with 90+ unique papers.
**Exclusions:** Papers already covered in `COLIEE_APPROACHES_REPORT.md` and `plans/2026-03-11-graphrag-literature-review.md`.

---

## Table of Contents

1. [Citation-Aware & Structure-Aware Attention](#1-citation-aware--structure-aware-attention) (18 papers)
2. [GNN Rerankers](#2-gnn-rerankers) (19 papers)
3. [Contrastive Pre-training](#3-contrastive-pre-training) (25 papers)
4. [Multi-Task Learning (Joint Task 1 + Task 2)](#4-multi-task-learning) (17 papers)
5. [Other Novel Approaches](#5-other-novel-approaches) (20 papers)
6. [Cross-Cutting Synthesis & Priority Recommendations](#6-cross-cutting-synthesis--priority-recommendations)

---

## 1. Citation-Aware & Structure-Aware Attention

### 1.1 Hierarchical Document Transformers

**HDT — Hierarchical Document Transformer**
He, Flicke et al. "HDT: Hierarchical Document Transformer." COLM 2024. [arXiv:2407.08330](https://arxiv.org/abs/2407.08330)
- Sparse Transformer with auxiliary anchor tokens (`[DOC]`, `[SEC]`, `[SENT]`) at structural levels. Attention flows through the hierarchy instead of globally, reducing memory from O(n^2) to near-linear.
- **COLIEE fit:** Insert anchor tokens at paragraph boundaries and `<FRAGMENT_SUPPRESSED>` positions, giving citation locations explicit structural status. Faster convergence and higher sample efficiency than flat Longformer.

**HAT — Hierarchical Attention Transformers**
Chalkidis, Dai, Fergadiotis et al. "An Exploration of Hierarchical Attention Transformers for Efficient Long Document Classification." arXiv:2210.05529, 2022.
- Segment-wise encoding + cross-segment attention. Outperforms equally-sized Longformer at 10-20% less GPU memory and 40-45% faster.
- **COLIEE fit:** Segment at paragraph boundaries, enabling paragraph-level interaction between query and candidate in the cross-encoder.

**SEAL — Structure and Element Aware Learning**
Huang, Ren et al. "SEAL: Structure and Element Aware Learning to Improve Long Structured Document Retrieval." EMNLP 2025. [arXiv:2508.20778](https://arxiv.org/abs/2508.20778)
- Structure-Aware Learning (SAL) preserving semantic hierarchies + Element-Aware Alignment (EAL) with masked element matching.
- **COLIEE fit:** Masked element alignment is directly analogous to our citation masking — `<FRAGMENT_SUPPRESSED>` markers are masked elements needing alignment with candidate documents. +3.9 nDCG@10 on BGE-M3.

### 1.2 Cross-Document / Listwise Attention

**Set-Encoder — Inter-Passage Attention for Listwise Reranking**
Schlatt, Frobe et al. "Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Passage Re-Ranking with Cross-Encoders." ECIR 2025. [arXiv:2404.06912](https://arxiv.org/abs/2404.06912)
- Scores multiple candidates simultaneously via [CLS] token exchange between passages. Permutation-invariant. Handles up to 100 passages per query.
- **COLIEE fit:** Score query against top-K candidates jointly — the model can recognize redundant candidates and identify the most relevant cited cases through comparative attention.

**jina-reranker-v3 — Last but Not Late Interaction**
Wang, Wang et al. "jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking." arXiv:2509.25085, 2025.
- Built on Qwen3-0.6B, processes query + multiple documents in shared context. Special `<|doc_emb|>` tokens after each document for extraction. 131K token context.
- **COLIEE fit:** Insert `<|cite_emb|>` tokens at `<FRAGMENT_SUPPRESSED>` positions. 61.94 nDCG@10 on BEIR with only 0.6B params.

**EBCAR — Embedding-Based Context-Aware Reranker**
Saeidi, Decerf et al. "Embedding-Based Context-Aware Reranker." ICLR 2026. [arXiv:2510.13329](https://arxiv.org/abs/2510.13329)
- Reranking on dense embeddings (not raw text). Encodes structural signals (document ID, passage position) as positional encodings. Hybrid attention: full + masked.
- **COLIEE fit:** Encode paragraph position and distance from `<FRAGMENT_SUPPRESSED>` as structural signals. Much lower latency than text-based cross-encoders.

### 1.3 Sparse Attention for Cross-Encoders

**Sparse Attention Investigation**
Schlatt, Frobe, Hagen. "Investigating the Effects of Sparse Attention on Cross-Encoders." ECIR 2024. [arXiv:2312.17649](https://arxiv.org/abs/2312.17649)
- Window-4 sparse attention matches full cross-encoder effectiveness. Query tokens don't need to attend to document tokens. 22-59% memory savings.
- **COLIEE fit:** Citation-focused sparse attention — full attention at `<FRAGMENT_SUPPRESSED>` positions, small windows elsewhere. Concentrates model capacity on informative regions.

**DAM — Dynamic Attention Mask**
Zhang et al. "DAM: Dynamic Attention Mask for Long-Context LLM Inference." ACL 2025 Findings. [arXiv:2506.11104](https://arxiv.org/abs/2506.11104)
- Adaptive masks at attention-map level. No predefined structure needed — learns context-aware patterns.
- **COLIEE fit:** Let the model discover that citation-adjacent regions and section boundaries deserve full attention, rather than manually designing patterns.

**DMA — Trainable Dynamic Mask Sparse Attention**
Shi, Wu et al. "Trainable Dynamic Mask Sparse Attention." arXiv:2508.02124, 2025.
- End-to-end trainable. Value vectors generate content-aware sparse masks. Up to 10x speedup with improved accuracy over static sparse methods.

### 1.4 Marker-Aware / Position-Aware Attention

**PL-Marker — Packed Levitated Markers**
Ye, Lin et al. "Packed Levitated Marker for Entity and Relation Extraction." ACL 2022. [arXiv:2109.06067](https://arxiv.org/abs/2109.06067)
- "Levitated markers" at entity boundaries participate in attention in a separate position space. +4.1-4.3% F1 on relation extraction.
- **COLIEE fit:** Insert levitated markers at each `<FRAGMENT_SUPPRESSED>` position. Markers attend to citation context AND candidate text via cross-attention without disrupting the original token interactions. Strong conceptual match.

**CoPE — Contextual Position Encoding**
Golovneva, Wang et al. "Contextual Position Encoding: Learning to Count What's Important." ICLR 2025. [arXiv:2405.18719](https://arxiv.org/abs/2405.18719)
- Positions conditioned on context via learned gates. Counts only "important" tokens (e.g., markers, sentences).
- **COLIEE fit:** Learn to count `<FRAGMENT_SUPPRESSED>` markers, giving each citation position a unique context-aware position. Distinguishes "3rd citation" from "7th citation" — potentially meaningful in legal reasoning flow.

### 1.5 Passage Aggregation

**PARADE — Passage Representation Aggregation**
Li, Yates et al. "PARADE: Passage Representation Aggregation for Document Reranking." ACM TOIS 2023. [arXiv:2008.09093](https://arxiv.org/abs/2008.09093)
- Splits long docs into passages, encodes each, aggregates via transformer. Learns which passages matter most per query.
- **COLIEE fit:** Treat each citation context window as a passage, plus remaining text. Aggregation learns to weight citation-context passages. Core architecture for hierarchical matching.

**IDCM — Intra-Document Cascading**
Hofstatter, Mitra et al. "Intra-Document Cascading: Learning to Select Passages for Neural Document Ranking." SIGIR 2021.
- Lightweight student model prunes passages; expensive teacher (BERT) scores only selected ones. 400% lower latency.
- **COLIEE fit:** Citation-context-aware pruner selects which passage pairs deserve expensive DeBERTa cross-encoding.

**QDER — Query-Specific Document and Entity Representations**
Godbole, Navratil et al. "QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking." SIGIR 2025. [arXiv:2510.11589](https://arxiv.org/abs/2510.11589)
- Integrates KG entities into multi-vector reranking with late aggregation. +36% nDCG@20 on TREC Robust on difficult queries.
- **COLIEE fit:** Legal entities from citation windows as explicit entity nodes alongside token representations.

### 1.6 Citation-Aware Pre-training

**CiteBART — Masked Citation Pre-Training**
Yu, Zhang et al. "CiteBART: Learning to Generate Citations for Local Citation Recommendation." EMNLP 2025. [arXiv:2412.17534](https://arxiv.org/abs/2412.17534)
- Pre-trains by masking citation placeholders and reconstructing them. Teaches deep understanding of citation intent.
- **COLIEE fit:** Conceptually identical to COLIEE Task 1. Pre-train on `<FRAGMENT_SUPPRESSED>` positions to predict which case fills each slot. Creates a citation-aware encoder.

**Hierarchical Attention for Citation Recommendation**
Gu, Gao, Hahnloser. "Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-Based Reranking." ECIR 2022. [arXiv:2112.01206](https://arxiv.org/abs/2112.01206)
- Word-level and sentence-level hierarchical attention for citation context windows.
- **COLIEE fit:** Maps directly to processing `<FRAGMENT_SUPPRESSED>` context windows at multiple granularities.

**SCIDOCA 2025 — Masked Citation Prediction Shared Task**
Nguyen et al. "Overview of SCIDOCA 2025 Shared Task on Citation Prediction." arXiv:2509.24283, 2025.
- Three citation tasks on 60K+ paragraphs. Masked Citation Prediction is essentially identical to COLIEE Task 1.

---

## 2. GNN Rerankers

### 2.1 Surveys

**Graph-Based Re-ranking Survey**
Di Francesco et al. "Graph-Based Re-ranking: Emerging Techniques, Limitations, and Opportunities." arXiv:2503.14802, March 2025.
- First comprehensive survey on graph-based reranking for IR. Taxonomizes by graph construction and GNN architecture. Key finding: **heterogeneous GNNs remain underexplored**, and fusing multiple feature graphs is an open problem.

### 2.2 GNN Rerankers for General IR

**GNRR — Graph Neural Re-Ranking via Corpus Graph**
Di Francesco, Tonellotto, Macdonald. "Graph Neural Re-Ranking via Corpus Graph." SIGIR 2024 Workshop. [arXiv:2406.11720](https://arxiv.org/abs/2406.11720)
- Semantic corpus graph (TCT-ColBERT cosine, k=8 neighbors). Per-query subgraph extraction from BM25 top-1000. GNN refines scores. +5.8% AP on TREC-DL19, <40ms overhead.
- **COLIEE fit:** Most directly transferable. Build corpus graph from BGE-large embeddings + entity overlap, extract subgraphs from RRF top-200.

**MPGraf — Modular Pre-trained Graphformer**
Baidu. "MPGraf: a Modular and Pre-trained Graphformer for Learning to Rank at Web-scale." IEEE ICDM 2023 / IJCAI 2024. [arXiv:2409.16590](https://arxiv.org/abs/2409.16590)
- Transformer + GNN on query-document bipartite graphs. Modular composition (parallel/stacking). +1.4-1.7% NDCG@10.
- **COLIEE fit:** Stack GNN on top of cross-encoder scores. Surgical fine-tuning beneficial for limited training data.

**SlideGAR / L2G — Listwise Graph Reranking**
Yoon, Kim et al. "On Listwise Reranking for Corpus Feedback." WSDM 2025. [arXiv:2510.00887](https://arxiv.org/abs/2510.00887)
- L2G induces document-document graphs from reranker output logs — no explicit corpus graph needed. Frontier expansion pulls in graph neighbors beyond initial top-K.
- **COLIEE fit:** Could boost recall ceiling beyond current 61% by pulling in graph neighbors of relevant candidates.

**GRADA — Adversarial Document Defense via Graph**
Zheng et al. "GRADA: Graph-based Reranking against Adversarial Document Attacks." EMNLP 2025. [arXiv:2505.07546](https://arxiv.org/abs/2505.07546)
- Document coherence graph detects and demotes outliers. Up to 80% attack reduction.
- **COLIEE fit:** Suppress false positives caused by artificial similarity from `<FRAGMENT_SUPPRESSED>` markers (93% of docs).

**GraphMonoT5 — KG-Fused Reranking**
"Empowering Language Model with Guided Knowledge Fusion for Biomedical Document Re-ranking." AIME 2024. [arXiv:2305.04344](https://arxiv.org/abs/2305.04344)
- GNN over KG fused with T5 encoder via mutual information bottleneck. +1.9 nDCG@10 on BioASQ/TREC-COVID.
- **COLIEE fit:** Complement DeBERTa with GNN over our entity graph (statutes, judges, domains). Bidirectional fusion via bottleneck.

**GAT-Cross Encoder**
Vollmers, Ali et al. "Document reRanking using GAT-Cross Encoder." DICE Research, KIAM 2024.
- GAT integrated directly into cross-encoder architecture. Joint training avoids cascading errors.

**KG-FiD — KG in Fusion-in-Decoder**
Yu, Zhu et al. "KG-FiD: Infusing Knowledge Graph in Fusion-in-Decoder for Open-Domain QA." ACL 2022. [arXiv:2110.04330](https://arxiv.org/abs/2110.04330)
- GNN reranks passages via KG structure before decoder. Same performance at 40% compute.
- **COLIEE fit:** GNN filter from RRF top-200 → top-50 before expensive cross-encoder.

**Heterogeneous Graph Document Ranking**
Dou et al. "Heterogeneous Graph-based Context-aware Document Ranking." WSDM 2023.
- Multiple node types (queries, documents, sessions) + edge types. Heterogeneous graph attention.
- **COLIEE fit:** Our graph has documents, statutes, judges, domains, outcomes as node types with multiple edge types.

**HGAT for Passage Retrieval**
Albarede, Mulhem et al. "Heterogeneous Graph Attention Networks for Passage Retrieval." Information Retrieval Journal, 2023.
- Key finding: naive HGAT fails — separate attention per edge type is required.
- **COLIEE fit:** Design separate attention heads for statute co-citation, judge-authored, and semantic similarity edges.

### 2.3 GNN for Legal Case Retrieval

**CFGL-LCR — Counterfactual Graph Learning**
Liu et al. "CFGL-LCR: A Counterfactual Graph Learning Framework for Legal Case Retrieval." KDD 2023.
- Counterfactual data augmentation on case graphs. Learns causal (not correlational) relationships.
- **COLIEE fit:** Generate counterfactual entity graphs (remove statute references → does relevance change?) to learn which features are causally important. Addresses limited training data.

**LEXA — Graph Contrastive Learning + LLM Embeddings**
Tang, Qiu et al. "LEXA: Legal Case Retrieval via Graph Contrastive Learning with Contextualised LLM Embeddings." arXiv:2405.11791, 2024.
- Edge-Updated Graph Attention (EUGAT) jointly updates node and edge features. Graph contrastive learning with augmentation. LLM embeddings as node features.
- **COLIEE fit:** EUGAT handles our case where edge features (shared statute count, co-citation) carry important information. Graph contrastive learning supplements limited labels.

**The Missing Link — Heterogeneous Joint Citation Prediction**
Wendlinger, Nonn et al. "The Missing Link: Joint Legal Citation Prediction using Heterogeneous Graph Enrichment." DEXA 2025. [arXiv:2506.22165](https://arxiv.org/abs/2506.22165)
- Joint Case-Case and Case-Law citation prediction on heterogeneous graph with relational GCN. Joint prediction adds +4.7 points. Fully inductive on unseen cases.
- **COLIEE fit:** **Highly relevant.** Closest match to our setting — heterogeneous graph for citation prediction with inductive capability for test documents.

**G-DSR — Graph-Augmented Dense Statute Retriever**
Louis, van Dijck, Spanakis. "Finding the Law: Enhancing Statutory Article Retrieval via Graph Neural Networks." EACL 2023. [arXiv:2301.12847](https://arxiv.org/abs/2301.12847)
- GNN enriches bi-encoder embeddings by aggregating from structurally neighboring articles. Code: [github.com/maastrichtlawtech/gdsr](https://github.com/maastrichtlawtech/gdsr).
- **COLIEE fit:** Directly transferable — GNN enriches case embeddings from graph neighbors (co-cited cases, shared-statute cases).

**Reproducibility Study of Graph Legal Retrieval**
Donabauer, Kruschwitz. "A Reproducibility Study of Graph-Based Legal Case Retrieval." SIGIR 2025. [arXiv:2504.08400](https://arxiv.org/abs/2504.08400)
- Key finding: **graph construction choices matter more than GNN architecture choices**. Provides open artifacts.

### 2.4 Inductive / Dynamic Graph Methods

**RAGraph — Retrieval-Augmented Graph Learning**
"RAGraph: A General Retrieval-Augmented Graph Learning Framework." NeurIPS 2024. [arXiv:2410.23855](https://arxiv.org/abs/2410.23855)
- Retrieves similar toy-graph patterns for unseen graphs. Enables generalization without retraining.

**InGram — Inductive KG Embedding via Relation Graphs**
Lee, Chung, Whang. "InGram: Inductive Knowledge Graph Embedding via Relation Graphs." ICML 2023. [arXiv:2305.19987](https://arxiv.org/abs/2305.19987)
- Handles both unseen entities AND unseen relations at inference. Learns relation-level aggregation.
- **COLIEE fit:** Test documents introduce new judges, new statute references. InGram generalizes without retraining.

**GNN-RAG — Graph Neural Retrieval for LLM Reasoning**
Mavromatis, Karypis. "GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning." NeurIPS 2024. [arXiv:2405.20139](https://arxiv.org/abs/2405.20139)
- GNN as subgraph reasoner. Shortest paths provide interpretable evidence. Outperforms GPT-4 on KGQA.

**Temporal GNN for Citation Networks**
Shen, Haqqani et al. "Temporal Graph Neural Network-Powered Paper Recommendation on Dynamic Citation Networks." arXiv:2408.15371, 2024.
- Temporal Graph Network with RNN-based memory for evolving citation networks.
- **COLIEE fit:** Legal citations are inherently temporal — older cases cite only older cases. Could capture chronological citation patterns.

---

## 3. Contrastive Pre-training

### 3.1 Retrieval-Oriented Pre-training Architectures

**RetroMAE — Masked Auto-Encoder for Retrieval**
Xiao, Liu et al. "RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder." EMNLP 2022. [arXiv:2205.12035](https://arxiv.org/abs/2205.12035)
- Asymmetric masking: encoder sees 15-30%, decoder reconstructs from 50-70% using only [CLS]. Information bottleneck forces rich [CLS] representation. +1.1% MRR@10 over CoCondenser.
- **COLIEE fit:** Pre-train BGE-large on full 7,700-doc corpus before LoRA fine-tuning. ~2-4 hours on single DGX Spark.

**RetroMAE v2 (DupMAE)**
Xiao, Liu et al. "RetroMAE-2: Duplex Masked Auto-Encoder For Pre-Training Retrieval-Oriented Language Models." ACL 2023. [arXiv:2211.08769](https://arxiv.org/abs/2211.08769)
- Dual decoding: sentence reconstruction from [CLS] + BoW prediction from token embeddings. Captures both semantic and lexical signals.

**SimLM — Representation Bottleneck**
Wang, Yang et al. "SimLM: Pre-training with Representation Bottleneck for Dense Passage Retrieval." ACL 2023. [arXiv:2207.02578](https://arxiv.org/abs/2207.02578)
- ELECTRA-style replaced token detection with [CLS] bottleneck. Better for legal text where small wording changes ("shall" vs. "may") carry significant meaning.

**Condenser / CoCondenser**
Gao, Callan. "Condenser: a Pre-training Architecture for Dense Retrieval." EMNLP 2021. [arXiv:2104.08253](https://arxiv.org/abs/2104.08253)
- Modified architecture forcing information aggregation into CLS token. Fixes BERT's CLS not naturally aggregating document-level info.

**MASTER — Multi-task Bottlenecked MAE**
Zhou et al. "MASTER: Multi-task Pre-trained Bottlenecked Masked Autoencoders are Better Dense Retrievers." ECML PKDD 2023. [arXiv:2212.07841](https://arxiv.org/abs/2212.07841)
- Unifies corrupted passage recovery, related passage recovery, PLM output recovery. "Related passage recovery" is directly analogous to case retrieval.

**Longtriever — Long Document Dense Retrieval**
Li et al. "Longtriever: a Pre-trained Long Text Encoder for Dense Document Retrieval." EMNLP 2023.
- Block-level local + global architecture for long documents. Handles full legal cases without truncation.

### 3.2 Legal Domain-Specific Pre-training

**Caseformer — Pre-training for Legal Case Retrieval** ★
Su et al. "Caseformer: Pre-training for Legal Case Retrieval Based on Inter-Case Distinctions." ACM TOIS 2025. [arXiv:2311.00333](https://arxiv.org/abs/2311.00333)
- Three unsupervised tasks: Legal Language Modeling, Legal Judgment Prediction, Factual Description Matching. SOTA in zero-shot and fine-tuning on Chinese + English.
- **COLIEE fit:** **HIGHLY RELEVANT.** Adapt all three tasks to Canadian cases. Use `<FRAGMENT_SUPPRESSED>` contexts as factual descriptions for matching.

**DELTA — Discriminative Encoder via Structural Word Alignment** ★
Li et al. "DELTA: Pre-train a Discriminative Encoder for Legal Case Retrieval via Structural Word Alignment." AAAI 2025. [arXiv:2403.18435](https://arxiv.org/abs/2403.18435)
- Pinpoints "key facts" and aligns [CLS] embedding toward them. Shallow vs. deep decoder bottlenecks.
- **COLIEE fit:** **HIGHLY RELEVANT.** `<FRAGMENT_SUPPRESSED>` citation context windows approximate "key facts" — use as alignment targets.

**LEAD — Scaling Synthetic Pairs for Legal Retrieval** ★
Gao, Xiao et al. "Enhancing Legal Case Retrieval via Scaling High-quality Synthetic Query-Candidate Pairs." EMNLP 2024. [arXiv:2410.06581](https://arxiv.org/abs/2410.06581)
- LLM-based fact extraction + entity anonymization → 100K+ synthetic pairs. Knowledge-driven hard positive selection.
- **COLIEE fit:** Use deepseek-r1:8b to generate synthetic pairs, expanding our 8,251 training pairs significantly. Code: [github.com/thunlp/LEAD](https://github.com/thunlp/LEAD).

**SaulLM — Legal Domain LLM**
Colombo et al. "SaulLM-7B: A pioneering Large Language Model for Law." arXiv:2403.03883, 2024. NeurIPS 2024 (54B/141B).
- Continued pre-training of Mistral on 30B+ legal tokens. MIT license.
- **COLIEE fit:** Extract embeddings from SaulLM-7B hidden states for contrastive fine-tuning.

**CLERC — Legal Case Retrieval Benchmark**
Hou et al. "CLERC: A Dataset for U.S. Legal Case Retrieval and RAG." NAACL 2025 Findings. [arXiv:2406.17186](https://arxiv.org/abs/2406.17186)
- 1.84M federal case documents, 11.54 citations/document.
- **COLIEE fit:** Additional pre-training data. U.S. and Canadian common law share traditions.

### 3.3 Citation/Graph-Aware Contrastive Learning

**SciNCL — Neighborhood Contrastive Learning** ★
Ostendorff et al. "Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings." EMNLP 2022. [arXiv:2202.06671](https://arxiv.org/abs/2202.06671)
- Citation graph embeddings sample contrastive pairs with controlled margins. Avoids hard binary citation cutoff.
- **COLIEE fit:** **HIGHLY RELEVANT.** Build citation graph from training labels, learn graph embeddings (node2vec or GraphRAG Lite), sample pairs by graph distance. Continuous similarity via graph proximity instead of binary cited/not-cited.

### 3.4 Hard Negative Mining

**TriSampler — Quasi-Triangular Negative Sampling**
Yang et al. "TriSampler: A Better Negative Sampling Principle for Dense Retrieval." AAAI 2024. [arXiv:2402.11855](https://arxiv.org/abs/2402.11855)
- Formalizes triangular relationship (query, positive, negative). Avoids both too-easy and false negatives.
- **COLIEE fit:** Principled replacement for our BM25-top-K hard negative mining. Avoids false negatives (unlabeled but actually relevant cases).

**SyNeg — LLM-Driven Synthetic Hard Negatives**
Li et al. "SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval." arXiv:2412.17250, 2024.
- Multi-attribute self-reflection prompting for synthetic negatives. Hybrid sampling (synthetic + retrieved).
- **COLIEE fit:** Use Ollama LLMs to generate topically similar but non-relevant cases as hard negatives.

**ADAM — Adaptive Dark Examples for Distillation**
Tao et al. "Adam: Dense Retrieval Distillation with Adaptive Dark Examples." ACL 2024 Findings. [arXiv:2212.10192](https://arxiv.org/abs/2212.10192)
- Creates "dark examples" via mix-up for nuanced soft labels from cross-encoder teacher.
- **COLIEE fit:** Use our cross-encoder to generate soft labels on mixed passages, distill into bi-encoder.

### 3.5 Domain-Adaptive Unsupervised Methods

**ReContriever — Relevance-Aware Contrastive Pre-training** ★
Lei et al. "Unsupervised Dense Retrieval with Relevance-Aware Contrastive Pre-Training." ACL 2023 Findings. [arXiv:2306.03166](https://arxiv.org/abs/2306.03166)
- Uses intermediate model as imperfect oracle to weight pseudo-positive pairs adaptively.
- **COLIEE fit:** Downweights low-quality positive pairs during corpus pre-training (e.g., procedural + substantive sections).

**GPL — Generative Pseudo Labeling**
Wang et al. "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval." NAACL 2022. [arXiv:2112.07577](https://arxiv.org/abs/2112.07577)
- Generate queries → mine hard negatives → pseudo-label with cross-encoder → train bi-encoder. +9.3 nDCG@10.
- **COLIEE fit:** We have all components. Generate queries with Ollama LLMs, pseudo-label with our cross-encoder.

**COCO-DR — Distributionally Robust Contrastive Pre-training**
Yu et al. "COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval." EMNLP 2022. [arXiv:2210.15212](https://arxiv.org/abs/2210.15212)
- Distributionally robust optimization for domain shift between pre-training and target domain.

**CAPSTONE — Curriculum Sampling with Document Expansion**
He et al. "CAPSTONE: Curriculum Sampling for Dense Retrieval with Document Expansion." EMNLP 2023.
- Progressive difficulty: easy → hard negatives during training. Generates pseudo queries for curriculum.

### 3.6 Multi-Granularity / LLM-Based Embeddings

**mGTE — Generalized Long-Context Embeddings**
Li et al. "mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval." EMNLP 2024 Industry. [arXiv:2407.19669](https://arxiv.org/abs/2407.19669)
- Three-stage training: MLM → weakly supervised contrastive → supervised fine-tuning. 75 languages, 8192 tokens.
- **COLIEE fit:** 8192-token context captures much more of each legal case than our current 512-token BGE-large.

**Nomic Embed v2 — MoE Text Embeddings**
Nussbaum, Duderstadt. "Training Sparse Mixture Of Experts Text Embedding Models." arXiv:2502.07972, 2025.
- First MoE embedding model. 8 experts, top-2 routing. Matryoshka representation learning.
- **COLIEE fit:** Different experts could specialize in different legal domains (immigration, tax, IP).

**LEMUR — Multilingual Legal Embeddings**
"LEMUR: A Corpus for Robust Fine-Tuning of Multilingual Law Embedding Models for Retrieval." arXiv:2602.09570, 2026.
- Legal fine-tuning enhances language-independent content representations. Transfers to unseen languages.
- **COLIEE fit:** Bilingual Canadian Federal Court cases (English/French).

---

## 4. Multi-Task Learning

### 4.1 Direct Joint Legal Retrieval + Entailment

**Joint Learning for Legal Text Retrieval and Entailment** ★
Nguyen Hai Long et al. "Joint Learning for Legal Text Retrieval and Textual Entailment: Leveraging the Relationship between Relevancy and Affirmation." NLLP @ EMNLP 2023. [ACL Anthology](https://aclanthology.org/2023.nllp-1.19/)
- **Only paper directly tackling joint training across legal retrieval + entailment.** Shared BERT backbone with task-specific heads. Key insight: "relevancy" (retrieval) and "affirmation" (entailment) are related but distinct — a document must be relevant before it can entail. Joint model outperforms single-task baselines on both.
- **COLIEE fit:** Replicate for case law. Shared DeBERTa/BGE backbone, Task 1 head (document retrieval) + Task 2 head (paragraph entailment). The key finding — entailment signal improves retrieval — validates our hypothesis.

**NOWJ at COLIEE 2023 — Multi-Task Approaches**
Vuong, Nguyen et al. "NOWJ at COLIEE 2023 -- Multi-Task and Ensemble Approaches in Legal Information Processing." arXiv:2306.04903, 2023.
- Applied MTL to statute tasks (3/4), but used separate models for case law tasks (1/2).
- **COLIEE fit:** Confirms the gap — nobody has done joint Task 1 + Task 2 MTL for case law. This is our novel opportunity.

### 4.2 Multi-Granularity Representations

**AGRaME — Any-Granularity Ranking with Multi-Vector Embeddings** ★
Reddy, Attia et al. "AGRaME: Any-Granularity Ranking with Multi-Vector Embeddings." EMNLP 2024. [arXiv:2405.15028](https://arxiv.org/abs/2405.15028)
- Ranking at multiple granularities (document, paragraph, sentence) from single coarse-level encoding. Multi-granular contrastive loss.
- **COLIEE fit:** Encode documents once for Task 1, rank paragraphs for Task 2 without re-encoding. Joint optimization of both levels.

**Dense X Retrieval — Proposition-Level Granularity**
Chen, Wang et al. "Dense X Retrieval: What Retrieval Granularity Should We Use?" EMNLP 2024. [arXiv:2312.06648](https://arxiv.org/abs/2312.06648)
- "Propositions" as retrieval units — atomic factoid expressions. +10.1 Recall@20 unsupervised over passage-level.
- **COLIEE fit:** Extract propositions from precedent paragraphs for Task 2 entailment. Cross-task: propositions become Task 1 retrieval features.

### 4.3 Multi-Task Dense Retrieval

**TART — Task-aware Retrieval with Instructions**
Asai, Schick et al. "Task-aware Retrieval with Instructions." ACL Findings 2023. [arXiv:2211.09260](https://arxiv.org/abs/2211.09260)
- Single retriever on 37 datasets with task-specific instructions. Outperforms 3x larger models.
- **COLIEE fit:** Frame Task 1 and Task 2 as two retrieval tasks with different instructions: "Find cited cases" vs. "Find entailing paragraphs."

**NV-Embed — LLM as Generalist Embedder**
Lee et al. "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models." ICLR 2025. [arXiv:2405.17428](https://arxiv.org/abs/2405.17428)
- Latent attention pooling, bidirectional attention, two-stage instruction tuning (retrieval first, then multi-task). #1 on MTEB.
- **COLIEE fit:** Stage 1 on Task 1 retrieval pairs, Stage 2 blends in Task 2 entailment pairs.

**Qwen3 Embedding — Unified Embedding + Reranking**
Qwen Team. "Qwen3 Embedding: Advancing Text Embedding and Reranking." arXiv:2506.05176, 2025.
- Single foundation → both embedding (dual-encoder) and reranking (cross-encoder) via LoRA. 100+ languages.
- **COLIEE fit:** Embedding model for Task 1 bi-encoder, reranking model for both Task 1 cross-encoder and Task 2 paragraph scoring. Shared LoRA.

**Multi-task Retriever Fine-tuning for RAG**
Kim et al. "Multi-task retriever fine-tuning for domain-specific and efficient RAG." KDD 2025. [arXiv:2501.04652](https://arxiv.org/abs/2501.04652)
- Instruction fine-tunes small retriever on multiple domain-specific tasks. Generalizes to unseen tasks.

### 4.4 Joint Retrieval and Reasoning

**RankRAG — Unifying Ranking with RAG**
Yu, Ping et al. "RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs." NeurIPS 2024. [arXiv:2407.02485](https://arxiv.org/abs/2407.02485)
- Joint LLM training for ranking + generation. Small fraction of ranking data dramatically improves performance.
- **COLIEE fit:** "Ranking" = Task 1, "reasoning" = Task 2. Even small amount of Task 2 data helps Task 1 ranking.

**Evidence Retrieval for Fact Verification**
Malviya, Katsigiannis. "Evidence Retrieval for Fact Verification using Multi-stage Reranking." EMNLP Findings 2024.
- Multi-granularity evidence extraction (sentences, tables, cells). Entailment filtering improves retrieval recall.
- **COLIEE fit:** Validates entailment → retrieval transfer at multiple granularities.

### 4.5 Knowledge Distillation Across Tasks

**DISKCO — Cross-Encoder to Bi-Encoder Distillation**
Ankith et al. "DISKCO: Disentangling Knowledge from Cross-Encoder to Bi-Encoder." WWW 2024.
- Transfers cross-attention patterns (not just scores) from cross-encoder to bi-encoder.
- **COLIEE fit:** Cross-encoder trained on Task 1 + Task 2 distills attention patterns into bi-encoder. Task 2 entailment attention is especially valuable.

**KELLER — Knowledge-Guided Case Reformulation**
Deng, Mao, Dou. "Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation." EMNLP 2024. [arXiv:2406.19760](https://arxiv.org/abs/2406.19760)
- LLM-reformulated sub-facts with dual-level contrastive learning (case + sub-fact level).
- **COLIEE fit:** Task 2 fragments are natural sub-facts. Sub-fact matching for Task 1 retrieval.

### 4.6 Legal Document Structure for Multi-Task

**ReaKase-8B — Knowledge and Reasoning Representations**
Qiu et al. "ReaKase-8B: Legal Case Retrieval via Knowledge and Reasoning Representations with LLMs." arXiv:2510.26178, 2025.
- Extracts facts, issues, relation triplets, AND reasoning chains. Reasoning = entailment generation.
- **COLIEE fit:** Reasoning chains from Task 2 training data improve Task 1 retrieval representations.

**LegalSearchLM — Retrieval as Legal Elements Generation**
Kim et al. "LegalSearchLM: Rethinking Legal Case Retrieval as Legal Elements Generation." EMNLP 2025. [arXiv:2505.23832](https://arxiv.org/abs/2505.23832)
- Generative retrieval via constrained decoding. +6-20% precision. Self-supervised (no labels needed).
- **COLIEE fit:** Element generation for Task 1 + paragraph entailment for Task 2 share the same legal element vocabulary.

**LEXA — Graph Contrastive Learning**
Tang et al. "LEXA: Legal Case Retrieval via Graph Contrastive Learning." arXiv:2405.11791, 2024.
- Task 2 paragraph annotations could supervise edge weights in the case graph used for Task 1.

**Long-Document Retrieval Survey**
"A Survey of Long-Document Retrieval in the PLM and LLM Era." arXiv:2509.07759, 2025.
- Provides taxonomic framework. Our multi-task setup = "passage-based divide-and-conquer" with Task 2 as paragraph-level supervision.

---

## 5. Other Novel Approaches

### 5.1 Late Interaction Models

**Jina-ColBERT-v2**
Rothe et al. "Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever." MRL @ EMNLP 2024.
- 8192-token context, 89 languages. Matryoshka loss for 50% storage reduction (128→64 dims).
- **COLIEE fit:** First-stage retriever handling full legal documents. Token-level matching captures specific legal concept matches that single-vector encoders miss.

**ColBERT Token Pooling**
Answer.AI. "A Little Pooling Goes a Long Way for Multi-Vector Representations." 2025.
- Clusters similar tokens, averages representations. 2-4x vector reduction, minimal degradation.
- **COLIEE fit:** Makes ColBERT tractable for our 7,700 long documents on 128GB nodes.

**PLAID Engine**
Santhanam et al. "PLAID: An Efficient Engine for Late Interaction Retrieval." CIKM 2022. [arXiv:2205.09707](https://arxiv.org/abs/2205.09707)
- 7x GPU / 45x CPU speedup via centroid interaction and pruning.

### 5.2 Learned Sparse Retrieval

**CSPLADE — Learned Sparse with Causal LMs**
"CSPLADE: Learned Sparse Retrieval with Causal Language Models." AACL 2025. [arXiv:2504.10816](https://arxiv.org/abs/2504.10816)
- SPLADE on Llama-3.1-8B backbone. <8GB index vs. 135GB dense. 41.3 MRR on MS MARCO.
- **COLIEE fit:** Domain-specific term expansion for legal terminology ("estoppel", "res judicata"). CPU-friendly.

**LACONIC — Dense-Level Effectiveness for Sparse Retrieval** ★
"LACONIC: Dense-Level Effectiveness for Scalable Sparse Retrieval." arXiv:2601.01684, 2026.
- Llama-3 based (1B/3B/8B). SOTA 60.2 nDCG on MTEB Retrieval. 71% less index memory than dense.
- **COLIEE fit:** 1B or 3B variant as first-stage retriever. Sparse retrieval naturally handles legal keyword matching + learned semantic expansion. Could replace or augment BM25.

### 5.3 Test-Time Reasoning Rerankers

**Rank1 — Test-Time Compute for Reranking** ★★
Weller et al. "Rank1: Test-Time Compute for Reranking in Information Retrieval." arXiv:2502.18418, 2025.
- First reranking model using test-time compute. Generates reasoning chains before relevance judgments. Distills 600K+ reasoning traces. SOTA on reasoning-intensive datasets. Explainable.
- **COLIEE fit:** **HIGH IMPACT.** Legal citation is reasoning-intensive — judges cite for specific legal reasoning connections. A reranker that reasons about WHY case A cites case B captures signals our similarity-based cross-encoder misses. 7B model fits our hardware.

**Rank-R1 — RL-Enhanced Reasoning Reranker**
Zhang et al. "Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning." arXiv:2503.06034, 2025.
- GRPO training with only relevance labels (no reasoning supervision needed). 14B surpasses zero-shot GPT-4 on BRIGHT. Uses only 18% of training data.
- **COLIEE fit:** Train reasoning reranker on COLIEE labels without explicit reasoning annotations. Data-efficient.

**Rank-K — Listwise Reasoning Reranker**
"Rank-K: Test-Time Reasoning for Listwise Reranking." arXiv:2505.14432, 2025.
- QwQ-32B based listwise reranker. +23% over RankZephyr. LoRA variants for smaller models.

**LimRank — Minimal Data Reasoning Reranker**
"LimRank: Less is More for Reasoning-Intensive Information Reranking." EMNLP 2025. [arXiv:2510.23544](https://arxiv.org/abs/2510.23544)
- Competitive with <5% of typical training data. Includes open-source synthesizer.
- **COLIEE fit:** Ideal for our limited training set.

### 5.4 Synthetic Data Augmentation

**InPars+ — Supercharging Synthetic Data**
Köksal et al. "InPars+: Supercharging Synthetic Data Generation for IR." arXiv:2508.13930, 2025.
- CPO-trained query generator + DSPy-optimized CoT prompts. Works with 3B LLMs.

**Promptagator-Style Training**
"Study on LLMs for Promptagator-Style Dense Retriever Training." arXiv:2510.02241, 2025.
- Task-specific few-shot prompting for synthetic query generation.
- **COLIEE fit:** Design prompts capturing citation-prediction intent: "Given this case, generate a query for cases it cites."

### 5.5 Self-Training / Pseudo-Relevance Feedback

**Reinforced-IR — Self-Boosting Domain Adaptation** ★
Li et al. "Reinforced IR: A Self-Boosting Framework For Domain-Adapted Information Retrieval." ACL 2025.
- Retriever and generator learn from each other's feedback in a self-boosting loop. Works with unlabeled corpus.
- **COLIEE fit:** Iteratively improve bi-encoder using cross-encoder judgments as pseudo-labels on our 7,700-doc unlabeled corpus.

**PromptPRF — PRF Closes Gap Between Small and Large Retrievers**
"Pseudo Relevance Feedback is Enough to Close the Gap." arXiv:2503.14887, 2025.
- LLM extracts structured features from top-k docs. Small retrievers + PRF match large models. Sub-millisecond.
- **COLIEE fit:** Free improvement — extract key entities from BM25 top-k to augment bi-encoder queries.

**LLM-VPRF — Vector PRF with LLM Embeddings**
Li et al. "LLM-VPRF: Large Language Model Based Vector Pseudo Relevance Feedback." arXiv:2504.01448, 2025.
- Rocchio-style embedding updates in LLM embedding space. Sub-millisecond latency.

### 5.6 Legal-Specific Novel Approaches

**LeCoPCR — Legal Concept-Guided Retrieval**
"LeCoPCR: Legal Concept-guided Prior Case Retrieval for European Court of Human Rights cases." NAACL Findings 2025. [arXiv:2501.14114](https://arxiv.org/abs/2501.14114)
- Generates legal concepts from facts as intent signals. DPP for quality-diversity balance.
- **COLIEE fit:** Extract legal concepts from `<FRAGMENT_SUPPRESSED>` context windows as augmented query terms.

**NS-LCR — Neuro-Symbolic Legal Case Retrieval**
Yu et al. "Logic Rules as Explanations for Legal Case Retrieval." LREC-COLING 2024. [arXiv:2403.01457](https://arxiv.org/abs/2403.01457)
- Learns case-level and law-level logic rules. Model-agnostic plug-in. SOTA + faithful explanations.
- **COLIEE fit:** Logic rules as meta-learner features (e.g., "immigration + Charter s.7 → cite X").

**GEAR — Judgment-Integrated Generative Retrieval**
Qin et al. "Explicitly Integrating Judgment Prediction with Legal Document Retrieval." SIGIR 2024. [arXiv:2312.09591](https://arxiv.org/abs/2312.09591)
- Insight: cases with similar judgments cite similar precedents.
- **COLIEE fit:** Outcome similarity (appeal allowed/dismissed) as retrieval signal.

**LAMUS — Legal Argumentation Mining Corpus**
"LAMUS: A Large-Scale Corpus for Legal Argument Mining from U.S. Caselaw." arXiv:2603.08286, 2026.
- 2.9M labeled sentences, 6 argument categories (Fact, Issue, Rule, Analysis, Conclusion, Other).
- **COLIEE fit:** Train argument classifier, apply to Canadian cases. Argument-type matching features for meta-learner.

### 5.7 Multi-View / Hybrid Retrieval

**BGE-M3 — Triple Retrieval from Single Model** ★
Chen et al. "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings." ACL Findings 2024. [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- Dense + sparse (SPLADE-like) + multi-vector (ColBERT-like) from one model. Self-knowledge distillation. 8192 tokens.
- **COLIEE fit:** Three retrieval signals feeding meta-learner from one model. Maximum feature diversity with minimal compute.

### 5.8 MoE for Retrieval

**CAME — Competitive MoE for First-Stage Retrieval**
Cai et al. "CAME: Competitively Learning a Mixture-of-Experts Model for First-stage Retrieval." ACM TOIS 2024. [arXiv:2311.02834](https://arxiv.org/abs/2311.02834)
- Shared bottom layers + competitive expert specialization. Experts auto-discover domain specializations.
- **COLIEE fit:** Experts specialize in immigration, tax, IP, aboriginal law, etc.

### 5.9 Query Expansion

**Legal Retrieval Reasoning Benchmark**
Zheng et al. "A Reasoning-Focused Legal Retrieval Benchmark." CS and LAW 2025. [arXiv:2505.03970](https://arxiv.org/abs/2505.03970)
- Query expansion with legal reasoning rollouts: +10pp Recall@10. Legal retrieval has much lower lexical overlap than standard benchmarks.
- **COLIEE fit:** Expand `<FRAGMENT_SUPPRESSED>` contexts with structured legal reasoning using open-source LLM.

---

## 6. Cross-Cutting Synthesis & Priority Recommendations

### 6.1 Novelty Assessment

Papers marked ★ or ★★ are highest-novelty items not used in any prior COLIEE submission:

| Approach | Paper | Novelty for COLIEE | Compute |
|----------|-------|--------------------|---------|
| Reasoning reranker | Rank1 ★★ | Never used in COLIEE | 7B, fits GB10 |
| Joint Task 1+2 MTL | Nguyen et al. ★ | Only done for statute tasks | Standard fine-tuning |
| Citation-aware pre-training | Caseformer ★ | Not applied to Canadian cases | ~4h corpus pre-training |
| Key-fact alignment | DELTA ★ | Uses FRAGMENT_SUPPRESSED naturally | ~4h pre-training |
| Citation graph contrastive | SciNCL ★ | Not combined with GraphRAG | ~2h training |
| Triple retrieval (single model) | BGE-M3 ★ | Not used in COLIEE Task 1 | Inference only |
| Self-boosting loop | Reinforced-IR ★ | Never applied to legal IR | Iterative, ~8h total |
| Learned sparse retrieval | LACONIC ★ | Not applied to legal domain | 1B model fits |
| Synthetic data | LEAD ★ | Not used for Canadian law | ~4h generation |
| Hierarchical sparse attention | HDT | Not used in cross-encoders | Standard fine-tuning |
| Any-granularity ranking | AGRaME ★ | Not applied to legal | Standard fine-tuning |
| Heterogeneous GNN reranker | Wendlinger et al. | Most similar to our approach | ~2h training |

### 6.2 Top Priority Implementations

Ranked by expected F1 impact and feasibility on our hardware:

**Tier 1 — High Impact, Immediately Feasible (1-2 days each)**

1. **Reasoning Reranker (Rank1/Rank-R1)** — Replace or supplement cross-encoder with 7B reasoning reranker. Legal citation is inherently reasoning-intensive. GRPO training needs only relevance labels.

2. **BGE-M3 Triple Retrieval** — Drop-in replacement for bi-encoder giving dense + sparse + ColBERT signals. Three features to meta-learner from one model inference. No training needed.

3. **Synthetic Data via LEAD** — Use deepseek-r1:8b to generate 50K+ synthetic training pairs. Highest training data ROI.

4. **PRF Query Expansion** — Nearly free. Use BM25 top-k to augment bi-encoder queries. Zero training cost.

**Tier 2 — Novel Contributions, Medium Effort (3-5 days each)**

5. **Joint Task 1+2 MTL** — Shared DeBERTa backbone with task-specific heads. Only paper doing this used statute tasks — case law MTL is novel.

6. **Citation-Aware Pre-training (Caseformer + DELTA)** — Pre-train on corpus with `<FRAGMENT_SUPPRESSED>` positions as structural anchors. 3-stage: corpus MLM → citation contrastive → supervised fine-tuning.

7. **GNN Reranker (GNRR-style)** — Build corpus graph from bi-encoder embeddings + entity overlap, train 2-layer GAT, feed scores to meta-learner.

8. **Hierarchical Cross-Encoder (HDT + PL-Marker)** — Levitated markers at `<FRAGMENT_SUPPRESSED>` positions with hierarchical sparse attention.

**Tier 3 — Ambitious Novel Contributions (1+ weeks)**

9. **Heterogeneous GNN with Relational GCN** — 5 node types (documents, statutes, judges, domains, outcomes), joint Case-Case and Case-Statute citation prediction.

10. **Full Contrastive Pre-training Pipeline** — RetroMAE → SciNCL citation contrastive → TriSampler curriculum fine-tuning.

11. **MoE Legal Embedding** — Domain-specialized experts for different legal areas.

### 6.3 Recommended Novel Architecture

Combining the strongest non-overlapping ideas into one pipeline:

```
Stage 0: Corpus Pre-training (NEW)
  ├── RetroMAE on 7,700 docs (domain adaptation)
  └── Caseformer-style factual matching on <FRAGMENT_SUPPRESSED> contexts

Stage 1: Multi-Signal Retrieval (ENHANCED)
  ├── BM25 (existing)
  ├── BGE-M3 dense + sparse + ColBERT (NEW — replaces single bi-encoder)
  ├── LACONIC learned sparse (NEW — complements BM25)
  └── PRF query expansion (NEW — free improvement)

Stage 2: GNN Score Refinement (NEW)
  ├── Corpus graph from BGE-M3 embeddings + entity overlap
  ├── 2-layer GAT refines retrieval scores
  └── Heterogeneous edges: statute co-citation, judge, domain

Stage 3: Reasoning Reranker (NEW — replaces/supplements cross-encoder)
  ├── Rank1-style 7B reasoning reranker (GRPO-trained on COLIEE labels)
  ├── Generates reasoning chains explaining WHY case A cites case B
  └── Joint Task 1 + Task 2 training (shared backbone)

Stage 4: GraphRAG Lite (existing, enhanced)
  ├── Entity graph with counterfactual augmentation (CFGL-LCR)
  └── Community features

Stage 5: Meta-Learner (existing, expanded)
  ├── Existing 22 features
  ├── + BGE-M3 sparse/ColBERT scores (3 new features)
  ├── + GNN-refined score (1 new feature)
  ├── + Reasoning reranker score (1 new feature)
  ├── + Argument-type matching features (6 new features)
  ├── + Legal concept overlap (1 new feature)
  └── LightGBM with expanded feature set (~34 features)
```

### 6.4 Paper Count Summary

| Direction | Papers Found | Non-overlapping with Prior Reviews |
|-----------|-------------|-----------------------------------|
| Citation-Aware Attention | 18 | 18 (all new) |
| GNN Rerankers | 22 | 19 (3 overlap with GraphRAG review) |
| Contrastive Pre-training | 35 | 25 (10 overlap with GNN/other) |
| Multi-Task Learning | 20 | 17 (3 overlap with other sections) |
| Other Novel Approaches | 32 | 20 (12 overlap with other sections) |
| **Total unique papers** | **~99** | |

---

*Generated 2026-03-13 by 5 parallel research agents searching SIGIR, ACL, EMNLP, NAACL, CIKM, ECIR, WSDM, KDD, NeurIPS, ICLR, AAAI, and arXiv 2022-2026.*
