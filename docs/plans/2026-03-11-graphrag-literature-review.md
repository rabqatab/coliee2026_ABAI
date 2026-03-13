# GraphRAG for Legal Case Retrieval: Literature Review & Feasibility Analysis

**Date:** 2026-03-11
**Purpose:** Evaluate whether GraphRAG is worth adopting for COLIEE 2026 Task 1, based on comprehensive literature search across legal, biomedical, patent, financial, and general IR domains.

---

## Executive Summary

### The Verdict

**GraphRAG (Microsoft-style community summarization) was NOT designed for document retrieval** — it was designed for query-focused summarization. No paper benchmarks it on precision/recall retrieval tasks. Multiple rigorous evaluations find it frequently underperforms vanilla RAG on standard tasks.

**However**, lightweight GraphRAG variants and graph-based retrieval patterns from other domains **do transfer well** to legal case retrieval. The key is extracting the useful components (entity graphs, community detection, graph-based reranking) while discarding the expensive, QA-focused components (LLM extraction, community summarization, map-reduce generation).

### Three Key Papers That Change the Design

| Paper | Venue | Key Insight |
|-------|-------|-------------|
| **Practical GraphRAG** (SAP) | arXiv 2507.03226 | SpaCy dependency parsing achieves **94% of LLM-based extraction** at fraction of cost. Hybrid retrieval (vector + graph + RRF) gives +12% context precision. |
| **E2GraphRAG** | arXiv 2505.24226 | SpaCy entity extraction + co-occurrence relations = **10x faster indexing than GraphRAG, 100x faster retrieval than LightRAG**, competitive quality. |
| **G-RAG** (Graph-based Reranking) | arXiv 2405.18414 | GNN reranker between retriever and reader **outperforms even PaLM 2 as a reranker** with smaller compute footprint. |

### Recommendation for Option C

Adopt a **"Practical GraphRAG"** approach: SpaCy/regex entity extraction → entity co-occurrence graph → Leiden communities → graph features for meta-learner. Skip LLM extraction, skip community summarization. Total added compute: ~30 minutes CPU.

---

## 1. GraphRAG: What It Actually Is (and Isn't)

### Microsoft GraphRAG (Original)

**Source:** "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" — Edge et al., Microsoft Research, arXiv 2404.16130 (April 2024, updated Feb 2025)

**What it does:**
1. LLM extracts entities/relationships from every text chunk
2. Builds entity knowledge graph
3. Leiden community detection creates hierarchical clusters
4. LLM generates summaries for each community
5. At query time: map-reduce over community summaries to answer global questions

**What it does NOT do:**
- Return a ranked list of documents
- Compute precision/recall/F1 for retrieval
- Handle pairwise document relevance scoring

**Designed for:** "What are the main themes in this dataset?" (global summarization)
**NOT designed for:** "Which documents does this query cite?" (pairwise retrieval)

### Rigorous Evaluations Show Moderate Gains

**"RAG vs. GraphRAG: A Systematic Evaluation"** — arXiv 2502.11371 (Feb 2025)
- Unified evaluation across 4 GraphRAG categories
- **Finding:** GraphRAG "frequently underperforms vanilla RAG on many real-world tasks"
- RAG excels on single-hop; GraphRAG helps for multi-hop and summarization

**"When to Use Graphs in RAG" (GraphRAG-Bench)** — arXiv 2506.05690, **accepted at ICLR 2026**
- **Finding:** GraphRAG "may overemphasize high-level statements at the expense of fine-grained details"
- Existing benchmarks "inadequately assess GraphRAG"

**"How Significant Are the Real Performance Gains?"** — arXiv 2506.06331 (May 2025)
- Applied unbiased evaluation to 3 GraphRAG methods
- **Finding:** "Performance gains are much more moderate than previously reported"

### Implication for COLIEE

Full Microsoft GraphRAG is a **poor fit** for Task 1. The task requires ranked document retrieval evaluated by micro-F1, not global summarization. The community summaries would be too coarse for pairwise relevance scoring. The cost (~40 hours LLM extraction for 7,708 docs) is prohibitive with unreliable quality.

---

## 2. What Actually Works for Legal Case Retrieval (Graph-Based)

### CaseLink — 2nd Place COLIEE 2025 (F1=0.2962)

**Source:** Tang et al., "CaseLink: Inductive Graph Learning for Legal Case Retrieval" — ACM SIGIR 2024, arXiv 2403.17780

- Constructs Global Case Graphs (GCG) with Case-Case, Case-Charge, Charge-Charge edges
- LLM text embeddings as node features
- GNN message-passing for case representation
- Contrastive loss + degree regularization training
- **UQLegalAI secured positions 2-4** with F1 0.2940-0.2962

**Key insight:** This is a GNN approach (not GraphRAG). It uses the citation graph structure, not community summaries.

### CaseGNN — SOTA on COLIEE 2022 & 2023

**Source:** Tang et al., ECIR 2024, arXiv 2312.11229

- Converts each case into Text-Attributed Case Graph (TACG)
- Edge Graph Attention Layer + readout for representation
- Contrastive loss with hard negative sampling

### LEXA — Extension of CaseGNN

**Source:** Tang et al., arXiv 2405.11791

- Edge-Updated Graph Attention Layer (EUGAT)
- Graph contrastive learning with augmentation
- LLM-generated node/edge features
- Significantly improves over CaseGNN

### CFGL-LCR — Counterfactual Graph Learning

**Source:** Zhang et al., ACM SIGKDD 2023

- Counterfactual data augmentation to learn causal relationships
- Relational GNNs for case representation
- Significantly outperforms previous SOTA

### Joint Legal Citation Prediction

**Source:** "The Missing Link" — arXiv 2506.22165, DEXA 2025

- GNN link prediction for both Case-Case AND Case-Statute citations
- Heterogeneous graph with semantic meta-information
- **Joint prediction gives +4.7 points improvement** at doubled efficiency
- +13% mAP for similar case matching, +11% accuracy for retrieval

### Reproducibility Study

**Source:** Donabauer et al., SIGIR 2025, arXiv 2504.08400

- Reproduces CaseLink with open LLMs instead of closed APIs
- Validates reliability and generalizability of graph-based legal IR

### COLIEE 2025 Winner (Non-Graph)

**JNLP** won Task 1 with F1=0.3353 using BM25 + enhanced semantic features (no graph). The gap from CaseLink (0.2962) to JNLP (0.3353) was bridged by feature engineering, not fundamentally different retrieval.

---

## 3. Legal-Domain GraphRAG Applications

### SAT-Graph RAG — Ontology-Driven Legal Norms

**Source:** De Martim et al., arXiv 2505.00039 (May 2025)

- Structure-Aware Temporal Graph RAG for Brazilian legislation
- LRMoo-inspired ontology: abstract legal Works vs. versioned Expressions
- Legislative events as first-class Action nodes
- Planner-guided deterministic query resolution
- **Not a retrieval system** — focuses on temporal/structural legal QA

### Vietnamese Legal Text Graph RAG

**Source:** Huynh et al., ICCIES 2025 (Springer, July 2025)

- GraphRAG adapted for Vietnamese legal documents
- Handles complex structures, specialized terminology

### Smart-Slic — RAG + KG + NMF for Legal

**Source:** Barron et al., arXiv 2502.20364, ICAIL 2025

- Combines vector stores + Neo4j knowledge graphs + hierarchical NMF
- **NMF uncovers latent topics** without LLM — lightweight topic discovery
- Chunking + NMF improves accuracy on large unstructured case law
- **Hybrid RAG-KG achieved 70% pass rate vs. 37.5% for RAG-only**

### Azure GraphRAG Legal Cases (Microsoft Sample)

**Source:** Azure-Samples/graphrag-legalcases-postgres (GitHub)

- End-to-end implementation on U.S. Case Law (500K cases)
- Exploits natural citation graph of case law
- PostgreSQL + Apache AGE for graph queries
- **Key finding:** "GraphRAG fits legal research especially well because case law has a natural graph structure"

### Myanmar Law Cases with GraphRAG

**Source:** IEEE ICCIES 2024, IEEE Xplore 10825155

- GraphRAG for retrieving and summarizing archival case documents

---

## 4. Cross-Domain Insights (Transferable to Legal)

### Patent Retrieval — Structurally Identical to COLIEE

**"Building a Graph-Based Patent Search Engine"** — Bjorkqvist & Kallio, SIGIR 2023
- Converts each patent into entity-relation graph
- GNN learns to find prior art using examiner citation data as training signal
- **HIGHLY RELEVANT:** Patent prior art search = legal case retrieval

**"Efficient Patent Searching Using Graph Transformers"** — arXiv 2508.10496 (Aug 2025)
- Graph Transformer dense retrieval
- **"Substantially higher recall on citation retrieval than text-based Transformer models"**
- More efficient than text Transformers for long documents

**"Patent Retrieval with Text + KG Embeddings"** — Siddharth et al., arXiv 2211.01976
- Combines Sentence-BERT (text) + TransE (citation graph) + inventor metadata
- Multi-faceted embedding concatenation outperforms text-only
- **Transfers to legal:** text embeddings + citation graph embeddings + judge/court metadata

### Biomedical — Similar Long-Document Challenges

**MedGraphRAG** — ACL 2025
- Triple-linked structure: user docs → authoritative sources → controlled vocabularies
- U-Retrieval: top-down precise + bottom-up refinement
- **Transfers to legal:** case docs → cited cases → legal concepts/statutes

**MedSumGraph** — Artificial Intelligence in Medicine (Nov 2025)
- KG construction to summarize structured knowledge
- Summarization-based global search + graph-based local search
- No fine-tuning required — relevant for open-source constraint

### G-RAG — Graph-Based Reranking (General Domain)

**Source:** Dong et al., arXiv 2405.18414 (May 2024)

- GNN-based reranker between retriever and reader
- Uses Abstract Meaning Representation (AMR) graphs for inter-document relationships
- **Outperforms SOTA rerankers including PaLM 2** with smaller compute
- **HIGHLY RELEVANT:** GNN reranker fits perfectly in BM25 → rerank → predict pipeline

### CommunityKG-RAG — Community Detection for Retrieval

**Source:** arXiv 2408.08535 (Aug 2024)

- KG from articles → Louvain community detection → word embeddings on nodes
- Focuses retrieval on tight-knit clusters rather than full corpus
- Zero-shot (no training needed)
- 56.24% accuracy with LLaMA 7B
- **Transfers to legal:** Cases sharing statutes/judges form communities → retrieve within community first

### Financial GraphRAG — Efficiency Gains

**Source:** GenAIK Workshop at ACL 2025

- GraphRAG reduces token usage by 734x vs. conventional RAG
- 6% hallucination reduction
- **Key:** Computational efficiency from graph-structured retrieval over flat retrieval

---

## 5. Lightweight GraphRAG Variants (Practical for COLIEE)

### LazyGraphRAG — No LLM During Indexing

**Source:** Microsoft Research Blog (Nov 2024)

- NLP noun phrase extraction for concepts, co-occurrence for relations
- **Indexing cost: 0.1% of full GraphRAG** (1,000x cheaper)
- Combines best-first and breadth-first search at query time
- Quality comparable to full GraphRAG global search

### E2GraphRAG — SpaCy-Based, 10x Faster

**Source:** arXiv 2505.24226 (May 2025)

- **SpaCy (not LLMs)** for entity extraction: named entities + common nouns
- Relations: entity co-occurrence within sentences (undirected weighted edges)
- Recursive document summary trees
- **10x faster indexing than GraphRAG, 100x faster retrieval than LightRAG**
- Adaptive local/global retrieval mode selection

### LinearRAG — Relation-Free, Linear Scale

**Source:** arXiv 2510.10114, **accepted at ICLR 2026**

- **No relation extraction at all** — avoids noisy/inconsistent relations
- Relation-free hierarchical "Tri-Graph" with lightweight entity extraction + semantic linking
- Scales linearly with corpus size, zero extra token consumption
- Two-stage: entity activation via semantic bridging → passage retrieval via importance aggregation

### Practical GraphRAG (SAP) — 94% of LLM Quality

**Source:** arXiv 2507.03226 (Jul 2025)

- **Dependency parsing** achieves 94% of LLM-based extraction (61.87% vs. 65.83%)
- SpaCy-based KG construction (no LLM needed)
- Hybrid retrieval: vector + graph traversal via RRF
- Separate embeddings for entities, chunks, and relations
- **+12% context precision, -32% "no coverage" answers** vs. dense-vector RAG

### GliNER — Zero-Shot NER Without LLMs

**Source:** NAACL 2024, GitHub urchade/GLiNER

- Bidirectional transformer encoder for zero-shot NER
- **Outperforms ChatGPT and fine-tuned LLMs** on NER benchmarks
- Runs on CPUs and consumer hardware
- Recommended by GraphRAG documentation as NER solution

### nano-graphrag — Minimal Implementation

**Source:** GitHub gusye1234/nano-graphrag

- **~1,100 lines of Python** (excluding tests)
- Supports NetworkX and Neo4j backends
- Three query modes: naive (vector), local (entity neighborhood), global (community)
- Supports Ollama for local open-source LLMs
- Most widely adopted open-source GraphRAG implementation

### LightRAG — Dual-Level Retrieval

**Source:** arXiv 2410.05779, **EMNLP 2025 Findings**

- Dual-level: low-level entity-specific + high-level thematic retrieval
- Incremental graph updates without full regeneration
- 6,000x fewer tokens than Microsoft GraphRAG per retrieval
- ~30% query latency reduction
- Supports Ollama; tested with models as small as gemma2:2b

---

## 6. Key Production Lessons

### Entity Extraction Quality

- **LLM extraction is noisy and expensive:** "visually appealing but impractical" — auto-extracted entities/relations contain significant noise, redundancy, and errors (TDS production lessons)
- **Open-source LLMs fail at structured extraction:** Llama 3.1 70B indexing fails ~50% of time due to format non-compliance (Ollama pitfalls guide)
- **SpaCy/GliNER are practical alternatives:** 94% of LLM quality, runs on CPU, no API costs

### Cost and Scalability

- Standard GraphRAG spends **75% of token budget before a single question is asked**
- Entity extraction + dedup + summarization can consume **several to dozens of times more tokens** than original text
- **Budget-tier models** for extraction cost 15-50x less than flagship models
- **GPU-accelerated Leiden** (cuGraph) delivers up to **47x speedup** over CPU

### Hybrid Retrieval

- Pure vector search: ~75-80% recall. Hybrid (BM25 + vector): +5-10pp. Adding RRF smooths precision.
- SAP paper: Hybrid with RRF showed **up to 15% improvement** over vanilla vector retrieval
- BM25 → vector expansion → cross-encoder rerank is the standard multi-stage pattern

### GraphRAG Index Fragility

- Updates trigger ripple effects requiring recomputation of community summaries
- Alternatives that skip summarization are more maintainable
- LazyGraphRAG and E2GraphRAG avoid this problem entirely

---

## 7. Synthesis: What Should COLIEE 2026 Adopt?

### What the Literature Supports

| Component | Evidence | Adopt? |
|-----------|----------|--------|
| Entity KG from regex/SpaCy (no LLM) | SAP Practical GraphRAG (94% quality), E2GraphRAG (10x faster), GliNER (outperforms ChatGPT on NER) | **Yes** |
| Leiden community detection on entity graph | CommunityKG-RAG, SemToG, Microsoft GraphRAG (core insight) | **Yes** |
| Community membership as meta-learner feature | CommunityKG-RAG (narrows search space), original GraphRAG insight | **Yes** |
| LLM entity extraction for graph | Production lessons (noisy, 50% failure with open-source), our own benchmarks (empty results) | **No** |
| LLM community summarization | LazyGraphRAG shows it's unnecessary, fragile index updates, not designed for retrieval | **No** |
| LLM reasoning chains as edge attributes | Compute-infeasible, no literature support for retrieval benefit | **No** |
| GNN-based reranking | CaseLink (2nd COLIEE 2025), G-RAG (outperforms PaLM 2), patent search (SIGIR 2023) | **Consider (Phase 2)** |
| Hybrid BM25 + vector + graph + RRF | Universal across all domains, SAP (+15%), COLIEE winners | **Yes** |
| Multi-faceted embeddings (text + graph + metadata) | Patent retrieval, Graphiti, multiple papers | **Consider** |

### What This Means for the Pipeline Options Document

**GraphRAG Lite (as currently designed) is well-supported by literature.** The approach of:
1. Regex/SpaCy entity extraction (not LLM)
2. Bipartite entity graph → Leiden communities
3. Community features in LightGBM meta-learner

...aligns with the practical patterns from E2GraphRAG, SAP Practical GraphRAG, and CommunityKG-RAG.

**One additional component worth considering:**
- **GliNER** (zero-shot NER, NAACL 2024) as a supplement to regex extraction. It could catch legal concepts that regex misses (e.g., "reasonableness standard", "procedural fairness") without LLM cost. Runs on CPU/consumer hardware.

**One component to potentially add in Phase 2:**
- **GNN reranking** (inspired by G-RAG and CaseLink). After BM25 + bi-encoder produces candidates, a lightweight GNN reranker over the entity graph could outperform the cross-encoder for some query types. This is speculative but supported by G-RAG and CaseLink results.

---

## 8. Complete Reference List

### Directly Relevant to COLIEE

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 1 | CaseLink: Inductive Graph Learning for Legal Case Retrieval | SIGIR 2024 | 2024 | 2nd place COLIEE 2025, GNN on case graphs |
| 2 | CaseGNN: GNNs for Legal Case Retrieval | ECIR 2024 | 2024 | SOTA on COLIEE 2022/2023 |
| 3 | LEXA: Legal Case Retrieval via Graph Contrastive Learning | arXiv | 2024 | Extension of CaseGNN with LLM features |
| 4 | CFGL-LCR: Counterfactual Graph Learning for Legal Case Retrieval | KDD 2023 | 2023 | Counterfactual augmentation + relational GNN |
| 5 | Joint Legal Citation Prediction using Heterogeneous Graphs | DEXA 2025 | 2025 | Joint case-case + case-statute prediction (+4.7 pts) |
| 6 | Reproducibility Study of Graph-Based Legal Case Retrieval | SIGIR 2025 | 2025 | Validates CaseLink with open LLMs |
| 7 | UQLegalAI@COLIEE2025 | arXiv | 2025 | CaseLink at COLIEE 2025, F1=0.2962 |

### Legal-Domain GraphRAG

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 8 | SAT-Graph RAG: Ontology-Driven Graph RAG for Legal Norms | arXiv | 2025 | Temporal/structural legal KG |
| 9 | Vietnamese Legal Text Graph RAG | ICCIES 2025 | 2025 | GraphRAG for non-English legal |
| 10 | Smart-Slic: RAG + KG + NMF for Legal | ICAIL 2025 | 2025 | Hybrid RAG-KG achieves 70% vs. 37.5% RAG-only |
| 11 | Myanmar Law Cases with GraphRAG | IEEE 2024 | 2024 | GraphRAG for case retrieval/summarization |
| 12 | Azure GraphRAG Legal Cases | GitHub | 2024 | Microsoft sample on 500K U.S. cases |

### GraphRAG Core & Evaluations

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 13 | From Local to Global (Microsoft GraphRAG) | arXiv | 2024 | Original GraphRAG paper |
| 14 | Graph Retrieval-Augmented Generation: A Survey | ACM TOIS | 2025 | Comprehensive taxonomy |
| 15 | RAG vs. GraphRAG: A Systematic Evaluation | arXiv | 2025 | GraphRAG often underperforms vanilla RAG |
| 16 | GraphRAG-Bench (When to Use Graphs in RAG) | ICLR 2026 | 2025 | Comprehensive benchmark, identifies limitations |
| 17 | Unbiased Evaluation of GraphRAG | arXiv | 2025 | Gains "much more moderate than reported" |

### Lightweight GraphRAG Variants

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 18 | LightRAG | EMNLP 2025 | 2024 | Dual-level retrieval, 6000x fewer tokens |
| 19 | E2GraphRAG | arXiv | 2025 | SpaCy extraction, 10x/100x faster |
| 20 | LinearRAG | ICLR 2026 | 2025 | Relation-free, linear scale |
| 21 | Practical GraphRAG (SAP) | arXiv | 2025 | SpaCy = 94% of LLM quality, hybrid RRF |
| 22 | LazyGraphRAG | Microsoft Research | 2024 | No LLM indexing, 0.1% cost |
| 23 | GRAG | NAACL 2025 | 2024 | Divide-and-conquer subgraph retrieval |
| 24 | nano-graphrag | GitHub | 2024 | 1,100 lines, Ollama support |

### Cross-Domain (Transferable)

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 25 | Graph-Based Patent Search Engine | SIGIR 2023 | 2023 | Patent = legal analog, GNN on entity graphs |
| 26 | Efficient Patent Searching using Graph Transformers | arXiv | 2025 | Higher recall than text Transformers |
| 27 | Patent Retrieval: Text + KG Embeddings | arXiv | 2022 | Multi-faceted embeddings beat text-only |
| 28 | MedGraphRAG | ACL 2025 | 2025 | Triple-linked structure for medical docs |
| 29 | MedSumGraph | AI in Medicine | 2025 | Summarization + graph for medical QA |
| 30 | G-RAG: Graph-based Reranking | arXiv | 2024 | GNN reranker outperforms PaLM 2 |
| 31 | CommunityKG-RAG | arXiv | 2024 | Community detection for search narrowing |
| 32 | SemToG: Semantic Think-on-Graph | SD State Thesis | 2025 | Semantic-aware community detection |
| 33 | ReGraphRAG | EMNLP 2025 | 2025 | Reconnects fragmented KG subgraphs |
| 34 | GraphRAG for Finance | ACL 2025 GenAIK | 2025 | 734x token reduction, 6% less hallucination |
| 35 | GliNER: Zero-Shot NER | NAACL 2024 | 2024 | Outperforms ChatGPT on NER, runs on CPU |

### Legal RAG Benchmarks

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 36 | LegalBench-RAG | arXiv | 2024 | 6,858 legal QA pairs, chunking strategies |
| 37 | LRAGE: Legal RAG Evaluation Tool | arXiv | 2025 | Extends LM Eval Harness for legal |
| 38 | Legal RAG Bench | arXiv | 2026 | Retrieval failures cause "hallucinations" |
| 39 | Benchmarking Legal RAG: Statutory Surveys | arXiv | 2026 | Purpose-built RAG beats Westlaw/Lexis |

### Production & Implementation

| # | Source | Year | Key Insight |
|---|--------|------|-------------|
| 40 | Ollama GraphRAG Pitfalls Guide | 2024 | Open-source LLMs fail 50% on extraction |
| 41 | Production RAG Lessons (TDS) | 2025 | Auto-extracted entities are noisy; reranking is essential |
| 42 | FalkorDB Cost Optimization | 2025 | Budget models 15-50x cheaper; cache aggressively |
| 43 | Do You Really Need GraphRAG? (TDS) | 2025 | Basic RAG often comparable; GraphRAG for complex reasoning |
| 44 | Graph RAG 2026 Practitioner's Guide | 2026 | Cost solvable, but gains often modest |

---

*Document version: 1.0*
*Created: 2026-03-11*
*Total sources: 44 papers/articles across 4 search domains*
*Status: COMPLETE*
