# Legal GraphRAG for COLIEE 2026 Task 1 — Design Document

**Date:** 2026-03-11
**Task:** Task 1 — Legal Case Retrieval
**Goal:** Build a novel GraphRAG pipeline that constructs a corpus-wide legal knowledge graph with reasoning chains, then uses multi-signal retrieval (graph traversal + community matching + embeddings + reasoning chain similarity) to identify cited cases.

---

## 1. Motivation

Prior COLIEE approaches fall into two camps:
- **Lexical + neural pipelines** (BM25 + BERT/LLM reranking) — dominant but plateauing
- **Graph-based methods** (CaseLink, CaseGNN) — 2nd place 2025, but limited to case-case similarity edges

No team has attempted:
1. **Intra-document knowledge graphs** — extracting entities and relationships *within* each case
2. **Community-level retrieval** — Microsoft GraphRAG's core idea applied to legal case retrieval
3. **Reasoning chains** — LLM-generated explanations of *why* two cases are related, used as a retrieval signal

This design combines all three into a novel multi-signal retrieval pipeline.

---

## 2. Competition Constraints

- **Open-source models only** — no GPT-4o, Gemini, or other closed-source LLMs
- **Fully automated** — no human intervention at any stage
- **No model release date restriction** for Tasks 1 & 2
- **Max 3 submission runs** per team per task
- **Evaluation:** micro-averaged F1

---

## 3. Knowledge Graph Schema

### Node Types

| Node Type | Example | Extraction Method |
|-----------|---------|-------------------|
| `Case` | `086870.txt` | File identity |
| `Statute` | IRPA s. 72(1) | Regex + LLM |
| `LegalConcept` | "reasonableness standard" | LLM |
| `LegalTest` | "Dunsmuir test" | LLM |
| `Party` | "Minister of Citizenship and Immigration" | LLM |
| `Judge` | "Mosley J." | Regex + LLM |
| `Outcome` | "application dismissed" | Regex |

### Edge Types

| Edge | From → To | Meaning |
|------|-----------|---------|
| `APPLIES` | Case → Statute | Case applies this statute |
| `INVOKES_CONCEPT` | Case → LegalConcept | Case reasons using this concept |
| `APPLIES_TEST` | Case → LegalTest | Case applies this named test |
| `DECIDED_BY` | Case → Judge | Judge authored decision |
| `HAS_OUTCOME` | Case → Outcome | Case outcome type |
| `IN_DOMAIN` | Case → Domain | Legal domain classification |
| `SIMILAR_REASONING` | Case → Case | LLM reasoning chain explains relationship |
| `BM25_NEIGHBOR` | Case → Case | Top-k BM25 similarity |

### Reasoning Chain Edges

For the top-k most similar case pairs, the LLM generates a 2-3 sentence reasoning chain explaining the legal relationship:

```
"Both cases apply the Dunsmuir reasonableness standard to
immigration officer decisions. Case A finds the officer's
reasoning inadequate; Case B upholds it. Both cite Baker v.
Canada for procedural fairness obligations."
```

The chain is embedded and stored as an edge attribute for reasoning-aware retrieval.

### Expected Graph Size

```
Case nodes:      ~7,708
Entity nodes:    ~1K-3K (statutes, concepts, tests, judges, etc.)
Total nodes:     ~9K-11K
Total edges:     ~350K
```

---

## 4. Pipeline Architecture

### Phase 1: Entity Extraction (~8-12 hours)

```
7,708 docs -> Chunk by [N] paragraph markers
           -> Regex pass (statutes, judges, outcomes)
           -> LLM structured JSON extraction (concepts, tests, holdings, domain)
           -> Merge & deduplicate entities
```

**Regex handles:** statute references, judge names, outcome lines, paragraph markers (~60-70% of entities).

**LLM handles:** legal concepts, named tests, case type, domain classification, key holdings.

**Long document strategy:**
- Documents <8K words: single LLM call
- Documents >8K words: chunk into sections, extract per section, merge
- Deduplication pass: normalize statute names, cluster similar concepts

### Phase 2: Graph Construction (~1 hour)

```
Normalized entities -> NetworkX graph
Case-Case edges from BM25 top-20 neighbors
Entity normalization:
  - Statutes: regex normalization + alias mapping
  - Concepts: embed all, agglomerative clustering (cosine, threshold=0.85)
  - Judges: regex normalization
Serialized as GraphML + JSON
```

### Phase 3: Community Detection & Summarization (~2 hours)

```
Project to weighted Case-only graph:
  weight = 0.3*shared_statutes + 0.3*shared_concepts + 0.3*BM25 + 0.05*same_judge + 0.05*same_domain

Leiden algorithm -> ~100-300 communities
LLM summarizes each community's legal theme
Embed summaries with embedding model -> community vectors
```

### Phase 4: Reasoning Chains (~4-6 hours)

```
For each of 2,001 labeled queries:
  BM25 top-50 candidates -> LLM generates reasoning chain per pair
  ~100K reasoning chains total
  Embed chains with embedding model
  Store as SIMILAR_REASONING edges
```

**Total indexing time estimate: ~15-20 hours on GB10**

### Query-Time Retrieval

```
Query case -> same entity extraction pipeline
           |
Signal 1: BM25 (top-200)
Signal 2: Entity graph traversal (weighted shared entities)
Signal 3: Community matching (query entities -> nearest communities -> member cases)
Signal 4: Embedding similarity (BGE-m3 or Qwen3-Embedding)
Signal 5: Reasoning chain similarity (LLM-generated, top-50 only)
           |
Stage 1: Signals 1-4 (fast) -> RRF -> top-50
Stage 2: Signal 5 (LLM) -> only on top-50
Stage 3: Learned weighted fusion (LightGBM) -> threshold -> predictions
```

---

## 5. Entity Extraction Prompts

### Pass 1 — Regex (no LLM)

```python
Statutes:    r"([\w\s]+Act),?\s*(S\.C\.|R\.S\.C\.)\s*\d{4},?\s*c\.\s*[\w.-]+"
Judges:      r"(\w+),?\s*J\.(?:A\.)?\s*$"
Outcomes:    r"(application|appeal|motion)\s+(is\s+)?(dismissed|allowed|granted)"
Para markers: r"\[(\d+)\]"
```

### Pass 2 — LLM Structured Extraction

```
You are a legal information extractor. Given a Federal Court of Canada
case, extract the following entities in JSON format.

EXTRACT ONLY what is explicitly stated. Do not infer or hallucinate.

{
  "legal_concepts": [
    // Abstract legal principles the case reasons about
    // e.g., "standard of review", "procedural fairness"
  ],
  "legal_tests": [
    // Named legal tests or frameworks applied
    // e.g., "Dunsmuir test", "Baker factors", "Oakes test"
  ],
  "statutes_applied": [
    // Statutes actually applied (not just mentioned), with section if stated
    // e.g., {"name": "IRPA", "section": "s. 72(1)", "context": "judicial review application"}
  ],
  "key_holdings": [
    // 1-2 sentence summary of each major holding
  ],
  "case_type": "judicial_review | appeal | motion | trial | other",
  "legal_domain": "immigration | IP | tax | aboriginal | criminal | administrative | other"
}
```

### Reasoning Chain Prompt

```
Given two Federal Court of Canada cases, explain in 2-3 sentences
WHY they are legally related. Focus on:
1. Shared legal principles or tests applied
2. Similar factual patterns or issues
3. How one case's reasoning builds on or departs from the other

Case A summary: {extracted holdings + concepts from Case A}
Case B summary: {extracted holdings + concepts from Case B}

Reasoning chain:
```

### Quality Control

- Validate JSON schema on every extraction (retry on malformed output, max 2 retries)
- Log extraction failures
- Sample 50 random extractions and spot-check before proceeding

---

## 6. Graph Construction Details

### Entity Normalization

```
Statutes:
  "Immigration and Refugee Protection Act" / "IRPA" / "the Act (IRPA)"
  -> all map to canonical "IRPA"

Concepts:
  Embed all extracted strings with embedding model
  Agglomerative clustering (cosine distance, threshold=0.85)
  Most frequent string becomes canonical name

Judges:
  "Mosley J." / "Mosley, J." / "The Honourable Mr. Justice Mosley"
  -> normalized to "Mosley J."
```

### Community Detection

```
1. Project to weighted Case-only graph
2. Leiden algorithm (resolution tuned for ~100-300 communities)
3. LLM summarizes each community's legal theme
4. Embed summaries -> community vectors
```

### Community Weight Formula

```
edge_weight(case_a, case_b) =
    0.30 * (shared_statutes / max_shared_statutes) +
    0.30 * (shared_concepts / max_shared_concepts) +
    0.30 * (bm25_neighbor_score) +
    0.05 * (same_judge ? 0.1 : 0) +
    0.05 * (same_domain ? 0.2 : 0)
```

---

## 7. Retrieval Signal Details

| Signal | Input | Score | Speed |
|--------|-------|-------|-------|
| S1: BM25 | Raw text | BM25 score, normalized [0,1] | Fast |
| S2: Entity Graph | Extracted entities | Weighted Jaccard (statutes 0.35, concepts 0.30, tests 0.20, domain 0.10, judge 0.05) | Fast |
| S3: Community | Entity profile embedding | Max cosine sim to community vectors | Fast |
| S4: Embedding | Document embedding | Cosine similarity | Fast |
| S5: Reasoning Chain | LLM-generated chain | Cosine similarity of chain embeddings | Slow (LLM) |

### Fusion Strategy

```
Stage 1: RRF(S1, S2, S3, S4) -> top-50 candidates
Stage 2: Generate S5 for top-50 only
Stage 3: LightGBM on [S1, S2, S3, S4, S5] -> final score
         Weights learned via 5-fold GroupKFold on training set
Stage 4: Threshold optimization on dev set -> binary predictions
```

---

## 8. Model Benchmarking

### LLM Candidates (Entity Extraction + Reasoning Chains)

| Model | Active Params | Ollama Tag | Rationale |
|-------|--------------|------------|-----------|
| Qwen3:32B | 32B | `qwen3:32b` | Already installed, strong structured output |
| Llama 4 Scout | 17B active (MoE) | `llama4:scout` | 10M context, native JSON, latest Meta |
| Gemma 3:27B | 27B | `gemma3:27b` | Google's latest, strong instruction following |
| DeepSeek-R1:8B | 8B | `deepseek-r1:8b` | Already installed, reasoning-focused, speed anchor |

### Embedding Model Candidates

| Model | Params | Dims | Max Tokens | MTEB Score | Rationale |
|-------|--------|------|------------|------------|-----------|
| Qwen3-Embedding-8B | 8B | 4096 | 32K | 70.58 (#1) | Current SOTA, Apache 2.0 |
| Qwen3-Embedding-0.6B | 0.6B | 1024 | 32K | Top for size | Speed baseline |
| BGE-m3 | 568M | 1024 | 8192 | 63.0 | Proven standard |
| GTE-Qwen2-7B-instruct | 7B | 3584 | 32K | Top-5 | LLM-based, long context |
| nomic-embed-text | 137M | 768 | 8192 | Solid | Fastest option |

### Benchmark Protocol

**LLM Benchmark (50 docs x 4 models):**

| Metric | Description |
|--------|-------------|
| Valid JSON rate | % of outputs that parse correctly |
| Entity count | Statutes, concepts, tests per doc |
| Precision | % of extracted entities that are real (manual check, 10 docs) |
| Recall | % of real entities extracted (pre-annotated, 10 docs) |
| Time per document | Mean, p95 |
| Consistency | Run same doc 3x, measure overlap |

**Reasoning Chain Benchmark (20 pairs x 4 models):**

| Metric | Description |
|--------|-------------|
| Relevance | Mentions real shared legal concepts? |
| Specificity | Generic vs case-specific (1-5 Likert) |
| Factual grounding | Hallucinated statutes/tests? |
| Discriminativeness | Positive vs negative pair quality difference |
| Time per pair | Mean, p95 |

**Embedding Benchmark (7,708 docs x 5 models):**

| Metric | Description |
|--------|-------------|
| Recall@50/100/200 | On 2,001 labeled queries |
| MRR | Mean Reciprocal Rank |
| Indexing time | Time to embed full corpus |
| Memory usage | Peak GPU/RAM |
| Entity clustering quality | Silhouette score on known-similar entities |

---

## 9. Evaluation Strategy

### Data Split

```
Training data: 2,001 queries with labels (2026 dataset)
Split: 80/20 stratified by citation count -> 1,600 train / 401 dev
```

### Metrics

- **Micro-averaged F1** (primary, matches competition)
- Precision, Recall
- Per-query AUC (diagnostic)

### Ablation Plan

```
Run 1: BM25 only (baseline)
Run 2: BM25 + Entity Graph (S1+S2)
Run 3: BM25 + Entity Graph + Communities (S1+S2+S3)
Run 4: BM25 + Entity Graph + Communities + Embeddings (S1+S2+S3+S4)
Run 5: Full pipeline -- all 5 signals (S1+S2+S3+S4+S5)
```

### Competition Submission (3 runs)

Best 3 configurations from ablation, likely:
- Run A: Full pipeline (all signals)
- Run B: Best subset from ablation
- Run C: Variant with different threshold or fusion weights

---

## 10. Compute Environment

- **Hardware:** NVIDIA GB10 (Project DIGITS / Spark), aarch64
- **Memory:** 128GB unified CPU/GPU memory
- **Ollama:** v0.13.5, running as system service
- **Pre-installed models:** Qwen3:32B (20GB), DeepSeek-R1:8B (5.2GB)
- **Second Spark device:** available but not currently connected

### Time Budget

| Phase | Estimated Time |
|-------|---------------|
| Model benchmarking | ~6-8 hours |
| Entity extraction (full corpus) | ~8-12 hours |
| Graph construction | ~1 hour |
| Community detection + summarization | ~2 hours |
| Reasoning chains | ~4-6 hours |
| Embedding indexing | ~1-3 hours |
| Training + evaluation | ~2-4 hours |
| **Total** | **~24-36 hours (~2-3 days)** |

Within the 5-day budget, leaving room for iteration and debugging.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Noisy entity extraction | Bad graph, bad retrieval | Regex first pass, JSON validation, spot-check 50 docs |
| LLM hallucination in reasoning chains | False connections | Ground chains in extracted entities only, not raw text |
| Community granularity wrong | Too broad or too narrow | Tune Leiden resolution parameter on dev set |
| Signal 5 too slow at test time | Can't run on full test set | Only run on top-50 from stage 1 |
| Memory pressure from large models | OOM on GB10 | Monitor memory, fall back to smaller models |
| Graph too sparse (few shared entities) | Entity signals weak | Add BM25_NEIGHBOR edges as fallback connectivity |

---

*Design approved: 2026-03-11*
