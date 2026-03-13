# COLIEE 2026 Task 1: Pipeline Strategy Options

**Date:** 2026-03-11
**Goal:** Win top-3 placement with a novel, publishable approach
**Constraint:** Open-source models only, fully automated, max 3 submission runs, must incorporate GraphRAG
**Hardware:** NVIDIA GB10 (128GB unified memory), Ollama for local inference
**Evaluation:** Micro-averaged F1

---

## Context: Competition Landscape

### Historical F1 Progression (Task 1)

| Year | Best F1 | Key Breakthrough |
|------|---------|------------------|
| 2019 | 0.19 | Early BERT fine-tuning |
| 2021 | 0.29 | Dense retrieval + BM25 fusion |
| 2023 | 0.35 | Structural BERT + ensemble (THUIR) |
| 2025 | 0.40 | GNN on citation graphs (CaseGNN) + LLM embeddings (ReaKase-8B) |

### What Top Teams Do

- **THUIR** (4x winner, 2020-2023): BM25 → paragraph-level structural BERT → cross-encoder → ensemble. Won by combining lexical and semantic signals with attention-weighted paragraph aggregation.
- **CaseGNN** (2023-2025): GNN message-passing over citation graphs. Added +3-5% over text-only methods by capturing precedent relationships.
- **ReaKase-8B** (2025): Fine-tuned 8B LLM to generate reasoning-augmented embeddings. Added +5-7% over standard embeddings.
- **NOWJ** (2025): BGE-m3 + LLM2Vec + LLM reranking with CoT. Multi-stage with RRF fusion.
- **Unicamp/NeuralMind** (2021): Vanilla BM25 alone got **2nd place**, proving lexical baselines are strong.

### What Nobody Has Tried

1. **Systematic exploitation of `<FRAGMENT_SUPPRESSED>` context windows** — the suppressed markers are redacted citations; surrounding text reveals citation intent
2. **Contrastive fine-tuning on COLIEE training labels** with BM25-mined hard negatives — most teams use off-the-shelf models
3. **Query decomposition** — treating each citation context as a separate sub-query instead of one whole-doc query
4. **Learned meta-fusion** — LightGBM over multi-signal features (vs. simple RRF or fixed weights)

### Current Project State

- **Signal validation**: LightGBM achieves AUC=0.936 with 6 lexical features. Median per-query AUC=0.988. ~30-50 "hard" queries with AUC < 0.6.
- **Embedding benchmark**: qwen3-embedding:0.6b R@200=0.794, MRR=0.462. bge-m3 R@200=0.740, MRR=0.401.
- **LLM benchmark**: deepseek-r1:8b 100% valid JSON, 19s/doc, 3.85 concepts. Other models slower or less reliable.
- **Entity extraction**: 87/7,708 docs extracted. Quality is inconsistent — many docs produce zero concepts/tests/holdings (e.g., `000002.json`, `000125.json`).
- **Corpus**: 93.1% of files contain `<FRAGMENT_SUPPRESSED>` markers. Average ~20 markers per file. 100% have `[N]` paragraph markers.

### Why the Current GraphRAG Pipeline Needs Restructuring

The existing GraphRAG design (`2026-03-11-legal-graphrag-design.md`) has the right intuition — global community structure captures patterns invisible to local retrieval — but three components are infeasible:

1. **LLM entity extraction**: The 8B model produces empty/thin results on long documents (e.g., `000002.json`: 5,436 words → zero concepts, zero tests, zero holdings; `000125.json`: 25,146 words → zero concepts). Estimated ~40 hours for the full corpus, with unreliable output.
2. **O(n²) community detection**: The current design builds a weighted case-similarity graph via pairwise BM25 over 7,708 docs = ~30M comparisons. Infeasible within time budget.
3. **LLM reasoning chains**: Generating explanations for top-K pairs across the corpus requires millions of LLM calls. Compute-infeasible.

**However**, the GraphRAG concept is valuable if restructured as a lightweight component:
- Build the knowledge graph from **regex-extracted entities only** (statutes, judges, outcomes — fast and reliable)
- Run community detection on the **entity bipartite graph** (sparse, not O(n²))
- Use community membership as **features in a meta-learner** (not a standalone retrieval signal)
- Skip LLM summarization and reasoning chains entirely

This "GraphRAG Lite" approach preserves the core insight (global community structure) at ~30 minutes of compute instead of 24-36 hours. See Option C for full integration details.

### Additional Constraint: GraphRAG Integration

The team has decided that **GraphRAG must be part of the final pipeline** — not the traditional GNN approach (like CaseGNN), but Microsoft's GraphRAG-style knowledge graph with community detection. This is both a design constraint and a novelty opportunity: no prior COLIEE team has applied GraphRAG to legal case retrieval. The challenge is making it practical within compute and quality constraints.

---

## The Key Insight: Citation Context Windows

Looking at the raw data, each `<FRAGMENT_SUPPRESSED>` marker has surrounding text that reveals **why the citation exists**:

```
"must be reviewed according to the standard of correctness (<FRAGMENT_SUPPRESSED>; 344 N.R. 257; 2005 FCA 404)"
→ Citing authority for standard of review

"The role of the Court was established in <FRAGMENT_SUPPRESSED>, at paragraph 47:"
→ Citing foundational case for judicial review principles

"<FRAGMENT_SUPPRESSED>; 2008 FC 655, the Court wrote that there was no breach of procedural fairness if..."
→ Citing precedent for procedural fairness principle
```

From a legal perspective: every citation in a Federal Court judgment serves a specific purpose — establishing a standard, supporting a factual finding, applying a test. The context window around `<FRAGMENT_SUPPRESSED>` is a **direct signal of citation intent** that no prior COLIEE team has exploited.

From an AI perspective: this transforms a single whole-doc query into ~20 targeted sub-queries, each with a clear semantic focus. This dramatically improves retrieval precision for the "hard" queries where full-document similarity fails.

---

## Option A: Citation Context Decomposition

**Philosophy:** Novel query decomposition via citation context mining. Maximum novelty, moderate complexity.

### Pipeline

```
STAGE 1: CITATION CONTEXT EXTRACTION
  Input: Query document with <FRAGMENT_SUPPRESSED> markers
  Process:
    - Locate all <FRAGMENT_SUPPRESSED> markers
    - Extract ±150 words around each marker → "citation context window"
    - Merge overlapping windows (markers within 50 words of each other)
    - Classify each window by citation type (statutory authority / precedent / factual analogy)
      using simple heuristics (presence of "s." or "Act" → statutory; "test" or "standard" → precedent)
  Output: List of citation context windows per query (~15-25 per doc on average)
  Fallback: For the ~7% of docs without markers, use full document as single context

STAGE 2: MULTI-GRANULARITY BM25
  S1: BM25 on full document text → top-200
  S2: BM25 per citation-context window → top-50 per window
  Union and deduplicate → candidate pool (~300-500 unique candidates)

STAGE 3: DENSE RETRIEVAL (fine-tuned bi-encoder)
  Model: BAAI/bge-large-en-v1.5, fine-tuned with LoRA on COLIEE training pairs
  Training:
    - Positive pairs: (query, cited_case) from training labels
    - Hard negatives: BM25 top-50 per query that are NOT in gold set
    - Loss: InfoNCE with in-batch negatives
    - Paragraph-level: encode by [N] paragraphs, max-pool for doc score
  Inference:
    - Full-doc embedding → top-200
    - Citation-context embedding matched against paragraph-level candidate embeddings → top-50/context
  Fusion: RRF across S1, S2, S3 → top-200 candidates

STAGE 4: CROSS-ENCODER RERANKING (top-200 → top-50)
  Model: microsoft/deberta-v3-large (304M params), fine-tuned as binary classifier
  Input: (citation_context_i, candidate_paragraph_j) pairs
  For each (query, candidate):
    - Score all (context_i, paragraph_j) pairs
    - Take max score across contexts and paragraphs
  Output: top-50 candidates with cross-encoder scores

STAGE 5: SCORE AGGREGATION & THRESHOLD
  LightGBM meta-learner:
    Features: BM25_full, BM25_context_max, biencoder_score, crossencoder_score,
              statute_jaccard, shared_bigrams, domain_match,
              candidate_citation_frequency, n_contexts_matched
  Training: 5-fold GroupKFold CV, grouped by query
  Threshold: grid search to maximize micro-F1
```

### What to Build (New Code)

| Module | Description | Effort |
|--------|-------------|--------|
| `citation_context.py` | Extract and classify citation context windows from docs | Medium |
| `finetune_biencoder.py` | LoRA fine-tuning of BGE-large with hard negative mining | High |
| `finetune_crossencoder.py` | Binary classification fine-tuning of DeBERTa-v3 | High |
| `meta_learner.py` | LightGBM feature assembly and training | Low (adapt from signal_validation) |
| `run_pipeline_v2.py` | End-to-end pipeline orchestration | Medium |

### What to Reuse

- `preprocess.py` — text cleaning
- `bm25.py` — BM25 indexing (extend for per-context queries)
- `embed.py` — embedding infrastructure (replace Ollama with HuggingFace for fine-tuned model)
- `extract_regex.py` — statute extraction for overlap feature
- `metrics.py` — micro-F1, threshold optimization
- `normalize.py` — statute normalization for Jaccard

### Compute Budget

| Step | Time Estimate |
|------|---------------|
| Citation context extraction (regex-based) | ~5 minutes |
| BM25 indexing + full-doc retrieval | ~10 minutes |
| BM25 per-context retrieval (1,678 queries × ~20 contexts) | ~30 minutes |
| Bi-encoder fine-tuning (LoRA, BGE-large) | ~4-6 hours |
| Dense embedding of 7,708 docs (paragraph-level) | ~2-3 hours |
| Dense retrieval | ~20 minutes |
| Cross-encoder fine-tuning | ~3-4 hours |
| Cross-encoder inference (200 candidates × 1,678 queries) | ~12-24 hours |
| LightGBM training + CV | ~5 minutes |
| **Total** | **~2-3 days** |

### Strengths

- **Genuinely novel**: Citation context decomposition has not been attempted in prior COLIEE work. Publishable contribution.
- **Strong inductive bias**: The citation context windows directly capture citation intent — the best possible signal for this task.
- **~20x more retrieval queries per document**: Instead of matching one 5K-word query against the corpus, you match ~20 targeted 300-word sub-queries. This catches citations that would be lost in full-doc noise.
- **Fine-tuning on COLIEE data**: Contrastive bi-encoder and cross-encoder adapted to the actual task distribution.

### Risks

- **Citation context extraction quality**: Some markers appear in dense clusters (3-4 in one sentence). Merging heuristics need tuning.
- **7% of docs lack markers**: Fallback to full-doc query is necessary but may underperform for those queries.
- **Cross-encoder inference is the bottleneck**: 200 candidates × 1,678 queries × ~20 (context, paragraph) pairs = significant compute. May need to reduce to top-100 candidates or sample paragraphs.
- **Fine-tuning with limited data**: Only ~1,678 queries × ~4.1 citations = ~6,881 positive pairs. Risk of overfitting mitigated by GroupKFold CV and LoRA (fewer trainable params).

### Expected Outcome

- **F1: 0.38-0.44**
- **Placement: 2nd-4th**
- **Paper quality: Excellent** (novel method with clear ablation story)

---

## Option B: Contrastive Legal Retriever + LLM Reranker

**Philosophy:** Follow the proven winning formula with modern components. Minimum risk, maximum reliability.

### Pipeline

```
STAGE 1: BM25 FIRST STAGE
  Standard BM25 (k1=1.5, b=0.75) → top-200 per query

STAGE 2: CONTRASTIVE BI-ENCODER (fine-tuned)
  Model: BAAI/bge-large-en-v1.5 or intfloat/e5-large-v2
  Training:
    - Positive pairs: (query, cited_case) from training labels
    - Hard negatives: BM25 top-50 minus gold set (proven difficult negatives)
    - Loss: InfoNCE with in-batch negatives + hard negatives
    - Paragraph-level encoding: encode by [N] paragraphs, attention-weighted aggregation
    - LoRA fine-tuning (rank=16, ~6 hours on GB10)
  Inference: rerank top-200 from BM25

STAGE 3: CROSS-ENCODER RERANKER (fine-tuned)
  Model: microsoft/deberta-v3-large (304M params)
  Training:
    - Input: [CLS] query_excerpt [SEP] candidate_excerpt [SEP]
    - Binary classification (cited / not cited)
    - Multiple excerpt pairs per doc pair → max pool scores
    - Balanced sampling: 1 positive + 3 hard negatives per query
  Inference: rerank top-50 from Stage 2

STAGE 4: LLM VERIFICATION (top-15 only)
  Model: Qwen2.5-32B-Instruct (4-bit quantized, ~20GB)
  Prompt:
    "You are a Federal Court of Canada law clerk. Given Query Case A
     and Candidate Case B (excerpts below), would a judge cite Case B
     when writing Case A?

     Analyze:
     1. Do they share legal issues or statutory provisions?
     2. Do they apply similar legal tests or standards?
     3. Is there factual analogy?

     Rate relevance 1-5 and explain briefly."
  Processing: 1,678 queries × 15 candidates = ~25,170 LLM calls
  Time: ~30s per call → ~8-9 days sequential → must batch/parallelize

STAGE 5: ENSEMBLE + THRESHOLD
  RRF or learned weights over [BM25, bi-encoder, cross-encoder, LLM] scores
  Alternative: LightGBM meta-learner on all scores
  Threshold: grid search on 5-fold GroupKFold CV
```

### What to Build (New Code)

| Module | Description | Effort |
|--------|-------------|--------|
| `finetune_biencoder.py` | LoRA fine-tuning with hard negative mining | High |
| `finetune_crossencoder.py` | DeBERTa-v3 binary classification fine-tuning | High |
| `llm_reranker.py` | LLM verification prompting + score extraction | Medium |
| `ensemble.py` | Score fusion (RRF / LightGBM) | Low |
| `run_pipeline_v2.py` | End-to-end orchestration | Medium |

### Compute Budget

| Step | Time Estimate |
|------|---------------|
| BM25 indexing + retrieval | ~10 minutes |
| Bi-encoder fine-tuning (LoRA) | ~4-6 hours |
| Dense embedding + retrieval | ~3 hours |
| Cross-encoder fine-tuning | ~3-4 hours |
| Cross-encoder inference (50 × 1,678) | ~6-8 hours |
| LLM reranking (25K calls at ~30s each) | ~8-9 days (bottleneck) |
| Ensemble + threshold | ~10 minutes |
| **Total** | **~10-11 days** (dominated by LLM stage) |

**Note:** LLM stage can be reduced by:
- Reducing to top-10 candidates (16,780 calls → ~5-6 days)
- Using a smaller model (Qwen2.5-7B: ~5s/call → ~1.5 days)
- Dropping LLM stage entirely and relying on cross-encoder (saves ~8 days, costs ~1-2% F1)

### Strengths

- **Proven formula**: Every component (BM25 → bi-encoder → cross-encoder → LLM) has been validated by prior winners.
- **Fine-tuning is the key differentiator**: Most COLIEE teams use off-the-shelf models. Fine-tuning on COLIEE labels with hard negatives is the single highest-impact thing available.
- **Low methodological risk**: Each stage adds value; pipeline degrades gracefully if any stage underperforms.
- **LLM reranking captures reasoning**: The 30-50 "hard" queries where lexical similarity fails may be solvable by LLM reasoning over legal principles.

### Risks

- **Low novelty for the paper**: "BM25 + fine-tuned BERT + LLM reranker" is an incremental improvement, not a publishable contribution. COLIEE prizes papers alongside results.
- **LLM reranking compute**: 25K calls at 30s each = 8+ days. This is the bottleneck and may need to be dropped or approximated.
- **Qwen2.5-32B reliability**: Quantized 32B model may produce inconsistent scores. Calibration needed.
- **Fine-tuning overfitting**: 6,881 positive pairs is small for contrastive learning. LoRA and GroupKFold mitigate but don't eliminate this risk.

### Expected Outcome

- **F1: 0.36-0.42**
- **Placement: 3rd-5th**
- **Paper quality: Average** (solid engineering, limited novelty)

---

## Option C: Hybrid Multi-View with Citation Context + GraphRAG Lite (RECOMMENDED)

**Philosophy:** Combine Option A's novel citation context mining with Option B's proven fine-tuning pipeline, plus a lightweight GraphRAG community signal. Best expected value across both F1 and paper quality. Satisfies the GraphRAG integration constraint.

### Pipeline

```
STAGE 1: MULTI-VIEW QUERY REPRESENTATION

  View 1 — Full Document:
    Preprocessed full text (strip FRAGMENT_SUPPRESSED, rejoin statutes, collapse whitespace)

  View 2 — Citation Context Windows:
    For each <FRAGMENT_SUPPRESSED> marker:
      - Extract ±150 words around marker
      - Merge windows within 50 words of each other
      - Classify by type (statutory / precedent / factual) via heuristic
    ~15-25 windows per query on average
    Fallback for docs without markers: extract key paragraphs by [N] markers

  View 3 — Statute Profile:
    Extracted via regex (fast, reliable, already implemented in extract_regex.py)
    Normalized via normalize.py → canonical statute list

  View 4 — Legal Issues Summary (optional, one LLM call per query):
    "List the 3-5 key legal issues decided in this case."
    Used as a short-form semantic query (100-200 words vs. 5,000+)

STAGE 2: MULTI-SIGNAL CANDIDATE GENERATION (→ top-200)

  S1: BM25 full-document → top-200
      Standard BM25 (k1=1.5, b=0.75) on preprocessed full text

  S2: BM25 per-citation-context → union of top-30 per context window
      Each citation context window is a separate BM25 query
      Union all results, keep max BM25 score per candidate

  S3: Dense bi-encoder → top-200
      Model: BAAI/bge-large-en-v1.5, fine-tuned with LoRA on COLIEE pairs
      Hard negatives: BM25 top-50 minus gold set
      Paragraph-level encoding with max-pool aggregation

  S4: Statute overlap → score all candidates
      Jaccard similarity of normalized statute sets between query and candidate
      Cheap to compute, reliable signal (proven in signal validation)

  S5: GraphRAG Lite community signal → score all candidates
      (See "GraphRAG Lite Component" section below for full details)
      Provides: same_community, community_jaccard, community_entity_sim
      Cheap to compute once graph is built (~30 min total indexing)

  Fusion: Reciprocal Rank Fusion across S1-S5 → top-200 candidates

STAGE 3: CROSS-ENCODER RERANKING (top-200 → top-50)

  Model: microsoft/deberta-v3-large, fine-tuned as binary classifier

  Scoring strategy:
    For each (query, candidate) pair:
      - Form (citation_context_i, candidate_paragraph_j) pairs
      - Score each pair with cross-encoder
      - Aggregate: max score across all (context, paragraph) combinations
    This naturally captures which specific citation context matches which
    specific paragraph — the most granular relevance signal possible.

  Optimization: batch high-scoring BM25 candidates first; skip low-BM25 pairs
  Output: top-50 candidates ranked by cross-encoder score

STAGE 4: META-LEARNER (top-50 → final predictions)

  Model: LightGBM (already proven AUC=0.936 on 6 lexical features)

  Features per (query, candidate) pair:
    Retrieval signals:
      - bm25_full_score          (from S1)
      - bm25_context_max_score   (from S2, max across citation contexts)
      - bm25_context_mean_score  (from S2, mean across citation contexts)
      - biencoder_score          (from S3)
      - statute_jaccard          (from S4)

    Cross-encoder signals:
      - crossencoder_max_score   (from Stage 3)
      - crossencoder_mean_score  (from Stage 3)
      - n_context_paragraph_matches_above_threshold  (count of high-scoring pairs)

    Lexical features:
      - tfidf_cosine             (full document)
      - shared_bigrams_jaccard   (full document)

    GraphRAG community features:
      - same_community           (binary: query and candidate in same Leiden community)
      - community_jaccard        (Jaccard of community sets if overlapping/multi-resolution)
      - community_entity_sim     (cosine of entity-type frequency vectors between communities)
      - community_size_norm      (1/community_size — penalize uninformative large communities)

    Metadata features:
      - candidate_citation_frequency  (how often this doc is cited in training set)
      - domain_match             (binary: same legal domain)
      - length_ratio             (min/max word count)

  Training: 5-fold GroupKFold CV, grouped by query ID
  Class balancing: is_unbalance=True

  Output: probability score per (query, candidate) pair

STAGE 5: THRESHOLD OPTIMIZATION & SUBMISSION

  For each fold, grid search threshold (0.1-0.9, step 0.01) to maximize micro-F1
  Average threshold across folds → final threshold

  Three submission runs:
    Run 1: Full Option C pipeline (best expected F1)
    Run 2: Ablation — remove citation context signals (S2, bm25_context_*, n_context_matches)
            Purpose: prove citation context adds value → paper contribution
    Run 3: Different threshold (conservative: +0.05 above optimal)
            Purpose: hedge against train/test distribution shift
```

### Architecture Diagram

```
Query Document
    │
    ├─────────────────────────┬──────────────────┬───────────────────┐
    ▼                         ▼                  ▼                   ▼
 Full Text              Citation Contexts    Statute List       Legal Issues
 (View 1)               (View 2)           (View 3)           (View 4, optional)
    │                         │                  │                   │
    ▼                         ▼                  ▼                   │
 BM25 Full (S1)        BM25 per-Context    Statute Jaccard        │
    │                    (S2)               (S4)                    │
    │                         │                  │                   │
    └──────────┬──────────────┘                  │                   │
               │                                 │                   │
    Dense Bi-Encoder (S3) ◄──────────────────────┘                   │
    (fine-tuned on COLIEE)                                           │
               │                                                     │
               ▼                                                     │
        RRF Fusion → top-200 candidates                              │
               │                                                     │
               ▼                                                     │
     Cross-Encoder Reranking                                         │
     (citation_context × candidate_paragraph)                        │
               │                                                     │
               ▼                                                     │
        top-50 candidates                                            │
               │                                                     │
               ▼                                                     │
     LightGBM Meta-Learner ◄────────────────────────────────────────┘
     (17 features from all stages incl. GraphRAG community)
               │
               ▼
     Threshold → Binary Predictions → Submission
```

### GraphRAG Lite Component (Signal S5)

This is the lightweight GraphRAG integration that satisfies the design constraint without the compute-killing components of the original design. See `2026-03-11-graphrag-literature-review.md` for the full 44-paper literature review backing these decisions.

#### Why GraphRAG (Not GNN)

- **GNN (CaseGNN-style)**: CaseLink (GNN) got 2nd place at COLIEE 2025 (F1=0.2962, SIGIR 2024). However, it requires actual citation edges between cases. At test time, citation edges are unknown (that's what we're predicting). CaseGNN/CaseLink used training citation edges to learn embeddings, creating a train/test information leak for new query cases. GNNs also require significant training infrastructure.
- **GraphRAG (Microsoft-style)**: Builds a knowledge graph from document content, detects communities, uses community structure for retrieval. Works at test time because it relies on entity extraction (not citation edges). The key insight: community-level patterns capture global topical structure that local pairwise retrieval misses.

**Important caveat from literature review:** Microsoft GraphRAG was designed for **global summarization, NOT document retrieval**. Multiple rigorous evaluations (ICLR 2026 GraphRAG-Bench, arXiv 2506.06331 unbiased evaluation) found gains are "much more moderate than previously reported" and it "frequently underperforms vanilla RAG" on standard tasks. We adopt the useful subcomponents (entity graph, community detection) while discarding the QA-focused components (community summarization, map-reduce generation).

#### Why "Lite" (Literature-Backed)

The full GraphRAG pipeline is infeasible AND unnecessary:
- **LLM extraction is noisy:** Our benchmarks show empty results on long docs. Production RAG lessons (TDS) confirm auto-extracted entities are "visually appealing but impractical." Open-source LLMs fail ~50% on structured extraction (Ollama pitfalls guide).
- **SpaCy/regex achieves 94% of LLM quality:** SAP "Practical GraphRAG" (arXiv 2507.03226) demonstrated dependency parsing achieves 61.87% vs 65.83% for LLM extraction, at a fraction of the cost.
- **E2GraphRAG (arXiv 2505.24226):** SpaCy entity extraction + co-occurrence relations = 10x faster indexing than GraphRAG, 100x faster retrieval than LightRAG, competitive quality.
- **LazyGraphRAG (Microsoft Research):** No LLM during indexing. Indexing cost is 0.1% of full GraphRAG. Quality comparable to full GraphRAG global search.
- **CommunityKG-RAG (arXiv 2408.08535):** Louvain community detection on entity graphs effectively narrows search space, validated for fact-checking at zero-shot (no training needed).

GraphRAG Lite follows the E2GraphRAG/Practical GraphRAG pattern: regex+SpaCy entity extraction → co-occurrence graph → Leiden communities → features for meta-learner. No LLM calls required.

#### Optional Enhancement: GliNER for Legal Concept Extraction

**GliNER** (NAACL 2024, github.com/urchade/GLiNER) is a zero-shot NER model that **outperforms ChatGPT on NER benchmarks** and runs on CPU/consumer hardware. It could supplement regex extraction to catch legal concepts (e.g., "reasonableness standard", "procedural fairness") that regex misses, without LLM cost. This is a P1 enhancement — regex-only is the P0 baseline.

#### Construction Pipeline

```
STEP 1: KNOWLEDGE GRAPH CONSTRUCTION (~15 minutes, CPU-only)

  Entity extraction (regex-only, already implemented):
    - Statutes: extract_regex.extract_statutes() → normalize_statute()
    - Judges: extract_regex.extract_judges() → normalize_judge()
    - Outcomes: extract_regex.extract_outcome()
    - Domain: keyword heuristic classifier
      (presence of "immigration"/"refugee" → immigration,
       "tax"/"income" → tax, "patent"/"trademark" → IP, etc.)

  Graph construction (NetworkX):
    Node types:
      - Case (7,708 nodes)
      - Statute (~500-1,000 unique normalized statutes)
      - Judge (~200-400 unique judges)
      - Domain (~10-15 categories)

    Edge types:
      - Case → APPLIES → Statute
      - Case → DECIDED_BY → Judge
      - Case → IN_DOMAIN → Domain

    Expected graph: ~9K-10K nodes, ~50K-80K edges

STEP 2: BIPARTITE PROJECTION → CASE-CASE GRAPH (~5 minutes)

  Project the bipartite (Case ↔ Entity) graph into a weighted Case-Case graph:
    For each pair of cases (A, B) that share at least one entity:
      weight = Σ over shared entities of entity_type_weight:
        - shared_statute:  0.50 per shared statute
        - shared_judge:    0.15 per shared judge
        - shared_domain:   0.10 per shared domain
        - shared_outcome:  0.05 per shared outcome type

  This is SPARSE by construction:
    - Most case pairs share zero entities → no edge
    - Only cases with overlapping statute/judge profiles get connected
    - Expected: ~100K-300K edges (not 30M)
    - Stored as sparse adjacency matrix

STEP 3: COMMUNITY DETECTION (~10 minutes)

  Algorithm: Leiden (already implemented in community.py, needs adaptation)
  Input: weighted Case-Case graph from Step 2
  Resolution parameter: tuned for ~100-300 communities
  Multi-resolution (optional): run at 2-3 resolution values for overlapping communities

  Output per case:
    - Primary community ID
    - Community size
    - Community entity profile (frequency vector of entity types within community)

  No LLM summarization — community identity is captured by:
    - The set of statutes most frequently cited by community members
    - The dominant domain(s)
    - The most common judges
    These are computed as simple frequency counts, not LLM calls.

STEP 4: COMMUNITY FEATURES FOR META-LEARNER

  For each (query, candidate) pair, compute:

  1. same_community (binary):
     Are query and candidate in the same primary Leiden community?
     Intuition: cases in the same community share legal topic, making citation more likely.

  2. community_jaccard (float, 0-1):
     If using multi-resolution communities, Jaccard of community sets.
     Captures partial community overlap.

  3. community_entity_sim (float, 0-1):
     Cosine similarity of entity-type frequency vectors between query's community
     and candidate's community.
     Example: query's community is 60% immigration statutes, 30% procedural fairness.
     Candidate's community is 55% immigration statutes, 35% procedural fairness.
     → High similarity even if different primary community IDs.

  4. community_size_norm (float):
     1 / community_size. Penalizes large, uninformative communities.
     A community of 50 cases sharing IRPA statutes is more informative than
     a community of 2,000 cases sharing generic "Federal Court Act" references.
```

#### What This Preserves from the Original GraphRAG Design

| Original Component | Status | Rationale |
|-------------------|--------|-----------|
| Knowledge graph with typed nodes/edges | **Kept** | Built from regex (fast, reliable) |
| Statute normalization + alias mapping | **Kept** | Already implemented in `normalize.py` |
| Leiden community detection | **Kept** | Adapted to bipartite projection (sparse, not O(n²)) |
| Community membership as retrieval signal | **Kept** | Core GraphRAG insight preserved |
| Entity type weights | **Kept** | Tuned weights for bipartite projection |
| LLM entity extraction | **Dropped** | Too noisy, too slow |
| O(n²) pairwise BM25 similarity graph | **Dropped** | Replaced by sparse bipartite projection |
| LLM community summarization | **Dropped** | Replaced by frequency-count profiles |
| LLM reasoning chains | **Dropped** | Compute-infeasible |
| Concept clustering (embedding-based) | **Dropped** | Requires reliable LLM extraction |

#### Expected Impact (Literature-Backed)

- **On easy queries (AUC > 0.95, ~85% of queries)**: Minimal impact — BM25 and embeddings already handle these. Community features may add marginal signal via LightGBM.
- **On hard queries (AUC < 0.6, ~30-50 queries)**: This is where GraphRAG earns its keep. CommunityKG-RAG (arXiv 2408.08535) showed community-based retrieval narrows search space effectively. SemToG showed semantic-aware communities give consistent 2-5% accuracy gains. These queries fail on lexical similarity because citation is based on shared legal principles — community membership provides a "soft topic match" signal.
- **Hybrid retrieval with graph features adds +5-15%**: SAP Practical GraphRAG showed +12% context precision and -32% "no coverage" answers when adding graph signals to vector retrieval via RRF. The financial domain GraphRAG (ACL 2025) showed 6% hallucination reduction. Smart-Slic (ICAIL 2025) showed hybrid RAG-KG achieved 70% vs. 37.5% for RAG-only on legal data.
- **For the paper**: "Citation Context Decomposition + GraphRAG Community Features" is a significantly stronger narrative than either alone. Two novel components with ablation = excellent paper. No prior COLIEE team has applied either technique.

#### Compute Budget (GraphRAG Lite only)

| Step | Time | Hardware |
|------|------|----------|
| Regex entity extraction (7,708 docs) | ~5 minutes | CPU |
| Statute/judge normalization | ~1 minute | CPU |
| Knowledge graph construction (NetworkX) | ~2 minutes | CPU |
| Bipartite projection → Case-Case graph | ~5 minutes | CPU |
| Leiden community detection | ~5 minutes | CPU |
| Community feature computation | ~2 minutes | CPU |
| **Total** | **~20 minutes** | **CPU-only** |

### What to Build (New Code)

| Module | Description | Lines (est.) | Effort | Priority |
|--------|-------------|-------------|--------|----------|
| `citation_context.py` | Extract citation context windows from docs, merge overlapping, classify type | ~150 | Medium | P0 |
| `graphrag_lite.py` | Build KG from regex entities, bipartite projection, Leiden communities, compute community features | ~250 | Medium | P0 |
| `finetune_biencoder.py` | LoRA fine-tuning of BGE-large with InfoNCE + hard negatives | ~250 | High | P0 |
| `finetune_crossencoder.py` | DeBERTa-v3 binary classification fine-tuning on (context, paragraph) pairs | ~250 | High | P0 |
| `meta_learner.py` | Feature assembly from all stages (incl. GraphRAG), LightGBM training, threshold search | ~250 | Medium | P0 |
| `run_pipeline_v2.py` | End-to-end orchestration replacing current `run_pipeline.py` | ~350 | Medium | P0 |
| `paragraph_index.py` | Paragraph-level BM25 and embedding index for candidates | ~100 | Low | P1 |
| `legal_issues.py` | LLM-based query summarization (View 4, optional) | ~80 | Low | P2 |

**Total new code: ~1,680 lines**

### What to Reuse from Current Codebase

| Existing Module | Reuse For | Modifications |
|-----------------|-----------|---------------|
| `preprocess.py` | Text cleaning, paragraph splitting, chunking | None |
| `bm25.py` | BM25 indexing and retrieval | Extend `query()` to accept short texts (citation contexts) |
| `embed.py` | Embedding infrastructure | Replace Ollama backend with HuggingFace Transformers for fine-tuned model |
| `extract_regex.py` | Statute/judge/outcome extraction → feeds both S4 and GraphRAG Lite | None |
| `normalize.py` | Statute/judge normalization → clean KG nodes | None |
| `graph.py` | Knowledge graph construction | **Simplify**: remove LLM entity node types (concepts, tests, holdings), keep Case/Statute/Judge/Domain nodes |
| `community.py` | Leiden community detection | **Rewrite**: use bipartite projection instead of O(n²) pairwise BM25 graph |
| `metrics.py` | Micro-F1, threshold optimization | None |
| `config.py` | Central configuration | Add new config params for fine-tuning, cross-encoder, meta-learner, GraphRAG Lite |
| `retrieve.py` | RRF fusion function | Reuse `reciprocal_rank_fusion()`, add S5 signal |

### What to Drop from Current Pipeline

| Module | Reason |
|--------|--------|
| `extract_llm.py` | Too noisy (empty concepts on large docs), too slow (19s/doc × 7,708 = 40+ hours). GraphRAG Lite uses regex extraction only. |
| `reasoning.py` | LLM reasoning chains infeasible (millions of calls) |
| `run_index.py` | Orchestrates dropped components; replaced by `run_pipeline_v2.py` |
| `run_extract.py` | LLM extraction orchestration; replaced by regex-only extraction in `graphrag_lite.py` |

**Note:** `graph.py` and `community.py` are **simplified and reused**, not dropped. The regex extraction (`extract_regex.py`) is the backbone of both the statute overlap signal (S4) and the GraphRAG Lite knowledge graph (S5).

### Compute Budget

| Step | Time Estimate | Hardware |
|------|---------------|----------|
| Citation context extraction (all docs) | ~5 minutes | CPU |
| BM25 indexing (full corpus) | ~10 minutes | CPU |
| BM25 full-doc retrieval (1,678 queries) | ~5 minutes | CPU |
| BM25 per-context retrieval (~30K sub-queries) | ~30 minutes | CPU |
| Bi-encoder fine-tuning (LoRA, BGE-large, 335M) | ~4-6 hours | GPU |
| Dense embedding of corpus (paragraph-level) | ~2-3 hours | GPU |
| Dense retrieval (similarity search) | ~20 minutes | CPU |
| Cross-encoder fine-tuning (DeBERTa-v3-large, 304M) | ~3-4 hours | GPU |
| Cross-encoder inference (200 × 1,678 queries) | ~12-24 hours | GPU |
| **GraphRAG Lite: regex extraction (7,708 docs)** | **~5 minutes** | **CPU** |
| **GraphRAG Lite: KG build + bipartite projection** | **~7 minutes** | **CPU** |
| **GraphRAG Lite: Leiden community detection** | **~5 minutes** | **CPU** |
| **GraphRAG Lite: community feature computation** | **~2 minutes** | **CPU** |
| LightGBM meta-learner training + CV | ~5 minutes | CPU |
| Threshold optimization | ~1 minute | CPU |
| **Total** | **~2-3 days** | |

**Note:** GraphRAG Lite adds only ~20 minutes of CPU-only compute to the pipeline. It runs in parallel with BM25 indexing since both depend only on the raw corpus.

### Dependencies (New)

```toml
# Add to pyproject.toml
transformers = ">=4.45"
torch = ">=2.1"
peft = ">=0.13"       # LoRA fine-tuning
datasets = ">=3.0"     # HuggingFace datasets for training
sentence-transformers = ">=3.0"  # Bi-encoder training utilities
accelerate = ">=1.0"   # Training acceleration
```

### Implementation Order

```
Phase 1: Foundation (~1 day)
  1. citation_context.py + tests
  2. Extend bm25.py for per-context queries
  3. Validate citation context extraction quality on sample docs

Phase 1b: GraphRAG Lite (~0.5 day, can parallel with Phase 1)
  4. graphrag_lite.py — regex extraction → KG → bipartite projection → Leiden → features
  5. Simplify graph.py — remove LLM entity types, keep Case/Statute/Judge/Domain
  6. Rewrite community.py — bipartite projection instead of O(n²) pairwise
  7. Validate: inspect community quality (are immigration cases grouped together? tax cases?)

Phase 2: Fine-tuning (~2 days)
  8. finetune_biencoder.py — LoRA on BGE-large with hard negatives
  9. finetune_crossencoder.py — DeBERTa-v3 binary classification
  10. Validate both on held-out fold

Phase 3: Integration (~1 day)
  11. meta_learner.py — feature assembly (incl. GraphRAG community features) + LightGBM
  12. run_pipeline_v2.py — end-to-end orchestration
  13. 5-fold CV evaluation → micro-F1

Phase 4: Ablation & Submission (~1 day)
  14. Run ablations:
      a. Full pipeline (all signals)
      b. Without citation context signals (prove citation context value)
      c. Without GraphRAG community signals (prove GraphRAG value)
  15. Generate 3 submission runs
  16. Write paper section on method
```

### Strengths

1. **Two novel, publishable contributions**: Citation context decomposition AND GraphRAG community features for legal retrieval. Neither has been attempted in prior COLIEE work. The ablation design (with/without each) provides a clean contribution story for the paper.

2. **Proven backbone**: Fine-tuned bi-encoder + cross-encoder follows the exact formula used by 4 consecutive THUIR wins and the 2025 SOTA. This is not speculative.

3. **Data-driven fusion**: The LightGBM meta-learner is already validated at AUC=0.936 with 6 features. Adding neural features (bi-encoder, cross-encoder) and GraphRAG community features (4 new features) should push it significantly higher. The meta-learner automatically learns optimal signal weights — if GraphRAG features don't help, LightGBM assigns them zero importance. No downside risk.

4. **Multi-granularity matching**: Matching citation contexts against paragraphs captures relevance at the right semantic level. A 150-word citation context about "standard of reasonableness" matching a specific paragraph in the candidate case is far more precise than 5,000-word document-level similarity.

5. **GraphRAG targets the hard queries**: The ~30-50 queries with per-query AUC < 0.6 are where lexical similarity fails. Community membership provides a "soft topic match" signal for these queries — if query and candidate share a Leiden community (e.g., "immigration procedural fairness"), that's a strong citation signal even when BM25 scores are low.

6. **Compute-feasible**: Total ~2-3 days on GB10. GraphRAG Lite adds only ~20 minutes of CPU compute. No LLM calls for graph construction.

7. **Graceful degradation**: Each stage adds value independently. If cross-encoder fine-tuning fails, BM25 + bi-encoder + GraphRAG + LightGBM still competes. If citation context extraction is noisy, the full-doc and community signals carry the pipeline.

8. **Three-run ablation strategy**: Run 1 (full pipeline), Run 2 (without citation context), Run 3 (without GraphRAG OR conservative threshold) maximizes both paper quality and placement probability.

### Risks

1. **Cross-encoder is the compute bottleneck**: 200 candidates × 1,678 queries × multiple (context, paragraph) pairs. Mitigation: early stopping on low-BM25 pairs, paragraph sampling, batch optimization.

2. **Fine-tuning with limited data**: ~6,881 positive pairs may be insufficient for contrastive learning. Mitigation: LoRA reduces parameter count; GroupKFold prevents leakage; data augmentation via paragraph-level pairs.

3. **Citation context quality varies**: Some markers are in dense clusters; some contexts are formulaic ("as stated in FRAGMENT_SUPPRESSED"). Mitigation: merging heuristics, filtering short/generic contexts, fallback to full-doc.

4. **Train/test distribution shift**: The noise analysis shows train files have 16.1% noise vs. test 5.6%. Models trained on noisy train data may underperform on cleaner test data. Mitigation: preprocessing normalizes both; conservative threshold (Run 3) hedges.

5. **Dependency on HuggingFace models**: Fine-tuning requires `transformers` + `torch` + `peft` — significant new dependencies. Mitigation: these are standard ML libraries; well-documented, stable APIs.

### Expected Outcome

- **F1: 0.40-0.46**
- **Placement: 1st-3rd**
- **Paper quality: Excellent** (novel method + ablation + competitive results)

---

## Comparison Matrix

| Criterion | Option A | Option B | **Option C** |
|-----------|----------|----------|-------------|
| **Novelty** | High | Low | **High** |
| **Expected F1** | 0.38-0.44 | 0.36-0.42 | **0.40-0.46** |
| **Implementation effort** | Medium | Medium | **Medium-High** |
| **Compute time** | ~2 days | ~10 days | **~2-3 days** |
| **Risk level** | Medium | Low | **Medium-Low** |
| **Paper quality** | Excellent | Average | **Excellent** |
| **Expected placement** | 2nd-4th | 3rd-5th | **1st-3rd** |
| **Codebase reuse** | High | Medium | **High** |
| **Graceful degradation** | Medium | High | **High** |

---

## Recommendation

**Option C is the recommended approach.** It combines two novel contributions (citation context decomposition + GraphRAG community features) with the most proven competitive backbone (fine-tuned bi-encoder + cross-encoder + learned fusion), all within a feasible compute budget. It satisfies the GraphRAG integration constraint without the compute-killing components of the original design.

The three-run ablation strategy (full pipeline / without citation context / without GraphRAG or conservative threshold) maximizes expected value across both competition placement and paper quality.

### Decision Points for Reviewer

1. **Should View 4 (LLM legal issues summary) be included?** It adds ~1,678 LLM calls (~9 hours) for a potentially marginal signal. Could be P2/optional.

2. **Should the LLM reranker from Option B be added as a final stage?** It adds significant compute (~8 days) for potentially +1-2% F1. Could be a Run 3 variant instead of conservative threshold.

3. **Which bi-encoder base model?** BGE-large-en-v1.5 (335M, well-benchmarked) vs. E5-large-v2 (335M, slightly newer) vs. BGE-m3 (568M, multilingual, already benchmarked at R@200=0.740).

4. **Paragraph-level vs. document-level cross-encoder?** (citation_context, paragraph) pairs are more granular but more expensive. (citation_context, full_doc_excerpt) pairs are cheaper but less precise.

5. **GraphRAG Lite: entity type weights for bipartite projection.** Current proposal: statute=0.50, judge=0.15, domain=0.10, outcome=0.05. Should these be tuned via CV? Or is a simple unweighted count sufficient?

6. **GraphRAG Lite: Leiden resolution parameter.** Targeting ~100-300 communities. Too few = uninformative mega-communities; too many = every case in its own community. Multi-resolution (run at 2-3 values) gives overlapping communities but adds complexity.

7. **GraphRAG Lite: should domain classification use LLM or keyword heuristics?** Keyword heuristics (~0.5s per doc) are fast but coarse. A single LLM call per doc for domain classification (~19s × 7,708 = ~40 hours) is accurate but slow. Could compromise: keyword heuristics for the majority, LLM for ambiguous docs only.

8. **Should GliNER (NAACL 2024) supplement regex extraction?** GliNER is a zero-shot NER model that outperforms ChatGPT on NER benchmarks, runs on CPU, and could catch legal concepts regex misses (e.g., "standard of review", "procedural fairness"). Adds ~30 min compute. Potential P1 enhancement.

9. **Should a GNN reranker be added as Phase 2?** G-RAG (arXiv 2405.18414) showed GNN reranking outperforms PaLM 2 as a reranker. CaseLink (SIGIR 2024) used GNN for case representation and got 2nd at COLIEE 2025. A lightweight GNN reranker over the entity graph could complement the cross-encoder. Significant additional implementation effort.

10. **Run 3 strategy: ablation without GraphRAG vs. conservative threshold?** Ablation gives better paper data. Conservative threshold gives better placement hedge. Which matters more?

---

## Appendix: Task 2 Considerations

This document focuses on Task 1, but the same citation context insight applies to Task 2 (Legal Case Entailment):

- Task 2 asks: which paragraph from a precedent case supports a given decision fragment?
- The cross-encoder trained on (citation_context, paragraph) pairs for Task 1 directly transfers to Task 2
- The fine-tuned bi-encoder can encode both the entailed fragment and candidate paragraphs
- A Task 2 submission would primarily reuse Task 1's models with a thin adaptation layer

A separate design document should be created for Task 2 if the team decides to compete on both tasks.

---

*Document version: 1.2*
*Created: 2026-03-11*
*Updated: 2026-03-11 — v1.1: Added GraphRAG Lite integration per team constraint*
*Updated: 2026-03-11 — v1.2: Literature-backed GraphRAG Lite design (44 papers reviewed, see `2026-03-11-graphrag-literature-review.md`)*
*Status: PENDING REVIEW*
