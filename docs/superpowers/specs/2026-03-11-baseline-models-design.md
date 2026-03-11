# Design Spec: SoTA Baseline Models for COLIEE 2026 Task 1

**Date**: 2026-03-11
**Status**: Reviewed (v2)
**Author**: Claude Code + user
**Purpose**: Faithful reproduction of 5 SoTA baseline models for fair comparison against our GraphRAG pipeline

---

## 1. Overview

### Goal
Implement 5 state-of-the-art baseline models from COLIEE 2024-2025 competition winners/runners-up, evaluated under identical conditions, to benchmark our GraphRAG pipeline against proven approaches.

### Baselines

| ID | Name | Source | Task 1 F1 | Year |
|----|------|--------|-----------|------|
| E | Vanilla BM25 | Standard IR | ~0.19 | 2021 |
| A | BM25 + SAILER + LightGBM | JNLP 2025 (1st) | 0.3353 | 2025 |
| B | LTR Fusion | TQM 2024 (1st) | 0.4432 | 2024 |
| C | Propositions + Judge + FF NN | UMNLP 2024 (2nd) | 0.4134 | 2024 |
| D | CaseLink GNN | UQLegalAI 2025 (2nd) | 0.2962 | 2025 |

### Constraints
- **Open-source models only** (competition requirement)
- **Faithful reproductions** — match paper methods as closely as possible
- **Competition-mimic evaluation** — fixed train/val split, micro-averaged F1
- **SAILER**: Use pre-trained HuggingFace checkpoint (`CSHaitao/SAILER_en_finetune`), not trained from scratch. Reference `github.com/CSHaitao/SAILER` for correct inference code (asymmetric encoder-decoder architecture).
- **Python 3.12**, executed via `uv run`

### Reference Papers
- Paper 26 (TQM): arXiv 2404.00947
- Paper 27 (UMNLP): GitHub `dc435/COLIEE_2024_Task1`
- Paper 29 (JNLP): COLIEE 2025 Proceedings
- Paper 30 (CaseLink): arXiv 2505.20743, GitHub `yanran-tang/CaseLink`
- Paper 15 (BM25): arXiv 2105.05686

---

## 2. Architecture

### Directory Structure

```
src/baselines/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── base_model.py        # BaselineModel ABC
│   ├── data_loader.py       # Corpus loading, train/val split
│   ├── bm25_index.py        # Shared BM25 index
│   ├── eval_harness.py      # Run baselines, comparison table
│   └── metrics.py           # Micro F1, threshold optimization, bootstrap test
├── bm25/
│   ├── __init__.py
│   └── model.py             # Vanilla BM25 baseline
├── jnlp/
│   ├── __init__.py
│   ├── sailer_encoder.py    # SAILER checkpoint download + encoding + cache
│   ├── features.py          # QLD, dynamic BM25 features
│   └── model.py             # LightGBM reranker
├── tqm/
│   ├── __init__.py
│   ├── bi_encoder.py        # Fine-tune sentence-transformer + cache
│   ├── features.py          # TF-IDF, dense cosine, heuristics
│   ├── postprocess.py       # Post-processing heuristics from paper
│   └── model.py             # LambdaMART via LightGBM lambdarank
├── umnlp/
│   ├── __init__.py
│   ├── propositions.py      # Citation context extraction, proposition similarity
│   ├── judge_match.py       # Judge name extraction and matching
│   ├── quotation.py         # Verbatim quotation detection
│   ├── features.py          # Paragraph/sentence-level similarity
│   └── model.py             # FF NN classifier (PyTorch)
├── caselink/
│   ├── __init__.py
│   ├── graph.py             # Global Case Graph construction
│   ├── node_features.py     # Text -> embedding for each case node
│   ├── gnn.py               # GNN architecture (PyTorch Geometric)
│   └── model.py             # Training loop, inductive inference
├── graphrag/
│   ├── __init__.py
│   └── model.py             # Adapter wrapping src/graphrag/ pipeline
├── run_all.py               # Run all baselines, produce comparison
└── run_baseline.py          # Run a single baseline by name
```

### Module Interface

```python
from abc import ABC, abstractmethod

class BaselineModel(ABC):
    """Interface that every baseline implements."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for comparison tables."""
        ...

    @abstractmethod
    def train(
        self,
        corpus: dict[str, str],          # {doc_id: full_text}
        train_queries: list[str],         # query doc_ids
        labels: dict[str, list[str]],     # {query_id: [noticed_ids]}
        bm25_index: BM25Index | None,     # shared index (None for CaseLink)
    ) -> None:
        """Train/fit the model on training data."""
        ...

    @abstractmethod
    def predict(
        self,
        query_id: str,
        corpus: dict[str, str],
        bm25_index: BM25Index | None,
    ) -> list[tuple[str, float]]:
        """Return ranked list of (candidate_id, relevance_score).

        IMPORTANT: Implementations MUST exclude query_id from results
        (a case cannot cite itself).
        """
        ...

    def optimize_threshold(
        self,
        train_queries: list[str],
        corpus: dict[str, str],
        labels: dict[str, list[str]],
        bm25_index: BM25Index | None,
    ) -> float:
        """Find optimal score cutoff on train set. Return threshold.

        Default implementation: run predict() for all train queries,
        sweep thresholds, maximize micro F1. Override only if
        a baseline has a non-standard thresholding approach.
        """
        # Default implementation in base class — not abstract
        ...
```

---

## 3. Shared Infrastructure (`common/`)

### 3.1 Data Loader (`data_loader.py`)

```python
@dataclass
class Dataset:
    corpus: dict[str, str]             # {doc_id: text} for ALL docs (train + test files)
    train_queries: list[str]           # query doc_ids for training
    val_queries: list[str]             # query doc_ids for validation
    labels: dict[str, list[str]]       # {query_id: [noticed_ids]}
```

**Loading logic:**
1. Read all `.txt` files from BOTH `data/task1/task1_train_files_2026/` AND `data/task1/task1_test_files_2026/` into a single corpus dict (~9,556 total docs). This matches COLIEE's evaluation setup where candidates are drawn from the full corpus — noticed cases of validation queries may reside in either directory.
2. Load labels from `task1_train_labels_2026.json`
3. Split queries (keys of labels dict) into train/val:
   - Sort query IDs numerically (higher IDs = more recent cases)
   - Last 20% of queries become validation set (~400 queries)
   - Remaining 80% become training set (~1,600 queries)
   - This mimics COLIEE's chronological train/test split
4. Corpus includes ALL documents from both directories
5. Import path constants from `src/graphrag/config.py` (`TRAIN_DOCS_DIR`, `TEST_DOCS_DIR`, `TRAIN_LABELS`) to avoid path duplication

**Caching:** Corpus text is loaded once and kept in memory (~2-3 GB for ~9,556 docs).

### 3.2 BM25 Index (`bm25_index.py`)

Shared BM25 index over the full corpus, built once.

- Tokenization: lowercase, split on whitespace + punctuation
- Standard BM25 parameters: k1=1.2, b=0.75
- API: `query(text, top_k=200) -> list[tuple[doc_id, score]]`
- Built on `rank-bm25` (already in dependencies)
- Exclude the query document itself from results

### 3.3 Evaluation Harness (`eval_harness.py`)

```python
def run_baselines(
    baselines: list[BaselineModel],
    dataset: Dataset,
) -> pd.DataFrame:
    """
    For each baseline:
    1. Call train() on training data
    2. Call optimize_threshold() on training data
    3. Call predict() for each val query
    4. Apply threshold to get final predictions
    5. Compute micro F1, precision, recall
    Return DataFrame with one row per baseline.
    """
```

**Output:**
- Console: formatted comparison table
- CSV: `output/baselines/comparison.csv`
- Per-query: `output/baselines/per_query_results.csv` (baseline x query F1 matrix)
- Plots: F1 bar chart, per-query F1 distribution boxplot

### 3.4 Metrics (`metrics.py`)

Reuse logic from existing `src/graphrag/metrics.py`:
- Micro-averaged F1, precision, recall
- Threshold optimization via grid search over score values
- **Add**: paired bootstrap significance test (B=10,000) for pairwise baseline comparison

---

## 4. Baseline E — Vanilla BM25

### Purpose
Establish the floor. Every other baseline must beat this.

### Method
1. Query the shared BM25 index with the full query document text
2. Retrieve top-200 candidates
3. Optimize score threshold on train set to maximize F1
4. Apply threshold: candidates above threshold are predicted as noticed

### Implementation
- ~50 lines. Wraps `BM25Index.query()` + threshold optimization.
- No training step (just threshold calibration).

### Expected F1
~0.15-0.20 on our val set (based on historical BM25 baselines).

---

## 5. Baseline A — JNLP 2025 (BM25 + SAILER + LightGBM)

### Purpose
Reproduce the 2025 Task 1 winner.

### Reference
- Paper 29 in approaches report
- JNLP built on UMNLP 2024 framework, extended with SAILER scores + LightGBM

**Note on UMNLP foundation:** The JNLP 2025 paper states it "built on UMNLP framework." Their full feature set likely includes UMNLP-style proposition features alongside BM25 and SAILER. However, the paper primarily documents BM25+SAILER+LightGBM as the novel contribution. Our reproduction focuses on the documented novel elements. If UMNLP features are needed for completeness, they can be imported from the UMNLP baseline (Phase 4) after both are built.

### Method

#### Stage 1: BM25 Candidate Retrieval
- Retrieve top-200 candidates per query using shared BM25 index
- This stage achieves 76-85% recall per the paper

#### Stage 2: Feature Extraction
For each (query, candidate) pair in the top-200:

| Feature | Description | Source |
|---------|-------------|--------|
| `bm25_full` | BM25 score using full query text | Shared BM25 index |
| `bm25_para_max` | Max BM25 score across query paragraphs | Paragraph-split query |
| `qld_score` | Query Likelihood with Dirichlet smoothing | Custom implementation |
| `sailer_sim` | Cosine similarity of SAILER embeddings | Pre-trained checkpoint |
| `bm25_rank` | Rank position in BM25 results (1=top) | Derived |
| `bm25_ratio` | Score / top-1 score | Derived |
| `query_len` | Query document word count | Derived |
| `doc_len` | Candidate document word count | Derived |

#### Stage 3: LightGBM Reranker
- Objective: binary classification (relevant / not relevant)
- Train on feature vectors from training queries
- Predict relevance probability for val candidates
- Optimize threshold on train set

### SAILER Integration (`sailer_encoder.py`)

```python
class SAILEREncoder:
    """
    Downloads CSHaitao/SAILER_en_finetune from HuggingFace.

    IMPORTANT: SAILER uses an asymmetric encoder-decoder architecture
    (deep encoder + shallow decoders for different document segments).
    It is NOT a standard BERT/AutoModel — requires specific loading
    code from github.com/CSHaitao/SAILER for correct inference.

    Caches embeddings to disk (output/baselines/sailer_embeddings/).
    """
    def encode_corpus(self, corpus: dict[str, str]) -> dict[str, np.ndarray]: ...
    def similarity(self, query_id: str, candidate_id: str) -> float: ...
```

SAILER segments documents into structural parts (Facts, Issues, Analysis, Conclusion) and uses separate decoder heads for each segment type. The encoder produces structure-aware representations. Reference the SAILER repo (`github.com/CSHaitao/SAILER`) for the correct encoding procedure.

**Fallback**: If the checkpoint is incompatible with current `transformers`, fall back to `CSHaitao/SAILER_en` (base, not fine-tuned on COLIEE) or `nlpaueb/legal-bert-base-uncased` as last resort.

**Caching**: Embeddings for all corpus docs saved to disk as `.npy` files. Computed once, loaded on subsequent runs.

### QLD Implementation (`features.py`)

Query Likelihood with Dirichlet smoothing:
```
P(q|d) = product_over_terms( (tf(t,d) + mu * P(t|C)) / (|d| + mu) )
```
Where:
- `tf(t,d)` = term frequency in document
- `P(t|C)` = term probability in corpus
- `mu` = Dirichlet smoothing parameter (default 2000)
- Score = log of the product

### LightGBM Configuration
```python
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
num_boost_round = 500
early_stopping_rounds = 50  # on a held-out portion of train
```

### Expected F1
~0.28-0.35 on our val set (the paper achieved 0.3353 on the competition test set).

---

## 6. Baseline B — TQM 2024 (LTR Fusion)

### Purpose
Reproduce the 2024 Task 1 winner. No public code — reverse-engineered from Paper 26.

### Reference
- Paper 26 in approaches report (arXiv 2404.00947)
- TQM is from Tsinghua University (same group as THUIR)

### Method

#### Pre-processing
- Remove boilerplate headers/footers from case text
- Strip formatting artifacts, normalize whitespace
- Paragraph segmentation by `[n]` markers (consistent with THUIR 2023 approach)

#### Lexical Matching
- BM25 scores from shared index
- TF-IDF cosine similarity (scikit-learn `TfidfVectorizer`)

#### Semantic Retrieval
- Fine-tune a sentence-transformer bi-encoder on training pairs:
  - Base model: `sentence-transformers/all-MiniLM-L6-v2` (or legal variant if available)
  - Positive pairs: (query, noticed_case) from training labels
  - Negative pairs: hard negatives from BM25 top-200 that are NOT noticed
  - Loss: MultipleNegativesRankingLoss
  - Training: 3-5 epochs, batch size 32
- Encode all corpus docs + queries, cache embeddings
- Cosine similarity as dense retrieval feature

#### Feature Fusion — LambdaMART
Learning-to-rank via LightGBM with `lambdarank` objective.

**Note:** The paper describes the method at a high level without enumerating exact features. Features marked `[inferred]` are reasonable guesses based on the paper's description and THUIR's prior work. Features marked `[documented]` are explicitly mentioned. Implementers have creative latitude on inferred features.

| Feature | Description | Source |
|---------|-------------|--------|
| `bm25_score` | BM25 relevance score | [documented] |
| `tfidf_cosine` | TF-IDF cosine similarity | [documented] — "classical methods" |
| `dense_cosine` | Fine-tuned bi-encoder cosine similarity | [documented] — "dense vector retrieval" |
| `query_len` | Query word count | [inferred] — "simple features (e.g., case length)" |
| `doc_len` | Candidate word count | [inferred] |
| `len_ratio` | query_len / doc_len | [inferred] |
| `paragraph_count` | Number of paragraphs in candidate | [inferred] |
| `shared_top_k` | How many of this candidate's BM25 top-50 overlap with query's top-50 | [inferred] |
| `year_proximity` | Absolute difference in case year (if extractable) | [inferred] |

#### Post-Processing Heuristics
From the paper's description of "heuristic strategies based on common properties of relevant cases." **These are inferred** — the paper does not specify exact heuristics:
1. **Shared-neighbor boost** [inferred]: If candidate C appears in BM25 top-50 of multiple other top-ranked candidates, boost C's score
2. **Score gap filtering** [inferred]: If there's a large gap between candidate N and N+1, cut at N
3. These heuristics are applied after LambdaMART scoring

#### LightGBM LambdaMART Configuration
```python
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5, 10, 20],
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
}
```

### New Dependencies
- `sentence-transformers` (for bi-encoder fine-tuning)

### Expected F1
~0.30-0.40 on our val set.

---

## 7. Baseline C — UMNLP 2024 (Propositions + Judge Matching + FF NN)

### Purpose
Reproduce the 2024 Task 1 runner-up. Public code available.

### Reference
- Paper 27 in approaches report
- GitHub: `dc435/COLIEE_2024_Task1`
- We will reference the GitHub repo for exact feature engineering and NN architecture, adapting to our data format

### Method

#### Proposition Extraction (`propositions.py`)

"Propositions" are compact summaries of *why* a case was cited. Extracted by:
1. In training data: identify text windows (e.g., 100 words) around known citation locations
2. These windows capture the legal reasoning that motivated the citation
3. Encode propositions using a sentence-transformer
4. At inference: compare query text segments against candidate propositions via cosine similarity

**Implementation approach:**
- Parse training documents for citation markers or references to known noticed cases
- Extract surrounding text as "proposition" candidates
- Build a proposition index per corpus document
- Compute max proposition similarity as a feature

#### Judge Name Matching (`judge_match.py`)

- Reuse regex patterns from existing `src/graphrag/extract_regex.py`
- Extract judge names from case headers
- Feature: binary (same judge = 1, different = 0) + frequency-based score (rare judges matter more)

#### Verbatim Quotation Detection (`quotation.py`)

- Detect if the query document contains verbatim text from the candidate
- Use longest common substring (LCS) or n-gram overlap (n=5+)
- Feature: length of longest verbatim match / query length

#### Paragraph/Sentence Similarity (`features.py`)

- Split both query and candidate into paragraphs and sentences
- Encode with a sentence-transformer (can reuse the same model as TQM, or use a separate one)
- Features:
  - `para_max_sim`: max cosine similarity across all paragraph pairs
  - `para_mean_sim`: mean cosine similarity
  - `sent_max_sim`: max across sentence pairs
  - `sent_mean_sim`: mean across sentence pairs

#### FF Neural Network (`model.py`)

Faithful to UMNLP's architecture:
```python
class PropositionNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
```

- Training: BCE loss, Adam optimizer, lr=1e-3
- Batch size: 256
- Epochs: 20-30 with early stopping on train holdout
- Exact hyperparameters will be cross-referenced with GitHub repo

### Feature Vector
```
[proposition_sim, judge_match, judge_rarity_score, quotation_overlap,
 para_max_sim, para_mean_sim, sent_max_sim, sent_mean_sim,
 bm25_score, doc_len_ratio]
```
~10 features. Final count determined by GitHub reference.

### New Dependencies
- `torch` (for FF NN — also needed by CaseLink)

### Expected F1
~0.30-0.40 on our val set.

---

## 8. Baseline D — CaseLink/UQLegalAI 2025 (GNN)

### Purpose
Reproduce the 2025 Task 1 runner-up. Public code available.

### Reference
- Paper 30 in approaches report (arXiv 2505.20743)
- GitHub: `yanran-tang/CaseLink`

### Method

#### Global Case Graph Construction (`graph.py`)

Build a heterogeneous graph with three edge types.

**Terminology note:** The CaseLink paper uses "Case-Charge" edges for a broader legal context. Since COLIEE's Federal Court of Canada corpus is predominantly civil law (immigration, tax, IP, administrative), not criminal, we adapt the terminology: "Charge" nodes become **"Statute/Regulation"** nodes, extracted via the existing `src/graphrag/extract_regex.py` statute patterns.

1. **Case-Case edges**: From training labels — directed edges from query to each noticed case
2. **Case-Statute edges**: Extract statute/regulation references from case text (reuse `src/graphrag/extract_regex.py`), create edges from case nodes to statute nodes
3. **Statute-Statute edges**: Two statutes are connected if they co-occur in the same case above a threshold frequency

Node types:
- Case nodes (~9,556 — full corpus including test files)
- Statute/Regulation nodes (variable, estimated ~500-2,000)

#### Node Feature Generation (`node_features.py`)

- Encode each case's full text into a dense vector using a sentence-transformer
- For long documents: chunk into paragraphs, encode each, mean-pool
- Base model: `sentence-transformers/all-MiniLM-L6-v2` (or model used by CaseLink repo)
- Cache all embeddings to `output/baselines/caselink_embeddings/`
- Statute nodes: aggregate embeddings of associated cases

#### GNN Architecture (`gnn.py`)

Faithful to CaseLink repo:
```python
class CaseLinkGNN(nn.Module):
    """
    Multi-layer GNN with:
    - GraphSAGE or GAT message passing
    - Edge-type-specific aggregation
    - Degree regularization
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        ...

    def forward(self, x, edge_index, edge_type):
        # Message passing layers
        # Output: updated node embeddings
        ...
```

- 2-3 GNN layers (per repo)
- Hidden dimension: 256 or 384 (per repo)
- Aggregation: mean or attention-based (per repo)

#### Training Loop (`model.py`)

- **Loss**: InfoNCE contrastive loss
  - Positive: (query, noticed_case) from training labels
  - Negatives: random non-noticed cases (in-batch negatives)
- **Regularization**: Degree-based — penalize hub nodes from dominating representations
- **Optimizer**: Adam, lr=1e-3
- **Epochs**: 100-200 with early stopping

#### Inductive Inference
- Test query cases not in training graph
- Compute node features from text embeddings
- Run GNN forward pass using edges available at test time (Case-Statute edges can be computed for test cases; Case-Case edges only available for training)
- Rank corpus cases by cosine similarity of GNN-produced embeddings

### New Dependencies
```
torch
torch-geometric
```

**Note on `torch-scatter` / `torch-sparse`:** These are historically difficult to install (must be compiled against specific PyTorch+CUDA versions). Modern `torch-geometric` (>=2.4) has reduced its dependency on these for many operations. First attempt implementation using only `torch-geometric` core. If specific GNN layers require scatter/sparse, install with `pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{VERSION}+{CUDA}.html` matching the installed PyTorch version.

### Expected F1
~0.25-0.30 on our val set.

---

## 9. GraphRAG Adapter

### Purpose
Integrate our own pipeline into the same evaluation harness for direct comparison.

### Implementation
```python
class GraphRAGModel(BaselineModel):
    """Adapter wrapping src/graphrag/ pipeline."""

    def name(self) -> str:
        return "GraphRAG (ours)"

    def train(self, corpus, train_queries, labels, bm25_index):
        # Trigger extraction + graph construction + community detection
        # Or load pre-computed artifacts from output/
        ...

    def predict(self, query_id, corpus, bm25_index):
        # Call src/graphrag/retrieve.py with multi-signal fusion
        # Return ranked candidates
        ...
```

This adapter is built last, after the GraphRAG pipeline itself is complete. It requires no changes to `src/graphrag/` — just a thin wrapper calling into existing modules.

---

## 10. Evaluation Design

### Train/Validation Split

```python
# Sort query IDs numerically (higher = more recent)
all_query_ids = sorted(labels.keys(), key=lambda x: int(x.replace(".txt", "")))
split_idx = int(len(all_query_ids) * 0.8)
train_queries = all_query_ids[:split_idx]
val_queries = all_query_ids[split_idx:]
```

This produces a fixed, deterministic split. All baselines use the same split.

### Metrics

| Metric | Description |
|--------|-------------|
| **Micro F1** | Primary metric (matches COLIEE evaluation) |
| **Precision** | Micro-averaged precision |
| **Recall** | Micro-averaged recall |
| **Per-query F1** | F1 for each validation query (for distribution analysis) |
| **Threshold** | Optimal score cutoff found on train set |
| **Paired bootstrap p-value** | Statistical significance of F1 difference between each pair of baselines |

### Comparison Table Output

```
===================================================================================
COLIEE 2026 Task 1 — Baseline Comparison
===================================================================================
Baseline                      F1      Prec    Rec     Thresh  Train(s)  Infer(s/q)
-----------------------------------------------------------------------------------
BM25 (vanilla)               0.182   0.145   0.244   0.34    0         0.02
JNLP (BM25+SAILER+LGBM)     0.312   0.298   0.327   0.52    45        0.05
TQM (LTR fusion)             0.351   0.402   0.312   0.48    3600      0.08
UMNLP (propositions+NN)      0.338   0.356   0.322   0.55    1200      0.10
CaseLink (GNN)               0.271   0.265   0.278   0.41    7200      0.03
GraphRAG (ours)              ----    ----    ----    ----    ----      ----
===================================================================================
```

(Values above are illustrative, not predictions. Train(s) = training time in seconds. Infer(s/q) = inference time per query in seconds.)

**Note on cross-validation:** The existing GraphRAG pipeline uses `N_FOLDS = 5` (from `config.py`). This baseline evaluation uses a single 80/20 split for efficiency. When the GraphRAG adapter is built, it should use the same single split for fair comparison. If CV results are needed for a paper, a separate CV evaluation script can be added later.

### Output Artifacts

All saved to `output/baselines/`:
```
output/baselines/
├── comparison.csv                    # Main results table
├── per_query_results.csv             # Query x baseline F1 matrix
├── significance_tests.csv            # Pairwise bootstrap p-values
├── plots/
│   ├── f1_comparison_bar.png         # Bar chart of F1 scores
│   └── per_query_f1_boxplot.png      # Distribution of per-query F1
├── sailer_embeddings/                # Cached SAILER embeddings
├── tqm_embeddings/                   # Cached bi-encoder embeddings
├── caselink_embeddings/              # Cached GNN node embeddings
└── models/                           # Saved model weights/checkpoints
    ├── jnlp_lightgbm.txt
    ├── tqm_lambdamart.txt
    ├── umnlp_nn.pt
    └── caselink_gnn.pt
```

---

## 11. Build Order

### Phase 1 — Shared Infrastructure (no dependencies)
```
common/base_model.py
common/data_loader.py
common/bm25_index.py
common/metrics.py
common/eval_harness.py
```

### Phase 2 — BM25 Baseline (depends: Phase 1)
```
bm25/model.py
run_baseline.py
run_all.py (initial version, BM25 only)
```
**Checkpoint**: Run BM25 baseline end-to-end, verify harness works.

### Phase 3 — JNLP Baseline (depends: Phase 1)
```
jnlp/sailer_encoder.py
jnlp/features.py
jnlp/model.py
```
**Checkpoint**: Run JNLP + BM25, compare results.

### Phase 4 — UMNLP Baseline (depends: Phase 1; parallel with Phase 3)
```
umnlp/propositions.py
umnlp/judge_match.py
umnlp/quotation.py
umnlp/features.py
umnlp/model.py
```

### Phase 5 — TQM Baseline (depends: Phase 1; parallel with Phase 3-4)
```
tqm/bi_encoder.py
tqm/features.py
tqm/postprocess.py
tqm/model.py
```

### Phase 6 — CaseLink Baseline (depends: Phase 1; parallel with Phase 3-5)
```
caselink/graph.py
caselink/node_features.py
caselink/gnn.py
caselink/model.py
```
**Note**: Requires `torch-geometric` dependency addition.

### Phase 7 — GraphRAG Adapter (depends: Phase 1 + GraphRAG pipeline completion)
```
graphrag/model.py
```

### Phase 8 — Comparison & Reporting (depends: all prior phases)
```
run_all.py (final version)
Comparison table, plots, significance tests
```

### Dependency Graph
```
Phase 1 (shared infra)
├── Phase 2 (BM25) ──────┐
├── Phase 3 (JNLP)  ─┐   │
├── Phase 4 (UMNLP)  ─┼───┼── Phase 8 (comparison)
├── Phase 5 (TQM)   ─┘   │
├── Phase 6 (CaseLink) ──┘
└── Phase 7 (GraphRAG adapter) ── requires GraphRAG pipeline
```

---

## 12. New Dependencies

```toml
# Already in pyproject.toml (no action needed)
lightgbm = "*"
scikit-learn = "*"
rank-bm25 = "*"
numpy = "*"
pandas = "*"
matplotlib = "*"
seaborn = "*"
torch = "*"
sentence-transformers = ">=5.2.3"
transformers = "*"

# New additions needed
torch-geometric = ">=2.4"
# torch-scatter and torch-sparse only if required by specific GNN layers (see Section 8)
```

### Dependency Notes
- `torch`, `sentence-transformers`, and `transformers` are already in `pyproject.toml`
- `torch-geometric` is the only genuinely new dependency (needed only by CaseLink)
- SAILER checkpoint loaded via `transformers` — requires specific SAILER inference code from the SAILER repo

### Build Config
Update `pyproject.toml` `[tool.hatch.build.targets.wheel]` to include the baselines package:
```toml
packages = ["src/graphrag", "src/baselines"]
```

---

## 13. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| SAILER checkpoint incompatible with transformers version | Blocks JNLP baseline | Test download early; SAILER has asymmetric encoder-decoder (not standard AutoModel). Reference SAILER repo for loading code. Fall back to `CSHaitao/SAILER_en` (base) or `nlpaueb/legal-bert-base-uncased` if needed |
| TQM paper underspecifies features/heuristics | Incomplete reproduction | Document assumptions; note where we deviated from paper |
| CaseLink repo has breaking dependency issues | Blocks GNN baseline | Pin versions; adapt architecture from paper if repo fails |
| Memory constraints (7,708 docs x embeddings) | OOM | Batch encoding, disk-cached embeddings, float16 |
| Bi-encoder fine-tuning overfits on small train set | Poor TQM performance | Early stopping, small learning rate, regularization |
| Proposition extraction ambiguous in UMNLP paper | Weak feature | Reference GitHub repo for exact implementation |

---

## 14. Success Criteria

1. All 5 baselines produce F1 scores on the validation set
2. Comparison table shows all baselines evaluated on identical data
3. BM25 < JNLP/TQM/UMNLP (sanity check: neural methods beat lexical)
4. Results are in the same ballpark as competition scores (within ~0.05 F1, accounting for different test sets)
5. Statistical significance tests identify which differences are meaningful
6. GraphRAG adapter plugs in cleanly when the pipeline is ready

---

*Design spec authored: 2026-03-11*
*Baselines based on: COLIEE 2024-2025 competition results*
*Reference documents: docs/COLIEE_APPROACHES_REPORT.md, docs/COLIEE_2024_RESULTS.md, docs/COLIEE_2025_RESULTS.md*
