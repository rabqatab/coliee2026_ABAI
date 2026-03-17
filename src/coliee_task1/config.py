"""Central configuration for the GraphRAG pipeline (Option C)."""
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TASK1_DIR = DATA_DIR / "task1"

TRAIN_DOCS_DIR = TASK1_DIR / "task1_train_files_2026"
TEST_DOCS_DIR = TASK1_DIR / "task1_test_files_2026"
TRAIN_LABELS = TASK1_DIR / "task1_train_labels_2026.json"
TEST_LABELS = TASK1_DIR / "task1_test_no_labels_2026.json"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
EXTRACTIONS_DIR = OUTPUT_DIR / "extractions"
GRAPH_DIR = OUTPUT_DIR / "graph"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
BENCHMARK_DIR = OUTPUT_DIR / "benchmarks"
MODELS_DIR = OUTPUT_DIR / "models_v2"

# === Ollama (used by synthetic data) ===
OLLAMA_BASE_URL = "http://localhost:11434"

# === Citation Context (View 2) ===
CONTEXT_WINDOW_WORDS = 150  # ±150 words around <FRAGMENT_SUPPRESSED>
CONTEXT_MERGE_DISTANCE = 50  # Merge windows within 50 words
CONTEXT_MIN_LENGTH = 30  # Minimum words for a valid context

# === BM25 (Stage 2) ===
BM25_TOP_K = 200  # Full-doc retrieval
BM25_CONTEXT_TOP_K = 30  # Per-context-window retrieval
RRF_K = 60  # RRF smoothing parameter

# === Bi-Encoder (Stage 3) ===
BIENCODER_MODEL = "BAAI/bge-large-en-v1.5"
BIENCODER_LORA_RANK = 16
BIENCODER_LORA_ALPHA = 32
BIENCODER_LR = 2e-5
BIENCODER_EPOCHS = 3
BIENCODER_BATCH_SIZE = 16
BIENCODER_HARD_NEG_K = 50  # BM25 top-K for hard negative mining
BIENCODER_TOP_K = 200

# === BGE-M3 Multi-Signal Retrieval (Stage 3 replacement) ===
USE_BGE_M3 = True  # Toggle: True = BGE-M3 triple retrieval, False = original bi-encoder
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_BATCH_SIZE = 8  # Smaller batch - model is larger than BGE-large
BGE_M3_MAX_LENGTH = 8192  # BGE-M3 supports up to 8192 tokens
BGE_M3_WEIGHTS = {"dense": 0.4, "sparse": 0.3, "colbert": 0.3}  # Fusion weights

# === Cross-Encoder (Stage 4) ===
CROSSENCODER_MODEL = "microsoft/deberta-v3-large"
CROSSENCODER_LR = 1e-5
CROSSENCODER_EPOCHS = 3
CROSSENCODER_BATCH_SIZE = 16
CROSSENCODER_MAX_LENGTH = 512
CROSSENCODER_TOP_K = 50  # Rerank top-50 from RRF
CROSSENCODER_PRUNE_TOP_CONTEXTS = 5  # Top citation contexts per query
CROSSENCODER_PRUNE_TOP_PARAGRAPHS = 10  # Top paragraphs per candidate
CROSSENCODER_MODE = "smart"  # "smart", "longctx", or "passage"

# Long-context cross-encoder (used when CROSSENCODER_MODE="longctx")
CROSSENCODER_LONG_MODEL = "BAAI/bge-reranker-v2-m3"
CROSSENCODER_LONG_MAX_LENGTH = 4096
CROSSENCODER_LONG_BATCH_SIZE = 4
CROSSENCODER_LONG_LR = 2e-5

# === GraphRAG Lite (Stage 5) ===
BIPARTITE_WEIGHTS = {
    "statute": 0.50,
    "judge": 0.15,
    "domain": 0.10,
    "outcome": 0.05,
}
LEIDEN_RESOLUTIONS = [0.5, 1.0, 2.0]  # Multi-resolution community detection

# === GNN Score Refinement (Stage 5.5) ===
USE_GNN_RERANKER = True
GNN_HIDDEN_DIM = 64
GNN_NUM_LAYERS = 2
GNN_HEADS = 4  # GAT attention heads
GNN_DROPOUT = 0.1
GNN_LR = 1e-3
GNN_EPOCHS = 50
GNN_K_NEIGHBORS = 8  # Semantic graph neighborhood size
GNN_ENTITY_WEIGHT = 0.3  # Weight for entity-overlap edges vs semantic edges

# Domain classification keywords
DOMAIN_KEYWORDS = {
    "immigration": ["immigration", "refugee", "irpa", "prra", "deportation", "removal order",
                     "permanent resident", "inland", "overseas", "visa", "citizenship"],
    "tax": ["income tax", "tax court", "taxation", "cra", "revenue", "assessment", "deduction"],
    "IP": ["patent", "trademark", "copyright", "intellectual property", "trade-mark", "infringement"],
    "aboriginal": ["first nation", "aboriginal", "indigenous", "indian act", "treaty", "reserve"],
    "criminal": ["criminal code", "criminal", "sentence", "conviction", "offence", "accused"],
    "administrative": ["judicial review", "administrative", "tribunal", "board", "commission",
                        "standard of review", "reasonableness", "correctness"],
    "labour": ["labour", "labor", "employment", "workplace", "union", "collective agreement"],
    "maritime": ["admiralty", "maritime", "vessel", "shipping", "navigation"],
    "environmental": ["environmental", "cepa", "pollution", "species at risk"],
}

# === Reasoning Reranker (Stage 4.5) ===
USE_REASONING_RERANKER = True
REASONING_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Open-source, fits GB10
REASONING_MAX_LENGTH = 4096
REASONING_BATCH_SIZE = 1  # Sequential for reasoning chains
REASONING_TOP_K = 30  # Rerank top-30 from RRF (reasoning is slow)
REASONING_TEMPERATURE = 0.1

# === Synthetic Data Augmentation ===
USE_SYNTHETIC_DATA = False  # Disabled by default -- enable when ready
SYNTHETIC_LLM_MODEL = "deepseek-r1:8b"  # Via Ollama (already running)
SYNTHETIC_N_PAIRS = 20000  # Target number of synthetic pairs
SYNTHETIC_MAX_WORDS = 300  # Max words per extracted summary

# === Meta-Learner (Stage 6) ===
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 800,
    "early_stopping_rounds": 80,
    "min_child_samples": 20,
}

# === Training ===
N_FOLDS = 5
RANDOM_SEED = 42

# === Enhancement Flags (toggle for A/B testing) ===
USE_STRATIFIED_NEGATIVES = False  # Option 19: Stratified hard/medium/easy negatives
TOP1_GUARANTEE = True        # Option 18: Always predict >= 1 per query
MULTI_SEED_RUNS = 1          # Option 16: Number of seed runs (1 = disabled)

