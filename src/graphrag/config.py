"""Central configuration for the GraphRAG pipeline."""
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

# === Ollama ===
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "deepseek-r1:8b"  # Benchmark winner: 100% JSON, 19s/doc, most concepts
EMBED_MODEL = "qwen3-embedding:0.6b"  # Benchmark winner: R@200=0.794, MRR=0.462

# === Extraction ===
MAX_WORDS_SINGLE_CALL = 8000  # Docs under this: single LLM call
CHUNK_OVERLAP_WORDS = 200
LLM_TEMPERATURE = 0.1
LLM_MAX_RETRIES = 2

# === Graph ===
ENTITY_WEIGHTS = {
    "statutes": 0.35,
    "concepts": 0.30,
    "tests": 0.20,
    "domain": 0.10,
    "judge": 0.05,
}

COMMUNITY_EDGE_WEIGHTS = {
    "shared_statutes": 0.30,
    "shared_concepts": 0.30,
    "bm25": 0.30,
    "same_judge": 0.05,
    "same_domain": 0.05,
}

LEIDEN_RESOLUTION = 1.0  # Tuned for ~100-300 communities
CONCEPT_CLUSTER_THRESHOLD = 0.85

# === Retrieval ===
BM25_TOP_K = 200
STAGE1_TOP_K = 50  # Candidates passed to stage 2 (reasoning chains)
RRF_K = 60  # RRF smoothing parameter

# === Embedding ===
EMBED_BATCH_SIZE = 32
EMBED_DIM = 4096  # Qwen3-Embedding-8B default

# === Training ===
N_FOLDS = 5
RANDOM_SEED = 42
