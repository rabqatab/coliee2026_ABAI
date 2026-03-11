# Legal GraphRAG Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a novel GraphRAG pipeline for COLIEE 2026 Task 1 that combines intra-document knowledge graphs, community detection, reasoning chains, and multi-signal fusion retrieval.

**Architecture:** Two-pass entity extraction (regex + LLM via Ollama) builds a corpus-wide knowledge graph. Leiden community detection groups related cases. Five retrieval signals (BM25, entity graph, community matching, embeddings, reasoning chains) are fused via RRF + LightGBM for final predictions.

**Tech Stack:** Python 3.12, NetworkX, leidenalg, igraph, httpx (Ollama API), LightGBM, scikit-learn, numpy, tqdm

---

## File Structure

```
src/graphrag/
    __init__.py              # Package init
    config.py                # All paths, model names, hyperparameters
    ollama_client.py         # Thin async wrapper around Ollama HTTP API
    preprocess.py            # Text cleaning and chunking
    extract_regex.py         # Regex-based entity extraction (statutes, judges, outcomes)
    extract_llm.py           # LLM-based entity extraction (concepts, tests, holdings)
    normalize.py             # Entity normalization and deduplication
    graph.py                 # Knowledge graph construction with NetworkX
    community.py             # Community detection (Leiden) and summarization
    embed.py                 # Embedding via Ollama API
    reasoning.py             # LLM reasoning chain generation
    bm25.py                  # BM25 index wrapper
    retrieve.py              # Multi-signal retrieval and fusion
    metrics.py               # Evaluation metrics (micro-F1, threshold optimization)
    run_benchmark_llm.py     # LLM model benchmark script
    run_benchmark_embed.py   # Embedding model benchmark script
    run_extract.py           # Full corpus extraction runner
    run_index.py             # Full indexing pipeline runner
    run_pipeline.py          # End-to-end retrieval + scoring
tests/
    conftest.py              # Shared fixtures
    test_extract_regex.py    # Regex extraction tests
    test_normalize.py        # Normalization tests
    test_graph.py            # Graph construction tests
    test_retrieve.py         # Retrieval fusion tests
    test_ollama_client.py    # Ollama client tests
```

---

## Chunk 1: Foundation (Tasks 1-3)

### Task 1: Project Setup & Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `src/graphrag/__init__.py`
- Create: `src/graphrag/config.py`

- [ ] **Step 1: Add dependencies to pyproject.toml**

```bash
cd /home/alphabridge/Research/coliee2026
uv add networkx leidenalg igraph httpx tqdm pytest
```

- [ ] **Step 2: Create package init**

```python
# src/graphrag/__init__.py
"""Legal GraphRAG pipeline for COLIEE 2026 Task 1."""
```

- [ ] **Step 3: Create config module**

```python
# src/graphrag/config.py
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
LLM_MODEL = "qwen3:32b"  # Updated after benchmarking
EMBED_MODEL = "qwen3-embedding:8b"  # Updated after benchmarking

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
```

- [ ] **Step 4: Verify setup**

```bash
uv run python -c "from graphrag.config import PROJECT_ROOT; print(f'Root: {PROJECT_ROOT}')"
```

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/__init__.py src/graphrag/config.py pyproject.toml uv.lock
git commit -m "feat: add graphrag package with config and dependencies"
```

---

### Task 2: Ollama Client

**Files:**
- Create: `src/graphrag/ollama_client.py`
- Create: `tests/test_ollama_client.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_ollama_client.py
"""Tests for Ollama client wrapper."""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from graphrag.ollama_client import OllamaClient


@pytest.fixture
def client():
    return OllamaClient(base_url="http://localhost:11434")


class TestGenerate:
    def test_generate_returns_text(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello world"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.generate("test-model", "Say hello")
            assert result == "Hello world"

    def test_generate_json_parses_response(self, client):
        json_str = '{"legal_concepts": ["fairness"], "legal_tests": []}'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": json_str}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.generate_json("test-model", "Extract entities")
            assert result == {"legal_concepts": ["fairness"], "legal_tests": []}


class TestEmbed:
    def test_embed_returns_vectors(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", return_value=mock_response):
            result = client.embed("test-model", ["text1", "text2"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]


class TestListModels:
    def test_list_models(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:32b", "size": 20_000_000_000},
                {"name": "deepseek-r1:8b", "size": 5_200_000_000},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "get", return_value=mock_response):
            result = client.list_models()
            assert len(result) == 2
            assert result[0]["name"] == "qwen3:32b"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ollama_client.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: Implement Ollama client**

```python
# src/graphrag/ollama_client.py
"""Thin wrapper around the Ollama HTTP API."""
import json
import logging
import time
from typing import Any

import httpx

from graphrag.config import OLLAMA_BASE_URL, LLM_TEMPERATURE, LLM_MAX_RETRIES

logger = logging.getLogger(__name__)


class OllamaClient:
    """Synchronous client for the Ollama REST API."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = 300.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text completion."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        resp = self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = LLM_TEMPERATURE,
    ) -> dict[str, Any]:
        """Generate and parse JSON output. Retries on malformed JSON."""
        for attempt in range(LLM_MAX_RETRIES + 1):
            text = self.generate(
                model, prompt, system=system, temperature=temperature
            )
            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s...",
                    attempt + 1,
                    LLM_MAX_RETRIES + 1,
                    text[:200],
                )
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(1)
        raise ValueError(f"Failed to parse JSON after {LLM_MAX_RETRIES + 1} attempts")

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a batch of texts."""
        resp = self._client.post(
            "/api/embed",
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def list_models(self) -> list[dict[str, Any]]:
        """List locally available models."""
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return resp.json()["models"]

    def pull_model(self, model: str) -> None:
        """Pull a model from the Ollama registry."""
        logger.info("Pulling model: %s", model)
        resp = self._client.post(
            "/api/pull",
            json={"name": model, "stream": False},
            timeout=1800.0,  # 30 min for large models
        )
        resp.raise_for_status()
        logger.info("Model pulled: %s", model)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_ollama_client.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/ollama_client.py tests/test_ollama_client.py
git commit -m "feat: add Ollama HTTP client wrapper with tests"
```

---

### Task 3: Text Preprocessing

**Files:**
- Create: `src/graphrag/preprocess.py`

- [ ] **Step 1: Implement preprocessing**

```python
# src/graphrag/preprocess.py
"""Text preprocessing for legal case documents."""
import re
from pathlib import Path


def preprocess(text: str) -> str:
    """Clean a raw case document for extraction."""
    # Remove FRAGMENT_SUPPRESSED placeholders
    text = re.sub(r"<FRAGMENT_SUPPRESSED>", "", text)
    # Remove end-of-document markers
    text = re.sub(r"\[End of document\]", "", text)
    # Rejoin broken statute names (lines starting with lowercase after short line)
    text = re.sub(r"(\w)\n([a-z])", r"\1 \2", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_paragraphs(text: str) -> list[tuple[int, str]]:
    """Extract numbered paragraphs from a case document.

    Returns list of (paragraph_number, paragraph_text) tuples.
    """
    # Match [N] paragraph markers
    parts = re.split(r"\[(\d+)\]", text)
    paragraphs = []
    for i in range(1, len(parts), 2):
        para_num = int(parts[i])
        para_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if para_text:
            paragraphs.append((para_num, para_text))
    return paragraphs


def chunk_for_llm(text: str, max_words: int = 8000, overlap_words: int = 200) -> list[str]:
    """Split text into chunks suitable for LLM processing.

    Documents under max_words are returned as a single chunk.
    Larger documents are split at paragraph boundaries with overlap.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Keep last paragraph as overlap
            overlap_paras = []
            overlap_count = 0
            for p in reversed(current_chunk):
                pw = len(p.split())
                if overlap_count + pw > overlap_words:
                    break
                overlap_paras.insert(0, p)
                overlap_count += pw
            current_chunk = overlap_paras
            current_words = overlap_count
        current_chunk.append(para)
        current_words += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def load_corpus(docs_dir: Path) -> dict[str, str]:
    """Load all .txt files from a directory into a dict keyed by filename."""
    corpus = {}
    for path in sorted(docs_dir.glob("*.txt")):
        corpus[path.name] = path.read_text(encoding="utf-8", errors="replace")
    return corpus
```

- [ ] **Step 2: Quick smoke test**

```bash
uv run python -c "
from graphrag.preprocess import preprocess, chunk_for_llm
text = '<FRAGMENT_SUPPRESSED>\n\n\n\n[1] Hello world.\n[2] Second para.\n[End of document]'
clean = preprocess(text)
print(repr(clean))
chunks = chunk_for_llm('word ' * 10000, max_words=5000)
print(f'Chunks: {len(chunks)}')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/preprocess.py
git commit -m "feat: add text preprocessing and chunking utilities"
```

---

## Chunk 2: Entity Extraction (Tasks 4-7)

### Task 4: Regex-Based Extraction

**Files:**
- Create: `src/graphrag/extract_regex.py`
- Create: `tests/test_extract_regex.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_extract_regex.py
"""Tests for regex-based entity extraction."""
import pytest
from graphrag.extract_regex import extract_statutes, extract_judges, extract_outcome


class TestExtractStatutes:
    def test_irpa_full_name(self):
        text = "Immigration and Refugee Protection Act, S.C. 2001, c. 27"
        result = extract_statutes(text)
        assert any("Immigration and Refugee Protection Act" in s for s in result)

    def test_section_reference(self):
        text = "pursuant to subsection 72(1) of the IRPA"
        result = extract_statutes(text)
        assert any("72(1)" in s for s in result)

    def test_multiple_statutes(self):
        text = """
        The Federal Courts Act, R.S.C. 1985, c. F-7
        and the Immigration and Refugee Protection Act, S.C. 2001, c. 27
        """
        result = extract_statutes(text)
        assert len(result) >= 2

    def test_no_statutes(self):
        text = "This is a simple text with no statute references."
        result = extract_statutes(text)
        assert len(result) == 0


class TestExtractJudges:
    def test_simple_judge(self):
        text = "Mosley J."
        result = extract_judges(text)
        assert "Mosley J." in result

    def test_judge_with_comma(self):
        text = "Before: Mosley, J."
        result = extract_judges(text)
        assert len(result) >= 1

    def test_justice_full_title(self):
        text = "The Honourable Mr. Justice Mosley"
        result = extract_judges(text)
        assert len(result) >= 1


class TestExtractOutcome:
    def test_dismissed(self):
        text = "The application is dismissed."
        result = extract_outcome(text)
        assert result == "application dismissed"

    def test_allowed(self):
        text = "The appeal is allowed."
        result = extract_outcome(text)
        assert result == "appeal allowed"

    def test_no_outcome(self):
        text = "The parties submitted their arguments."
        result = extract_outcome(text)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_extract_regex.py -v
```

- [ ] **Step 3: Implement regex extraction**

```python
# src/graphrag/extract_regex.py
"""Regex-based entity extraction from legal case documents."""
import re


def extract_statutes(text: str) -> list[str]:
    """Extract statute references from case text.

    Captures full Act names with citations and section references.
    """
    statutes = []

    # Full statute citations: "Act Name, S.C./R.S.C. YYYY, c. X"
    full_pattern = r"([\w\s]+Act),?\s*(S\.C\.|R\.S\.C\.)\s*\d{4},?\s*c\.\s*[\w.\-]+"
    for match in re.finditer(full_pattern, text):
        statutes.append(match.group(0).strip())

    # Section references: "section/subsection/paragraph N(N) of the ACT"
    section_pattern = (
        r"(?:section|subsection|paragraph|s\.)\s*(\d+(?:\(\d+\))?(?:\.\d+)?)"
        r"\s+of\s+(?:the\s+)?([\w\s]+(?:Act|Regulations?))"
    )
    for match in re.finditer(section_pattern, text, re.IGNORECASE):
        statutes.append(f"{match.group(2).strip()} s. {match.group(1)}")

    # Abbreviated references: "IRPA", "PRRA", "CBSA"
    abbrev_pattern = r"\b(IRPA|PRRA|CBSA|FCTD|FCA|SCC)\b"
    for match in re.finditer(abbrev_pattern, text):
        statutes.append(match.group(1))

    return list(set(statutes))


def extract_judges(text: str) -> list[str]:
    """Extract judge names from case text."""
    judges = []

    # "Name J." or "Name, J." or "Name J.A."
    pattern1 = r"(\b[A-Z][a-z]+),?\s*J\.(?:A\.)?"
    for match in re.finditer(pattern1, text):
        judges.append(f"{match.group(1)} J.")

    # "The Honourable Mr./Madam Justice Name"
    pattern2 = r"(?:The\s+)?Honou?rable\s+(?:Mr\.|Madam)\s+Justice\s+(\w+)"
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        judges.append(f"{match.group(1)} J.")

    # "Before: Name J."
    pattern3 = r"Before:\s*(\w+),?\s*J\.(?:A\.)?"
    for match in re.finditer(pattern3, text):
        judges.append(f"{match.group(1)} J.")

    return list(set(judges))


def extract_outcome(text: str) -> str | None:
    """Extract case outcome from text.

    Returns normalized string like 'application dismissed' or None.
    """
    pattern = r"(?:the\s+)?(application|appeal|motion)\s+(?:is\s+|be\s+)?(dismissed|allowed|granted)"
    # Search in the last 2000 characters (outcome is typically at the end)
    search_text = text[-2000:] if len(text) > 2000 else text
    match = re.search(pattern, search_text, re.IGNORECASE)
    if match:
        return f"{match.group(1).lower()} {match.group(2).lower()}"
    return None


def extract_paragraph_markers(text: str) -> list[int]:
    """Extract paragraph numbers from [N] markers."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", text)]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_extract_regex.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/extract_regex.py tests/test_extract_regex.py
git commit -m "feat: add regex entity extraction for statutes, judges, outcomes"
```

---

### Task 5: LLM-Based Extraction

**Files:**
- Create: `src/graphrag/extract_llm.py`

- [ ] **Step 1: Implement LLM extraction**

```python
# src/graphrag/extract_llm.py
"""LLM-based entity extraction from legal case documents."""
import logging
from typing import Any

from graphrag.config import LLM_MODEL, MAX_WORDS_SINGLE_CALL
from graphrag.ollama_client import OllamaClient
from graphrag.preprocess import chunk_for_llm

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a legal information extractor. Given a Federal Court of Canada
case, extract the following entities in JSON format.

EXTRACT ONLY what is explicitly stated. Do not infer or hallucinate.

Return ONLY valid JSON with this exact schema:
{
  "legal_concepts": [
    "abstract legal principles the case reasons about, e.g. standard of review, procedural fairness"
  ],
  "legal_tests": [
    "named legal tests or frameworks applied, e.g. Dunsmuir test, Baker factors, Oakes test"
  ],
  "statutes_applied": [
    {"name": "Act name or abbreviation", "section": "section number if stated", "context": "how it is applied"}
  ],
  "key_holdings": [
    "1-2 sentence summary of each major holding"
  ],
  "case_type": "judicial_review | appeal | motion | trial | other",
  "legal_domain": "immigration | IP | tax | aboriginal | criminal | administrative | other"
}"""

EXTRACTION_USER_TEMPLATE = """Extract legal entities from this case text:

{text}

Return ONLY the JSON object, no other text."""


def extract_entities_llm(
    client: OllamaClient,
    text: str,
    model: str = LLM_MODEL,
) -> dict[str, Any]:
    """Extract entities from a case document using LLM.

    Long documents are chunked and extractions are merged.
    """
    chunks = chunk_for_llm(text, max_words=MAX_WORDS_SINGLE_CALL)

    if len(chunks) == 1:
        prompt = EXTRACTION_USER_TEMPLATE.format(text=chunks[0])
        return client.generate_json(model, prompt, system=EXTRACTION_SYSTEM_PROMPT)

    # Multiple chunks: extract per chunk, then merge
    extractions = []
    for i, chunk in enumerate(chunks):
        logger.debug("Extracting chunk %d/%d", i + 1, len(chunks))
        prompt = EXTRACTION_USER_TEMPLATE.format(text=chunk)
        try:
            result = client.generate_json(
                model, prompt, system=EXTRACTION_SYSTEM_PROMPT
            )
            extractions.append(result)
        except ValueError:
            logger.warning("Failed to extract from chunk %d, skipping", i + 1)

    return _merge_extractions(extractions)


def _merge_extractions(extractions: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge extractions from multiple chunks into a single result."""
    merged: dict[str, Any] = {
        "legal_concepts": [],
        "legal_tests": [],
        "statutes_applied": [],
        "key_holdings": [],
        "case_type": "other",
        "legal_domain": "other",
    }

    seen_concepts: set[str] = set()
    seen_tests: set[str] = set()
    seen_statutes: set[str] = set()

    for ext in extractions:
        # Merge concepts (deduplicate by lowercase)
        for concept in ext.get("legal_concepts", []):
            key = concept.lower().strip()
            if key not in seen_concepts:
                seen_concepts.add(key)
                merged["legal_concepts"].append(concept)

        # Merge tests
        for test in ext.get("legal_tests", []):
            key = test.lower().strip()
            if key not in seen_tests:
                seen_tests.add(key)
                merged["legal_tests"].append(test)

        # Merge statutes (deduplicate by name+section)
        for statute in ext.get("statutes_applied", []):
            if isinstance(statute, dict):
                key = f"{statute.get('name', '')}-{statute.get('section', '')}".lower()
            else:
                key = str(statute).lower()
            if key not in seen_statutes:
                seen_statutes.add(key)
                merged["statutes_applied"].append(statute)

        # Collect holdings
        merged["key_holdings"].extend(ext.get("key_holdings", []))

        # Take first non-"other" case_type and domain
        ct = ext.get("case_type", "other")
        if ct != "other" and merged["case_type"] == "other":
            merged["case_type"] = ct

        ld = ext.get("legal_domain", "other")
        if ld != "other" and merged["legal_domain"] == "other":
            merged["legal_domain"] = ld

    return merged
```

- [ ] **Step 2: Smoke test with one document**

```bash
uv run python -c "
from graphrag.ollama_client import OllamaClient
from graphrag.extract_llm import extract_entities_llm
from graphrag.preprocess import preprocess
from graphrag.config import TRAIN_DOCS_DIR
import json

doc = sorted(TRAIN_DOCS_DIR.glob('*.txt'))[0]
text = preprocess(doc.read_text())
with OllamaClient() as client:
    result = extract_entities_llm(client, text[:3000])
    print(json.dumps(result, indent=2))
"
```

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/extract_llm.py
git commit -m "feat: add LLM-based entity extraction with chunking support"
```

---

### Task 6: Entity Normalization

**Files:**
- Create: `src/graphrag/normalize.py`
- Create: `tests/test_normalize.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_normalize.py
"""Tests for entity normalization."""
import pytest
from graphrag.normalize import normalize_statute, normalize_judge, merge_regex_llm


class TestNormalizeStatute:
    def test_irpa_full(self):
        assert normalize_statute("Immigration and Refugee Protection Act") == "IRPA"

    def test_irpa_with_citation(self):
        result = normalize_statute(
            "Immigration and Refugee Protection Act, S.C. 2001, c. 27"
        )
        assert result == "IRPA"

    def test_already_abbreviated(self):
        assert normalize_statute("IRPA") == "IRPA"

    def test_federal_courts_act(self):
        result = normalize_statute("Federal Courts Act, R.S.C. 1985, c. F-7")
        assert result == "Federal Courts Act"

    def test_section_preserved(self):
        result = normalize_statute("IRPA s. 72(1)")
        assert "IRPA" in result
        assert "72(1)" in result


class TestNormalizeJudge:
    def test_simple(self):
        assert normalize_judge("Mosley J.") == "Mosley J."

    def test_with_comma(self):
        assert normalize_judge("Mosley, J.") == "Mosley J."

    def test_extra_whitespace(self):
        assert normalize_judge("  Mosley  J. ") == "Mosley J."


class TestMergeRegexLlm:
    def test_merges_statutes(self):
        regex = {
            "statutes": ["IRPA", "Federal Courts Act, R.S.C. 1985, c. F-7"],
            "judges": ["Mosley J."],
            "outcome": "application dismissed",
        }
        llm = {
            "legal_concepts": ["standard of review"],
            "legal_tests": ["Dunsmuir test"],
            "statutes_applied": [
                {"name": "IRPA", "section": "s. 72(1)", "context": "judicial review"}
            ],
            "key_holdings": ["Officer's decision unreasonable"],
            "case_type": "judicial_review",
            "legal_domain": "immigration",
        }
        merged = merge_regex_llm(regex, llm)
        assert "IRPA" in merged["statutes"]
        assert "Mosley J." in merged["judges"]
        assert "standard of review" in merged["concepts"]
        assert merged["domain"] == "immigration"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_normalize.py -v
```

- [ ] **Step 3: Implement normalization**

```python
# src/graphrag/normalize.py
"""Entity normalization and deduplication."""
import re
from typing import Any

# Canonical statute aliases
STATUTE_ALIASES: dict[str, str] = {
    "immigration and refugee protection act": "IRPA",
    "irpa": "IRPA",
    "the act (irpa)": "IRPA",
    "citizenship act": "Citizenship Act",
    "canada evidence act": "Canada Evidence Act",
    "criminal code": "Criminal Code",
    "charter of rights and freedoms": "Charter",
    "canadian charter of rights and freedoms": "Charter",
    "charter": "Charter",
    "federal courts act": "Federal Courts Act",
    "income tax act": "Income Tax Act",
    "canada labour code": "Canada Labour Code",
    "customs act": "Customs Act",
    "access to information act": "Access to Information Act",
    "privacy act": "Privacy Act",
    "patent act": "Patent Act",
    "copyright act": "Copyright Act",
    "trade-marks act": "Trade-marks Act",
    "competition act": "Competition Act",
    "bankruptcy and insolvency act": "Bankruptcy and Insolvency Act",
    "indian act": "Indian Act",
    "national defence act": "National Defence Act",
    "canada elections act": "Canada Elections Act",
}


def normalize_statute(raw: str) -> str:
    """Normalize a statute name to its canonical form.

    Handles full names, abbreviations, and citation suffixes.
    """
    # Strip citation suffix (e.g., ", S.C. 2001, c. 27")
    name = re.sub(r",?\s*(?:S\.C\.|R\.S\.C\.)\s*\d{4}.*$", "", raw).strip()

    # Check for section reference
    section_match = re.search(r"\bs\.\s*(\d+(?:\(\d+\))?(?:\.\d+)?)", name)
    section = section_match.group(0) if section_match else None

    # Remove section from name for lookup
    lookup = re.sub(r"\bs\.\s*\d+.*$", "", name).strip()
    lookup_lower = lookup.lower().rstrip(",. ")

    canonical = STATUTE_ALIASES.get(lookup_lower, lookup)

    if section:
        return f"{canonical} {section}"
    return canonical


def normalize_judge(raw: str) -> str:
    """Normalize a judge name to 'Lastname J.' format."""
    name = raw.strip()
    # Remove comma before J.
    name = re.sub(r",\s*J\.", " J.", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def merge_regex_llm(
    regex_result: dict[str, Any],
    llm_result: dict[str, Any],
) -> dict[str, Any]:
    """Merge regex and LLM extraction results into a unified entity record.

    Returns a flat dict with normalized entities.
    """
    # Normalize regex statutes
    statutes = set()
    for s in regex_result.get("statutes", []):
        statutes.add(normalize_statute(s))

    # Add LLM statutes
    for s in llm_result.get("statutes_applied", []):
        if isinstance(s, dict):
            name = s.get("name", "")
            section = s.get("section", "")
            full = f"{name} {section}".strip() if section else name
        else:
            full = str(s)
        statutes.add(normalize_statute(full))

    # Normalize judges
    judges = set()
    for j in regex_result.get("judges", []):
        judges.add(normalize_judge(j))

    # Concepts and tests from LLM
    concepts = list(set(
        c.lower().strip() for c in llm_result.get("legal_concepts", [])
    ))
    tests = list(set(
        t.strip() for t in llm_result.get("legal_tests", [])
    ))

    return {
        "statutes": sorted(statutes),
        "judges": sorted(judges),
        "outcome": regex_result.get("outcome"),
        "concepts": concepts,
        "tests": tests,
        "holdings": llm_result.get("key_holdings", []),
        "case_type": llm_result.get("case_type", "other"),
        "domain": llm_result.get("legal_domain", "other"),
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_normalize.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/normalize.py tests/test_normalize.py
git commit -m "feat: add entity normalization with statute aliases and merge logic"
```

---

### Task 7: Corpus Extraction Runner

**Files:**
- Create: `src/graphrag/run_extract.py`

- [ ] **Step 1: Implement extraction runner**

```python
# src/graphrag/run_extract.py
"""Full corpus entity extraction with resume support."""
import json
import logging
import sys
import time
from pathlib import Path

from graphrag.config import (
    TRAIN_DOCS_DIR,
    TEST_DOCS_DIR,
    EXTRACTIONS_DIR,
    LLM_MODEL,
)
from graphrag.ollama_client import OllamaClient
from graphrag.preprocess import preprocess
from graphrag.extract_regex import extract_statutes, extract_judges, extract_outcome
from graphrag.extract_llm import extract_entities_llm
from graphrag.normalize import merge_regex_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def extract_single_document(
    client: OllamaClient,
    doc_path: Path,
    model: str = LLM_MODEL,
) -> dict:
    """Run full extraction pipeline on a single document."""
    raw_text = doc_path.read_text(encoding="utf-8", errors="replace")
    clean_text = preprocess(raw_text)

    # Pass 1: Regex
    regex_result = {
        "statutes": extract_statutes(clean_text),
        "judges": extract_judges(clean_text),
        "outcome": extract_outcome(clean_text),
    }

    # Pass 2: LLM
    llm_result = extract_entities_llm(client, clean_text, model=model)

    # Merge
    merged = merge_regex_llm(regex_result, llm_result)
    merged["doc_id"] = doc_path.stem
    merged["word_count"] = len(clean_text.split())

    return merged


def run_extraction(
    docs_dir: Path,
    output_dir: Path,
    model: str = LLM_MODEL,
    resume: bool = True,
) -> None:
    """Extract entities from all documents in a directory.

    Saves one JSON file per document for resume support.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_paths = sorted(docs_dir.glob("*.txt"))
    logger.info("Found %d documents in %s", len(doc_paths), docs_dir)

    # Check what's already done
    if resume:
        done = {p.stem for p in output_dir.glob("*.json")}
        remaining = [p for p in doc_paths if p.stem not in done]
        logger.info("Already extracted: %d, remaining: %d", len(done), len(remaining))
    else:
        remaining = doc_paths

    with OllamaClient() as client:
        start_time = time.time()
        for i, doc_path in enumerate(remaining):
            try:
                result = extract_single_document(client, doc_path, model=model)
                out_path = output_dir / f"{doc_path.stem}.json"
                out_path.write_text(json.dumps(result, indent=2))

                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] %s — %.1f docs/min, ETA %.0f min",
                    i + 1,
                    len(remaining),
                    doc_path.name,
                    rate * 60,
                    eta / 60,
                )
            except Exception:
                logger.exception("Failed to extract %s", doc_path.name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract entities from corpus")
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="train",
    )
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    dirs = []
    if args.split in ("train", "both"):
        dirs.append(("train", TRAIN_DOCS_DIR))
    if args.split in ("test", "both"):
        dirs.append(("test", TEST_DOCS_DIR))

    for split_name, docs_dir in dirs:
        out_dir = EXTRACTIONS_DIR / split_name
        logger.info("=== Extracting %s split ===", split_name)
        run_extraction(docs_dir, out_dir, model=args.model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test with 3 documents**

```bash
uv run python -m graphrag.run_extract --split train 2>&1 | head -20
# (Ctrl+C after a few documents to verify it works)
```

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/run_extract.py
git commit -m "feat: add corpus extraction runner with resume support"
```

---

## Chunk 3: Model Benchmarking (Tasks 8-9)

### Task 8: LLM Benchmark

**Files:**
- Create: `src/graphrag/run_benchmark_llm.py`

- [ ] **Step 1: Implement LLM benchmark**

```python
# src/graphrag/run_benchmark_llm.py
"""Benchmark LLM models for entity extraction quality and speed."""
import json
import logging
import random
import time
from pathlib import Path

from graphrag.config import (
    TRAIN_DOCS_DIR,
    BENCHMARK_DIR,
    RANDOM_SEED,
)
from graphrag.ollama_client import OllamaClient
from graphrag.preprocess import preprocess
from graphrag.extract_llm import extract_entities_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Models to benchmark
LLM_CANDIDATES = [
    "qwen3:32b",
    "gemma3:27b",
    "deepseek-r1:8b",
    # "llama4:scout",  # Add when available on aarch64
]

N_SAMPLE_DOCS = 50
N_CONSISTENCY_DOCS = 10
N_CONSISTENCY_RUNS = 3


def sample_documents(docs_dir: Path, n: int, seed: int = RANDOM_SEED) -> list[Path]:
    """Sample n documents from the corpus."""
    rng = random.Random(seed)
    all_docs = sorted(docs_dir.glob("*.txt"))
    return rng.sample(all_docs, min(n, len(all_docs)))


def benchmark_single_model(
    client: OllamaClient,
    model: str,
    docs: list[Path],
) -> dict:
    """Benchmark a single model on a set of documents."""
    results = []
    times = []
    valid_json_count = 0

    for doc_path in docs:
        text = preprocess(doc_path.read_text(encoding="utf-8", errors="replace"))
        start = time.time()
        try:
            result = extract_entities_llm(client, text, model=model)
            elapsed = time.time() - start
            valid_json_count += 1
            results.append({
                "doc": doc_path.name,
                "extraction": result,
                "time_seconds": elapsed,
            })
        except (ValueError, Exception) as e:
            elapsed = time.time() - start
            results.append({
                "doc": doc_path.name,
                "error": str(e),
                "time_seconds": elapsed,
            })
        times.append(elapsed)

    # Aggregate stats
    entity_counts = {
        "concepts": [],
        "tests": [],
        "statutes": [],
        "holdings": [],
    }
    for r in results:
        ext = r.get("extraction", {})
        entity_counts["concepts"].append(len(ext.get("legal_concepts", [])))
        entity_counts["tests"].append(len(ext.get("legal_tests", [])))
        entity_counts["statutes"].append(len(ext.get("statutes_applied", [])))
        entity_counts["holdings"].append(len(ext.get("key_holdings", [])))

    import numpy as np
    return {
        "model": model,
        "n_docs": len(docs),
        "valid_json_rate": valid_json_count / len(docs),
        "mean_time_seconds": np.mean(times),
        "p95_time_seconds": np.percentile(times, 95) if times else 0,
        "mean_concepts": np.mean(entity_counts["concepts"]),
        "mean_tests": np.mean(entity_counts["tests"]),
        "mean_statutes": np.mean(entity_counts["statutes"]),
        "mean_holdings": np.mean(entity_counts["holdings"]),
        "results": results,
    }


def main():
    output_dir = BENCHMARK_DIR / "llm"
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = sample_documents(TRAIN_DOCS_DIR, N_SAMPLE_DOCS)
    logger.info("Sampled %d documents for benchmarking", len(docs))

    with OllamaClient() as client:
        # Check available models
        available = {m["name"] for m in client.list_models()}
        logger.info("Available models: %s", available)

        for model in LLM_CANDIDATES:
            # Check if model name matches (Ollama uses name:tag format)
            model_available = any(model in m for m in available)
            if not model_available:
                logger.warning("Model %s not available, attempting to pull", model)
                try:
                    client.pull_model(model)
                except Exception:
                    logger.exception("Failed to pull %s, skipping", model)
                    continue

            logger.info("=== Benchmarking %s ===", model)
            result = benchmark_single_model(client, model, docs)

            # Save results
            safe_name = model.replace(":", "_").replace("/", "_")
            out_path = output_dir / f"{safe_name}.json"
            out_path.write_text(json.dumps(result, indent=2, default=str))

            logger.info(
                "%s: valid_json=%.0f%%, mean_time=%.1fs, concepts=%.1f, tests=%.1f, statutes=%.1f",
                model,
                result["valid_json_rate"] * 100,
                result["mean_time_seconds"],
                result["mean_concepts"],
                result["mean_tests"],
                result["mean_statutes"],
            )

    # Print comparison table
    print("\n=== LLM Benchmark Results ===")
    print(f"{'Model':<25} {'JSON%':>6} {'Time(s)':>8} {'Concepts':>9} {'Tests':>6} {'Statutes':>9}")
    print("-" * 70)
    for model in LLM_CANDIDATES:
        safe_name = model.replace(":", "_").replace("/", "_")
        path = output_dir / f"{safe_name}.json"
        if path.exists():
            r = json.loads(path.read_text())
            print(
                f"{r['model']:<25} {r['valid_json_rate']*100:>5.0f}% {r['mean_time_seconds']:>7.1f} "
                f"{r['mean_concepts']:>8.1f} {r['mean_tests']:>5.1f} {r['mean_statutes']:>8.1f}"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark (expect ~6-8 hours)**

```bash
uv run python -m graphrag.run_benchmark_llm
```

- [ ] **Step 3: Commit results**

```bash
git add src/graphrag/run_benchmark_llm.py
git commit -m "feat: add LLM benchmark runner for entity extraction"
```

---

### Task 9: Embedding Benchmark

**Files:**
- Create: `src/graphrag/run_benchmark_embed.py`
- Create: `src/graphrag/embed.py`

- [ ] **Step 1: Implement embedding module**

```python
# src/graphrag/embed.py
"""Embedding utilities using Ollama API."""
import logging
import numpy as np
from typing import Sequence

from graphrag.config import EMBED_MODEL, EMBED_BATCH_SIZE
from graphrag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def embed_texts(
    client: OllamaClient,
    texts: Sequence[str],
    model: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed a list of texts into vectors.

    Returns numpy array of shape (len(texts), embed_dim).
    """
    all_embeddings = []

    if show_progress:
        from tqdm import tqdm
        batches = range(0, len(texts), batch_size)
        iterator = tqdm(batches, desc="Embedding", unit="batch")
    else:
        iterator = range(0, len(texts), batch_size)

    for start in iterator:
        batch = list(texts[start : start + batch_size])
        vectors = client.embed(model, batch)
        all_embeddings.extend(vectors)

    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of vectors.

    Args:
        a: shape (m, d)
        b: shape (n, d)

    Returns:
        shape (m, n) similarity matrix
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T
```

- [ ] **Step 2: Implement embedding benchmark**

```python
# src/graphrag/run_benchmark_embed.py
"""Benchmark embedding models for retrieval quality."""
import json
import logging
import time
import numpy as np
from pathlib import Path

from graphrag.config import (
    TRAIN_DOCS_DIR,
    TRAIN_LABELS,
    BENCHMARK_DIR,
)
from graphrag.ollama_client import OllamaClient
from graphrag.preprocess import preprocess, load_corpus
from graphrag.embed import embed_texts, cosine_similarity_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EMBED_CANDIDATES = [
    "qwen3-embedding:8b",
    "qwen3-embedding:0.6b",
    "bge-m3",
    "nomic-embed-text",
]


def truncate_for_embedding(text: str, max_words: int = 4000) -> str:
    """Truncate text to fit embedding model context."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def compute_retrieval_metrics(
    sim_matrix: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    labels: dict[str, list[str]],
    k_values: list[int] = [50, 100, 200],
) -> dict:
    """Compute Recall@K and MRR from a similarity matrix."""
    recalls = {k: [] for k in k_values}
    mrrs = []

    for i, qid in enumerate(query_ids):
        true_cited = set(labels.get(qid, []))
        if not true_cited:
            continue

        # Get ranked corpus indices (descending similarity)
        scores = sim_matrix[i]
        ranked_indices = np.argsort(-scores)

        # Filter out self-match
        ranked_corpus = [corpus_ids[idx] for idx in ranked_indices if corpus_ids[idx] != qid]

        # Recall@K
        for k in k_values:
            retrieved = set(ranked_corpus[:k])
            recall = len(true_cited & retrieved) / len(true_cited)
            recalls[k].append(recall)

        # MRR
        for rank, cid in enumerate(ranked_corpus, 1):
            if cid in true_cited:
                mrrs.append(1.0 / rank)
                break
        else:
            mrrs.append(0.0)

    return {
        **{f"recall@{k}": np.mean(v) for k, v in recalls.items()},
        "mrr": np.mean(mrrs),
        "n_queries": len(mrrs),
    }


def main():
    output_dir = BENCHMARK_DIR / "embed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus and labels
    logger.info("Loading corpus...")
    corpus = load_corpus(TRAIN_DOCS_DIR)
    labels = json.loads(TRAIN_LABELS.read_text())

    # Preprocess all documents
    corpus_ids = sorted(corpus.keys())
    corpus_texts = [truncate_for_embedding(preprocess(corpus[cid])) for cid in corpus_ids]
    logger.info("Corpus: %d documents", len(corpus_ids))

    # Query IDs are keys in labels
    query_ids = sorted(labels.keys())
    logger.info("Queries: %d", len(query_ids))

    with OllamaClient(timeout=600.0) as client:
        available = {m["name"] for m in client.list_models()}

        for model in EMBED_CANDIDATES:
            model_available = any(model in m for m in available)
            if not model_available:
                logger.warning("Model %s not available, attempting to pull", model)
                try:
                    client.pull_model(model)
                except Exception:
                    logger.exception("Failed to pull %s, skipping", model)
                    continue

            logger.info("=== Benchmarking %s ===", model)
            start = time.time()

            try:
                # Embed full corpus
                embeddings = embed_texts(client, corpus_texts, model=model)
                embed_time = time.time() - start
                logger.info("Embedding time: %.1f seconds", embed_time)

                # Build query-corpus similarity matrix
                query_indices = [corpus_ids.index(qid) for qid in query_ids if qid in corpus_ids]
                valid_query_ids = [qid for qid in query_ids if qid in corpus_ids]
                query_embeddings = embeddings[query_indices]

                sim_matrix = cosine_similarity_matrix(query_embeddings, embeddings)

                # Compute metrics
                metrics = compute_retrieval_metrics(
                    sim_matrix, valid_query_ids, corpus_ids, labels
                )
                metrics["model"] = model
                metrics["embed_time_seconds"] = embed_time
                metrics["embed_dim"] = embeddings.shape[1]
                metrics["memory_mb"] = embeddings.nbytes / 1e6

                # Save
                safe_name = model.replace(":", "_").replace("/", "_")
                out_path = output_dir / f"{safe_name}.json"
                out_path.write_text(json.dumps(metrics, indent=2, default=str))

                logger.info(
                    "%s: R@50=%.3f R@100=%.3f R@200=%.3f MRR=%.3f time=%.0fs dim=%d",
                    model,
                    metrics["recall@50"],
                    metrics["recall@100"],
                    metrics["recall@200"],
                    metrics["mrr"],
                    embed_time,
                    metrics["embed_dim"],
                )
            except Exception:
                logger.exception("Failed to benchmark %s", model)

    # Print comparison table
    print("\n=== Embedding Benchmark Results ===")
    print(f"{'Model':<25} {'R@50':>6} {'R@100':>6} {'R@200':>6} {'MRR':>6} {'Time(s)':>8} {'Dim':>5}")
    print("-" * 65)
    for model in EMBED_CANDIDATES:
        safe_name = model.replace(":", "_").replace("/", "_")
        path = output_dir / f"{safe_name}.json"
        if path.exists():
            r = json.loads(path.read_text())
            print(
                f"{r['model']:<25} {r['recall@50']:>5.3f} {r['recall@100']:>5.3f} "
                f"{r['recall@200']:>5.3f} {r['mrr']:>5.3f} {r['embed_time_seconds']:>7.0f} "
                f"{r['embed_dim']:>4d}"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run embedding benchmark**

```bash
uv run python -m graphrag.run_benchmark_embed
```

- [ ] **Step 4: Commit**

```bash
git add src/graphrag/embed.py src/graphrag/run_benchmark_embed.py
git commit -m "feat: add embedding module and benchmark runner"
```

---

## Chunk 4: Graph Construction & Communities (Tasks 10-11)

### Task 10: Knowledge Graph Construction

**Files:**
- Create: `src/graphrag/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_graph.py
"""Tests for knowledge graph construction."""
import json
import tempfile
from pathlib import Path

import pytest
import networkx as nx

from graphrag.graph import build_knowledge_graph, save_graph, load_graph


@pytest.fixture
def sample_extractions(tmp_path):
    """Create sample extraction files."""
    docs = {
        "000001": {
            "doc_id": "000001",
            "statutes": ["IRPA", "IRPA s. 72(1)"],
            "judges": ["Mosley J."],
            "outcome": "application dismissed",
            "concepts": ["standard of review", "procedural fairness"],
            "tests": ["Dunsmuir test"],
            "holdings": ["Officer's decision was reasonable"],
            "case_type": "judicial_review",
            "domain": "immigration",
        },
        "000002": {
            "doc_id": "000002",
            "statutes": ["IRPA", "IRPA s. 96"],
            "judges": ["Russell J."],
            "outcome": "application allowed",
            "concepts": ["standard of review", "credibility"],
            "tests": ["Dunsmuir test"],
            "holdings": ["Board failed to consider evidence"],
            "case_type": "judicial_review",
            "domain": "immigration",
        },
    }
    ext_dir = tmp_path / "extractions"
    ext_dir.mkdir()
    for doc_id, data in docs.items():
        (ext_dir / f"{doc_id}.json").write_text(json.dumps(data))
    return ext_dir


class TestBuildGraph:
    def test_creates_case_nodes(self, sample_extractions):
        G = build_knowledge_graph(sample_extractions)
        case_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Case"]
        assert len(case_nodes) == 2

    def test_creates_statute_nodes(self, sample_extractions):
        G = build_knowledge_graph(sample_extractions)
        statute_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Statute"]
        assert len(statute_nodes) >= 2  # IRPA and at least one section

    def test_creates_concept_nodes(self, sample_extractions):
        G = build_knowledge_graph(sample_extractions)
        concept_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "LegalConcept"]
        assert len(concept_nodes) >= 2

    def test_shared_entities_create_paths(self, sample_extractions):
        G = build_knowledge_graph(sample_extractions)
        # Both cases share IRPA and "standard of review"
        # So there should be a path between them through shared entities
        assert nx.has_path(G, "case:000001", "case:000002")


class TestSaveLoad:
    def test_roundtrip(self, sample_extractions, tmp_path):
        G = build_knowledge_graph(sample_extractions)
        graph_path = tmp_path / "graph"
        save_graph(G, graph_path)
        G2 = load_graph(graph_path)
        assert set(G.nodes()) == set(G2.nodes())
        assert len(G.edges()) == len(G2.edges())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_graph.py -v
```

- [ ] **Step 3: Implement graph construction**

```python
# src/graphrag/graph.py
"""Knowledge graph construction from extracted entities."""
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def load_extractions(extractions_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all extraction JSON files from a directory."""
    extractions = {}
    for path in sorted(extractions_dir.glob("*.json")):
        data = json.loads(path.read_text())
        extractions[data["doc_id"]] = data
    logger.info("Loaded %d extractions from %s", len(extractions), extractions_dir)
    return extractions


def build_knowledge_graph(
    extractions_dir_or_dict: Path | dict[str, dict[str, Any]],
) -> nx.Graph:
    """Build a knowledge graph from entity extractions.

    Node types: Case, Statute, LegalConcept, LegalTest, Judge, Outcome, Domain
    Edge types: APPLIES, INVOKES_CONCEPT, APPLIES_TEST, DECIDED_BY, HAS_OUTCOME, IN_DOMAIN
    """
    if isinstance(extractions_dir_or_dict, Path):
        extractions = load_extractions(extractions_dir_or_dict)
    else:
        extractions = extractions_dir_or_dict

    G = nx.Graph()

    for doc_id, data in extractions.items():
        case_id = f"case:{doc_id}"

        # Add Case node
        G.add_node(
            case_id,
            type="Case",
            doc_id=doc_id,
            case_type=data.get("case_type", "other"),
            word_count=data.get("word_count", 0),
        )

        # Statutes
        for statute in data.get("statutes", []):
            statute_id = f"statute:{statute}"
            G.add_node(statute_id, type="Statute", name=statute)
            G.add_edge(case_id, statute_id, relation="APPLIES")

        # Legal concepts
        for concept in data.get("concepts", []):
            concept_id = f"concept:{concept}"
            G.add_node(concept_id, type="LegalConcept", name=concept)
            G.add_edge(case_id, concept_id, relation="INVOKES_CONCEPT")

        # Legal tests
        for test in data.get("tests", []):
            test_id = f"test:{test}"
            G.add_node(test_id, type="LegalTest", name=test)
            G.add_edge(case_id, test_id, relation="APPLIES_TEST")

        # Judges
        for judge in data.get("judges", []):
            judge_id = f"judge:{judge}"
            G.add_node(judge_id, type="Judge", name=judge)
            G.add_edge(case_id, judge_id, relation="DECIDED_BY")

        # Outcome
        outcome = data.get("outcome")
        if outcome:
            outcome_id = f"outcome:{outcome}"
            G.add_node(outcome_id, type="Outcome", name=outcome)
            G.add_edge(case_id, outcome_id, relation="HAS_OUTCOME")

        # Domain
        domain = data.get("domain", "other")
        domain_id = f"domain:{domain}"
        G.add_node(domain_id, type="Domain", name=domain)
        G.add_edge(case_id, domain_id, relation="IN_DOMAIN")

    # Log graph stats
    node_types = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    logger.info(
        "Graph built: %d nodes (%s), %d edges",
        G.number_of_nodes(),
        ", ".join(f"{k}:{v}" for k, v in sorted(node_types.items())),
        G.number_of_edges(),
    )
    return G


def save_graph(G: nx.Graph, output_dir: Path) -> None:
    """Save graph as GraphML + JSON metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, output_dir / "knowledge_graph.graphml")

    # Save as adjacency JSON for easier loading
    data = nx.node_link_data(G)
    (output_dir / "knowledge_graph.json").write_text(
        json.dumps(data, indent=2, default=str)
    )
    logger.info("Graph saved to %s", output_dir)


def load_graph(graph_dir: Path) -> nx.Graph:
    """Load graph from JSON."""
    data = json.loads((graph_dir / "knowledge_graph.json").read_text())
    return nx.node_link_graph(data)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_graph.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/graph.py tests/test_graph.py
git commit -m "feat: add knowledge graph construction from entity extractions"
```

---

### Task 11: Community Detection & Summarization

**Files:**
- Create: `src/graphrag/community.py`

- [ ] **Step 1: Implement community detection**

```python
# src/graphrag/community.py
"""Community detection and summarization for the knowledge graph."""
import logging
from typing import Any

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np

from graphrag.config import (
    COMMUNITY_EDGE_WEIGHTS,
    LEIDEN_RESOLUTION,
    LLM_MODEL,
)
from graphrag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

COMMUNITY_SUMMARY_PROMPT = """You are a legal analyst. Given a group of Federal Court of Canada cases
that form a legal community, summarize in 2-3 sentences the shared legal themes, common statutes,
and typical case outcomes. Be specific and factual.

Cases in this community:

{case_descriptions}

Summary:"""


def build_case_similarity_graph(
    G: nx.Graph,
    bm25_neighbors: dict[str, list[tuple[str, float]]] | None = None,
) -> nx.Graph:
    """Project the knowledge graph to a weighted Case-only graph.

    Edge weight formula:
        0.30 * shared_statutes + 0.30 * shared_concepts +
        0.30 * bm25_score + 0.05 * same_judge + 0.05 * same_domain
    """
    case_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Case"]
    logger.info("Building case similarity graph for %d cases", len(case_nodes))

    # Precompute entity sets per case
    case_entities: dict[str, dict[str, set[str]]] = {}
    for case_id in case_nodes:
        neighbors = G[case_id]
        entities: dict[str, set[str]] = {
            "statutes": set(),
            "concepts": set(),
            "tests": set(),
            "judges": set(),
            "domains": set(),
        }
        for neighbor, edge_data in neighbors.items():
            node_data = G.nodes[neighbor]
            ntype = node_data.get("type", "")
            if ntype == "Statute":
                entities["statutes"].add(neighbor)
            elif ntype == "LegalConcept":
                entities["concepts"].add(neighbor)
            elif ntype == "LegalTest":
                entities["tests"].add(neighbor)
            elif ntype == "Judge":
                entities["judges"].add(neighbor)
            elif ntype == "Domain":
                entities["domains"].add(neighbor)
        case_entities[case_id] = entities

    # Build weighted edges
    sim_graph = nx.Graph()
    sim_graph.add_nodes_from(case_nodes)

    # Normalize BM25 scores
    bm25_max = 1.0
    if bm25_neighbors:
        all_scores = [s for neighbors in bm25_neighbors.values() for _, s in neighbors]
        bm25_max = max(all_scores) if all_scores else 1.0

    w = COMMUNITY_EDGE_WEIGHTS
    processed = set()

    for case_a in case_nodes:
        ents_a = case_entities[case_a]
        for case_b in case_nodes:
            if case_a >= case_b:
                continue
            pair = (case_a, case_b)
            if pair in processed:
                continue
            processed.add(pair)

            ents_b = case_entities[case_b]

            shared_stat = len(ents_a["statutes"] & ents_b["statutes"])
            shared_conc = len(ents_a["concepts"] & ents_b["concepts"])
            same_judge = 1.0 if ents_a["judges"] & ents_b["judges"] else 0.0
            same_domain = 1.0 if ents_a["domains"] & ents_b["domains"] else 0.0

            # BM25 score
            bm25_score = 0.0
            doc_a = case_a.replace("case:", "")
            doc_b = case_b.replace("case:", "")
            if bm25_neighbors:
                for neighbor_id, score in bm25_neighbors.get(f"{doc_a}.txt", []):
                    if neighbor_id == f"{doc_b}.txt":
                        bm25_score = score / bm25_max
                        break

            # Compute weight
            weight = (
                w["shared_statutes"] * min(shared_stat / 3.0, 1.0)
                + w["shared_concepts"] * min(shared_conc / 3.0, 1.0)
                + w["bm25"] * bm25_score
                + w["same_judge"] * same_judge
                + w["same_domain"] * same_domain
            )

            if weight > 0.05:  # Threshold to avoid too many edges
                sim_graph.add_edge(case_a, case_b, weight=weight)

    logger.info(
        "Case similarity graph: %d nodes, %d edges",
        sim_graph.number_of_nodes(),
        sim_graph.number_of_edges(),
    )
    return sim_graph


def detect_communities(
    sim_graph: nx.Graph,
    resolution: float = LEIDEN_RESOLUTION,
) -> dict[str, int]:
    """Run Leiden algorithm on the case similarity graph.

    Returns dict mapping case_id -> community_id.
    """
    # Convert to igraph
    nodes = list(sim_graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))
    edges = []
    weights = []
    for u, v, d in sim_graph.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(d.get("weight", 1.0))

    ig_graph.add_edges(edges)

    # Run Leiden
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=42,
    )

    # Map back to case IDs
    communities = {}
    for comm_id, members in enumerate(partition):
        for idx in members:
            communities[nodes[idx]] = comm_id

    n_communities = len(partition)
    sizes = [len(m) for m in partition]
    logger.info(
        "Detected %d communities (min=%d, max=%d, median=%.0f)",
        n_communities,
        min(sizes),
        max(sizes),
        np.median(sizes),
    )
    return communities


def summarize_communities(
    communities: dict[str, int],
    extractions: dict[str, dict],
    client: OllamaClient,
    model: str = LLM_MODEL,
    max_cases_per_summary: int = 20,
) -> dict[int, str]:
    """Generate LLM summaries for each community's legal theme."""
    # Group cases by community
    comm_cases: dict[int, list[str]] = {}
    for case_id, comm_id in communities.items():
        comm_cases.setdefault(comm_id, []).append(case_id)

    summaries = {}
    for comm_id, case_ids in sorted(comm_cases.items()):
        # Build case descriptions
        descriptions = []
        for case_id in case_ids[:max_cases_per_summary]:
            doc_id = case_id.replace("case:", "")
            ext = extractions.get(doc_id, {})
            desc = (
                f"- {doc_id}: {ext.get('domain', 'unknown')} | "
                f"concepts: {', '.join(ext.get('concepts', [])[:3])} | "
                f"statutes: {', '.join(ext.get('statutes', [])[:3])} | "
                f"outcome: {ext.get('outcome', 'unknown')}"
            )
            descriptions.append(desc)

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            case_descriptions="\n".join(descriptions)
        )
        try:
            summary = client.generate(model, prompt, temperature=0.3, max_tokens=300)
            summaries[comm_id] = summary.strip()
        except Exception:
            logger.exception("Failed to summarize community %d", comm_id)
            summaries[comm_id] = f"Community {comm_id}: {len(case_ids)} cases"

        logger.info("Community %d (%d cases): %s...", comm_id, len(case_ids), summaries[comm_id][:80])

    return summaries
```

- [ ] **Step 2: Smoke test community detection**

```bash
uv run python -c "
import networkx as nx
from graphrag.community import detect_communities

G = nx.Graph()
for i in range(20):
    G.add_node(f'case:{i:06d}')
# Create two clusters
for i in range(10):
    for j in range(i+1, 10):
        G.add_edge(f'case:{i:06d}', f'case:{j:06d}', weight=0.8)
for i in range(10, 20):
    for j in range(i+1, 20):
        G.add_edge(f'case:{i:06d}', f'case:{j:06d}', weight=0.8)
# Weak cross-link
G.add_edge('case:000005', 'case:000015', weight=0.1)

comms = detect_communities(G)
print(f'Communities: {len(set(comms.values()))}')
print(comms)
"
```

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/community.py
git commit -m "feat: add community detection (Leiden) and LLM summarization"
```

---

## Chunk 5: Retrieval Signals & Fusion (Tasks 12-15)

### Task 12: BM25 Index

**Files:**
- Create: `src/graphrag/bm25.py`

- [ ] **Step 1: Implement BM25 wrapper**

```python
# src/graphrag/bm25.py
"""BM25 index for first-stage retrieval."""
import logging
import re
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """BM25 index over a corpus of documents."""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._doc_ids: list[str] = []

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Build BM25 index."""
        self._doc_ids = list(doc_ids)
        tokenized = [tokenize(t) for t in texts]
        self._index = BM25Okapi(tokenized)
        logger.info("BM25 index built: %d documents", len(doc_ids))

    def query(self, text: str, top_k: int = 200) -> list[tuple[str, float]]:
        """Query the index. Returns list of (doc_id, score) sorted by score descending."""
        assert self._index is not None, "Index not built. Call fit() first."
        tokens = tokenize(text)
        scores = self._index.get_scores(tokens)
        top_indices = np.argsort(-scores)[:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_indices]

    def get_all_neighbors(self, texts: dict[str, str], top_k: int = 20) -> dict[str, list[tuple[str, float]]]:
        """Get top-k BM25 neighbors for every document in the corpus.

        Args:
            texts: dict mapping doc_id -> text (must match fit() doc_ids)
            top_k: number of neighbors per document

        Returns:
            dict mapping doc_id -> [(neighbor_id, score), ...]
        """
        assert self._index is not None
        neighbors = {}
        for doc_id in self._doc_ids:
            results = self.query(texts[doc_id], top_k=top_k + 1)
            # Filter out self
            neighbors[doc_id] = [(did, s) for did, s in results if did != doc_id][:top_k]
        return neighbors
```

- [ ] **Step 2: Quick test**

```bash
uv run python -c "
from graphrag.bm25 import BM25Index
idx = BM25Index()
idx.fit(['a', 'b', 'c'], ['the cat sat on mat', 'dog barked loudly', 'cat and dog played'])
results = idx.query('cat sat', top_k=2)
print(results)
"
```

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/bm25.py
git commit -m "feat: add BM25 index wrapper for first-stage retrieval"
```

---

### Task 13: Multi-Signal Retrieval & Fusion

**Files:**
- Create: `src/graphrag/retrieve.py`
- Create: `tests/test_retrieve.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_retrieve.py
"""Tests for multi-signal retrieval and fusion."""
import pytest
import numpy as np
from graphrag.retrieve import reciprocal_rank_fusion, weighted_entity_score


class TestRRF:
    def test_basic_fusion(self):
        rankings = {
            "signal_1": [("a", 10), ("b", 8), ("c", 5)],
            "signal_2": [("b", 10), ("c", 8), ("a", 5)],
        }
        fused = reciprocal_rank_fusion(rankings, k=60)
        # All three docs should appear
        doc_ids = [d for d, _ in fused]
        assert set(doc_ids) == {"a", "b", "c"}
        # b should rank highest (rank 1+2 in signals) or a (rank 1+3)
        # b: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0326
        # a: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
        assert fused[0][0] == "b"

    def test_empty_rankings(self):
        result = reciprocal_rank_fusion({}, k=60)
        assert result == []


class TestWeightedEntityScore:
    def test_shared_entities(self):
        query_entities = {
            "statutes": {"IRPA", "Charter"},
            "concepts": {"standard of review"},
            "tests": {"Dunsmuir test"},
            "domain": "immigration",
            "judge": "Mosley J.",
        }
        candidate_entities = {
            "statutes": {"IRPA"},
            "concepts": {"standard of review", "procedural fairness"},
            "tests": {"Dunsmuir test"},
            "domain": "immigration",
            "judge": "Russell J.",
        }
        score = weighted_entity_score(query_entities, candidate_entities)
        assert 0 < score <= 1.0

    def test_no_overlap(self):
        query = {
            "statutes": {"IRPA"},
            "concepts": {"fairness"},
            "tests": set(),
            "domain": "immigration",
            "judge": "Mosley J.",
        }
        candidate = {
            "statutes": {"Patent Act"},
            "concepts": {"novelty"},
            "tests": set(),
            "domain": "IP",
            "judge": "Russell J.",
        }
        score = weighted_entity_score(query, candidate)
        assert score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_retrieve.py -v
```

- [ ] **Step 3: Implement retrieval module**

```python
# src/graphrag/retrieve.py
"""Multi-signal retrieval and fusion for case retrieval."""
import logging
from typing import Any, Sequence

import numpy as np

from graphrag.config import ENTITY_WEIGHTS, RRF_K, BM25_TOP_K, STAGE1_TOP_K

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: dict mapping signal_name -> [(doc_id, score), ...] sorted descending
        k: smoothing parameter

    Returns:
        Fused ranking as [(doc_id, rrf_score), ...] sorted descending
    """
    scores: dict[str, float] = {}
    for signal_name, ranked_list in rankings.items():
        for rank, (doc_id, _) in enumerate(ranked_list, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: -x[1])


def weighted_entity_score(
    query_entities: dict[str, Any],
    candidate_entities: dict[str, Any],
) -> float:
    """Compute weighted Jaccard-like entity overlap score.

    Weights: statutes 0.35, concepts 0.30, tests 0.20, domain 0.10, judge 0.05
    """
    w = ENTITY_WEIGHTS
    score = 0.0

    # Set overlap for statutes, concepts, tests
    for key in ("statutes", "concepts", "tests"):
        q_set = set(query_entities.get(key, set()))
        c_set = set(candidate_entities.get(key, set()))
        if q_set or c_set:
            union = q_set | c_set
            intersection = q_set & c_set
            score += w[key] * (len(intersection) / len(union) if union else 0)

    # Domain match
    if query_entities.get("domain") == candidate_entities.get("domain"):
        if query_entities.get("domain") != "other":
            score += w["domain"]

    # Judge match
    if query_entities.get("judge") and query_entities.get("judge") == candidate_entities.get("judge"):
        score += w["judge"]

    return score


def signal_entity_graph(
    query_id: str,
    query_entities: dict[str, Any],
    corpus_entities: dict[str, dict[str, Any]],
    top_k: int = BM25_TOP_K,
) -> list[tuple[str, float]]:
    """Signal S2: Entity graph traversal scoring."""
    scores = []
    for doc_id, ents in corpus_entities.items():
        if doc_id == query_id:
            continue
        score = weighted_entity_score(query_entities, ents)
        if score > 0:
            scores.append((doc_id, score))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def signal_community(
    query_embedding: np.ndarray,
    community_embeddings: np.ndarray,
    community_members: dict[int, list[str]],
    top_k: int = BM25_TOP_K,
) -> list[tuple[str, float]]:
    """Signal S3: Community matching via embedding similarity.

    Finds nearest communities to query, returns member cases with scores.
    """
    # Cosine similarity to each community
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    c_norms = community_embeddings / (
        np.linalg.norm(community_embeddings, axis=1, keepdims=True) + 1e-10
    )
    sims = c_norms @ q_norm

    # Get top communities
    top_comm_indices = np.argsort(-sims)[:5]

    # Assign scores to member cases
    scores: dict[str, float] = {}
    for comm_idx in top_comm_indices:
        comm_score = float(sims[comm_idx])
        for case_id in community_members.get(int(comm_idx), []):
            doc_id = case_id.replace("case:", "") + ".txt"
            scores[doc_id] = max(scores.get(doc_id, 0), comm_score)

    result = sorted(scores.items(), key=lambda x: -x[1])
    return result[:top_k]


def signal_embedding(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: list[str],
    query_id: str,
    top_k: int = BM25_TOP_K,
) -> list[tuple[str, float]]:
    """Signal S4: Dense embedding similarity."""
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    c_norms = corpus_embeddings / (
        np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
    )
    sims = c_norms @ q_norm

    top_indices = np.argsort(-sims)
    results = []
    for idx in top_indices:
        if corpus_ids[idx] == query_id:
            continue
        results.append((corpus_ids[idx], float(sims[idx])))
        if len(results) >= top_k:
            break
    return results


def retrieve_multi_signal(
    query_id: str,
    bm25_results: list[tuple[str, float]],
    entity_results: list[tuple[str, float]],
    community_results: list[tuple[str, float]],
    embedding_results: list[tuple[str, float]],
    top_k: int = STAGE1_TOP_K,
) -> list[tuple[str, float]]:
    """Stage 1: Fuse signals S1-S4 via RRF to get top-k candidates.

    These candidates are then passed to Stage 2 (reasoning chains).
    """
    rankings = {
        "bm25": bm25_results,
        "entity_graph": entity_results,
        "community": community_results,
        "embedding": embedding_results,
    }
    fused = reciprocal_rank_fusion(rankings)
    return fused[:top_k]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_retrieve.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/graphrag/retrieve.py tests/test_retrieve.py
git commit -m "feat: add multi-signal retrieval with RRF fusion"
```

---

### Task 14: Reasoning Chains

**Files:**
- Create: `src/graphrag/reasoning.py`

- [ ] **Step 1: Implement reasoning chain generation**

```python
# src/graphrag/reasoning.py
"""LLM reasoning chain generation for case pair similarity."""
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from graphrag.config import LLM_MODEL, EMBED_MODEL, OUTPUT_DIR
from graphrag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

REASONING_SYSTEM_PROMPT = """You are a legal analyst specializing in Federal Court of Canada case law."""

REASONING_USER_TEMPLATE = """Given two Federal Court of Canada cases, explain in 2-3 sentences
WHY they are legally related. Focus on:
1. Shared legal principles or tests applied
2. Similar factual patterns or issues
3. How one case's reasoning builds on or departs from the other

Case A ({case_a_id}):
- Domain: {case_a_domain}
- Concepts: {case_a_concepts}
- Tests: {case_a_tests}
- Statutes: {case_a_statutes}
- Holdings: {case_a_holdings}

Case B ({case_b_id}):
- Domain: {case_b_domain}
- Concepts: {case_b_concepts}
- Tests: {case_b_tests}
- Statutes: {case_b_statutes}
- Holdings: {case_b_holdings}

Reasoning chain:"""


def generate_reasoning_chain(
    client: OllamaClient,
    case_a: dict[str, Any],
    case_b: dict[str, Any],
    model: str = LLM_MODEL,
) -> str:
    """Generate a reasoning chain explaining the relationship between two cases."""
    prompt = REASONING_USER_TEMPLATE.format(
        case_a_id=case_a.get("doc_id", "unknown"),
        case_a_domain=case_a.get("domain", "unknown"),
        case_a_concepts=", ".join(case_a.get("concepts", [])[:5]),
        case_a_tests=", ".join(case_a.get("tests", [])[:3]),
        case_a_statutes=", ".join(case_a.get("statutes", [])[:5]),
        case_a_holdings="; ".join(case_a.get("holdings", [])[:2]),
        case_b_id=case_b.get("doc_id", "unknown"),
        case_b_domain=case_b.get("domain", "unknown"),
        case_b_concepts=", ".join(case_b.get("concepts", [])[:5]),
        case_b_tests=", ".join(case_b.get("tests", [])[:3]),
        case_b_statutes=", ".join(case_b.get("statutes", [])[:5]),
        case_b_holdings="; ".join(case_b.get("holdings", [])[:2]),
    )
    return client.generate(
        model,
        prompt,
        system=REASONING_SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=300,
    ).strip()


def generate_chains_for_pairs(
    client: OllamaClient,
    pairs: list[tuple[str, str]],
    extractions: dict[str, dict[str, Any]],
    output_dir: Path | None = None,
    model: str = LLM_MODEL,
) -> dict[tuple[str, str], str]:
    """Generate reasoning chains for a list of case pairs with resume support.

    Args:
        pairs: list of (query_id, candidate_id) tuples (without .txt extension)
        extractions: dict mapping doc_id -> extraction result
        output_dir: if provided, saves chains incrementally for resume

    Returns:
        dict mapping (query_id, candidate_id) -> reasoning chain text
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check what's already done
    done: dict[tuple[str, str], str] = {}
    if output_dir:
        chains_file = output_dir / "reasoning_chains.json"
        if chains_file.exists():
            raw = json.loads(chains_file.read_text())
            for key, chain in raw.items():
                parts = key.split("|")
                if len(parts) == 2:
                    done[(parts[0], parts[1])] = chain
            logger.info("Loaded %d existing chains", len(done))

    remaining = [p for p in pairs if p not in done]
    logger.info("Generating %d reasoning chains (%d already done)", len(remaining), len(done))

    chains = dict(done)
    start_time = time.time()

    for i, (qid, cid) in enumerate(remaining):
        case_a = extractions.get(qid, {})
        case_b = extractions.get(cid, {})

        try:
            chain = generate_reasoning_chain(client, case_a, case_b, model=model)
            chains[(qid, cid)] = chain
        except Exception:
            logger.exception("Failed to generate chain for %s-%s", qid, cid)
            chains[(qid, cid)] = ""

        # Save periodically
        if output_dir and (i + 1) % 100 == 0:
            _save_chains(chains, output_dir)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "[%d/%d] %.1f chains/min, ETA %.0f min",
                i + 1, len(remaining), rate * 60, eta / 60,
            )

    if output_dir:
        _save_chains(chains, output_dir)

    return chains


def _save_chains(chains: dict[tuple[str, str], str], output_dir: Path) -> None:
    """Save chains to disk."""
    serializable = {f"{k[0]}|{k[1]}": v for k, v in chains.items()}
    (output_dir / "reasoning_chains.json").write_text(
        json.dumps(serializable, indent=2)
    )


def score_reasoning_chains(
    client: OllamaClient,
    query_chain_text: str,
    candidate_chains: dict[str, str],
    model: str = EMBED_MODEL,
) -> list[tuple[str, float]]:
    """Score candidates by reasoning chain embedding similarity.

    Signal S5: embed the query's relationship description and compare to
    pre-computed candidate chain embeddings.
    """
    if not candidate_chains:
        return []

    # Embed query chain
    q_vec = np.array(client.embed(model, [query_chain_text])[0])

    # Embed candidate chains
    cand_ids = list(candidate_chains.keys())
    cand_texts = [candidate_chains[cid] for cid in cand_ids]

    # Filter out empty chains
    valid = [(cid, t) for cid, t in zip(cand_ids, cand_texts) if t.strip()]
    if not valid:
        return []

    valid_ids, valid_texts = zip(*valid)
    cand_vecs = np.array(client.embed(model, list(valid_texts)))

    # Cosine similarity
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    c_norms = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-10)
    sims = c_norms @ q_norm

    results = [(cid, float(sim)) for cid, sim in zip(valid_ids, sims)]
    results.sort(key=lambda x: -x[1])
    return results
```

- [ ] **Step 2: Commit**

```bash
git add src/graphrag/reasoning.py
git commit -m "feat: add reasoning chain generation and scoring"
```

---

### Task 15: Scoring Metrics

**Files:**
- Create: `src/graphrag/metrics.py`

- [ ] **Step 1: Implement scoring and threshold optimization**

```python
# src/graphrag/metrics.py
"""Scoring metrics and threshold optimization for COLIEE Task 1."""
import logging
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def micro_f1(
    predictions: dict[str, list[str]],
    labels: dict[str, list[str]],
) -> dict[str, float]:
    """Compute micro-averaged F1, precision, and recall.

    This matches the official COLIEE evaluation metric.
    """
    tp = 0
    fp = 0
    fn = 0

    for query_id in labels:
        true_set = set(labels[query_id])
        pred_set = set(predictions.get(query_id, []))

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def scores_to_predictions(
    scores: dict[str, list[tuple[str, float]]],
    threshold: float,
) -> dict[str, list[str]]:
    """Convert scored candidates to binary predictions using a threshold.

    Args:
        scores: dict mapping query_id -> [(candidate_id, score), ...]
        threshold: minimum score to predict as positive

    Returns:
        dict mapping query_id -> [predicted_candidate_ids]
    """
    predictions = {}
    for query_id, candidates in scores.items():
        predictions[query_id] = [
            cid for cid, score in candidates if score >= threshold
        ]
    return predictions


def optimize_threshold(
    scores: dict[str, list[tuple[str, float]]],
    labels: dict[str, list[str]],
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the threshold that maximizes micro-F1 on the given data.

    Returns:
        (best_threshold, best_metrics_dict)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics = {}

    for t in thresholds:
        preds = scores_to_predictions(scores, float(t))
        metrics = micro_f1(preds, labels)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(t)
            best_metrics = metrics

    logger.info(
        "Best threshold=%.3f: F1=%.4f P=%.4f R=%.4f",
        best_threshold,
        best_metrics["f1"],
        best_metrics["precision"],
        best_metrics["recall"],
    )
    return best_threshold, best_metrics
```

- [ ] **Step 2: Quick test**

```bash
uv run python -c "
from graphrag.metrics import micro_f1, scores_to_predictions, optimize_threshold
import numpy as np

labels = {'q1': ['a', 'b'], 'q2': ['c']}
scores = {'q1': [('a', 0.9), ('b', 0.7), ('d', 0.3)], 'q2': [('c', 0.8), ('e', 0.2)]}
preds = scores_to_predictions(scores, 0.5)
print('Predictions:', preds)
m = micro_f1(preds, labels)
print('Metrics:', m)
best_t, best_m = optimize_threshold(scores, labels)
print(f'Best threshold: {best_t}, F1: {best_m[\"f1\"]}')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/graphrag/metrics.py
git commit -m "feat: add scoring metrics and threshold optimization"
```

---

## Chunk 6: Full Pipeline Orchestration (Task 16)

### Task 16: Indexing & End-to-End Pipeline

**Files:**
- Create: `src/graphrag/run_index.py`
- Create: `src/graphrag/run_pipeline.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create shared test fixtures**

```python
# tests/conftest.py
"""Shared test fixtures."""
import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_corpus(tmp_path):
    """Create a small sample corpus for testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    docs = {
        "000001.txt": "[1] This is a judicial review of an immigration decision. "
        "The Immigration and Refugee Protection Act, S.C. 2001, c. 27 applies. "
        "Mosley J. The standard of review is reasonableness per Dunsmuir. "
        "The application is dismissed.",
        "000002.txt": "[1] Another immigration case reviewing officer credibility finding. "
        "IRPA s. 96 applies. Russell J. Procedural fairness was breached. "
        "The application is allowed.",
        "000003.txt": "[1] Patent infringement case under the Patent Act. "
        "The test for claim construction follows Free World Trust. "
        "Hughes J. The motion is dismissed.",
    }
    for name, text in docs.items():
        (docs_dir / name).write_text(text)

    labels = {"000001.txt": ["000002.txt"]}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    return docs_dir, labels_path
```

- [ ] **Step 2: Create indexing runner**

```python
# src/graphrag/run_index.py
"""Full indexing pipeline: extraction -> graph -> communities -> embeddings."""
import json
import logging
import time
from pathlib import Path

from graphrag.config import (
    TRAIN_DOCS_DIR,
    EXTRACTIONS_DIR,
    GRAPH_DIR,
    EMBEDDINGS_DIR,
    LLM_MODEL,
    EMBED_MODEL,
)
from graphrag.ollama_client import OllamaClient
from graphrag.preprocess import preprocess, load_corpus
from graphrag.run_extract import run_extraction
from graphrag.graph import build_knowledge_graph, load_extractions, save_graph
from graphrag.community import (
    build_case_similarity_graph,
    detect_communities,
    summarize_communities,
)
from graphrag.bm25 import BM25Index
from graphrag.embed import embed_texts

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_indexing(skip_extraction: bool = False):
    """Run the full indexing pipeline."""
    total_start = time.time()

    # Phase 1: Entity Extraction
    if not skip_extraction:
        logger.info("=== Phase 1: Entity Extraction ===")
        run_extraction(TRAIN_DOCS_DIR, EXTRACTIONS_DIR / "train", model=LLM_MODEL)
    else:
        logger.info("Skipping extraction (using existing files)")

    # Phase 2: Graph Construction
    logger.info("=== Phase 2: Graph Construction ===")
    extractions = load_extractions(EXTRACTIONS_DIR / "train")
    G = build_knowledge_graph(extractions)

    # Add BM25 neighbor edges
    logger.info("Computing BM25 neighbors...")
    corpus = load_corpus(TRAIN_DOCS_DIR)
    corpus_texts = {k: preprocess(v) for k, v in corpus.items()}
    bm25 = BM25Index()
    bm25.fit(list(corpus_texts.keys()), list(corpus_texts.values()))
    bm25_neighbors = bm25.get_all_neighbors(corpus_texts, top_k=20)

    # Add BM25 edges to graph
    for doc_id, neighbors in bm25_neighbors.items():
        case_a = f"case:{doc_id.replace('.txt', '')}"
        for neighbor_id, score in neighbors:
            case_b = f"case:{neighbor_id.replace('.txt', '')}"
            if G.has_node(case_a) and G.has_node(case_b):
                G.add_edge(case_a, case_b, relation="BM25_NEIGHBOR", weight=score)

    save_graph(G, GRAPH_DIR)

    # Phase 3: Community Detection
    logger.info("=== Phase 3: Community Detection ===")
    sim_graph = build_case_similarity_graph(G, bm25_neighbors=bm25_neighbors)
    communities = detect_communities(sim_graph)

    # Save communities
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    (GRAPH_DIR / "communities.json").write_text(
        json.dumps(communities, indent=2)
    )

    # Summarize communities
    with OllamaClient() as client:
        summaries = summarize_communities(
            communities, extractions, client, model=LLM_MODEL
        )
        (GRAPH_DIR / "community_summaries.json").write_text(
            json.dumps(summaries, indent=2)
        )

        # Phase 4: Embed community summaries
        logger.info("=== Phase 4: Embedding ===")
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        # Embed community summaries
        summary_texts = [summaries[i] for i in sorted(summaries.keys())]
        comm_embeddings = embed_texts(client, summary_texts, model=EMBED_MODEL)
        np.save(EMBEDDINGS_DIR / "community_embeddings.npy", comm_embeddings)

        # Embed full corpus
        corpus_ids = sorted(corpus_texts.keys())
        # Truncate to ~4000 words for embedding
        truncated = [" ".join(corpus_texts[cid].split()[:4000]) for cid in corpus_ids]
        corpus_embeddings = embed_texts(client, truncated, model=EMBED_MODEL)
        np.save(EMBEDDINGS_DIR / "corpus_embeddings.npy", corpus_embeddings)

        # Save corpus ID order
        (EMBEDDINGS_DIR / "corpus_ids.json").write_text(json.dumps(corpus_ids))

    elapsed = time.time() - total_start
    logger.info("=== Indexing complete in %.1f hours ===", elapsed / 3600)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run full indexing pipeline")
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()
    run_indexing(skip_extraction=args.skip_extraction)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create end-to-end pipeline runner**

```python
# src/graphrag/run_pipeline.py
"""End-to-end retrieval pipeline with ablation support."""
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

from graphrag.config import (
    TRAIN_DOCS_DIR,
    TRAIN_LABELS,
    EXTRACTIONS_DIR,
    GRAPH_DIR,
    EMBEDDINGS_DIR,
    OUTPUT_DIR,
    LLM_MODEL,
    EMBED_MODEL,
    BM25_TOP_K,
    STAGE1_TOP_K,
    N_FOLDS,
    RANDOM_SEED,
)
from graphrag.ollama_client import OllamaClient
from graphrag.preprocess import preprocess, load_corpus
from graphrag.graph import load_extractions, load_graph
from graphrag.bm25 import BM25Index
from graphrag.embed import embed_texts, cosine_similarity_matrix
from graphrag.retrieve import (
    signal_entity_graph,
    signal_community,
    signal_embedding,
    retrieve_multi_signal,
    reciprocal_rank_fusion,
)
from graphrag.reasoning import generate_chains_for_pairs, score_reasoning_chains
from graphrag.metrics import micro_f1, scores_to_predictions, optimize_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_all_resources():
    """Load all pre-built resources needed for retrieval."""
    logger.info("Loading resources...")

    # Labels
    labels = json.loads(TRAIN_LABELS.read_text())

    # Corpus
    corpus = load_corpus(TRAIN_DOCS_DIR)
    corpus_texts = {k: preprocess(v) for k, v in corpus.items()}

    # Extractions
    extractions = load_extractions(EXTRACTIONS_DIR / "train")

    # Build entity lookup
    entity_lookup = {}
    for doc_id, ext in extractions.items():
        entity_lookup[f"{doc_id}.txt"] = {
            "statutes": set(ext.get("statutes", [])),
            "concepts": set(ext.get("concepts", [])),
            "tests": set(ext.get("tests", [])),
            "domain": ext.get("domain", "other"),
            "judge": ext.get("judges", [""])[0] if ext.get("judges") else "",
        }

    # BM25
    bm25 = BM25Index()
    bm25.fit(list(corpus_texts.keys()), list(corpus_texts.values()))

    # Embeddings
    corpus_ids = json.loads((EMBEDDINGS_DIR / "corpus_ids.json").read_text())
    corpus_embeddings = np.load(EMBEDDINGS_DIR / "corpus_embeddings.npy")
    community_embeddings = np.load(EMBEDDINGS_DIR / "community_embeddings.npy")

    # Communities
    communities = json.loads((GRAPH_DIR / "communities.json").read_text())
    community_members: dict[int, list[str]] = {}
    for case_id, comm_id in communities.items():
        community_members.setdefault(comm_id, []).append(case_id)

    return {
        "labels": labels,
        "corpus_texts": corpus_texts,
        "extractions": extractions,
        "entity_lookup": entity_lookup,
        "bm25": bm25,
        "corpus_ids": corpus_ids,
        "corpus_embeddings": corpus_embeddings,
        "community_embeddings": community_embeddings,
        "community_members": community_members,
    }


def run_ablation(resources: dict, configs: list[dict]) -> None:
    """Run ablation study across different signal configurations."""
    labels = resources["labels"]
    query_ids = sorted(labels.keys())

    results_dir = OUTPUT_DIR / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        config_name = config["name"]
        signals = config["signals"]
        logger.info("=== Ablation: %s (signals: %s) ===", config_name, signals)

        all_scores: dict[str, list[tuple[str, float]]] = {}

        for qi, qid in enumerate(query_ids):
            rankings = {}
            query_text = resources["corpus_texts"].get(qid, "")

            # S1: BM25
            if "bm25" in signals:
                rankings["bm25"] = resources["bm25"].query(query_text, top_k=BM25_TOP_K)

            # S2: Entity graph
            if "entity" in signals:
                q_ents = resources["entity_lookup"].get(qid, {})
                rankings["entity"] = signal_entity_graph(
                    qid, q_ents, resources["entity_lookup"]
                )

            # S3: Community
            if "community" in signals:
                # Get query embedding
                if qid in resources["corpus_ids"]:
                    q_idx = resources["corpus_ids"].index(qid)
                    q_emb = resources["corpus_embeddings"][q_idx]
                    rankings["community"] = signal_community(
                        q_emb,
                        resources["community_embeddings"],
                        resources["community_members"],
                    )

            # S4: Embedding
            if "embedding" in signals:
                if qid in resources["corpus_ids"]:
                    q_idx = resources["corpus_ids"].index(qid)
                    q_emb = resources["corpus_embeddings"][q_idx]
                    rankings["embedding"] = signal_embedding(
                        q_emb,
                        resources["corpus_embeddings"],
                        resources["corpus_ids"],
                        qid,
                    )

            # Fuse available signals
            fused = reciprocal_rank_fusion(rankings)
            all_scores[qid] = fused

            if (qi + 1) % 200 == 0:
                logger.info("  Processed %d/%d queries", qi + 1, len(query_ids))

        # Optimize threshold
        best_t, best_metrics = optimize_threshold(all_scores, labels)

        # Save results
        result = {
            "config": config_name,
            "signals": signals,
            "best_threshold": best_t,
            **best_metrics,
        }
        (results_dir / f"{config_name}.json").write_text(
            json.dumps(result, indent=2)
        )

        logger.info(
            "%s: F1=%.4f P=%.4f R=%.4f (threshold=%.3f)",
            config_name,
            best_metrics["f1"],
            best_metrics["precision"],
            best_metrics["recall"],
            best_t,
        )

    # Print comparison table
    print("\n=== Ablation Results ===")
    print(f"{'Config':<35} {'F1':>7} {'P':>7} {'R':>7} {'Threshold':>10}")
    print("-" * 70)
    for config in configs:
        path = results_dir / f"{config['name']}.json"
        if path.exists():
            r = json.loads(path.read_text())
            print(
                f"{r['config']:<35} {r['f1']:>6.4f} {r['precision']:>6.4f} "
                f"{r['recall']:>6.4f} {r['best_threshold']:>9.3f}"
            )


def main():
    resources = load_all_resources()

    # Ablation configurations
    configs = [
        {"name": "01_bm25_only", "signals": ["bm25"]},
        {"name": "02_bm25_entity", "signals": ["bm25", "entity"]},
        {"name": "03_bm25_entity_community", "signals": ["bm25", "entity", "community"]},
        {"name": "04_bm25_entity_community_embed", "signals": ["bm25", "entity", "community", "embedding"]},
        {"name": "05_all_signals", "signals": ["bm25", "entity", "community", "embedding", "reasoning"]},
    ]

    run_ablation(resources, configs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add src/graphrag/run_index.py src/graphrag/run_pipeline.py tests/conftest.py
git commit -m "feat: add indexing pipeline and end-to-end retrieval with ablation"
```

---

## Execution Sequence

After all code is written and committed, execute in this order:

```bash
# 1. Install dependencies
uv sync

# 2. Pull required models (if not already available)
uv run python -c "
from graphrag.ollama_client import OllamaClient
with OllamaClient() as c:
    models = [m['name'] for m in c.list_models()]
    print('Available:', models)
    for m in ['gemma3:27b', 'qwen3-embedding:8b', 'bge-m3', 'nomic-embed-text']:
        if not any(m in n for n in models):
            print(f'Pulling {m}...')
            c.pull_model(m)
"

# 3. Run LLM benchmark (~6-8 hours)
uv run python -m graphrag.run_benchmark_llm

# 4. Run embedding benchmark (~2-4 hours)
uv run python -m graphrag.run_benchmark_embed

# 5. Update config.py with best models from benchmarks

# 6. Run full indexing pipeline (~15-20 hours)
uv run python -m graphrag.run_index

# 7. Run ablation study
uv run python -m graphrag.run_pipeline
```

---

## Success Criteria

- All tests pass: `uv run pytest tests/ -v`
- LLM benchmark produces valid JSON >90% of the time with best model
- Embedding benchmark shows Recall@200 improvement over BM25-only baseline
- Knowledge graph has ~9K-11K nodes and ~350K edges
- Community detection finds ~100-300 communities
- Ablation shows incremental improvement as signals are added
- Full pipeline micro-F1 exceeds BM25-only baseline on dev set
