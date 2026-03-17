"""Synthetic Data Augmentation: LLM-based training pair generation.

Generates additional (query, positive_candidate) training pairs by:
1. Extracting key legal facts from positive documents using an LLM
2. Generating synthetic search queries from those facts
3. Pairing synthetic queries with their source documents

This expands the training set beyond the 8,251 gold pairs, improving
bi-encoder and cross-encoder training.

References:
  - Gao et al., "LEAD" (EMNLP 2024, arXiv:2410.06581)
  - InPars+ (arXiv:2508.13930)
"""
import json
import logging
import time
from pathlib import Path

import requests

from coliee_task1.config import (
    SYNTHETIC_LLM_MODEL,
    SYNTHETIC_N_PAIRS,
    SYNTHETIC_MAX_WORDS,
    OLLAMA_BASE_URL,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a legal document analyst. Extract the key legal information from this Canadian federal court case excerpt.

Provide a structured summary with:
1. Area of law (e.g., immigration, tax, IP, criminal, administrative)
2. Main legal issue or question
3. Key statutes or regulations referenced
4. Core facts of the case
5. Outcome or ruling

Keep your response under {max_words} words. Be precise and factual.

CASE EXCERPT:
{text}

STRUCTURED SUMMARY:"""

QUERY_GENERATION_PROMPT = """You are a legal research assistant. Based on the following summary of a Canadian federal court case, write a realistic legal research query that someone would use to find this case or similar cases.

The query should:
- Sound like a natural legal research question
- Reference the key legal issues and area of law
- Be 2-4 sentences long
- Not directly copy phrases from the summary

CASE SUMMARY:
{summary}

RESEARCH QUERY:"""


def build_extraction_prompt(text: str, max_words: int = SYNTHETIC_MAX_WORDS) -> str:
    """Construct fact extraction prompt with truncated case text.

    Args:
        text: Full case text
        max_words: Max words for the response

    Returns:
        Formatted prompt string
    """
    # Truncate input to ~1000 words to keep prompt manageable
    words = text.split()[:1000]
    return EXTRACTION_PROMPT.format(
        text=" ".join(words),
        max_words=max_words,
    )


def build_synthetic_pair(
    query_summary: str,
    candidate_text: str,
    candidate_id: str,
) -> dict:
    """Create a synthetic training pair.

    Args:
        query_summary: Generated query text
        candidate_text: Original case text (the positive document)
        candidate_id: Document ID of the positive document

    Returns:
        Dict with query, candidate_id, candidate_text, label
    """
    return {
        "query": query_summary,
        "candidate_id": candidate_id,
        "candidate_text": candidate_text[:2000],  # truncate for storage
        "label": 1,
        "synthetic": True,
    }


def _ollama_generate(prompt: str, model: str = SYNTHETIC_LLM_MODEL) -> str:
    """Call Ollama API for text generation.

    Args:
        prompt: Input prompt
        model: Ollama model name

    Returns:
        Generated text
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        logger.warning("Ollama call failed: %s", e)
        return ""


def generate_synthetic_pairs(
    corpus_texts: dict[str, str],
    labels: dict[str, list[str]],
    n_pairs: int = SYNTHETIC_N_PAIRS,
    output_path: Path | None = None,
) -> list[dict]:
    """Generate synthetic training pairs from labeled positive documents.

    Pipeline:
    1. Collect all unique positive document IDs from labels
    2. For each positive doc: extract key facts via LLM
    3. Generate a synthetic query from the facts
    4. Pair the synthetic query with the source document

    Args:
        corpus_texts: {doc_id: text} for all documents
        labels: {query_id: [positive_doc_ids]}
        n_pairs: Target number of synthetic pairs
        output_path: Path to save JSONL output (default: output/synthetic_pairs.jsonl)

    Returns:
        List of synthetic pair dicts
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "synthetic_pairs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect unique positive documents
    positive_docs = set()
    for qid, pos_list in labels.items():
        for pid in pos_list:
            if pid in corpus_texts:
                positive_docs.add(pid)

    positive_docs = sorted(positive_docs)
    logger.info("Generating synthetic pairs from %d positive documents (target: %d pairs)",
                len(positive_docs), n_pairs)

    # Limit to what we need
    import numpy as np
    rng = np.random.default_rng(42)
    if len(positive_docs) > n_pairs:
        selected = list(rng.choice(positive_docs, size=n_pairs, replace=False))
    else:
        # Repeat documents to reach target
        repeats = (n_pairs // len(positive_docs)) + 1
        selected = (positive_docs * repeats)[:n_pairs]

    pairs = []
    t0 = time.time()

    with open(output_path, "w") as f:
        for i, doc_id in enumerate(selected):
            text = corpus_texts.get(doc_id, "")
            if not text:
                continue

            # Step 1: Extract key facts
            extract_prompt = build_extraction_prompt(text)
            summary = _ollama_generate(extract_prompt)
            if not summary.strip():
                continue

            # Step 2: Generate synthetic query
            query_prompt = QUERY_GENERATION_PROMPT.format(summary=summary)
            synthetic_query = _ollama_generate(query_prompt)
            if not synthetic_query.strip():
                continue

            # Step 3: Create pair
            pair = build_synthetic_pair(synthetic_query, text, doc_id)
            pairs.append(pair)

            # Write incrementally
            f.write(json.dumps(pair) + "\n")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed * 3600
                logger.info(
                    "  Synthetic: %d/%d pairs (%.0f pairs/hr, %d saved)",
                    i + 1, len(selected), rate, len(pairs),
                )

    logger.info("Generated %d synthetic pairs in %.1f minutes (saved to %s)",
                len(pairs), (time.time() - t0) / 60, output_path)
    return pairs
