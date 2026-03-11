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
