"""LLM reasoning chain generation for case pair similarity."""
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from graphrag.config import LLM_MODEL, EMBED_MODEL
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
