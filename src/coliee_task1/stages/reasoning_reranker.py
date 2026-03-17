"""Reasoning Reranker: chain-of-thought scoring for legal citation relevance.

Uses a 7B instruction-tuned LLM to generate explicit reasoning about why
a query case would cite a candidate case, producing reasoning-informed scores.

Features produced per (query, candidate) pair:
  - reasoning_score: LLM-assessed citation relevance (0-1)
  - reasoning_rank: Rank by reasoning_score within query

References:
  - Weller et al., "Rank1" (arXiv:2502.18418)
  - Zhang et al., "Rank-R1" (arXiv:2503.06034)
"""
import logging
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coliee_task1.config import (
    REASONING_MODEL,
    REASONING_MAX_LENGTH,
    REASONING_TOP_K,
    REASONING_TEMPERATURE,
)

logger = logging.getLogger(__name__)

REASONING_PROMPT_TEMPLATE = """You are a legal citation analysis expert specializing in Canadian federal court law.

Given a QUERY case and a CANDIDATE case, determine whether the query case would cite the candidate case as a legal precedent.

Analyze step by step:
1. Area of law: Do both cases deal with the same area (immigration, tax, IP, etc.)?
2. Legal issues: Do they address similar legal questions or principles?
3. Precedent relationship: Does the candidate establish principles the query case would rely on?
4. Shared statutes: Do they reference the same legislation or regulatory framework?
5. Factual similarity: Are the underlying facts analogous?

QUERY CASE (excerpt):
{query_text}

CANDIDATE CASE (excerpt):
{candidate_text}

After your analysis, provide:
- Relevant: Yes or No
- Score: A number between 0.00 and 1.00 indicating citation likelihood

Format your final answer exactly as:
Relevant: [Yes/No]
Score: [0.00-1.00]"""


def build_reasoning_prompt(query_text: str, candidate_text: str, max_words: int = 500) -> str:
    """Construct reasoning prompt with truncated case texts.

    Args:
        query_text: Full text of the query case
        candidate_text: Full text of the candidate case
        max_words: Maximum words per text excerpt

    Returns:
        Formatted prompt string
    """
    q_words = query_text.split()[:max_words]
    c_words = candidate_text.split()[:max_words]
    return REASONING_PROMPT_TEMPLATE.format(
        query_text=" ".join(q_words),
        candidate_text=" ".join(c_words),
    )


def parse_relevance_score(output: str) -> float:
    """Extract relevance score from model output.

    Looks for 'Score: X.XX' pattern. Falls back to:
    - 0.75 if 'Yes' found without explicit score
    - 0.25 if 'No' found without explicit score
    - 0.5 if unparseable (neutral)
    """
    # Try to find explicit score
    score_match = re.search(r"Score:\s*([\d.]+)", output)
    if score_match:
        try:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

    # Fallback: look for Yes/No verdict
    relevant_match = re.search(r"Relevant:\s*(Yes|No)", output, re.IGNORECASE)
    if relevant_match:
        return 0.75 if relevant_match.group(1).lower() == "yes" else 0.25

    return 0.5  # neutral fallback


def load_reasoning_model(
    model_name: str = REASONING_MODEL,
) -> tuple:
    """Load HuggingFace causal LM for reasoning.

    Returns:
        (model, tokenizer)
    """
    logger.info("Loading reasoning model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    logger.info("Reasoning model loaded")
    return model, tokenizer


def reasoning_rerank(
    model,
    tokenizer,
    query_text: str,
    candidates: list[tuple[str, str]],
    max_length: int = REASONING_MAX_LENGTH,
    temperature: float = REASONING_TEMPERATURE,
) -> list[tuple[str, float, str]]:
    """Score candidates using reasoning chain.

    Args:
        model: HuggingFace causal LM
        tokenizer: Associated tokenizer
        query_text: Query case text
        candidates: List of (candidate_id, candidate_text)
        max_length: Max input tokens
        temperature: Generation temperature

    Returns:
        List of (candidate_id, score, reasoning_text)
    """
    results = []

    for cid, cand_text in candidates:
        prompt = build_reasoning_prompt(query_text, cand_text)

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        score = parse_relevance_score(generated)
        results.append((cid, score, generated))

    return results


def batch_reasoning_rerank(
    query_ids: list[str],
    corpus_texts: dict[str, str],
    rrf_results: dict[str, list[tuple[str, float]]],
    top_k: int = REASONING_TOP_K,
) -> dict[str, dict[str, float]]:
    """Run reasoning reranker across all queries.

    Args:
        query_ids: List of query document IDs
        corpus_texts: {doc_id: text} for all documents
        rrf_results: {query_id: [(candidate_id, score), ...]} from BM25/RRF
        top_k: Number of top candidates to rerank per query

    Returns:
        {query_id: {candidate_id: reasoning_score}}
    """
    model, tokenizer = load_reasoning_model()

    logger.info("Reasoning reranker: %d queries, top-%d candidates each", len(query_ids), top_k)
    t0 = time.time()

    results: dict[str, dict[str, float]] = {}

    for i, qid in enumerate(query_ids):
        query_text = corpus_texts.get(qid, "")
        if not query_text:
            continue

        candidates_raw = rrf_results.get(qid, [])[:top_k]
        candidates = [
            (cid, corpus_texts.get(cid, ""))
            for cid, _ in candidates_raw
            if cid in corpus_texts
        ]

        if not candidates:
            continue

        reranked = reasoning_rerank(model, tokenizer, query_text, candidates)
        results[qid] = {cid: score for cid, score, _ in reranked}

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 3600
            logger.info(
                "  Reasoning: %d/%d queries (%.0f queries/hr)",
                i + 1, len(query_ids), rate,
            )

    logger.info("Reasoning reranker complete in %.1f minutes", (time.time() - t0) / 60)
    return results
