"""LLM-based query expansion for BM25 recall improvement.

Generates a pseudo-document from query text using an LLM, then
concatenates it with the original query for expanded BM25 retrieval.

Ref: Wang et al. (EMNLP 2023) — "Query2Doc"
"""
import logging
import re

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """Given the following legal case excerpt, generate a brief summary of the key legal issues, relevant statutes, and type of proceeding. Include specific legal terminology that would appear in related cases.

Case excerpt:
{excerpt}

Summary of key legal issues (2-3 sentences):"""


def expand_queries_llm(
    query_texts: dict[str, str],
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_excerpt_words: int = 500,
    batch_size: int = 1,
) -> dict[str, str]:
    """Expand queries using LLM-generated pseudo-documents.

    For each query, extracts a key excerpt, prompts the LLM for a summary
    of legal issues, and appends the expansion to the original query text.

    Args:
        query_texts: {query_id: full_query_text}
        model_name: HuggingFace model for generation
        max_excerpt_words: Max words from query to include in prompt
        batch_size: Generation batch size (1 for sequential)

    Returns:
        {query_id: expanded_query_text}
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    logger.info("Loading LLM for query expansion: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )

    expanded = {}
    for i, (qid, text) in enumerate(query_texts.items()):
        words = text.split()[:max_excerpt_words]
        excerpt = " ".join(words)

        prompt = EXPANSION_PROMPT.format(excerpt=excerpt)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, temperature=0.3,
                do_sample=True, top_p=0.9,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated = generated.strip().split("\n")[0]  # Take first paragraph only

        expanded[qid] = text + " " + generated

        if (i + 1) % 100 == 0:
            logger.info("  Query expansion: %d/%d queries", i + 1, len(query_texts))

    logger.info("Query expansion complete: %d queries expanded", len(expanded))
    return expanded
