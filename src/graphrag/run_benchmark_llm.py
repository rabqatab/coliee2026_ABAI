"""Benchmark LLM models for entity extraction quality and speed."""
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

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
]

N_SAMPLE_DOCS = 50


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

    return {
        "model": model,
        "n_docs": len(docs),
        "valid_json_rate": valid_json_count / len(docs),
        "mean_time_seconds": float(np.mean(times)),
        "p95_time_seconds": float(np.percentile(times, 95)) if times else 0,
        "mean_concepts": float(np.mean(entity_counts["concepts"])),
        "mean_tests": float(np.mean(entity_counts["tests"])),
        "mean_statutes": float(np.mean(entity_counts["statutes"])),
        "mean_holdings": float(np.mean(entity_counts["holdings"])),
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
