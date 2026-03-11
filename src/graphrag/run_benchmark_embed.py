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
    "nomic-embed-text",
    "bge-m3",
    "qwen3-embedding:0.6b",
    # "qwen3-embedding:8b",  # Too slow on GB10 (~130s/batch, impractical for 7.7K docs)
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
    k_values: list[int] | None = None,
) -> dict:
    """Compute Recall@K and MRR from a similarity matrix."""
    if k_values is None:
        k_values = [50, 100, 200]

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
        **{f"recall@{k}": float(np.mean(v)) for k, v in recalls.items()},
        "mrr": float(np.mean(mrrs)),
        "n_queries": len(mrrs),
    }


N_SAMPLE_QUERIES = 200  # Subsample queries for faster benchmarking
CORPUS_POOL_MULTIPLIER = 5  # Include top-N * multiplier candidate docs per query


def _subsample_corpus(
    corpus: dict[str, str],
    labels: dict[str, list[str]],
    n_queries: int = N_SAMPLE_QUERIES,
    seed: int = 42,
) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    """Subsample the corpus for faster benchmarking.

    Selects n_queries and their cited docs, plus a random negative pool.
    This gives valid Recall@K metrics without embedding the full 7,708 docs.
    """
    import random
    rng = random.Random(seed)

    # Sample queries
    all_query_ids = sorted(labels.keys())
    sampled_queries = rng.sample(all_query_ids, min(n_queries, len(all_query_ids)))

    # Collect all cited docs + the queries themselves
    needed_ids = set(sampled_queries)
    for qid in sampled_queries:
        needed_ids.update(labels[qid])

    # Add random negatives to make retrieval challenging
    all_ids = set(corpus.keys())
    remaining = list(all_ids - needed_ids)
    n_negatives = min(len(remaining), n_queries * CORPUS_POOL_MULTIPLIER)
    needed_ids.update(rng.sample(remaining, n_negatives))

    sub_corpus = {k: corpus[k] for k in sorted(needed_ids) if k in corpus}
    sub_labels = {q: labels[q] for q in sampled_queries}

    return sub_corpus, sub_labels, sorted(sub_corpus.keys())


def main():
    output_dir = BENCHMARK_DIR / "embed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus and labels
    logger.info("Loading corpus...")
    corpus = load_corpus(TRAIN_DOCS_DIR)
    labels = json.loads(TRAIN_LABELS.read_text())

    # Subsample for speed
    corpus, labels, corpus_ids = _subsample_corpus(corpus, labels)

    # Preprocess all documents
    corpus_texts = [truncate_for_embedding(preprocess(corpus[cid])) for cid in corpus_ids]
    logger.info("Corpus: %d documents (subsampled)", len(corpus_ids))

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
                metrics["embed_dim"] = int(embeddings.shape[1])
                metrics["memory_mb"] = float(embeddings.nbytes / 1e6)

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
