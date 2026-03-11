"""Full indexing pipeline: extraction -> graph -> communities -> embeddings."""
import json
import logging
import time
from pathlib import Path

import numpy as np

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
