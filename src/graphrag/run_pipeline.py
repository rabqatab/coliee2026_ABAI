"""End-to-end retrieval pipeline with ablation support."""
import json
import logging
import time

import numpy as np

from graphrag.config import (
    TRAIN_DOCS_DIR,
    TRAIN_LABELS,
    EXTRACTIONS_DIR,
    GRAPH_DIR,
    EMBEDDINGS_DIR,
    OUTPUT_DIR,
    BM25_TOP_K,
)
from graphrag.ollama_client import OllamaClient
from graphrag.preprocess import preprocess, load_corpus
from graphrag.graph import load_extractions
from graphrag.bm25 import BM25Index
from graphrag.embed import embed_texts
from graphrag.retrieve import (
    signal_entity_graph,
    signal_community,
    signal_embedding,
    reciprocal_rank_fusion,
)
from graphrag.metrics import micro_f1, optimize_threshold

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
