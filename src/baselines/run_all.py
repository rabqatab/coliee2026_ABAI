"""Run all baseline models and produce comparison table.

Usage: PYTHONPATH=src uv run python -m baselines.run_all
"""
import logging

from baselines.common.data_loader import load_dataset
from baselines.common.bm25_index import build_shared_bm25
from baselines.common.run_harness import run_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("baselines")


def get_all_baselines():
    """Import and instantiate all available baselines."""
    models = []

    from baselines.bm25.model import VanillaBM25
    models.append(VanillaBM25())

    try:
        from baselines.jnlp.model import JNLPBaseline
        models.append(JNLPBaseline())
    except ImportError:
        logger.warning("JNLP baseline not available")

    try:
        from baselines.tqm.model import TQMBaseline
        models.append(TQMBaseline())
    except ImportError:
        logger.warning("TQM baseline not available")

    try:
        from baselines.umnlp.model import UMNLPBaseline
        models.append(UMNLPBaseline())
    except ImportError:
        logger.warning("UMNLP baseline not available")

    try:
        from baselines.caselink.model import CaseLinkBaseline
        models.append(CaseLinkBaseline())
    except ImportError:
        logger.warning("CaseLink baseline not available")

    try:
        from baselines.graphrag_adapter.model import GraphRAGAdapter
        models.append(GraphRAGAdapter())
    except ImportError:
        logger.warning("GraphRAG adapter not available")

    return models


def main():
    dataset = load_dataset()
    _, bm25_candidates = build_shared_bm25(dataset.corpus)

    models = get_all_baselines()
    logger.info("Running %d baselines", len(models))

    run_comparison(models, dataset, bm25_candidates)


if __name__ == "__main__":
    main()
