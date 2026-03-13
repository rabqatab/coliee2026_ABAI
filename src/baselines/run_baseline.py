"""Run a single baseline model by name.

Usage: PYTHONPATH=src uv run python -m baselines.run_baseline bm25
"""
import logging
import sys

from baselines.common.data_loader import load_dataset
from baselines.common.bm25_index import build_shared_bm25
from baselines.common.run_harness import assess_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

BASELINE_REGISTRY: dict[str, type] = {}


def register(name, cls):
    BASELINE_REGISTRY[name] = cls


def _register_all():
    from baselines.bm25.model import VanillaBM25
    register("bm25", VanillaBM25)

    try:
        from baselines.jnlp.model import JNLPBaseline
        register("jnlp", JNLPBaseline)
    except ImportError:
        pass

    try:
        from baselines.tqm.model import TQMBaseline
        register("tqm", TQMBaseline)
    except ImportError:
        pass

    try:
        from baselines.umnlp.model import UMNLPBaseline
        register("umnlp", UMNLPBaseline)
    except ImportError:
        pass

    try:
        from baselines.caselink.model import CaseLinkBaseline
        register("caselink", CaseLinkBaseline)
    except ImportError:
        pass


def main():
    _register_all()

    if len(sys.argv) < 2 or sys.argv[1] not in BASELINE_REGISTRY:
        print(f"Usage: python -m baselines.run_baseline <name>")
        print(f"Available: {', '.join(sorted(BASELINE_REGISTRY.keys()))}")
        sys.exit(1)

    name = sys.argv[1]
    model = BASELINE_REGISTRY[name]()

    dataset = load_dataset()
    _, bm25_candidates = build_shared_bm25(dataset.corpus)
    result = assess_baseline(model, dataset, bm25_candidates)
    print(f"\n{name}: val F1 = {result['val_f1']:.4f}")


if __name__ == "__main__":
    main()
