"""Ablation study for performance push phases.

Tests each improvement independently and in combination to measure
the contribution of each phase to overall F1.

Usage:
    uv run python scripts/run_phase_ablation.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coliee_task1 import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CONFIGS = {
    "baseline": {
        "INJECT_GOLD_IN_POOL": True,
        "USE_TEMPORAL_SPLIT": False,
        "USE_DENSE_FIRST_STAGE": False,
        "CROSSENCODER_TOP_K": 50,
        "THRESHOLD_METHOD": "global",
    },
    "phase1_no_gold_injection": {
        "INJECT_GOLD_IN_POOL": False,
        "USE_TEMPORAL_SPLIT": False,
        "USE_DENSE_FIRST_STAGE": False,
        "CROSSENCODER_TOP_K": 50,
        "THRESHOLD_METHOD": "global",
    },
    "phase1_temporal_split": {
        "INJECT_GOLD_IN_POOL": False,
        "USE_TEMPORAL_SPLIT": True,
        "USE_DENSE_FIRST_STAGE": False,
        "CROSSENCODER_TOP_K": 50,
        "THRESHOLD_METHOD": "global",
    },
    "phase2_hybrid_retrieval": {
        "INJECT_GOLD_IN_POOL": False,
        "USE_TEMPORAL_SPLIT": True,
        "USE_DENSE_FIRST_STAGE": True,
        "CROSSENCODER_TOP_K": 50,
        "THRESHOLD_METHOD": "global",
    },
    "phase3_expanded_ce": {
        "INJECT_GOLD_IN_POOL": False,
        "USE_TEMPORAL_SPLIT": True,
        "USE_DENSE_FIRST_STAGE": True,
        "CROSSENCODER_TOP_K": 100,
        "THRESHOLD_METHOD": "global",
    },
    "phase5_evt_threshold": {
        "INJECT_GOLD_IN_POOL": False,
        "USE_TEMPORAL_SPLIT": True,
        "USE_DENSE_FIRST_STAGE": True,
        "CROSSENCODER_TOP_K": 100,
        "THRESHOLD_METHOD": "evt",
    },
    "full_pipeline": {
        "INJECT_GOLD_IN_POOL": False,
        "USE_TEMPORAL_SPLIT": True,
        "USE_DENSE_FIRST_STAGE": True,
        "CROSSENCODER_TOP_K": 100,
        "THRESHOLD_METHOD": "evt",
    },
}


def run_config(name: str, params: dict) -> dict:
    """Run pipeline with specific config and return metrics."""
    logger.info("=" * 60)
    logger.info("Running config: %s", name)
    logger.info("=" * 60)

    # Apply config overrides
    for key, value in params.items():
        setattr(config, key, value)

    from coliee_task1.pipeline import run_train_pipeline
    metrics = run_train_pipeline(use_cache=False)

    logger.info("Config %s: F1=%.4f, P=%.4f, R=%.4f",
                name, metrics.get("f1", 0), metrics.get("precision", 0), metrics.get("recall", 0))
    return {"config": name, "params": params, **metrics}


def main():
    results = []
    for name, params in CONFIGS.items():
        try:
            result = run_config(name, params)
            results.append(result)
        except Exception as e:
            logger.error("Config %s FAILED: %s", name, e)
            results.append({"config": name, "error": str(e)})

    # Save results
    output_path = Path("output/ablation/phase_ablation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Config':<30} {'F1':>8} {'Prec':>8} {'Recall':>8} {'dF1':>8}")
    print("-" * 70)
    baseline_f1 = results[0].get("f1", 0) if results else 0
    for r in results:
        f1 = r.get("f1", 0)
        p = r.get("precision", 0)
        rec = r.get("recall", 0)
        delta = f1 - baseline_f1
        print(f"{r['config']:<30} {f1:>8.4f} {p:>8.4f} {rec:>8.4f} {delta:>+8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
