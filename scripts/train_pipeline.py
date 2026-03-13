"""Run Option C training pipeline inside GPU Docker container.

Thin wrapper around run_pipeline_v2.run_train_pipeline() with Docker-specific
logging setup. All enhancement flags are read from graphrag.config.

Usage (from host):
    docker exec -e PYTHONPATH=/workspace/coliee2026/src coliee_optionc \
        python /workspace/coliee2026/scripts/train_pipeline.py [--no-cache]
"""
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/coliee2026/output/training.log"),
    ],
)
logger = logging.getLogger("pipeline")

import torch

logger.info("=" * 60)
logger.info("  Option C Pipeline — GPU Training")
logger.info("  GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Option C GPU Training")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable stage caching (recompute everything from scratch).",
    )
    args = parser.parse_args()

    from graphrag.run_pipeline_v2 import run_train_pipeline

    t0 = time.time()
    metrics = run_train_pipeline(
        finetune=True,
        use_cache=not args.no_cache,
    )

    logger.info("=" * 60)
    logger.info("  DONE in %.1f minutes", (time.time() - t0) / 60)
    logger.info("  F1=%.4f  P=%.4f  R=%.4f",
                metrics["f1"], metrics["precision"], metrics["recall"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
