"""Compare smart CE vs longctx CE by swapping stage4 cache and re-running meta-learner.

Both stage4.pkl files exist:
  - stage4.pkl (smart CE, current)
  - stage4_longctx.pkl (longctx CE from Node 2)
"""
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("output/pipeline_cache")
CONFIG_PATH = Path("src/graphrag/config.py")
OUTPUT_DIR = Path("output")


def set_config_flag(flag_name: str, value: str) -> str:
    text = CONFIG_PATH.read_text()
    lines = text.split("\n")
    original = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{flag_name} =") or line.strip().startswith(f"{flag_name}="):
            original = line
            lines[i] = f"{flag_name} = {value}"
            break
    if original is None:
        raise ValueError(f"Flag {flag_name} not found in config")
    CONFIG_PATH.write_text("\n".join(lines))
    return original


def restore_config_line(original_line: str, flag_name: str):
    text = CONFIG_PATH.read_text()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{flag_name} =") or line.strip().startswith(f"{flag_name}="):
            lines[i] = original_line
            break
    CONFIG_PATH.write_text("\n".join(lines))


def run_pipeline() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "graphrag.run_pipeline_v2", "train"],
        capture_output=True, text=True, timeout=1200,
    )
    output = result.stdout + result.stderr
    metrics = {}
    for line in output.split("\n"):
        if "CV F1=" in line:
            parts = line.split("CV F1=")[1]
            f1 = float(parts.split()[0])
            p = float(parts.split("P=")[1].split()[0])
            r = float(parts.split("R=")[1].split()[0])
            metrics = {"f1": f1, "precision": p, "recall": r}
        if "Threshold re-optimized" in line:
            t_part = line.split("t=")[1].split(")")[0]
            metrics["threshold"] = float(t_part)
    return metrics


def main():
    smart_ce = CACHE_DIR / "stage4.pkl"
    longctx_ce = CACHE_DIR / "stage4_longctx.pkl"
    backup = CACHE_DIR / "stage4_smart_backup.pkl"

    if not longctx_ce.exists():
        logger.error("stage4_longctx.pkl not found!")
        return

    results = {}

    # Ensure reasoning is off
    orig_reason = set_config_flag("USE_REASONING_RERANKER", "False")

    try:
        # Test 1: Smart CE (current best config: stratified + multiseed3)
        logger.info("=" * 60)
        logger.info("Test 1: Smart CE + stratified + multiseed3")
        logger.info("=" * 60)
        orig_strat = set_config_flag("USE_STRATIFIED_NEGATIVES", "True")
        orig_seed = set_config_flag("MULTI_SEED_RUNS", "3")
        t0 = time.time()
        m = run_pipeline()
        m["time_seconds"] = time.time() - t0
        results["smart_ce_strat_seed3"] = m
        logger.info("  Result: F1=%.4f P=%.4f R=%.4f (%.1f min)",
                    m.get("f1", 0), m.get("precision", 0), m.get("recall", 0),
                    m.get("time_seconds", 0) / 60)
        restore_config_line(orig_strat, "USE_STRATIFIED_NEGATIVES")
        restore_config_line(orig_seed, "MULTI_SEED_RUNS")

        # Swap to longctx CE
        logger.info("Swapping stage4.pkl: smart → longctx")
        shutil.copy2(smart_ce, backup)
        shutil.copy2(longctx_ce, smart_ce)

        # Test 2: Longctx CE baseline
        logger.info("=" * 60)
        logger.info("Test 2: Longctx CE baseline")
        logger.info("=" * 60)
        t0 = time.time()
        m = run_pipeline()
        m["time_seconds"] = time.time() - t0
        results["longctx_ce_baseline"] = m
        logger.info("  Result: F1=%.4f P=%.4f R=%.4f (%.1f min)",
                    m.get("f1", 0), m.get("precision", 0), m.get("recall", 0),
                    m.get("time_seconds", 0) / 60)

        # Test 3: Longctx CE + stratified + multiseed3
        logger.info("=" * 60)
        logger.info("Test 3: Longctx CE + stratified + multiseed3")
        logger.info("=" * 60)
        orig_strat = set_config_flag("USE_STRATIFIED_NEGATIVES", "True")
        orig_seed = set_config_flag("MULTI_SEED_RUNS", "3")
        t0 = time.time()
        m = run_pipeline()
        m["time_seconds"] = time.time() - t0
        results["longctx_ce_strat_seed3"] = m
        logger.info("  Result: F1=%.4f P=%.4f R=%.4f (%.1f min)",
                    m.get("f1", 0), m.get("precision", 0), m.get("recall", 0),
                    m.get("time_seconds", 0) / 60)
        restore_config_line(orig_strat, "USE_STRATIFIED_NEGATIVES")
        restore_config_line(orig_seed, "MULTI_SEED_RUNS")

    finally:
        # Restore smart CE
        if backup.exists():
            logger.info("Restoring smart CE as stage4.pkl")
            shutil.copy2(backup, smart_ce)
            backup.unlink()
        restore_config_line(orig_reason, "USE_REASONING_RERANKER")

    # Save results
    out_path = OUTPUT_DIR / "ce_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\n" + "=" * 60)
    logger.info("CE COMPARISON RESULTS:")
    logger.info("=" * 60)
    for name, m in sorted(results.items(), key=lambda x: -x[1].get("f1", 0)):
        if "error" in m:
            logger.info("  %-30s ERROR: %s", name, m["error"])
        else:
            logger.info("  %-30s F1=%.4f  P=%.4f  R=%.4f  (%.1f min)",
                       name, m["f1"], m["precision"], m["recall"],
                       m.get("time_seconds", 0) / 60)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
