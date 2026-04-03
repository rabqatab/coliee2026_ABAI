"""Run A/B tests on enhancement flags by modifying config and re-running Stage 6 only.

Each test toggles one flag, runs the meta-learner, and logs the result.
All stages 1-5 + 3m are cached, so each run takes ~5-10 min.
"""
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("src/graphrag/config.py")
OUTPUT_DIR = Path("output")


def set_config_flag(flag_name: str, value: str) -> str:
    """Set a config flag and return the original line for restoration."""
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
    """Restore original config line."""
    text = CONFIG_PATH.read_text()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{flag_name} =") or line.strip().startswith(f"{flag_name}="):
            lines[i] = original_line
            break
    CONFIG_PATH.write_text("\n".join(lines))


def run_pipeline() -> dict:
    """Run the pipeline in train mode and extract results."""
    result = subprocess.run(
        [sys.executable, "-m", "graphrag.run_pipeline_v2", "train"],
        capture_output=True, text=True, timeout=1200,
    )
    output = result.stdout + result.stderr

    # Extract CV F1
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
    results = {}

    # Define A/B tests — GNN + BGE-M3 enabled as baseline (both cached)
    tests = [
        ("baseline_gnn_m3", {}),  # Current best: GNN + BGE-M3 + Smart CE
        ("lambdarank", {"USE_LAMBDARANK": "True"}),
        ("convex_fusion", {"USE_CONVEX_FUSION": "True"}),
        ("stratified_neg", {"USE_STRATIFIED_NEGATIVES": "True"}),
        ("multi_seed_3", {"MULTI_SEED_RUNS": "3"}),
        ("no_gnn", {"USE_GNN_RERANKER": "False"}),  # Ablation: remove GNN
        ("no_m3", {"USE_BGE_M3": "False"}),  # Ablation: remove BGE-M3
        ("no_gnn_no_m3", {"USE_GNN_RERANKER": "False", "USE_BGE_M3": "False"}),  # Double ablation
    ]

    # Disable reasoning (not cached yet)
    orig_reason = set_config_flag("USE_REASONING_RERANKER", "False")

    try:
        for test_name, flags in tests:
            logger.info("=" * 60)
            logger.info("Running test: %s (flags: %s)", test_name, flags or "none")
            logger.info("=" * 60)

            # Set flags
            originals = {}
            for flag, value in flags.items():
                originals[flag] = set_config_flag(flag, value)

            t0 = time.time()
            try:
                metrics = run_pipeline()
                elapsed = time.time() - t0
                metrics["time_seconds"] = elapsed
                results[test_name] = metrics
                logger.info("  Result: F1=%.4f P=%.4f R=%.4f (%.1f min)",
                           metrics.get("f1", 0), metrics.get("precision", 0),
                           metrics.get("recall", 0), elapsed / 60)
            except Exception as e:
                logger.error("  FAILED: %s", e)
                results[test_name] = {"error": str(e)}

            # Restore flags
            for flag, orig in originals.items():
                restore_config_line(orig, flag)

    finally:
        # Always restore reasoning flag
        restore_config_line(orig_reason, "USE_REASONING_RERANKER")

    # Save results
    out_path = OUTPUT_DIR / "ab_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\n" + "=" * 60)
    logger.info("ALL RESULTS:")
    logger.info("=" * 60)
    for name, m in sorted(results.items(), key=lambda x: -x[1].get("f1", 0)):
        if "error" in m:
            logger.info("  %-25s ERROR: %s", name, m["error"])
        else:
            logger.info("  %-25s F1=%.4f  P=%.4f  R=%.4f  t=%.3f  (%.1f min)",
                       name, m["f1"], m["precision"], m["recall"],
                       m.get("threshold", 0), m.get("time_seconds", 0) / 60)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
