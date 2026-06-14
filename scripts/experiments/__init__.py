"""Reproducible COLIEE 2026 journal experiments.

Each module is a self-contained experiment that imports its shared building
blocks from `scripts.experiments.common` and writes a JSON result to
`output/experiments/<name>.json` via `common.write_result`.

See `scripts/experiments/README.md` for the experiment index, reproduce
commands, and headline results.
"""
