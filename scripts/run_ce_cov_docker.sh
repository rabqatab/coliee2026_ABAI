#!/bin/bash
set -euo pipefail
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NVIDIA_DISABLE_REQUIRE=1 -e HF_HOME=/hf -e PYTHONPATH=/workspace/src \
  -v /home/alphabridge/Research/coliee2026:/workspace \
  -v /home/alphabridge/.sparkq/hf_cache:/hf \
  -w /workspace nvcr.io/nvidia/pytorch:25.09-py3 \
  bash -c "pip install -q 'transformers>=4.44' sentencepiece pandas && python scripts/experiments/ce_coverage.py --device cuda --limit 100"
