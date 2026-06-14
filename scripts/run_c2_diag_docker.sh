#!/bin/bash
# C2 diagnostic (e5-mistral-7b, bf16, @512) on GB10 via nvcr pytorch container.
set -euo pipefail
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NVIDIA_DISABLE_REQUIRE=1 -e HF_HOME=/hf -e PYTHONPATH=/workspace/src \
  -v /home/alphabridge/Research/coliee2026:/workspace \
  -v /home/alphabridge/.sparkq/hf_cache:/hf \
  -w /workspace nvcr.io/nvidia/pytorch:25.09-py3 \
  bash -c "pip install -q 'transformers>=4.44' && python scripts/w2_c2_diag.py"
