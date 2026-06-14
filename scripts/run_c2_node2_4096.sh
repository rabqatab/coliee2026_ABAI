#!/bin/bash
# Option 3: full dense @4096 on NODE 2. Repo is NFS-read-only on node 2, so mount it
# read-only and write output + HF cache to node-2-local writable dirs.
set -euo pipefail
mkdir -p /home/nvidia/c2hf /home/nvidia/c2out
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NVIDIA_DISABLE_REQUIRE=1 -e HF_HOME=/hf -e PYTHONPATH=/workspace/src \
  -v /home/alphabridge/Research/coliee2026:/workspace:ro \
  -v /home/nvidia/c2hf:/hf \
  -v /home/nvidia/c2out:/out \
  -w /workspace nvcr.io/nvidia/pytorch:25.09-py3 \
  bash -c "pip install -q 'transformers>=4.44' && python scripts/w2_c2_dense_recall.py --max-length 4096 --batch-size 8 --out /out/c2_dense_4096.json"
echo "=== result ==="; cat /home/nvidia/c2out/c2_dense_4096.json
