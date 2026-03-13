# Node 2 (Worker) Access Guide

> **Last updated:** 2026-03-13
> **Purpose:** How to access and operate on the second DGX Spark GPU node for COLIEE 2026 training.

---

## Cluster Overview

| | Node 1 (Master) | Node 2 (Worker) |
|---|---|---|
| **Hostname** | spark-0 | gx10-3d56 |
| **IP** | 192.168.200.12 | 192.168.200.13 |
| **GPU** | NVIDIA GB10 (128 GB unified) | NVIDIA GB10 (128 GB unified) |
| **User** | alphabridge (primary), nvidia | nvidia |
| **Interconnect** | 200GbE QSFP (ConnectX-7) | 200GbE QSFP (ConnectX-7) |

---

## SSH Access

From Node 1 (this machine):

```bash
# Interactive shell
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13

# Run a single command
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 "command here"
```

No password needed — key-based auth is configured.

---

## Docker Container

The COLIEE training container on Node 2:

```bash
# Container name: coliee_longctx
# Image: nvcr.io/nvidia/pytorch:25.09-py3
# Bind mount: /home/nvidia/coliee2026 → /workspace/coliee2026

# Check if running
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 "docker ps --filter name=coliee_longctx"

# Execute a command inside the container
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "docker exec -e PYTHONPATH=/workspace/coliee2026/src coliee_longctx COMMAND"

# Shell into the container
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 "docker exec -it coliee_longctx bash"

# View training log
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "tail -20 /home/nvidia/coliee2026/output/training_longctx.log"
```

### Creating the Container (if needed)

```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 "docker run -d --name coliee_longctx \
  --runtime=nvidia --gpus all \
  -v /home/nvidia/coliee2026:/workspace/coliee2026 \
  nvcr.io/nvidia/pytorch:25.09-py3 \
  sleep infinity"
```

### Installing Dependencies (if fresh container)

```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "docker exec coliee_longctx pip install \
    transformers sentence-transformers tiktoken sentencepiece \
    protobuf lightgbm rank_bm25 scikit-learn leidenalg igraph"
```

---

## File Layout on Node 2

```
/home/nvidia/coliee2026/
├── src/graphrag/              # Pipeline source code (rsynced from Node 1)
├── scripts/                   # Entry point scripts
├── data/task1/                # Full corpus (9,556 docs)
│   ├── task1_train_files_2026/   # 7,708 train docs
│   ├── task1_test_files_2026/    # 1,848 test docs
│   ├── task1_train_labels_2026.json
│   └── task1_test_no_labels_2026.json
├── output/
│   ├── pipeline_cache/        # Stages 1-3 cached (rsynced from Node 1)
│   │   ├── stage1.pkl         # 907 MB — preprocessing + contexts
│   │   ├── stage2.pkl         # 21 MB — BM25 results
│   │   └── stage3.pkl         # 5.4 MB — bi-encoder scores
│   ├── models_v2/crossencoder/  # Cross-encoder model output
│   └── training_longctx.log   # Current training log
├── pyproject.toml
└── uv.lock
```

**Important:** Node 2's codebase is an rsync copy, NOT a live mount. After editing code on Node 1, you must re-sync:

```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  /home/alphabridge/Research/coliee2026/src/ \
  nvidia@192.168.200.13:/home/nvidia/coliee2026/src/
```

---

## Syncing Data Between Nodes

### Code changes (fast, <1 sec)
```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  /home/alphabridge/Research/coliee2026/src/ \
  nvidia@192.168.200.13:/home/nvidia/coliee2026/src/
```

### Pipeline cache (slow, ~930 MB)
```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  /home/alphabridge/Research/coliee2026/output/pipeline_cache/ \
  nvidia@192.168.200.13:/home/nvidia/coliee2026/output/pipeline_cache/
```

### Fetch results back from Node 2
```bash
# Copy trained model back to Node 1
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  nvidia@192.168.200.13:/home/nvidia/coliee2026/output/models_v2/crossencoder/ \
  /home/alphabridge/Research/coliee2026/output/models_v2/crossencoder_longctx/

# Copy training log
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  nvidia@192.168.200.13:/home/nvidia/coliee2026/output/training_longctx.log \
  /home/alphabridge/Research/coliee2026/output/
```

---

## Monitoring

### Check training progress
```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "tail -10 /home/nvidia/coliee2026/output/training_longctx.log"
```

### Check GPU utilization
```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 "nvidia-smi"
```

### Check process status
```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "docker exec coliee_longctx pgrep -af python"
```

### Check memory
```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 "free -h"
```

---

## Switching Cross-Encoder Mode

Node 2's `config.py` is independent from Node 1. To change the mode:

```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "sed -i 's/CROSSENCODER_MODE = \".*\"/CROSSENCODER_MODE = \"NEW_MODE\"/' \
   /home/nvidia/coliee2026/src/graphrag/config.py"
```

Valid modes: `"smart"`, `"longctx"`, `"passage"`

---

## Current Training (as of 2026-03-13)

| Setting | Value |
|---------|-------|
| **Mode** | `longctx` |
| **Model** | `BAAI/bge-reranker-v2-m3` |
| **Max tokens** | 4,096 |
| **Batch size** | 4 |
| **Steps/epoch** | 10,314 |
| **Epochs** | 3 |
| **Log** | `/home/nvidia/coliee2026/output/training_longctx.log` |
| **Started** | ~02:27 UTC |

---

## Troubleshooting

### Container not running
```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 "docker start coliee_longctx"
```

### Process died silently
Check the log for errors, then restart:
```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "docker exec -d -e PYTHONPATH=/workspace/coliee2026/src coliee_longctx \
   bash -c 'python /workspace/coliee2026/scripts/train_pipeline.py \
   > /workspace/coliee2026/output/training_longctx.log 2>&1'"
```

### Out of memory
The BGE-reranker-v2-m3 with batch_size=4 and max_length=4096 uses ~50 GB GPU memory. If OOM occurs, reduce batch size in Node 2's `config.py`:
```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "sed -i 's/CROSSENCODER_LONG_BATCH_SIZE = 4/CROSSENCODER_LONG_BATCH_SIZE = 2/' \
   /home/nvidia/coliee2026/src/graphrag/config.py"
```

### Need to kill training
```bash
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "docker exec coliee_longctx pkill -f train_pipeline"
```
