# COLIEE 2026 — Training Status & Model Inventory

> **Last updated:** 2026-03-13 ~04:00 UTC
> **For:** Claude Code session handoff

---

## Code Versioning

### Git State

**Last commit:** `740a6aa` (2026-03-11 19:38 KST) — original GraphRAG pipeline only.

**ALL Option C pipeline code is UNCOMMITTED.** The following critical files were created after the last commit and exist only on disk:

| File | Status | Last Modified (UTC) |
|------|--------|---------------------|
| `src/graphrag/finetune_crossencoder.py` | **NEW** (untracked) | 02:08 Mar 13 |
| `src/graphrag/run_pipeline_v2.py` | **NEW** (untracked) | 02:09 Mar 13 |
| `src/graphrag/meta_learner.py` | **NEW** (untracked) | 06:53 Mar 12 |
| `src/graphrag/citation_context.py` | **NEW** (untracked) | 10:28 Mar 11 |
| `src/graphrag/graphrag_lite.py` | **NEW** (untracked) | 06:51 Mar 12 |
| `src/graphrag/finetune_biencoder.py` | **NEW** (untracked) | 10:30 Mar 11 |
| `scripts/train_pipeline.py` | **NEW** (untracked) | 15:17 Mar 12 |
| `src/graphrag/config.py` | **MODIFIED** (tracked) | 02:08 Mar 13 |
| `src/graphrag/bm25.py` | **MODIFIED** (tracked) | 06:51 Mar 12 |
| `src/graphrag/metrics.py` | **MODIFIED** (tracked) | 07:26 Mar 12 |

### Code Loaded by Each Running Process

Python caches module imports at startup. The Docker bind mount shows the host filesystem, but running processes use the version loaded at start time.

**Verification:** No `.py` files were modified after either process started. Both processes are running the exact current on-disk code.

#### Node 1 "smart" (PID 2831, started 02:12:41 UTC)

| File | MD5 | Matches disk? |
|------|-----|---------------|
| `finetune_crossencoder.py` | `da92a7d2` | Yes (last write 02:08, import 02:09) |
| `run_pipeline_v2.py` | `0e46e880` | Yes (last write 02:09, import 02:12) |
| `config.py` | `bfc91b2e` | Yes (last write 02:08, import 02:09) |
| `meta_learner.py` | `00620efa` | Yes (last write 06:53 Mar 12) |
| `bm25.py` | `8a182f28` | Yes (last write 06:51 Mar 12) |

**Config:** `CROSSENCODER_MODE = "smart"`, `CROSSENCODER_MODEL = "microsoft/deberta-v3-large"`, `CROSSENCODER_MAX_LENGTH = 512`, `CROSSENCODER_LR = 1e-5`

#### Node 2 "longctx" (PID 238, started 02:25:49 UTC)

| File | MD5 | Matches Node 1? |
|------|-----|-----------------|
| `finetune_crossencoder.py` | `da92a7d2` | **Identical** |
| `run_pipeline_v2.py` | `0e46e880` | **Identical** |
| `config.py` | `2d4eff88` | **Differs** (sed changed mode) |
| `meta_learner.py` | `00620efa` | **Identical** |
| `bm25.py` | `8a182f28` | **Identical** |

**Config:** `CROSSENCODER_MODE = "longctx"`, `CROSSENCODER_LONG_MODEL = "BAAI/bge-reranker-v2-m3"`, `CROSSENCODER_LONG_MAX_LENGTH = 4096`, `CROSSENCODER_LONG_LR = 2e-5`

**Only difference between nodes:** One line in `config.py` — `CROSSENCODER_MODE = "smart"` vs `"longctx"`. All other files byte-identical (confirmed by md5sum).

### Pipeline Cache Provenance

All 3 cache files are **byte-identical** between Node 1 and Node 2 (confirmed by md5sum).

| Cache | Created (UTC) | Created by | Contents |
|-------|---------------|------------|----------|
| `stage1.pkl` (907 MB) | 00:20 Mar 13 | GPU Run 2 | 9,556 docs (train+test), clean corpus, citation contexts |
| `stage2.pkl` (21 MB) | 00:28 Mar 13 | GPU Run 2 | BM25 RRF results for 2,001 queries |
| `stage3.pkl` (5.4 MB) | 01:21 Mar 13 | GPU Run 3 | Bi-encoder scores from **fine-tuned** model (LoRA adapter) |

**stage3 provenance:** Created during GPU Run 3, which loaded the saved fine-tuned bi-encoder from `output/models/biencoder/final/` (trained 2026-03-11/12, ~19h). The scores are from the LoRA-adapted model, NOT the base BGE model.

---

## Currently Running Training Jobs

### [ACTIVE] Node 1: Cross-Encoder "smart" (DeBERTa-v3-large)

| Field | Value |
|-------|-------|
| **Node** | Node 1 (192.168.200.12), container `coliee_optionc` |
| **PID** | 2831 (main), 2886/2887 (DataLoader workers) |
| **Mode** | `smart` — smart truncation with citation context windows |
| **Model** | `microsoft/deberta-v3-large` (304M params, **forced fp32**) |
| **Input strategy** | head 100w + citation context windows + tail 100w = ~500w, tokenized to max 512 tokens |
| **Training data** | 8,251 pos + 33,004 neg pairs (1:4 ratio, BM25 hard negatives) |
| **Epochs** | 3, 2,579 steps/epoch (total 7,737 steps) |
| **Batch size** | 16 |
| **LR** | 1e-5, linear warmup 10%, weight decay 0.01 |
| **Loss** | CrossEntropyLoss (2-class: not relevant / relevant) |
| **Started** | 02:12:41 UTC, Mar 13 |
| **Progress** | Epoch 1, step 1800/2579, loss=0.421, acc=81.9% |
| **Speed** | ~5.5 min per 100 steps |
| **Est. per epoch** | ~2h 22m |
| **Est. completion** | ~09:30 UTC Mar 13 (3 epochs + stages 5-6) |
| **GPU usage** | 47,813 MiB, 96% utilization |
| **Log** | `output/training_smart.log` |
| **Output model** | `output/models_v2/crossencoder/final/` |
| **Key fix** | `model.float()` prevents DeBERTa-v3 XSoftmax fp16 NaN overflow |
| **NaN guard** | Skips batches with NaN/Inf loss (0 so far) |

```bash
# Monitor
tail -5 output/training_smart.log
grep "Epoch.*step" output/training_smart.log | tail -5
docker exec coliee_optionc nvidia-smi
```

### [ACTIVE] Node 2: Cross-Encoder "longctx" (BGE-reranker-v2-m3)

| Field | Value |
|-------|-------|
| **Node** | Node 2 (192.168.200.13), container `coliee_longctx` |
| **PID** | 238 (main), 320/321 (DataLoader workers) |
| **Mode** | `longctx` — long-context reranker, up to 4096 tokens |
| **Model** | `BAAI/bge-reranker-v2-m3` (568M params, XLM-RoBERTa-large) |
| **Input strategy** | First 3000 words, tokenized to max 4096 tokens |
| **Training data** | 8,251 pos + 33,004 neg pairs (1:4 ratio, BM25 hard negatives) |
| **Epochs** | 3, 10,314 steps/epoch (total 30,942 steps) |
| **Batch size** | 4 (limited by 4096-token sequence length) |
| **LR** | 2e-5, linear warmup 10%, weight decay 0.01 |
| **Loss** | BCEWithLogitsLoss (single-logit output, sigmoid for inference) |
| **Started** | 02:25:49 UTC, Mar 13 |
| **Progress** | Epoch 1, step 1200/10314, loss=0.600, acc=79.9% |
| **Speed** | ~7.5 min per 100 steps |
| **Est. per epoch** | ~12h 54m |
| **Est. completion** | ~16:30 UTC Mar 14 (~38h total) |
| **Log** | Node 2: `/home/nvidia/coliee2026/output/training_longctx.log` |
| **Output model** | Node 2: `output/models_v2/crossencoder/final/` |

```bash
# Monitor
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 \
  "tail -5 /home/nvidia/coliee2026/output/training_longctx.log"
ssh -i ~/.ssh/id_ed25519 nvidia@192.168.200.13 "nvidia-smi"
```

### [ACTIVE] Node 1: Baselines (`python -m baselines.run_all`)

| Field | Value |
|-------|-------|
| **PID** | 2006 (inside `coliee_optionc`) |
| **Started** | 00:14 UTC, Mar 13 |
| **Resource usage** | ~94% of 1 CPU core, 925 MiB GPU, 6.3 GB RAM |
| **Purpose** | Prior-winner reproduction baselines (JNLP, TQM, UMNLP, CaseLink) |
| **Output** | `output/baselines/` |
| **Launched by** | Previous Claude Code session |
| **Code** | `src/baselines/` (all untracked, created by other session) |

Not significantly competing with cross-encoder training (GPU saturated by our job at 96%).

---

## Queued Training Job

### [QUEUED] Node 1: Cross-Encoder "passage" (DeBERTa-v3-large)

Will run on Node 1 after the "smart" approach completes.

| Field | Value |
|-------|-------|
| **Mode** | `passage` — passage-level scoring with max-pooling |
| **Model** | `microsoft/deberta-v3-large` (same as smart, fp32) |
| **Input strategy** | Chunk docs into 200w overlapping passages (50w overlap), select top-5 by word overlap with query, score all passage pairs, aggregate via max-pool |
| **Config change needed** | Set `CROSSENCODER_MODE = "passage"` in config.py, delete `output/pipeline_cache/stage4.pkl` |
| **Training** | Same pairs/epochs as smart — trained identically, only inference differs |
| **Est. start** | After smart completes (~09:30 UTC Mar 13) |
| **Training time** | ~7h (same as smart), inference slower (5x passage pairs per candidate) |

---

## Completed Models

### Bi-Encoder (BGE-large-en-v1.5 + LoRA)

| Field | Value |
|-------|-------|
| **Status** | COMPLETED and cached — stage3.pkl contains its scores |
| **Base model** | `BAAI/bge-large-en-v1.5` (335M params, 1024-dim embeddings) |
| **Fine-tuning** | LoRA (PEFT v0.18.1) |
| **LoRA config** | rank=16, alpha=32, dropout=0.1, bias=none |
| **LoRA targets** | query, key, value, dense |
| **Task type** | FEATURE_EXTRACTION |
| **Trainable params** | 7,110,656 / 342,252,544 total (2.08%) |
| **Training** | InfoNCE loss, 57,757 triplets, 3 epochs, 10,830 steps |
| **Training time** | ~19 hours on GB10 GPU (2026-03-11 13:18 – 2026-03-12 ~08:00 UTC) |
| **Trained by** | GPU Run 1 (old `scripts/train_pipeline.py`, NOT `run_pipeline_v2.py`) |
| **Saved to** | `output/models/biencoder/final/` (28 MB adapter + tokenizer) |
| **Symlink** | `output/models_v2/biencoder/` → `../models/biencoder` (relative, Docker-compatible) |
| **Cached scores** | `output/pipeline_cache/stage3.pkl` (5.4 MB, created 01:21 UTC Mar 13) |

### Meta-Learner (LightGBM) — STALE, will be overwritten

| Field | Value |
|-------|-------|
| **Status** | STALE — from CPU no-finetune run, will be overwritten by current GPU runs |
| **Result** | CV F1=0.5924, P=0.8530, R=0.4538 (threshold=0.690) |
| **Features** | 22 (6 retrieval + 5 lexical + 9 GraphRAG + 2 citation context) |
| **Missing** | No cross-encoder features (stage 4 was skipped in CPU no-finetune mode) |
| **Bi-encoder scores** | From BASE model (not fine-tuned) — used `output/pipeline_cache_cpu_backup/` |
| **Saved to** | `output/models_v2/meta_learner/` |
| **Note** | Current best score. GPU runs will retrain with fine-tuned bi-encoder + cross-encoder features. |

---

## Baselines & Results Summary

### Our Simple Baselines (5-fold CV, train-only 7,708-doc corpus)

| Method | CV F1 | Precision | Recall | Notes |
|--------|-------|-----------|--------|-------|
| BM25-only | 0.0230 | 0.0117 | 0.5685 | High recall, no precision |
| TF-IDF Cosine | 0.0842 | 0.0728 | 0.0999 | Similarity threshold |
| **BM25 + Lexical → LightGBM** | **0.3489** | 0.3007 | 0.4312 | 6 features, reference baseline |

### Prior Winner Reproductions (80/20 split, 9,556-doc corpus, NOT comparable to CV)

| System | Val F1 | Train F1 | Status |
|--------|--------|----------|--------|
| BM25 (vanilla) | 0.0234 | 0.0214 | Baseline |
| JNLP 2025 (BM25+SAILER+LightGBM) | 0.1212 | 0.3889 | Massive overfit |
| TQM 2024 (LTR Fusion) | 0.0993 | 0.3172 | Massive overfit |
| UMNLP 2024 (Propositions+NN) | 0.0000 | 0.0000 | Failed completely |
| CaseLink 2025 (GNN) | 0.0307 | 0.0353 | Underfit |

> These used an 80/20 split with the 2026 dataset. All prior winners degraded severely, suggesting 2026 is much harder.

### Competition Benchmarks (on their respective year's test set)

| Year | Rank 1 | F1 | Method |
|------|--------|-----|--------|
| 2024 | TQM | 0.4432 | LTR fusion (BM25 + neural) |
| 2025 | JNLP | 0.3353 | BM25 + SAILER + LightGBM |

### Our Option C Pipeline Results (5-fold CV, 9,556-doc corpus)

| Run | Date | Bi-encoder | Cross-encoder | CV F1 | Status |
|-----|------|------------|---------------|-------|--------|
| Option C run 2 | 03-12 | Base (no LoRA) | None | 0.0912 | Deprecated |
| Option C lexical fix | 03-12 | Base | None | 0.2484 | Deprecated |
| Option C v2 | 03-12 | Base | None | 0.4475 | Deprecated |
| **Option C v4 (CPU) ★** | 03-12 | **Base** | **None** | **0.5924** | Current best |
| **Option C GPU "smart"** | 03-13 | **LoRA fine-tuned** | **DeBERTa-v3 (smart)** | **TBD** | Training |
| **Option C GPU "longctx"** | 03-13 | **LoRA fine-tuned** | **BGE-reranker (longctx)** | **TBD** | Training |
| Option C GPU "passage" | — | LoRA fine-tuned | DeBERTa-v3 (passage) | **TBD** | Queued |

**Key difference:** CPU runs used the BASE bi-encoder (no LoRA). GPU runs use the FINE-TUNED bi-encoder (cached in stage3.pkl). The cross-encoder is entirely new — no CPU run had one.

---

## Deprecated / Outdated Items

### [DEPRECATED] GPU Run 1 Cross-Encoder (2026-03-11 to 2026-03-12)
- DeBERTa-v3-large training that died silently at ~08:14 UTC
- No model was saved, empty `output/models/crossencoder/` dir
- Root cause: Unknown (no traceback, no OOM evidence)
- Replaced by the 3-approach training launched 2026-03-13

### [DEPRECATED] GPU Run 2 (2026-03-13 00:20–00:30)
- Killed because it was retraining bi-encoder from scratch despite saved model
- Fix: Reordered `stage3_biencoder()` to check `model_path.exists()` first
- **Created stage1.pkl and stage2.pkl caches** (still in use)

### [DEPRECATED] GPU Run 3 (2026-03-13 01:16–01:22)
- Crashed at stage 4: `ValueError: tiktoken is required`, then `sentencepiece` missing
- Fix: `pip install tiktoken sentencepiece protobuf` in Docker
- **Created stage3.pkl cache** (still in use)

### [DEPRECATED] GPU Run 4 — NaN run (2026-03-13 01:28–02:08)
- DeBERTa-v3-large cross-encoder training produced `loss=nan` from step ~100 onwards
- Root cause: XSoftmax in disentangled attention overflows fp16
- Fix: `model.float()` to force fp32, plus NaN guard to skip bad batches
- Killed and replaced by current "smart" run
- Log: `output/training_gpu.log` (still on disk, shows NaN pattern)

### [STALE] `output/models_v2/meta_learner/`
- LightGBM models from CPU no-finetune run (CV F1=0.5924)
- Used BASE bi-encoder scores, no cross-encoder features
- Will be overwritten by stage 6 after each cross-encoder training completes

### [STALE] `output/pipeline_cache_cpu_backup/`
- Old CPU caches with only 7,708 train docs (not full 9,556 corpus)
- Bi-encoder scores from BASE model (not fine-tuned)
- Safe to delete

### [STALE] `output/training.log`
- From GPU Run 1 (2026-03-11/12). Contains bi-encoder training completion + cross-encoder crash.

### [STALE] `output/training_gpu.log`
- From GPU Run 4 (NaN run). Shows loss=nan from step 100 onwards.

---

## Resource Status (as of ~04:00 UTC)

### Node 1 (192.168.200.12)

| Resource | Value |
|----------|-------|
| **GPU** | 96% util, 48.7 GB used (CE training 47.8 GB + baselines 0.9 GB) |
| **RAM** | 77 / 119 GiB used, 42 GiB available |
| **CPU** | Load 2.46 on 20 cores |
| **Swap** | 902 MiB / 15 GiB |
| **GPU Temp** | 73°C |

### Node 2 (192.168.200.13)

| Resource | Value |
|----------|-------|
| **Container** | `coliee_longctx` (nvcr.io/nvidia/pytorch:25.09-py3) |
| **Disk** | 517 GB available |

---

## After Training Completes

1. **Check final F1** in the training log (grep for "Training Complete" and "F1=")
2. **For longctx model:** Rsync the trained model back from Node 2:
   ```bash
   rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
     nvidia@192.168.200.13:/home/nvidia/coliee2026/output/models_v2/crossencoder/ \
     /home/alphabridge/Research/coliee2026/output/models_v2/crossencoder_longctx/
   ```
3. **A/B test** each approach through the meta-learner (~11 min per run, stages 1-5 cached)
4. **Launch "passage" approach** on Node 1 after "smart" finishes:
   ```bash
   # Change mode
   sed -i 's/CROSSENCODER_MODE = "smart"/CROSSENCODER_MODE = "passage"/' src/graphrag/config.py
   # Remove stage4 cache so it retrains
   rm output/pipeline_cache/stage4.pkl
   # Launch
   docker exec -d -e PYTHONPATH=/workspace/coliee2026/src coliee_optionc \
     bash -c "python /workspace/coliee2026/scripts/train_pipeline.py > /workspace/coliee2026/output/training_passage.log 2>&1"
   ```
5. **COMMIT THE CODE** — all pipeline work is uncommitted (2+ days of development)
6. **Pick best 3 for submission** (competition allows max 3 runs)
