"""OOD probe (#4): is the official 2026 TEST query set distributionally shifted from TRAIN?

Label-free (test gold is withheld). Embed train queries (2001) and official test queries (400)
with e5-mistral (EOS recipe), then measure how far test queries sit from the TRAIN query
manifold vs how far train queries sit from each other (leave-one-out). Systematically larger
test->train nearest-neighbour distance = measurable OOD — our hypothesised collapse driver.
Runs in the nvcr pytorch container (venv torch is CPU-only). ~5 min @512.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parent.parent
# OOD manifold probe is embedder-agnostic; use a small CLS-pooled model to avoid GB10
# unified-memory contention (a 7B model OOMs when other jobs share the 128GB pool).
MODEL = "BAAI/bge-large-en-v1.5"
MAXLEN = 512
INSTR = ""  # bge: no instruction prefix for this probe
OUT = REPO / "output" / "w2" / "ood_embedding.json"


def load_clean():
    sys.path.insert(0, str(REPO / "src"))
    import coliee_task1
    import coliee_task1.stages.citation_context as cc
    for m in ("graphrag", "graphrag.stages", "graphrag.citation_context", "graphrag.stages.citation_context"):
        sys.modules.setdefault(m, cc if "citation" in m else coliee_task1)
    with open(REPO / "output" / "pipeline_cache" / "stage1.pkl", "rb") as f:
        _, clean, _ = pickle.load(f)
    return clean


def embed(texts, tok, mod, bs=32):
    """CLS pooling for BGE (h[:,0]), L2-normalized."""
    out = []
    for i in range(0, len(texts), bs):
        chunk = [INSTR + t for t in texts[i:i + bs]]
        enc = tok(chunk, max_length=MAXLEN, truncation=True, padding=True,
                  return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = mod(**enc).last_hidden_state[:, 0]  # CLS
            e = torch.nn.functional.normalize(h, p=2, dim=1)
        out.append(e.float().cpu().numpy())
        if (i // bs) % 40 == 0:
            print(f"  {i}/{len(texts)}", flush=True)
    return np.vstack(out).astype(np.float32)


def knn_mean(q, ref, k=5, exclude_self=False):
    """mean cosine to the top-k nearest rows of `ref` for each row of `q`."""
    sims = q @ ref.T  # (nq, nref), all L2-normalized
    if exclude_self:
        np.fill_diagonal(sims, -1.0)
    part = np.partition(-sims, k, axis=1)[:, :k]
    return (-part).mean(axis=1)  # (nq,)


def main():
    clean = load_clean()
    train_labels = json.load(open(REPO / "data" / "task1" / "task1_train_labels_2026.json"))
    test_ids = list(json.load(open(REPO / "data" / "task1" / "task1_test_no_labels_2026.json")))
    train_ids = [q for q in sorted(train_labels) if q in clean]
    # test query texts from the test files dir
    tdir = REPO / "data" / "task1" / "task1_test_files_2026"
    test_texts, test_ok = [], []
    for q in test_ids:
        p = tdir / q
        if p.exists():
            test_texts.append(p.read_text(errors="ignore")); test_ok.append(q)
    print(f"train queries={len(train_ids)} test queries={len(test_ok)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    print("embedding train queries...", flush=True)
    Etr = embed([clean[q] for q in train_ids], tok, mod)
    print("embedding test queries...", flush=True)
    Ete = embed(test_texts, tok, mod)

    # nearest-train-neighbour cosine: train (LOO) vs test
    tr_nn = knn_mean(Etr, Etr, k=5, exclude_self=True)
    te_nn = knn_mean(Ete, Etr, k=5)
    # centroid distance
    c = Etr.mean(0); c /= (np.linalg.norm(c) + 1e-9)
    tr_c = Etr @ c
    te_c = Ete @ c

    def stats(a):
        a = np.asarray(a)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25))}

    tr_p5 = float(np.percentile(tr_nn, 5))
    frac_test_below = float((te_nn < tr_p5).mean())
    # effect size (Cohen's d) on NN cosine
    pooled = np.sqrt((tr_nn.var() + te_nn.var()) / 2) + 1e-9
    cohend = float((tr_nn.mean() - te_nn.mean()) / pooled)

    out = {
        "model": MODEL, "max_length": MAXLEN,
        "n_train": len(train_ids), "n_test": len(test_ok),
        "nn5_cosine_to_train": {"train_LOO": stats(tr_nn), "test": stats(te_nn)},
        "centroid_cosine": {"train": stats(tr_c), "test": stats(te_c)},
        "frac_test_below_train_p5_nn": frac_test_below,
        "cohens_d_nn_cosine_train_minus_test": cohend,
        "reading": "lower test NN-cosine and high frac_below = test queries sit farther from the train manifold (OOD)",
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
