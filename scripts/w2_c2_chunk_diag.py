"""C2 chunk-MaxSim diagnostic (NOWJ/ColBERT style): does chunk-level late interaction
beat single-vector dense for citation retrieval?

Same 50 queries + full corpus as w2_c2_diag.py (comparable to single-vector R@200=0.506).
Docs split into word-chunks; e5-mistral (EOS recipe) embeds every chunk; per (query,doc):
  score = sum_{qchunk} max_{dchunk} cos(qchunk, dchunk)   (ColBERT MaxSim)
Reports MaxSim Recall@{50,100,200} + BM25-RRF reference. ~30-40 min on one GB10 @384.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parent.parent
MODEL = "intfloat/e5-mistral-7b-instruct"
MAXLEN = 384
N_QUERIES = 50
CHUNK_WORDS = 300
MAX_CHUNKS = 4
INSTR = "Instruct: Given a legal case, retrieve prior cases it cites\nQuery: "
OUT = REPO / "output" / "w2" / "c2_chunk_diag.json"


def load_stage1():
    sys.path.insert(0, str(REPO / "src"))
    import coliee_task1
    import coliee_task1.stages.citation_context as cc
    for m in ("graphrag", "graphrag.stages", "graphrag.citation_context", "graphrag.stages.citation_context"):
        sys.modules.setdefault(m, cc if "citation" in m else coliee_task1)
    with open(REPO / "output" / "pipeline_cache" / "stage1.pkl", "rb") as f:
        _, clean, _ = pickle.load(f)
    return clean


def chunks_of(text):
    w = text.split()
    if not w:
        return [text]
    return [" ".join(w[i:i + CHUNK_WORDS]) for i in range(0, min(len(w), CHUNK_WORDS * MAX_CHUNKS), CHUNK_WORDS)]


def last_token_pool(h, mask):
    if mask[:, -1].sum() == mask.shape[0]:
        return h[:, -1]
    sl = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), sl]


def embed(texts, tok, mod, instr=False, bs=16):
    out = []
    t0 = time.time()
    eos = tok.eos_token_id
    for i in range(0, len(texts), bs):
        chunk = [(INSTR + t) if instr else t for t in texts[i:i + bs]]
        bd = tok(chunk, max_length=MAXLEN - 1, truncation=True, padding=False,
                 add_special_tokens=True, return_attention_mask=False)
        bd["input_ids"] = [ids + [eos] for ids in bd["input_ids"]]
        enc = tok.pad(bd, padding=True, return_attention_mask=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = mod(**enc).last_hidden_state
            e = torch.nn.functional.normalize(last_token_pool(h, enc["attention_mask"]), p=2, dim=1)
        out.append(e.float().cpu().numpy())
        if (i // bs) % 100 == 0:
            print(f"  embedded {i}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    return np.vstack(out).astype(np.float32)


def recall_at(rankings, labels, qids, ks=(50, 100, 200)):
    out = {}
    for k in ks:
        g = h = 0
        for q in qids:
            gold = set(labels.get(q, []))
            if not gold:
                continue
            top = [c for c, _ in rankings[q][:k]]
            g += len(gold); h += len(gold & set(top))
        out[f"R@{k}"] = h / g if g else 0.0
    return out


def main():
    clean = load_stage1()
    labels = json.load(open(REPO / "data" / "task1" / "task1_train_labels_2026.json"))
    qids = []
    for q in sorted(labels):
        if q in clean and any(g in clean for g in labels[q]):
            qids.append(q)
        if len(qids) >= N_QUERIES:
            break
    cids = sorted(clean.keys())

    # build corpus chunks + doc index mapping
    all_chunks, doc_idx = [], []
    for di, c in enumerate(cids):
        ch = chunks_of(clean[c])
        all_chunks.extend(ch)
        doc_idx.extend([di] * len(ch))
    doc_idx = np.array(doc_idx)
    print(f"corpus={len(cids)} chunks={len(all_chunks)} queries={len(qids)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

    print("embedding corpus chunks ...", flush=True)
    dce = embed(all_chunks, tok, mod, instr=False)  # (Nc, d)
    ndoc = len(cids)
    id_to_idx = {c: i for i, c in enumerate(cids)}

    print("MaxSim retrieval ...", flush=True)
    rankings = {}
    for q in qids:
        qe = embed(chunks_of(clean[q]), tok, mod, instr=True)  # (qc, d)
        sims = qe @ dce.T  # (qc, Nc)
        score = np.zeros(ndoc, dtype=np.float32)
        for row in sims:  # per query chunk: max over each doc's chunks, then sum
            dm = np.full(ndoc, -1e9, dtype=np.float32)
            np.maximum.at(dm, doc_idx, row)
            score += dm
        if q in id_to_idx:
            score[id_to_idx[q]] = -1e9
        top = np.argpartition(-score, 200)[:200]
        top = top[np.argsort(-score[top])]
        rankings[q] = [(cids[i], float(score[i])) for i in top]

    res = {"chunk_maxsim": recall_at(rankings, labels, qids)}
    try:
        with open(REPO / "output" / "pipeline_cache" / "stage2.pkl", "rb") as f:
            rrf = pickle.load(f)[0]
        res["bm25_rrf_ref"] = recall_at({q: rrf.get(q, []) for q in qids}, labels, qids)
    except Exception as e:
        res["bm25_rrf_ref"] = {"error": str(e)}
    res["note"] = "compare chunk_maxsim vs single-vector R@200=0.506 (same 50 queries, w2_c2_diag)"

    out = {"model": MODEL, "chunk_words": CHUNK_WORDS, "max_chunks": MAX_CHUNKS,
           "n_queries": len(qids), "n_corpus": len(cids), "n_chunks": len(all_chunks), "results": res}
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
