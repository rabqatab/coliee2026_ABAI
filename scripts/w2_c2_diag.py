"""C2 diagnostic: why is naive full-doc dense near-random? Compare QUERY representations.

Same passages (corpus doc-heads @max_length), full corpus (9,556), a 50-query subset.
Query variants (e5-mistral, last-token, query instruction):
  full_doc  - query = full query document (head)           [control; expect ~random]
  contexts  - query = the query's citation-context windows  [hypothesis: matches cited cases]
Reports dense Recall@{50,100,200} per variant + the BM25-RRF reference on the same queries.
Runs in the nvcr pytorch container (venv torch is CPU-only). ~15-20 min @512.
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
MAXLEN = 512
N_QUERIES = 50
INSTR = "Instruct: Given a legal case, retrieve prior cases it cites\nQuery: "
OUT = REPO / "output" / "w2" / "c2_diag.json"


def load_stage1():
    sys.path.insert(0, str(REPO / "src"))
    import coliee_task1
    import coliee_task1.stages.citation_context as cc
    sys.modules.setdefault("graphrag", coliee_task1)
    sys.modules.setdefault("graphrag.stages", coliee_task1.stages)
    sys.modules.setdefault("graphrag.citation_context", cc)
    sys.modules.setdefault("graphrag.stages.citation_context", cc)
    with open(REPO / "output" / "pipeline_cache" / "stage1.pkl", "rb") as f:
        raw, clean, contexts = pickle.load(f)
    return clean, contexts


def last_token_pool(h, mask):
    if mask[:, -1].sum() == mask.shape[0]:
        return h[:, -1]
    sl = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), sl]


def embed(texts, tok, mod, instr=False, bs=8):
    """Official e5-mistral recipe: truncate to max_len-1, APPEND eos, then pad.
    e5-mistral forms the embedding at the EOS position; omitting EOS -> degenerate."""
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
        if (i // bs) % 50 == 0:
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


def topk(qembs, cembs, cids, qids, k=200):
    idx = {c: i for i, c in enumerate(cids)}
    res = {}
    for qi, q in enumerate(qids):
        s = cembs @ qembs[qi]
        if q in idx:
            s[idx[q]] = -np.inf
        part = np.argpartition(-s, k)[:k]
        order = part[np.argsort(-s[part])]
        res[q] = [(cids[i], float(s[i])) for i in order]
    return res


def main():
    clean, contexts = load_stage1()
    labels = json.load(open(REPO / "data" / "task1" / "task1_train_labels_2026.json"))
    # subset queries with gold present
    qids = []
    for q in sorted(labels):
        if q in clean and any(g in clean for g in labels[q]):
            qids.append(q)
        if len(qids) >= N_QUERIES:
            break
    cids = sorted(clean.keys())
    print(f"corpus={len(cids)} queries={len(qids)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

    print("embedding corpus (passages, head)...", flush=True)
    cembs = embed([clean[c] for c in cids], tok, mod, instr=False)

    # query variant texts
    q_full = [clean[q] for q in qids]
    q_ctx = []
    for q in qids:
        dc = contexts.get(q)
        ctx = " ".join(c.text for c in dc.contexts) if dc else clean[q]
        q_ctx.append(ctx if ctx.strip() else clean[q])

    print("embedding queries: full_doc...", flush=True)
    e_full = embed(q_full, tok, mod, instr=True)
    print("embedding queries: contexts...", flush=True)
    e_ctx = embed(q_ctx, tok, mod, instr=True)

    res = {
        "full_doc": recall_at(topk(e_full, cembs, cids, qids), labels, qids),
        "contexts": recall_at(topk(e_ctx, cembs, cids, qids), labels, qids),
    }
    # BM25-RRF reference on same queries
    try:
        with open(REPO / "output" / "pipeline_cache" / "stage2.pkl", "rb") as f:
            rrf_results = pickle.load(f)[0]
        bm = {q: rrf_results.get(q, []) for q in qids}
        res["bm25_rrf_ref"] = recall_at(bm, labels, qids)
    except Exception as e:
        res["bm25_rrf_ref"] = {"error": str(e)}

    out = {"model": MODEL, "max_length": MAXLEN, "n_queries": len(qids),
           "n_corpus": len(cids), "results": res}
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
