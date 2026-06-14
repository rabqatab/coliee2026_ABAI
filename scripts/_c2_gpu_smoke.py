"""GPU smoke for C2: validate e5-mistral-7b loads + forwards in bf16 on GB10.

Triggers the model download into the shared HF cache (reused by the full run),
and confirms the Blackwell SM121 bf16 path works before the long embedding job.
"""
import torch
from transformers import AutoModel, AutoTokenizer

M = "intfloat/e5-mistral-7b-instruct"
print(f"[smoke] loading {M} ...", flush=True)
tok = AutoTokenizer.from_pretrained(M)
mod = AutoModel.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
p = next(mod.parameters())
print(f"[smoke] loaded: device={p.device} dtype={p.dtype} gpu={torch.cuda.get_device_name(0)}", flush=True)

txt = [
    "Instruct: Given a legal case, retrieve prior cases it cites\nQuery: refugee protection division judicial review",
    "The Federal Court allowed the application for judicial review of the decision.",
]
b = tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
with torch.no_grad():
    out = mod(**b)
emb = out.last_hidden_state[:, -1]  # crude last-token; full script does masked last-token
print(f"[smoke] OK forward: emb={tuple(emb.shape)} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
print("[smoke] PASS", flush=True)
