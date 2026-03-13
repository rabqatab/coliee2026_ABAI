"""SAILER-style encoder with embedding cache for JNLP baseline."""
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

FALLBACK_CHAIN = [
    "CSHaitao/SAILER_en_finetune",
    "nlpaueb/legal-bert-base-uncased",
    "bert-base-uncased",
]


class SAILEREncoder:
    def __init__(self, cache_dir: str = "output/baselines/sailer_embeddings"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model, self._tokenizer, self._model_name = self._load_model()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.train(False)  # set to inference mode (eval)

    def _load_model(self) -> tuple:
        for name in FALLBACK_CHAIN:
            try:
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModel.from_pretrained(name)
                logger.info("Loaded encoder: %s", name)
                return model, tokenizer, name
            except Exception as e:
                logger.warning("Failed to load %s: %s", name, e)
        raise RuntimeError("Could not load any encoder from fallback chain")

    @torch.no_grad()
    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            output = self._model(**encoded)
            cls_emb = output.last_hidden_state[:, 0, :]  # CLS pooling
            # L2 normalize
            norms = cls_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cls_emb = cls_emb / norms
            all_embeddings.append(cls_emb.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def encode_corpus(self, corpus: dict[str, str]) -> dict[str, np.ndarray]:
        # Check cache
        cache_file = self._cache_dir / "corpus_embeddings.npz"
        if cache_file.exists():
            logger.info("Loading cached SAILER embeddings from %s", cache_file)
            data = np.load(cache_file)
            cached = {k: data[k] for k in data.files}
            if set(cached.keys()) == set(corpus.keys()):
                return cached
            logger.info("Cache mismatch (%d cached vs %d corpus), re-encoding",
                        len(cached), len(corpus))

        doc_ids = sorted(corpus.keys())
        texts = [corpus[d] for d in doc_ids]
        logger.info("Encoding %d documents with %s...", len(texts), self._model_name)
        embeddings = self.encode_batch(texts)

        result = {}
        for i, did in enumerate(doc_ids):
            result[did] = embeddings[i]

        # Save to cache
        np.savez(cache_file, **result)
        logger.info("Saved SAILER embeddings to %s", cache_file)
        return result

    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))
