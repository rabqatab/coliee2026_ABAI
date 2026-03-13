"""Proposition extraction: text windows around <FRAGMENT_SUPPRESSED> markers.

Mirrors the UMNLP 2024 approach (Paper 27) of extracting "propositions"
from citation-suppressed regions and encoding them with a sentence transformer.
"""
import re
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class PropositionExtractor:
    def __init__(self, window_size=100, cache_dir="output/baselines/umnlp_embeddings"):
        self.window_size = window_size  # words on each side
        self.cache_dir = Path(cache_dir)
        self.model = None

    def extract_windows(self, text: str) -> list[str]:
        """Extract text windows around <FRAGMENT_SUPPRESSED> markers."""
        windows = []
        pattern = r'<FRAGMENT_SUPPRESSED>'
        for match in re.finditer(pattern, text):
            start = match.start()
            # Get surrounding text
            before = text[:start].split()[-self.window_size:]
            after = text[match.end():].split()[:self.window_size]
            window = ' '.join(before + after)
            if window.strip():
                windows.append(window)
        return windows if windows else [text[:500]]  # fallback to first 500 chars

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def encode_propositions(self, corpus: dict[str, str]) -> dict[str, np.ndarray]:
        """Encode proposition windows for all docs. Returns mean-pooled embedding per doc."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings = {}
        to_encode = {}

        for doc_id, text in corpus.items():
            cache_path = self.cache_dir / f"{doc_id}.npy"
            if cache_path.exists():
                embeddings[doc_id] = np.load(cache_path)
            else:
                to_encode[doc_id] = text

        if to_encode:
            logger.info("Encoding propositions for %d docs (cached: %d)",
                        len(to_encode), len(embeddings))
            model = self._get_model()

            # Process in batches
            ids = list(to_encode.keys())
            for i in range(0, len(ids), 100):
                batch_ids = ids[i:i + 100]
                all_windows = []
                doc_indices = []  # track which doc each window belongs to

                for doc_id in batch_ids:
                    windows = self.extract_windows(to_encode[doc_id])
                    for w in windows:
                        all_windows.append(w[:512])  # truncate
                        doc_indices.append(doc_id)

                if all_windows:
                    window_embs = model.encode(
                        all_windows, batch_size=64,
                        normalize_embeddings=True, show_progress_bar=False,
                    )

                    # Mean-pool per doc
                    from collections import defaultdict
                    doc_embs = defaultdict(list)
                    for idx, doc_id in enumerate(doc_indices):
                        doc_embs[doc_id].append(window_embs[idx])

                    for doc_id in batch_ids:
                        if doc_id in doc_embs:
                            emb = np.mean(doc_embs[doc_id], axis=0).astype(np.float32)
                            emb = emb / (np.linalg.norm(emb) + 1e-8)
                        else:
                            emb = np.zeros(384, dtype=np.float32)
                        np.save(self.cache_dir / f"{doc_id}.npy", emb)
                        embeddings[doc_id] = emb

                if (i // 100) % 10 == 0:
                    logger.info("Encoded %d/%d docs",
                                min(i + 100, len(ids)), len(ids))

        return embeddings

    def proposition_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two proposition embeddings."""
        return float(np.dot(emb1, emb2))
