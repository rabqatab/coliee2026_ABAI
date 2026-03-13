"""Generate node feature embeddings for the CaseLink GNN."""
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class NodeFeatureGenerator:
    """Encodes case texts into dense vectors using a sentence-transformer.

    Embeddings are cached to disk to avoid redundant computation across runs.
    """

    def __init__(self, cache_dir: str = "output/baselines/caselink_embeddings"):
        self.cache_dir = Path(cache_dir)
        self.model = None

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def encode_cases(self, corpus: dict[str, str]) -> dict[str, np.ndarray]:
        """Encode all cases with sentence-transformer, cache to disk.

        Args:
            corpus: Mapping from doc_id to full text.

        Returns:
            Mapping from doc_id to L2-normalized embedding (384-dim).
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings: dict[str, np.ndarray] = {}
        to_encode: dict[str, str] = {}

        for doc_id, text in corpus.items():
            cache_path = self.cache_dir / f"{doc_id}.npy"
            if cache_path.exists():
                embeddings[doc_id] = np.load(cache_path)
            else:
                to_encode[doc_id] = text

        if to_encode:
            logger.info(
                "Encoding %d case nodes (cached: %d)", len(to_encode), len(embeddings)
            )
            model = self._get_model()
            ids = list(to_encode.keys())
            # Truncate to first 512 chars for lightweight encoding
            texts = [to_encode[d][:512] for d in ids]
            embs = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            for i, doc_id in enumerate(ids):
                emb = embs[i].astype(np.float32)
                np.save(self.cache_dir / f"{doc_id}.npy", emb)
                embeddings[doc_id] = emb
        else:
            logger.info("All %d case embeddings loaded from cache", len(embeddings))

        return embeddings
