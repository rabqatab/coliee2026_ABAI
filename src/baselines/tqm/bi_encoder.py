"""Fine-tune a sentence-transformer bi-encoder for legal case retrieval.

Based on TQM 2024 approach (arXiv 2404.00947): uses MultipleNegativesRankingLoss
with hard negatives from BM25 candidates.
"""
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class TQMBiEncoder:
    """Bi-encoder wrapper for TQM pipeline: fine-tune + encode + cache."""

    def __init__(
        self,
        model_dir: str = "output/baselines/models/tqm_biencoder",
        cache_dir: str = "output/baselines/tqm_embeddings",
    ):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.model = None

    def finetune(
        self,
        corpus: dict[str, str],
        train_queries: list[str],
        labels: dict[str, list[str]],
        bm25_candidates: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        """Fine-tune all-MiniLM-L6-v2 with MultipleNegativesRankingLoss.

        Creates (query_text, positive_text, hard_negative_text) triplets.
        Hard negatives = top BM25 candidates that are NOT in labels.

        3 epochs, batch_size=32, lr=2e-5.
        Saves to self.model_dir.
        """
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            logger.info("Loading existing fine-tuned model from %s", self.model_dir)
            self.model = SentenceTransformer(str(self.model_dir))
            return

        logger.info("Fine-tuning bi-encoder on %d queries", len(train_queries))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Build triplets
        examples = []
        for qid in train_queries:
            positives = labels.get(qid, [])
            if not positives:
                continue
            # Get hard negatives from BM25 candidates
            if bm25_candidates and qid in bm25_candidates:
                hard_negs = [
                    cid
                    for cid, _ in bm25_candidates[qid]
                    if cid != qid and cid not in positives
                ][:10]
            else:
                hard_negs = []

            for pos_id in positives:
                if pos_id not in corpus:
                    continue
                neg_id = hard_negs[0] if hard_negs else None
                if neg_id and neg_id in corpus:
                    examples.append(
                        InputExample(
                            texts=[
                                corpus[qid][:512],
                                corpus[pos_id][:512],
                                corpus[neg_id][:512],
                            ]
                        )
                    )
                else:
                    examples.append(
                        InputExample(
                            texts=[corpus[qid][:512], corpus[pos_id][:512]]
                        )
                    )

        logger.info("Built %d training examples", len(examples))
        loader = DataLoader(examples, shuffle=True, batch_size=32)
        loss = losses.MultipleNegativesRankingLoss(self.model)

        self.model.fit(
            train_objectives=[(loader, loss)],
            epochs=3,
            warmup_steps=100,
            show_progress_bar=True,
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.model_dir))
        logger.info("Saved fine-tuned model to %s", self.model_dir)

    def _load_model(self) -> None:
        """Load model if not already loaded."""
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            logger.info("Loading fine-tuned model from %s", self.model_dir)
            self.model = SentenceTransformer(str(self.model_dir))
        else:
            logger.info("No fine-tuned model found, using base all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode_corpus(self, corpus: dict[str, str]) -> dict[str, np.ndarray]:
        """Encode all docs, cache to disk as .npy files."""
        self._load_model()

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
                "Encoding %d docs (cached: %d)", len(to_encode), len(embeddings)
            )
            ids = list(to_encode.keys())
            texts = [to_encode[d][:512] for d in ids]
            embs = self.model.encode(
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
            logger.info("All %d embeddings loaded from cache", len(embeddings))

        return embeddings

    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity (embeddings are already L2-normalized)."""
        return float(np.dot(emb1, emb2))
