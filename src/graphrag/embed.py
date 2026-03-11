"""Embedding utilities using Ollama API."""
import logging

import numpy as np
from typing import Sequence

from graphrag.config import EMBED_MODEL, EMBED_BATCH_SIZE
from graphrag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def embed_texts(
    client: OllamaClient,
    texts: Sequence[str],
    model: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed a list of texts into vectors.

    Returns numpy array of shape (len(texts), embed_dim).
    """
    all_embeddings = []

    if show_progress:
        from tqdm import tqdm
        batches = range(0, len(texts), batch_size)
        iterator = tqdm(batches, desc="Embedding", unit="batch")
    else:
        iterator = range(0, len(texts), batch_size)

    for start in iterator:
        batch = list(texts[start : start + batch_size])
        vectors = client.embed(model, batch)
        all_embeddings.extend(vectors)

    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of vectors.

    Args:
        a: shape (m, d)
        b: shape (n, d)

    Returns:
        shape (m, n) similarity matrix
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T
