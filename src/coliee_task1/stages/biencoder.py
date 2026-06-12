"""Fine-tune a bi-encoder (BGE-large) with LoRA for legal case retrieval.

Trains on COLIEE query-candidate pairs with BM25-mined hard negatives,
using InfoNCE (in-batch negatives) contrastive loss via sentence-transformers.
"""
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from coliee_task1.stages.bm25 import BM25Index
from coliee_task1.config import (
    BIENCODER_MODEL,
    BIENCODER_LORA_RANK,
    BIENCODER_LORA_ALPHA,
    BIENCODER_LR,
    BIENCODER_EPOCHS,
    BIENCODER_BATCH_SIZE,
    BIENCODER_HARD_NEG_K,
    BIENCODER_TOP_K,
    MODELS_DIR,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def mine_hard_negatives(
    labels: dict[str, list[str]],
    bm25_index: BM25Index,
    corpus_texts: dict[str, str],
    n_negatives: int = 7,
    hard_neg_k: int = BIENCODER_HARD_NEG_K,
) -> list[dict[str, str]]:
    """Mine training triplets: (query, positive, hard_negative).

    Hard negatives are BM25 top-K candidates that are NOT in the label set.
    This provides challenging negatives that share lexical features with
    the query but are not actual citations.

    Returns list of dicts with keys: anchor, positive, negative
    """
    triplets = []
    for query_id, positives in labels.items():
        if query_id not in corpus_texts:
            continue
        query_text = corpus_texts[query_id]

        # Get BM25 candidates
        bm25_results = bm25_index.query(query_text, top_k=hard_neg_k)
        positive_set = set(positives)

        # Hard negatives: high BM25 rank but not positive
        hard_negs = [
            doc_id for doc_id, _ in bm25_results
            if doc_id not in positive_set and doc_id != query_id
        ]

        for pos_id in positives:
            if pos_id not in corpus_texts:
                continue
            pos_text = corpus_texts[pos_id]

            # Sample hard negatives
            neg_sample = hard_negs[:n_negatives] if len(hard_negs) >= n_negatives else hard_negs
            for neg_id in neg_sample:
                if neg_id in corpus_texts:
                    triplets.append({
                        "anchor": query_text[:2048],  # Truncate for memory
                        "positive": pos_text[:2048],
                        "negative": corpus_texts[neg_id][:2048],
                    })

    random.shuffle(triplets)
    logger.info("Mined %d training triplets from %d queries", len(triplets), len(labels))
    return triplets


def build_training_dataset(
    labels: dict[str, list[str]],
    corpus_texts: dict[str, str],
    bm25_index: BM25Index,
) -> Dataset:
    """Build HuggingFace Dataset of (anchor, positive, negative) triplets."""
    triplets = mine_hard_negatives(labels, bm25_index, corpus_texts)
    return Dataset.from_list(triplets)


def finetune_biencoder(
    labels: dict[str, list[str]],
    corpus_texts: dict[str, str],
    bm25_index: BM25Index,
    output_dir: Path | None = None,
    model_name: str = BIENCODER_MODEL,
    lora_rank: int = BIENCODER_LORA_RANK,
    lora_alpha: int = BIENCODER_LORA_ALPHA,
    lr: float = BIENCODER_LR,
    epochs: int = BIENCODER_EPOCHS,
    batch_size: int = BIENCODER_BATCH_SIZE,
) -> SentenceTransformer:
    """Fine-tune BGE-large with LoRA on COLIEE training data.

    Uses MultipleNegativesRankingLoss (InfoNCE) which treats all other
    positives in the batch as additional negatives, giving (batch_size - 1)
    effective negatives per sample.

    Args:
        labels: Training labels {query_id: [positive_ids]}
        corpus_texts: {doc_id: text} for all documents
        bm25_index: Pre-built BM25 index for hard negative mining
        output_dir: Where to save the fine-tuned model
        model_name: Base model to fine-tune
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Fine-tuned SentenceTransformer model
    """
    if output_dir is None:
        output_dir = MODELS_DIR / "biencoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", model_name)
    model = SentenceTransformer(model_name)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"],
    )
    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
    trainable = sum(p.numel() for p in model[0].auto_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model[0].auto_model.parameters())
    logger.info("LoRA applied: %d trainable / %d total params (%.2f%%)", trainable, total, 100 * trainable / total)

    # Build training data
    logger.info("Building training dataset with hard negatives...")
    train_dataset = build_training_dataset(labels, corpus_texts, bm25_index)
    logger.info("Training dataset: %d examples", len(train_dataset))

    # Loss: InfoNCE with in-batch negatives
    loss = MultipleNegativesRankingLoss(model)

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        seed=RANDOM_SEED,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_num_workers=2,
    )

    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    logger.info("Starting bi-encoder training for %d epochs...", epochs)
    trainer.train()

    # Save
    model.save(str(output_dir / "final"))
    logger.info("Bi-encoder saved to %s", output_dir / "final")

    return model


def encode_corpus(
    model: SentenceTransformer,
    corpus_texts: dict[str, str],
    batch_size: int = 32,
    max_length: int = 512,
) -> tuple[list[str], np.ndarray]:
    """Encode all documents in the corpus using the fine-tuned bi-encoder.

    Returns:
        (doc_ids, embeddings) where embeddings is (n_docs, embed_dim) array
    """
    doc_ids = sorted(corpus_texts.keys())
    texts = [corpus_texts[did][:2048] for did in doc_ids]

    logger.info("Encoding %d documents...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return doc_ids, np.array(embeddings)


def biencoder_retrieve(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_doc_ids: list[str],
    top_k: int = BIENCODER_TOP_K,
) -> list[tuple[str, float]]:
    """Retrieve top-K candidates using cosine similarity."""
    similarities = query_embedding @ corpus_embeddings.T
    top_indices = np.argsort(-similarities)[:top_k]
    return [(corpus_doc_ids[i], float(similarities[i])) for i in top_indices]


def dense_retrieve_full_corpus(
    query_ids: list[str],
    doc_ids: list[str],
    embeddings: np.ndarray,
    top_k: int = 200,
) -> dict[str, list[tuple[str, float]]]:
    """Full-corpus dense retrieval using pre-computed embeddings.

    For each query, computes cosine similarity against ALL corpus documents
    and returns top-K results. Complementary to BM25 — finds semantically
    similar documents that may lack lexical overlap.
    """
    import logging
    _logger = logging.getLogger(__name__)

    doc_idx = {did: i for i, did in enumerate(doc_ids)}

    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = embeddings / norms

    dense_results: dict[str, list[tuple[str, float]]] = {}
    for i, qid in enumerate(query_ids):
        if qid not in doc_idx:
            continue

        q_emb = normed[doc_idx[qid]]
        sims = normed @ q_emb
        sims[doc_idx[qid]] = -1.0  # exclude self

        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        dense_results[qid] = [
            (doc_ids[idx], float(sims[idx])) for idx in top_indices
        ]

        if (i + 1) % 200 == 0:
            _logger.info("  Dense retrieval: %d/%d queries", i + 1, len(query_ids))

    _logger.info("Dense retrieval complete: %d queries, top-%d", len(dense_results), top_k)
    return dense_results
