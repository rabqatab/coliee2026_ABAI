"""Fine-tune a cross-encoder for legal case reranking.

Supports 4 approaches for handling long documents:
  - "smart":      Smart truncation using citation context windows (default)
  - "longctx":    Long-context model (BGE-reranker-v2-m3, up to 8192 tokens)
  - "passage":    Passage-level scoring with max-pooling aggregation
  - "modernbert": ModernBERT-large (8192 native tokens, 5000-word budget)

All approaches produce the same output format: {query_id: {candidate_id: score}}
"""
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from coliee_task1.config import (
    CROSSENCODER_MODEL,
    CROSSENCODER_LR,
    CROSSENCODER_EPOCHS,
    CROSSENCODER_BATCH_SIZE,
    CROSSENCODER_MAX_LENGTH,
    CROSSENCODER_TOP_K,
    CROSSENCODER_PRUNE_TOP_CONTEXTS,
    MODELS_DIR,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text preparation helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_words: int) -> str:
    """Truncate text to max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def smart_truncate(
    text: str,
    context_texts: list[str] | None = None,
    max_words: int = 500,
    head_words: int = 100,
    tail_words: int = 100,
    word_budget: int = 500,
) -> str:
    """Smart truncation: head + citation contexts + tail.

    Instead of taking the first N words (which is often just boilerplate),
    this selects the most informative parts of a legal document:
      1. Opening (parties, background) - head_words
      2. Citation context windows (surroundings of FRAGMENT_SUPPRESSED) - fill middle
      3. Closing (disposition, conclusion) - tail_words

    Falls back to naive truncation if no contexts available.

    Args:
        word_budget: Overall word budget; if document fits within this, skip
            truncation entirely.  Also used as the effective max_words for the
            head+context+tail assembly when truncation *is* needed.
    """
    words = text.split()
    if len(words) <= word_budget:
        return text  # No truncation needed -- full document fits

    if not context_texts:
        return " ".join(words[:word_budget])

    head = " ".join(words[:head_words])
    tail = " ".join(words[-tail_words:]) if len(words) > tail_words else ""

    # Fill middle budget with citation context windows
    middle_budget = word_budget - head_words - tail_words
    middle_parts = []
    used_words = 0
    for ctx in context_texts:
        ctx_words = ctx.split()
        if used_words + len(ctx_words) > middle_budget:
            remaining = middle_budget - used_words
            if remaining > 20:
                middle_parts.append(" ".join(ctx_words[:remaining]))
            break
        middle_parts.append(ctx)
        used_words += len(ctx_words)

    parts = [head]
    if middle_parts:
        parts.append(" [SEP] ".join(middle_parts))
    if tail:
        parts.append(tail)
    return " [SEP] ".join(parts)


def chunk_document(
    text: str,
    chunk_words: int = 200,
    overlap_words: int = 50,
) -> list[str]:
    """Split document into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    start = 0
    step = chunk_words - overlap_words
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def select_top_passages(
    doc_text: str,
    reference_text: str,
    max_passages: int = 5,
    chunk_words: int = 200,
    overlap_words: int = 50,
) -> list[str]:
    """Chunk a document and select top passages by word overlap with reference."""
    chunks = chunk_document(doc_text, chunk_words, overlap_words)
    if len(chunks) <= max_passages:
        return chunks

    ref_words = set(reference_text.lower().split())
    scored = []
    for chunk in chunks:
        chunk_word_set = set(chunk.lower().split())
        overlap = len(ref_words & chunk_word_set)
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: -x[0])
    return [chunk for _, chunk in scored[:max_passages]]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PairDataset(TorchDataset):
    """Dataset of (text_a, text_b, label) pairs for 2-class cross-encoder."""

    def __init__(
        self,
        pairs: list[tuple[str, str, int]],
        tokenizer: AutoTokenizer,
        max_length: int = CROSSENCODER_MAX_LENGTH,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_a, text_b, label = self.pairs[idx]
        encoding = self.tokenizer(
            text_a, text_b,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class RegressionPairDataset(TorchDataset):
    """Dataset for single-logit reranker models (e.g., BGE-reranker)."""

    def __init__(
        self,
        pairs: list[tuple[str, str, int]],
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_a, text_b, label = self.pairs[idx]
        encoding = self.tokenizer(
            text_a, text_b,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Training pair construction
# ---------------------------------------------------------------------------

def build_training_pairs(
    labels: dict[str, list[str]],
    corpus_texts: dict[str, str],
    bm25_candidates: dict[str, list[tuple[str, float]]],
    contexts: dict | None = None,
    mode: str = "smart",
    max_words: int = 500,
    neg_ratio: int = 4,
    dense_results: dict[str, list[tuple[str, float]]] | None = None,
) -> list[tuple[str, str, int]]:
    """Build (query_text, candidate_text, label) pairs for cross-encoder training.

    Args:
        contexts: {doc_id: DocumentContexts} for smart truncation
        mode: "smart", "longctx", or "passage"
        max_words: Word budget per document
        dense_results: Optional dense retrieval results for semantic hard negatives.
            Format: {query_id: [(candidate_id, score), ...]}
            When provided, negatives are mixed 70% BM25 + 30% dense.
    """
    def _prepare_text(doc_id: str) -> str:
        text = corpus_texts.get(doc_id, "")
        if mode == "modernbert" and contexts:
            dc = contexts.get(doc_id)
            ctx_texts = [c.text for c in dc.contexts] if dc and dc.contexts else []
            return smart_truncate(text, ctx_texts, max_words=max_words, word_budget=max_words)
        elif mode == "smart" and contexts:
            dc = contexts.get(doc_id)
            ctx_texts = [c.text for c in dc.contexts] if dc and dc.contexts else []
            return smart_truncate(text, ctx_texts, max_words=max_words)
        elif mode == "longctx":
            return _truncate(text, max_words)
        else:
            return _truncate(text, max_words)

    pairs = []
    for query_id, positives in labels.items():
        if query_id not in corpus_texts:
            continue
        query_text = _prepare_text(query_id)
        positive_set = set(positives)

        for pos_id in positives:
            if pos_id not in corpus_texts:
                continue
            pairs.append((query_text, _prepare_text(pos_id), 1))

        bm25_results = bm25_candidates.get(query_id, [])
        hard_negs = [
            doc_id for doc_id, _ in bm25_results
            if doc_id not in positive_set and doc_id != query_id
        ]

        # When dense results available, mix 70% BM25 + 30% dense negatives
        if dense_results is not None:
            dense_negs = [
                cid for cid, _ in dense_results.get(query_id, [])
                if cid not in positive_set and cid not in set(hard_negs[:20])
            ]
            n_total = len(positives) * neg_ratio
            n_bm25 = int(n_total * 0.7)
            n_dense = n_total - n_bm25
            selected_negs = hard_negs[:n_bm25] + dense_negs[:n_dense]
        else:
            selected_negs = hard_negs[:len(positives) * neg_ratio]

        for neg_id in selected_negs:
            if neg_id in corpus_texts:
                pairs.append((query_text, _prepare_text(neg_id), 0))

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    n_pos = sum(1 for _, _, l in pairs if l == 1)
    n_neg = len(pairs) - n_pos
    logger.info(
        "Training pairs [%s]: %d positive, %d negative (ratio 1:%.1f)",
        mode, n_pos, n_neg, n_neg / max(n_pos, 1),
    )
    return pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune_crossencoder(
    labels: dict[str, list[str]],
    corpus_texts: dict[str, str],
    bm25_candidates: dict[str, list[tuple[str, float]]],
    contexts: dict | None = None,
    output_dir: Path | None = None,
    model_name: str = CROSSENCODER_MODEL,
    lr: float = CROSSENCODER_LR,
    epochs: int = CROSSENCODER_EPOCHS,
    batch_size: int = CROSSENCODER_BATCH_SIZE,
    max_length: int = CROSSENCODER_MAX_LENGTH,
    mode: str = "smart",
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Fine-tune a cross-encoder reranker.

    Args:
        mode: "smart" (DeBERTa + citation contexts),
              "longctx" (BGE-reranker, long input),
              "passage" (DeBERTa + passage-level, trained same as smart),
              "modernbert" (ModernBERT-large, 8192-token context)
    """
    if output_dir is None:
        output_dir = MODELS_DIR / "crossencoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s, Mode: %s", device, mode)

    is_longctx = mode == "longctx"

    # --- Mode-specific overrides ---
    if mode == "modernbert":
        from coliee_task1.config import (
            CROSSENCODER_MODERNBERT_MODEL,
            CROSSENCODER_MODERNBERT_MAX_LENGTH,
            CROSSENCODER_MODERNBERT_BATCH_SIZE,
            CROSSENCODER_MODERNBERT_LR,
            CROSSENCODER_MODERNBERT_WORD_BUDGET,
        )
        model_name = CROSSENCODER_MODERNBERT_MODEL
        max_length = CROSSENCODER_MODERNBERT_MAX_LENGTH
        batch_size = CROSSENCODER_MODERNBERT_BATCH_SIZE
        lr = CROSSENCODER_MODERNBERT_LR
        num_labels = 2

    # --- Model setup ---
    logger.info("Loading base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_longctx:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1,
        ).to(device)
    elif mode == "modernbert":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels,
        ).to(device)
    else:
        # DeBERTa-v3: force fp32 to avoid NaN (known XSoftmax overflow in fp16)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2,
        ).to(device).float()
        logger.info("Forced fp32 for DeBERTa-v3 NaN prevention")

    # --- Training pairs ---
    max_words = CROSSENCODER_MODERNBERT_WORD_BUDGET if mode == "modernbert" else (3000 if is_longctx else 500)
    pairs = build_training_pairs(
        labels, corpus_texts, bm25_candidates,
        contexts=contexts, mode=mode, max_words=max_words,
    )

    if is_longctx:
        dataset = RegressionPairDataset(pairs, tokenizer, max_length)
    else:
        dataset = PairDataset(pairs, tokenizer, max_length)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # --- Training loop ---
    logger.info(
        "Starting cross-encoder training: %d epochs, %d steps/epoch",
        epochs, len(dataloader),
    )
    model.train()
    nan_count = 0

    if is_longctx:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n_correct = 0
        n_total = 0
        valid_steps = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            if is_longctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits.squeeze(-1), labels_tensor)
                preds = (outputs.logits.squeeze(-1) > 0).long()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_tensor,
                )
                loss = outputs.loss
                preds = outputs.logits.argmax(dim=-1)

            # NaN guard (DeBERTa-v3 can still occasionally NaN)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count <= 5:
                    logger.warning(
                        "NaN/Inf loss at epoch %d step %d, skipping",
                        epoch + 1, step + 1,
                    )
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            valid_steps += 1
            n_correct += (preds == labels_tensor.long()).sum().item()
            n_total += labels_tensor.size(0)

            if (step + 1) % 100 == 0:
                avg = total_loss / max(valid_steps, 1)
                acc = n_correct / max(n_total, 1)
                logger.info(
                    "  Epoch %d step %d/%d: loss=%.4f acc=%.4f%s",
                    epoch + 1, step + 1, len(dataloader), avg, acc,
                    f" (nan_skip={nan_count})" if nan_count else "",
                )

        avg_loss = total_loss / max(valid_steps, 1)
        accuracy = n_correct / max(n_total, 1)
        logger.info("Epoch %d: loss=%.4f, acc=%.4f", epoch + 1, avg_loss, accuracy)

    if nan_count > 0:
        logger.warning("Total NaN/Inf batches skipped: %d", nan_count)

    # Save model
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    (output_dir / "final" / "ce_mode.json").write_text(
        json.dumps({"mode": mode, "model_name": model_name, "max_length": max_length})
    )
    logger.info("Cross-encoder saved to %s (mode=%s)", output_dir / "final", mode)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference / reranking
# ---------------------------------------------------------------------------

def crossencoder_rerank(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    query_text: str,
    candidates: list[tuple[str, str]],
    max_length: int = CROSSENCODER_MAX_LENGTH,
    batch_size: int = 32,
    mode: str = "smart",
    query_contexts: list[str] | None = None,
    candidate_contexts: dict[str, list[str]] | None = None,
) -> list[tuple[str, float]]:
    """Rerank candidates using the cross-encoder.

    Args:
        mode: "smart", "longctx", "passage", or "modernbert"
        query_contexts: Citation context texts for the query (smart/passage/modernbert modes)
        candidate_contexts: {candidate_id: [context_texts]} (smart/modernbert mode)

    Returns:
        [(doc_id, score), ...] sorted by score descending.
    """
    device = next(model.parameters()).device
    model.eval()
    num_labels = model.config.num_labels

    if mode == "passage":
        return _rerank_passage(
            model, tokenizer, query_text, candidates,
            max_length, batch_size, num_labels,
            query_contexts=query_contexts,
        )

    # For smart and longctx: prepare texts then score
    doc_ids = [did for did, _ in candidates]

    if mode == "modernbert":
        from coliee_task1.config import (
            CROSSENCODER_MODERNBERT_MAX_LENGTH,
            CROSSENCODER_MODERNBERT_BATCH_SIZE,
            CROSSENCODER_MODERNBERT_WORD_BUDGET,
        )
        max_length = CROSSENCODER_MODERNBERT_MAX_LENGTH
        batch_size = CROSSENCODER_MODERNBERT_BATCH_SIZE
        wb = CROSSENCODER_MODERNBERT_WORD_BUDGET
        q_text = smart_truncate(query_text, query_contexts, max_words=wb, word_budget=wb)
        prepared_candidates = []
        for did, text in candidates:
            ctx = (candidate_contexts or {}).get(did, [])
            prepared_candidates.append(smart_truncate(text, ctx, max_words=wb, word_budget=wb))
    elif mode == "smart":
        q_text = smart_truncate(query_text, query_contexts, max_words=500)
        prepared_candidates = []
        for did, text in candidates:
            ctx = (candidate_contexts or {}).get(did, [])
            prepared_candidates.append(smart_truncate(text, ctx, max_words=500))
    elif mode == "longctx":
        q_text = _truncate(query_text, 3000)
        prepared_candidates = [_truncate(text, 3000) for _, text in candidates]
    else:
        q_text = _truncate(query_text, 500)
        prepared_candidates = [_truncate(text, 500) for _, text in candidates]

    scores = _score_pairs(
        model, tokenizer, q_text, prepared_candidates,
        max_length, batch_size, num_labels, device,
    )

    results = list(zip(doc_ids, scores))
    results.sort(key=lambda x: -x[1])
    return results


def _score_pairs(
    model, tokenizer, query_text: str, candidate_texts: list[str],
    max_length: int, batch_size: int, num_labels: int, device,
) -> list[float]:
    """Score a list of (query, candidate) pairs."""
    scores = []
    for i in range(0, len(candidate_texts), batch_size):
        batch_texts = candidate_texts[i:i + batch_size]
        texts_a = [query_text] * len(batch_texts)

        encoding = tokenizer(
            texts_a, batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            if num_labels == 1:
                batch_scores = torch.sigmoid(outputs.logits.squeeze(-1))
            else:
                batch_scores = torch.softmax(outputs.logits, dim=-1)[:, 1]
            scores.extend(batch_scores.cpu().tolist())
    return scores


def _rerank_passage(
    model, tokenizer, query_text: str,
    candidates: list[tuple[str, str]],
    max_length: int, batch_size: int, num_labels: int,
    query_contexts: list[str] | None = None,
    max_passages_per_doc: int = 5,
) -> list[tuple[str, float]]:
    """Passage-level reranking: chunk both docs, score passage pairs, max-pool.

    For query: use citation context windows if available, else chunk.
    For candidate: chunk and select top passages by overlap with query.
    """
    device = next(model.parameters()).device

    # Prepare query passages
    if query_contexts and len(query_contexts) > 0:
        q_passages = query_contexts[:max_passages_per_doc]
    else:
        q_passages = chunk_document(query_text, chunk_words=200, overlap_words=50)
        q_passages = q_passages[:max_passages_per_doc]

    results = []
    for did, doc_text in candidates:
        c_passages = select_top_passages(
            doc_text, query_text,
            max_passages=max_passages_per_doc,
        )

        # Build all (query_passage, candidate_passage) pairs
        all_q_texts = []
        all_c_texts = []
        for qp in q_passages:
            for cp in c_passages:
                all_q_texts.append(qp)
                all_c_texts.append(cp)

        if not all_q_texts:
            results.append((did, 0.0))
            continue

        # Batch score
        pair_scores = []
        for i in range(0, len(all_q_texts), batch_size):
            batch_q = all_q_texts[i:i + batch_size]
            batch_c = all_c_texts[i:i + batch_size]

            encoding = tokenizer(
                batch_q, batch_c,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoding)
                if num_labels == 1:
                    s = torch.sigmoid(outputs.logits.squeeze(-1))
                else:
                    s = torch.softmax(outputs.logits, dim=-1)[:, 1]
                pair_scores.extend(s.cpu().tolist())

        # Max-pool across all passage pairs
        results.append((did, max(pair_scores)))

    results.sort(key=lambda x: -x[1])
    return results
