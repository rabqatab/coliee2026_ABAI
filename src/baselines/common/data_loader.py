"""Unified data loading and train/val split for baseline comparison."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from coliee_task1.config import TRAIN_DOCS_DIR, TEST_DOCS_DIR, TRAIN_LABELS
from coliee_task1.stages.preprocess import preprocess, load_corpus

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Loaded dataset with chronological train/val split."""
    corpus: dict[str, str]
    train_queries: list[str]
    val_queries: list[str]
    labels: dict[str, list[str]]


def load_dataset(val_fraction: float = 0.2) -> Dataset:
    """Load full corpus and create chronological train/val split.

    Higher numeric doc IDs = more recent cases -> validation set.
    """
    raw_train = load_corpus(TRAIN_DOCS_DIR)
    raw_test = load_corpus(TEST_DOCS_DIR)
    raw_corpus = {**raw_train, **raw_test}
    corpus = {did: preprocess(txt) for did, txt in raw_corpus.items()}
    logger.info("Loaded %d documents (%d train + %d test files)",
                len(corpus), len(raw_train), len(raw_test))

    with open(TRAIN_LABELS) as f:
        labels = json.load(f)

    all_query_ids = sorted(labels.keys(),
                           key=lambda x: int(x.replace(".txt", "")))
    split_idx = int(len(all_query_ids) * (1 - val_fraction))
    train_queries = all_query_ids[:split_idx]
    val_queries = all_query_ids[split_idx:]

    logger.info("Split: %d train queries, %d val queries (%.0f%%)",
                len(train_queries), len(val_queries),
                100 * len(val_queries) / len(all_query_ids))

    return Dataset(
        corpus=corpus,
        train_queries=train_queries,
        val_queries=val_queries,
        labels=labels,
    )
