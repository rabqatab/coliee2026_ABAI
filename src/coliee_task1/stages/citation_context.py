"""Citation context extraction from <FRAGMENT_SUPPRESSED> markers.

This module implements the core novel technique of the Option C pipeline:
extracting targeted sub-queries from the context windows surrounding
suppressed citation fragments in COLIEE query cases.
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from coliee_task1.config import (
    CONTEXT_WINDOW_WORDS,
    CONTEXT_MERGE_DISTANCE,
    CONTEXT_MIN_LENGTH,
)

logger = logging.getLogger(__name__)

FRAGMENT_PATTERN = re.compile(r"<FRAGMENT_SUPPRESSED>")


@dataclass
class CitationContext:
    """A context window around a suppressed citation."""
    doc_id: str
    index: int  # 0-based position among all markers in the document
    text: str  # The extracted context window text
    start_char: int  # Character offset in the original document
    end_char: int
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class DocumentContexts:
    """All citation contexts extracted from a single document."""
    doc_id: str
    raw_text: str = ""
    contexts: list[CitationContext] = field(default_factory=list)
    full_text_cleaned: str = ""  # Text with markers removed for full-doc retrieval

    @property
    def n_markers(self) -> int:
        return len(self.contexts)


def _get_word_boundaries(text: str) -> list[tuple[int, int]]:
    """Get (start, end) character offsets for every word in text."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def extract_contexts(
    raw_text: str,
    doc_id: str = "",
    window_words: int = CONTEXT_WINDOW_WORDS,
    merge_distance: int = CONTEXT_MERGE_DISTANCE,
    min_length: int = CONTEXT_MIN_LENGTH,
) -> DocumentContexts:
    """Extract citation context windows from a raw case document.

    For each <FRAGMENT_SUPPRESSED> marker, extracts ±window_words of surrounding
    text. Nearby windows are merged if their gap is within merge_distance words.
    Windows shorter than min_length words are discarded.

    Args:
        raw_text: Raw document text with <FRAGMENT_SUPPRESSED> markers intact
        doc_id: Document identifier
        window_words: Number of words on each side of a marker
        merge_distance: Merge windows whose gaps are within this many words
        min_length: Minimum word count for a valid context window

    Returns:
        DocumentContexts with all extracted context windows
    """
    # Find all marker positions
    markers = list(FRAGMENT_PATTERN.finditer(raw_text))
    if not markers:
        return DocumentContexts(
            doc_id=doc_id,
            raw_text=raw_text,
            full_text_cleaned=FRAGMENT_PATTERN.sub("", raw_text).strip(),
        )

    # Get word boundaries in the raw text
    word_bounds = _get_word_boundaries(raw_text)
    if not word_bounds:
        return DocumentContexts(doc_id=doc_id, raw_text=raw_text)

    # For each marker, find the context window character offsets
    raw_windows: list[tuple[int, int]] = []
    for marker in markers:
        marker_pos = marker.start()

        # Find the word index closest to the marker
        word_idx = _bisect_word_index(word_bounds, marker_pos)

        # Window bounds in word indices
        start_word = max(0, word_idx - window_words)
        end_word = min(len(word_bounds) - 1, word_idx + window_words)

        # Convert to character offsets
        start_char = word_bounds[start_word][0]
        end_char = word_bounds[end_word][1]
        raw_windows.append((start_char, end_char))

    # Merge overlapping or nearby windows
    merged_windows = _merge_windows(raw_windows, word_bounds, merge_distance)

    # Extract text and build CitationContext objects
    contexts = []
    idx = 0
    for start_char, end_char in merged_windows:
        window_text = raw_text[start_char:end_char]
        # Remove any markers from the window text itself
        clean_window = FRAGMENT_PATTERN.sub("", window_text).strip()
        # Collapse whitespace
        clean_window = re.sub(r"\s+", " ", clean_window)

        if len(clean_window.split()) >= min_length:
            contexts.append(CitationContext(
                doc_id=doc_id,
                index=idx,
                text=clean_window,
                start_char=start_char,
                end_char=end_char,
            ))
            idx += 1

    result = DocumentContexts(
        doc_id=doc_id,
        raw_text=raw_text,
        contexts=contexts,
        full_text_cleaned=FRAGMENT_PATTERN.sub("", raw_text).strip(),
    )

    return result


def _bisect_word_index(word_bounds: list[tuple[int, int]], char_pos: int) -> int:
    """Find the word index closest to a character position using binary search."""
    lo, hi = 0, len(word_bounds) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if word_bounds[mid][1] < char_pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _merge_windows(
    windows: list[tuple[int, int]],
    word_bounds: list[tuple[int, int]],
    merge_distance_words: int,
) -> list[tuple[int, int]]:
    """Merge overlapping or nearby windows.

    Two windows are merged if the gap between them contains fewer than
    merge_distance_words words.
    """
    if not windows:
        return []

    # Sort by start position
    sorted_windows = sorted(windows)
    merged = [sorted_windows[0]]

    for start, end in sorted_windows[1:]:
        prev_start, prev_end = merged[-1]

        if start <= prev_end:
            # Overlapping — merge
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            # Check gap in words
            gap_words = _count_words_in_range(word_bounds, prev_end, start)
            if gap_words <= merge_distance_words:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

    return merged


def _count_words_in_range(
    word_bounds: list[tuple[int, int]],
    start_char: int,
    end_char: int,
) -> int:
    """Count words whose start falls within [start_char, end_char)."""
    count = 0
    for ws, we in word_bounds:
        if ws >= end_char:
            break
        if ws >= start_char:
            count += 1
    return count


def extract_all_contexts(
    corpus: dict[str, str],
    **kwargs,
) -> dict[str, DocumentContexts]:
    """Extract citation contexts from every document in a corpus.

    Args:
        corpus: dict mapping doc_id -> raw text (with markers intact)
        **kwargs: Passed to extract_contexts()

    Returns:
        dict mapping doc_id -> DocumentContexts
    """
    results = {}
    n_total_contexts = 0

    for doc_id, raw_text in corpus.items():
        dc = extract_contexts(raw_text, doc_id=doc_id, **kwargs)
        results[doc_id] = dc
        n_total_contexts += dc.n_markers

    logger.info(
        "Extracted contexts from %d docs: %d total windows",
        len(results), n_total_contexts,
    )
    return results


def load_raw_corpus(docs_dir: Path) -> dict[str, str]:
    """Load corpus WITHOUT preprocessing — keeps <FRAGMENT_SUPPRESSED> markers."""
    corpus = {}
    for path in sorted(docs_dir.glob("*.txt")):
        corpus[path.name] = path.read_text(encoding="utf-8", errors="replace")
    return corpus
