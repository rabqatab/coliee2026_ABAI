"""Text preprocessing for legal case documents."""
import re
from pathlib import Path


def preprocess(text: str) -> str:
    """Clean a raw case document for extraction."""
    # Remove FRAGMENT_SUPPRESSED placeholders
    text = re.sub(r"<FRAGMENT_SUPPRESSED>", "", text)
    # Remove end-of-document markers
    text = re.sub(r"\[End of document\]", "", text)
    # Rejoin broken statute names (lines starting with lowercase after short line)
    text = re.sub(r"(\w)\n([a-z])", r"\1 \2", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_paragraphs(text: str) -> list[tuple[int, str]]:
    """Extract numbered paragraphs from a case document.

    Returns list of (paragraph_number, paragraph_text) tuples.
    """
    # Match [N] paragraph markers
    parts = re.split(r"\[(\d+)\]", text)
    paragraphs = []
    for i in range(1, len(parts), 2):
        para_num = int(parts[i])
        para_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if para_text:
            paragraphs.append((para_num, para_text))
    return paragraphs


def chunk_for_llm(text: str, max_words: int = 8000, overlap_words: int = 200) -> list[str]:
    """Split text into chunks suitable for LLM processing.

    Documents under max_words are returned as a single chunk.
    Larger documents are split at paragraph boundaries with overlap.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Keep last paragraph as overlap
            overlap_paras = []
            overlap_count = 0
            for p in reversed(current_chunk):
                pw = len(p.split())
                if overlap_count + pw > overlap_words:
                    break
                overlap_paras.insert(0, p)
                overlap_count += pw
            current_chunk = overlap_paras
            current_words = overlap_count
        current_chunk.append(para)
        current_words += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def load_corpus(docs_dir: Path) -> dict[str, str]:
    """Load all .txt files from a directory into a dict keyed by filename."""
    corpus = {}
    for path in sorted(docs_dir.glob("*.txt")):
        corpus[path.name] = path.read_text(encoding="utf-8", errors="replace")
    return corpus
