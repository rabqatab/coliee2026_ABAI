"""Full corpus entity extraction with resume support."""
import json
import logging
import sys
import time
from pathlib import Path

from graphrag.config import (
    TRAIN_DOCS_DIR,
    TEST_DOCS_DIR,
    EXTRACTIONS_DIR,
    LLM_MODEL,
)
from graphrag.ollama_client import OllamaClient
from graphrag.preprocess import preprocess
from graphrag.extract_regex import extract_statutes, extract_judges, extract_outcome
from graphrag.extract_llm import extract_entities_llm
from graphrag.normalize import merge_regex_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def extract_single_document(
    client: OllamaClient,
    doc_path: Path,
    model: str = LLM_MODEL,
) -> dict:
    """Run full extraction pipeline on a single document."""
    raw_text = doc_path.read_text(encoding="utf-8", errors="replace")
    clean_text = preprocess(raw_text)

    # Pass 1: Regex
    regex_result = {
        "statutes": extract_statutes(clean_text),
        "judges": extract_judges(clean_text),
        "outcome": extract_outcome(clean_text),
    }

    # Pass 2: LLM
    llm_result = extract_entities_llm(client, clean_text, model=model)

    # Merge
    merged = merge_regex_llm(regex_result, llm_result)
    merged["doc_id"] = doc_path.stem
    merged["word_count"] = len(clean_text.split())

    return merged


def run_extraction(
    docs_dir: Path,
    output_dir: Path,
    model: str = LLM_MODEL,
    resume: bool = True,
) -> None:
    """Extract entities from all documents in a directory.

    Saves one JSON file per document for resume support.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_paths = sorted(docs_dir.glob("*.txt"))
    logger.info("Found %d documents in %s", len(doc_paths), docs_dir)

    # Check what's already done
    if resume:
        done = {p.stem for p in output_dir.glob("*.json")}
        remaining = [p for p in doc_paths if p.stem not in done]
        logger.info("Already extracted: %d, remaining: %d", len(done), len(remaining))
    else:
        remaining = doc_paths

    with OllamaClient() as client:
        start_time = time.time()
        for i, doc_path in enumerate(remaining):
            try:
                result = extract_single_document(client, doc_path, model=model)
                out_path = output_dir / f"{doc_path.stem}.json"
                out_path.write_text(json.dumps(result, indent=2))

                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] %s — %.1f docs/min, ETA %.0f min",
                    i + 1,
                    len(remaining),
                    doc_path.name,
                    rate * 60,
                    eta / 60,
                )
            except Exception:
                logger.exception("Failed to extract %s", doc_path.name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract entities from corpus")
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="train",
    )
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    dirs = []
    if args.split in ("train", "both"):
        dirs.append(("train", TRAIN_DOCS_DIR))
    if args.split in ("test", "both"):
        dirs.append(("test", TEST_DOCS_DIR))

    for split_name, docs_dir in dirs:
        out_dir = EXTRACTIONS_DIR / split_name
        logger.info("=== Extracting %s split ===", split_name)
        run_extraction(docs_dir, out_dir, model=args.model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
