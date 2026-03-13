"""Full corpus entity extraction with resume support and dual-node parallelism."""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx

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

# Two DGX Spark nodes, each with its own GB10 GPU
OLLAMA_INSTANCES = [
    "http://localhost:11434",       # Node 1 (local)
    "http://192.168.200.13:11434",  # Node 2 (worker)
]


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


def _worker_loop(
    docs: list[Path],
    output_dir: Path,
    model: str,
    ollama_url: str,
    counter: dict,
    lock: Lock,
    total: int,
    start_time: float,
) -> list[str]:
    """Process a batch of documents using one Ollama instance."""
    failed = []
    with OllamaClient(base_url=ollama_url, timeout=120.0) as client:
        for doc_path in docs:
            try:
                result = extract_single_document(client, doc_path, model=model)
                out_path = output_dir / f"{doc_path.stem}.json"
                out_path.write_text(json.dumps(result, indent=2))

                with lock:
                    counter["done"] += 1
                    done = counter["done"]
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    if done % 20 == 0 or done <= 5:
                        logger.info(
                            "[%d/%d] %s — %.1f docs/min, ETA %.0f min",
                            done, total, doc_path.name, rate * 60, eta / 60,
                        )
            except Exception:
                logger.exception("Failed: %s on %s", doc_path.name, ollama_url)
                failed.append(doc_path.name)
    return failed


def run_extraction(
    docs_dir: Path,
    output_dir: Path,
    model: str = LLM_MODEL,
    resume: bool = True,
    ollama_urls: list[str] | None = None,
) -> None:
    """Extract entities using multiple Ollama instances on separate GPUs.

    Each instance runs on a different DGX Spark node with its own GPU,
    so true parallelism is achieved with no contention.
    """
    if ollama_urls is None:
        ollama_urls = OLLAMA_INSTANCES

    # Filter to reachable instances
    live_urls = []
    for url in ollama_urls:
        try:
            resp = httpx.get(f"{url}/api/tags", timeout=5)
            resp.raise_for_status()
            live_urls.append(url)
            logger.info("Ollama reachable: %s", url)
        except Exception:
            logger.warning("Ollama NOT reachable: %s (skipping)", url)
    if not live_urls:
        raise RuntimeError("No reachable Ollama instances!")

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

    if not remaining:
        logger.info("Nothing to extract.")
        return

    n_instances = len(live_urls)
    logger.info("Starting extraction with %d Ollama instances", n_instances)

    # Split docs evenly across instances
    batches = [[] for _ in range(n_instances)]
    for i, doc in enumerate(remaining):
        batches[i % n_instances].append(doc)

    counter = {"done": 0}
    lock = Lock()
    start_time = time.time()
    all_failed = []

    with ThreadPoolExecutor(max_workers=n_instances) as pool:
        futures = {}
        for batch, url in zip(batches, live_urls):
            logger.info("  %s: %d docs assigned", url, len(batch))
            future = pool.submit(
                _worker_loop,
                batch, output_dir, model, url,
                counter, lock, len(remaining), start_time,
            )
            futures[future] = url

        for future in as_completed(futures):
            url = futures[future]
            try:
                failed = future.result()
                all_failed.extend(failed)
            except Exception:
                logger.exception("Worker %s crashed", url)

    elapsed = time.time() - start_time
    n_done = counter["done"]
    logger.info(
        "Extraction complete: %d docs in %.1f min (%.1f docs/min), %d failed",
        n_done, elapsed / 60, n_done / (elapsed / 60) if elapsed > 0 else 0,
        len(all_failed),
    )


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
