"""CLI entry point for the COLIEE Task 1 pipeline.

Usage:
    uv run python -m coliee_task1 train
    uv run python -m coliee_task1 predict
    uv run python -m coliee_task1 evaluate -p predictions.json -g gold.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path


def cmd_train(args: argparse.Namespace) -> None:
    """Run the full training pipeline."""
    from coliee_task1.pipeline import run_train_pipeline
    run_train_pipeline(
        finetune=not args.no_finetune,
        use_cache=not args.no_cache,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    """Run prediction on test queries."""
    from coliee_task1.pipeline import run_predict_pipeline
    output = Path(args.output) if args.output else None
    run_predict_pipeline(output_path=output)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate predictions against gold labels."""
    from coliee_task1.utils.metrics import micro_f1

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"Error: {pred_path} not found", file=sys.stderr)
        sys.exit(1)

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"Error: {gold_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(pred_path) as f:
        predictions = json.load(f)
    with open(gold_path) as f:
        gold = json.load(f)

    metrics = micro_f1(predictions, gold)
    print(f"Queries: {len(set(predictions) & set(gold))}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="coliee-task1",
        description="COLIEE 2026 Task 1: Legal Case Retrieval Pipeline",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Run the full training pipeline")
    train_parser.add_argument(
        "--no-finetune", action="store_true",
        help="Skip fine-tuning neural models (CPU-friendly mode)",
    )
    train_parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable stage caching (recompute everything)",
    )

    # predict
    predict_parser = subparsers.add_parser("predict", help="Run prediction on test queries")
    predict_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for predictions JSON",
    )
    predict_parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable stage caching (recompute everything)",
    )

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions against gold labels")
    eval_parser.add_argument(
        "--predictions", "-p", required=True,
        help="Path to predictions JSON file",
    )
    eval_parser.add_argument(
        "--gold", "-g", required=True,
        help="Path to gold labels JSON file",
    )

    args = parser.parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
