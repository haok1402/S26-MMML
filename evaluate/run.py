"""
CLI entry point for VLM swing prediction evaluation.

Usage:
    HF_HOME=/tmp/hf-home python -m evaluate.run --model qwen3-vl-8b --strategy zero-shot
    HF_HOME=/tmp/hf-home python -m evaluate.run --model qwen3-vl-8b --strategy reasoning
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf-home")

from evaluate.datasets import load_examples
from evaluate.inference import run_inference
from evaluate.metrics import compute_metrics, compute_zone_metrics, print_metrics, print_zone_metrics


def main():
    """
    Run VLM swing prediction evaluation and save results.
    """
    parser = argparse.ArgumentParser(description="Evaluate VLM on baseball swing prediction")
    parser.add_argument("--data", type=str, default="workspace/datasets/evaluation.parquet")
    parser.add_argument("--model", type=str, default="qwen3-vl-8b")
    parser.add_argument("--strategy", type=str, default="zero-shot", choices=["zero-shot", "naive-reasoning", "structured-reasoning", "3-history", "zone-ocr"])
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples (for debugging)")
    args = parser.parse_args()

    print(f"Loading examples from {args.data}...")
    examples = load_examples(Path(args.data))
    print(f"Loaded {len(examples)} pitch examples with images")

    if args.max_examples is not None:
        examples = examples[:args.max_examples]
        print(f"Limited to {len(examples)} examples")

    print(f"Running inference: model={args.model}, strategy={args.strategy}")
    results = run_inference(examples, model=args.model, strategy=args.strategy)

    if args.strategy == "zone-ocr":
        metrics = compute_zone_metrics(examples, results)
        print_zone_metrics(metrics)
    else:
        metrics = compute_metrics(examples, results)
        print_metrics(metrics)

    out_dir = Path("workspace", "results", args.model, args.strategy)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(out_dir, "metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics written to {metrics_path}")

    predictions_path = Path(out_dir, "predictions.jsonl")
    with open(predictions_path, "w") as f:
        for ex, res in zip(examples, results):
            entry = dict()
            entry["filename"] = ex["filename"]
            entry["ground_truth"] = ex["zone"] if args.strategy == "zone-ocr" else ex["swing"]
            entry["prediction"] = res["prediction"]
            entry["in_zone"] = ex["in_zone"]
            entry["confidence"] = res["confidence"]
            entry["raw_output"] = res["raw_output"]
            f.write(json.dumps(entry) + "\n")
    print(f"Predictions written to {predictions_path}")


if __name__ == "__main__":
    main()
