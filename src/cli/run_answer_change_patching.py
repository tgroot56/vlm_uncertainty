"""CLI entrypoint for the answer-change activation patching experiment.

Patches the final prompt token at specified layers and classifies the
answer transition to distinguish confidence modulation from answer reranking.

Usage
-----
  # Single layer:
  python -m src.cli.run_answer_change_patching \\
      --patching_dataset /path/to/patching_dataset_vqa-v2.json \\
      --vlm llava-hf/llava-1.5-7b-hf \\
      --layers 31

  # Multi-layer sweep:
  python -m src.cli.run_answer_change_patching \\
      --patching_dataset /path/to/patching_dataset_vqa-v2.json \\
      --vlm llava-hf/llava-1.5-7b-hf \\
      --layers 14 17 20 24 28 31

  # Re-plot from existing results:
  python -m src.cli.run_answer_change_patching \\
      --patching_dataset /path/to/patching_dataset_vqa-v2.json \\
      --vlm llava-hf/llava-1.5-7b-hf \\
      --output_dir /path/to/existing/results \\
      --plot_only
"""

import argparse
import json
import os

import torch

from ..storage_paths import configure_hf_cache_env

configure_hf_cache_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Answer-change analysis: patch the final prompt token at specified "
            "layers and classify transitions (wrong->same wrong, wrong->correct, etc.)."
        )
    )
    parser.add_argument(
        "--patching_dataset", type=str, required=True,
        help="Path to the patching dataset JSON.",
    )
    parser.add_argument(
        "--vlm", type=str, required=True,
        help="HuggingFace model id (e.g. llava-hf/llava-1.5-7b-hf).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Output directory for results. "
            "Defaults to <storage_root>/patching/answer_change_results/<vlm>/<dataset>/."
        ),
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[31],
        help=(
            "One or more decoder layer indices to patch (space-separated). "
            "Default: 31 (last layer of LLaVA-1.5-7b)."
        ),
    )
    parser.add_argument(
        "--severity", type=str, default="moderate",
        choices=["mild", "moderate", "severe", "all"],
        help="Which blur severity to test (default: moderate).",
    )
    parser.add_argument(
        "--final_prompt_offset", type=int, default=-1,
        help=(
            "Which final non-visual prompt token to patch. "
            "-1 = last (default), -2 = second-to-last, etc."
        ),
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Process only the first N samples (default: all).",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=25,
        help="Save a checkpoint every K completed samples.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample per-layer transition details.",
    )
    parser.add_argument(
        "--plot_only", action="store_true",
        help="Skip patching and just re-plot from an existing results parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.patching.run_answer_change_patching import (
        AnswerChangeConfig,
        plot_answer_change_results,
        run_answer_change_patching,
    )

    # ---- load patching dataset ----
    print(f"Loading patching dataset from {args.patching_dataset}")
    with open(args.patching_dataset) as f:
        data = json.load(f)
    metadata = data["metadata"]
    records = data["samples"]
    print(f"Loaded {len(records)} records.")

    if args.max_samples is not None:
        records = records[: args.max_samples]
        print(f"Truncated to {len(records)} samples.")

    dataset_id = metadata.get("dataset_id", "vqa-v2")
    print(f"Dataset: {dataset_id}")

    # ---- resolve output dir ----
    if args.output_dir is None:
        storage_root = os.environ.get("VLM_UQ_STORAGE_ROOT", "/projects/prjs2014")
        vlm_short = args.vlm.replace("/", "_").replace(".", "-")
        args.output_dir = os.path.join(
            storage_root, "patching", "answer_change_results",
            vlm_short, dataset_id,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- plot-only mode ----
    if args.plot_only:
        parquet_path = os.path.join(args.output_dir, "answer_change_results.parquet")
        print(f"Plot-only mode: reading {parquet_path}")
        plot_answer_change_results(parquet_path, args.output_dir)
        return

    # ---- load model ----
    from src.vlm.adapters import load_model

    print(f"Loading model: {args.vlm}")
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)

    # ---- run answer-change patching ----
    cfg = AnswerChangeConfig(
        dataset_id=dataset_id,
        layers=args.layers,
        severity=args.severity,
        final_prompt_offset=args.final_prompt_offset,
        checkpoint_every=args.checkpoint_every,
        verbose=args.verbose,
    )

    run_answer_change_patching(
        model=model,
        processor=processor,
        device=device,
        vlm_adapter=vlm_adapter,
        patching_records=records,
        output_dir=args.output_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
