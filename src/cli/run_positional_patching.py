"""
CLI entrypoint for the single-token positional activation patching experiment.

Loads a previously built patching dataset, then for each sample patches one
prompt token position at a time at a single layer and measures how recovery
varies with token position. By default, the sweep includes every non-visual
prompt token plus a sparse sample of visual tokens.

Usage:
  python -m src.cli.run_positional_patching \
      --patching_dataset /path/to/patching_dataset_vqa-v2.json \
      --vlm llava-hf/llava-1.5-7b-hf \
      --layer 17

  # Re-plot from existing results:
  python -m src.cli.run_positional_patching \
      --patching_dataset /path/to/patching_dataset_vqa-v2.json \
      --vlm llava-hf/llava-1.5-7b-hf \
      --output_dir /path/to/existing/results \
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
        description="Run single-token positional activation patching experiment"
    )
    parser.add_argument(
        "--patching_dataset", type=str, required=True,
        help="Path to the patching dataset JSON (patching_dataset_<dataset>.json)",
    )
    parser.add_argument(
        "--vlm", type=str, required=True,
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results. "
             "Defaults to <storage_root>/patching/positional_patching_results/<vlm>/<dataset>/",
    )
    parser.add_argument(
        "--layer", type=int, default=17,
        help="Target LM layer index for patching (default 17)",
    )
    parser.add_argument(
        "--severity", type=str, default="moderate",
        choices=["mild", "moderate", "severe", "extreme", "all"],
        help="Which blur severity to test (default: moderate)",
    )
    parser.add_argument(
        "--sample_visual_every", type=int, default=10,
        help=(
            "Include every Nth visual token position in the positional sweep. "
            "Set 0 to exclude visual tokens from the strided sample (default: 10)."
        ),
    )
    parser.add_argument(
        "--include_visual_last", type=int, default=10,
        help=(
            "Include the last K visual token positions in addition to any strided "
            "visual sampling. Set 0 to disable (default: 10)."
        ),
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Process only the first N samples (default: all)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=10,
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-position patching results for each sample",
    )
    parser.add_argument(
        "--plot_only", action="store_true",
        help="Skip patching, just re-plot from existing results parquet",
    )
    parser.add_argument(
        "--qwen_prefill_empty_thinking", action="store_true",
        help=(
            "Append <think>\\n</think>\\n preamble to the Qwen prompt so the model "
            "answers directly (comparable to LLaVA). Adds 4 new positions to the "
            "assistant_marker span and shifts position -1 to the final preamble token."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pathlib import Path

    from src.cli.plot_sweep_filtered import plot_positional_sweep_combined
    from src.patching.run_positional_patching import (
        PositionalPatchingConfig,
        run_positional_patching,
    )

    # ---- load patching dataset ----
    print(f"Loading patching dataset from {args.patching_dataset}")
    with open(args.patching_dataset) as f:
        data = json.load(f)
    metadata = data["metadata"]
    records = data["samples"]
    print(f"Loaded {len(records)} records.")

    if args.max_samples is not None:
        records = records[:args.max_samples]
        print(f"Truncated to {len(records)} samples.")

    dataset_id = metadata.get("dataset_id", "vqa-v2")
    print(f"Dataset: {dataset_id}")

    # ---- resolve output dir ----
    if args.output_dir is None:
        storage_root = os.environ.get("VLM_UQ_STORAGE_ROOT", "/projects/prjs2014")
        vlm_short = args.vlm.replace("/", "_").replace(".", "-")
        args.output_dir = os.path.join(
            storage_root, "patching", "positional_patching_results",
            vlm_short, dataset_id,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- plot-only mode ----
    if args.plot_only:
        parquet_path = os.path.join(args.output_dir, "positional_patching_results.parquet")
        print(f"Plot-only mode: reading {parquet_path}")
        plot_positional_sweep_combined(parquet_path, Path(args.output_dir), raw=False)
        plot_positional_sweep_combined(parquet_path, Path(args.output_dir), raw=True)
        return

    # ---- load model ----
    from src.vlm.adapters import load_model

    print(f"Loading model: {args.vlm}")
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)

    # ---- run positional patching ----
    cfg = PositionalPatchingConfig(
        dataset_id=dataset_id,
        layer=args.layer,
        severity=args.severity,
        sample_visual_every=args.sample_visual_every,
        include_visual_last=args.include_visual_last,
        checkpoint_every=args.checkpoint_every,
        verbose=args.verbose,
        qwen_prefill_empty_thinking=args.qwen_prefill_empty_thinking,
    )

    run_positional_patching(
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
