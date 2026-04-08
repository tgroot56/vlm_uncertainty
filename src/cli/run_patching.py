"""
CLI entrypoint for running the activation patching experiment.

Loads a previously built patching dataset and pre-cached activations, then
runs patched forward passes to measure how swapping clean activations into
degraded inputs affects confidence and accuracy.

Usage:
  python -m src.cli.run_patching \
      --patching_dataset /path/to/patching_dataset_vqa-v2.json \
      --activations_dir /path/to/supervised_datasets/llava-hf_llava-1-5-7b-hf/ \
      --vlm llava-hf/llava-1.5-7b-hf
"""

import argparse
import json
import os

import torch

from ..storage_paths import configure_hf_cache_env

configure_hf_cache_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run activation patching experiment"
    )
    parser.add_argument(
        "--patching_dataset", type=str, required=True,
        help="Path to the patching dataset JSON (patching_dataset_<dataset>.json)",
    )
    parser.add_argument(
        "--activations_dir", type=str, required=True,
        help="Path to the cached activations directory "
             "(the model_name/ subdirectory from collect_patching_activations)",
    )
    parser.add_argument(
        "--vlm", type=str, required=True,
        help="HuggingFace model id (must match the one used to collect activations)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results. "
             "Defaults to <activations_dir>/patching_results/",
    )
    parser.add_argument(
        "--middle_layer", type=int, default=16,
        help="Middle LM layer index (default 16 for LLaVA-7B)",
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
        "--checkpoint_every", type=int, default=50,
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-condition patching results for each sample",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.vlm.adapters import load_model
    from src.patching.run_patching import (
        PatchingExecutionConfig,
        run_activation_patching,
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
        args.output_dir = os.path.join(args.activations_dir, "patching_results")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load model ----
    print(f"Loading model: {args.vlm}")
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)

    # ---- run patching ----
    cfg = PatchingExecutionConfig(
        dataset_id=dataset_id,
        middle_layer=args.middle_layer,
        checkpoint_every=args.checkpoint_every,
        verbose=args.verbose,
    )

    run_activation_patching(
        model=model,
        processor=processor,
        device=device,
        vlm_adapter=vlm_adapter,
        patching_records=records,
        activations_dir=args.activations_dir,
        output_dir=args.output_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
