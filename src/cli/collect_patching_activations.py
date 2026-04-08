"""
CLI entrypoint for collecting per-span activations on the patching dataset.

Loads a previously built patching dataset (JSON from build_patching_dataset.py)
and runs targeted activation extraction for each sample across conditions
(clean + blurred severities).  Saves .pt activation files, a Parquet output-
distribution table, and per-sample metadata JSON.

Output structure::

    <output_dir>/<model_name>/<sample_id>/<version>/
        lm_visual_layer_16.pt
        question_layer_16.pt
        ...
        output_logits.pt
    <output_dir>/<model_name>/output_distributions.parquet
    <output_dir>/<model_name>/<sample_id>/metadata.json

Usage:
  python -m src.cli.collect_patching_activations \\
      --patching_dataset /path/to/patching_dataset_vqa-v2.json \\
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
        description="Collect per-span activations on the patching dataset"
    )
    parser.add_argument(
        "--patching_dataset", type=str, required=True,
        help="Path to the patching dataset JSON (patching_dataset_<dataset>.json)",
    )
    parser.add_argument(
        "--vlm", type=str, required=True,
        help="HuggingFace model id (must match the one used to build the patching dataset)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Root output dir. Defaults to $VLM_UQ_USER_STORAGE_ROOT/patching/supervised_datasets/",
    )
    parser.add_argument(
        "--middle_layer", type=int, default=16,
        help="Middle LM layer index (default 16 for LLaVA-7B). "
             "Activations are cached at layers [mid, mid+1, mid+2].",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4,
        help="Max tokens to generate per sample (default 4)",
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
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print question, model answer, ground truth, and score for each sample",
    )
    return parser.parse_args()


def _model_dir_name(model_id: str) -> str:
    """Derive a filesystem-safe directory name from a HF model id."""
    return model_id.replace("/", "_").replace(".", "-")


def main() -> None:
    args = parse_args()

    from src.vlm.adapters import load_model
    from src.patching.collect_activations import (
        PatchingActivationConfig,
        collect_patching_activations,
    )

    # ---- load patching dataset ----
    print(f"Loading patching dataset from {args.patching_dataset}")
    with open(args.patching_dataset) as f:
        data = json.load(f)
    metadata = data["metadata"]
    records = data["samples"]
    print(f"Loaded {len(records)} records.")
    print(f"Calibrated sigmas: {metadata.get('calibration', {}).get('selected_sigmas')}")

    if args.max_samples is not None:
        records = records[: args.max_samples]
        print(f"Truncated to {len(records)} samples.")

    # ---- resolve output dir ----
    if args.output_dir is None:
        storage_root = os.environ.get(
            "VLM_UQ_USER_STORAGE_ROOT",
            os.path.join(os.environ.get("VLM_UQ_STORAGE_ROOT", "/projects/prjs2014"), os.environ.get("USER", "user")),
        )
        args.output_dir = os.path.join(storage_root, "patching", "supervised_datasets")
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = _model_dir_name(args.vlm)

    # ---- load model ----
    print(f"Loading model: {args.vlm}")
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)

    # ---- resolve dataset_id from metadata ----
    dataset_id = metadata.get("dataset_id", "vqa-v2")
    print(f"Dataset: {dataset_id}")

    # ---- run collection ----
    cfg = PatchingActivationConfig(
        dataset_id=dataset_id,
        middle_layer=args.middle_layer,
        max_new_tokens=args.max_new_tokens,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        verbose=args.verbose,
    )

    collect_patching_activations(
        model=model,
        processor=processor,
        device=device,
        vlm_adapter=vlm_adapter,
        patching_records=records,
        output_dir=args.output_dir,
        model_name=model_name,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
