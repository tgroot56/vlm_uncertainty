"""
CLI entrypoint for building the activation-patching paired dataset.

Two-phase pipeline:
  Phase 1 — Calibration: sweep blur sigma on a held-out subset to find
            mild / moderate / severe levels (or load a previous calibration).
  Phase 2 — Build:       run VLM on clean + blurred images for every sample
            and save the paired dataset as JSON.

Usage examples:

  # Full run (calibrate + build) on VQA-v2 with LLaVA:
  python -m src.cli.build_patching_dataset \
      --dataset vqa-v2 --vlm llava-hf/llava-1.5-7b-hf \
      --max_samples 1000 --calibration_samples 100

  # Reuse a previous calibration:
  python -m src.cli.build_patching_dataset \
      --dataset vqa-v2 --vlm llava-hf/llava-1.5-7b-hf \
      --max_samples 2000 \
      --load_calibration /path/to/calibration.json

  # Calibration-only (no dataset build):
  python -m src.cli.build_patching_dataset \
      --dataset vqa-v2 --vlm llava-hf/llava-1.5-7b-hf \
      --calibration_only --calibration_samples 100
"""

import argparse
import os

import torch

from ..storage_paths import configure_hf_cache_env, default_supervision_output_root

configure_hf_cache_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired clean-degraded dataset for activation patching"
    )

    # ---- Core identifiers ----
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset identifier (e.g. vqa-v2, pope, coco-qa-vi)",
    )
    parser.add_argument(
        "--vlm", type=str, required=True,
        help="HuggingFace model id (e.g. llava-hf/llava-1.5-7b-hf)",
    )
    parser.add_argument(
        "--dataset_split", type=str, default=None,
        help="Override the default dataset split",
    )

    # ---- Sample counts ----
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples for the main dataset build (None = full split)",
    )
    parser.add_argument(
        "--calibration_samples", type=int, default=100,
        help="Number of held-out samples for blur calibration (default 100)",
    )
    


    # ---- Calibration control ----
    parser.add_argument(
        "--load_calibration", type=str, default=None,
        help="Path to a previously saved calibration JSON. Skips calibration phase.",
    )
    parser.add_argument(
        "--calibration_only", action="store_true",
        help="Run calibration only; do not build the full dataset.",
    )
    
    parser.add_argument(
        "--sigma_min", type=int, default=1,
        help="Minimum sigma for calibration sweep (default 1)",
    )
    parser.add_argument(
        "--sigma_max", type=int, default=40,
        help="Maximum sigma for calibration sweep (default 40)",
    )

    # ---- Output ----
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory. Defaults to <storage_root>/patching/<vlm_short>/<dataset>",
    )
    parser.add_argument(
        "--no_save_images", action="store_true",
        help="Skip saving blurred/clean images to disk (only store metadata in JSON)",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=50,
        help="Save checkpoint every N samples (default 50)",
    )

    # ---- Runtime ----
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: auto-detect cuda)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset sampling",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4,
        help="Max tokens to generate per sample (default 4)",
    )

    return parser.parse_args()


def _vlm_short_name(model_id: str) -> str:
    """Derive a short directory-safe name from a HF model id."""
    return model_id.replace("/", "_").replace(".", "-")


def main() -> None:
    args = parse_args()

    if args.calibration_only and args.max_samples is None:
        args.max_samples = args.calibration_samples

    # Lazy imports so --help is fast
    from src.data.load import load_dataset_prepared
    from src.vlm.adapters import get_vlm_adapter, load_model
    from src.patching.calibrate_blur import (
        calibrate_blur_sigmas,
        load_calibration,
        save_calibration,
        save_blur_examples,
    )
    from src.patching.build_dataset import build_patching_dataset

    # ---- resolve output directory ----
    if args.output_dir is None:
        root = default_supervision_output_root()
        args.output_dir = os.path.join(
            root, "patching", _vlm_short_name(args.vlm), args.dataset,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load model ----
    print(f"Loading model: {args.vlm}")
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)

    # ---- load dataset ----
    print(f"Loading dataset: {args.dataset}")
    all_samples = load_dataset_prepared(
        dataset_id=args.dataset,
        split=args.dataset_split,
        seed_offset=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(all_samples)} samples")

    # ---- calibration phase ----
    if args.load_calibration:
        print(f"Loading calibration from {args.load_calibration}")
        calibration = load_calibration(args.load_calibration)
        print(f"Loaded sigmas: {calibration.selected_sigmas}")
    else:
        # Use the first N samples for calibration, remaining for build
        n_cal = min(args.calibration_samples, len(all_samples))
        cal_samples = all_samples[:n_cal]
        print(f"Running calibration on {n_cal} samples …")

        calibration = calibrate_blur_sigmas(
            model=model,
            processor=processor,
            device=device,
            vlm_adapter=vlm_adapter,
            samples=cal_samples,
            sigma_range=range(args.sigma_min, args.sigma_max + 1),
            max_new_tokens=args.max_new_tokens,
            dataset_id=args.dataset,
        )

        cal_path = os.path.join(args.output_dir, "calibration.json")
        save_calibration(calibration, cal_path)

        # Save visual examples of each blur severity for inspection.
        save_blur_examples(
            result=calibration,
            samples=cal_samples,
            output_dir=args.output_dir,
        )

    if args.calibration_only:
        print("Calibration-only mode — done.")
        return

    # ---- build phase ----
    # When calibration was run inline, use only the non-calibration samples
    # for the main build to avoid data leakage.
    if args.load_calibration:
        build_samples = all_samples
    else:
        n_cal = min(args.calibration_samples, len(all_samples))
        build_samples = all_samples[n_cal:]
        if len(build_samples) == 0:
            print("WARNING: no samples left after reserving calibration set. "
                  "Increase --max_samples or decrease --calibration_samples.")
            return

    print(f"Building dataset with {len(build_samples)} samples …")
    build_patching_dataset(
        model=model,
        processor=processor,
        device=device,
        vlm_adapter=vlm_adapter,
        samples=build_samples,
        calibration=calibration,
        output_dir=args.output_dir,
        model_id=args.vlm,
        dataset_id=args.dataset,
        save_images=not args.no_save_images,
        max_new_tokens=args.max_new_tokens,
        checkpoint_every=args.checkpoint_every,
    )

    print("Done.")


if __name__ == "__main__":
    main()
