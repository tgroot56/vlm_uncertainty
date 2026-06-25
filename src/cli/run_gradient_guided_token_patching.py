"""CLI for the qualitative gradient-guided token patching pilot.

This runs a single POPE sample through two greedy token-selection objectives:
answer recovery and clean-answer confidence recovery. Layer arguments are
human-facing decoder layer numbers; the runner converts them to code indices.
"""

from __future__ import annotations

import argparse
import os

import torch

from ..storage_paths import configure_hf_cache_env

configure_hf_cache_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run gradient-guided greedy token activation patching for a single "
            "qualitative POPE example."
        )
    )
    parser.add_argument(
        "--patching_dataset",
        type=str,
        required=True,
        help="Path to patching_dataset_pope.json.",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        required=True,
        help="HuggingFace model id, e.g. llava-hf/llava-1.5-7b-hf.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<storage_root>/patching/gradient_guided_token_patching/<vlm>/pope/random/sample_<sample_id>."
        ),
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default="795",
        help="POPE sample id to run (default: 795).",
    )
    parser.add_argument(
        "--severity",
        type=str,
        default="extreme",
        choices=["mild", "moderate", "severe", "extreme"],
        help="Corruption severity to use (default: extreme).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[7, 10, 13, 18],
        help=(
            "Requested decoder layer numbers to analyze. These are converted "
            "to zero-based code indices internally (default: 7 10 13 18)."
        ),
    )
    parser.add_argument(
        "--max_patches",
        type=int,
        default=8,
        help="Maximum greedy patches per layer/objective (default: 8).",
    )
    parser.add_argument(
        "--visual_top_k",
        type=int,
        default=64,
        help=(
            "At each greedy step, exactly evaluate all non-visual tokens plus "
            "the top-K visual tokens by gradient score (default: 64)."
        ),
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=[
            "answer_recovery",
            "confidence_recovery",
            "maximize_p_yes",
            "maximize_p_no",
        ],
        default=["answer_recovery", "confidence_recovery"],
        help="Greedy objectives to run.",
    )
    parser.add_argument(
        "--early_stop_window",
        type=int,
        default=0,
        help="Stop after this many recent steps satisfy the rolling threshold; 0 disables it.",
    )
    parser.add_argument(
        "--early_stop_abs_sum_threshold",
        type=float,
        default=0.0,
        help="Stop when the sum of absolute objective changes in the rolling window is below this value.",
    )
    parser.add_argument(
        "--require_corrupted_no",
        action="store_true",
        help="Require the selected corrupted example to be an incorrect no prediction.",
    )
    parser.add_argument(
        "--qwen_prefill_empty_thinking",
        action="store_true",
        help="Use the codebase's Qwen empty-thinking prefill so generation starts at the visible answer token.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--allow_non_yes",
        action="store_true",
        help=(
            "Disable the pilot guard that requires sample 795-style clean/correct "
            "yes examples. Future no-answer pilots should also update the recovery predicate."
        ),
    )
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip thesis-style qualitative figure generation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional progress details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.patching.run_gradient_guided_token_patching import (
        GradientGuidedTokenPatchingConfig,
        run_gradient_guided_token_patching,
    )
    from src.vlm.adapters import load_model

    if args.output_dir is None:
        storage_root = os.environ.get("VLM_UQ_STORAGE_ROOT", "/projects/prjs2014")
        vlm_short = args.vlm.replace("/", "_").replace(".", "-")
        args.output_dir = os.path.join(
            storage_root,
            "patching",
            "gradient_guided_token_patching",
            vlm_short,
            "pope",
            "random",
            f"sample_{args.sample_id}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.vlm}")
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)

    cfg = GradientGuidedTokenPatchingConfig(
        sample_id=str(args.sample_id),
        severity=args.severity,
        human_layers=list(args.layers),
        max_patches=args.max_patches,
        visual_top_k=args.visual_top_k,
        require_clean_yes=not args.allow_non_yes,
        verbose=args.verbose,
        make_figures=not args.no_figures,
        run_types=list(args.objectives),
        early_stop_window=args.early_stop_window,
        early_stop_abs_sum_threshold=args.early_stop_abs_sum_threshold,
        require_corrupted_no=args.require_corrupted_no,
        qwen_prefill_empty_thinking=args.qwen_prefill_empty_thinking,
    )

    run_gradient_guided_token_patching(
        model=model,
        processor=processor,
        device=device,
        vlm_adapter=vlm_adapter,
        patching_dataset=args.patching_dataset,
        output_dir=args.output_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
