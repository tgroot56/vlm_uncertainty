"""CLI entrypoint for the layer-sweep activation patching experiment.

Patches the final N non-visual prompt token positions at every LM layer and
measures score / top-1 / entropy recovery vs layer index.

Usage
-----
  python -m src.cli.run_layer_sweep_patching \\
      --patching_dataset /path/to/patching_dataset_vqa-v2.json \\
      --vlm llava-hf/llava-1.5-7b-hf \\
      --severity moderate

  # Re-plot from existing results without re-running:
  python -m src.cli.run_layer_sweep_patching \\
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
            "Layer-sweep activation patching: patch the final N prompt token "
            "activations across all LM layers and plot recovery vs layer."
        )
    )
    parser.add_argument(
        "--patching_dataset", type=str, required=True,
        help="Path to the patching dataset JSON produced by build_patching_dataset.",
    )
    parser.add_argument(
        "--vlm", type=str, required=True,
        help="HuggingFace model id (e.g. llava-hf/llava-1.5-7b-hf).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Output directory for results. "
            "Defaults to <storage_root>/patching/layer_sweep_results/<vlm>/<dataset>/."
        ),
    )
    parser.add_argument(
        "--n_prompt_tokens", type=int, default=3,
        help=(
            "Number of final non-visual prompt token positions to patch "
            "(default: 3). Only used when --position_set=final."
        ),
    )
    parser.add_argument(
        "--position_set", type=str, default="final",
        choices=["final", "question_visual"],
        help=(
            "Which token positions to patch at every layer. "
            "'final' = last N non-visual prompt tokens (default). "
            "'question_visual' = last question token + last LM visual token."
        ),
    )
    parser.add_argument(
        "--severity", type=str, default="moderate",
        choices=["mild", "moderate", "severe", "extreme", "all"],
        help="Which blur severity to test (default: moderate).",
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
        "--checkpoint_every", type=int, default=10,
        help="Save a checkpoint every K completed samples.",
    )
    parser.add_argument(
        "--plot_at_n_samples", type=int, default=None,
        help=(
            "If set, write a parquet + plot snapshot into "
            "<output_dir>/intermediate_at_<N>/ once N samples have completed. "
            "The main run continues to --max_samples."
        ),
    )
    parser.add_argument(
        "--qwen_prefill_empty_thinking",
        action="store_true",
        help=(
            "Qwen-only: append an empty <think></think> block to the prompt "
            "before generation, so generated step 0 is the first answer token."
        ),
    )
    parser.add_argument(
        "--save_clean_answer_probability",
        action="store_true",
        help=(
            "For every clean/degraded/patched row, save the probability assigned "
            "to the clean run's stripped generated answer tokens."
        ),
    )
    parser.add_argument(
        "--save_stripped_answer_softmax",
        action="store_true",
        help=(
            "For every row, save full softmax distributions only for the stripped "
            "generated-answer token positions. This can make outputs large."
        ),
    )
    parser.add_argument(
        "--stripped_answer_softmax_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Storage dtype for stripped-answer softmax payloads.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-(layer, position) patching results for each sample.",
    )
    parser.add_argument(
        "--plot_only", action="store_true",
        help="Skip patching and just re-plot from an existing results parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pathlib import Path

    from src.cli.plot_sweep_filtered import plot_layer_sweep_combined
    from src.patching.run_layer_sweep_patching import (
        LayerSweepConfig,
        run_layer_sweep_patching,
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
            storage_root, "patching", "layer_sweep_results",
            vlm_short, dataset_id,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- plot-only mode ----
    if args.plot_only:
        parquet_path = os.path.join(args.output_dir, "layer_sweep_results.parquet")
        print(f"Plot-only mode: reading {parquet_path}")
        plot_layer_sweep_combined(parquet_path, Path(args.output_dir), raw=False)
        plot_layer_sweep_combined(parquet_path, Path(args.output_dir), raw=True)
        return

    # ---- load model ----
    from src.vlm.adapters import load_model

    print(f"Loading model: {args.vlm}")
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)

    # ---- run layer-sweep patching ----
    cfg = LayerSweepConfig(
        dataset_id=dataset_id,
        n_prompt_tokens=args.n_prompt_tokens,
        severity=args.severity,
        checkpoint_every=args.checkpoint_every,
        verbose=args.verbose,
        position_set=args.position_set,
        plot_at_n_samples=args.plot_at_n_samples,
        qwen_prefill_empty_thinking=args.qwen_prefill_empty_thinking,
        save_clean_answer_probability=args.save_clean_answer_probability,
        save_stripped_answer_softmax=args.save_stripped_answer_softmax,
        stripped_answer_softmax_dtype=args.stripped_answer_softmax_dtype,
    )

    run_layer_sweep_patching(
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
