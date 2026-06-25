"""CLI entrypoint for the softmax-changes proof-of-concept.

Scans a shuffled pool of patching-dataset samples, retains those that satisfy
the main layer-sweep cohort filter (clean answer correct, corrupted answer
incorrect), and for each retained sample records how the output softmax top-5
changes across the layer sweep (clean baseline, corrupted baseline, and the
patched run at every LM layer — final prompt token, ``position_offset == -1``).

Usage
-----
  python -m src.cli.run_softmax_changes \\
      --patching_dataset /path/to/patching_dataset_coco-qa-vi.json \\
      --vlm llava-hf/llava-1.5-7b-hf \\
      --severity extreme --n_samples 10

  # Qwen (matches the production layer sweep prefill convention):
  python -m src.cli.run_softmax_changes \\
      --patching_dataset /path/to/patching_dataset_coco-qa-vi.json \\
      --vlm Qwen/Qwen3.5-9B --severity extreme --n_samples 10 \\
      --qwen_prefill_empty_thinking
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
            "Softmax-changes POC: track clean/corrupted/patched output top-5 "
            "softmax across the layer sweep for a few filter-passing samples."
        )
    )
    parser.add_argument(
        "--patching_dataset", type=str, required=True,
        help="Path to the patching dataset JSON produced by build_patching_dataset.",
    )
    parser.add_argument(
        "--vlm", type=str, required=True,
        help="HuggingFace model id (e.g. llava-hf/llava-1.5-7b-hf or Qwen/Qwen3.5-9B).",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help=(
            "Output JSON path. Defaults to "
            "<storage_root>/patching/softmax_changes/<vlm>/<dataset>/softmax_changes.json."
        ),
    )
    parser.add_argument(
        "--severity", type=str, default="extreme",
        choices=["mild", "moderate", "severe", "extreme"],
        help="Which blur severity to corrupt with (default: extreme, matching coco-qa-vi).",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of filter-passing samples to retain (default: 10).",
    )
    parser.add_argument(
        "--max_candidates", type=int, default=100,
        help="Max samples to scan looking for retained ones (default: 100).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for candidate order.")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k tokens to record per run.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Override the dataset-specific generation budget.",
    )
    parser.add_argument(
        "--qwen_prefill_empty_thinking", action="store_true",
        help="Qwen-only: append an empty <think></think> block before generation.",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample keep/skip decisions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.patching.run_softmax_changes import (
        SoftmaxChangesConfig,
        run_softmax_changes,
    )

    print(f"Loading patching dataset from {args.patching_dataset}")
    with open(args.patching_dataset) as f:
        data = json.load(f)
    metadata = data["metadata"]
    records = data["samples"]
    dataset_id = metadata.get("dataset_id", "coco-qa-vi")
    print(f"Loaded {len(records)} records. Dataset: {dataset_id}")

    # ---- resolve output path ----
    vlm_short = args.vlm.replace("/", "_").replace(".", "-")
    if args.output_path is None:
        storage_root = os.environ.get("VLM_UQ_STORAGE_ROOT", "/projects/prjs2014")
        dataset_subdir = "pope/random" if dataset_id == "pope" else dataset_id
        args.output_path = os.path.join(
            storage_root, "patching", "softmax_changes",
            vlm_short, dataset_subdir, "softmax_changes.json",
        )
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # ---- load model ----
    from src.vlm.adapters import load_model

    print(f"Loading model: {args.vlm}")
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)

    cfg = SoftmaxChangesConfig(
        dataset_id=dataset_id,
        severity=args.severity,
        n_samples=args.n_samples,
        max_candidates=args.max_candidates,
        seed=args.seed,
        top_k=args.top_k,
        max_new_tokens_override=args.max_new_tokens,
        qwen_prefill_empty_thinking=args.qwen_prefill_empty_thinking,
        verbose=args.verbose,
    )

    run_softmax_changes(
        model=model,
        processor=processor,
        device=device,
        vlm_adapter=vlm_adapter,
        patching_records=records,
        output_path=args.output_path,
        model_label=args.vlm,
        model_id=args.vlm,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
