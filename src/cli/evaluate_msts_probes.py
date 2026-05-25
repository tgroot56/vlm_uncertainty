from __future__ import annotations

import argparse

from src.generalization.evaluate_msts import (
    DEFAULT_MODEL_TYPES,
    DEFAULT_MSTS_RUN_DIR,
    DEFAULT_OUTPUT_ROOT,
    evaluate_msts_probes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained LLaVA probes on the manually annotated MSTS dataset."
    )
    parser.add_argument("--msts_run_dir", type=str, default=DEFAULT_MSTS_RUN_DIR)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model_types", type=str, nargs="+", default=list(DEFAULT_MODEL_TYPES))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_ece_bins", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate_msts_probes(
        msts_run_dir=args.msts_run_dir,
        output_root=args.output_root,
        model_types=args.model_types,
        batch_size=args.batch_size,
        device=args.device,
        num_ece_bins=args.num_ece_bins,
    )
    print("MSTS probe evaluation complete")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
