"""
CLI entrypoint for generating supervised UQ datasets.

End-to-end pipeline:
- load task dataset
- load VLM
- generate model outputs
- extract hidden / confidence features
- construct and save supervised UQ dataset
"""

import argparse
from ..pipelines.generate_dataset import run_generate_dataset
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate supervised uncertainty dataset from VLM hidden features"
    )

    # ---- Core identifiers ----
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset identifier (e.g. vqa2, advqa, coqa)"
    )
    parser.add_argument(
        "--vlm",
        type=str,
        required=True,
        help="VLM identifier (e.g. llava-1.6-7b, blip2-flan-t5)"
    )

    # ---- Output & reproducibility ----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/supervised_datasets",
        help="Root directory to store generated supervised datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splits and generation"
    )

    # ---- Runtime control ----
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (e.g. cuda, cuda:0, cpu). Defaults to auto-detect."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=1,
    #     help="Batch size for VLM inference (keep small for large models)"
    # )

    return parser.parse_args()



def main():
    print("Starting supervised UQ dataset generation pipeline...")
    args = parse_args()

    # 1 Load the relevant model
    run_generate_dataset(args)

if __name__ == "__main__":
    main()

