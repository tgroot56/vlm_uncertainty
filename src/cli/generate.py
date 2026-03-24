"""
CLI entrypoint for generating supervised UQ datasets.

End-to-end pipeline:
- load task dataset
- load VLM
- generate model outputs
- extract hidden / confidence features
- construct and save supervised UQ dataset
"""

import torch
import argparse

from ..storage_paths import configure_hf_cache_env, default_supervision_output_root

configure_hf_cache_env()

from ..pipelines.generate_dataset import run_generate_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate supervised uncertainty dataset from VLM hidden features"
    )

    # ---- Core identifiers ----
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset identifier (supported: pope, vqa-v2, coco-qa-vi, imagenet-r)"
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
        default=default_supervision_output_root(),
        help="Root directory to store generated supervised datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splits and generation"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="Optional dataset split override. Useful for datasets like POPE Full "
             "with multiple splits such as adversarial, popular, or random."
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
    parser.add_argument(
        "--vision_features_only",
        action="store_true",
        help="Qwen-only mode: run only vision-encoder forward passes and save "
             "a separate supervision_dataset_visual.pt using labels from an existing "
             "supervision_dataset.pt for the same run."
    )
    parser.add_argument(
        "--source_supervision_path",
        type=str,
        default=None,
        help="Optional path to an existing supervision_dataset.pt to reuse labels/rows "
             "when --vision_features_only is enabled. Defaults to the matching run folder."
    )
    parser.add_argument(
        "--vision_batch_size",
        type=int,
        default=4,
        help="Batch size for Qwen vision-only extraction. Only used with --vision_features_only."
    )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=1,
    #     help="Batch size for VLM inference (keep small for large models)"
    # )

    # imagenet-r specific options
    parser.add_argument(
        "--k_options",
        type=int,
        default=4,
        help="(ImageNet-R) Number of multiple-choice options K (default: 4)."
    )

    parser.add_argument(
        "--clip_negatives",
        action="store_true",
        help="(ImageNet-R) Use CLIP-based hard negatives instead of random distractors."
    )

    parser.add_argument(
        "--clip_mode",
        type=str,
        default="gt_to_text",
        choices=["gt_to_text", "image_to_text"],
        help="(ImageNet-R) CLIP negative sampling mode. "
            "gt_to_text is cheaper; image_to_text is harder but slower."
    )

    parser.add_argument(
        "--clip_top_pool",
        type=int,
        default=30,
        help="(ImageNet-R) Sample distractors from the top-N CLIP-similar labels (default: 30)."
    )

    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None
    )
    parser.add_argument(
        "--all_lm_mean_layers",
        type=str,
        nargs="+",
        default=None,
        choices=["visual", "question", "answer"],
        help=(
            "LLaVA layer-sweep mode: extract mean-pooled LM features for every LM layer "
            "for the selected spans. When set, the generator disables the standard default "
            "feature set and stores only the requested per-layer mean features."
        ),
    )






    return parser.parse_args()



def main():
    print("Starting supervised UQ dataset generation pipeline...")
    args = parse_args()

    # 1 Load the relevant model
    run_generate_dataset(args)

if __name__ == "__main__":
    main()
