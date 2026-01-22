"""Main script for running ImageNet-R experiments with LLaVA."""

import argparse
from data_loader import load_dataset_by_name, build_mc_prompt_imagenet_r
from utils.model_loader import load_llava_model
from utils.inference import predict_letter_and_logits
from utils.experiment import run_imagenet_r_experiment, save_results
from generate_dataset import get_data_split


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ImageNet-R classification experiments with LLaVA"
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model identifier"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="axiong/imagenet-r",
        help="Dataset name from HuggingFace"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="imagenet_r_results.pkl",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="Save logits in results (increases file size)"
    )
    
    parser.add_argument(
        "--no_save_logits",
        dest="save_logits",
        action="store_false",
        help="Do not save logits in results"
    )
    parser.set_defaults(save_logits=True)

    parser.add_argument(
        "--generate|_ds",
        action="store_true",
        help="Generate supervision dataset"
    )
    
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=42,
        help="Starting seed for reproducibility"
    )
    
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=100,
        help="Print progress every N samples"
    )
    
    return parser.parse_args()

def get_baseline_model(model_id: str) -> str: # check later if this works
    """
    Get the corresponding baseline accuracy and calibration results for a given model.
    
    Args:
        model_id: model identifier string
    """
    # Run experiment
    print(f"\nRunning inference...")
    results = run_imagenet_r_experiment(
        model=model,
        processor=processor,
        device=device,
        test_split=test_split,
        all_class_names=all_class_names,
        build_prompt_fn=build_mc_prompt_imagenet_r,
        predict_fn=predict_letter_and_logits,
        seed_offset=args.seed_offset,
        save_logits=args.save_logits,
        progress_interval=args.progress_interval,
    )
    
    # Save results
    additional_info = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "seed_offset": args.seed_offset,
    }
    save_results(results, args.output, additional_info)
    
    print("\nExperiment completed successfully!")

def generate_ds(model_id, dataset):
    # 1 get data split
    data_split = get_data_split(model_id, dataset) # select first 10 samples for testing pipeline, replace this later with full dataset

    # Forward pass loop
    for idx in range(len(data_split)):
        sample = data_split[idx]
        # Further processing and saving logic goes here



def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset_by_name(args.dataset)
    test_split = dataset["test"]
    all_class_names = sorted(set(test_split["class_name"]))
    print(f"Dataset loaded: {len(test_split)} test samples, {len(all_class_names)} classes")
    
    # Load model
    model, processor, device = load_llava_model(args.model_id)
    
    # Get baseline model
    baseline_model = get_baseline_model(args.model_id)

    # generate supervision dataset
    if args.generate_ds:
        generate_ds(baseline_model, dataset)



if __name__ == "__main__":
    main()
