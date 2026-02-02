import argparse
from utils.model_loader import load_model
from utils.data_loader import load_dataset_prepared
from utils.uq_ds_generator import generate_supervised_uq_dataset, SupervisionGenConfig
from utils.inference import predict_letter_and_logits_with_features

def run_generate_dataset(args):
    """
    Run the full pipeline to generate supervised UQ dataset.
    
    Args:
        args: Namespace object from argparse with all CLI arguments
    """
    # Access arguments as attributes
    dataset_id = args.dataset
    vlm_id = args.vlm
    output_dir = args.output_dir
    seed = args.seed
    device = args.device
    
    # Rest of your pipeline code here
    print(f"Generating dataset for {dataset_id} using {vlm_id}")
    
    # Step 1: Load model
    model, processor, device = load_model(vlm_id, device)

    print(f"{model} successfully loaded on {device}")

    # Step 2: Load dataset(s)
    dataset = load_dataset_prepared(dataset_id)
    print(f"Dataset {dataset_id} loaded with {len(dataset)} samples")

    # Step 3: Generate supervised UQ dataset
    cfg = SupervisionGenConfig(
        dataset_id=dataset_id,
        model_id=vlm_id,
        output_root=output_dir,
        seed_offset=seed,
        max_samples=None,  # or args.max_samples if you add this argument
        verbose=False,
    )

    # Step 4: Generate supervised UQ dataset
    supervision_dataset = generate_supervised_uq_dataset(
        model=model,
        processor=processor,
        device=device,
        samples=dataset,
        cfg=cfg,
        predict_fn=predict_letter_and_logits_with_features,
    )


    