from utils.model_loader import load_model
from src.data import load_dataset_prepared
from src.uq_dataset_generation import (
    generate_qwen_visual_supervised_dataset,
    generate_supervised_uq_dataset,
    SupervisionGenConfig,
)
from src.uq_dataset_generation.registry import get_task

from utils.inference import predict_letter_and_logits_with_features, predict_answer_and_features

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
    verbose = args.verbose
    
    print(f"Generating dataset for {dataset_id} using {vlm_id}")
    
    # Step 1: Load model
    model, processor, device, vlm_adapter = load_model(vlm_id, device)
    task = get_task(dataset_id)

    print(f"{model} successfully loaded on {device}")

    # Step 2: Load dataset(s)
    dataset_kwargs = task.dataset_load_kwargs(args, device)

    dataset = load_dataset_prepared(
        dataset_id=dataset_id,
        split=args.dataset_split,
        seed_offset=seed,
        use_cache=True,
        max_samples=args.max_samples, 
        dataset_kwargs=dataset_kwargs,
    )

    print(f"Dataset {dataset_id} loaded with {len(dataset)} samples")

    # Step 3: Generate supervised UQ dataset
    cfg = SupervisionGenConfig(
        dataset_id=dataset_id,
        model_id=vlm_id,
        output_root=output_dir,
        dataset_split=args.dataset_split,
        seed_offset=seed,
        max_samples=args.max_samples,  
        verbose=verbose,
    )

    all_lm_mean_layers = tuple(getattr(args, "all_lm_mean_layers", None) or [])
    if all_lm_mean_layers:
        # Dedicated layer-sweep mode: keep only the requested per-layer LM mean features.
        cfg.use_vision_middle = False
        cfg.use_vision_final = False
        cfg.use_lm_visual_middle = False
        cfg.use_lm_visual_final = False
        cfg.use_lm_question_middle = False
        cfg.use_lm_question_final = False
        cfg.use_lm_answer_middle = False
        cfg.use_lm_answer_final = False
        cfg.use_lm_visual_middle_lasttoken = False
        cfg.use_lm_visual_final_lasttoken = False
        cfg.use_lm_question_middle_lasttoken = False
        cfg.use_lm_question_final_lasttoken = False
        cfg.use_lm_answer_middle_lasttoken = False
        cfg.use_lm_answer_final_lasttoken = False
        cfg.use_answer_prob_entropy_stats = False
        cfg.use_lm_fullspan_middle = False
        cfg.use_lm_fullspan_final = False
        cfg.use_lm_fullspan_middle_lasttoken = False
        cfg.use_lm_fullspan_final_lasttoken = False
        cfg.use_lm_question_including_middle = False
        cfg.use_lm_question_including_final = False
        cfg.use_lm_question_including_middle_lasttoken = False
        cfg.use_lm_question_including_final_lasttoken = False

        cfg.use_lm_visual_all_layers_mean = "visual" in all_lm_mean_layers
        cfg.use_lm_question_all_layers_mean = "question" in all_lm_mean_layers
        cfg.use_lm_answer_all_layers_mean = "answer" in all_lm_mean_layers

    if task.prediction_mode == "multiple_choice":
        predict_fn = predict_letter_and_logits_with_features
    elif task.prediction_mode == "open_ended_vqa":
        predict_fn = predict_answer_and_features
    else:
        raise ValueError(f"Unsupported prediction_mode '{task.prediction_mode}' for dataset_id '{dataset_id}'")

    if getattr(args, "vision_features_only", False):
        if "qwen" not in vlm_id.lower():
            raise ValueError("--vision_features_only is currently only supported for Qwen models.")

        generate_qwen_visual_supervised_dataset(
            model=model,
            processor=processor,
            device=device,
            vlm_adapter=vlm_adapter,
            samples=dataset,
            cfg=cfg,
            source_supervision_path=getattr(args, "source_supervision_path", None),
            vision_batch_size=getattr(args, "vision_batch_size", 4),
        )
        return

    generate_supervised_uq_dataset(
        model=model,
        processor=processor,
        device=device,
        vlm_adapter=vlm_adapter,
        samples=dataset,
        cfg=cfg,
        predict_fn=predict_fn,
    )


    
