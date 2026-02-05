# """Main script for running ImageNet-R experiments with LLaVA."""

# import argparse
# from tqdm import tqdm
# from data_loader import load_dataset_by_name, construct_or_load_mc_dataset, load_or_construct_mc_dataset_optimized
# from utils.model_loader import load_llava_model
# from utils.inference import predict_letter_and_logits, predict_letter_and_logits_with_features
# from utils.experiment import run_imagenet_r_experiment, save_results
# from generate_dataset import get_supervision_samples, extract_features_from_sample, extract_lm_features_from_sample


# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Run ImageNet-R classification experiments with LLaVA"
#     )
    
#     parser.add_argument(
#         "--model_id",
#         type=str,
#         default="llava-hf/llava-1.5-7b-hf",
#         help="HuggingFace model identifier"
#     )
    
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default="axiong/imagenet-r",
#         help="Dataset name from HuggingFace"
#     )
    
#     parser.add_argument(
#         "--output",
#         type=str,
#         default="imagenet_r_results.pkl",
#         help="Output file for results"
#     )
    
#     parser.add_argument(
#         "--save_logits",
#         action="store_true",
#         help="Save logits in results (increases file size)"
#     )
    
#     parser.add_argument(
#         "--no_save_logits",
#         dest="save_logits",
#         action="store_false",
#         help="Do not save logits in results"
#     )
#     parser.set_defaults(save_logits=True)

#     parser.add_argument(
#         "--baseline",
#         action="store_true",
#         help="Run baseline inference without any adaptations"
#     )

#     parser.add_argument(
#         "--generate_ds",
#         action="store_true",
#         help="Generate supervision dataset"
#     )

#     parser.add_argument(
#         "--subset",
#         action="store_true",
#         help="Generate supervision dataset on a subset (first 10 samples) for testing"
#     )
    
#     parser.add_argument(
#         "--seed_offset",
#         type=int,
#         default=42,
#         help="Starting seed for reproducibility"
#     )
    
#     parser.add_argument(
#         "--progress_interval",
#         type=int,
#         default=100,
#         help="Print progress every N samples"
#     )
    
#     return parser.parse_args()

# def get_baseline_model(model, processor, device, mc_dataset, args):
#     """
#     Run baseline inference with the given model on the pre-constructed MC dataset.
    
#     Args:
#         model: Loaded model
#         processor: Model processor
#         device: torch device
#         mc_dataset: Pre-constructed multiple choice dataset
#         args: Command line arguments
        
#     Returns:
#         results: Experiment results
#     """
#     # Run experiment
#     print(f"\nRunning inference...")
#     results = run_imagenet_r_experiment(
#         model=model,
#         processor=processor,
#         device=device,
#         mc_dataset=mc_dataset,
#         predict_fn=predict_letter_and_logits,
#         save_logits=args.save_logits,
#         progress_interval=args.progress_interval,
#     )
    
#     return results

# def generate_supervision_ds(model, processor, device, mc_dataset, subset=False):
#     """
#     Generate supervision dataset by running forward passes on the first 10 samples or all samples if subset is False.
#     Extracts and saves vision features from middle and final layers.
    
#     Args:
#         model: Loaded model
#         processor: Model processor
#         device: torch device
#         mc_dataset: Pre-constructed multiple choice dataset
#         subset: If True, only process 10 samples and print verbose debug info
        
#     Returns:
#         List of dicts containing samples with extracted features
#     """
#     # Get first 10 samples or all if subset is False
#     num_samples = 10 if subset else len(mc_dataset)
#     supervision_samples = get_supervision_samples(mc_dataset, num_samples=num_samples)
#     if subset == False:
#         verbose = False
#     else:
#         verbose = True
#     print("verbose mode:", verbose)
    
#     print(f"\n{'='*80}")
#     print(f"Running forward passes on {'first 10 samples (verbose mode)' if subset else f'{len(supervision_samples)} samples'}")
#     print(f"{'='*80}\n")
    
#     # Store results with features
#     supervision_dataset = []
    
#     # Loop through samples and do forward passes with feature extraction
#     # Use tqdm for progress bar (disable in verbose mode to avoid cluttering output)
#     progress_bar = tqdm(enumerate(supervision_samples), total=len(supervision_samples), 
#                        desc="Extracting features", disable=verbose)
    
#     for idx, sample in progress_bar:
#         if verbose:
#             print(f"\n--- Sample {idx + 1} ---")
#             print(f"Ground Truth Class: {sample['gt_class']}")
#             print(f"Ground Truth Letter: {sample['gt_letter']}")
        
#         # Run forward pass with vision and LM feature extraction
#         pred_letter, option_probs, option_logits, raw_text, vision_hidden_states, lm_hidden_states, token_spans, answer_hidden_states = predict_letter_and_logits_with_features(
#             model=model,
#             processor=processor,
#             device=device,
#             image=sample['image'],
#             prompt=sample['prompt'],
#         )
        
#         # Extract vision features from both middle and final layers
#         vision_middle_layer_features = None
#         vision_final_layer_features = None
#         lm_middle_visual_features = None
#         lm_final_visual_features = None
#         lm_middle_prompt_features = None
#         lm_final_prompt_features = None
#         lm_middle_answer_features = None
#         lm_final_answer_features = None
        
#         if vision_hidden_states is not None:
#             num_vision_layers = len(vision_hidden_states)
#             middle_vision_layer_idx = num_vision_layers // 2
            
#             # Extract from middle layer
#             vision_middle_layer_features = extract_features_from_sample(
#                 vision_hidden_states=vision_hidden_states,
#                 layer_idx=middle_vision_layer_idx,
#             )
#             if verbose:
#                 print(f"\nVision Middle Layer ({middle_vision_layer_idx}) Features Shape: {vision_middle_layer_features.shape}")
#                 print(f"Vision Middle Layer Features (first 5 dims): {vision_middle_layer_features[0, :5].tolist()}")
            
#             # Extract from final layer
#             vision_final_layer_features = extract_features_from_sample(
#                 vision_hidden_states=vision_hidden_states,
#                 layer_idx=-1,
#             )
#             if verbose:
#                 print(f"Vision Final Layer Features Shape: {vision_final_layer_features.shape}")
#                 print(f"Vision Final Layer Features (first 5 dims): {vision_final_layer_features[0, :5].tolist()}")
        
#         # Extract LM features from both middle and final layers for visual and prompt spans
#         if lm_hidden_states is not None and token_spans:
#             num_lm_layers = len(lm_hidden_states)
#             middle_lm_layer_idx = num_lm_layers // 2
            
#             visual_start = token_spans['visual_start']
#             visual_end = token_spans['visual_end']
#             prompt_start = token_spans['prompt_start']
#             prompt_end = token_spans['prompt_end']
            
#             if verbose:
#                 print(f"\n{'='*60}")
#                 print("Token Span Detection:")
#                 print(f"{'='*60}")
#                 print(f"Image Token ID: {token_spans.get('image_token_id', 'N/A')}")
#                 print(f"Image Token Position in input_ids: {token_spans.get('image_token_position', 'N/A')}")
#                 print(f"Input sequence length: {token_spans.get('input_seq_len', 'N/A')}")
#                 print(f"Hidden states sequence length: {token_spans.get('hidden_seq_len', 'N/A')}")
#                 print(f"Number of visual tokens: {token_spans.get('num_visual_tokens', 'N/A')}")
#                 print(f"Visual span: [{visual_start}:{visual_end}] (length={visual_end - visual_start})")
#                 print(f"Prompt span: [{prompt_start}:{prompt_end}] (length={prompt_end - prompt_start})")
#                 print(f"{'='*60}")
            
#             # Extract from middle LM layer
#             lm_middle_visual_features = extract_lm_features_from_sample(
#                 lm_hidden_states=lm_hidden_states,
#                 layer_idx=middle_lm_layer_idx,
#                 token_start=visual_start,
#                 token_end=visual_end,
#             )
#             if verbose:
#                 print(f"LM Middle Layer ({middle_lm_layer_idx}) Visual Features Shape: {lm_middle_visual_features.shape}")
            
#             lm_middle_prompt_features = extract_lm_features_from_sample(
#                 lm_hidden_states=lm_hidden_states,
#                 layer_idx=middle_lm_layer_idx,
#                 token_start=prompt_start,
#                 token_end=prompt_end,
#             )
#             if verbose:
#                 print(f"LM Middle Layer ({middle_lm_layer_idx}) Prompt Features Shape: {lm_middle_prompt_features.shape}")
            
#             # Extract from final LM layer
#             lm_final_visual_features = extract_lm_features_from_sample(
#                 lm_hidden_states=lm_hidden_states,
#                 layer_idx=-1,
#                 token_start=visual_start,
#                 token_end=visual_end,
#             )
#             if verbose:
#                 print(f"LM Final Layer Visual Features Shape: {lm_final_visual_features.shape}")
            
#             lm_final_prompt_features = extract_lm_features_from_sample(
#                 lm_hidden_states=lm_hidden_states,
#                 layer_idx=-1,
#                 token_start=prompt_start,
#                 token_end=prompt_end,
#             )
#             if verbose:
#                 print(f"LM Final Layer Prompt Features Shape: {lm_final_prompt_features.shape}")
        
#         # Extract answer features from both middle and final layers
#         if answer_hidden_states is not None and token_spans:
#             num_answer_layers = len(answer_hidden_states)
#             middle_answer_layer_idx = num_answer_layers // 2
            
#             answer_start = token_spans['answer_start']
#             answer_end = token_spans['answer_end']
            
#             if verbose:
#                 print(f"\nAnswer span: [{answer_start}:{answer_end}] (length={token_spans['answer_length']})")
            
#             # Extract from middle layer
#             lm_middle_answer_features = extract_lm_features_from_sample(
#                 lm_hidden_states=answer_hidden_states,
#                 layer_idx=middle_answer_layer_idx,
#                 token_start=answer_start,
#                 token_end=answer_end,
#             )
#             if verbose:
#                 print(f"LM Middle Layer ({middle_answer_layer_idx}) Answer Features Shape: {lm_middle_answer_features.shape}")
            
#             # Extract from final layer
#             lm_final_answer_features = extract_lm_features_from_sample(
#                 lm_hidden_states=answer_hidden_states,
#                 layer_idx=-1,
#                 token_start=answer_start,
#                 token_end=answer_end,
#             )
#             if verbose:
#                 print(f"LM Final Layer Answer Features Shape: {lm_final_answer_features.shape}")
        
#         # Store sample with all information including features
#         sample_data = {
#             "idx": sample["idx"],
#             "gt_class": sample["gt_class"],
#             "gt_letter": sample["gt_letter"],
#             "pred_letter": pred_letter,
#             "is_correct": pred_letter == sample["gt_letter"],
#             "option_probs": option_probs,
#             "option_logits": option_logits,
#             "option_map": sample["option_map"],
#             "raw_text": raw_text,
#             "vision_middle_layer_features": vision_middle_layer_features.cpu().float() if vision_middle_layer_features is not None else None,
#             "vision_final_layer_features": vision_final_layer_features.cpu().float() if vision_final_layer_features is not None else None,
#             "lm_middle_visual_features": lm_middle_visual_features.cpu().float() if lm_middle_visual_features is not None else None,
#             "lm_final_visual_features": lm_final_visual_features.cpu().float() if lm_final_visual_features is not None else None,
#             "lm_middle_prompt_features": lm_middle_prompt_features.cpu().float() if lm_middle_prompt_features is not None else None,
#             "lm_final_prompt_features": lm_final_prompt_features.cpu().float() if lm_final_prompt_features is not None else None,
#             "lm_middle_answer_features": lm_middle_answer_features.cpu().float() if lm_middle_answer_features is not None else None,
#             "lm_final_answer_features": lm_final_answer_features.cpu().float() if lm_final_answer_features is not None else None,
#             "token_spans": token_spans,
#         }
#         supervision_dataset.append(sample_data)
        
#         # Print results
#         if verbose:
#             print(f"\nModel Answer: {pred_letter}")
#             print(f"Raw Output: {raw_text}")
#             print(f"Correct: {pred_letter == sample['gt_letter']}")
#             print(f"\nProbabilities:")
#             for letter in sorted(option_probs.keys()):
#                 prob = option_probs[letter]
#                 class_name = sample['option_map'][letter]
#                 marker = "[PREDICTED]" if letter == pred_letter else ""
#                 gt_marker = "[GT]" if letter == sample['gt_letter'] else ""
#                 print(f"  {letter}: {class_name} - {prob:.4f} {marker} {gt_marker}")
#             print(f"{'-'*80}")
    
#     return supervision_dataset 



# def main():
#     """Main execution function."""
#     # Parse arguments
#     args = parse_args()
    
#     # Optimized: Check cache first, only load original dataset if needed
#     print(f"Preparing dataset: {args.dataset}")
#     mc_dataset = load_or_construct_mc_dataset_optimized(
#         dataset_name=args.dataset,
#         seed_offset=args.seed_offset,
#     )
#     print(f"Multiple choice dataset ready: {len(mc_dataset)} samples")
    
#     # Load model
#     print(f"\nLoading model: {args.model_id}")
#     model, processor, device = load_llava_model(args.model_id)
    
#     # Get baseline results
#     if args.baseline:
#         results = get_baseline_model(model, processor, device, mc_dataset, args)
    
#         # Save results
#         additional_info = {
#             "model_id": args.model_id,
#             "dataset": args.dataset,
#             "seed_offset": args.seed_offset,
#         }
#         save_results(results, args.output, additional_info)

#         print("\nExperiment completed successfully!")

#     # Generate supervision dataset
#     if args.generate_ds:
#         supervision_dataset = generate_supervision_ds(model, processor, device, mc_dataset, subset=args.subset)
        
#         # Save the supervision dataset with features
#         import pickle
#         output_path = "supervision_dataset_with_features.pkl"
#         with open(output_path, "wb") as f:
#             pickle.dump(supervision_dataset, f)
#         print(f"\n{'='*80}")
#         print(f"Supervision dataset saved to: {output_path}")
#         print(f"Total samples: {len(supervision_dataset)}")
#         print(f"Each sample contains: {', '.join(supervision_dataset[0].keys())}")
#         print(f"{'='*80}")



# if __name__ == "__main__":
#     main()
