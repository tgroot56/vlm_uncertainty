"""Utilities for running experiments and saving results."""

import pickle
from typing import Dict, List, Optional
from tqdm import tqdm


def run_imagenet_r_experiment(
    model,
    processor,
    device,
    mc_dataset: List[Dict],
    predict_fn,
    save_logits: bool = True,
    progress_interval: int = 100,
) -> List[Dict]:
    """
    Run ImageNet-R experiment on a pre-constructed multiple choice dataset.
    
    Args:
        model: Loaded model
        processor: Model processor
        device: torch device
        mc_dataset: Pre-constructed dataset with prompts and options
        predict_fn: Function to make predictions
        save_logits: Whether to save logits in results
        progress_interval: Print progress every N samples
        
    Returns:
        List of result dictionaries
    """
    results = []
    correct = 0
    total = 0
    
    for mc_sample in tqdm(mc_dataset, desc="Processing test set"):
        # Get prediction and probabilities using pre-constructed prompt
        pred_letter, option_probs, option_logits, raw_text = predict_fn(
            model=model,
            processor=processor,
            device=device,
            image=mc_sample["image"],
            prompt=mc_sample["prompt"],
        )
        
        is_correct = (pred_letter == mc_sample["gt_letter"])
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        result = {
            "idx": mc_sample["idx"],
            "gt_letter": mc_sample["gt_letter"],
            "pred_letter": pred_letter,
            "is_correct": is_correct,
            "option_probs": option_probs,  # softmax over A/B/C/D
            "gt_class": mc_sample["gt_class"],
            "option_map": mc_sample["option_map"],
            "raw_text": raw_text,
        }
        
        # Optionally include logits
        if save_logits:
            result["option_logits"] = option_logits
        
        results.append(result)
        
        # Print progress
        if (total) % progress_interval == 0:
            acc = correct / total
            print(f"\nSamples processed: {total}, Accuracy so far: {acc:.4f}")
    
    # Final accuracy
    final_accuracy = correct / total
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"{'='*60}")
    
    return results


def save_results(
    results: List[Dict],
    output_file: str,
    additional_info: Optional[Dict] = None
):
    """
    Save experiment results to a pickle file.
    
    Args:
        results: List of result dictionaries
        output_file: Output file path
        additional_info: Optional additional information to save
    """
    # Calculate statistics
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r["is_correct"])
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # Prepare data to save
    data = {
        "results": results,
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
    }
    
    # Add any additional info
    if additional_info:
        data.update(additional_info)
    
    # Save to file
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    
    print(f"\nResults saved to {output_file}")
    print(f"Each result contains: {', '.join(results[0].keys())}")
    if "option_probs" in results[0]:
        print("Use option_probs for ECE calculation (softmax confidences over A/B/C/D)")
