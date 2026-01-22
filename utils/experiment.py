"""Utilities for running experiments and saving results."""

import pickle
from typing import Dict, List, Optional
from tqdm import tqdm


def run_imagenet_r_experiment(
    model,
    processor,
    device,
    test_split,
    all_class_names: List[str],
    build_prompt_fn,
    predict_fn,
    seed_offset: int = 42,
    save_logits: bool = True,
    progress_interval: int = 100,
) -> List[Dict]:
    """
    Run ImageNet-R experiment on the test split.
    
    Args:
        model: Loaded model
        processor: Model processor
        device: torch device
        test_split: Dataset test split
        all_class_names: List of all class names
        build_prompt_fn: Function to build prompts
        predict_fn: Function to make predictions
        seed_offset: Starting seed for reproducibility
        save_logits: Whether to save logits in results
        progress_interval: Print progress every N samples
        
    Returns:
        List of result dictionaries
    """
    results = []
    correct = 0
    total = 0
    
    for idx in tqdm(range(len(test_split)), desc="Processing test set"):
        sample = test_split[idx]
        
        # Build prompt with reproducible seed per sample
        prompt, option_map, gt_letter = build_prompt_fn(
            sample, all_class_names, seed=seed_offset + idx
        )
        
        # Get prediction and probabilities
        pred_letter, option_probs, option_logits, raw_text = predict_fn(
            model=model,
            processor=processor,
            device=device,
            image=sample["image"],
            prompt=prompt,
        )
        
        is_correct = (pred_letter == gt_letter)
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        result = {
            "idx": idx,
            "gt_letter": gt_letter,
            "pred_letter": pred_letter,
            "is_correct": is_correct,
            "option_probs": option_probs,  # softmax over A/B/C/D
            "gt_class": sample["class_name"],
            "option_map": option_map,
            "raw_text": raw_text,
        }
        
        # Optionally include logits
        if save_logits:
            result["option_logits"] = option_logits
        
        results.append(result)
        
        # Print progress
        if (idx + 1) % progress_interval == 0:
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
