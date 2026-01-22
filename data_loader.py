"""Data loading and prompt building utilities."""

from datasets import load_dataset
import random
import pickle
import os
import hashlib
from typing import Dict, List, Tuple, Optional

LETTERS = ["A", "B", "C", "D"]
CACHE_DIR = "cached_datasets"


def load_dataset_by_name(name: str):
    """
    Load a dataset by name from HuggingFace.
    
    Args:
        name: Dataset name/path
        
    Returns:
        Dataset object
    """
    return load_dataset(name)


def build_mc_prompt_imagenet_r(
    sample: Dict,
    all_class_names: List[str],
    seed: Optional[int] = None,
) -> Tuple[str, Dict[str, str], str]:
    """
    Build a multiple-choice prompt for ImageNet-R classification.
    
    Args:
        sample: Dataset sample containing 'class_name' and 'image'
        all_class_names: List of all possible class names
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (prompt_text, option_map, ground_truth_letter)
    """
    rng = random.Random(seed)
    gt_label = sample["class_name"]

    # Sample 3 distractors
    distractors = rng.sample([c for c in all_class_names if c != gt_label], 3)
    options = [gt_label] + distractors
    rng.shuffle(options)

    # Map letters to options
    option_map = {LETTERS[i]: options[i] for i in range(4)}
    gt_letter = next(k for k, v in option_map.items() if v == gt_label)

    # Build prompt
    prompt = (
        "Which object is shown in the image?\n\n"
        "Choices:\n"
        + "\n".join(f"{k}. {v}" for k, v in option_map.items())
        + "\n\nAnswer with only A, B, C, or D."
    )
    
    return prompt, option_map, gt_letter


def construct_mc_dataset(
    test_split,
    all_class_names: List[str],
    seed_offset: int = 42,
) -> List[Dict]:
    """
    Construct a dataset with multiple choice options pre-generated.
    
    Args:
        test_split: Dataset test split
        all_class_names: List of all class names
        seed_offset: Starting seed for reproducibility
        
    Returns:
        List of samples with prompts and options pre-generated
    """
    mc_dataset = []
    
    for idx in range(len(test_split)):
        sample = test_split[idx]
        
        # Build prompt with reproducible seed per sample
        prompt, option_map, gt_letter = build_mc_prompt_imagenet_r(
            sample, all_class_names, seed=seed_offset + idx
        )
        
        # Store the sample with pre-generated MC options
        mc_sample = {
            "idx": idx,
            "image": sample["image"],
            "gt_class": sample["class_name"],
            "prompt": prompt,
            "option_map": option_map,
            "gt_letter": gt_letter,
        }
        
        mc_dataset.append(mc_sample)
    
    return mc_dataset


def get_cache_filename(dataset_name: str, seed_offset: int, num_samples: int) -> str:
    """
    Generate a unique cache filename based on dataset parameters.
    
    Args:
        dataset_name: Name of the dataset
        seed_offset: Seed offset used for reproducibility
        num_samples: Number of samples in the dataset
        
    Returns:
        Cache filename
    """
    # Create a hash of the parameters for a unique filename
    cache_key = f"{dataset_name}_{seed_offset}_{num_samples}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    return f"mc_dataset_{cache_hash}.pkl"


def save_mc_dataset_cache(mc_dataset: List[Dict], dataset_name: str, seed_offset: int) -> str:
    """
    Save the constructed MC dataset to cache.
    
    Args:
        mc_dataset: Constructed multiple choice dataset
        dataset_name: Name of the dataset
        seed_offset: Seed offset used
        
    Returns:
        Path to the cached file
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    cache_filename = get_cache_filename(dataset_name, seed_offset, len(mc_dataset))
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    
    # Save metadata along with dataset
    cache_data = {
        "mc_dataset": mc_dataset,
        "dataset_name": dataset_name,
        "seed_offset": seed_offset,
        "num_samples": len(mc_dataset),
    }
    
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    
    print(f"Cached MC dataset saved to: {cache_path}")
    return cache_path


def load_mc_dataset_cache(dataset_name: str, seed_offset: int, num_samples: int) -> Optional[List[Dict]]:
    """
    Load the MC dataset from cache if it exists.
    
    Args:
        dataset_name: Name of the dataset
        seed_offset: Seed offset to match
        num_samples: Expected number of samples
        
    Returns:
        Cached MC dataset or None if not found
    """
    cache_filename = get_cache_filename(dataset_name, seed_offset, num_samples)
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        
        # Verify metadata matches
        if (cache_data["dataset_name"] == dataset_name and
            cache_data["seed_offset"] == seed_offset and
            cache_data["num_samples"] == num_samples):
            print(f"Loaded MC dataset from cache: {cache_path}")
            return cache_data["mc_dataset"]
        else:
            print("Cache metadata mismatch, will rebuild dataset")
            return None
    except Exception as e:
        print(f"Error loading cache: {e}, will rebuild dataset")
        return None


def construct_or_load_mc_dataset(
    test_split,
    all_class_names: List[str],
    dataset_name: str,
    seed_offset: int = 42,
) -> List[Dict]:
    """
    Construct or load from cache a dataset with multiple choice options.
    
    Args:
        test_split: Dataset test split
        all_class_names: List of all class names
        dataset_name: Name of the dataset (for caching)
        seed_offset: Starting seed for reproducibility
        
    Returns:
        List of samples with prompts and options pre-generated
    """
    num_samples = len(test_split)
    
    # Try to load from cache first
    mc_dataset = load_mc_dataset_cache(dataset_name, seed_offset, num_samples)
    
    if mc_dataset is not None:
        return mc_dataset
    
    # Cache miss - construct the dataset
    print("Cache not found, constructing MC dataset...")
    mc_dataset = construct_mc_dataset(test_split, all_class_names, seed_offset)
    
    # Save to cache
    save_mc_dataset_cache(mc_dataset, dataset_name, seed_offset)
    
    return mc_dataset


def main():
    """Test data loading."""
    dataset = load_dataset_by_name("axiong/imagenet-r")
    print(dataset["test"][0])


if __name__ == "__main__":
    main()