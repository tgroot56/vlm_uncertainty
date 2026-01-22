"""Data loading and prompt building utilities."""

from datasets import load_dataset
import random
from typing import Dict, List, Tuple, Optional

LETTERS = ["A", "B", "C", "D"]


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


def main():
    """Test data loading."""
    dataset = load_dataset_by_name("axiong/imagenet-r")
    print(dataset["test"][0])


if __name__ == "__main__":
    main()