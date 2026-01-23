"""Dataset generation utilities for supervision data."""

from utils.extract_features import extract_vision_features_mean_pool
from typing import List, Dict
import torch


def get_supervision_samples(mc_dataset, num_samples: int = 10):
    """
    Get the first num_samples from the pre-constructed MC dataset for supervision.
    
    Args:
        mc_dataset: Pre-constructed multiple choice dataset
        num_samples: Number of samples to retrieve (default: 10)
        
    Returns:
        List of the first num_samples from mc_dataset
    """
    num_samples = min(num_samples, len(mc_dataset))
    return mc_dataset[:num_samples]


def extract_features_from_sample(
    vision_hidden_states: torch.Tensor,
    layer_idx: int = -1,
) -> torch.Tensor:
    """
    Extract mean-pooled vision features from a sample's hidden states.
    
    Args:
        vision_hidden_states: Tuple of hidden states from vision model
        layer_idx: Layer index to extract from (default: -1 for final layer)
        
    Returns:
        Mean-pooled features tensor of shape (batch_size, hidden_dim)
    """
    features = extract_vision_features_mean_pool(vision_hidden_states, layer_idx)
    
    # Validation checks
    assert features.dim() == 2, f"Expected 2D tensor, got shape {features.shape}"
    assert not torch.isnan(features).any(), "Features contain NaN values"
    assert not torch.isinf(features).any(), "Features contain Inf values"
    
    return features


