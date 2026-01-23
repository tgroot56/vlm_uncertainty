"""Test script for feature extraction functionality."""

import torch
from utils.extract_features import extract_vision_features_mean_pool


def test_extract_vision_features():
    """Test the vision feature extraction function."""
    print("Testing vision feature extraction...")
    
    # Create mock hidden states (simulating CLIP vision tower output)
    batch_size = 2
    num_tokens = 197  # 1 CLS + 196 patch tokens (14x14 patches)
    hidden_dim = 768
    num_layers = 24
    
    # Create tuple of hidden states (one per layer)
    mock_hidden_states = tuple([
        torch.randn(batch_size, num_tokens, hidden_dim)
        for _ in range(num_layers)
    ])
    
    print(f"Mock input: {num_layers} layers, shape per layer: {mock_hidden_states[0].shape}")
    
    # Test extraction from different layers
    for layer_idx in [0, num_layers // 2, -1]:
        features = extract_vision_features_mean_pool(mock_hidden_states, layer_idx)
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Output shape: {features.shape}")
        print(f"  Expected: ({batch_size}, {hidden_dim})")
        print(f"  Shape correct: {features.shape == (batch_size, hidden_dim)}")
        print(f"  Contains NaN: {torch.isnan(features).any()}")
        print(f"  Contains Inf: {torch.isinf(features).any()}")
        print(f"  Mean value: {features.mean().item():.4f}")
        print(f"  Std value: {features.std().item():.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_extract_vision_features()
