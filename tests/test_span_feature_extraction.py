"""Test suite for validating span-based feature extraction from LM hidden states.

This test verifies that:
1. Token spans are correctly identified
2. Visual and textual span features are properly extracted
3. Features from different spans are distinguishable
4. Features attend to the correct tokens
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.extract_features import extract_lm_features_mean_pool


def test_span_extraction_basic():
    """Test basic functionality of span-based feature extraction."""
    print("="*80)
    print("Test 1: Basic Span Extraction")
    print("="*80)
    
    batch_size = 2
    seq_len = 100
    hidden_dim = 512
    num_layers = 12
    
    # Create mock hidden states
    hidden_states = tuple([
        torch.randn(batch_size, seq_len, hidden_dim)
        for _ in range(num_layers)
    ])
    
    # Define two non-overlapping spans
    visual_start, visual_end = 10, 30
    prompt_start, prompt_end = 50, 80
    
    # Extract features from final layer
    visual_features = extract_lm_features_mean_pool(
        hidden_states, layer_idx=-1, 
        token_start=visual_start, token_end=visual_end
    )
    
    prompt_features = extract_lm_features_mean_pool(
        hidden_states, layer_idx=-1,
        token_start=prompt_start, token_end=prompt_end
    )
    
    # Assertions
    assert visual_features.shape == (batch_size, hidden_dim), f"Expected shape ({batch_size}, {hidden_dim}), got {visual_features.shape}"
    assert prompt_features.shape == (batch_size, hidden_dim), f"Expected shape ({batch_size}, {hidden_dim}), got {prompt_features.shape}"
    
    print(f"✓ Visual features shape: {visual_features.shape}")
    print(f"✓ Prompt features shape: {prompt_features.shape}")
    print(f"✓ No NaN in visual features: {not torch.isnan(visual_features).any()}")
    print(f"✓ No NaN in prompt features: {not torch.isnan(prompt_features).any()}")
    print()


def test_span_features_are_different():
    """Test that features from different spans are distinguishable."""
    print("="*80)
    print("Test 2: Different Spans Produce Different Features")
    print("="*80)
    
    batch_size = 1
    seq_len = 100
    hidden_dim = 512
    num_layers = 12
    
    # Create hidden states with distinct patterns in different spans
    base_hidden_states = []
    for layer_idx in range(num_layers):
        layer_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Add distinct patterns to different spans
        # Visual span (10-30): add positive bias
        layer_states[:, 10:30, :] += 2.0
        # Prompt span (50-80): add negative bias
        layer_states[:, 50:80, :] -= 2.0
        
        base_hidden_states.append(layer_states)
    
    hidden_states = tuple(base_hidden_states)
    
    # Extract features
    visual_features = extract_lm_features_mean_pool(
        hidden_states, layer_idx=-1,
        token_start=10, token_end=30
    )
    
    prompt_features = extract_lm_features_mean_pool(
        hidden_states, layer_idx=-1,
        token_start=50, token_end=80
    )
    
    # Calculate difference
    mean_visual = visual_features.mean().item()
    mean_prompt = prompt_features.mean().item()
    difference = abs(mean_visual - mean_prompt)
    
    print(f"Visual features mean: {mean_visual:.4f}")
    print(f"Prompt features mean: {mean_prompt:.4f}")
    print(f"Absolute difference: {difference:.4f}")
    
    # Features should be significantly different
    assert difference > 1.0, f"Features are too similar (diff={difference:.4f}), expected > 1.0"
    
    # Visual features should have higher mean (we added +2.0)
    assert mean_visual > mean_prompt, "Visual features should have higher mean than prompt features"
    
    print(f"✓ Features from different spans are distinguishable")
    print(f"✓ Pattern correctly reflects token span content")
    print()


def test_span_boundaries():
    """Test that features correctly respect span boundaries."""
    print("="*80)
    print("Test 3: Span Boundaries Are Respected")
    print("="*80)
    
    batch_size = 1
    seq_len = 100
    hidden_dim = 256
    num_layers = 6
    
    # Create hidden states with a clear signal in a specific span
    base_hidden_states = []
    for layer_idx in range(num_layers):
        layer_states = torch.zeros(batch_size, seq_len, hidden_dim)
        
        # Only tokens 20-40 have non-zero values
        layer_states[:, 20:40, :] = 5.0
        
        base_hidden_states.append(layer_states)
    
    hidden_states = tuple(base_hidden_states)
    
    # Extract features that fully overlap the signal
    features_overlap = extract_lm_features_mean_pool(
        hidden_states, layer_idx=-1,
        token_start=20, token_end=40
    )
    
    # Extract features that partially overlap the signal
    features_partial = extract_lm_features_mean_pool(
        hidden_states, layer_idx=-1,
        token_start=15, token_end=25
    )
    
    # Extract features with no overlap
    features_no_overlap = extract_lm_features_mean_pool(
        hidden_states, layer_idx=-1,
        token_start=50, token_end=70
    )
    
    mean_overlap = features_overlap.mean().item()
    mean_partial = features_partial.mean().item()
    mean_no_overlap = features_no_overlap.mean().item()
    
    print(f"Full overlap mean: {mean_overlap:.4f}")
    print(f"Partial overlap mean: {mean_partial:.4f}")
    print(f"No overlap mean: {mean_no_overlap:.4f}")
    
    # Full overlap should equal the signal value
    assert abs(mean_overlap - 5.0) < 0.01, f"Full overlap should be ~5.0, got {mean_overlap:.4f}"
    
    # Partial overlap should be between 0 and 5
    assert 0 < mean_partial < 5.0, f"Partial overlap should be between 0 and 5, got {mean_partial:.4f}"
    
    # No overlap should be zero
    assert abs(mean_no_overlap) < 0.01, f"No overlap should be ~0.0, got {mean_no_overlap:.4f}"
    
    print(f"✓ Span boundaries are correctly respected")
    print(f"✓ Features accurately represent only their span content")
    print()


def test_different_layers_produce_different_features():
    """Test that different layers produce different features."""
    print("="*80)
    print("Test 4: Different Layers Produce Different Features")
    print("="*80)
    
    batch_size = 1
    seq_len = 100
    hidden_dim = 512
    num_layers = 12
    
    # Create hidden states where each layer has a unique pattern
    base_hidden_states = []
    for layer_idx in range(num_layers):
        # Each layer has a different scale
        layer_states = torch.randn(batch_size, seq_len, hidden_dim) * (layer_idx + 1)
        base_hidden_states.append(layer_states)
    
    hidden_states = tuple(base_hidden_states)
    
    # Extract features from different layers
    first_layer = extract_lm_features_mean_pool(
        hidden_states, layer_idx=0,
        token_start=10, token_end=50
    )
    
    middle_layer = extract_lm_features_mean_pool(
        hidden_states, layer_idx=num_layers // 2,
        token_start=10, token_end=50
    )
    
    final_layer = extract_lm_features_mean_pool(
        hidden_states, layer_idx=-1,
        token_start=10, token_end=50
    )
    
    # Calculate standard deviations (should increase with layer depth)
    std_first = first_layer.std().item()
    std_middle = middle_layer.std().item()
    std_final = final_layer.std().item()
    
    print(f"First layer (0) std: {std_first:.4f}")
    print(f"Middle layer ({num_layers // 2}) std: {std_middle:.4f}")
    print(f"Final layer ({num_layers - 1}) std: {std_final:.4f}")
    
    # Verify increasing pattern
    assert std_middle > std_first, "Middle layer should have larger std than first layer"
    assert std_final > std_middle, "Final layer should have larger std than middle layer"
    
    print(f"✓ Different layers produce features with different characteristics")
    print()


def test_batch_consistency():
    """Test that features are computed consistently across batch dimension."""
    print("="*80)
    print("Test 5: Batch Consistency")
    print("="*80)
    
    batch_size = 4
    seq_len = 100
    hidden_dim = 512
    num_layers = 6
    
    # Create identical hidden states for all batch items
    single_states = torch.randn(1, seq_len, hidden_dim)
    
    base_hidden_states = []
    for layer_idx in range(num_layers):
        # Repeat the same states for all batch items
        layer_states = single_states.repeat(batch_size, 1, 1)
        base_hidden_states.append(layer_states)
    
    hidden_states = tuple(base_hidden_states)
    
    # Extract features
    features = extract_lm_features_mean_pool(
        hidden_states, layer_idx=-1,
        token_start=10, token_end=50
    )
    
    # All batch items should have identical features
    for i in range(1, batch_size):
        difference = (features[0] - features[i]).abs().max().item()
        assert difference < 1e-5, f"Batch item {i} differs from item 0 by {difference}"
    
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ All {batch_size} batch items produce identical features")
    print(f"✓ Max difference between batch items: {difference:.2e}")
    print()


def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*80)
    print("SPAN FEATURE EXTRACTION TEST SUITE")
    print("="*80 + "\n")
    
    tests = [
        test_span_extraction_basic,
        test_span_features_are_different,
        test_span_boundaries,
        test_different_layers_produce_different_features,
        test_batch_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"✗ TEST FAILED: {e}\n")
        except Exception as e:
            failed += 1
            print(f"✗ TEST ERROR: {e}\n")
    
    print("="*80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
