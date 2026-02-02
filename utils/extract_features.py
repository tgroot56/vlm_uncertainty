"""Feature extraction utilities for Vision-Language Models.

This module provides functions to extract hidden representations from
different components of LLaVA-style models (vision tower, projector, language model).
"""

import torch
from typing import Tuple, Union, List


@torch.no_grad()
def extract_vision_features_mean_pool(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_idx: int,
) -> torch.Tensor:
    """
    Extract mean-pooled patch-token features from vision model hidden states.
    
    The CLIP vision model outputs hidden states where:
    - Token 0 is the CLS token
    - Tokens 1: are patch tokens
    
    This function extracts features from a specific layer and mean-pools
    over patch tokens only (excluding the CLS token).
    
    Args:
        hidden_states: Tuple of hidden state tensors from the vision model
            (obtained via output_hidden_states=True). Each element has shape
            (batch_size, num_tokens, hidden_dim) where num_tokens includes
            the CLS token plus all patch tokens.
        layer_idx: Integer index of the layer to extract from (0-indexed).
            For example, use -1 for the final layer, or len(hidden_states)//2
            for a middle layer.
    
    Returns:
        Tensor of shape (batch_size, hidden_dim) containing the mean-pooled
        patch token representations from the specified layer.
    
    Example:
        >>> # Assuming vision_model outputs hidden_states
        >>> outputs = vision_model(pixel_values, output_hidden_states=True)
        >>> hidden_states = outputs.hidden_states
        >>> # Extract from final layer
        >>> features = extract_vision_features_mean_pool(hidden_states, -1)
        >>> # features.shape: (batch_size, hidden_dim)
    """
    # Extract the specified layer's hidden states
    # Shape: (batch_size, num_tokens, hidden_dim)
    layer_hidden_states = hidden_states[layer_idx]
    
    # Exclude CLS token (index 0), keep only patch tokens (index 1:)
    # Shape: (batch_size, num_patch_tokens, hidden_dim)
    patch_tokens = layer_hidden_states[:, 1:, :]
    
    # Mean pool over patch tokens (dim=1)
    # Shape: (batch_size, hidden_dim)
    mean_pooled = patch_tokens.mean(dim=1)
    
    return mean_pooled


@torch.no_grad()
def extract_lm_features_mean_pool(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_idx: int,
    token_start: int,
    token_end: int,
) -> torch.Tensor:
    """
    Extract mean-pooled features from language model hidden states over a token span.
    
    Args:
        hidden_states: Tuple of hidden state tensors from the language model
            (obtained via output_hidden_states=True). Each element has shape
            (batch_size, seq_len, hidden_dim).
        layer_idx: Integer index of the layer to extract from (0-indexed).
            For example, use -1 for the final layer, or len(hidden_states)//2
            for a middle layer.
        token_start: Starting token index (inclusive) for the span to pool over.
        token_end: Ending token index (exclusive) for the span to pool over.
    
    Returns:
        Tensor of shape (batch_size, hidden_dim) containing the mean-pooled
        representations from the specified layer and token span.
    
    Example:
        >>> # Extract features over visual token span
        >>> outputs = model(input_ids, output_hidden_states=True)
        >>> hidden_states = outputs.hidden_states
        >>> features = extract_lm_features_mean_pool(
        ...     hidden_states, layer_idx=-1, token_start=1, token_end=577
        ... )
    """
    # Extract the specified layer's hidden states
    # Shape: (batch_size, seq_len, hidden_dim)
    layer_hidden_states = hidden_states[layer_idx]
    
    # Extract the token span
    # Shape: (batch_size, span_length, hidden_dim)
    span_tokens = layer_hidden_states[:, token_start:token_end, :]
    
    # Mean pool over the token span (dim=1)
    # Shape: (batch_size, hidden_dim)
    mean_pooled = span_tokens.mean(dim=1)
    
    return mean_pooled


@torch.no_grad()
def extract_lm_last_k_tokens(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_idx: int,
    token_start: int,
    token_end: int,
    k: int = 1,
) -> torch.Tensor:
    """
    Extract the last-k token hidden states from a given span [token_start, token_end).

    Returns:
      Tensor of shape (batch_size, k, hidden_dim)

    If k=1, this is the "final token representation" of the span.
    """
    assert token_end > token_start, f"Empty/invalid span: [{token_start}, {token_end})"
    span_len = token_end - token_start
    assert k >= 1, "k must be >= 1"
    k = min(k, span_len)  # clamp if span shorter than k

    layer_h = hidden_states[layer_idx]  # (B, seq_len, H)
    # last k tokens within the span:
    start = token_end - k
    return layer_h[:, start:token_end, :]


