"""Linear probe models for correctness prediction."""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Random Forest model is available in random_forest_model.py
# from probe.random_forest_model import RandomForestProbe, FeatureSelector


class LinearProbe(nn.Module):
    """
    Simple linear probe for predicting correctness probability.
    
    Maps features to a probability in [0, 1] using a linear layer + sigmoid.
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize the linear probe.
        
        Args:
            input_dim: Dimension of input features
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
        # Initialize with small weights for better calibration
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape [batch_size, input_dim]
            
        Returns:
            Predicted probabilities of shape [batch_size]
        """
        logits = self.linear(x)  # [batch_size, 1]
        probs = torch.sigmoid(logits).squeeze(-1)  # [batch_size]
        return probs


class MLPProbe(nn.Module):
    """
    Multi-layer perceptron probe for richer feature learning.
    
    Uses hidden layers with nonlinearities to learn complex patterns.
    Compatible with BrierScoreLoss and all existing metrics.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128],
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Initialize the MLP probe.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
            activation: Activation function ('relu', 'gelu', 'elu')
        """
        super().__init__()
        
        # Choose activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape [batch_size, input_dim]
            
        Returns:
            Predicted probabilities of shape [batch_size]
        """
        logits = self.network(x)  # [batch_size, 1]
        probs = torch.sigmoid(logits).squeeze(-1)  # [batch_size]
        return probs


class BrierScoreLoss(nn.Module):
    """
    Brier Score loss function with optional class weighting.
    
    The Brier score is defined as: BS = (1/N) * sum((p - y)^2)
    where p is the predicted probability and y is the true label (0 or 1).
    
    With class weighting: BS = (1/N) * sum(w * (p - y)^2)
    where w is higher for the minority class.
    
    This loss has a probabilistic interpretation and encourages well-calibrated
    probability estimates.
    """
    
    def __init__(self, reduction: str = 'mean', neg_weight: float = 1.0):
        """
        Initialize Brier Score loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
            neg_weight: Weight for negative class (class 0). Use > 1 to upweight minority class.
        """
        super().__init__()
        self.reduction = reduction
        self.neg_weight = neg_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Brier score with class weighting.
        
        Args:
            predictions: Predicted probabilities of shape [batch_size]
            targets: True labels (0 or 1) of shape [batch_size]
            
        Returns:
            Brier score loss
        """
        # Compute squared error between predicted probability and true label
        brier_score = (predictions - targets) ** 2
        
        # Apply class weights: weight positive class (targets=1) more
        weights = torch.where(targets == 0, self.neg_weight, 1.0)
        weighted_brier = brier_score * weights
        if self.reduction == 'mean':
            return weighted_brier.mean()
        elif self.reduction == 'sum':
            return weighted_brier.sum()
        else:  # 'none'
            return weighted_brier
            


