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

class ComponentAttentionMLPProbe(nn.Module):
    """
    Component-attention MLP probe for confidence / correctness prediction.

    Expected input:
        x of shape [batch_size, input_dim]
    where input_dim = num_components * component_dim

    The model internally reshapes x to:
        [batch_size, num_components, component_dim]

    It then:
    1. computes an attention score for each component
    2. forms a weighted sum of the component vectors
    3. predicts a probability with an MLP head

    This is designed to integrate with an existing pipeline that already
    concatenates multiple component vectors into one flat feature vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128],
        dropout: float = 0.2,
        activation: str = "relu",
        num_components: int = 4,
        attn_hidden_dim: int = 128,
        use_batchnorm: bool = True,
        return_attention: bool = False,
    ):
        """
        Args:
            input_dim: Total input feature dimension = num_components * component_dim
            hidden_dims: Hidden dimensions for the regression MLP head
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'elu')
            num_components: Number of concatenated component vectors
            attn_hidden_dim: Hidden size for the attention scoring MLP
            use_batchnorm: Whether to use BatchNorm in the regression head
            return_attention: If True, forward returns (probs, attention_weights).
                              If False, forward returns probs only.
        """
        super().__init__()

        if input_dim % num_components != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by num_components "
                f"({num_components}) for ComponentAttentionMLPProbe."
            )

        self.input_dim = input_dim
        self.num_components = num_components
        self.component_dim = input_dim // num_components
        self.return_attention = return_attention
        self.use_batchnorm = use_batchnorm

        # Choose activation
        if activation == "relu":
            self.activation_name = "relu"
            activation_layer = nn.ReLU
        elif activation == "gelu":
            self.activation_name = "gelu"
            activation_layer = nn.GELU
        elif activation == "elu":
            self.activation_name = "elu"
            activation_layer = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Attention network applied independently to each component vector
        # Input:  [B, C, D]
        # Output: [B, C, 1]
        self.attn_net = nn.Sequential(
            nn.Linear(self.component_dim, attn_hidden_dim),
            activation_layer(),
            nn.Linear(attn_hidden_dim, 1)
        )

        # Regression head on the pooled representation [B, D]
        mlp_layers = []
        prev_dim = self.component_dim

        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(activation_layer())
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.regressor = nn.Sequential(*mlp_layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _reshape_components(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat concatenated input into component structure.

        Args:
            x: Tensor of shape [B, input_dim]

        Returns:
            Tensor of shape [B, num_components, component_dim]
        """
        if x.ndim != 2:
            raise ValueError(
                f"Expected x to have shape [B, input_dim], got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got x.shape[1]={x.shape[1]}"
            )

        return x.view(x.shape[0], self.num_components, self.component_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input features of shape [batch_size, input_dim]

        Returns:
            If return_attention=False:
                probs of shape [batch_size]
            If return_attention=True:
                (probs, attention_weights)
                probs: [batch_size]
                attention_weights: [batch_size, num_components]
        """
        # [B, C, D]
        x_components = self._reshape_components(x)

        # Compute component attention scores: [B, C, 1]
        attn_scores = self.attn_net(x_components)

        # Normalize over components: [B, C, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum pooling: [B, D]
        pooled = (attn_weights * x_components).sum(dim=1)

        # Regression head: [B, 1]
        logits = self.regressor(pooled)

        # Probability output: [B]
        probs = torch.sigmoid(logits).squeeze(-1)

        if self.return_attention:
            return probs, attn_weights.squeeze(-1)

        return probs

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights for each component.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            attention weights [batch_size, num_components]
        """
        x_components = self._reshape_components(x)
        attn_scores = self.attn_net(x_components)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)


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
            


