"""Linear probe models for correctness prediction."""

import torch
import torch.nn as nn
from typing import Optional


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


class BrierScoreLoss(nn.Module):
    """
    Brier Score loss function.
    
    The Brier score is defined as: BS = (1/N) * sum((p - y)^2)
    where p is the predicted probability and y is the true label (0 or 1).
    
    This loss has a probabilistic interpretation and encourages well-calibrated
    probability estimates.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize Brier Score loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Brier score.
        
        Args:
            predictions: Predicted probabilities of shape [batch_size]
            targets: True labels (0 or 1) of shape [batch_size]
            
        Returns:
            Brier score loss
        """
        # Compute squared error between predicted probability and true label
        brier_score = (predictions - targets) ** 2
        
        if self.reduction == 'mean':
            return brier_score.mean()
        elif self.reduction == 'sum':
            return brier_score.sum()
        else:  # 'none'
            return brier_score


def compute_calibration_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10
) -> dict:
    """
    Compute calibration metrics including ECE (Expected Calibration Error).
    
    Args:
        predictions: Predicted probabilities [N]
        targets: True labels (0 or 1) [N]
        num_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with calibration metrics
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (predictions > bin_lower.item()) & (predictions <= bin_upper.item())
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(None)
            bin_confidences.append(None)
            bin_counts.append(0)
    
    return {
        'ece': ece,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
    }
