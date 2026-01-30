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
            

def compute_calibration_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Standard ECE for binary classification:
    ECE = sum_m (|acc(B_m) - conf(B_m)| * |B_m|/N)
    """
    preds = predictions.detach().float().cpu().numpy()
    targs = targets.detach().float().cpu().numpy()

    # Safety: if preds might not be clipped already
    preds = np.clip(preds, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    N = len(preds)

    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Use [lower, upper) except last bin includes 1.0
        if i == num_bins - 1:
            in_bin = (preds >= lower) & (preds <= upper)
        else:
            in_bin = (preds >= lower) & (preds < upper)

        count = int(in_bin.sum())
        bin_counts.append(count)

        if count > 0:
            acc = float(targs[in_bin].mean())
            conf = float(preds[in_bin].mean())
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
            ece += abs(conf - acc) * (count / N)
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)

    return {
        "ece": float(ece),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges.tolist(),
    }


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


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    """
    Compute classification metrics including AUROC, F1, balanced accuracy.
    
    Args:
        predictions: Predicted probabilities [N]
        targets: True labels (0 or 1) [N]
        threshold: Decision threshold for binary classification
        
    Returns:
        Dictionary with various metrics
    """
    # Convert to numpy for sklearn
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Binary predictions with given threshold
    binary_preds = (preds_np >= threshold).astype(int)
    
    # Basic metrics
    from sklearn.metrics import (
        roc_auc_score, f1_score, balanced_accuracy_score,
        precision_score, recall_score, confusion_matrix
    )
    
    # AUROC (threshold-independent)
    try:
        auroc = roc_auc_score(targets_np, preds_np)
    except:
        auroc = 0.5  # Only one class present
    
    # Threshold-dependent metrics
    f1 = f1_score(targets_np, binary_preds, zero_division=0)
    balanced_acc = balanced_accuracy_score(targets_np, binary_preds)
    precision = precision_score(targets_np, binary_preds, zero_division=0)
    recall = recall_score(targets_np, binary_preds, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets_np, binary_preds).ravel()
    
    # Accuracy (for completeness)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'auroc': auroc,
        'f1': f1,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
    }


def plot_reliability_diagram(
    calibration_metrics: Dict[str, Any],
    output_path: str,
    title: str = "Reliability Diagram",
    show_diagonal: bool = True,
):
    """
    Plot a reliability diagram (accuracy vs confidence bins).

    Args:
        calibration_metrics: Output dict from compute_calibration_metrics
        output_path: Path to save the plot (e.g. 'reliability.png')
        title: Plot title
        show_diagonal: Whether to plot y=x perfect calibration line
    """
    bin_accuracies = np.array(calibration_metrics["bin_accuracies"], dtype=float)
    bin_confidences = np.array(calibration_metrics["bin_confidences"], dtype=float)
    bin_counts = np.array(calibration_metrics["bin_counts"])
    bin_edges = np.array(calibration_metrics["bin_edges"])

    # Use bin centers for x-axis
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Filter empty bins
    valid = ~np.isnan(bin_accuracies)

    plt.figure(figsize=(7, 6))

    # Bar plot: accuracy per confidence bin
    plt.bar(
        bin_centers[valid],
        bin_accuracies[valid],
        width=bin_width * 0.9,
        edgecolor="black",
        alpha=0.8,
        label="Empirical accuracy",
    )

    # Optional perfect calibration line
    if show_diagonal:
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            linewidth=2,
            label="Perfect calibration",
        )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Predicted confidence", fontsize=12)
    plt.ylabel("Empirical accuracy", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Reliability diagram saved to: {output_path}")


def find_optimal_threshold(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric: str = 'f1',
    num_thresholds: int = 100
) -> tuple:
    """
    Find optimal decision threshold by maximizing a metric.
    
    Args:
        predictions: Predicted probabilities [N]
        targets: True labels (0 or 1) [N]
        metric: Metric to optimize ('f1', 'balanced_accuracy')
        num_thresholds: Number of thresholds to try
        
    Returns:
        (optimal_threshold, best_metric_value, all_thresholds, all_scores)
    """
    import numpy as np
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    thresholds = np.linspace(0, 1, num_thresholds)
    scores = []
    
    from sklearn.metrics import f1_score, balanced_accuracy_score
    
    for threshold in thresholds:
        binary_preds = (preds_np >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(targets_np, binary_preds, zero_division=0)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(targets_np, binary_preds)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return optimal_threshold, best_score, thresholds, scores
