import torch
import torch.nn as nn
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


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