import numpy as np
import torch
from typing import Dict, Any
import matplotlib.pyplot as plt


def compute_ece(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10,
) -> Dict[str, Any]:
    preds = predictions.detach().float().cpu().numpy()
    targs = targets.detach().float().cpu().numpy()

    preds = np.clip(preds, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    N = len(preds)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
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
        "bin_edges": bin_edges.tolist(),
        "bin_counts": bin_counts,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
    }


def compute_auroc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    preds_np = predictions.detach().float().cpu().numpy()
    targs_np = targets.detach().float().cpu().numpy()

    if len(np.unique(targs_np)) < 2:
        return 0.5

    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(targs_np, preds_np))


def plot_reliability_diagram(
    ece_dict: Dict[str, Any],
    output_path: str,
    title: str = "Reliability diagram",
    show_diagonal: bool = True,
) -> None:
    bin_acc = np.array(ece_dict["bin_accuracies"], dtype=float)
    bin_conf = np.array(ece_dict["bin_confidences"], dtype=float)
    bin_counts = np.array(ece_dict["bin_counts"], dtype=float)
    bin_edges = np.array(ece_dict["bin_edges"], dtype=float)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = (bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 0.1

    valid = ~np.isnan(bin_acc)

    plt.figure(figsize=(7, 6))
    plt.bar(
        bin_centers[valid],
        bin_acc[valid],
        width=bin_width * 0.9,
        edgecolor="black",
        alpha=0.85,
        label="Empirical accuracy",
    )

    if show_diagonal:
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2, label="Perfect calibration")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"{title}\nECE = {ece_dict['ece']:.4f}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
