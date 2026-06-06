from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_NAME = "lm_fullspan_lasttok_layer_16"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "probe_results/msts_eval/llava"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "probe_results/_plots/generalization_msts_auroc"


@dataclass(frozen=True)
class ProbeFamily:
    key: str
    label: str
    color: str


PROBE_FAMILIES = (
    ProbeFamily("llava", "LLaVA-trained", "#2c7fb8"),
    ProbeFamily("qwen", "Qwen-trained", "#41ab5d"),
    ProbeFamily("both_models", "LLaVA+Qwen-trained", "#8856a7"),
)


def feature_slug(feature_name: str) -> str:
    return feature_name.strip().replace("/", "_").replace(" ", "_")


def predictions_path(results_root: Path, family: ProbeFamily, feature_name: str) -> Path:
    return results_root / f"{family.key}_{feature_slug(feature_name)}" / "predictions.csv"


def load_prediction_mean(path: Path) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[float] = []
    labels: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["probe_name"] != "ensemble_mean":
                continue
            predictions.append(float(row["prediction"]))
            labels.append(float(row["label"]))
    if not predictions:
        raise RuntimeError(f"No prediction-mean rows found in {path}")
    return np.array(predictions, dtype=float), np.array(labels, dtype=float)


def calibration_bins(predictions: np.ndarray, labels: np.ndarray, num_bins: int) -> dict[str, object]:
    predictions = np.clip(predictions, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    accuracies: list[float | None] = []
    confidences: list[float | None] = []
    counts: list[int] = []
    ece = 0.0

    for idx in range(num_bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == num_bins - 1:
            mask = (predictions >= lower) & (predictions <= upper)
        else:
            mask = (predictions >= lower) & (predictions < upper)
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            accuracies.append(None)
            confidences.append(None)
            continue
        acc = float(labels[mask].mean())
        conf = float(predictions[mask].mean())
        accuracies.append(acc)
        confidences.append(conf)
        ece += abs(acc - conf) * count / len(labels)

    brier = float(np.mean((predictions - labels) ** 2))
    return {
        "ece": float(ece),
        "brier": brier,
        "bin_edges": edges.tolist(),
        "bin_counts": counts,
        "bin_accuracies": accuracies,
        "bin_confidences": confidences,
        "num_samples": int(len(labels)),
        "num_positive": int(labels.sum()),
        "num_negative": int((labels == 0).sum()),
        "mean_prediction": float(predictions.mean()),
        "mean_label": float(labels.mean()),
    }


def plot_reliability(
    *,
    results_root: Path,
    output_dir: Path,
    feature_name: str,
    num_bins: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"llava_msts_all_datasets_sources_{feature_slug(feature_name)}_reliability.pdf"

    family_payloads = []
    for family in PROBE_FAMILIES:
        path = predictions_path(results_root, family, feature_name)
        preds, labels = load_prediction_mean(path)
        bins = calibration_bins(preds, labels, num_bins)
        family_payloads.append((family, bins, path))

    fig, axes = plt.subplots(1, len(family_payloads), figsize=(15, 5), sharex=True, sharey=True)
    if len(family_payloads) == 1:
        axes = [axes]

    for ax, (family, bins, path) in zip(axes, family_payloads):
        edges = np.array(bins["bin_edges"], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = (edges[1] - edges[0]) * 0.9
        accuracies = np.array(
            [np.nan if value is None else float(value) for value in bins["bin_accuracies"]],
            dtype=float,
        )
        counts = np.array(bins["bin_counts"], dtype=int)
        valid = ~np.isnan(accuracies)

        ax.bar(
            centers[valid],
            accuracies[valid],
            width=width,
            color=family.color,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
            label="Empirical accuracy",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.8, alpha=0.7)
        for center, acc, count in zip(centers[valid], accuracies[valid], counts[valid]):
            ax.text(center, min(acc + 0.035, 0.98), f"n={count}", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{family.label}\nECE={bins['ece']:.3f}, Brier={bins['brier']:.3f}")
        ax.set_xlabel("Predicted correctness")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.text(
            0.03,
            0.94,
            f"mean pred={bins['mean_prediction']:.3f}\nlabel mean={bins['mean_label']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )

    axes[0].set_ylabel("Empirical accuracy")
    fig.suptitle("MSTS Reliability Diagrams: All-Datasets Probe Predictions", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "feature_name": feature_name,
        "num_bins": num_bins,
        "prediction_row": "ensemble_mean",
        "families": [
            {"family": family.key, "label": family.label, "predictions_path": str(path), **bins}
            for family, bins, path in family_payloads
        ],
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reliability diagrams for all-datasets probes evaluated on MSTS."
    )
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature_name", type=str, default=DEFAULT_FEATURE_NAME)
    parser.add_argument("--num_bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_reliability(
        results_root=args.results_root,
        output_dir=args.output_dir,
        feature_name=args.feature_name,
        num_bins=args.num_bins,
    )
    print(f"plot_pdf: {output_path}")
    print(f"plot_png: {output_path.with_suffix('.png')}")
    print(f"plot_json: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
