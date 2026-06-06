from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generalization.paths import DEFAULT_OUTPUT_ROOT
from src.generalization.train_leave_one_out import _train_probe_run
from src.probe.data import build_feature_slice_map
from utils.probe_viewer.msts_all_datasets_probe_app import (
    DEFAULT_ANNOTATION_CSV,
    DEFAULT_FEATURE_NAME,
    DEFAULT_MSTS_RUN_DIR,
    ProbeSpec,
    evaluate_probe_specs,
)
from utils.tables_and_plots.plot_style import PROBE_PLOT_FONT_SIZES


DEFAULT_EVAL_ROOT = PROJECT_ROOT / "probe_results/msts_eval"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "probe_results/_plots/generalization_msts_auroc"
DEFAULT_MODEL_DATASET_PATHS = {
    "llava": Path(DEFAULT_OUTPUT_ROOT)
    / "generalization/datasets/llava/generalizability_supervision_dataset.pt",
    "qwen": Path(DEFAULT_OUTPUT_ROOT)
    / "generalization/datasets/qwen/generalizability_supervision_dataset.pt",
}


@dataclass(frozen=True)
class ProbeFamily:
    key: str
    label: str
    model_key: str
    checkpoint_model_key: str
    color: str


PROBE_FAMILIES = (
    ProbeFamily("llava", "LLaVA-trained", "llava", "llava", "#2c7fb8"),
    ProbeFamily("qwen", "Qwen-trained", "qwen", "qwen", "#41ab5d"),
    ProbeFamily("both_models", "LLaVA+Qwen-trained", "both_models", "both_models", "#8856a7"),
)


def feature_slug(feature_name: str) -> str:
    return feature_name.strip().replace("/", "_").replace(" ", "_")


def checkpoint_root(output_root: Path, model_key: str, feature_name: str) -> Path:
    return output_root / "generalization/all_datasets" / model_key / "_shared_train" / feature_name / "seed_runs"


def checkpoint_path(output_root: Path, model_key: str, feature_name: str, seed: int) -> Path:
    return (
        checkpoint_root(output_root, model_key, feature_name)
        / f"seed_{seed}"
        / "train_all_datasets"
        / "shared_train"
        / "mlp"
        / feature_name
        / "best_model.pt"
    )


def probe_specs(output_root: Path, family: ProbeFamily, feature_name: str, seeds: Sequence[int]) -> list[ProbeSpec]:
    return [
        ProbeSpec(
            name=f"{family.key}_seed_{seed}",
            checkpoint_path=checkpoint_path(output_root, family.checkpoint_model_key, feature_name, seed),
        )
        for seed in seeds
    ]


def select_feature(payload: dict, feature_name: str) -> torch.Tensor:
    x_all = payload["X"].float()
    slices = build_feature_slice_map(payload["feature_names"], payload.get("feature_dims"), x_all.shape[1])
    if feature_name not in slices:
        raise ValueError(f"Missing feature {feature_name!r}; available: {list(slices)}")
    return x_all[:, slices[feature_name]].float()


def select_split(payload: dict, feature_name: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.tensor([value == split for value in payload["split_labels"]], dtype=torch.bool)
    return select_feature(payload, feature_name)[mask], payload["y"].float()[mask]


def selection_summary(payloads: dict[str, dict], split: str) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for model_key, payload in payloads.items():
        model_summary: dict[str, int] = {}
        for dataset_name, split_label in zip(payload["dataset_names"], payload["split_labels"]):
            if split_label == split:
                key = f"{model_key}/{dataset_name}"
                model_summary[key] = model_summary.get(key, 0) + 1
        summary.update({key: {split: value} for key, value in model_summary.items()})
    return summary


def train_both_models_probe(
    *,
    output_root: Path,
    feature_name: str,
    seeds: Sequence[int],
    dataset_paths: dict[str, Path],
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    activation: str,
    device_name: str,
    force: bool,
) -> None:
    missing = [
        path
        for seed in seeds
        for path in [checkpoint_path(output_root, "both_models", feature_name, seed)]
        if force or not path.exists()
    ]
    if not missing:
        return

    payloads = {
        model_key: torch.load(path, map_location="cpu", weights_only=False)
        for model_key, path in dataset_paths.items()
    }
    train_parts = [select_split(payload, feature_name, "train") for payload in payloads.values()]
    val_parts = [select_split(payload, feature_name, "val") for payload in payloads.values()]
    train_X = torch.cat([part[0] for part in train_parts], dim=0)
    train_y = torch.cat([part[1] for part in train_parts], dim=0)
    val_X = torch.cat([part[0] for part in val_parts], dim=0)
    val_y = torch.cat([part[1] for part in val_parts], dim=0)

    input_dims = {model_key: select_feature(payload, feature_name).shape[1] for model_key, payload in payloads.items()}
    if len(set(input_dims.values())) != 1:
        raise ValueError(f"Cannot train pooled hidden-state probe with mismatched dims: {input_dims}")

    device = torch.device(device_name)
    for seed in seeds:
        target = checkpoint_path(output_root, "both_models", feature_name, seed)
        if target.exists() and not force:
            continue
        shared_seed_root = checkpoint_root(output_root, "both_models", feature_name) / f"seed_{seed}"
        _train_probe_run(
            feature_names=[feature_name],
            output_dir=str(shared_seed_root / "train_all_datasets" / "shared_train"),
            run_variant="train_all_datasets_both_models",
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            train_selection_summary=selection_summary(payloads, "train"),
            val_selection_summary=selection_summary(payloads, "val"),
            sampling_info={"source_models": sorted(payloads.keys())},
            model_type="mlp",
            hidden_dims=None,
            dropout=dropout,
            activation=activation,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            normalize=True,
            seed=seed,
            device=device,
            use_class_weights=False,
            shuffle_train_labels=False,
        )


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def family_eval_dir(eval_root: Path, family: ProbeFamily, target_model: str, feature_name: str) -> Path:
    return eval_root / target_model / f"{family.key}_{feature_slug(feature_name)}"


def evaluate_family(
    *,
    output_root: Path,
    eval_root: Path,
    family: ProbeFamily,
    target_model: str,
    feature_name: str,
    seeds: Sequence[int],
    annotation_csv: Path,
    msts_run_dir: Path,
    batch_size: int,
    device: str,
    skip_eval: bool,
) -> Path:
    specs = probe_specs(output_root, family, feature_name, seeds)
    missing = [spec.checkpoint_path for spec in specs if not spec.checkpoint_path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(str(path) for path in missing))

    output_dir = family_eval_dir(eval_root, family, target_model, feature_name)
    metrics_path = output_dir / "metrics.csv"
    if not skip_eval or not metrics_path.exists():
        evaluate_probe_specs(
            probe_specs=specs,
            feature_name=feature_name,
            annotation_csv=annotation_csv,
            msts_run_dir=msts_run_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            device_name=device,
        )
    return metrics_path


def aggregate_family_metrics(metrics_path: Path, family: ProbeFamily) -> dict[str, object]:
    rows = read_metrics(metrics_path)
    seed_rows = [row for row in rows if row["probe_name"].startswith(f"{family.key}_seed_")]
    ensemble_rows = [row for row in rows if row["probe_name"] == "ensemble_mean"]
    if not seed_rows:
        raise RuntimeError(f"No seed rows for {family.key} in {metrics_path}")
    aurocs = np.array([float(row["auroc"]) for row in seed_rows], dtype=float)
    prediction_mean_auroc = float(ensemble_rows[0]["auroc"]) if ensemble_rows else float("nan")
    first = seed_rows[0]
    return {
        "family": family.key,
        "label": family.label,
        "mean_auroc": float(aurocs.mean()),
        "std_auroc": float(aurocs.std(ddof=0)),
        "seed_aurocs": aurocs.tolist(),
        "prediction_mean_across_seeds_auroc": prediction_mean_auroc,
        "num_samples": int(float(first["num_samples"])),
        "num_positive": int(float(first["num_positive"])),
        "num_negative": int(float(first["num_negative"])),
        "metrics_path": str(metrics_path),
    }


def plot_auroc(
    *,
    aggregates: list[dict[str, object]],
    families: Sequence[ProbeFamily],
    output_dir: Path,
    target_model: str,
    feature_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target_model}_msts_all_datasets_sources_{feature_slug(feature_name)}_auroc.pdf"

    labels = [str(item["label"]) for item in aggregates]
    means = np.array([float(item["mean_auroc"]) for item in aggregates], dtype=float)
    stds = np.array([float(item["std_auroc"]) for item in aggregates], dtype=float)
    ensembles = np.array([float(item["prediction_mean_across_seeds_auroc"]) for item in aggregates], dtype=float)
    family_by_key = {family.key: family for family in families}
    colors = [family_by_key[str(item["family"])].color for item in aggregates]
    fonts = PROBE_PLOT_FONT_SIZES
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(aggregates))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=1.1,
        label="Mean across seeds",
    )
    ax.scatter(
        x,
        ensembles,
        marker="D",
        s=64,
        color="white",
        edgecolor="black",
        zorder=4,
        label="Prediction mean across seeds",
    )
    ax.axhline(0.5, linestyle="--", linewidth=2, color="black", alpha=0.7, label="Chance AUROC")

    for idx, (bar, mean, ensemble) in enumerate(zip(bars, means, ensembles)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + stds[idx] + 0.025,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=fonts.annotation + 1,
        )
        if not np.isnan(ensemble):
            ax.text(
                idx,
                ensemble - 0.045,
                f"{ensemble:.3f}",
                ha="center",
                va="top",
                fontsize=fonts.annotation,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fonts.tick_label)
    ax.set_ylabel("Test AUROC (Higher is better)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True)
    ax.set_title("All Datasets Probe Generalization to MSTS")

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"feature_name": feature_name, "target_model": target_model, "families": aggregates}, handle, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all-datasets LLaVA/Qwen/pooled probes on annotated MSTS and plot AUROC."
        )
    )
    parser.add_argument("--annotation_csv", type=Path, default=DEFAULT_ANNOTATION_CSV)
    parser.add_argument("--msts_run_dir", type=Path, default=Path(DEFAULT_MSTS_RUN_DIR))
    parser.add_argument("--target_model", type=str, default="llava", choices=["llava", "qwen"])
    parser.add_argument("--output_root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--eval_root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--plot_dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--feature_name", type=str, default=DEFAULT_FEATURE_NAME)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "elu"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_train_both", action="store_true")
    parser.add_argument("--force_train_both", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target_model == "qwen" and "qwen" not in str(args.msts_run_dir).lower():
        print(
            "[warning] target_model=qwen but msts_run_dir does not look like a Qwen MSTS run. "
            "Hidden-state probe evaluation is only clean when target activations match target_model.",
            file=sys.stderr,
        )

    if not args.skip_train_both:
        train_both_models_probe(
            output_root=args.output_root,
            feature_name=args.feature_name,
            seeds=args.seeds,
            dataset_paths=DEFAULT_MODEL_DATASET_PATHS,
            batch_size=args.train_batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            activation=args.activation,
            device_name=args.device,
            force=args.force_train_both,
        )

    aggregates = []
    for family in PROBE_FAMILIES:
        metrics_path = evaluate_family(
            output_root=args.output_root,
            eval_root=args.eval_root,
            family=family,
            target_model=args.target_model,
            feature_name=args.feature_name,
            seeds=args.seeds,
            annotation_csv=args.annotation_csv,
            msts_run_dir=args.msts_run_dir,
            batch_size=args.batch_size,
            device=args.device,
            skip_eval=args.skip_eval,
        )
        aggregates.append(aggregate_family_metrics(metrics_path, family))

    output_path = plot_auroc(
        aggregates=aggregates,
        families=PROBE_FAMILIES,
        output_dir=args.plot_dir,
        target_model=args.target_model,
        feature_name=args.feature_name,
    )
    print(f"plot_pdf: {output_path}")
    print(f"plot_png: {output_path.with_suffix('.png')}")
    print(f"plot_json: {output_path.with_suffix('.json')}")
    for item in aggregates:
        print(
            f"{item['label']}: seed mean AUROC={float(item['mean_auroc']):.4f} "
            f"+/- {float(item['std_auroc']):.4f}; "
            f"prediction-mean={float(item['prediction_mean_across_seeds_auroc']):.4f}"
        )


if __name__ == "__main__":
    main()
