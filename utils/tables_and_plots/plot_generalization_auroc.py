from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Allow direct execution via `python utils/.../plot_generalization_auroc.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generalization.paths import (
    DEFAULT_OUTPUT_ROOT,
    all_datasets_feature_root,
    model_gen_feature_root,
    plot_root,
    task_gen_feature_root,
)

DEFAULT_ROOT = Path(DEFAULT_OUTPUT_ROOT)
DEFAULT_DATASETS = ("coco-qa-vi", "vqa-v2", "imagenet-r", "pope/random")
DEFAULT_ALL_DATASETS_FEATURE = "lm_answer_mean_layer_16"
DEFAULT_FEATURE_NAMES = (
    "lm_answer_mean_layer_16",
    "answer_gen_entropy_mean",
    "lm_fullspan_lasttok_layer_16",
)

MODEL_DISPLAY = {
    "llava": "LLaVA",
    "qwen": "Qwen",
}

DATASET_DISPLAY = {
    "coco-qa-vi": "COCO-QA-VI",
    "vqa-v2": "VQA-v2",
    "imagenet-r": "ImageNet-R",
    "pope/random": "POPE Random",
}

VARIANT_DISPLAY = {
    "leave_one_out": "Leave-one-out",
    "in_domain": "In-domain",
    "in_domain_same_dataset_count": "In-domain (same own-data count)",
    "shuffled_label_control": "Shuffled-label control",
    "train_all_datasets": "Train on all datasets",
    "llava_to_qwen": "LLaVA -> Qwen",
    "qwen_to_llava": "Qwen -> LLaVA",
    "llava_to_qwen_shuffled": "LLaVA -> Qwen (shuffled)",
    "qwen_to_llava_shuffled": "Qwen -> LLaVA (shuffled)",
}


def _feature_slug(feature_name: str) -> str:
    return feature_name.strip().replace("/", "_").replace(" ", "_")


def _mode_uses_feature_specific_summaries(mode: str) -> bool:
    return mode in {"leave_one_out", "all_datasets", "model_gen"}


def canonicalize_model_name(model_name: str) -> str:
    key = model_name.strip().lower().replace("_", "-")
    if key in {"qwen", "qwen-3.5-9b", "qwen3.5-9b", "qwen/qwen3.5-9b"}:
        return "qwen"
    if key in {"llava", "llava-1.5-7b", "llava1.5-7b", "llava-hf/llava-1.5-7b-hf"}:
        return "llava"
    raise ValueError(f"Unsupported model name: {model_name!r}")


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_generalization_aurocs(
    root: Path,
    model_name: str,
    datasets: List[str],
    feature_name: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    collected: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in datasets:
        summary_path = task_gen_feature_root(str(root), model_name, dataset, feature_name) / "comparison_summary.json"
        if not summary_path.exists():
            print(f"[skip] Missing summary: {summary_path}")
            continue

        summary = _load_json(summary_path)
        variant_scores: Dict[str, Dict[str, float]] = {}
        for variant in ("leave_one_out", "in_domain"):
            variant_summary = summary.get("variants", {}).get(variant, {})
            if "mean_metrics" in variant_summary:
                mean_value = variant_summary.get("mean_metrics", {}).get("test_auroc")
                std_value = variant_summary.get("std_metrics", {}).get("test_auroc")
            else:
                mean_value = variant_summary.get("metrics", {}).get("test_auroc")
                std_value = 0.0 if mean_value is not None else np.nan

            variant_scores[variant] = {
                "mean": np.nan if mean_value is None else float(mean_value),
                "std": np.nan if std_value is None else float(std_value),
            }
        collected[dataset] = variant_scores
    return collected


def collect_all_datasets_aurocs(
    root: Path,
    model_name: str,
    datasets: List[str],
    feature_name: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    collected: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in datasets:
        summary_path = all_datasets_feature_root(str(root), model_name, dataset, feature_name) / "comparison_summary.json"
        if not summary_path.exists():
            print(f"[skip] Missing summary: {summary_path}")
            continue

        summary = _load_json(summary_path)
        dataset_scores: Dict[str, Dict[str, float]] = {}
        for variant in (
            "train_all_datasets",
            "in_domain",
            "in_domain_same_dataset_count",
        ):
            dataset_summary = summary.get("variants", {}).get(variant, {})
            if "mean_metrics" in dataset_summary:
                mean_value = dataset_summary.get("mean_metrics", {}).get("test_auroc")
                std_value = dataset_summary.get("std_metrics", {}).get("test_auroc")
            else:
                mean_value = dataset_summary.get("metrics", {}).get("test_auroc")
                std_value = 0.0 if mean_value is not None else np.nan

            dataset_scores[variant] = {
                "mean": np.nan if mean_value is None else float(mean_value),
                "std": np.nan if std_value is None else float(std_value),
            }
        collected[dataset] = dataset_scores
    return collected


def collect_model_gen_aurocs(
    root: Path,
    datasets: List[str],
    feature_name: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    direction_paths = {
        "llava_to_qwen": "llava_to_qwen",
        "qwen_to_llava": "qwen_to_llava",
    }
    collected: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in datasets:
        dataset_scores: Dict[str, Dict[str, float]] = {}
        found_any = False
        for direction_key, direction_model_key in direction_paths.items():
            summary_path = model_gen_feature_root(str(root), direction_model_key, dataset, feature_name) / "comparison_summary.json"
            if not summary_path.exists():
                print(f"[skip] Missing summary: {summary_path}")
                dataset_scores[direction_key] = {"mean": np.nan, "std": np.nan}
                continue

            summary = _load_json(summary_path)
            found_any = True
            for summary_variant, plot_variant in (
                ("model_gen", direction_key),
            ):
                variant_summary = summary.get("variants", {}).get(summary_variant, {})
                if "mean_metrics" in variant_summary:
                    mean_value = variant_summary.get("mean_metrics", {}).get("test_auroc")
                    std_value = variant_summary.get("std_metrics", {}).get("test_auroc")
                else:
                    mean_value = variant_summary.get("metrics", {}).get("test_auroc")
                    std_value = 0.0 if mean_value is not None else np.nan

                dataset_scores[plot_variant] = {
                    "mean": np.nan if mean_value is None else float(mean_value),
                    "std": np.nan if std_value is None else float(std_value),
                }
        if found_any:
            collected[dataset] = dataset_scores
    return collected


def plot_generalization_auroc(
    *,
    root: Path,
    model_name: str,
    datasets: List[str],
    mode: str = "leave_one_out",
    feature_name: str = DEFAULT_ALL_DATASETS_FEATURE,
    output_path: Path | None = None,
) -> Path:
    if mode == "leave_one_out":
        scores = collect_generalization_aurocs(
            root=root,
            model_name=model_name,
            datasets=datasets,
            feature_name=feature_name,
        )
        variant_order = ("leave_one_out", "in_domain")
        title = (
            f"Generalization Experiment AUROC - {MODEL_DISPLAY.get(model_name, model_name.title())}\n"
            f"Feature: {feature_name} (Mean +/- Std Across Seeds)"
        )
        default_filename = f"{model_name}_generalization_{_feature_slug(feature_name)}_auroc.pdf"
    elif mode == "all_datasets":
        scores = collect_all_datasets_aurocs(
            root=root,
            model_name=model_name,
            datasets=datasets,
            feature_name=feature_name,
        )
        variant_order = (
            "train_all_datasets",
            "in_domain",
            "in_domain_same_dataset_count",
        )
        title = (
            f"All-Datasets Probe AUROC - {MODEL_DISPLAY.get(model_name, model_name.title())}\n"
            f"Feature: {feature_name} (Mean +/- Std Across Seeds)"
        )
        default_filename = f"{model_name}_all_datasets_{_feature_slug(feature_name)}_auroc.pdf"
    elif mode == "model_gen":
        scores = collect_model_gen_aurocs(
            root=root,
            datasets=datasets,
            feature_name=feature_name,
        )
        variant_order = (
            "llava_to_qwen",
            "qwen_to_llava",
        )
        title = (
            "Cross-Model Generalization AUROC\n"
            f"Feature: {feature_name} (Mean +/- Std Across Seeds)"
        )
        default_filename = f"model_gen_{_feature_slug(feature_name)}_auroc.pdf"
    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'leave_one_out', 'all_datasets', or 'model_gen'.")

    available_datasets = [dataset for dataset in datasets if dataset in scores]
    if not available_datasets:
        raise FileNotFoundError(
            f"No comparison_summary.json files found for model {model_name!r} under {root}."
        )

    if output_path is None:
        output_dir = plot_root(str(root), mode)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / default_filename
    else:
        feature_slug = _feature_slug(feature_name)
        if feature_slug not in output_path.stem:
            output_path = output_path.with_name(f"{output_path.stem}_{feature_slug}{output_path.suffix}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(available_datasets))
    width_lookup = {
        2: 0.32,
        3: 0.24,
        4: 0.19,
    }
    width = width_lookup.get(len(variant_order), 0.18)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhline(0.5, linestyle="--", linewidth=2, color="black", alpha=0.7, label="Chance AUROC")

    color_map = {
        "leave_one_out": "#2c7fb8",
        "in_domain": "#41ab5d",
        "in_domain_same_dataset_count": "#74c476",
        "shuffled_label_control": "#9ecae1",
        "train_all_datasets": "#8856a7",
        "llava_to_qwen": "#8856a7",
        "qwen_to_llava": "#2b8cbe",
        "llava_to_qwen_shuffled": "#cab2d6",
        "qwen_to_llava_shuffled": "#9ecae1",
    }
    hatch_map = {
        "shuffled_label_control": "//",
        "llava_to_qwen_shuffled": "//",
        "qwen_to_llava_shuffled": "//",
    }

    bars_by_variant = []
    if len(variant_order) == 4:
        offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    elif len(variant_order) == 3:
        offsets = (-width, 0.0, width)
    elif len(variant_order) == 2:
        offsets = (-width / 2.0, width / 2.0)
    else:
        offsets = [0.0] * len(variant_order)

    for variant, offset in zip(variant_order, offsets):
        variant_scores = np.array([scores[d][variant]["mean"] for d in available_datasets], dtype=float)
        variant_stds = np.array([scores[d][variant]["std"] for d in available_datasets], dtype=float)
        bars = ax.bar(
            x + offset,
            variant_scores,
            width=width,
            yerr=variant_stds,
            capsize=4,
            color=color_map[variant],
            edgecolor="black",
            linewidth=1.2,
            hatch=hatch_map.get(variant, None),
            label=VARIANT_DISPLAY[variant],
        )
        bars_by_variant.append(bars)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY.get(dataset, dataset) for dataset in available_datasets], rotation=20, ha="right")
    ax.set_ylabel("Test AUROC (Higher is better)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True)

    for bars in bars_by_variant:
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path} (+ png)")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot test AUROC for the generalization experiment."
    )
    parser.add_argument("--model", required=True, help="Model name: llava or qwen")
    parser.add_argument(
        "--mode",
        choices=["leave_one_out", "all_datasets", "model_gen"],
        default="leave_one_out",
        help="Which generalization experiment layout to plot.",
    )
    parser.add_argument(
        "--feature_name",
        default=None,
        help=(
            "Optional feature name to plot. If omitted, plots are generated for: "
            f"{', '.join(DEFAULT_FEATURE_NAMES)}"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing generalization experiment outputs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Dataset(s) to include. Can be passed multiple times. "
            f"Defaults to: {', '.join(DEFAULT_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output PDF path. PNG is saved alongside it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = canonicalize_model_name(args.model)
    datasets = args.dataset if args.dataset is not None else list(DEFAULT_DATASETS)
    if args.feature_name is not None:
        feature_names = [args.feature_name]
    elif _mode_uses_feature_specific_summaries(args.mode):
        feature_names = list(DEFAULT_FEATURE_NAMES)

    for feature_name in feature_names:
        output_path = args.output
        if output_path is not None and len(feature_names) > 1:
            feature_slug = _feature_slug(feature_name)
            output_path = output_path.with_name(f"{output_path.stem}_{feature_slug}{output_path.suffix}")

        plot_generalization_auroc(
            root=args.root,
            model_name=model_name,
            datasets=datasets,
            mode=args.mode,
            feature_name=feature_name,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
