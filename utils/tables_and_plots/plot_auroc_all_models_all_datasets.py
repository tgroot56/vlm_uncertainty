from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_CONFIGS = {
    "llava/coco-qa-vi": {
        "true": Path("probe_results/llava/coco-qa-vi"),
        "shuffle": Path("probe_results/llava/coco-qa-vi/baselines"),
    },
    "llava/imagenet-r": {
        "true": Path("probe_results/llava/imagenet-r/16mc-clip"),
        "shuffle": Path("probe_results/llava/imagenet-r/16mc-clip/baselines"),
    },
    "llava/pope/random": {
        "true": Path("probe_results/llava/pope/random"),
        "shuffle": Path("probe_results/llava/pope/random/baselines"),
    },
    "qwen/coco-qa-vi": {
        "true": Path("probe_results/qwen/coco-qa-vi"),
        "shuffle": Path("probe_results/qwen/coco-qa-vi/baselines"),
    },
    "qwen/imagenet-r": {
        "true": Path("probe_results/qwen/imagenet-r"),
        "shuffle": Path("probe_results/qwen/imagenet-r/baselines"),
    },
    "qwen/pope/random": {
        "true": Path("probe_results/qwen/pope/random"),
        "shuffle": Path("probe_results/qwen/pope/random/baselines"),
    },
}

OUTDIR = Path("probe_results/_plots/brier_grouped_clean/all_models_all_datasets/auroc")
MODEL_TYPES = {"linear", "mlp"}
VQA_PLACEHOLDER_DATASETS = {"llava/vqa-v2", "qwen/vqa-v2"}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_feature_names(run_dir: Path) -> list[str]:
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            feature_names = _load_json(cfg).get("feature_names")
        except Exception:
            feature_names = None
        if isinstance(feature_names, list) and all(isinstance(x, str) for x in feature_names):
            return feature_names

    parts = [part for part in run_dir.name.split("__") if part]
    return parts if parts else [run_dir.name]


def safe_read_auroc_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "feature_label" not in df.columns:
        return None

    df.columns = [column.strip() for column in df.columns]
    for column in ["linear", "mlp"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan
    return df[["feature_label", "linear", "mlp"]]


def collect_auroc_from_metrics(root: Path) -> Optional[pd.DataFrame]:
    rows = []
    for metrics_path in root.rglob("metrics_test.json"):
        run_dir = metrics_path.parent
        model_type = next(
            (parent.name for parent in (run_dir,) + tuple(run_dir.parents) if parent.name in MODEL_TYPES),
            None,
        )
        if model_type is None:
            continue

        try:
            metrics = _load_json(metrics_path)
        except Exception:
            continue

        feature_names = _infer_feature_names(run_dir)
        feature_label = feature_names[0] if len(feature_names) == 1 else " + ".join(feature_names)
        rows.append({"feature_label": feature_label, model_type: metrics.get("test_auroc", np.nan)})

    if not rows:
        return None

    df = pd.DataFrame(rows).groupby("feature_label", as_index=False).first()
    for column in ["linear", "mlp"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[["feature_label", "linear", "mlp"]]


def load_auroc(root: Path) -> Optional[pd.DataFrame]:
    csv_df = safe_read_auroc_csv(root / "results_auroc.csv")
    if csv_df is not None:
        return csv_df
    return collect_auroc_from_metrics(root)


def load_merged_auroc(dataset: str) -> Optional[pd.DataFrame]:
    if dataset in VQA_PLACEHOLDER_DATASETS:
        return None

    config = DATASET_CONFIGS.get(dataset)
    if config is None:
        return None

    true_df = load_auroc(config["true"])
    shuffle_df = load_auroc(config["shuffle"])
    if true_df is None or shuffle_df is None:
        return None

    true_df = true_df.rename(columns={"linear": "linear_true", "mlp": "mlp_true"})
    shuffle_df = shuffle_df.rename(columns={"linear": "linear_shuffle", "mlp": "mlp_shuffle"})
    return pd.merge(true_df, shuffle_df, on="feature_label", how="inner")


def plot_model_comparison_component_comparison_mlp(dfs: dict[str, Optional[pd.DataFrame]]) -> None:
    selected_features = {
        "vision_mean_layer_-1": "Vision (mean)",
        "lm_visual_mean_layer_-1": "Lm Visual (mean)",
        "lm_question_lasttok_layer_-1": "LM Question (lasttok)",
        "lm_fullspan_lasttok_layer_-1": "LM Final Token",
        "lm_answer_lasttok_layer_-1": "LM Answer (lasttok)",
        "answer_gen_entropy_mean": "Entropy (mean)",
        "answer_gen_neglogp_mean": "NegLogP (mean)",
    }
    components = list(selected_features.values())
    columns = [
        ("vqa-v2", "VQA-v2"),
        ("coco-qa-vi", "COCO-QA-VI"),
        ("pope/random", "POPE Random"),
        ("imagenet-r", "ImageNet-R"),
    ]
    rows = [("llava", "LLaVA"), ("qwen", "Qwen")]

    fig, axes = plt.subplots(len(rows), len(columns), figsize=(28, 10), sharex=True, sharey=True)

    for row_idx, (model, model_label) in enumerate(rows):
        for col_idx, (dataset_suffix, dataset_label) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            dataset = f"{model}/{dataset_suffix}"
            df = dfs.get(dataset)

            if row_idx == 0:
                ax.set_title(dataset_label, fontsize=15)

            if dataset in VQA_PLACEHOLDER_DATASETS:
                ax.text(0.5, 0.5, "VQA-v2 AUROC\nplaceholder", ha="center", va="center", fontsize=13)
                ax.set_xticks([])
                ax.set_yticks(np.arange(0.0, 1.01, 0.1))
                ax.grid(axis="y", linestyle="--", alpha=0.35)
                if col_idx == 0:
                    ax.set_ylabel(f"{model_label}\nAUROC", fontsize=14)
                continue

            if df is None:
                ax.text(0.5, 0.5, "Data not available", ha="center", va="center", fontsize=13)
                ax.set_xticks([])
                if col_idx == 0:
                    ax.set_ylabel(f"{model_label}\nAUROC", fontsize=14)
                continue

            subset = df[df["feature_label"].isin(selected_features.keys())].copy()
            if subset.empty:
                ax.text(0.5, 0.5, "Data not available", ha="center", va="center", fontsize=13)
                ax.set_xticks([])
                if col_idx == 0:
                    ax.set_ylabel(f"{model_label}\nAUROC", fontsize=14)
                continue

            subset["Component"] = subset["feature_label"].map(selected_features)
            subset_sorted = subset.set_index("Component").reindex(components).dropna(subset=["mlp_true"])
            current_components = subset_sorted.index.tolist()

            x = np.arange(len(current_components))
            y_true = subset_sorted["mlp_true"].to_numpy()
            y_shuffle = subset_sorted["mlp_shuffle"].to_numpy()

            ax.axhline(0.5, linestyle="--", linewidth=2, color="black", label="No-skill baseline")
            ax.axhspan(0.0, 0.5, alpha=0.04, color="gray")
            ax.bar(x, y_true, color="C0", edgecolor="black", label="Probe (true labels)")
            if np.isfinite(y_shuffle).any():
                ax.bar(
                    x,
                    y_shuffle,
                    color="none",
                    edgecolor="black",
                    hatch="//",
                    linewidth=1.2,
                    label="Shuffled-label control",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(current_components, rotation=45, ha="right", fontsize=10)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.arange(0.0, 1.01, 0.1))
            ax.grid(axis="y", linestyle="--", alpha=0.6)

            if col_idx == 0:
                ax.set_ylabel(f"{model_label}\nAUROC", fontsize=14)

            if row_idx == 0:
                ax.tick_params(axis="x", labelbottom=False)

    handles, labels = [], []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))

    plt.suptitle("Component Comparison - MLP Probe (AUROC)", fontsize=18, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.925))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTDIR / "model_comparison_component_comparison_mlp_auroc.pdf"
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath} (+ png)")


def main() -> None:
    datasets = [
        "llava/vqa-v2",
        "llava/coco-qa-vi",
        "llava/pope/random",
        "llava/imagenet-r",
        "qwen/vqa-v2",
        "qwen/coco-qa-vi",
        "qwen/pope/random",
        "qwen/imagenet-r",
    ]
    dfs = {dataset: load_merged_auroc(dataset) for dataset in datasets}
    plot_model_comparison_component_comparison_mlp(dfs)


if __name__ == "__main__":
    main()
