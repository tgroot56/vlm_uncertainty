from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULTS_ROOT = Path(
    "/projects/prjs2014/llava/layer_experiment/probe_results/imagenet-r/mlp_layer_comparison/mlp"
)
DEFAULT_OUTPUT_PDF = Path("probe_results/layer_experiment/llava_imagenet_r_mlp_layer_brier.pdf")

FEATURE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "lm_visual_mean": re.compile(r"^lm_visual_mean_layer_(\d+)$"),
    "lm_question_mean": re.compile(r"^lm_question_mean_layer_(\d+)$"),
    "lm_answer_mean": re.compile(r"^lm_answer_mean_layer_(\d+)$"),
    "lm_fullspan_lasttok": re.compile(r"^lm_fullspan_lasttok_layer_(\d+)$"),
}

SERIES_LABELS = {
    "lm_visual_mean": "Visual span mean",
    "lm_question_mean": "Question span mean",
    "lm_answer_mean": "Answer span mean",
    "lm_fullspan_lasttok": "Full span last token",
}

SERIES_COLORS = {
    "lm_visual_mean": "#0f766e",
    "lm_question_mean": "#c2410c",
    "lm_answer_mean": "#4338ca",
    "lm_fullspan_lasttok": "#b91c1c",
}

ENTROPY_FEATURE_NAME = "answer_gen_entropy_mean"
ENTROPY_LABEL = "Mean entropy"
ENTROPY_COLOR = "#4b5563"
ENTROPY_X = 35

# Fill these in with the layer-experiment probe result roots you want to plot.
# Example:
# Path("/projects/prjs2014/llava/layer_experiment/probe_results/pope/random/mlp_layer_comparison/mlp")
MODEL_DATASET_CONFIGS: Dict[str, Dict[str, Dict[str, Path | str]]] = {
    "llava": {
        "imagenet-r": {
            "results_root": DEFAULT_RESULTS_ROOT,
            "output_pdf": DEFAULT_OUTPUT_PDF,
            "title": "LLaVA ImageNet-R: uncertainty signal across LM layers",
        },
        "pope/random": {
            "results_root": Path(
                "/projects/prjs2014/llava/layer_experiment/probe_results/pope/random/mlp_layer_comparison/mlp"
            ),
            "output_pdf": Path("probe_results/layer_experiment/llava_pope_random_mlp_layer_brier.pdf"),
            "title": "LLaVA POPE/random: uncertainty signal across LM layers",
        },
        "vqa-v2": {
            "results_root": Path(
                "/TODO/fill_in_llava_vqa_v2_layer_experiment_results_root/mlp"
            ),
            "output_pdf": Path("probe_results/layer_experiment/llava_vqa_v2_mlp_layer_brier.pdf"),
            "title": "LLaVA VQA-v2: uncertainty signal across LM layers",
        },
        "coco-qa-vi": {
            "results_root": Path(
                "/TODO/fill_in_llava_coco_qa_vi_layer_experiment_results_root/mlp"
            ),
            "output_pdf": Path("probe_results/layer_experiment/llava_coco_qa_vi_mlp_layer_brier.pdf"),
            "title": "LLaVA COCO-QA-VI: uncertainty signal across LM layers",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Brier score versus LM layer for a LLaVA layer experiment."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava",
        choices=sorted(MODEL_DATASET_CONFIGS),
        help="Model registry key used to select default paths and titles.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=sorted({dataset for datasets in MODEL_DATASET_CONFIGS.values() for dataset in datasets}),
        help=(
            "Dataset registry key used to select default paths and titles. "
            "If omitted, plots all configured datasets for --model."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Directory containing per-feature probe result folders with config.json files. "
            "Defaults to the hardcoded path for --model/--dataset."
        ),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Where to save the PDF plot. Defaults to the hardcoded path for --model/--dataset.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title. Defaults to the hardcoded title for --model/--dataset.",
    )
    parser.add_argument(
        "--entropy-results-root",
        type=Path,
        default=None,
        help=(
            "Deprecated compatibility option. Entropy is now discovered from "
            "probe_results/<model>/<dataset>/**/results_brier.csv."
        ),
    )
    parser.add_argument(
        "--entropy-brier-csv",
        type=Path,
        default=None,
        help=(
            "Optional results_brier.csv to read the entropy baseline from. "
            "Defaults to an automatic search under probe_results/<model>/<dataset>."
        ),
    )
    parser.add_argument(
        "--entropy-feature",
        type=str,
        default=ENTROPY_FEATURE_NAME,
        help="Single-feature probe name to plot as the entropy baseline.",
    )
    parser.add_argument(
        "--no-entropy",
        action="store_true",
        help="Disable the mean-entropy baseline point.",
    )
    return parser.parse_args()


def resolve_plot_config(args: argparse.Namespace, dataset: str) -> Tuple[Path, Path, str]:
    model_configs = MODEL_DATASET_CONFIGS.get(args.model)
    if model_configs is None or dataset not in model_configs:
        available = ", ".join(
            f"{model}/{dataset}"
            for model, datasets in MODEL_DATASET_CONFIGS.items()
            for dataset in datasets
        )
        raise ValueError(
            f"No hardcoded layer-experiment config for model={args.model!r}, "
            f"dataset={dataset!r}. Available: {available}"
        )

    config = model_configs[dataset]
    results_root = args.results_root or Path(config["results_root"])
    output_pdf = args.output_pdf or Path(config["output_pdf"])
    title = args.title or str(config["title"])
    return results_root, output_pdf, title


def selected_datasets(args: argparse.Namespace) -> List[str]:
    if args.dataset is not None:
        return [args.dataset]
    return sorted(MODEL_DATASET_CONFIGS[args.model])


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _score_from_config(config: Dict) -> Optional[float]:
    test_loss = config.get("test_loss")
    if test_loss is None:
        return None
    return float(test_loss)


def _iter_direct_feature_configs(results_root: Path) -> Iterable[Path]:
    yield from sorted(results_root.glob("*/config.json"))


def collect_layer_scores(results_root: Path) -> Dict[str, List[Tuple[int, float, Path]]]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    if results_root.is_file():
        raise ValueError(
            f"Results root points to a file, but this script needs the probe results "
            f"directory containing per-feature folders: {results_root}"
    )

    collected: Dict[str, List[Tuple[int, float, Path]]] = {key: [] for key in FEATURE_PATTERNS}

    for config_path in _iter_direct_feature_configs(results_root):
        config = load_json(config_path)
        feature_names = config.get("feature_names") or []
        if len(feature_names) != 1:
            continue

        feature_name = str(feature_names[0])
        score = _score_from_config(config)
        if score is None:
            continue

        for family, pattern in FEATURE_PATTERNS.items():
            match = pattern.match(feature_name)
            if not match:
                continue

            layer_idx = int(match.group(1))
            collected[family].append((layer_idx, score, config_path.parent))
            break

    for family, points in collected.items():
        points.sort(key=lambda item: item[0])
        if not points:
            print(f"[warning] No results found for {family} under {results_root}")

    if not any(collected.values()):
        raise RuntimeError(
            "No per-layer probe results were found. Expected folders like "
            "'lm_question_mean_layer_16/config.json' under the results root."
        )

    return collected


def _dataset_probe_results_root(model: str, dataset: str) -> Path:
    return Path("probe_results") / model / Path(*dataset.split("/"))


def _entropy_score_from_brier_csv(
    csv_path: Path,
    *,
    entropy_feature: str,
    probe_column: str,
    warn_missing_value: bool = True,
) -> Optional[float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("feature_label") != entropy_feature:
                continue
            value = row.get(probe_column)
            if value is None or value == "":
                if warn_missing_value:
                    print(f"[warning] {csv_path} has no {probe_column!r} value for {entropy_feature}")
                return None
            return float(value)
    return None


def _entropy_csv_rank(csv_path: Path, search_root: Path) -> Tuple[bool, bool, bool, int, str]:
    parts = set(csv_path.parts)
    relative_depth = len(csv_path.relative_to(search_root).parts)
    return (
        "baselines" in parts,
        "generalizability" in parts,
        any(part.startswith("l2_") for part in parts),
        relative_depth,
        str(csv_path),
    )


def collect_entropy_score_from_csv(
    *,
    model: str,
    dataset: str,
    entropy_brier_csv: Optional[Path],
    entropy_feature: str,
    probe_column: str = "mlp",
) -> Optional[Tuple[float, Path]]:
    if entropy_brier_csv is not None:
        if not entropy_brier_csv.exists():
            print(f"[warning] Entropy results_brier.csv does not exist: {entropy_brier_csv}")
            return None
        score = _entropy_score_from_brier_csv(
            entropy_brier_csv,
            entropy_feature=entropy_feature,
            probe_column=probe_column,
        )
        if score is None:
            print(f"[warning] No {entropy_feature} row found in {entropy_brier_csv}")
            return None
        return score, entropy_brier_csv

    search_root = _dataset_probe_results_root(model, dataset)
    if not search_root.exists():
        print(f"[warning] Entropy search root does not exist: {search_root}")
        return None

    candidates: List[Tuple[Tuple[bool, bool, bool, int, str], float, Path]] = []
    for csv_path in sorted(search_root.rglob("results_brier.csv")):
        score = _entropy_score_from_brier_csv(
            csv_path,
            entropy_feature=entropy_feature,
            probe_column=probe_column,
            warn_missing_value=False,
        )
        if score is not None:
            candidates.append((_entropy_csv_rank(csv_path, search_root), score, csv_path))

    if candidates:
        _, score, csv_path = min(candidates, key=lambda item: item[0])
        return score, csv_path

    print(f"[warning] No {entropy_feature} row found under {search_root}")
    return None


def plot_layer_scores(
    layer_scores: Dict[str, List[Tuple[int, float, Path]]],
    *,
    entropy_score: Optional[Tuple[float, Path]],
    title: str,
    output_pdf: Path,
) -> Path:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    all_layers = sorted({layer for points in layer_scores.values() for layer, _, _ in points})
    entropy_x = ENTROPY_X

    if entropy_score is not None:
        score, _ = entropy_score
        ax.scatter(
            [entropy_x],
            [score],
            marker="D",
            s=95,
            color=ENTROPY_COLOR,
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
            label=ENTROPY_LABEL,
        )
        ax.annotate(
            f"{score:.3f}",
            xy=(entropy_x, score),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color=ENTROPY_COLOR,
        )

    for family, points in layer_scores.items():
        if not points:
            continue

        layers = [layer for layer, _, _ in points]
        brier_scores = [score for _, score, _ in points]

        ax.plot(
            layers,
            brier_scores,
            marker="o",
            linewidth=2.4,
            markersize=6,
            label=SERIES_LABELS[family],
            color=SERIES_COLORS[family],
        )

        best_idx = min(range(len(points)), key=lambda idx: brier_scores[idx])
        best_layer = layers[best_idx]
        best_score = brier_scores[best_idx]
        ax.scatter(
            [best_layer],
            [best_score],
            s=90,
            color=SERIES_COLORS[family],
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )
        ax.annotate(
            f"L{best_layer}: {best_score:.3f}",
            xy=(best_layer, best_score),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9,
            color=SERIES_COLORS[family],
        )

    ax.set_xlabel("LM layer")
    ax.set_ylabel("Brier score")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)

    if all_layers:
        ax.set_xticks(all_layers + ([entropy_x] if entropy_score is not None else []))
        if entropy_score is not None:
            ax.set_xticklabels([str(layer) for layer in all_layers] + ["Mean\nentropy"])
    elif entropy_score is not None:
        ax.set_xticks([entropy_x])
        ax.set_xticklabels(["Mean\nentropy"])

    if entropy_score is not None:
        left = (min(all_layers) - 0.75) if all_layers else entropy_x - 1
        ax.set_xlim(left, entropy_x + 1.25)

    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_png


def print_summary(
    layer_scores: Dict[str, List[Tuple[int, float, Path]]],
    entropy_score: Optional[Tuple[float, Path]],
    entropy_feature: str,
) -> None:
    if entropy_score is not None:
        score, result_dir = entropy_score
        print(f"{entropy_feature}: brier={score:.6f}, result_dir={result_dir}")

    for family, points in layer_scores.items():
        if not points:
            continue
        best_layer, best_score, best_dir = min(points, key=lambda item: item[1])
        print(
            f"{family}: best layer={best_layer}, brier={best_score:.6f}, "
            f"result_dir={best_dir}"
        )


def main() -> None:
    args = parse_args()
    datasets = selected_datasets(args)
    for dataset in datasets:
        results_root, output_pdf, title = resolve_plot_config(args, dataset)
        if len(datasets) > 1 and not results_root.exists():
            print(f"[warning] Skipping {args.model}/{dataset}; results root does not exist: {results_root}")
            continue

        layer_scores = collect_layer_scores(results_root)
        entropy_score = None
        if not args.no_entropy:
            entropy_score = collect_entropy_score_from_csv(
                model=args.model,
                dataset=dataset,
                entropy_brier_csv=args.entropy_brier_csv,
                entropy_feature=args.entropy_feature,
            )
        output_png = plot_layer_scores(
            layer_scores,
            entropy_score=entropy_score,
            title=title,
            output_pdf=output_pdf,
        )
        print_summary(layer_scores, entropy_score, args.entropy_feature)
        print(f"[saved] {output_pdf}")
        print(f"[saved] {output_png}")


if __name__ == "__main__":
    main()
