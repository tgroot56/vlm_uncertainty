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

try:
    from .plot_style import PROBE_PLOT_FONT_SIZES
except ImportError:
    from plot_style import PROBE_PLOT_FONT_SIZES


DEFAULT_RESULTS_ROOT = Path(
    "/projects/prjs2014/llava/layer_experiment/probe_results/imagenet-r/mlp_layer_comparison_20k_from_checkpoint/mlp"
)
DEFAULT_OUTPUT_PDF = Path("probe_results/layer_experiment/llava_imagenet_r_mlp_layer_brier.pdf")

FEATURE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "vision_mean": re.compile(r"^vision_mean_layer_(\d+)$"),
    "lm_visual_mean": re.compile(r"^lm_visual_mean_layer_(\d+)$"),
    "lm_question_lasttok": re.compile(r"^lm_question_lasttok_layer_(\d+)$"),
    "lm_answer_lasttok": re.compile(r"^lm_answer_lasttok_layer_(\d+)$"),
    "lm_fullspan_lasttok": re.compile(r"^lm_fullspan_lasttok_layer_(\d+)$"),
}

LEGACY_IMAGENET_FEATURE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "lm_visual_mean": re.compile(r"^lm_visual_mean_layer_(\d+)$"),
    "lm_question_mean": re.compile(r"^lm_question_mean_layer_(\d+)$"),
    "lm_answer_mean": re.compile(r"^lm_answer_mean_layer_(\d+)$"),
    "lm_fullspan_lasttok": re.compile(r"^lm_fullspan_lasttok_layer_(\d+)$"),
}

SERIES_LABELS = {
    "vision_mean": "Vision (mean)",
    "lm_visual_mean": "LM visual (mean)",
    "lm_question_mean": "LM question (mean)",
    "lm_answer_mean": "LM answer (mean)",
    "lm_question_lasttok": "LM question (lasttok)",
    "lm_answer_lasttok": "LM answer (lasttok)",
    "lm_fullspan_lasttok": "LM Final Token",
}

SERIES_COLORS = {
    "vision_mean": "#2563eb",
    "lm_visual_mean": "#0f766e",
    "lm_question_mean": "#ea580c",
    "lm_answer_mean": "#6366f1",
    "lm_question_lasttok": "#c2410c",
    "lm_answer_lasttok": "#4338ca",
    "lm_fullspan_lasttok": "#b91c1c",
}

LAYER_EXPERIMENT_FONT_SCALE = 0.85
LAYER_EXPERIMENT_FONT_SIZES = PROBE_PLOT_FONT_SIZES.scaled(LAYER_EXPERIMENT_FONT_SCALE)

AX_TITLE_FONTSIZE = LAYER_EXPERIMENT_FONT_SIZES.axis_title
AX_LABEL_FONTSIZE = LAYER_EXPERIMENT_FONT_SIZES.axis_label
TICK_LABEL_FONTSIZE = LAYER_EXPERIMENT_FONT_SIZES.tick_label
LEGEND_FONTSIZE = LAYER_EXPERIMENT_FONT_SIZES.legend
COMBINED_FIG_TITLE_FONTSIZE = LAYER_EXPERIMENT_FONT_SIZES.combined_figure_title
ANNOTATION_FONTSIZE = LAYER_EXPERIMENT_FONT_SIZES.annotation
LAYER_XTICK_STRIDE = 4
OUTPUT_DIST_X_GAP = 6

SCALAR_FEATURES = {
    "answer_gen_geom_mean_probability": {
        "label": "Answer geom. mean prob.",
        "color": "#4b5563",
    },
}

CSV_SCALAR_FEATURES = {
    "answer_gen_neglogp_mean": {
        "label": "NegLogP (mean)",
        "color": "#111827",
        "marker": "s",
        "tick_label": "Output Dist",
        "x_group": "output_dist",
    },
    "answer_gen_entropy_mean": {
        "label": "Entropy (mean)",
        "color": "#f97316",
        "marker": "^",
        "tick_label": "Output Dist",
        "x_group": "output_dist",
    },
}

SCALAR_VARIANTS = [
    ("neglogp_mean", "answer_gen_neglogp_mean"),
    ("entropy_mean", "answer_gen_entropy_mean"),
]

ENTROPY_FEATURE_NAME = "answer_gen_entropy_mean"

COMBINED_DATASET_ORDER = ["vqa-v2", "coco-qa-vi", "pope/random", "imagenet-r"]
COMBINED_DATASET_LABELS = {
    "vqa-v2": "VQA-v2",
    "coco-qa-vi": "COCO-QA",
    "pope/random": "POPE",
    "imagenet-r": "ImageNet-R",
}
COMBINED_OUTPUT_PDF = Path("probe_results/layer_experiment/llava_all_datasets_mlp_layer_brier.pdf")
COMBINED_OUTPUT_PDFS = {
    "llava": COMBINED_OUTPUT_PDF,
    "qwen": Path("probe_results/layer_experiment/qwen_all_datasets_mlp_layer_brier.pdf"),
}

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
            "title": "LLaVA POPE: uncertainty signal across LM layers",
        },
        "vqa-v2": {
            "results_root": Path(
                "/projects/prjs2014/llava/layer_experiment/probe_results/vqa-v2/mlp_layer_comparison/mlp"
            ),
            "output_pdf": Path("probe_results/layer_experiment/llava_vqa_v2_mlp_layer_brier.pdf"),
            "title": "LLaVA VQA-v2: uncertainty signal across LM layers",
        },
        "coco-qa-vi": {
            "results_root": Path(
                "/projects/prjs2014/llava/layer_experiment/probe_results/coco-qa-vi/mlp_layer_comparison/mlp"
            ),
            "output_pdf": Path("probe_results/layer_experiment/llava_coco_qa_vi_mlp_layer_brier.pdf"),
            "title": "LLaVA COCO-QA: uncertainty signal across LM layers",
        },
    },
    "qwen": {
        "imagenet-r": {
            "results_root": Path(
                "/projects/prjs2014/qwen/layer_experiment/probe_results/imagenet-r/mlp_layer_comparison/mlp"
            ),
            "output_pdf": Path("probe_results/layer_experiment/qwen_imagenet_r_mlp_layer_brier.pdf"),
            "title": "Qwen ImageNet-R: uncertainty signal across LM layers",
        },
        "pope/random": {
            "results_root": Path(
                "/projects/prjs2014/qwen/layer_experiment/probe_results/pope/random/mlp_layer_comparison/mlp"
            ),
            "output_pdf": Path("probe_results/layer_experiment/qwen_pope_random_mlp_layer_brier.pdf"),
            "title": "Qwen POPE: uncertainty signal across LM layers",
        },
        "vqa-v2": {
            "results_root": Path(
                "/projects/prjs2014/qwen/layer_experiment/probe_results/vqa-v2/mlp_layer_comparison/mlp"
            ),
            "output_pdf": Path("probe_results/layer_experiment/qwen_vqa_v2_mlp_layer_brier.pdf"),
            "title": "Qwen VQA-v2: uncertainty signal across LM layers",
        },
        "coco-qa-vi": {
            "results_root": Path(
                "/projects/prjs2014/qwen/layer_experiment/probe_results/coco-qa-vi/mlp_layer_comparison/mlp"
            ),
            "output_pdf": Path("probe_results/layer_experiment/qwen_coco_qa_vi_mlp_layer_brier.pdf"),
            "title": "Qwen COCO-QA: uncertainty signal across LM layers",
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
    parser.add_argument(
        "--combined-all-datasets",
        action="store_true",
        help="Save a 2x2 all-dataset layer-sweep figure for --model.",
    )
    parser.add_argument(
        "--combined-output-pdf",
        type=Path,
        default=None,
        help="Where to save --combined-all-datasets. A PNG is also written.",
    )
    parser.add_argument(
        "--layer-run-suffix",
        type=str,
        default=None,
        help=(
            "Optional suffix appended to configured layer-experiment directories, "
            "e.g. _1k maps mlp_layer_comparison/mlp to mlp_layer_comparison_1k/mlp."
        ),
    )
    return parser.parse_args()


def _local_probe_results_root(model: str, dataset: str) -> Path:
    return Path("probe_results") / model / Path(*dataset.split("/")) / "mlp"


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
    run_suffix = getattr(args, "layer_run_suffix", None)
    if args.results_root is None and run_suffix:
        parts = list(results_root.parts)
        for idx, part in enumerate(parts):
            if part == "mlp_layer_comparison":
                parts[idx] = f"{part}{run_suffix}"
                results_root = Path(*parts)
                break
    if args.results_root is None and not run_suffix and not results_root.exists():
        local_results_root = _local_probe_results_root(args.model, dataset)
        if local_results_root.exists():
            print(
                f"[info] Falling back to local probe results for {args.model}/{dataset}: "
                f"{local_results_root}"
            )
            results_root = local_results_root
    output_pdf = args.output_pdf or Path(config["output_pdf"])
    title = args.title or str(config["title"])
    return results_root, output_pdf, title


def selected_datasets(args: argparse.Namespace) -> List[str]:
    if args.dataset is not None:
        return [args.dataset]
    return sorted(MODEL_DATASET_CONFIGS[args.model])


def default_output_dist_features(args: argparse.Namespace) -> List[str]:
    if args.no_entropy:
        return []
    features = [feature for _, feature in SCALAR_VARIANTS]
    if args.entropy_feature != ENTROPY_FEATURE_NAME:
        features = [
            args.entropy_feature if feature == ENTROPY_FEATURE_NAME else feature
            for feature in features
        ]
    return list(dict.fromkeys(features))


def feature_patterns_for_dataset(dataset: str) -> Dict[str, re.Pattern[str]]:
    return FEATURE_PATTERNS


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


def collect_layer_scores(
    results_root: Path,
    *,
    feature_patterns: Dict[str, re.Pattern[str]] = FEATURE_PATTERNS,
) -> Dict[str, List[Tuple[int, float, Path]]]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    if results_root.is_file():
        raise ValueError(
            f"Results root points to a file, but this script needs the probe results "
            f"directory containing per-feature folders: {results_root}"
    )

    collected: Dict[str, List[Tuple[int, float, Path]]] = {key: [] for key in feature_patterns}

    for config_path in _iter_direct_feature_configs(results_root):
        config = load_json(config_path)
        feature_names = config.get("feature_names") or []
        if len(feature_names) != 1:
            continue

        feature_name = str(feature_names[0])
        score = _score_from_config(config)
        if score is None:
            continue

        for family, pattern in feature_patterns.items():
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


def collect_scalar_scores(
    results_root: Path,
    *,
    scalar_features: Dict[str, Dict[str, str]] = SCALAR_FEATURES,
) -> Dict[str, Tuple[float, Path]]:
    scalar_scores: Dict[str, Tuple[float, Path]] = {}

    for config_path in _iter_direct_feature_configs(results_root):
        config = load_json(config_path)
        feature_names = config.get("feature_names") or []
        if len(feature_names) != 1:
            continue

        feature_name = str(feature_names[0])
        if feature_name not in scalar_features:
            continue

        score = _score_from_config(config)
        if score is None:
            continue

        scalar_scores[feature_name] = (score, config_path.parent)

    for feature_name in scalar_features:
        if feature_name not in scalar_scores:
            print(f"[warning] No results found for scalar feature {feature_name} under {results_root}")

    return scalar_scores


def collect_single_feature_score_from_results_root(
    results_root: Path,
    *,
    feature_name: str,
) -> Optional[Tuple[float, Path]]:
    if not results_root.exists() or results_root.is_file():
        return None

    for config_path in _iter_direct_feature_configs(results_root):
        config = load_json(config_path)
        feature_names = config.get("feature_names") or []
        if len(feature_names) != 1 or str(feature_names[0]) != feature_name:
            continue

        score = _score_from_config(config)
        if score is None:
            continue
        return score, config_path.parent

    return None


def _extra_tick_label(feature_name: str, scalar_styles: Dict[str, Dict[str, str]]) -> str:
    style = scalar_styles[feature_name]
    if "tick_label" in style:
        return str(style["tick_label"])
    return str(style["label"]).replace(" ", "\n", 2)


def _scalar_positions(
    layer_scores: Dict[str, List[Tuple[int, float, Path]]],
    scalar_scores: Dict[str, Tuple[float, Path]],
    scalar_styles: Dict[str, Dict[str, str]],
) -> Dict[str, int]:
    all_layers = sorted({layer for points in layer_scores.values() for layer, _, _ in points})
    scalar_start_x = (max(all_layers) + OUTPUT_DIST_X_GAP) if all_layers else 1
    positions: Dict[str, int] = {}
    group_positions: Dict[str, int] = {}
    next_x = scalar_start_x
    for feature_name, style in scalar_styles.items():
        if feature_name not in scalar_scores:
            continue
        group = str(style.get("x_group", feature_name))
        if group not in group_positions:
            group_positions[group] = next_x
            next_x += 1
        positions[feature_name] = group_positions[group]
    return positions


def draw_layer_scores_on_axis(
    ax: plt.Axes,
    layer_scores: Dict[str, List[Tuple[int, float, Path]]],
    *,
    scalar_scores: Dict[str, Tuple[float, Path]],
    scalar_styles: Dict[str, Dict[str, str]],
    title: str,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_legend: bool = True,
    annotate_best: bool = True,
    annotate_scalar_scores: bool = True,
) -> None:
    all_layers = sorted({layer for points in layer_scores.values() for layer, _, _ in points})
    scalar_positions = _scalar_positions(layer_scores, scalar_scores, scalar_styles)

    for feature_name, x_pos in scalar_positions.items():
        score, _ = scalar_scores[feature_name]
        style = scalar_styles[feature_name]
        color = str(style["color"])
        ax.scatter(
            [x_pos],
            [score],
            marker=str(style.get("marker", "D")),
            s=76 if not annotate_best else 95,
            color=color,
            edgecolors="white",
            linewidths=1.1,
            zorder=4,
            label=str(style["label"]),
        )
        if annotate_best and annotate_scalar_scores:
            ax.annotate(
                f"{score:.3f}",
                xy=(x_pos, score),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=ANNOTATION_FONTSIZE,
                color=color,
            )

    plotted_scores = [
        score
        for points in layer_scores.values()
        for _, score, _ in points
    ]
    if plotted_scores:
        score_min = min(plotted_scores)
        score_max = max(plotted_scores)
        score_span = max(score_max - score_min, 1e-6)
    else:
        score_min = 0.0
        score_span = 1.0

    all_layer_values = sorted({layer for points in layer_scores.values() for layer, _, _ in points})
    max_layer = max(all_layer_values) if all_layer_values else None

    for family_idx, (family, points) in enumerate(layer_scores.items()):
        if not points:
            continue

        layers = [layer for layer, _, _ in points]
        brier_scores = [score for _, score, _ in points]
        ax.plot(
            layers,
            brier_scores,
            marker="o",
            linewidth=2.0 if not annotate_best else 2.4,
            markersize=4.5 if not annotate_best else 6,
            label=SERIES_LABELS[family],
            color=SERIES_COLORS[family],
        )

        if annotate_best:
            best_idx = min(range(len(points)), key=lambda idx: brier_scores[idx])
            best_layer = layers[best_idx]
            best_score = brier_scores[best_idx]
            y_fraction = (best_score - score_min) / score_span
            y_offset = 11 if y_fraction < 0.28 else -13
            y_offset += (family_idx % 3 - 1) * 3
            if max_layer is not None and best_layer >= max_layer - 2:
                x_offset = -10
                ha = "right"
            else:
                x_offset = 7
                ha = "left"
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
                f"L{best_layer}",
                xy=(best_layer, best_score),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=ANNOTATION_FONTSIZE,
                color=SERIES_COLORS[family],
                ha=ha,
                va="center",
                zorder=6,
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )

    ax.set_title(title, fontsize=AX_TITLE_FONTSIZE, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.8)

    tick_labels_by_position: Dict[int, str] = {}
    for feature_name, x_pos in scalar_positions.items():
        tick_labels_by_position.setdefault(
            x_pos,
            _extra_tick_label(feature_name, scalar_styles),
        )
    extra_ticks = list(tick_labels_by_position)
    extra_labels = [tick_labels_by_position[x_pos] for x_pos in extra_ticks]
    displayed_layers = [
        layer
        for layer in all_layers
        if layer % LAYER_XTICK_STRIDE == 0
    ]

    if all_layers:
        ax.set_xticks(displayed_layers + extra_ticks)
        if extra_ticks:
            ax.set_xticklabels([str(layer) for layer in displayed_layers] + extra_labels)
    elif extra_ticks:
        ax.set_xticks(extra_ticks)
        ax.set_xticklabels(extra_labels)

    if extra_ticks:
        left = (min(all_layers) - 0.75) if all_layers else min(extra_ticks) - 1
        ax.set_xlim(left, max(extra_ticks) + 1.25)

    ax.set_xlabel("Layer index" if show_xlabel else "", fontsize=AX_LABEL_FONTSIZE)
    ax.set_ylabel("Brier score" if show_ylabel else "", fontsize=AX_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    if show_legend:
        ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)


def _dataset_probe_results_root(model: str, dataset: str) -> Path:
    return Path("probe_results") / model / Path(*dataset.split("/"))


def _single_feature_score_from_brier_csv(
    csv_path: Path,
    *,
    feature_name: str,
    probe_column: str,
    warn_missing_value: bool = True,
) -> Optional[float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("feature_label") != feature_name:
                continue
            value = row.get(probe_column)
            if value is None or value == "":
                if warn_missing_value:
                    print(f"[warning] {csv_path} has no {probe_column!r} value for {feature_name}")
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


def collect_single_feature_score_from_csv(
    *,
    model: str,
    dataset: str,
    brier_csv: Optional[Path],
    feature_name: str,
    probe_column: str = "mlp",
) -> Optional[Tuple[float, Path]]:
    if brier_csv is not None:
        if not brier_csv.exists():
            print(f"[warning] Feature results_brier.csv does not exist: {brier_csv}")
            return None
        score = _single_feature_score_from_brier_csv(
            brier_csv,
            feature_name=feature_name,
            probe_column=probe_column,
        )
        if score is None:
            print(f"[warning] No {feature_name} row found in {brier_csv}")
            return None
        return score, brier_csv

    search_root = _dataset_probe_results_root(model, dataset)
    if not search_root.exists():
        print(f"[warning] Feature search root does not exist: {search_root}")
        return None

    candidates: List[Tuple[Tuple[bool, bool, bool, int, str], float, Path]] = []
    for csv_path in sorted(search_root.rglob("results_brier.csv")):
        score = _single_feature_score_from_brier_csv(
            csv_path,
            feature_name=feature_name,
            probe_column=probe_column,
            warn_missing_value=False,
        )
        if score is not None:
            candidates.append((_entropy_csv_rank(csv_path, search_root), score, csv_path))

    if candidates:
        _, score, csv_path = min(candidates, key=lambda item: item[0])
        return score, csv_path

    print(f"[warning] No {feature_name} row found under {search_root}")
    return None


def plot_layer_scores(
    layer_scores: Dict[str, List[Tuple[int, float, Path]]],
    *,
    scalar_scores: Dict[str, Tuple[float, Path]],
    scalar_styles: Dict[str, Dict[str, str]],
    title: str,
    output_pdf: Path,
    annotate_scalar_scores: bool = True,
) -> Path:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    draw_layer_scores_on_axis(
        ax,
        layer_scores,
        scalar_scores=scalar_scores,
        scalar_styles=scalar_styles,
        title=title,
        show_xlabel=True,
        show_ylabel=True,
        show_legend=True,
        annotate_best=True,
        annotate_scalar_scores=annotate_scalar_scores,
    )

    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_png


def plot_combined_all_datasets(
    *,
    model: str,
    output_pdf: Path,
    extra_feature: Optional[str],
    extra_features: Optional[List[str]] = None,
    extra_feature_source_csv: Optional[Path] = None,
    layer_run_suffix: Optional[str] = None,
    include_default_scalars: bool = True,
    annotate_scalar_scores: bool = True,
    title: str = "LLaVA Layer Sweep",
) -> Path:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=False, sharey=False)
    axes_flat = list(axes.flat)
    legend_handles = {}

    for idx, (ax, dataset) in enumerate(zip(axes_flat, COMBINED_DATASET_ORDER)):
        results_root, _, _ = resolve_plot_config(
            argparse.Namespace(
                model=model,
                results_root=None,
                output_pdf=None,
                title=None,
                layer_run_suffix=layer_run_suffix,
            ),
            dataset,
        )
        layer_scores = collect_layer_scores(
            results_root,
            feature_patterns=feature_patterns_for_dataset(dataset),
        )
        scalar_scores = collect_scalar_scores(results_root) if include_default_scalars else {}
        scalar_styles = dict(SCALAR_FEATURES) if include_default_scalars else {}
        requested_extra_features = list(extra_features or [])
        if extra_feature is not None:
            requested_extra_features.append(extra_feature)
        for requested_feature in dict.fromkeys(requested_extra_features):
            extra_score = collect_single_feature_score_from_results_root(
                results_root,
                feature_name=requested_feature,
            )
            if extra_score is None:
                extra_score = collect_single_feature_score_from_csv(
                    model=model,
                    dataset=dataset,
                    brier_csv=extra_feature_source_csv,
                    feature_name=requested_feature,
                )
            if extra_score is not None:
                scalar_scores[requested_feature] = extra_score
                scalar_styles[requested_feature] = CSV_SCALAR_FEATURES.get(
                    requested_feature,
                    {
                        "label": requested_feature,
                        "color": "#4b5563",
                    },
                )

        draw_layer_scores_on_axis(
            ax,
            layer_scores,
            scalar_scores=scalar_scores,
            scalar_styles=scalar_styles,
            title=COMBINED_DATASET_LABELS[dataset],
            show_xlabel=idx >= 2,
            show_ylabel=idx % 2 == 0,
            show_legend=False,
            annotate_best=True,
            annotate_scalar_scores=annotate_scalar_scores,
        )

        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_handles.setdefault(label, handle)

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle(title, fontsize=COMBINED_FIG_TITLE_FONTSIZE, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_png


def print_summary(
    layer_scores: Dict[str, List[Tuple[int, float, Path]]],
    scalar_scores: Dict[str, Tuple[float, Path]],
) -> None:
    for feature_name, (score, result_dir) in scalar_scores.items():
        print(f"{feature_name}: brier={score:.6f}, result_dir={result_dir}")

    for family, points in layer_scores.items():
        if not points:
            continue
        best_layer, best_score, best_dir = min(points, key=lambda item: item[1])
        print(
            f"{family}: best layer={best_layer}, brier={best_score:.6f}, "
            f"result_dir={best_dir}"
        )


def output_with_suffix(output_pdf: Path, suffix: str) -> Path:
    return output_pdf.with_name(f"{output_pdf.stem}_{suffix}{output_pdf.suffix}")


def plot_scalar_swap_variants(
    *,
    model: str,
    dataset: str,
    results_root: Path,
    layer_scores: Dict[str, List[Tuple[int, float, Path]]],
    title: str,
    output_pdf: Path,
    brier_csv: Optional[Path],
) -> None:
    for suffix, feature_name in SCALAR_VARIANTS:
        scalar_scores: Dict[str, Tuple[float, Path]] = {}
        scalar_styles: Dict[str, Dict[str, str]] = {}
        score = collect_single_feature_score_from_results_root(
            results_root,
            feature_name=feature_name,
        )
        if score is None:
            score = collect_single_feature_score_from_csv(
                model=model,
                dataset=dataset,
                brier_csv=brier_csv,
                feature_name=feature_name,
            )
        if score is not None:
            scalar_scores[feature_name] = score
            scalar_styles[feature_name] = CSV_SCALAR_FEATURES.get(
                feature_name,
                {
                    "label": feature_name,
                    "color": "#4b5563",
                },
            )
        if not scalar_scores:
            continue
        variant_pdf = output_with_suffix(output_pdf, suffix)
        variant_png = plot_layer_scores(
            layer_scores,
            scalar_scores=scalar_scores,
            scalar_styles=scalar_styles,
            title=title,
            output_pdf=variant_pdf,
            annotate_scalar_scores=False,
        )
        print_summary(layer_scores, scalar_scores)
        print(f"[saved] {variant_pdf}")
        print(f"[saved] {variant_png}")


def plot_combined_scalar_swap_variants(
    *,
    model: str,
    output_pdf: Path,
    layer_run_suffix: Optional[str] = None,
) -> None:
    for suffix, feature_name in SCALAR_VARIANTS:
        variant_pdf = output_with_suffix(output_pdf, suffix)
        variant_png = plot_combined_all_datasets(
            model=model,
            output_pdf=variant_pdf,
            extra_feature=None,
            extra_features=[feature_name],
            layer_run_suffix=layer_run_suffix,
            include_default_scalars=False,
            annotate_scalar_scores=False,
            title=f"{'LLaVA' if model == 'llava' else 'Qwen'} Layer Sweep",
        )
        print(f"[saved] {variant_pdf}")
        print(f"[saved] {variant_png}")


def main() -> None:
    args = parse_args()
    if args.combined_all_datasets:
        combined_output_pdf = args.combined_output_pdf or COMBINED_OUTPUT_PDFS[args.model]
        extra_features = default_output_dist_features(args)
        output_png = plot_combined_all_datasets(
            model=args.model,
            output_pdf=combined_output_pdf,
            extra_feature=None,
            extra_features=extra_features,
            layer_run_suffix=args.layer_run_suffix,
            include_default_scalars=False,
            annotate_scalar_scores=False,
            title=f"{'LLaVA' if args.model == 'llava' else 'Qwen'} Layer Sweep",
        )
        print(f"[saved] {combined_output_pdf}")
        print(f"[saved] {output_png}")
        plot_combined_scalar_swap_variants(
            model=args.model,
            output_pdf=combined_output_pdf,
            layer_run_suffix=args.layer_run_suffix,
        )
        return

    datasets = selected_datasets(args)
    for dataset in datasets:
        results_root, output_pdf, title = resolve_plot_config(args, dataset)
        if len(datasets) > 1 and not results_root.exists():
            print(f"[warning] Skipping {args.model}/{dataset}; results root does not exist: {results_root}")
            continue

        layer_scores = collect_layer_scores(
            results_root,
            feature_patterns=feature_patterns_for_dataset(dataset),
        )
        scalar_scores = collect_scalar_scores(results_root)
        scalar_styles = dict(SCALAR_FEATURES)
        for feature_name in default_output_dist_features(args):
            feature_score = collect_single_feature_score_from_results_root(
                results_root,
                feature_name=feature_name,
            )
            if feature_score is None:
                feature_score = collect_single_feature_score_from_csv(
                    model=args.model,
                    dataset=dataset,
                    brier_csv=args.entropy_brier_csv,
                    feature_name=feature_name,
                )
            if feature_score is not None:
                scalar_scores[feature_name] = feature_score
                scalar_styles[feature_name] = CSV_SCALAR_FEATURES.get(
                    feature_name,
                    {
                        "label": feature_name,
                        "color": "#4b5563",
                    },
                )
        output_png = plot_layer_scores(
            layer_scores,
            scalar_scores=scalar_scores,
            scalar_styles=scalar_styles,
            title=title,
            output_pdf=output_pdf,
        )
        print_summary(layer_scores, scalar_scores)
        print(f"[saved] {output_pdf}")
        print(f"[saved] {output_png}")
        plot_scalar_swap_variants(
            model=args.model,
            dataset=dataset,
            results_root=results_root,
            layer_scores=layer_scores,
            title=title,
            output_pdf=output_pdf,
            brier_csv=args.entropy_brier_csv,
        )


if __name__ == "__main__":
    main()
