from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULTS_ROOT = Path(
    "/projects/prjs2014/llava/layer_experiment/probe_results/pope/random/mlp_layer_comparison/mlp"
)
DEFAULT_OUTPUT_PDF = Path("probe_results/layer_experiment/llava_pope_random_mlp_layer_brier.pdf")

FEATURE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "lm_visual_mean": re.compile(r"^lm_visual_mean_layer_(\d+)$"),
    "lm_question_mean": re.compile(r"^lm_question_mean_layer_(\d+)$"),
    "lm_answer_mean": re.compile(r"^lm_answer_mean_layer_(\d+)$"),
}

SERIES_LABELS = {
    "lm_visual_mean": "Visual span mean",
    "lm_question_mean": "Question span mean",
    "lm_answer_mean": "Answer span mean",
}

SERIES_COLORS = {
    "lm_visual_mean": "#0f766e",
    "lm_question_mean": "#c2410c",
    "lm_answer_mean": "#4338ca",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Brier score versus LM layer for the LLaVA POPE/random layer experiment."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing per-feature probe result folders with config.json files.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Where to save the PDF plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="LLaVA POPE/random: uncertainty signal across LM layers",
        help="Plot title.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_layer_scores(results_root: Path) -> Dict[str, List[Tuple[int, float, Path]]]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    collected: Dict[str, List[Tuple[int, float, Path]]] = {key: [] for key in FEATURE_PATTERNS}

    for config_path in sorted(results_root.glob("*/config.json")):
        config = load_json(config_path)
        feature_names = config.get("feature_names") or []
        if len(feature_names) != 1:
            continue

        feature_name = str(feature_names[0])
        test_loss = config.get("test_loss")
        if test_loss is None:
            continue

        for family, pattern in FEATURE_PATTERNS.items():
            match = pattern.match(feature_name)
            if not match:
                continue

            layer_idx = int(match.group(1))
            collected[family].append((layer_idx, float(test_loss), config_path.parent))
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


def plot_layer_scores(
    layer_scores: Dict[str, List[Tuple[int, float, Path]]],
    *,
    title: str,
    output_pdf: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

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

    all_layers = sorted({layer for points in layer_scores.values() for layer, _, _ in points})
    if all_layers:
        ax.set_xticks(all_layers)

    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def print_summary(layer_scores: Dict[str, List[Tuple[int, float, Path]]]) -> None:
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
    layer_scores = collect_layer_scores(args.results_root)
    plot_layer_scores(layer_scores, title=args.title, output_pdf=args.output_pdf)
    print_summary(layer_scores)
    print(f"[saved] {args.output_pdf}")


if __name__ == "__main__":
    main()
