#!/usr/bin/env python
"""Render representative sample-level JS-divergence trajectories with images."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patching.plot_cross_model_vqa_layer_sweep_js_divergence import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    LLAVA_PARQUET,
    QWEN_PARQUET,
)
from scripts.patching.plot_vqa_layer_sweep_js_divergence import (
    compute_sample_layer_js_streaming,
)


PATCHING_ROOT = Path("/projects/prjs2014/patching")
MODEL_ENTRIES = [
    (
        "LLaVA-1.5-7B",
        LLAVA_PARQUET,
        PATCHING_ROOT
        / "llava-hf_llava-1-5-7b-hf/vqa-v2/patching_dataset_vqa-v2.json",
        "#2563eb",
    ),
    (
        "Qwen3.5-9B",
        QWEN_PARQUET,
        PATCHING_ROOT / "Qwen_Qwen3-5-9B/vqa-v2/patching_dataset_vqa-v2.json",
        "#ea580c",
    ),
]
OUTPUT_NAME = "cross_model_vqa_js_qualitative_examples"


def _slug(model_label: str) -> str:
    return model_label.lower().replace(".", "_").replace("-", "_")


def load_or_compute_trajectories(
    model_label: str,
    parquet_path: Path,
    output_dir: Path,
    *,
    recompute: bool,
) -> pd.DataFrame:
    output_path = output_dir / f"{OUTPUT_NAME}_{_slug(model_label)}_trajectories.csv"
    if (
        not recompute
        and output_path.exists()
        and output_path.stat().st_mtime >= parquet_path.stat().st_mtime
    ):
        print(f"Using current trajectories: {output_path}")
        return pd.read_csv(output_path, dtype={"sample_id": str}).fillna(
            {"clean_answer": "", "corrupted_answer": "", "patched_answer": ""}
        )

    print(f"Computing sample trajectories from {parquet_path}")
    frame = compute_sample_layer_js_streaming(parquet_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    return frame


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as handle:
        data = json.load(handle)
    return {str(sample["sample_id"]): sample for sample in data["samples"]}


def corrupted_image_path(record: dict[str, Any], severity: str) -> str:
    for perturbation in record.get("visual_perturbations", []):
        if perturbation.get("severity") == severity:
            return str(perturbation.get("blurred_image_path", ""))
    return ""


def select_examples(
    model_label: str,
    trajectories: pd.DataFrame,
    aggregate_path: Path,
) -> list[tuple[str, pd.DataFrame]]:
    aggregate = pd.read_csv(aggregate_path).set_index("display_layer")
    complete_ids = (
        trajectories.groupby("sample_id")["display_layer"]
        .nunique()
        .loc[lambda values: values == len(aggregate)]
        .index
    )
    complete = trajectories[trajectories["sample_id"].isin(complete_ids)].copy()

    stats = []
    aggregate_values = aggregate["mean_js_clean_patched"]
    for sample_id, group in complete.groupby("sample_id", sort=False):
        ordered = group.sort_values("display_layer")
        values = ordered.set_index("display_layer")["js_clean_patched"]
        rmse = float(np.sqrt(np.mean((values - aggregate_values) ** 2)))
        late_drop = float(values.loc[20] - values.loc[32])
        stats.append(
            {
                "sample_id": str(sample_id),
                "rmse": rmse,
                "late_drop": late_drop,
                "mid_drop": float(values.loc[12] - values.loc[24]),
                "onset_80": next(
                    (
                        int(layer)
                        for layer, value in values.items()
                        if value < 0.8 * float(ordered.iloc[0]["js_clean_corrupted"])
                    ),
                    99,
                ),
                "answer_changes": ordered["patched_answer"].astype(str).nunique(),
                "has_readable_baselines": bool(
                    str(ordered.iloc[0]["clean_answer"]).strip()
                    and str(ordered.iloc[0]["corrupted_answer"]).strip()
                ),
            }
        )
    ranking = pd.DataFrame(stats)
    readable = ranking[ranking["has_readable_baselines"]]
    answer_changing = readable[readable["answer_changes"] >= 2]
    pool = answer_changing if len(answer_changing) >= 2 else readable

    typical_id = str(pool.sort_values("rmse").iloc[0]["sample_id"])
    contrast_pool = pool[pool["sample_id"] != typical_id]
    if model_label.startswith("LLaVA"):
        contrast_row = contrast_pool.sort_values(
            ["onset_80", "mid_drop"],
            ascending=[True, False],
        ).iloc[0]
        contrast_label = "Early, gradual convergence"
    else:
        contrast_row = contrast_pool.sort_values(
            "late_drop",
            ascending=False,
        ).iloc[0]
        contrast_label = "Late, abrupt convergence"
    contrast_id = str(contrast_row["sample_id"])
    return [
        (
            "Representative aggregate-like trajectory",
            complete[complete["sample_id"] == typical_id].sort_values("display_layer"),
        ),
        (
            contrast_label,
            complete[complete["sample_id"] == contrast_id].sort_values("display_layer"),
        ),
    ]


def _load_image(path: str) -> Image.Image | None:
    if not path or not Path(path).exists():
        return None
    return Image.open(path).convert("RGB")


def _show_image(ax: plt.Axes, image: Image.Image | None, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=2)
    if image is None:
        ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", fontsize=7)
    else:
        ax.imshow(image)


def _answer_at(group: pd.DataFrame, display_layer: int) -> str:
    row = group[group["display_layer"] == display_layer]
    return str(row.iloc[0]["patched_answer"]) if not row.empty else ""


def render(
    selected: dict[str, list[tuple[str, pd.DataFrame]]],
    records: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(16, 13), facecolor="white")
    outer = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.15)
    selected_rows = []

    for col, (model_label, _, _, color) in enumerate(MODEL_ENTRIES):
        for row, (selection_label, group) in enumerate(selected[model_label]):
            sample_id = str(group.iloc[0]["sample_id"])
            severity = str(group.iloc[0]["severity"])
            record = records[model_label][sample_id]
            clean_image = _load_image(str(record.get("clean_image_path", "")))
            corrupted_image = _load_image(corrupted_image_path(record, severity))

            cell = outer[row, col].subgridspec(
                3,
                2,
                height_ratios=[1.05, 0.48, 1.45],
                hspace=0.10,
                wspace=0.06,
            )
            ax_clean = fig.add_subplot(cell[0, 0])
            ax_corrupted = fig.add_subplot(cell[0, 1])
            ax_text = fig.add_subplot(cell[1, :])
            ax_curve = fig.add_subplot(cell[2, :])
            _show_image(ax_clean, clean_image, "Clean")
            _show_image(ax_corrupted, corrupted_image, "Corrupted")

            ax_curve.plot(
                group["display_layer"],
                group["js_clean_patched"],
                "o-",
                color=color,
                linewidth=2,
                markersize=3.5,
            )
            ax_curve.axhline(
                float(group.iloc[0]["js_clean_corrupted"]),
                color="#dc2626",
                linestyle="--",
                linewidth=1.3,
                label="JS(clean, corrupted)",
            )
            ax_curve.set_xlim(1, 32)
            ax_curve.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
            ax_curve.set_xlabel("Patched layer", fontsize=9)
            ax_curve.set_ylabel("JS(clean, patched)", fontsize=9)
            ax_curve.grid(alpha=0.25)
            ax_curve.tick_params(labelsize=8)
            ax_curve.legend(frameon=False, fontsize=7, loc="upper right")

            clean_answer = str(group.iloc[0]["clean_answer"])
            corrupted_answer = str(group.iloc[0]["corrupted_answer"])
            layer_answers = " | ".join(
                f"L{layer}: {_answer_at(group, layer)}" for layer in (1, 16, 24, 32)
            )
            question = str(record.get("clean_question", ""))
            ground_truth = str(record.get("correct_answer", ""))
            ax_text.axis("off")
            ax_text.text(
                0,
                0.98,
                selection_label,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
            ax_text.text(
                0,
                0.70,
                textwrap.fill(
                    f"sample {sample_id} | Q: {question} | GT: {ground_truth}",
                    width=92,
                ),
                fontsize=8.5,
                va="top",
            )
            ax_text.text(
                0,
                0.34,
                textwrap.fill(
                    f"clean: {clean_answer} | corrupted: {corrupted_answer} | {layer_answers}",
                    width=100,
                ),
                fontsize=8.5,
                va="top",
            )
            if row == 0:
                ax_clean.text(
                    0.0,
                    1.30,
                    model_label,
                    transform=ax_clean.transAxes,
                    fontsize=17,
                    fontweight="bold",
                    va="bottom",
                )

            selected_rows.append(
                {
                    "model": model_label,
                    "selection": selection_label,
                    "sample_id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "clean_answer": clean_answer,
                    "corrupted_answer": corrupted_answer,
                    "layer_1_answer": _answer_at(group, 1),
                    "layer_16_answer": _answer_at(group, 16),
                    "layer_24_answer": _answer_at(group, 24),
                    "layer_32_answer": _answer_at(group, 32),
                    "js_layer_1": float(group.iloc[0]["js_clean_patched"]),
                    "js_layer_32": float(group.iloc[-1]["js_clean_patched"]),
                }
            )

    fig.suptitle(
        "Qualitative VQA-v2 Examples: Layer-wise JS-Divergence Behavior",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in [("pdf", {}), ("png", {"dpi": 200})]:
        path = output_dir / f"{OUTPUT_NAME}.{extension}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved {path}")
    plt.close(fig)
    pd.DataFrame(selected_rows).to_csv(
        output_dir / f"{OUTPUT_NAME}_selected.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for model_label, parquet_path, dataset_path, _ in MODEL_ENTRIES:
        trajectories = load_or_compute_trajectories(
            model_label,
            parquet_path,
            args.output_dir,
            recompute=args.recompute,
        )
        aggregate_path = args.output_dir / (
            f"{DEFAULT_OUTPUT_NAME}_{_slug(model_label)}.csv"
        )
        selected[model_label] = select_examples(model_label, trajectories, aggregate_path)
        records[model_label] = load_records(dataset_path)
    render(selected, records, args.output_dir)


if __name__ == "__main__":
    main()
