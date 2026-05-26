from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm_uncertainty_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

try:
    from .config import DATASET_LABELS, OUTPUT_ROOT, PAPER_EXAMPLES_DIR, SPAN_DISPLAY
    from .data_utils import (
        all_plot_rows,
        blurred_image_path,
        compact_row_dict,
        format_answer_with_conf,
        ground_truth_text,
        highlighted_degraded_image,
        latex_escape,
        prompt_tokens_for_row,
        sample_record,
        token_window,
    )
except ImportError:  # pragma: no cover
    from config import DATASET_LABELS, OUTPUT_ROOT, PAPER_EXAMPLES_DIR, SPAN_DISPLAY  # type: ignore
    from data_utils import (  # type: ignore
        all_plot_rows,
        blurred_image_path,
        compact_row_dict,
        format_answer_with_conf,
        ground_truth_text,
        highlighted_degraded_image,
        latex_escape,
        prompt_tokens_for_row,
        sample_record,
        token_window,
    )


def _unique_key(row: pd.Series) -> tuple[str, str, str, int]:
    return (
        str(row["model"]),
        str(row["dataset"]),
        str(row["sample_id"]),
        int(row["patched_token_position"]),
    )


def select_default_examples(rows: pd.DataFrame, per_category: int = 4) -> pd.DataFrame:
    selected: list[tuple[str, pd.Series]] = []
    seen: set[tuple[str, str, str, int]] = set()

    def add(category: str, frame: pd.DataFrame, sort_cols: list[str]) -> None:
        nonlocal selected, seen
        if frame.empty:
            return
        ranked = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(per_category * 4)
        added = 0
        for _, row in ranked.iterrows():
            key = _unique_key(row)
            if key in seen:
                continue
            seen.add(key)
            selected.append((category, row))
            added += 1
            if added >= per_category:
                break

    add(
        "final_prompt_accuracy_recovery",
        rows[
            rows["is_final_prompt_token"]
            & (rows["degraded_correctness"] < 1)
            & (rows["patched_correctness"] > rows["degraded_correctness"])
        ],
        ["accuracy_recovery", "patched_correctness", "confidence_recovery"],
    )
    add(
        "visual_token_accuracy_recovery",
        rows[
            (rows["span_label"] == "visual")
            & (rows["degraded_correctness"] < 1)
            & (rows["patched_correctness"] > rows["degraded_correctness"])
        ],
        ["accuracy_recovery", "patched_correctness", "confidence_recovery"],
    )
    add(
        "confidence_recovers_accuracy_not",
        rows[(rows["confidence_recovery"] > 0) & (rows["accuracy_recovery"] <= 0)],
        ["confidence_recovery", "patched_confidence"],
    )
    add(
        "accuracy_recovers_confidence_not",
        rows[(rows["accuracy_recovery"] > 0) & (rows["confidence_recovery"] <= 0)],
        ["accuracy_recovery", "patched_correctness"],
    )
    add(
        "confidence_exceeds_clean",
        rows[rows["confidence_exceeds_clean"]],
        ["patched_confidence", "confidence_recovery"],
    )
    add(
        "answer_changed",
        rows[rows["answer_changed"]],
        ["accuracy_recovery", "confidence_recovery"],
    )

    records = []
    for category, row in selected:
        item = row.to_dict()
        item["category"] = category
        records.append(item)
    return pd.DataFrame(records)


def _load_image(path: str) -> Image.Image | None:
    if not path or not Path(path).exists():
        return None
    return Image.open(path).convert("RGB")


def _show_image_or_text(ax: plt.Axes, image: Image.Image | None, title: str, fallback: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    if image is None:
        ax.text(0.5, 0.5, fallback, ha="center", va="center", wrap=True, fontsize=9)
        return
    ax.imshow(image)


def _token_snippet(tokens: list[dict[str, Any]], patched_pos: int) -> str:
    window = token_window(tokens, patched_pos, radius=7)
    parts = []
    for tok in window:
        text = str(tok.get("display", tok.get("token", "")))
        if int(tok["position"]) == patched_pos:
            parts.append(f"[PATCHED: {text}]")
        else:
            parts.append(text)
    return " ".join(parts)


def build_caption(row: pd.Series, record: dict[str, Any], category: str) -> str:
    dataset = DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"]))
    span = SPAN_DISPLAY.get(str(row["span_label"]), str(row["span_label"]))
    return (
        f"{row['model']} on {dataset}, sample {row['sample_id']}: patching the {span.lower()} "
        f"at token position {int(row['patched_token_position'])} changes the degraded output "
        f"from {format_answer_with_conf(row['degraded_answer'], row['degraded_confidence'])} "
        f"to {format_answer_with_conf(row['patched_answer'], row['patched_confidence'])}. "
        f"The clean output is {format_answer_with_conf(row['clean_answer'], row['clean_confidence'])}. "
        f"Selection category: {category.replace('_', ' ')}."
    )


def interpretation(row: pd.Series) -> str:
    acc_delta = float(row["patched_correctness"]) - float(row["degraded_correctness"])
    conf_delta = float(row["patched_confidence"]) - float(row["degraded_confidence"])
    if acc_delta > 0 and conf_delta > 0:
        pattern = "both answer correctness and expressed confidence increase"
    elif acc_delta > 0 and conf_delta <= 0:
        pattern = "answer correctness increases without a corresponding confidence increase"
    elif acc_delta <= 0 and conf_delta > 0:
        pattern = "expressed confidence increases while the correctness score does not improve"
    else:
        pattern = "the patch changes the output pattern without improving the stored recovery metrics"
    return (
        f"This example shows that patching the selected {SPAN_DISPLAY.get(str(row['span_label']), row['span_label']).lower()} "
        f"position can alter the degraded model behavior at the sample level. In this case, {pattern}. "
        f"The qualitative pattern should be interpreted as evidence about this specific intervention and sample, "
        f"with aggregate trends assessed by the positional-sweep recovery curves."
    )


def latex_table_row(row: pd.Series) -> str:
    values = [
        latex_escape(row["model"]),
        latex_escape(DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"]))),
        latex_escape(row["sample_id"]),
        latex_escape(SPAN_DISPLAY.get(str(row["span_label"]), str(row["span_label"]))),
        str(int(row["patched_token_position"])),
        latex_escape(row["clean_answer"]),
        f"{float(row['clean_confidence']):.2f}",
        f"{float(row['clean_correctness']):.2f}",
        latex_escape(row["degraded_answer"]),
        f"{float(row['degraded_confidence']):.2f}",
        f"{float(row['degraded_correctness']):.2f}",
        latex_escape(row["patched_answer"]),
        f"{float(row['patched_confidence']):.2f}",
        f"{float(row['patched_correctness']):.2f}",
        f"{float(row['accuracy_recovery']):.2f}",
        f"{float(row['confidence_recovery']):.2f}",
    ]
    return " & ".join(values) + r" \\"


def render_example(row: pd.Series, category: str, out_dir: Path, all_rows: pd.DataFrame) -> dict[str, Any]:
    record = sample_record(str(row["model"]), str(row["dataset"]), str(row["sample_id"]))
    clean_path = str(record.get("clean_image_path", ""))
    degraded_path = blurred_image_path(record, str(row["severity"]))
    tokens, spans, notes = prompt_tokens_for_row(all_rows, row)
    row = row.copy()
    for key in ("visual_start", "visual_end", "num_visual_tokens", "visual_geometry"):
        if key in spans:
            row[key] = spans[key]

    clean_img = _load_image(clean_path)
    degraded_img = _load_image(degraded_path)
    patched_visual_img = None
    patch_note = ""
    if str(row["span_label"]) == "visual":
        visual_start = int(row.get("visual_start", row["patched_token_position"]))
        visual_count = int(row.get("num_visual_tokens", 0) or 0)
        visual_rel = int(row["patched_token_position"]) - visual_start
        patched_visual_img, _, patch_note = highlighted_degraded_image(
            str(row["model"]),
            clean_path,
            degraded_path,
            visual_rel,
            visual_count,
            row.get("visual_geometry"),
        )

    uid = (
        f"{category}_{row['model'].replace('/', '_').replace(' ', '')}_"
        f"{row['dataset']}_{row['sample_id']}_pos{int(row['patched_token_position'])}"
    )
    safe_uid = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in uid)
    fig_png = out_dir / f"{safe_uid}.png"
    fig_pdf = out_dir / f"{safe_uid}.pdf"
    text_path = out_dir / f"{safe_uid}.md"

    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.4, 0.9, 1.3])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    _show_image_or_text(axes[0], clean_img, "Clean image", "Clean image unavailable")
    _show_image_or_text(axes[1], degraded_img, "Degraded image", "Degraded image unavailable")
    if patched_visual_img is not None:
        _show_image_or_text(axes[2], patched_visual_img, "Patched visual-token visualization", patch_note)
    else:
        axes[2].axis("off")
        axes[2].set_title("Patched token", fontsize=11, fontweight="bold")
        snippet = _token_snippet(tokens, int(row["patched_token_position"]))
        axes[2].text(0.02, 0.86, f"Span: {SPAN_DISPLAY.get(str(row['span_label']), row['span_label'])}", fontsize=10, transform=axes[2].transAxes)
        axes[2].text(0.02, 0.70, f"Position: {int(row['patched_token_position'])}", fontsize=10, transform=axes[2].transAxes)
        if patch_note:
            axes[2].text(0.02, 0.48, textwrap.fill(patch_note, width=42), fontsize=8, color="#555555", transform=axes[2].transAxes)
        axes[2].text(0.02, 0.05, textwrap.fill(snippet, width=42), fontsize=9, va="bottom", transform=axes[2].transAxes)

    ax_prompt = fig.add_subplot(gs[1, :])
    ax_prompt.axis("off")
    snippet = _token_snippet(tokens, int(row["patched_token_position"]))
    prompt_note = " ".join(notes) if notes else "Token strings reconstructed from cached processor."
    ax_prompt.text(0.01, 0.70, "Patched-token context", fontsize=11, fontweight="bold")
    ax_prompt.text(0.01, 0.36, textwrap.fill(snippet, width=150), fontsize=9, family="monospace")
    ax_prompt.text(0.01, 0.03, textwrap.fill(prompt_note, width=150), fontsize=8, color="#555555")

    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    table_data = [
        ["Clean", format_answer_with_conf(row["clean_answer"], row["clean_confidence"]), f"{float(row['clean_correctness']):.2f}"],
        ["Degraded", format_answer_with_conf(row["degraded_answer"], row["degraded_confidence"]), f"{float(row['degraded_correctness']):.2f}"],
        ["Patched", format_answer_with_conf(row["patched_answer"], row["patched_confidence"]), f"{float(row['patched_correctness']):.2f}"],
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Run", "Answer (confidence)", "Correctness"],
        loc="upper left",
        cellLoc="left",
        colWidths=[0.16, 0.58, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    gt = ground_truth_text(record, str(row["dataset"]))
    metric_text = (
        f"Ground truth: {gt}\n"
        f"Accuracy recovery: {float(row['accuracy_recovery']):.2f}; "
        f"confidence recovery: {float(row['confidence_recovery']):.2f}"
    )
    ax_table.text(0.01, 0.03, textwrap.fill(metric_text, width=150), fontsize=9, va="bottom")

    fig.suptitle(
        f"{row['model']} / {DATASET_LABELS.get(str(row['dataset']), row['dataset'])} / sample {row['sample_id']}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf, bbox_inches="tight")
    plt.close(fig)

    caption = build_caption(row, record, category)
    interp = interpretation(row)
    latex_row = latex_table_row(row)
    with text_path.open("w") as f:
        f.write(f"# {safe_uid}\n\n")
        f.write("## Caption\n\n")
        f.write(caption + "\n\n")
        f.write("## LaTeX table row\n\n")
        f.write("```latex\n")
        f.write(latex_row + "\n")
        f.write("```\n\n")
        f.write("## Interpretation\n\n")
        f.write(interp + "\n")

    compact = compact_row_dict(row, record, category=category)
    compact["figure_png"] = str(fig_png)
    compact["figure_pdf"] = str(fig_pdf)
    compact["text_file"] = str(text_path)
    compact["caption"] = caption
    compact["interpretation"] = interp
    compact["latex_table_row"] = latex_row
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PAPER_EXAMPLES_DIR)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--selection-csv", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = all_plot_rows()
    if args.selection_csv:
        requested = pd.read_csv(args.selection_csv)
        selected = rows.merge(
            requested,
            on=["model", "dataset", "sample_id", "patched_token_position"],
            how="inner",
        )
        if "category" not in selected.columns:
            selected["category"] = "custom_selection"
    else:
        selected = select_default_examples(rows, per_category=args.per_category)

    if selected.empty:
        raise SystemExit("No examples selected.")

    exported = []
    for _, row in selected.iterrows():
        category = str(row.get("category", "selected"))
        exported.append(render_example(row, category, args.output_dir, rows))

    summary = pd.DataFrame(exported)
    summary_path = args.output_dir / "selected_qualitative_examples.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Exported {len(summary)} examples to {args.output_dir}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
