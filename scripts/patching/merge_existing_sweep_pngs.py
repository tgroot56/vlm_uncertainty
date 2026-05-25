"""Merge existing sweep PNGs while keeping answer confidence and score panels."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PANEL_METRIC_TITLE_Y = 52
PANEL_TOP_CROP = 48
ROW_GAP = 28
HEADER_H = 78
LEFT_LABEL_W = 190
BG = "white"
TEXT = "#111827"
MUTED = "#374151"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _center_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=TEXT) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) // 2, y), text, fill=fill, font=font)


def _row(path: Path, dataset_label: str, target_width: int | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    third = w // 3
    left = img.crop((0, PANEL_TOP_CROP, third, h))
    right = img.crop((third * 2, PANEL_TOP_CROP, w, h))

    combined = Image.new("RGB", (left.width + right.width, left.height), BG)
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))

    draw = ImageDraw.Draw(combined)
    title_font = _font(24, bold=True)
    draw.rectangle((0, 0, combined.width, 128), fill=BG)
    _center_text(draw, (left.width // 2, PANEL_METRIC_TITLE_Y), "Answer confidence", title_font)
    _center_text(draw, (left.width + right.width // 2, PANEL_METRIC_TITLE_Y), "Score", title_font)

    if target_width is not None and combined.width != target_width:
        scale = target_width / combined.width
        combined = combined.resize((target_width, int(combined.height * scale)), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (LEFT_LABEL_W + combined.width, combined.height), BG)
    canvas.paste(combined, (LEFT_LABEL_W, 0))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(30, bold=True)
    bbox = draw.textbbox((0, 0), dataset_label, font=label_font)
    draw.text(
        (LEFT_LABEL_W - (bbox[2] - bbox[0]) - 24, 42),
        dataset_label,
        fill=MUTED,
        font=label_font,
    )
    return canvas


def merge(paths: list[tuple[str, Path]], output_base: Path, title: str) -> None:
    rows: list[Image.Image] = []
    target_width: int | None = None
    for label, path in paths:
        row = _row(path, label, target_width)
        target_width = row.width - LEFT_LABEL_W
        rows.append(row)

    width = max(row.width for row in rows)
    height = HEADER_H + sum(row.height for row in rows) + ROW_GAP * (len(rows) - 1)
    out = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(out)
    _center_text(draw, (width // 2, 18), title, _font(34, bold=True))

    y = HEADER_H
    for row in rows:
        out.paste(row, ((width - row.width) // 2, y))
        y += row.height + ROW_GAP

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    out.save(png_path)
    out.save(pdf_path, "PDF", resolution=200.0)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


def main() -> None:
    qwen_pos_root = Path("/projects/prjs2014/patching/positional_patching_results_including_visual/Qwen_Qwen3-5-9B")
    qwen_layer_root = Path("/projects/prjs2014/patching/layer_sweep_results/Qwen_Qwen3-5-9B")
    llava_pos_root = Path("/projects/prjs2014/patching/positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf")
    llava_layer_root = Path("/projects/prjs2014/patching/layer_sweep_results/llava-hf_llava-1-5-7b-hf")

    qwen_pos = [
        ("POPE", qwen_pos_root / "pope/extreme_answer_confidence/positional_sweep_combined_filtered_raw_nofilter.png"),
        ("VQA-v2", qwen_pos_root / "vqa-v2/extreme/positional_sweep_combined_filtered_raw_nofilter.png"),
        ("ImageNet-R", qwen_pos_root / "imagenet-r/extreme_answer_confidence/positional_sweep_combined_filtered_raw_nofilter.png"),
        ("COCO-QA-VI", qwen_pos_root / "coco-qa-vi/extreme/positional_sweep_combined_filtered_raw_nofilter.png"),
    ]
    qwen_layer = [
        ("POPE", qwen_layer_root / "pope/extreme_answer_confidence_finaltoken/layer_sweep_combined_filtered_raw_nofilter.png"),
        ("VQA-v2", qwen_layer_root / "vqa-v2/extreme_answer_confidence_finaltoken/layer_sweep_combined_filtered_raw_nofilter.png"),
        ("ImageNet-R", qwen_layer_root / "imagenet-r/extreme_answer_confidence_finaltoken/layer_sweep_combined_filtered_raw_nofilter.png"),
        ("COCO-QA-VI", qwen_layer_root / "coco-qa-vi/extreme_answer_confidence_finaltoken/layer_sweep_combined_filtered_raw_nofilter.png"),
    ]
    llava_pos = [
        ("POPE", llava_pos_root / "pope_bug_fix/extreme/positional_sweep_combined_filtered_raw_nofilter.png"),
        ("VQA-v2", llava_pos_root / "vqa-v2_bug_fix/extreme/positional_sweep_combined_filtered_raw_nofilter.png"),
        ("ImageNet-R", llava_pos_root / "imagenet-r_bug_fix/extreme/positional_sweep_combined_filtered_raw_nofilter.png"),
        ("COCO-QA-VI", llava_pos_root / "coco-qa-vi_bug_fix/extreme/positional_sweep_combined_filtered_raw_nofilter.png"),
    ]
    llava_layer = [
        ("POPE", llava_layer_root / "pope/extreme_answer_confidence/layer_sweep_combined_filtered_raw_nofilter.png"),
        ("VQA-v2", llava_layer_root / "vqa-v2/extreme_answer_confidence/layer_sweep_combined_filtered_raw_nofilter.png"),
        ("ImageNet-R", llava_layer_root / "imagenet-r/extreme_answer_confidence/layer_sweep_combined_filtered_raw_nofilter.png"),
        ("COCO-QA-VI", llava_layer_root / "coco-qa-vi/extreme_answer_confidence/layer_sweep_combined_filtered_raw_nofilter.png"),
    ]

    merge(qwen_pos, qwen_pos_root / "merged_answer_confidence_score_positional_sweep", "Qwen3-5-9B: Positional Sweep")
    merge(qwen_layer, qwen_layer_root / "merged_answer_confidence_score_layer_sweep", "Qwen3-5-9B: Layer Sweep")
    merge(llava_pos, llava_pos_root / "merged_answer_confidence_score_positional_sweep", "LLaVA-1.5-7B: Positional Sweep")
    merge(llava_layer, llava_layer_root / "merged_answer_confidence_score_layer_sweep", "LLaVA-1.5-7B: Layer Sweep")


if __name__ == "__main__":
    main()
