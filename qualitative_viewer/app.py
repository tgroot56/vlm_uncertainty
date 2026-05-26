from __future__ import annotations

import base64
import html
from functools import lru_cache
from io import BytesIO
import math
import os
from pathlib import Path
import re
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm_uncertainty_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from PIL import Image

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - shown when Streamlit is missing.
    raise SystemExit(
        "Streamlit is not installed in the active Python environment. "
        "Install it or run this app in an environment with streamlit available."
    ) from exc

try:
    from .config import DATASET_LABELS, MODEL_SPECS, OUTPUT_ROOT, SPAN_DISPLAY
    from .data_utils import (
        all_plot_rows,
        blurred_image_path,
        compact_row_dict,
        format_answer_with_conf,
        ground_truth_text,
        highlighted_degraded_image,
        prompt_tokens_for_row,
        sample_record,
        token_html,
    )
except ImportError:  # pragma: no cover - useful when running as `streamlit run app.py`.
    from config import DATASET_LABELS, MODEL_SPECS, OUTPUT_ROOT, SPAN_DISPLAY  # type: ignore
    from data_utils import (  # type: ignore
        all_plot_rows,
        blurred_image_path,
        compact_row_dict,
        format_answer_with_conf,
        ground_truth_text,
        highlighted_degraded_image,
        prompt_tokens_for_row,
        sample_record,
        token_html,
    )


st.set_page_config(
    page_title="Qualitative positional-patching viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading positional-sweep parquet files...")
def load_rows() -> pd.DataFrame:
    return all_plot_rows()


def apply_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    if preset == "Final prompt token restores answer":
        return df[
            df["is_final_prompt_token"]
            & (df["degraded_correctness"] < 1)
            & (df["patched_correctness"] > df["degraded_correctness"])
        ].sort_values(["accuracy_recovery", "patched_correctness"], ascending=[False, False])
    if preset == "Visual token restores answer":
        return df[
            (df["span_label"] == "visual")
            & (df["degraded_correctness"] < 1)
            & (df["patched_correctness"] > df["degraded_correctness"])
        ].sort_values(["accuracy_recovery", "patched_correctness"], ascending=[False, False])
    if preset == "Confidence recovers, accuracy does not":
        return df[(df["confidence_recovery"] > 0) & (df["accuracy_recovery"] <= 0)].sort_values(
            "confidence_recovery", ascending=False
        )
    if preset == "Accuracy recovers, confidence does not":
        return df[(df["accuracy_recovery"] > 0) & (df["confidence_recovery"] <= 0)].sort_values(
            "accuracy_recovery", ascending=False
        )
    if preset == "Confidence exceeds clean":
        return df[df["confidence_exceeds_clean"]].sort_values("patched_confidence", ascending=False)
    if preset == "Patching changes answer":
        return df[df["answer_changed"]].sort_values(["accuracy_recovery", "confidence_recovery"], ascending=False)
    if preset == "Top 20 by accuracy recovery":
        return df.sort_values("accuracy_recovery", ascending=False).head(20)
    if preset == "Top 20 by confidence recovery":
        return df.sort_values("confidence_recovery", ascending=False).head(20)
    return df


def filter_tri_state(df: pd.DataFrame, label: str, col: str) -> pd.DataFrame:
    choice = st.sidebar.selectbox(label, ["Any", "Yes", "No"], key=f"filter_{col}")
    if choice == "Yes":
        return df[df[col]]
    if choice == "No":
        return df[~df[col]]
    return df


PANEL_FONT_REGULAR = Path("/usr/share/fonts/urw-base35/URWBookman-Light.otf")
PANEL_FONT_BOLD = Path("/usr/share/fonts/urw-base35/URWBookman-Demi.otf")


def _font_face_css() -> str:
    if not PANEL_FONT_REGULAR.is_file() or not PANEL_FONT_BOLD.is_file():
        return ""
    regular = base64.b64encode(PANEL_FONT_REGULAR.read_bytes()).decode("ascii")
    bold = base64.b64encode(PANEL_FONT_BOLD.read_bytes()).decode("ascii")
    return (
        "@font-face{font-family:'QVPanelFont';"
        f"src:url(data:font/otf;base64,{regular}) format('opentype');"
        "font-weight:400;font-style:normal;}"
        "@font-face{font-family:'QVPanelFont';"
        f"src:url(data:font/otf;base64,{bold}) format('opentype');"
        "font-weight:700;font-style:normal;}"
    )


THREE_PANEL_CSS = "<style>" + _font_face_css() + """
.qv-three-panel {
    display: grid;
    grid-template-columns: repeat(3, minmax(280px, 1fr));
    gap: 18px;
    margin: 18px 0 24px;
    align-items: stretch;
    width: 100%;
    min-width: 960px;
}
.qv-three-panel-shell {
    width: 100%;
    overflow-x: auto;
    padding-bottom: 8px;
}
.qv-panel {
    min-height: 590px;
    display: flex;
    flex-direction: column;
    padding: 14px 18px 18px;
    border: 2px dashed #9ca3af;
    border-radius: 28px;
    background: #ffffff;
    color: #111827;
    font-family: "QVPanelFont", "Comic Sans MS", "Comic Sans", "Comic Neue", "Trebuchet MS", sans-serif;
    box-sizing: border-box;
}
.qv-panel-title {
    text-align: center;
    font-size: 1.28rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 12px;
}
.qv-image-box {
    height: 250px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
}
.qv-image-box img {
    max-width: 100%;
    max-height: 250px;
    object-fit: contain;
}
.qv-image-missing {
    width: 100%;
    min-height: 160px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #d1d5db;
    color: #6b7280;
    text-align: center;
    padding: 12px;
}
.qv-patch-note {
    font-size: 0.82rem;
    line-height: 1.3;
    color: #4b5563;
    margin: -2px 0 10px;
}
.qv-prompt {
    flex: 0 0 auto;
    font-family: inherit;
    white-space: normal;
    overflow-wrap: anywhere;
    font-size: 0.96rem;
    line-height: 1.42;
}
.qv-answer {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #e5e7eb;
    font-family: inherit;
    font-size: 1.04rem;
    line-height: 1.35;
}
.qv-patched-token {
    font-weight: 800;
    background: #fff1a8;
    border-radius: 3px;
    padding: 0 2px;
}
.qv-image-token {
    color: #374151;
    font-weight: 700;
}
</style>
"""


def _image_data_uri(source: str | Path | Image.Image | None) -> str:
    if source is None:
        return ""
    if isinstance(source, Image.Image):
        buffer = BytesIO()
        source.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    path_text = str(source)
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_file():
        return ""
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _image_html(source: str | Path | Image.Image | None, fallback: str) -> str:
    data_uri = _image_data_uri(source)
    if not data_uri:
        return f"<div class='qv-image-missing'>{html.escape(fallback)}</div>"
    return f"<img src='{html.escape(data_uri)}' alt='{html.escape(fallback)}'>"


def _token_piece(token: dict[str, Any]) -> str:
    raw = str(token.get("token", ""))
    decoded = token.get("decoded")
    if decoded is not None:
        piece = str(decoded)
        if raw == "▁":
            return " "
        if raw.startswith("▁") and piece and not piece.startswith(" "):
            return " " + piece
        if raw.startswith("Ġ") and piece and not piece.startswith(" "):
            return " " + piece
        if piece:
            return piece
    piece = str(token.get("display", token.get("token", "")))
    if bool(token.get("exact", False)):
        piece = piece.replace("▁", " ").replace("Ġ", " ")
    return piece


def _visible_bracket_text(piece: str, token: dict[str, Any]) -> str:
    if piece == "\n":
        return r"\n"
    if piece == "\t":
        return r"\t"
    if piece.strip():
        return piece
    display = str(token.get("display", token.get("token", "")))
    return display if display.strip() else " "


def _prompt_text_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>").replace("\t", "    ")


def _prompt_html_from_tokens(
    tokens: list[dict[str, Any]],
    patched_pos: int,
    highlight_patched_text: bool,
) -> str:
    if not tokens:
        return "<span style='color:#6b7280'>Prompt unavailable.</span>"

    pieces: list[str] = []
    in_visual_span = False
    for token in tokens:
        span = str(token.get("span_label", ""))
        pos = int(token["position"])
        if span == "visual":
            if not in_visual_span:
                pieces.append("<span class='qv-image-token'>[image tokens]</span>")
                in_visual_span = True
            continue

        in_visual_span = False
        piece = _token_piece(token)
        if highlight_patched_text and pos == patched_pos:
            bracketed = _visible_bracket_text(piece, token)
            pieces.append(
                f"<strong class='qv-patched-token'>[{html.escape(bracketed)}]</strong>"
            )
        else:
            pieces.append(_prompt_text_html(piece))
    return "".join(pieces).strip()


def _prompt_segments_for_export(
    tokens: list[dict[str, Any]],
    patched_pos: int,
    mark_patched_text: bool,
) -> list[tuple[str, bool]]:
    if not tokens:
        return [("Prompt unavailable.", False)]

    pieces: list[tuple[str, bool]] = []
    in_visual_span = False
    for token in tokens:
        span = str(token.get("span_label", ""))
        pos = int(token["position"])
        if span == "visual":
            if not in_visual_span:
                pieces.append(("[image tokens]", False))
                in_visual_span = True
            continue

        in_visual_span = False
        piece = _token_piece(token)
        if mark_patched_text and pos == patched_pos:
            piece = f"[{_visible_bracket_text(piece, token)}]"
            pieces.append((piece, True))
        else:
            pieces.append((piece, False))

    if pieces and not pieces[0][0].strip():
        pieces[0] = (pieces[0][0].lstrip(), pieces[0][1])
    if pieces and not pieces[-1][0].strip():
        pieces[-1] = (pieces[-1][0].rstrip(), pieces[-1][1])
    return pieces


def _answer_line(answer: Any, confidence: Any) -> str:
    return f"Answer: {format_answer_with_conf(answer, confidence)}"


def _patch_activation_label(row: pd.Series) -> str:
    span = SPAN_DISPLAY.get(str(row["span_label"]), str(row["span_label"]))
    return f"{span}, token position {int(row['patched_token_position'])}"


def _panel_html(
    title: str,
    image_source: str | Path | Image.Image | None,
    prompt_html: str,
    answer_text: str,
    fallback: str,
) -> str:
    return (
        "<section class='qv-panel'>"
        f"<div class='qv-panel-title'>{html.escape(title)}</div>"
        f"<div class='qv-image-box'>{_image_html(image_source, fallback)}</div>"
        f"<div class='qv-prompt'>{prompt_html}</div>"
        f"<div class='qv-answer'>{html.escape(answer_text)}</div>"
        "</section>"
    )


def _selected_run_images(row: pd.Series, record: dict) -> tuple[str, str, str | Image.Image | None]:
    clean_path = str(record.get("clean_image_path", ""))
    degraded_path = blurred_image_path(record, str(row["severity"]))
    patched_image: str | Image.Image | None = degraded_path

    if str(row["span_label"]) == "visual":
        patched_pos = int(row["patched_token_position"])
        visual_token_count = int(row.get("num_visual_tokens", 0) or 0)
        visual_rel = patched_pos - int(row.get("visual_start", patched_pos))
        highlighted, _, _ = highlighted_degraded_image(
            str(row["model"]),
            clean_path,
            degraded_path,
            visual_rel,
            visual_token_count,
            row.get("visual_geometry"),
            max_side=2400,
        )
        if highlighted is not None:
            patched_image = highlighted
    return clean_path, degraded_path, patched_image


def show_three_panel_layout(
    row: pd.Series,
    record: dict,
    token_rows: list[dict[str, Any]],
    notes: list[str],
) -> None:
    patched_pos = int(row["patched_token_position"])
    span_label = str(row["span_label"])
    clean_path, degraded_path, patched_image = _selected_run_images(row, record)

    prompt_html = _prompt_html_from_tokens(token_rows, patched_pos, highlight_patched_text=False)
    patched_prompt_html = _prompt_html_from_tokens(
        token_rows,
        patched_pos,
        highlight_patched_text=(span_label != "visual"),
    )

    if notes:
        for note in notes:
            st.info(note)

    panels = [
        _panel_html(
            "(1) Clean Run",
            clean_path,
            prompt_html,
            _answer_line(row["clean_answer"], row["clean_confidence"]),
            "Clean image unavailable",
        ),
        _panel_html(
            "(2) Corrupted Run",
            degraded_path,
            prompt_html,
            _answer_line(row["degraded_answer"], row["degraded_confidence"]),
            "Corrupted image unavailable",
        ),
        _panel_html(
            "(3) Patched Run",
            patched_image,
            patched_prompt_html,
            _answer_line(row["patched_answer"], row["patched_confidence"]),
            "Patched image unavailable",
        ),
    ]
    st.html(
        THREE_PANEL_CSS
        + "<div class='qv-three-panel-shell'><div class='qv-three-panel'>"
        + "".join(panels)
        + "</div></div>"
    )


def show_outputs(row: pd.Series) -> None:
    output_rows = [
        {
            "run": "clean",
            "answer": format_answer_with_conf(row["clean_answer"], row["clean_confidence"]),
            "correctness": f"{float(row['clean_correctness']):.3f}",
        },
        {
            "run": "degraded",
            "answer": format_answer_with_conf(row["degraded_answer"], row["degraded_confidence"]),
            "correctness": f"{float(row['degraded_correctness']):.3f}",
        },
        {
            "run": "patched",
            "answer": format_answer_with_conf(row["patched_answer"], row["patched_confidence"]),
            "correctness": f"{float(row['patched_correctness']):.3f}",
        },
    ]
    st.dataframe(pd.DataFrame(output_rows), hide_index=True, use_container_width=True)


def show_images(row: pd.Series, record: dict) -> None:
    clean_path = str(record.get("clean_image_path", ""))
    degraded_path = blurred_image_path(record, str(row["severity"]))
    cols = st.columns(3 if row["span_label"] == "visual" else 2)
    if clean_path and Path(clean_path).exists():
        cols[0].image(clean_path, caption="Clean image", use_container_width=True)
    else:
        cols[0].warning("Clean image path is missing or unreadable.")
    if degraded_path and Path(degraded_path).exists():
        cols[1].image(degraded_path, caption="Degraded image", use_container_width=True)
    else:
        cols[1].warning("Degraded image path is missing or unreadable.")

    if row["span_label"] != "visual":
        return

    visual_token_count = int(row.get("num_visual_tokens", 0) or 0)
    visual_rel = int(row["patched_token_position"]) - int(row.get("visual_start", row["patched_token_position"]))
    image, meta, note = highlighted_degraded_image(
        str(row["model"]),
        clean_path,
        degraded_path,
        visual_rel,
        visual_token_count,
        row.get("visual_geometry"),
    )
    if image is None:
        if meta.get("reason") == "boundary":
            cols[2].info(note)
        else:
            cols[2].warning(f"Could not map this visual token to an image patch. {note}")
    else:
        cols[2].image(image, caption="Degraded image with patched visual token highlighted", use_container_width=True)
        cols[2].caption(note)
        cols[2].caption(
            f"Grid cell row {meta.get('grid_row')}, col {meta.get('grid_col')} "
            f"in {meta.get('grid_rows')}x{meta.get('grid_cols')} grid."
        )


def _load_export_image(source: str | Path | Image.Image | None) -> Image.Image | None:
    if source is None:
        return None
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    path_text = str(source)
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


def _wrap_segments_for_export(
    segments: list[tuple[str, bool]],
    width: int = 36,
) -> list[list[tuple[str, bool]]]:
    lines: list[list[tuple[str, bool]]] = [[]]
    current_len = 0

    for text, highlighted in segments:
        for part in re.findall(r"\n|[^\S\n]+|[^\s]+", text):
            if part == "\n":
                lines.append([])
                current_len = 0
                continue

            if part.isspace():
                if current_len > 0:
                    lines[-1].append((part, highlighted))
                    current_len += len(part)
                continue

            if current_len > 0 and current_len + len(part) > width:
                lines.append([])
                current_len = 0

            lines[-1].append((part, highlighted))
            current_len += len(part)

    return lines or [[("Prompt unavailable.", False)]]


def _line_text(line: list[tuple[str, bool]]) -> str:
    return "".join(part for part, _ in line)


@lru_cache(maxsize=1)
def _export_font_properties() -> tuple[FontProperties, FontProperties]:
    if PANEL_FONT_REGULAR.is_file() and PANEL_FONT_BOLD.is_file():
        return FontProperties(fname=PANEL_FONT_REGULAR), FontProperties(fname=PANEL_FONT_BOLD)
    return FontProperties(family="DejaVu Sans"), FontProperties(family="DejaVu Sans", weight="bold")


def _draw_export_panel(
    ax: plt.Axes,
    title: str,
    image_source: str | Path | Image.Image | None,
    prompt_segments: list[tuple[str, bool]],
    answer: str,
    line_count: int,
) -> None:
    body_font, title_font = _export_font_properties()
    prompt_size = 11 if line_count <= 9 else max(7.0, 11 - 0.35 * (line_count - 9))
    prompt_top = 0.48
    axis_height_inches = max(1.0, ax.figure.get_figheight() * 0.90)
    line_step = (prompt_size / 72.0 / axis_height_inches) * 1.28
    prompt_bottom = prompt_top - line_step * max(1, line_count)
    divider_gap = min(0.048, line_step * 1.9)
    divider_y = max(0.105, prompt_bottom - divider_gap)
    answer_y = divider_y - min(0.030, line_step * 1.1)
    answer_line_step = (12 / 72.0 / axis_height_inches) * 1.25
    content_bottom = max(0.035, answer_y - answer_line_step - 0.025)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    border = FancyBboxPatch(
        (0.025, content_bottom),
        0.95,
        0.975 - content_bottom,
        boxstyle="round,pad=0.018,rounding_size=0.08",
        fill=False,
        linestyle=(0, (4, 4)),
        linewidth=1.1,
        edgecolor="#8f949d",
        transform=ax.transAxes,
    )
    ax.add_patch(border)
    ax.text(
        0.5,
        0.94,
        title,
        ha="center",
        va="top",
        fontsize=16,
        fontproperties=title_font,
        transform=ax.transAxes,
    )

    image = _load_export_image(image_source)
    image_ax = ax.inset_axes([0.18, 0.55, 0.64, 0.31])
    image_ax.axis("off")
    if image is None:
        image_ax.text(
            0.5,
            0.5,
            "Image unavailable",
            ha="center",
            va="center",
            fontsize=10,
            fontproperties=body_font,
        )
    else:
        image_ax.imshow(image)

    renderer = ax.figure.canvas.get_renderer()
    y = prompt_top
    for line in _wrap_segments_for_export(prompt_segments):
        x = 0.075
        for part, highlighted in line:
            bbox = None
            if highlighted:
                bbox = {
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "#fff1a8",
                    "edgecolor": "none",
                }
            text_artist = ax.text(
                x,
                y,
                part,
                ha="left",
                va="top",
                fontsize=prompt_size,
                fontproperties=body_font,
                bbox=bbox,
                transform=ax.transAxes,
            )
            extent = text_artist.get_window_extent(renderer=renderer)
            x += extent.width / ax.bbox.width
        y -= line_step

    ax.plot(
        [0.075, 0.925],
        [divider_y, divider_y],
        color="#e5e7eb",
        linewidth=0.8,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        0.075,
        answer_y,
        answer,
        ha="left",
        va="top",
        fontsize=12,
        fontproperties=body_font,
        transform=ax.transAxes,
    )


def _safe_export_stem(row: pd.Series) -> str:
    raw = (
        f"three_panel_{row['model']}_{row['dataset']}_"
        f"sample_{row['sample_id']}_pos_{int(row['patched_token_position'])}"
    )
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")


def render_three_panel_export(
    row: pd.Series,
    record: dict,
    token_rows: list[dict[str, Any]],
    file_format: str,
) -> bytes:
    patched_pos = int(row["patched_token_position"])
    span_label = str(row["span_label"])
    clean_path, degraded_path, patched_image = _selected_run_images(row, record)

    prompt_segments = _prompt_segments_for_export(token_rows, patched_pos, mark_patched_text=False)
    patched_prompt_segments = _prompt_segments_for_export(
        token_rows,
        patched_pos,
        mark_patched_text=(span_label != "visual"),
    )
    export_prompts = [prompt_segments, prompt_segments, patched_prompt_segments]
    max_lines = max(len(_wrap_segments_for_export(segments)) for segments in export_prompts)
    fig_height = max(6.0, min(10.5, 5.25 + 0.16 * max_lines))

    fig = plt.figure(figsize=(14.5, fig_height), facecolor="white")
    axes = [
        fig.add_axes([0.035 + i * 0.322, 0.045, 0.295, 0.91])
        for i in range(3)
    ]
    panels = [
        ("(1) Clean Run", clean_path, prompt_segments, _answer_line(row["clean_answer"], row["clean_confidence"])),
        ("(2) Corrupted Run", degraded_path, prompt_segments, _answer_line(row["degraded_answer"], row["degraded_confidence"])),
        (
            "(3) Patched Run",
            patched_image,
            patched_prompt_segments,
            _answer_line(row["patched_answer"], row["patched_confidence"]),
        ),
    ]
    for ax, (title, image_source, prompt_segments_for_panel, answer_text) in zip(axes, panels):
        line_count = len(_wrap_segments_for_export(prompt_segments_for_panel))
        _draw_export_panel(ax, title, image_source, prompt_segments_for_panel, answer_text, line_count)

    buffer = BytesIO()
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03, "facecolor": "white"}
    if file_format == "png":
        fig.savefig(buffer, format="png", dpi=450, **save_kwargs)
    elif file_format == "pdf":
        fig.savefig(buffer, format="pdf", dpi=450, **save_kwargs)
    else:
        plt.close(fig)
        raise ValueError(f"Unsupported export format: {file_format}")
    plt.close(fig)
    return buffer.getvalue()


def show_three_panel_downloads(row: pd.Series, record: dict, token_rows: list[dict[str, Any]]) -> None:
    stem = _safe_export_stem(row)
    pdf_bytes = render_three_panel_export(row, record, token_rows, "pdf")
    png_bytes = render_three_panel_export(row, record, token_rows, "png")
    left, right = st.columns(2)
    left.download_button(
        "Download three-panel PDF",
        data=pdf_bytes,
        file_name=f"{stem}.pdf",
        mime="application/pdf",
    )
    right.download_button(
        "Download three-panel PNG",
        data=png_bytes,
        file_name=f"{stem}.png",
        mime="image/png",
    )
    st.caption("PDF export keeps text and panel borders vector-sharp; images are embedded at high DPI.")


def show_original_layout(
    row: pd.Series,
    record: dict,
    token_rows: list[dict[str, Any]],
    notes: list[str],
) -> None:
    show_images(row, record)

    st.subheader("Prompt tokens")
    if notes:
        for note in notes:
            st.info(note)
    st.markdown(token_html(token_rows, int(row["patched_token_position"])), unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Model outputs")
        show_outputs(row)
    with right:
        st.subheader("Ground truth")
        st.write(ground_truth_text(record, str(row["dataset"])))
        st.caption("Correctness values are the dataset-specific scores stored in the positional-sweep parquet.")


def main() -> None:
    st.title("Qualitative positional-patching viewer")
    rows = load_rows()
    if rows.empty:
        st.error("No positional-sweep rows could be loaded.")
        return

    st.sidebar.header("Filters")
    preset = st.sidebar.selectbox(
        "Preset",
        [
            "None",
            "Final prompt token restores answer",
            "Visual token restores answer",
            "Confidence recovers, accuracy does not",
            "Accuracy recovers, confidence does not",
            "Confidence exceeds clean",
            "Patching changes answer",
            "Top 20 by accuracy recovery",
            "Top 20 by confidence recovery",
        ],
    )

    filtered = rows.copy()
    models = st.sidebar.multiselect("Model", sorted(rows["model"].unique()), default=sorted(rows["model"].unique()))
    datasets = st.sidebar.multiselect(
        "Dataset",
        sorted(rows["dataset"].unique()),
        default=sorted(rows["dataset"].unique()),
        format_func=lambda x: DATASET_LABELS.get(x, x),
    )
    spans = st.sidebar.multiselect(
        "Patched token span",
        sorted(rows["span_label"].dropna().unique()),
        default=sorted(rows["span_label"].dropna().unique()),
        format_func=lambda x: SPAN_DISPLAY.get(x, x),
    )
    filtered = filtered[
        filtered["model"].isin(models)
        & filtered["dataset"].isin(datasets)
        & filtered["span_label"].isin(spans)
    ]

    sample_filter = st.sidebar.text_input("Sample id contains", "")
    if sample_filter:
        filtered = filtered[filtered["sample_id"].astype(str).str.contains(sample_filter, case=False, regex=False)]

    pos_min = int(math.floor(rows["patched_token_position"].min()))
    pos_max = int(math.ceil(rows["patched_token_position"].max()))
    pos_range = st.sidebar.slider("Patched token position", pos_min, pos_max, (pos_min, pos_max))
    filtered = filtered[
        (filtered["patched_token_position"] >= pos_range[0])
        & (filtered["patched_token_position"] <= pos_range[1])
    ]

    filtered = filter_tri_state(filtered, "Answer changed", "answer_changed")
    filtered = filter_tri_state(filtered, "Correctness improved", "correctness_improved")
    filtered = filter_tri_state(filtered, "Confidence improved", "confidence_improved")

    sort_metric = st.sidebar.selectbox(
        "Sort",
        ["accuracy_recovery", "confidence_recovery", "patched_correctness", "patched_confidence"],
    )
    top_k = st.sidebar.number_input("Show top k rows", min_value=1, max_value=5000, value=200, step=10)
    sample_layout = st.sidebar.radio("Selected sample layout", ["Three-panel", "Original"], index=0)
    filtered = apply_preset(filtered, preset).sort_values(sort_metric, ascending=False).head(int(top_k))

    st.caption(
        f"Loaded {len(rows):,} plot-included patched rows from the exact positional-sweep parquets. "
        f"Showing {len(filtered):,} rows after filters."
    )

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    table_cols = [
        "model",
        "dataset",
        "sample_id",
        "span_label",
        "patched_token_position",
        "clean_answer_stripped",
        "degraded_answer_stripped",
        "patched_answer_stripped",
        "clean_correctness",
        "degraded_correctness",
        "patched_correctness",
        "accuracy_recovery",
        "confidence_recovery",
        "answer_changed",
    ]
    st.dataframe(filtered[table_cols], hide_index=True, use_container_width=True, height=260)

    selected_id = st.selectbox("Selected patch", filtered["row_id"].tolist())
    row = filtered[filtered["row_id"] == selected_id].iloc[0]
    record = sample_record(str(row["model"]), str(row["dataset"]), str(row["sample_id"]))

    st.subheader(f"{row['model']} / {DATASET_LABELS.get(row['dataset'], row['dataset'])} / sample {row['sample_id']}")
    st.write(
        f"Patched activation: **{SPAN_DISPLAY.get(row['span_label'], row['span_label'])}**, "
        f"token position **{int(row['patched_token_position'])}** "
        f"(span-relative {int(row['span_rel_pos'])})."
    )

    token_rows, spans, notes = prompt_tokens_for_row(rows, row)
    if spans:
        row = row.copy()
        for key in ("visual_start", "visual_end", "num_visual_tokens", "visual_geometry"):
            if key in spans:
                row[key] = spans[key]

    if sample_layout == "Three-panel":
        show_three_panel_layout(row, record, token_rows, notes)
        st.subheader("Export three-panel figure")
        show_three_panel_downloads(row, record, token_rows)
        st.subheader("Ground truth")
        st.write(ground_truth_text(record, str(row["dataset"])))
        st.caption("Correctness values are the dataset-specific scores stored in the positional-sweep parquet.")
    else:
        show_original_layout(row, record, token_rows, notes)

    st.subheader("Recovery metrics")
    metric_row = compact_row_dict(row, record)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "accuracy recovery": f"{metric_row['accuracy_recovery']:.3f}",
                    "confidence recovery": f"{metric_row['confidence_recovery']:.3f}",
                    "clean correctness": f"{metric_row['clean_correctness']:.3f}",
                    "degraded correctness": f"{metric_row['degraded_correctness']:.3f}",
                    "patched correctness": f"{metric_row['patched_correctness']:.3f}",
                    "clean confidence": f"{metric_row['clean_confidence']:.3f}",
                    "degraded confidence": f"{metric_row['degraded_confidence']:.3f}",
                    "patched confidence": f"{metric_row['patched_confidence']:.3f}",
                }
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"Outputs and exports are written under `{OUTPUT_ROOT}`.")


if __name__ == "__main__":
    main()
