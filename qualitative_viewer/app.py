from __future__ import annotations

import base64
import ast
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
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - shown when Streamlit is missing.
    raise SystemExit(
        "Streamlit is not installed in the active Python environment. "
        "Install it or run this app in an environment with streamlit available."
    ) from exc

try:
    from . import data_utils as data_utils_module
    from .config import (
        DATASET_LABELS,
        GRADIENT_GUIDED_DATASET,
        GRADIENT_GUIDED_LAYERS,
        GRADIENT_GUIDED_MODEL_LABEL,
        MODEL_SPECS,
        OUTPUT_ROOT,
        SPAN_DISPLAY,
    )
    from .data_utils import (
        all_layer_plot_rows,
        all_plot_rows,
        blurred_image_path,
        compact_row_dict,
        ensure_viewer_helper_columns,
        format_answer_with_conf,
        gradient_guided_sample_data,
        gradient_guided_sample_ids,
        ground_truth_text,
        highlighted_degraded_image,
        prompt_tokens_for_row,
        sample_record,
        softmax_changes_available,
        softmax_changes_data,
        softmax_storage_status,
        token_html,
        visual_token_rectangle,
    )
except ImportError:  # pragma: no cover - useful when running as `streamlit run app.py`.
    import data_utils as data_utils_module  # type: ignore
    from config import (  # type: ignore
        DATASET_LABELS,
        GRADIENT_GUIDED_DATASET,
        GRADIENT_GUIDED_LAYERS,
        GRADIENT_GUIDED_MODEL_LABEL,
        MODEL_SPECS,
        OUTPUT_ROOT,
        SPAN_DISPLAY,
    )
    from data_utils import (  # type: ignore
        all_layer_plot_rows,
        all_plot_rows,
        blurred_image_path,
        compact_row_dict,
        ensure_viewer_helper_columns,
        format_answer_with_conf,
        gradient_guided_sample_data,
        gradient_guided_sample_ids,
        ground_truth_text,
        highlighted_degraded_image,
        prompt_tokens_for_row,
        sample_record,
        softmax_changes_available,
        softmax_changes_data,
        softmax_storage_status,
        token_html,
        visual_token_rectangle,
    )


st.set_page_config(
    page_title="Qualitative activation-patching viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading sweep parquet files...")
def load_rows(sweep_type: str) -> pd.DataFrame:
    if sweep_type == "Layer sweep":
        return ensure_viewer_helper_columns(all_layer_plot_rows())
    return ensure_viewer_helper_columns(all_plot_rows())


def clear_app_caches() -> None:
    load_rows.clear()
    data_utils_module.clear_runtime_caches()
    _export_font_properties.cache_clear()


CORRECTNESS_PATTERNS = [
    "C -> C -> C",
    "C -> C -> I",
    "C -> I -> C",
    "C -> I -> I",
    "I -> C -> C",
    "I -> C -> I",
    "I -> I -> C",
    "I -> I -> I",
]

CORRECTNESS_PRESETS = {
    "None": None,
    "Accuracy recovered": "C -> I -> C",
    "Failed recovery": "C -> I -> I",
    "Already robust": "C -> C -> C",
    "Patch improves beyond clean": "I -> I -> C",
    "Patch harms accuracy": "C -> C -> I",
}

CONFIDENCE_DIRECTIONS = ["increased", "decreased", "unchanged"]
POPE_ANSWER_BUCKETS = ["yes", "no", "other"]
ANY_CHOICE = "Any"


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


def _direction_from_delta(delta: pd.Series, tolerance: float) -> pd.Series:
    values = pd.Series("unchanged", index=delta.index, dtype="object")
    values[delta > tolerance] = "increased"
    values[delta < -tolerance] = "decreased"
    return values


def add_confidence_direction_columns(rows: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows = rows.copy()
    rows["corruption_confidence_direction"] = _direction_from_delta(rows["delta_conf_corruption"], tolerance)
    rows["patch_confidence_direction"] = _direction_from_delta(rows["delta_conf_patch"], tolerance)
    rows["patch_vs_clean_confidence_direction"] = _direction_from_delta(
        rows["delta_conf_patch_vs_clean"], tolerance
    )
    rows["confidence_transition"] = (
        "corruption "
        + rows["corruption_confidence_direction"]
        + ", patch "
        + rows["patch_confidence_direction"]
    )
    return rows


def _apply_correctness_choice(df: pd.DataFrame, col: str, choice: str) -> pd.DataFrame:
    if choice == "Correct":
        return df[df[col]]
    if choice == "Incorrect":
        return df[~df[col]]
    return df


def _apply_yes_no_choice(df: pd.DataFrame, col: str, choice: str) -> pd.DataFrame:
    if choice == "Yes":
        return df[df[col]]
    if choice == "No":
        return df[~df[col]]
    return df


def _apply_direction_choice(df: pd.DataFrame, col: str, choice: str) -> pd.DataFrame:
    if choice == ANY_CHOICE:
        return df
    return df[df[col] == choice.lower()]


def _apply_answer_bucket_choice(df: pd.DataFrame, col: str, choice: str) -> pd.DataFrame:
    if choice == ANY_CHOICE:
        return df
    return df[df[col] == choice]


def _percentage(count: pd.Series, total: int) -> pd.Series:
    if total <= 0:
        return pd.Series(0.0, index=count.index)
    return (count.astype(float) / float(total) * 100.0).round(2)


def _selected_table_value(state: Any, table: pd.DataFrame, col: str) -> str:
    selection = getattr(state, "selection", {})
    rows = selection.get("rows", []) if isinstance(selection, dict) else getattr(selection, "rows", [])
    if not rows:
        return ANY_CHOICE
    row_index = int(rows[0])
    if row_index < 0 or row_index >= len(table):
        return ANY_CHOICE
    return str(table.iloc[row_index][col])


def correctness_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("correctness_pattern", dropna=False)
        .agg(
            count=("row_id", "size"),
            **{
                "mean clean confidence": ("clean_confidence", "mean"),
                "mean corrupted confidence": ("corrupted_confidence", "mean"),
                "mean patched confidence": ("patched_confidence", "mean"),
                "mean delta corrupted-clean confidence": ("delta_conf_corruption", "mean"),
                "mean delta patched-corrupted confidence": ("delta_conf_patch", "mean"),
            },
        )
        .reindex(CORRECTNESS_PATTERNS)
        .reset_index(names="correctness_pattern")
    )
    grouped["count"] = grouped["count"].fillna(0).astype(int)
    grouped.insert(2, "percentage", _percentage(grouped["count"], len(df)))
    return grouped


def confidence_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    transitions = [
        f"corruption {corruption}, patch {patch}"
        for corruption in CONFIDENCE_DIRECTIONS
        for patch in CONFIDENCE_DIRECTIONS
    ]
    grouped = (
        df.groupby("confidence_transition", dropna=False)
        .agg(
            count=("row_id", "size"),
            **{
                "mean clean correctness": ("clean_correctness", "mean"),
                "mean corrupted correctness": ("corrupted_correctness", "mean"),
                "mean patched correctness": ("patched_correctness", "mean"),
                "mean clean confidence": ("clean_confidence", "mean"),
                "mean corrupted confidence": ("corrupted_confidence", "mean"),
                "mean patched confidence": ("patched_confidence", "mean"),
            },
        )
        .reindex(transitions)
        .reset_index(names="confidence_transition")
    )
    grouped["count"] = grouped["count"].fillna(0).astype(int)
    grouped.insert(2, "percentage", _percentage(grouped["count"], len(df)))
    return grouped


def answer_summary_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    grouped = (
        df.groupby("answer_transition", dropna=False)
        .agg(
            count=("row_id", "size"),
            **{
                "mean clean correctness": ("clean_correctness", "mean"),
                "mean corrupted correctness": ("corrupted_correctness", "mean"),
                "mean patched correctness": ("patched_correctness", "mean"),
                "mean clean confidence": ("clean_confidence", "mean"),
                "mean corrupted confidence": ("corrupted_confidence", "mean"),
                "mean patched confidence": ("patched_confidence", "mean"),
            },
        )
        .sort_values("count", ascending=False)
    )
    if dataset == "pope":
        transitions = [
            f"{clean} -> {corrupted} -> {patched}"
            for clean in POPE_ANSWER_BUCKETS
            for corrupted in POPE_ANSWER_BUCKETS
            for patched in POPE_ANSWER_BUCKETS
        ]
        grouped = grouped.reindex(transitions)
    else:
        grouped = grouped.head(20)
    grouped = grouped.reset_index(names="answer_transition")
    grouped["count"] = grouped["count"].fillna(0).astype(int)
    grouped.insert(2, "percentage", _percentage(grouped["count"], len(df)))
    return grouped


def validation_summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = df.groupby(["model", "dataset"])["row_id"].size().rename("total")
    combos = df[["model", "dataset"]].drop_duplicates().assign(_join_key=1)

    correctness_counts = (
        df.groupby(["model", "dataset", "correctness_pattern"])["row_id"]
        .size()
        .rename("count")
        .reset_index()
    )
    correctness_categories = pd.DataFrame({"correctness_pattern": CORRECTNESS_PATTERNS, "_join_key": 1})
    correctness = (
        combos.merge(correctness_categories, on="_join_key", how="outer")
        .drop(columns=["_join_key"])
        .merge(correctness_counts, on=["model", "dataset", "correctness_pattern"], how="left")
        .merge(totals.reset_index(), on=["model", "dataset"], how="left")
    )
    correctness["count"] = correctness["count"].fillna(0).astype(int)
    correctness["percentage"] = (correctness["count"] / correctness["total"] * 100).round(2)
    correctness = correctness.drop(columns=["total"]).sort_values(["model", "dataset", "correctness_pattern"])

    confidence_counts = (
        df.groupby(["model", "dataset", "confidence_transition"])["row_id"]
        .size()
        .rename("count")
        .reset_index()
    )
    confidence_transitions = [
        f"corruption {corruption}, patch {patch}"
        for corruption in CONFIDENCE_DIRECTIONS
        for patch in CONFIDENCE_DIRECTIONS
    ]
    confidence_categories = pd.DataFrame({"confidence_transition": confidence_transitions, "_join_key": 1})
    confidence = (
        combos.merge(confidence_categories, on="_join_key", how="outer")
        .drop(columns=["_join_key"])
        .merge(confidence_counts, on=["model", "dataset", "confidence_transition"], how="left")
        .merge(totals.reset_index(), on=["model", "dataset"], how="left")
    )
    confidence["count"] = confidence["count"].fillna(0).astype(int)
    confidence["percentage"] = (confidence["count"] / confidence["total"] * 100).round(2)
    confidence = confidence.drop(columns=["total"]).sort_values(["model", "dataset", "confidence_transition"])
    return correctness, confidence


def render_summary_tables(
    summary_base: pd.DataFrame,
    dataset: str,
) -> tuple[str, str, str]:
    correctness_table = correctness_summary_table(summary_base)
    confidence_table = confidence_summary_table(summary_base)
    answer_table = answer_summary_table(summary_base, dataset)

    st.subheader("Summary tables")
    st.caption("Tables summarize the selected model/dataset before detailed sidebar filters are applied.")
    if "summary_table_reset_token" not in st.session_state:
        st.session_state["summary_table_reset_token"] = 0
    if st.button("Clear summary table selections"):
        st.session_state["summary_table_reset_token"] += 1
    reset_token = st.session_state["summary_table_reset_token"]

    correctness_filter = ANY_CHOICE
    confidence_filter = ANY_CHOICE
    answer_filter = ANY_CHOICE
    tab_correctness, tab_confidence, tab_answers = st.tabs(
        ["Correctness transitions", "Confidence transitions", "Answer transitions"]
    )

    with tab_correctness:
        state = st.dataframe(
            correctness_table,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"correctness_summary_{reset_token}",
        )
        clicked = _selected_table_value(state, correctness_table, "correctness_pattern")
        dropdown = st.selectbox(
            "Filter sample list by correctness transition",
            [ANY_CHOICE] + correctness_table["correctness_pattern"].tolist(),
            key="correctness_summary_dropdown",
        )
        correctness_filter = clicked if clicked != ANY_CHOICE else dropdown

    with tab_confidence:
        state = st.dataframe(
            confidence_table,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"confidence_summary_{reset_token}",
        )
        clicked = _selected_table_value(state, confidence_table, "confidence_transition")
        dropdown = st.selectbox(
            "Filter sample list by confidence transition",
            [ANY_CHOICE] + confidence_table["confidence_transition"].tolist(),
            key="confidence_summary_dropdown",
        )
        confidence_filter = clicked if clicked != ANY_CHOICE else dropdown

    with tab_answers:
        state = st.dataframe(
            answer_table,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"answer_summary_{reset_token}",
        )
        clicked = _selected_table_value(state, answer_table, "answer_transition")
        dropdown = st.selectbox(
            "Filter sample list by answer transition",
            [ANY_CHOICE] + answer_table["answer_transition"].tolist(),
            key="answer_summary_dropdown",
        )
        answer_filter = clicked if clicked != ANY_CHOICE else dropdown

    return correctness_filter, confidence_filter, answer_filter


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
    position: relative;
}
.qv-image-box img {
    max-width: 100%;
    max-height: 250px;
    object-fit: contain;
}
.qv-patch-visual {
    position: relative;
    margin-bottom: 14px;
}
.qv-patch-visual .qv-image-box {
    margin-bottom: 0;
}
.qv-patch-callout {
    position: absolute;
    z-index: 2;
    right: -24px;
    bottom: -6px;
    width: 172px;
    height: 172px;
    padding: 0;
    border: 0;
    background: transparent;
    box-shadow: 0 4px 12px rgba(17, 24, 39, 0.16);
    box-sizing: border-box;
    overflow: hidden;
}
.qv-patch-callout img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.qv-patch-connector {
    position: absolute;
    z-index: 3;
    inset: 0;
    width: 100%;
    height: 100%;
    overflow: visible;
    pointer-events: none;
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
.qv-four-panel {
    display: grid;
    grid-template-columns: repeat(4, minmax(270px, 1fr));
    gap: 16px;
    margin: 18px 0 24px;
    align-items: stretch;
    width: 100%;
    min-width: 1220px;
}
.qv-four-panel .qv-panel {
    min-height: 760px;
    padding-left: 15px;
    padding-right: 15px;
    display: grid;
    grid-template-rows: 30px 250px 92px minmax(210px, 1fr) 88px;
    row-gap: 10px;
    align-items: start;
}
.qv-four-panel .qv-panel-title {
    margin: 0;
    font-size: 1.05rem;
    line-height: 30px;
    white-space: nowrap;
}
.qv-four-panel .qv-image-box {
    width: 100%;
    height: 250px;
    margin: 0;
}
.qv-four-panel .qv-image-box img {
    width: 100%;
    height: 250px;
    max-width: 100%;
    max-height: 250px;
    object-fit: contain;
}
.qv-four-panel .qv-prompt {
    min-height: 210px;
    max-height: 210px;
    overflow-y: auto;
    font-size: 0.88rem;
}
.qv-gradient-token {
    display: inline;
    font-weight: 800;
    background: #fff1a8;
    border: 1px solid #d97706;
    border-radius: 4px;
    padding: 0 3px;
    white-space: nowrap;
}
.qv-gradient-order {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.2em;
    height: 1.2em;
    margin-right: 2px;
    border-radius: 999px;
    background: #b45309;
    color: #ffffff;
    font-size: 0.72em;
    font-weight: 700;
    vertical-align: 0.15em;
}
.qv-gradient-sequence {
    margin: 0;
    padding-top: 8px;
    border-top: 1px solid #e5e7eb;
    color: #4b5563;
    font-size: 0.78rem;
    line-height: 1.45;
    overflow-y: auto;
}
.qv-run-answers {
    width: 100%;
    min-height: 92px;
    margin: 0;
    padding: 9px 10px;
    border: 1px solid #e5e7eb;
    border-radius: 7px;
    background: #f8fafc;
    color: #111827;
    font-size: 0.80rem;
    line-height: 1.45;
    box-sizing: border-box;
}
.qv-recovery-up {
    color: #047857;
    font-weight: 700;
}
.qv-recovery-down {
    color: #b91c1c;
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
    if piece and all(ch in "\n\r\t" for ch in piece):
        return "".join({"\n": r"\n", "\r": r"\r", "\t": r"\t"}[ch] for ch in piece)
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


def _gradient_token_rows(prompt_tokens: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for token in prompt_tokens.sort_values("token_position").to_dict("records"):
        rows.append(
            {
                "position": int(token["token_position"]),
                "token": str(token.get("token", "")),
                "display": str(token.get("token_text", token.get("token", ""))),
                "decoded": token.get("decoded"),
                "span_label": str(token.get("span_label", "")),
                "exact": True,
            }
        )
    return rows


def _gradient_step_label(step: dict[str, Any]) -> str:
    token_text = str(step.get("token_text", "")).strip()
    if token_text:
        return token_text.replace("\n", r"\n").replace("\t", r"\t")
    return f"token {int(step['token_position'])}"


def _gradient_step_title(step: dict[str, Any]) -> str:
    delta = _fmt_metric(step.get("signed_incremental_objective_improvement"))
    return (
        f"Patch {int(step['patch_order'])}: position {int(step['token_position'])}, "
        f"{step.get('span_label', 'unknown')} span, objective change {delta}"
    )


def _gradient_incremental_recoveries(layer_steps: pd.DataFrame) -> dict[int, float]:
    recoveries: dict[int, float] = {}
    previous = 0.0
    for step in layer_steps.sort_values("patch_order").to_dict("records"):
        order = int(step["patch_order"])
        try:
            current = float(step.get("p_clean_recovery", previous))
        except Exception:
            current = previous
        recoveries[order] = current - previous
        previous = current
    return recoveries


def _gradient_recovery_label(value: float) -> tuple[str, str]:
    if value >= 0:
        return f"↑ +{value:.2f}", "qv-recovery-up"
    return f"↓ {value:.2f}", "qv-recovery-down"


def _gradient_prompt_html(
    tokens: list[dict[str, Any]],
    layer_steps: pd.DataFrame,
) -> str:
    if not tokens:
        return "<span style='color:#6b7280'>Prompt unavailable.</span>"

    steps = {
        int(step["token_position"]): step
        for step in layer_steps.sort_values("patch_order").to_dict("records")
    }
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
        step = steps.get(pos)
        if step is None:
            pieces.append(_prompt_text_html(piece))
            continue
        order = int(step["patch_order"])
        title = html.escape(_gradient_step_title(step))
        visible = html.escape(_visible_bracket_text(piece, token))
        pieces.append(
            f"<span class='qv-gradient-token' title='{title}'>"
            f"<span class='qv-gradient-order'>{order}</span>{visible}</span>"
        )
    return "".join(pieces).strip()


def _gradient_prompt_segments_for_export(
    tokens: list[dict[str, Any]],
    layer_steps: pd.DataFrame,
) -> list[tuple[str, bool]]:
    if not tokens:
        return [("Prompt unavailable.", False)]

    steps = {
        int(step["token_position"]): step
        for step in layer_steps.sort_values("patch_order").to_dict("records")
    }
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
        step = steps.get(pos)
        if step is None:
            pieces.append((piece, False))
        else:
            pieces.append(
                (
                    f"[{int(step['patch_order'])}:{_visible_bracket_text(piece, token)}]",
                    True,
                )
            )
    return pieces


def _gradient_sequence_html(layer_steps: pd.DataFrame) -> str:
    labels = []
    recoveries = _gradient_incremental_recoveries(layer_steps)
    for step in layer_steps.sort_values("patch_order").to_dict("records"):
        order = int(step["patch_order"])
        recovery_text, recovery_class = _gradient_recovery_label(recoveries.get(order, 0.0))
        labels.append(
            f"<span title='{html.escape(_gradient_step_title(step))}'><strong>#{order}</strong> "
            f"<span class='{recovery_class}'>{recovery_text}</span></span>"
        )
    return "<div class='qv-gradient-sequence'><strong>Patch increases:</strong> " + " &nbsp; ".join(labels) + "</div>"


def _gradient_sequence_text(layer_steps: pd.DataFrame) -> str:
    recoveries = _gradient_incremental_recoveries(layer_steps)
    labels = [
        f"#{int(step['patch_order'])} {_gradient_recovery_label(recoveries.get(int(step['patch_order']), 0.0))[0]}"
        for step in layer_steps.sort_values("patch_order").to_dict("records")
    ]
    return "Patch increases: " + "   ".join(labels)


def _gradient_sequence_lines(layer_steps: pd.DataFrame, per_line: int = 5) -> list[str]:
    recoveries = _gradient_incremental_recoveries(layer_steps)
    labels = [
        f"#{int(step['patch_order'])} {_gradient_recovery_label(recoveries.get(int(step['patch_order']), 0.0))[0]}"
        for step in layer_steps.sort_values("patch_order").to_dict("records")
    ]
    lines = []
    for start in range(0, len(labels), per_line):
        prefix = "Patch increases: " if start == 0 else ""
        lines.append(prefix + "   ".join(labels[start : start + per_line]))
    return lines or ["Patch increases: none"]


def _answer_line(answer: Any, confidence: Any) -> str:
    return f"Answer: {format_answer_with_conf(answer, confidence)}"


# Per-panel column lookup: (expressed answer, its confidence, P(clean answer) in that run).
_PANEL_RUN_FIELDS = {
    "clean": ("clean_answer", "clean_confidence", "clean_clean_answer_probability"),
    "corrupted": ("degraded_answer", "degraded_confidence", "corrupted_clean_answer_probability"),
    "patched": ("patched_answer", "patched_confidence", "patched_clean_answer_probability"),
}
# Per-panel POPE binary probabilities for the run.
_PANEL_POPE_FIELDS = {
    "clean": ("clean_pope_yes_probability", "clean_pope_no_probability"),
    "corrupted": ("corrupted_pope_yes_probability", "corrupted_pope_no_probability"),
    "patched": ("patched_pope_yes_probability", "patched_pope_no_probability"),
}


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric


def _panel_answer_block(row: pd.Series, kind: str) -> str:
    """Answer text for one panel (kind in {clean, corrupted, patched}).

    Line 1 is always the run's own expressed answer and its confidence.
    Line 2 is the *clean-answer confidence* — the probability this run assigned
    to the clean run's answer — shown only when it adds information:

    - POPE (binary): always show the complementary answer's probability, e.g.
      ``Answer: no (0.89)`` / ``yes (0.11)`` (uses per-run pope yes/no probs).
    - Other datasets: show the clean answer and the probability this run assigned
      to it, but only when the run's answer differs from the clean answer.

    Shared by the Positional sweep, Layer sweep, and Softmax changes views.
    """
    answer_col, conf_col, clean_prob_col = _PANEL_RUN_FIELDS[kind]
    answer = row.get(answer_col, "")
    confidence = row.get(conf_col)
    dataset = str(row.get("dataset", ""))

    if dataset == "pope":
        pope_line = _pope_panel_lines(row, kind, answer)
        if pope_line is not None:
            return pope_line

    lines = [_answer_line(answer, confidence)]
    clean_answer = row.get("clean_answer", "")
    if data_utils_module.norm_answer(answer) != data_utils_module.norm_answer(clean_answer):
        clean_prob = _finite_or_none(row.get(clean_prob_col))
        if clean_prob is not None:
            lines.append(
                f"Clean answer confidence: "
                f"{data_utils_module.stripped_answer(clean_answer)} ({clean_prob:.2f})"
            )
    return "\n".join(lines)


def _pope_panel_lines(row: pd.Series, kind: str, answer: Any) -> str | None:
    """POPE binary answer block: expressed answer + its prob, then the other answer.

    Returns ``None`` when per-run yes/no probabilities are unavailable so the
    caller can fall back to the generic clean-answer-confidence behaviour.
    """
    yes_col, no_col = _PANEL_POPE_FIELDS[kind]
    p_yes = _finite_or_none(row.get(yes_col))
    p_no = _finite_or_none(row.get(no_col))
    if p_yes is None or p_no is None:
        return None
    answer_text = data_utils_module.stripped_answer(answer)
    bucket = data_utils_module.pope_answer_bucket(answer)
    if bucket == "yes":
        return f"Answer: {answer_text} ({p_yes:.2f})\nno ({p_no:.2f})"
    if bucket == "no":
        return f"Answer: {answer_text} ({p_no:.2f})\nyes ({p_yes:.2f})"
    # Expressed answer is neither yes nor no: show both binary probabilities.
    return f"{_answer_line(answer, row.get(_PANEL_RUN_FIELDS[kind][1]))}\nyes ({p_yes:.2f}), no ({p_no:.2f})"


def _fmt_metric(value: Any) -> str:
    try:
        numeric = float(value)
        if math.isnan(numeric):
            return "NA"
        return f"{numeric:.3f}"
    except Exception:
        return "NA"


def _patch_activation_label(row: pd.Series) -> str:
    span = SPAN_DISPLAY.get(str(row["span_label"]), str(row["span_label"]))
    return f"{span}, token position {int(row['patched_token_position'])}"


def _panel_html(
    title: str,
    image_source: str | Path | Image.Image | None,
    prompt_html: str,
    answer_text: str,
    fallback: str,
    callout_image_source: str | Path | Image.Image | None = None,
    callout_anchor: tuple[float, float] | None = None,
    callout_fallback: str = "Clean patch unavailable",
) -> str:
    image_html = f"<div class='qv-image-box'>{_image_html(image_source, fallback)}</div>"
    if callout_image_source is not None:
        anchor_x, anchor_y = callout_anchor or (0.5, 0.5)
        marker_x = max(0.0, min(100.0, anchor_x * 100.0))
        marker_y = max(0.0, min(57.0, anchor_y * 57.0))
        image_html = (
            "<div class='qv-patch-visual'>"
            f"{image_html}"
            "<svg class='qv-patch-connector' viewBox='0 0 100 100' "
            "preserveAspectRatio='none' aria-hidden='true'>"
            "<defs><marker id='qv-patch-arrow' markerWidth='7' markerHeight='7' "
            "refX='6' refY='3.5' orient='auto'>"
            "<path d='M0,0 L7,3.5 L0,7 z' fill='#d97706'/></marker></defs>"
            f"<line x1='{marker_x:.2f}' y1='{marker_y:.2f}' x2='72' y2='63' "
            "stroke='#d97706' stroke-width='1.8' vector-effect='non-scaling-stroke' "
            "marker-end='url(#qv-patch-arrow)'/>"
            "</svg>"
            "<div class='qv-patch-callout'>"
            f"{_image_html(callout_image_source, callout_fallback)}"
            "</div></div>"
        )
    return (
        "<section class='qv-panel'>"
        f"<div class='qv-panel-title'>{html.escape(title)}</div>"
        f"{image_html}"
        f"<div class='qv-prompt'>{prompt_html}</div>"
        f"<div class='qv-answer'>{html.escape(answer_text).replace(chr(10), '<br>')}</div>"
        "</section>"
    )


def _gradient_panel_html(
    layer: int,
    highlighted_image: str | Path | Image.Image | None,
    answers_html: str,
    prompt_html: str,
    sequence_html: str,
) -> str:
    return (
        "<section class='qv-panel'>"
        f"<div class='qv-panel-title'>Patched Run — Layer {layer}</div>"
        f"<div class='qv-image-box'>{_image_html(highlighted_image, 'Clean image unavailable')}</div>"
        f"{answers_html}"
        f"<div class='qv-prompt'>{prompt_html}</div>"
        f"{sequence_html}"
        "</section>"
    )


def _gradient_geometry(metadata: dict[str, Any]) -> dict[str, Any] | None:
    token_spans = metadata.get("token_spans", {})
    geometry = token_spans.get("visual_geometry") if isinstance(token_spans, dict) else None
    if isinstance(geometry, dict):
        return geometry
    if isinstance(geometry, str):
        try:
            parsed = ast.literal_eval(geometry)
            return parsed if isinstance(parsed, dict) else None
        except (SyntaxError, ValueError):
            return None
    return None


def _gradient_visual_patch_image(
    metadata: dict[str, Any],
    layer_steps: pd.DataFrame,
    max_side: int = 2400,
) -> Image.Image | str:
    clean_path = str(metadata.get("clean_image_path", ""))
    if not clean_path or not Path(clean_path).is_file():
        return clean_path
    image = Image.open(clean_path).convert("RGB")
    token_spans = metadata.get("token_spans", {})
    geometry = _gradient_geometry(metadata)
    visual_start = int(token_spans.get("visual_start", 0)) if isinstance(token_spans, dict) else 0
    visual_count = int(token_spans.get("num_visual_tokens", 0)) if isinstance(token_spans, dict) else 0
    scale = min(max_side / max(image.size), 1.0)
    if scale < 1.0:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )

    draw = ImageDraw.Draw(image, "RGBA")
    font_size = max(12, round(min(image.size) * 0.035))
    try:
        font = ImageFont.truetype(str(PANEL_FONT_BOLD), font_size)
    except OSError:
        font = ImageFont.load_default()
    occupied_badges: list[tuple[int, int, int, int]] = []
    for step in layer_steps.sort_values("patch_order").to_dict("records"):
        if str(step.get("span_label", "")) != "visual":
            continue
        visual_rel = int(step["token_position"]) - visual_start
        rect, _, _ = visual_token_rectangle(
            tuple(round(dimension / scale) for dimension in image.size),
            visual_rel,
            visual_count,
            geometry,
        )
        if rect is None:
            continue
        x0, y0, x1, y1 = [round(value * scale) for value in rect]
        order = int(step["patch_order"])
        color = (245, 158, 11, 255)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(253, 224, 71, 95), outline=color, width=max(3, round(4 * scale)))
        label = str(order)
        bbox = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        label_w = bbox[2] - bbox[0] + 10
        label_h = bbox[3] - bbox[1] + 8
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        gap = 4
        candidates = [
            (center_x - label_w // 2, y0 - label_h - gap),
            (center_x - label_w // 2, y1 + gap),
            (x0 - label_w - gap, center_y - label_h // 2),
            (x1 + gap, center_y - label_h // 2),
            (x0 - label_w - gap, y0 - label_h - gap),
            (x1 + gap, y1 + gap),
        ]
        badge_x, badge_y = candidates[0]
        for candidate_x, candidate_y in candidates:
            candidate_x = max(0, min(image.width - label_w, candidate_x))
            candidate_y = max(0, min(image.height - label_h, candidate_y))
            candidate_box = (candidate_x, candidate_y, candidate_x + label_w, candidate_y + label_h)
            if not any(
                candidate_box[0] < box[2] + 2
                and candidate_box[2] + 2 > box[0]
                and candidate_box[1] < box[3] + 2
                and candidate_box[3] + 2 > box[1]
                for box in occupied_badges
            ):
                badge_x, badge_y = candidate_x, candidate_y
                break
        occupied_badges.append((badge_x, badge_y, badge_x + label_w, badge_y + label_h))
        draw.line(
            (center_x, center_y, badge_x + label_w // 2, badge_y + label_h // 2),
            fill=(180, 83, 9, 235),
            width=2,
        )
        draw.rounded_rectangle(
            (badge_x, badge_y, badge_x + label_w, badge_y + label_h),
            radius=max(3, label_h // 3),
            fill=(180, 83, 9, 245),
            outline=(255, 255, 255, 255),
            width=2,
        )
        draw.text(
            (badge_x + 5, badge_y + 3),
            label,
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(120, 53, 15, 255),
        )
    return image


def _gradient_baseline_answer(metadata: dict[str, Any], baseline_key: str) -> str:
    baseline = metadata.get(baseline_key, {})
    if not isinstance(baseline, dict):
        return "unavailable (NA)"
    confidence = baseline.get(
        "given_generated_answer_confidence",
        baseline.get("answer_geom_mean_probability", baseline.get("top1_probability")),
    )
    return format_answer_with_conf(baseline.get("predicted_answer", ""), confidence)


def _gradient_answer_lines(metadata: dict[str, Any], layer_steps: pd.DataFrame) -> list[str]:
    patched = "unavailable (NA)"
    if not layer_steps.empty:
        final_step = layer_steps.sort_values("patch_order").iloc[-1]
        patched = format_answer_with_conf(
            final_step.get("predicted_answer", ""),
            final_step.get(
                "given_generated_answer_confidence",
                final_step.get("top1_probability"),
            ),
        )
    return [
        f"(1) Clean Run Answer: {_gradient_baseline_answer(metadata, 'baseline_clean')}",
        f"(2) Corrupted Run Answer: {_gradient_baseline_answer(metadata, 'baseline_degraded')}",
        f"(3) Patched Run Answer: {patched}",
    ]


def _gradient_answers_html(metadata: dict[str, Any], layer_steps: pd.DataFrame) -> str:
    lines = "<br>".join(html.escape(line) for line in _gradient_answer_lines(metadata, layer_steps))
    return f"<div class='qv-run-answers'>{lines}</div>"


def _clean_visual_patch_crop(
    clean_path: str,
    visual_rel_pos: int,
    visual_token_count: int,
    visual_geometry: dict[str, Any] | None,
    output_side: int = 480,
    neighborhood_radius: int = 2,
) -> Image.Image | None:
    if not clean_path or not Path(clean_path).is_file():
        return None
    with Image.open(clean_path) as source:
        clean = source.convert("RGB")
    rect, meta, _ = visual_token_rectangle(
        clean.size,
        visual_rel_pos,
        visual_token_count,
        visual_geometry,
    )
    if rect is None:
        return None

    x0, y0, x1, y1 = rect
    grid_row = int(meta.get("grid_row", 0))
    grid_col = int(meta.get("grid_col", 0))
    grid_rows = int(meta.get("grid_rows", 0))
    grid_cols = int(meta.get("grid_cols", 0))
    visual_offset = int((visual_geometry or {}).get("visual_token_offset", 0) or 0)
    neighbor_rects: list[tuple[int, int, int, int]] = []
    for neighbor_row in range(
        max(0, grid_row - neighborhood_radius),
        min(grid_rows, grid_row + neighborhood_radius + 1),
    ):
        for neighbor_col in range(
            max(0, grid_col - neighborhood_radius),
            min(grid_cols, grid_col + neighborhood_radius + 1),
        ):
            neighbor_visual_rel = visual_offset + neighbor_row * grid_cols + neighbor_col
            neighbor_rect, _, _ = visual_token_rectangle(
                clean.size,
                neighbor_visual_rel,
                visual_token_count,
                visual_geometry,
            )
            if neighbor_rect is not None:
                neighbor_rects.append(neighbor_rect)

    if not neighbor_rects:
        neighbor_rects = [rect]
    crop_x0 = min(neighbor[0] for neighbor in neighbor_rects)
    crop_y0 = min(neighbor[1] for neighbor in neighbor_rects)
    crop_x1 = max(neighbor[2] for neighbor in neighbor_rects)
    crop_y1 = max(neighbor[3] for neighbor in neighbor_rects)
    crop = clean.crop((crop_x0, crop_y0, crop_x1, crop_y1))

    scale = output_side / max(crop.size)
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(resized, "RGBA")
    neighbor_outline_width = max(1, round(output_side / 240))
    for neighbor_x0, neighbor_y0, neighbor_x1, neighbor_y1 in neighbor_rects:
        neighbor_marker = (
            round((neighbor_x0 - crop_x0) * scale),
            round((neighbor_y0 - crop_y0) * scale),
            round((neighbor_x1 - crop_x0) * scale) - 1,
            round((neighbor_y1 - crop_y0) * scale) - 1,
        )
        draw.rectangle(
            neighbor_marker,
            outline=(255, 255, 255, 185),
            width=neighbor_outline_width,
        )
    marker = (
        round((x0 - crop_x0) * scale),
        round((y0 - crop_y0) * scale),
        round((x1 - crop_x0) * scale) - 1,
        round((y1 - crop_y0) * scale) - 1,
    )
    outline_width = max(3, round(output_side / 100))
    draw.rectangle(
        marker,
        fill=(245, 158, 11, 45),
        outline=(245, 158, 11, 255),
        width=outline_width,
    )
    return resized


def _selected_run_images(
    row: pd.Series,
    record: dict,
    show_clean_visual_callout: bool = False,
) -> tuple[
    str,
    str,
    str | Image.Image | None,
    Image.Image | None,
    tuple[float, float] | None,
]:
    clean_path = str(record.get("clean_image_path", ""))
    degraded_path = blurred_image_path(record, str(row["severity"]))
    patched_image: str | Image.Image | None = degraded_path
    clean_patch_crop: Image.Image | None = None
    patch_marker_anchor: tuple[float, float] | None = None

    if str(row["span_label"]) == "visual":
        patched_pos = int(row["patched_token_position"])
        visual_token_count = int(row.get("num_visual_tokens", 0) or 0)
        visual_rel = patched_pos - int(row.get("visual_start", patched_pos))
        highlighted, highlight_meta, _ = highlighted_degraded_image(
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
            marker_rect = highlight_meta.get("rect")
            if marker_rect:
                x0, y0, x1, y1 = marker_rect
                patch_marker_anchor = (
                    ((x0 + x1) / 2) / highlighted.width,
                    ((y0 + y1) / 2) / highlighted.height,
                )
        if show_clean_visual_callout:
            clean_patch_crop = _clean_visual_patch_crop(
                clean_path,
                visual_rel,
                visual_token_count,
                row.get("visual_geometry"),
            )
    return clean_path, degraded_path, patched_image, clean_patch_crop, patch_marker_anchor


def show_three_panel_layout(
    row: pd.Series,
    record: dict,
    token_rows: list[dict[str, Any]],
    notes: list[str],
    show_clean_visual_callout: bool = False,
) -> None:
    patched_pos = int(row["patched_token_position"])
    span_label = str(row["span_label"])
    clean_path, degraded_path, patched_image, clean_patch_crop, patch_marker_anchor = _selected_run_images(
        row,
        record,
        show_clean_visual_callout=show_clean_visual_callout,
    )

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
            _panel_answer_block(row, "clean"),
            "Clean image unavailable",
        ),
        _panel_html(
            "(2) Corrupted Run",
            degraded_path,
            prompt_html,
            _panel_answer_block(row, "corrupted"),
            "Corrupted image unavailable",
        ),
        _panel_html(
            "(3) Patched Run",
            patched_image,
            patched_prompt_html,
            _panel_answer_block(row, "patched"),
            "Patched image unavailable",
            callout_image_source=clean_patch_crop,
            callout_anchor=patch_marker_anchor,
            callout_fallback="Clean patch unavailable",
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


def show_top5_softmax_table(row: pd.Series) -> None:
    st.subheader("Top-5 softmax token information")
    status = softmax_storage_status(row)
    if status["status"] == "materialized":
        st.info(
            "Explicit top-5 fields were detected, but their schema is not yet standardized enough "
            "to render safely."
        )
        return

    source_label = (
        "Full softmax stored; top-5 not materialized"
        if status["status"] == "raw_distribution_only"
        else "Softmax distribution not stored"
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"run": run, "top-5 tokens": "Unavailable", "probabilities": "Unavailable", "source status": source_label}
                for run in ("Clean", "Corrupted", "Patched")
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.info(status["detail"])
    st.caption(
        "To make this table available, materialize first stripped-answer-step top-5 token IDs/text "
        "and probabilities while generating each run in "
        "`src/patching/run_patching.py::_compute_stripped_answer_softmax_stats`, then enable "
        "`--save_stripped_answer_softmax` in both layer- and positional-sweep jobs."
    )


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


def _export_content_bottom(
    line_count: int,
    answer: str,
    answers_below_image: bool,
    fig_height: float,
) -> float:
    """Compute the bottom edge (axes fraction) of a panel's dashed box.

    Panels with more answer/prompt lines push this lower. Used both to lay out a
    single panel and, in the export renderers, to find the lowest box bottom in a
    row so every panel can share it and stay bottom-aligned.
    """
    answer_lines = str(answer).splitlines() or [""]
    prompt_size = 11 if line_count <= 9 else max(7.0, 11 - 0.35 * (line_count - 9))
    prompt_top = 0.39 if answers_below_image else 0.48
    axis_height_inches = max(1.0, fig_height * 0.90)
    line_step = (prompt_size / 72.0 / axis_height_inches) * 1.28
    prompt_bottom = prompt_top - line_step * max(1, line_count)
    divider_gap = min(0.048, line_step * 1.9)
    divider_y = max(0.105, prompt_bottom - divider_gap)
    answer_y = 0.525 if answers_below_image else divider_y - min(0.030, line_step * 1.1)
    answer_line_step = (12 / 72.0 / axis_height_inches) * 1.25
    if answers_below_image:
        return max(0.035, prompt_bottom - 0.04)
    return max(0.035, answer_y - answer_line_step * len(answer_lines) - 0.025)


def _draw_export_panel(
    ax: plt.Axes,
    title: str,
    image_source: str | Path | Image.Image | None,
    prompt_segments: list[tuple[str, bool]],
    answer: str,
    line_count: int,
    callout_image_source: str | Path | Image.Image | None = None,
    callout_anchor: tuple[float, float] | None = None,
    answers_below_image: bool = False,
    box_bottom: float | None = None,
) -> None:
    body_font, title_font = _export_font_properties()
    answer_lines = str(answer).splitlines() or [""]
    prompt_size = 11 if line_count <= 9 else max(7.0, 11 - 0.35 * (line_count - 9))
    prompt_top = 0.39 if answers_below_image else 0.48
    axis_height_inches = max(1.0, ax.figure.get_figheight() * 0.90)
    line_step = (prompt_size / 72.0 / axis_height_inches) * 1.28
    prompt_bottom = prompt_top - line_step * max(1, line_count)
    divider_gap = min(0.048, line_step * 1.9)
    divider_y = max(0.105, prompt_bottom - divider_gap)
    answer_y = 0.525 if answers_below_image else divider_y - min(0.030, line_step * 1.1)
    answer_line_step = (12 / 72.0 / axis_height_inches) * 1.25
    content_bottom = _export_content_bottom(
        line_count, answer, answers_below_image, ax.figure.get_figheight()
    )
    # The dashed box may be extended down to a shared bottom so all panels in a
    # row stay bottom-aligned; text still lays out from this panel's own bottom.
    box_bottom = content_bottom if box_bottom is None else min(box_bottom, content_bottom)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    border = FancyBboxPatch(
        (0.025, box_bottom),
        0.95,
        0.975 - box_bottom,
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

    callout_image = _load_export_image(callout_image_source)
    if callout_image is not None:
        callout_ax = ax.inset_axes([0.59, 0.51, 0.34, 0.23])
        callout_ax.set_facecolor("none")
        for spine in callout_ax.spines.values():
            spine.set_visible(False)
        callout_ax.set_xticks([])
        callout_ax.set_yticks([])
        callout_ax.set_xlim(-0.5, callout_image.width - 0.5)
        callout_ax.set_ylim(callout_image.height - 0.5, -0.5)
        callout_ax.set_aspect("equal", adjustable="box")
        callout_ax.imshow(callout_image)
        anchor = callout_anchor or (0.5, 0.5)
        connector = ConnectionPatch(
            xyA=(anchor[0], 1.0 - anchor[1]),
            coordsA="axes fraction",
            axesA=image_ax,
            xyB=(0.5, 0.5),
            coordsB="axes fraction",
            axesB=callout_ax,
            arrowstyle="->",
            color="#d97706",
            linewidth=1.8,
            mutation_scale=15,
            shrinkA=1,
            shrinkB=1,
        )
        connector.set_zorder(10)
        ax.figure.add_artist(connector)

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

    if not answers_below_image:
        ax.plot(
            [0.075, 0.925],
            [divider_y, divider_y],
            color="#e5e7eb",
            linewidth=0.8,
            transform=ax.transAxes,
            clip_on=False,
        )
    for line_idx, answer_line in enumerate(answer_lines):
        ax.text(
            0.075,
            answer_y - answer_line_step * line_idx,
            answer_line,
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
    show_clean_visual_callout: bool = False,
) -> bytes:
    patched_pos = int(row["patched_token_position"])
    span_label = str(row["span_label"])
    clean_path, degraded_path, patched_image, clean_patch_crop, patch_marker_anchor = _selected_run_images(
        row,
        record,
        show_clean_visual_callout=show_clean_visual_callout,
    )

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
        (
            "(1) Clean Run",
            clean_path,
            prompt_segments,
            _panel_answer_block(row, "clean"),
            None,
            None,
        ),
        (
            "(2) Corrupted Run",
            degraded_path,
            prompt_segments,
            _panel_answer_block(row, "corrupted"),
            None,
            None,
        ),
        (
            "(3) Patched Run",
            patched_image,
            patched_prompt_segments,
            _panel_answer_block(row, "patched"),
            clean_patch_crop,
            patch_marker_anchor,
        ),
    ]
    # Extend every dashed box down to the lowest box bottom in the row so the
    # panels stay bottom-aligned even when one run carries an extra answer line.
    shared_box_bottom = min(
        _export_content_bottom(
            len(_wrap_segments_for_export(prompt_segments_for_panel)),
            answer_text,
            False,
            fig_height,
        )
        for _, _, prompt_segments_for_panel, answer_text, _, _ in panels
    )
    for ax, (
        title,
        image_source,
        prompt_segments_for_panel,
        answer_text,
        callout_image_source,
        callout_anchor,
    ) in zip(axes, panels):
        line_count = len(_wrap_segments_for_export(prompt_segments_for_panel))
        _draw_export_panel(
            ax,
            title,
            image_source,
            prompt_segments_for_panel,
            answer_text,
            line_count,
            callout_image_source=callout_image_source,
            callout_anchor=callout_anchor,
            box_bottom=shared_box_bottom,
        )

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


def show_three_panel_downloads(
    row: pd.Series,
    record: dict,
    token_rows: list[dict[str, Any]],
    show_clean_visual_callout: bool = False,
) -> None:
    stem = _safe_export_stem(row)
    pdf_bytes = render_three_panel_export(
        row,
        record,
        token_rows,
        "pdf",
        show_clean_visual_callout=show_clean_visual_callout,
    )
    png_bytes = render_three_panel_export(
        row,
        record,
        token_rows,
        "png",
        show_clean_visual_callout=show_clean_visual_callout,
    )
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


def _gradient_layer_answer(layer_steps: pd.DataFrame) -> str:
    if layer_steps.empty:
        return "Answer: unavailable (NA)"
    final_step = layer_steps.sort_values("patch_order").iloc[-1]
    confidence = final_step.get(
        "given_generated_answer_confidence",
        final_step.get("top1_probability"),
    )
    return _answer_line(final_step.get("predicted_answer", ""), confidence)


def _draw_gradient_export_panel(
    ax: plt.Axes,
    layer: int,
    image_source: str | Path | Image.Image | None,
    metadata: dict[str, Any],
    layer_steps: pd.DataFrame,
    prompt_segments: list[tuple[str, bool]],
) -> None:
    body_font, title_font = _export_font_properties()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.025, 0.035),
            0.95,
            0.94,
            boxstyle="round,pad=0.018,rounding_size=0.08",
            fill=False,
            linestyle=(0, (4, 4)),
            linewidth=1.1,
            edgecolor="#8f949d",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.945,
        f"Patched Run — Layer {layer}",
        ha="center",
        va="top",
        fontsize=13,
        fontproperties=title_font,
        transform=ax.transAxes,
    )

    image = _load_export_image(image_source)
    image_ax = ax.inset_axes([0.12, 0.59, 0.76, 0.27])
    image_ax.axis("off")
    if image is None:
        image_ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center", fontproperties=body_font)
    else:
        image_ax.imshow(image)

    answer_y = 0.565
    for line_idx, line in enumerate(_gradient_answer_lines(metadata, layer_steps)):
        ax.text(
            0.065,
            answer_y - 0.027 * line_idx,
            line,
            ha="left",
            va="top",
            fontsize=8.7,
            fontproperties=body_font,
            transform=ax.transAxes,
        )

    renderer = ax.figure.canvas.get_renderer()
    prompt_lines = _wrap_segments_for_export(prompt_segments, width=35)
    prompt_size = max(7.0, 9.2 - max(0, len(prompt_lines) - 8) * 0.22)
    y = 0.45
    line_step = 0.026
    for line in prompt_lines:
        x = 0.065
        for part, highlighted in line:
            bbox = None
            if highlighted:
                bbox = {"boxstyle": "round,pad=0.10", "facecolor": "#fff1a8", "edgecolor": "none"}
            artist = ax.text(
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
            x += artist.get_window_extent(renderer=renderer).width / ax.bbox.width
        y -= line_step

    ax.plot([0.065, 0.935], [0.115, 0.115], color="#e5e7eb", linewidth=0.8, transform=ax.transAxes)
    for line_idx, summary_line in enumerate(_gradient_sequence_lines(layer_steps)):
        ax.text(
            0.065,
            0.09 - 0.025 * line_idx,
            summary_line,
            ha="left",
            va="top",
            fontsize=7.3,
            fontproperties=body_font,
            transform=ax.transAxes,
        )


def render_gradient_four_panel_export(
    metadata: dict[str, Any],
    prompt_tokens: pd.DataFrame,
    selected_steps: pd.DataFrame,
    file_format: str,
) -> bytes:
    tokens = _gradient_token_rows(prompt_tokens)
    panel_prompts = [
        _gradient_prompt_segments_for_export(
            tokens,
            selected_steps[selected_steps["requested_layer"] == layer],
        )
        for layer in GRADIENT_GUIDED_LAYERS
    ]
    fig = plt.figure(figsize=(19.2, 8.2), facecolor="white")
    axes = [fig.add_axes([0.018 + i * 0.246, 0.045, 0.226, 0.91]) for i in range(4)]
    for ax, layer, prompt_segments in zip(axes, GRADIENT_GUIDED_LAYERS, panel_prompts):
        layer_steps = selected_steps[selected_steps["requested_layer"] == layer]
        _draw_gradient_export_panel(
            ax,
            layer,
            _gradient_visual_patch_image(metadata, layer_steps),
            metadata,
            layer_steps,
            prompt_segments,
        )

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


def show_gradient_guided_view() -> None:
    sample_ids = list(gradient_guided_sample_ids())
    st.sidebar.header("Filters")
    sample_filter = st.sidebar.text_input("Sample id contains", "", key="gradient_sample_filter")
    if sample_filter:
        sample_ids = [sample_id for sample_id in sample_ids if sample_filter.lower() in sample_id.lower()]
    if not sample_ids:
        st.warning("No LLaVA gradient-guided samples match the current filter.")
        return

    st.caption(
        "LLaVA-1.5-7B / POPE random / maximize P(yes). "
        "Numbered highlights show the order selected by gradient-guided patching."
    )
    selected_sample_id = st.selectbox("Selected sample", sample_ids, key="gradient_selected_sample")
    sample_data = gradient_guided_sample_data(selected_sample_id)
    metadata = sample_data["metadata"]
    prompt_tokens = sample_data["prompt_tokens"]
    selected_steps = sample_data["selected_steps"]
    tokens = _gradient_token_rows(prompt_tokens)

    st.subheader(
        f"{GRADIENT_GUIDED_MODEL_LABEL} / "
        f"{DATASET_LABELS.get(GRADIENT_GUIDED_DATASET, GRADIENT_GUIDED_DATASET)} / "
        f"sample {selected_sample_id}"
    )
    st.write(str(metadata.get("question", "")))

    summary_rows = []
    for layer in GRADIENT_GUIDED_LAYERS:
        layer_steps = selected_steps[selected_steps["requested_layer"] == layer]
        if layer_steps.empty:
            summary_rows.append({"layer": layer, "answer": "unavailable", "confidence": float("nan"), "patches": 0})
            continue
        final_step = layer_steps.sort_values("patch_order").iloc[-1]
        summary_rows.append(
            {
                "layer": layer,
                "answer": data_utils_module.stripped_answer(final_step.get("predicted_answer", "")),
                "confidence": float(
                    final_step.get(
                        "given_generated_answer_confidence",
                        final_step.get("top1_probability", float("nan")),
                    )
                ),
                "patches": len(layer_steps),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    panels = []
    for layer in GRADIENT_GUIDED_LAYERS:
        layer_steps = selected_steps[selected_steps["requested_layer"] == layer]
        panels.append(
            _gradient_panel_html(
                layer,
                _gradient_visual_patch_image(metadata, layer_steps),
                _gradient_answers_html(metadata, layer_steps),
                _gradient_prompt_html(tokens, layer_steps),
                _gradient_sequence_html(layer_steps),
            )
        )
    st.html(
        THREE_PANEL_CSS
        + "<div class='qv-three-panel-shell'><div class='qv-four-panel'>"
        + "".join(panels)
        + "</div></div>"
    )

    st.subheader("Export four-panel figure")
    pdf_bytes = render_gradient_four_panel_export(
        metadata,
        prompt_tokens,
        selected_steps,
        "pdf",
    )
    png_bytes = render_gradient_four_panel_export(
        metadata,
        prompt_tokens,
        selected_steps,
        "png",
    )
    stem = f"gradient_guided_llava_pope_sample_{selected_sample_id}_layers_1_11_15_18"
    left, right = st.columns(2)
    left.download_button(
        "Download four-panel PDF",
        data=pdf_bytes,
        file_name=f"{stem}.pdf",
        mime="application/pdf",
    )
    right.download_button(
        "Download four-panel PNG",
        data=png_bytes,
        file_name=f"{stem}.png",
        mime="image/png",
    )
    st.caption(
        "Hover over a highlighted token in the browser view to see its prompt position, span, "
        "and incremental objective change."
    )


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
        st.caption("Correctness values are the dataset-specific scores stored in the selected sweep parquet.")


def _softmax_top5_df(top5: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Two-column top-5 table: token + softmax probability."""
    rows = []
    for entry in top5 or []:
        text = str(entry.get("token_text", ""))
        display = text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        if display != "" and display.strip() == "":
            display = f"'{display}'"  # make pure-whitespace tokens visible
        rows.append(
            {
                "token": display,
                "softmax probability": round(float(entry.get("prob", float("nan"))), 4),
            }
        )
    if not rows:
        rows = [{"token": "(unavailable)", "softmax probability": float("nan")}]
    return pd.DataFrame(rows)


def _softmax_run_caption(run: dict[str, Any]) -> str:
    answer = format_answer_with_conf(run.get("answer", ""), run.get("top1_probability"))
    return f"Answer: **{answer}** · score {float(run.get('score', float('nan'))):.3f}"


def _softmax_ground_truth(sample: dict[str, Any]) -> str:
    gt = sample.get("ground_truth")
    if isinstance(gt, (list, tuple)):
        return ", ".join(str(x) for x in gt if str(x).strip())
    return str(gt) if gt is not None else "Not available"


def _softmax_synthetic_row(
    model_label: str,
    dataset: str,
    sample: dict[str, Any],
    layer_entry: dict[str, Any],
    severity: str,
) -> pd.Series:
    """Build a layer-sweep-shaped row so ``show_three_panel_layout`` can render
    a softmax-changes sample with the same panels and patched-token marker.

    ``span_label`` is set to ``final_prompt`` (non-visual) so the patched panel
    text-highlights the final prompt token at ``patched_token_position``,
    exactly like the Layer sweep view.
    """
    clean = sample.get("clean", {})
    corrupted = sample.get("corrupted", {})
    patched = layer_entry.get("patched", {})
    return pd.Series(
        {
            "model": model_label,
            "dataset": dataset,
            "sample_id": str(sample["sample_id"]),
            "severity": severity,
            "span_label": "final_prompt",
            "patched_token_position": int(sample.get("final_prompt_token_position", 0)),
            "clean_answer": clean.get("answer", ""),
            "clean_confidence": clean.get("top1_probability"),
            "degraded_answer": corrupted.get("answer", ""),
            "degraded_confidence": corrupted.get("top1_probability"),
            "patched_answer": patched.get("answer", ""),
            "patched_confidence": patched.get("top1_probability"),
            # P(clean answer) in each run -> "Clean answer confidence" panel line.
            # Present once the softmax-changes JSON is regenerated (schema >= 2).
            "clean_clean_answer_probability": clean.get("clean_answer_probability"),
            "corrupted_clean_answer_probability": corrupted.get("clean_answer_probability"),
            "patched_clean_answer_probability": patched.get("clean_answer_probability"),
        }
    )


def _softmax_shared_sample_ids(dataset: str) -> dict[str, Any]:
    """Cross-model overlap report for a dataset's softmax-changes outputs."""
    ids_by_model = {}
    for model_label, ds in softmax_changes_available():
        if ds != dataset:
            continue
        payload = softmax_changes_data(model_label, ds)
        ids_by_model[model_label] = {str(s["sample_id"]) for s in payload.get("samples", [])}
    shared = set.intersection(*ids_by_model.values()) if len(ids_by_model) > 1 else set()
    return {"ids_by_model": ids_by_model, "shared": sorted(shared, key=lambda x: int(x) if x.isdigit() else x)}


def _softmax_top5_columns(
    container,
    clean_run: dict[str, Any],
    corrupted_run: dict[str, Any],
    patched_run: dict[str, Any],
    selected_layer: int,
) -> None:
    """Render Clean | Corrupted | Patched@L top-5 tables in three columns."""
    cols = container.columns(3)
    for col, title, run in (
        (cols[0], "Clean run", clean_run),
        (cols[1], "Corrupted run", corrupted_run),
        (cols[2], f"Patched run @ layer {selected_layer}", patched_run),
    ):
        col.markdown(f"**{title}**")
        col.markdown(_softmax_run_caption(run))
        col.dataframe(_softmax_top5_df(run.get("top5")), hide_index=True, use_container_width=True)


def show_softmax_changes_view() -> None:
    st.caption(
        "Layer-sweep softmax proof-of-concept. For each retained sample we patch the "
        "**final prompt token** at every LM layer and read the model's output softmax at "
        "the first answer-token step. Clean and corrupted are unpatched baselines (constant "
        "across layers); the patched column changes with the patch layer. Setup matches the "
        "main layer-sweep cohort (clean correct, corrupted incorrect)."
    )

    available = list(softmax_changes_available())
    if not available:
        st.warning(
            "No softmax-changes results found. Generate them with "
            "`python -m src.cli.run_softmax_changes` (see the SLURM jobs under "
            "`scripts/patching/{llava,qwen}/coco-qa-vi/run_softmax_changes.job`)."
        )
        return

    model_labels = sorted({model_label for model_label, _ in available})
    model_label = st.sidebar.selectbox("Model", model_labels, key="softmax_changes_model")
    datasets = [dataset for ml, dataset in available if ml == model_label]
    dataset = st.sidebar.selectbox(
        "Dataset",
        sorted(datasets),
        format_func=lambda x: DATASET_LABELS.get(x, x),
        key="softmax_changes_dataset",
    )

    payload = softmax_changes_data(model_label, dataset)
    samples = payload.get("samples", [])
    meta = payload.get("metadata", {})
    if not samples:
        st.warning(f"The softmax-changes file for {model_label} / {dataset} contains no samples.")
        return

    # ---- cross-model overlap report (samples run for both models) ----
    overlap = _softmax_shared_sample_ids(dataset)
    if len(overlap["ids_by_model"]) > 1:
        shared = overlap["shared"]
        if shared:
            st.info(
                f"Samples run for **both** models on {DATASET_LABELS.get(dataset, dataset)}: "
                f"{len(shared)} — {', '.join(shared)}. Selecting one lets you compare models on "
                "the same sample."
            )
        else:
            st.warning(
                "The two models were run on **different** samples (0 shared). Browsing is "
                "model-specific; cross-model same-sample comparison is not available for the "
                "current outputs. Re-run both models on a common sample set to enable it."
            )

    st.caption(
        f"severity **{meta.get('severity', '?')}** · retained "
        f"**{meta.get('n_retained', len(samples))}/{meta.get('n_requested', '?')}** "
        f"(scanned {meta.get('n_candidates_scanned', '?')}, seed {meta.get('seed', '?')}) · "
        f"top-{meta.get('top_k', 5)} at {meta.get('answer_step', 'first answer token')} · "
        f"{meta.get('num_layers', '?')} layers."
    )

    sample_ids = [str(s["sample_id"]) for s in samples]
    sample_filter = st.sidebar.text_input("Sample id contains", "", key="softmax_changes_filter")
    if sample_filter:
        sample_ids = [sid for sid in sample_ids if sample_filter.lower() in sid.lower()]
    if not sample_ids:
        st.warning("No retained samples match the current filter.")
        return

    selected_id = st.selectbox("Selected sample", sample_ids, key="softmax_changes_sample")
    sample = next(s for s in samples if str(s["sample_id"]) == selected_id)

    layers = sample.get("layers", [])
    if not layers:
        st.info("This sample has no per-layer patched results.")
        return
    layer_indices = [int(layer["layer"]) for layer in layers]
    selected_layer = st.sidebar.selectbox(
        "Patch layer",
        layer_indices,
        index=len(layer_indices) - 1,
        key="softmax_changes_layer",
    )
    layer_entry = next(layer for layer in layers if int(layer["layer"]) == int(selected_layer))

    severity = str(meta.get("severity", "extreme"))
    clean_run = sample.get("clean", {})
    corrupted_run = sample.get("corrupted", {})
    patched_run = layer_entry.get("patched", {})

    # ---- header: current model / sample / layer / run context ----
    st.subheader(
        f"{model_label} / {DATASET_LABELS.get(dataset, dataset)} / sample {selected_id}"
    )
    st.markdown(
        f"**Model:** {model_label} · **Sample:** {selected_id} · **Patch layer:** "
        f"{selected_layer} (final prompt token, offset −1) · **Severity:** {severity}"
    )
    left, right = st.columns([2, 1])
    left.write(sample.get("question", ""))
    right.markdown(f"**Ground truth:** {_softmax_ground_truth(sample)}")

    # ---- three-panel layout (reused from the Layer sweep view) ----
    row = _softmax_synthetic_row(model_label, dataset, sample, layer_entry, severity)
    record = sample_record(model_label, dataset, selected_id)
    token_rows, spans, notes = prompt_tokens_for_row(pd.DataFrame([row.to_dict()]), row)
    if spans:
        row = row.copy()
        for key in ("visual_start", "visual_end", "num_visual_tokens", "visual_geometry"):
            if key in spans:
                row[key] = spans[key]
    show_three_panel_layout(row, record, token_rows, notes)

    # ---- top-5 softmax tables: Clean | Corrupted | Patched@L ----
    st.subheader(f"Top-5 softmax — {model_label} (clean / corrupted / patched @ layer {selected_layer})")
    _softmax_top5_columns(st, clean_run, corrupted_run, patched_run, selected_layer)

    # ---- per-layer patched top-1 scan ----
    st.subheader("Patched top-1 across all layers")
    scan_rows = []
    for layer in layers:
        patched = layer.get("patched", {})
        top5 = patched.get("top5") or []
        top1 = top5[0] if top5 else {}
        scan_rows.append(
            {
                "layer": int(layer["layer"]),
                "patched answer": data_utils_module.stripped_answer(patched.get("answer", "")),
                "score": round(float(patched.get("score", float("nan"))), 3),
                "top-1 token": str(top1.get("token_text", "")).replace("\n", "\\n"),
                "top-1 prob": round(float(top1.get("prob", float("nan"))), 4),
                "answer confidence": round(float(patched.get("top1_probability", float("nan"))), 4),
            }
        )
    st.dataframe(pd.DataFrame(scan_rows), hide_index=True, use_container_width=True, height=320)
    st.caption(f"Outputs are read from the {model_label} softmax-changes JSON.")


def main() -> None:
    st.title("Qualitative activation-patching viewer")
    if st.sidebar.button("Reload data", use_container_width=True):
        clear_app_caches()
        st.rerun()

    st.sidebar.header("Results")
    sweep_type = st.sidebar.radio(
        "Sweep",
        ["Positional sweep", "Layer sweep", "Gradient-guided patching", "Softmax changes"],
        index=2,
        help=(
            "Gradient-guided patching currently supports LLaVA POPE random samples only. "
            "Softmax changes is a layer-sweep proof-of-concept (coco-qa-vi)."
        ),
    )

    if sweep_type == "Gradient-guided patching":
        show_gradient_guided_view()
        return

    if sweep_type == "Softmax changes":
        show_softmax_changes_view()
        return

    raw_rows = load_rows(sweep_type)
    if raw_rows.empty:
        st.error(f"No {sweep_type.lower()} rows could be loaded.")
        return

    st.sidebar.header("Filters")
    models = st.sidebar.multiselect(
        "Model",
        sorted(raw_rows["model"].unique()),
        default=sorted(raw_rows["model"].unique()),
    )
    datasets = st.sidebar.multiselect(
        "Dataset",
        sorted(raw_rows["dataset"].unique()),
        default=sorted(raw_rows["dataset"].unique()),
        format_func=lambda x: DATASET_LABELS.get(x, x),
    )
    if sweep_type == "Layer sweep":
        available_layers = sorted(raw_rows["patch_layer"].dropna().astype(int).unique())
        selected_layers = st.sidebar.multiselect(
            "Layer",
            available_layers,
            default=available_layers,
            format_func=lambda layer: f"Layer {int(layer)}",
        )
        spans = []
    else:
        spans = st.sidebar.multiselect(
            "Patched token span",
            sorted(raw_rows["span_label"].dropna().unique()),
            default=sorted(raw_rows["span_label"].dropna().unique()),
            format_func=lambda x: SPAN_DISPLAY.get(x, x),
        )
        selected_layers = []
    sample_filter = st.sidebar.text_input("Sample id contains", "")
    prompt_filter = st.sidebar.text_input(
        "Prompt contains",
        "",
        help="Case-insensitive phrase search over the sample question.",
    )

    st.sidebar.subheader("Correctness")
    clean_correctness_choice = st.sidebar.selectbox(
        "Clean run", [ANY_CHOICE, "Correct", "Incorrect"], key="clean_correctness_filter"
    )
    corrupted_correctness_choice = st.sidebar.selectbox(
        "Corrupted run", [ANY_CHOICE, "Correct", "Incorrect"], key="corrupted_correctness_filter"
    )
    patched_correctness_choice = st.sidebar.selectbox(
        "Patched run", [ANY_CHOICE, "Correct", "Incorrect"], key="patched_correctness_filter"
    )

    st.sidebar.subheader("Confidence")
    confidence_tolerance = st.sidebar.slider(
        "Unchanged tolerance",
        min_value=0.0,
        max_value=0.20,
        value=0.02,
        step=0.005,
        format="%.3f",
        key="confidence_unchanged_tolerance",
    )
    corruption_direction_choice = st.sidebar.selectbox(
        "Corrupted run",
        [ANY_CHOICE, "Increased", "Decreased", "Unchanged"],
        help="Corrupted confidence relative to the clean run.",
        key="corrupted_confidence_direction_filter",
    )
    patch_direction_choice = st.sidebar.selectbox(
        "Patched run",
        [ANY_CHOICE, "Increased", "Decreased", "Unchanged"],
        help="Patched confidence relative to the corrupted run.",
        key="patched_confidence_direction_filter",
    )

    st.sidebar.subheader("Display")
    sort_options = {
        "Accuracy recovery": "accuracy_recovery",
        "Confidence recovery": "confidence_recovery",
        "Absolute confidence change after patching": "abs_delta_conf_patch",
        "Patched correctness": "patched_correctness",
        "Patched confidence": "patched_confidence",
    }
    sort_label = st.sidebar.selectbox("Sort sample list", list(sort_options.keys()))
    sort_order = st.sidebar.selectbox("Order", ["Highest first", "Lowest first"])
    top_k = st.sidebar.number_input("Show top k rows", min_value=1, max_value=5000, value=200, step=10)
    sample_layout = st.sidebar.radio("Selected sample layout", ["Three-panel", "Original"], index=0)
    if sweep_type == "Positional sweep":
        show_clean_visual_callout = st.sidebar.checkbox(
            "Enlarged clean patch callout",
            value=True,
            help="For visual-token patches, connect the corrupted-image marker to an enlarged crop of the clean region.",
        )
    else:
        show_clean_visual_callout = False

    rows = add_confidence_direction_columns(raw_rows, confidence_tolerance)
    filtered = rows[
        rows["model"].isin(models)
        & rows["dataset"].isin(datasets)
    ]
    if sweep_type == "Layer sweep":
        filtered = filtered[filtered["patch_layer"].isin(selected_layers)]
    else:
        filtered = filtered[filtered["span_label"].isin(spans)]
    if sample_filter:
        filtered = filtered[filtered["sample_id"].astype(str).str.contains(sample_filter, case=False, regex=False)]
    if prompt_filter:
        filtered = filtered[filtered["prompt_text"].str.contains(prompt_filter, case=False, regex=False, na=False)]

    filtered = _apply_correctness_choice(filtered, "clean_correct", clean_correctness_choice)
    filtered = _apply_correctness_choice(filtered, "corrupted_correct", corrupted_correctness_choice)
    filtered = _apply_correctness_choice(filtered, "patched_correct", patched_correctness_choice)

    filtered = _apply_direction_choice(
        filtered,
        "corruption_confidence_direction",
        corruption_direction_choice,
    )
    filtered = _apply_direction_choice(filtered, "patch_confidence_direction", patch_direction_choice)

    sort_metric = sort_options[sort_label]
    filtered = filtered.sort_values(sort_metric, ascending=(sort_order == "Lowest first")).head(int(top_k))

    if sweep_type == "Layer sweep":
        source_caption = (
            "final-prompt-token rows from the exact clean-correct/corrupted-incorrect final layer-plot cohort"
        )
    else:
        source_caption = (
            "plot-included rows from the exact clean-correct/corrupted-incorrect n=35 positional-plot cohort"
        )
    st.caption(f"Loaded {len(rows):,} {source_caption}. Showing {len(filtered):,} rows after filters.")

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    table_cols = [
        "model",
        "dataset",
        "sample_id",
        "prompt_text",
        "span_display",
        "patch_layer_display",
        "patched_token_position",
        "clean_answer_stripped",
        "corrupted_answer_stripped",
        "patched_answer_stripped",
        "clean_correctness_label",
        "corrupted_correctness_label",
        "patched_correctness_label",
        "correctness_pattern",
        "delta_conf_corruption",
        "delta_conf_patch",
        "confidence_transition",
        "accuracy_recovery",
        "confidence_recovery",
    ]
    sample_table = filtered[table_cols].rename(
        columns={
            "prompt_text": "prompt",
            "span_display": "span",
            "patch_layer_display": "patch layer",
            "patched_token_position": "token position",
            "clean_answer_stripped": "clean answer",
            "corrupted_answer_stripped": "corrupted answer",
            "patched_answer_stripped": "patched answer",
            "clean_correctness_label": "clean correctness",
            "corrupted_correctness_label": "corrupted correctness",
            "patched_correctness_label": "patched correctness",
            "delta_conf_corruption": "delta corrupted-clean confidence",
            "delta_conf_patch": "delta patched-corrupted confidence",
        }
    )
    st.dataframe(sample_table, hide_index=True, use_container_width=True, height=300)

    selected_id = st.selectbox("Selected patch", filtered["row_id"].tolist())
    row = filtered[filtered["row_id"] == selected_id].iloc[0]
    record = sample_record(str(row["model"]), str(row["dataset"]), str(row["sample_id"]))

    st.subheader(f"{row['model']} / {DATASET_LABELS.get(row['dataset'], row['dataset'])} / sample {row['sample_id']}")
    if sweep_type == "Layer sweep":
        st.write(
            f"Patched activation: final prompt token position **{int(row['patched_token_position'])}** "
            f"at hidden-state layer **{int(row['patch_layer'])}**."
        )
    else:
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
        show_three_panel_layout(
            row,
            record,
            token_rows,
            notes,
            show_clean_visual_callout=show_clean_visual_callout,
        )
        st.subheader("Export three-panel figure")
        show_three_panel_downloads(
            row,
            record,
            token_rows,
            show_clean_visual_callout=show_clean_visual_callout,
        )
        st.subheader("Ground truth")
        st.write(ground_truth_text(record, str(row["dataset"])))
        st.caption("Correctness values are the dataset-specific scores stored in the selected sweep parquet.")
    else:
        show_original_layout(row, record, token_rows, notes)

    show_top5_softmax_table(row)

    st.subheader("Recovery metrics")
    metric_row = compact_row_dict(row, record)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "accuracy recovery": f"{metric_row['accuracy_recovery']:.3f}",
                    "confidence recovery": f"{metric_row['confidence_recovery']:.3f}",
                    "clean-answer confidence recovery": (
                        _fmt_metric(metric_row["clean_answer_confidence_recovery"])
                    ),
                    "patch layer": str(metric_row["patch_layer"]),
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
