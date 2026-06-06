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
    from . import data_utils as data_utils_module
    from .config import DATASET_LABELS, MODEL_SPECS, OUTPUT_ROOT, SPAN_DISPLAY
    from .data_utils import (
        all_layer_plot_rows,
        all_plot_rows,
        blurred_image_path,
        compact_row_dict,
        ensure_viewer_helper_columns,
        format_answer_with_conf,
        ground_truth_text,
        highlighted_degraded_image,
        highlighted_clean_marker_image,
        prompt_tokens_for_row,
        sample_record,
        token_html,
    )
except ImportError:  # pragma: no cover - useful when running as `streamlit run app.py`.
    import data_utils as data_utils_module  # type: ignore
    from config import DATASET_LABELS, MODEL_SPECS, OUTPUT_ROOT, SPAN_DISPLAY  # type: ignore
    from data_utils import (  # type: ignore
        all_layer_plot_rows,
        all_plot_rows,
        blurred_image_path,
        compact_row_dict,
        ensure_viewer_helper_columns,
        format_answer_with_conf,
        ground_truth_text,
        highlighted_degraded_image,
        highlighted_clean_marker_image,
        prompt_tokens_for_row,
        sample_record,
        token_html,
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
.qv-inset {
    position: absolute;
    right: 8px;
    bottom: 8px;
    width: 34%;
    max-width: 118px;
    min-width: 82px;
    padding: 4px;
    border: 1px solid #f59e0b;
    background: #ffffff;
    box-shadow: 0 4px 12px rgba(17, 24, 39, 0.18);
    box-sizing: border-box;
}
.qv-inset img {
    display: block;
    width: 100%;
    max-height: 105px;
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


def _answer_line(answer: Any, confidence: Any) -> str:
    return f"Answer: {format_answer_with_conf(answer, confidence)}"


def _fmt_prob(value: Any) -> str:
    try:
        numeric = float(value)
        if math.isnan(numeric):
            return "NA"
        return f"{numeric:.2f}"
    except Exception:
        return "NA"


def _fmt_metric(value: Any) -> str:
    try:
        numeric = float(value)
        if math.isnan(numeric):
            return "NA"
        return f"{numeric:.3f}"
    except Exception:
        return "NA"


def _answers_match_for_row(row: pd.Series, left: Any, right: Any) -> bool:
    if str(row.get("dataset", "")) == "pope":
        return data_utils_module.pope_answer_bucket(left) == data_utils_module.pope_answer_bucket(right)
    return data_utils_module.norm_answer(left) == data_utils_module.norm_answer(right)


def _corrupted_answer_text(row: pd.Series) -> str:
    lines = [_answer_line(row["degraded_answer"], row["degraded_confidence"])]
    clean_answer_probability = row.get("corrupted_clean_answer_probability")
    try:
        has_clean_answer_probability = not math.isnan(float(clean_answer_probability))
    except Exception:
        has_clean_answer_probability = False
    if (
        has_clean_answer_probability
        and not _answers_match_for_row(row, row["degraded_answer"], row["clean_answer"])
    ):
        lines.append(
            "Clean answer: "
            f"{data_utils_module.stripped_answer(row['clean_answer'])} "
            f"(P(a_clean)={_fmt_prob(clean_answer_probability)})"
        )
    return "\n".join(lines)


def _patch_activation_label(row: pd.Series) -> str:
    span = SPAN_DISPLAY.get(str(row["span_label"]), str(row["span_label"]))
    return f"{span}, token position {int(row['patched_token_position'])}"


def _panel_html(
    title: str,
    image_source: str | Path | Image.Image | None,
    prompt_html: str,
    answer_text: str,
    fallback: str,
    inset_image_source: str | Path | Image.Image | None = None,
    inset_fallback: str = "Clean image inset unavailable",
) -> str:
    inset_html = ""
    if inset_image_source is not None:
        inset_html = f"<div class='qv-inset'>{_image_html(inset_image_source, inset_fallback)}</div>"
    return (
        "<section class='qv-panel'>"
        f"<div class='qv-panel-title'>{html.escape(title)}</div>"
        f"<div class='qv-image-box'>{_image_html(image_source, fallback)}{inset_html}</div>"
        f"<div class='qv-prompt'>{prompt_html}</div>"
        f"<div class='qv-answer'>{html.escape(answer_text).replace(chr(10), '<br>')}</div>"
        "</section>"
    )


def _selected_run_images(
    row: pd.Series,
    record: dict,
    show_clean_visual_inset: bool = False,
) -> tuple[str, str, str | Image.Image | None, Image.Image | None]:
    clean_path = str(record.get("clean_image_path", ""))
    degraded_path = blurred_image_path(record, str(row["severity"]))
    patched_image: str | Image.Image | None = degraded_path
    clean_marker_inset: Image.Image | None = None

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
        if show_clean_visual_inset:
            clean_marker, _, _ = highlighted_clean_marker_image(
                str(row["model"]),
                clean_path,
                visual_rel,
                visual_token_count,
                row.get("visual_geometry"),
                max_side=900,
            )
            clean_marker_inset = clean_marker
    return clean_path, degraded_path, patched_image, clean_marker_inset


def show_three_panel_layout(
    row: pd.Series,
    record: dict,
    token_rows: list[dict[str, Any]],
    notes: list[str],
    show_clean_visual_inset: bool = False,
) -> None:
    patched_pos = int(row["patched_token_position"])
    span_label = str(row["span_label"])
    clean_path, degraded_path, patched_image, clean_marker_inset = _selected_run_images(
        row,
        record,
        show_clean_visual_inset=show_clean_visual_inset,
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
            _answer_line(row["clean_answer"], row["clean_confidence"]),
            "Clean image unavailable",
        ),
        _panel_html(
            "(2) Corrupted Run",
            degraded_path,
            prompt_html,
            _corrupted_answer_text(row),
            "Corrupted image unavailable",
        ),
        _panel_html(
            "(3) Patched Run",
            patched_image,
            patched_prompt_html,
            _answer_line(row["patched_answer"], row["patched_confidence"]),
            "Patched image unavailable",
            inset_image_source=clean_marker_inset,
            inset_fallback="Clean image with patched visual-token marker unavailable",
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
    inset_image_source: str | Path | Image.Image | None = None,
) -> None:
    body_font, title_font = _export_font_properties()
    answer_lines = str(answer).splitlines() or [""]
    prompt_size = 11 if line_count <= 9 else max(7.0, 11 - 0.35 * (line_count - 9))
    prompt_top = 0.48
    axis_height_inches = max(1.0, ax.figure.get_figheight() * 0.90)
    line_step = (prompt_size / 72.0 / axis_height_inches) * 1.28
    prompt_bottom = prompt_top - line_step * max(1, line_count)
    divider_gap = min(0.048, line_step * 1.9)
    divider_y = max(0.105, prompt_bottom - divider_gap)
    answer_y = divider_y - min(0.030, line_step * 1.1)
    answer_line_step = (12 / 72.0 / axis_height_inches) * 1.25
    content_bottom = max(0.035, answer_y - answer_line_step * len(answer_lines) - 0.025)

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

    inset_image = _load_export_image(inset_image_source)
    if inset_image is not None:
        inset_ax = ax.inset_axes([0.64, 0.555, 0.23, 0.15])
        inset_ax.set_facecolor("white")
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.1)
            spine.set_edgecolor("#f59e0b")
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.imshow(inset_image)

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
    show_clean_visual_inset: bool = False,
) -> bytes:
    patched_pos = int(row["patched_token_position"])
    span_label = str(row["span_label"])
    clean_path, degraded_path, patched_image, clean_marker_inset = _selected_run_images(
        row,
        record,
        show_clean_visual_inset=show_clean_visual_inset,
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
        ("(1) Clean Run", clean_path, prompt_segments, _answer_line(row["clean_answer"], row["clean_confidence"]), None),
        (
            "(2) Corrupted Run",
            degraded_path,
            prompt_segments,
            _corrupted_answer_text(row),
            None,
        ),
        (
            "(3) Patched Run",
            patched_image,
            patched_prompt_segments,
            _answer_line(row["patched_answer"], row["patched_confidence"]),
            clean_marker_inset,
        ),
    ]
    for ax, (title, image_source, prompt_segments_for_panel, answer_text, inset_image_source) in zip(axes, panels):
        line_count = len(_wrap_segments_for_export(prompt_segments_for_panel))
        _draw_export_panel(
            ax,
            title,
            image_source,
            prompt_segments_for_panel,
            answer_text,
            line_count,
            inset_image_source=inset_image_source,
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
    show_clean_visual_inset: bool = False,
) -> None:
    stem = _safe_export_stem(row)
    pdf_bytes = render_three_panel_export(
        row,
        record,
        token_rows,
        "pdf",
        show_clean_visual_inset=show_clean_visual_inset,
    )
    png_bytes = render_three_panel_export(
        row,
        record,
        token_rows,
        "png",
        show_clean_visual_inset=show_clean_visual_inset,
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


def main() -> None:
    st.title("Qualitative activation-patching viewer")
    if st.sidebar.button("Reload data", use_container_width=True):
        clear_app_caches()
        st.rerun()

    st.sidebar.header("Results")
    sweep_type = st.sidebar.radio(
        "Sweep",
        ["Positional sweep", "Layer sweep"],
        help=(
            "Layer sweep loads the parquets used for "
            "two_metrics_layer_sweep_nofilter and keeps position_offset == -1."
        ),
    )

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
        show_clean_visual_inset = st.sidebar.checkbox(
            "Clean inset for visual patches",
            value=True,
            help="In the patched panel, show a small clean-image inset with the visual-token marker.",
        )
    else:
        show_clean_visual_inset = False

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
            "final-prompt-token patched rows from the exact layer-sweep parquets used for "
            "two_metrics_layer_sweep_nofilter"
        )
    else:
        source_caption = "plot-included patched rows from the exact positional-sweep parquets"
    st.caption(f"Loaded {len(rows):,} {source_caption}. Showing {len(filtered):,} rows after filters.")

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    table_cols = [
        "model",
        "dataset",
        "sample_id",
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
            show_clean_visual_inset=show_clean_visual_inset,
        )
        st.subheader("Export three-panel figure")
        show_three_panel_downloads(
            row,
            record,
            token_rows,
            show_clean_visual_inset=show_clean_visual_inset,
        )
        st.subheader("Ground truth")
        st.write(ground_truth_text(record, str(row["dataset"])))
        st.caption("Correctness values are the dataset-specific scores stored in the selected sweep parquet.")
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
