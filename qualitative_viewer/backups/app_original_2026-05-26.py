from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

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
        f"Patched span: **{SPAN_DISPLAY.get(row['span_label'], row['span_label'])}**, "
        f"token position **{int(row['patched_token_position'])}** "
        f"(span-relative {int(row['span_rel_pos'])})."
    )

    token_rows, spans, notes = prompt_tokens_for_row(rows, row)
    if spans:
        row = row.copy()
        for key in ("visual_start", "visual_end", "num_visual_tokens", "visual_geometry"):
            if key in spans:
                row[key] = spans[key]

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
