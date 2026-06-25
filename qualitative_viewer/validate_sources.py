from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

try:
    from .config import DATASET_LABELS, DATASETS, FINAL_POSITIONAL_PLOT, MIN_POS_SAMPLES, MODEL_SPECS, OUTPUT_ROOT, all_parquet_paths
    from .data_utils import aggregate_recovery_ci, all_plot_rows, raw_parquet
except ImportError:  # pragma: no cover
    from config import DATASET_LABELS, DATASETS, FINAL_POSITIONAL_PLOT, MIN_POS_SAMPLES, MODEL_SPECS, OUTPUT_ROOT, all_parquet_paths  # type: ignore
    from data_utils import aggregate_recovery_ci, all_plot_rows, raw_parquet  # type: ignore


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown_table(df: pd.DataFrame) -> str:
    text_df = df.fillna("").astype(str)
    headers = list(text_df.columns)
    rows = text_df.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"

    out = [fmt(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    out.extend(fmt([str(value) for value in row]) for row in rows)
    return "\n".join(out)


def aggregate_values(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    group_cols = ["model", "dataset", "severity", "span_label", "span_rel_pos", "canonical_x"]
    for keys, sub in rows.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        rec["n_samples"] = int(sub["sample_id"].nunique())
        confidence_sub = pd.DataFrame(
            {
                "top1_probability": sub["patched_plot_confidence"],
                "clean_top1_probability": sub["clean_plot_confidence"],
                "degraded_top1_probability": sub["degraded_plot_confidence"],
            }
        )
        accuracy_sub = pd.DataFrame(
            {
                "score": sub["patched_correctness"],
                "clean_score": sub["clean_correctness"],
                "degraded_score": sub["degraded_correctness"],
            }
        )
        for metric, label, metric_sub in [
            ("top1_probability", "confidence", confidence_sub),
            ("score", "accuracy", accuracy_sub),
        ]:
            mean, lo, hi = aggregate_recovery_ci(metric_sub, metric)
            rec[f"{label}_aggregate_recovery"] = mean
            rec[f"{label}_ci_low"] = lo
            rec[f"{label}_ci_high"] = hi
        out.append(rec)
    return pd.DataFrame(out).sort_values(["model", "dataset", "canonical_x"])


def build_report(out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot = all_plot_rows()
    loaded = []
    for (model_label, dataset), path in all_parquet_paths().items():
        exists = path.exists()
        if not exists:
            loaded.append(
                {
                    "model": model_label,
                    "dataset": dataset,
                    "parquet": str(path),
                    "exists": False,
                    "sha256": "",
                    "rows": 0,
                    "clean_samples": 0,
                    "plot_rows": 0,
                    "plot_samples": 0,
                    "spans": "",
                    "columns": "",
                }
            )
            continue
        raw = raw_parquet(model_label, dataset)
        plot_sub = plot[(plot["model"] == model_label) & (plot["dataset"] == dataset)]
        loaded.append(
            {
                "model": model_label,
                "dataset": dataset,
                "parquet": str(path),
                "exists": True,
                "sha256": sha256_file(path),
                "rows": int(len(raw)),
                "clean_samples": int(raw[raw["condition"] == "clean"]["sample_id"].nunique()),
                "plot_rows": int(len(plot_sub)),
                "plot_samples": int(plot_sub["sample_id"].nunique()),
                "spans": ", ".join(sorted(str(x) for x in raw["span_label"].dropna().unique())),
                "columns": ", ".join(raw.columns),
            }
        )
    summary = pd.DataFrame(loaded)
    summary_path = out_dir / "loaded_parquets_summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = aggregate_values(plot)
    aggregate_path = out_dir / "aggregate_recovery_values.csv"
    aggregate.to_csv(aggregate_path, index=False)

    report_path = out_dir / "validation_report.md"
    with report_path.open("w") as f:
        f.write("# Qualitative viewer validation report\n\n")
        f.write(f"Target plot: `{FINAL_POSITIONAL_PLOT}`\n\n")
        f.write(
            "The viewer uses the same eight positional-sweep parquet families and applies "
            "the clean-correct/corrupted-incorrect cohort filter and "
            f"`min_pos_samples={MIN_POS_SAMPLES}` preparation used by "
            "`scripts/plot_filtered_clean_answer_cross_model_sweeps.py`.\n\n"
        )
        f.write("## Loaded parquet files\n\n")
        f.write(markdown_table(summary[["model", "dataset", "exists", "clean_samples", "plot_samples", "plot_rows", "parquet"]]))
        f.write("\n\n")
        f.write("## Available token spans\n\n")
        for _, row in summary.iterrows():
            f.write(f"- {row['model']} / {DATASET_LABELS.get(row['dataset'], row['dataset'])}: {row['spans']}\n")
        f.write("\n## Available columns\n\n")
        for _, row in summary.iterrows():
            f.write(f"- {row['model']} / {row['dataset']}: `{row['columns']}`\n")
        f.write("\n## Aggregate recovery validation\n\n")
        f.write(
            "Aggregate confidence and accuracy recovery values were recomputed with "
            "`src.cli.plot_sweep_filtered._prepare_positional_sweep` and "
            "`aggregate_recovery_ci`, the same functions used by the target plot. "
            "The numerical values are saved in `aggregate_recovery_values.csv`. "
            "No separate numeric dump from the original PDF was found, so the match is by exact source pipeline equivalence rather than by reading values back from the rendered PDF.\n\n"
        )
        max_counts = (
            aggregate.groupby(["model", "dataset"])["n_samples"]
            .agg(["min", "max"])
            .reset_index()
            .rename(columns={"min": "min_samples_per_position", "max": "max_samples_per_position"})
        )
        f.write(markdown_table(max_counts))
        f.write("\n\n")
        f.write("## Known limitations logged for requested qualitative features\n\n")
        f.write("- The parquet stores prompt token positions and span labels, but not prompt token strings. The app reconstructs exact token strings from cached Hugging Face processors when available and otherwise displays an explicit fallback warning.\n")
        f.write("- Visual-token highlights use model-specific geometry reconstructed from cached Hugging Face processors. If exact geometry cannot be recovered, the app does not draw a pixel highlight.\n")
        f.write("- Displayed confidence uses `answer_geom_mean_probability` when present. Recovery values use `top1_probability`, matching the positional-sweep plotting pipeline; in the inspected parquets these columns carry the same answer-level geometric confidence values.\n")
    return report_path, summary_path, aggregate_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "validation")
    args = parser.parse_args()
    report_path, summary_path, aggregate_path = build_report(args.output_dir)
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
