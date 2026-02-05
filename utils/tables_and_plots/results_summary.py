# results_summary.py
"""
Collect probe results from:
  probe_results/<model_type>/<feature_names_concat>/metrics_test.json

and write:
  - results_auroc.csv
  - results_ece.csv

Each CSV has:
  rows    = feature name (single feature or concatenated)
  columns = ["linear", "mlp"]

Sorted by feature group (vision / lm_visual / lm_prompt / lm_answer / answer_gen).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

METRICS_FILENAMES = [
    "metrics_test.json",    # older variant you used earlier
]

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _find_metrics_file(run_dir: Path) -> Optional[Path]:
    for name in METRICS_FILENAMES:
        p = run_dir / name
        if p.exists():
            return p
    return None

def _infer_feature_names(run_dir: Path) -> List[str]:
    """
    Prefer config.json if present (robust when directory name is hashed/truncated),
    otherwise infer from folder name split by '__'.
    """
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            j = _load_json(cfg)
            feats = j.get("feature_names", None)
            if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
                return feats
        except Exception:
            pass

    # Fallback: directory name convention
    # feature_names_concat uses "__" join
    name = run_dir.name
    parts = [p for p in name.split("__") if p]
    return parts if parts else [name]

def _feature_group(feat: str) -> str:
    # Order matters (more specific first)
    if feat.startswith("vision_"):
        return "vision"
    if feat.startswith("lm_visual_"):
        return "lm_visual"
    if feat.startswith("lm_prompt_"):
        return "lm_prompt"
    if feat.startswith("lm_answer_"):
        return "lm_answer"
    if feat.startswith("answer_gen_"):
        return "answer_gen"
    return "other"

def _row_label(feature_names: List[str]) -> str:
    # For single-feature runs, this is the feature name.
    # For multi-feature runs, keep a readable join.
    if len(feature_names) == 1:
        return feature_names[0]
    return " + ".join(feature_names)

def _sort_key(label: str) -> Tuple[int, str]:
    """
    Sort by group, then label.
    """
    group_order = {
        "vision": 0,
        "lm_visual": 1,
        "lm_prompt": 2,
        "lm_answer": 3,
        "answer_gen": 4,
        "other": 5,
    }
    g = _feature_group(label.split(" + ", 1)[0])
    return (group_order.get(g, 99), label)


# ----------------------------
# Main collection
# ----------------------------

def collect_results(root: Path) -> pd.DataFrame:
    """
    Returns a long dataframe with columns:
      feature_label, group, model_type, test_auroc, test_ece, test_loss, best_epoch, best_val_loss, run_dir
    """
    rows = []

    for model_type_dir in root.iterdir():
        if not model_type_dir.is_dir():
            continue
        model_type = model_type_dir.name
        if model_type not in {"linear", "mlp"}:
            # ignore anything else
            continue

        for run_dir in model_type_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metrics_path = _find_metrics_file(run_dir)
            if metrics_path is None:
                continue

            try:
                metrics = _load_json(metrics_path)
            except Exception:
                continue

            feature_names = _infer_feature_names(run_dir)
            label = _row_label(feature_names)
            group = _feature_group(feature_names[0]) if feature_names else "other"

            rows.append(
                {
                    "feature_label": label,
                    "group": group,
                    "model_type": model_type,
                    "test_auroc": metrics.get("test_auroc", None),
                    "test_ece": metrics.get("test_ece", None),
                    "test_loss": metrics.get("test_loss", None),
                    "best_epoch": metrics.get("best_epoch", None),
                    "best_val_loss": metrics.get("best_val_loss", None),
                    "run_dir": str(run_dir),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_label","group","model_type",
                "test_auroc","test_ece","test_loss","best_epoch","best_val_loss","run_dir"
            ]
        )

    df = pd.DataFrame(rows)
    return df


def pivot_metric(df_long: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Create wide table: rows=feature_label, cols=[linear, mlp], values=metric_col
    """
    wide = (
        df_long.pivot_table(
            index="feature_label",
            columns="model_type",
            values=metric_col,
            aggfunc="mean",  # if duplicates exist, average them
        )
        .reset_index()
    )

    # Ensure columns exist
    for c in ["linear", "mlp"]:
        if c not in wide.columns:
            wide[c] = pd.NA

    # Sort by group then label
    wide = wide.sort_values("feature_label", key=lambda s: s.map(lambda x: _sort_key(str(x))))
    wide = wide.set_index("feature_label")

    # Column order
    wide = wide[["linear", "mlp"]]
    return wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="probe_results",
        help="Root folder containing probe_results/<model_type>/<run_dir>/metrics_test.json",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Where to write CSVs (default: same as --root).",
    )
    ap.add_argument(
        "--save_long",
        action="store_true",
        help="Also save a long-form CSV with all metrics and run_dir paths.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir) if args.outdir is not None else root
    outdir.mkdir(parents=True, exist_ok=True)

    df_long = collect_results(root)

    if df_long.empty:
        print(f"No metrics found under: {root}")
        return

    # Two separate tables, as requested
    df_auroc = pivot_metric(df_long, "test_auroc")
    df_ece = pivot_metric(df_long, "test_ece")
    df_brier = pivot_metric(df_long, "test_loss")  # if you want to include loss as well

    auroc_path = outdir / "results_auroc.csv"
    ece_path = outdir / "results_ece.csv"
    brier_path = outdir / "results_brier.csv"

    df_auroc.to_csv(auroc_path)
    df_ece.to_csv(ece_path)
    df_brier.to_csv(brier_path)

    print(f"Wrote: {auroc_path}")
    print(f"Wrote: {ece_path}")
    print(f"Wrote: {brier_path}")

    if args.save_long:
        long_path = outdir / "results_long.csv"
        df_long.sort_values(["group", "feature_label", "model_type"]).to_csv(long_path, index=False)
        print(f"Wrote: {long_path}")


if __name__ == "__main__":
    main()
