#!/usr/bin/env python
"""Plot JS divergence for VQA-v2 layer-sweep answer-token softmaxes.

This script expects a layer-sweep parquet produced with
``--save_stripped_answer_softmax``. It uses only rows whose stripped generated
answer consists of a single token. For each layer, it compares:

  - clean vs corrupted/degraded baseline
  - clean vs patched at that layer

The two means are computed on the same per-layer subset: samples where clean,
corrupted, and patched distributions are all single-token. This makes the two
curves directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path(
    "/projects/prjs2014/patching/layer_sweep_results/"
    "llava-hf_llava-1-5-7b-hf/vqa-v2/"
    "extreme_clean_answer_confidence_stripped_softmax"
)
EPSILON = 1e-12


def _parse_shape(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(v) for v in json.loads(value)]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise ValueError(f"Cannot parse softmax shape from {value!r}")


def _softmax_vector(row: pd.Series) -> np.ndarray | None:
    """Return the single stripped-answer softmax vector for one row."""
    if int(row.get("stripped_answer_token_count", 0)) != 1:
        return None

    shape = _parse_shape(row["stripped_answer_softmax_shape"])
    if len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
        return None

    dtype_name = str(row["stripped_answer_softmax_dtype"]).lower()
    if dtype_name == "float16":
        dtype = np.float16
    elif dtype_name == "float32":
        dtype = np.float32
    else:
        raise ValueError(f"Unsupported softmax dtype: {dtype_name}")

    payload = row["stripped_answer_softmax_bytes"]
    if payload is None:
        return None
    arr = np.frombuffer(payload, dtype=dtype)
    expected = shape[0] * shape[1]
    if arr.size != expected:
        raise ValueError(
            f"Softmax payload has {arr.size} values, expected {expected} "
            f"from shape {shape}."
        )
    return arr.reshape(shape)[0].astype(np.float64, copy=False)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = EPSILON) -> float:
    """Jensen-Shannon divergence in nats."""
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {q.shape}")

    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    m = 0.5 * (p_safe + q_safe)

    kl_pm = np.sum(p_safe * np.log(p_safe / np.clip(m, eps, None)))
    kl_qm = np.sum(q_safe * np.log(q_safe / np.clip(m, eps, None)))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _load_layer_sweep(parquet_path: Path) -> pd.DataFrame:
    columns = [
        "sample_id",
        "severity",
        "condition",
        "layer",
        "position_offset",
        "span_label",
        "stripped_answer_token_count",
        "stripped_answer_softmax_shape",
        "stripped_answer_softmax_dtype",
        "stripped_answer_softmax_bytes",
    ]
    return pd.read_parquet(parquet_path, columns=columns)


def compute_layer_js(
    df: pd.DataFrame,
    *,
    position_offset: int = -1,
) -> pd.DataFrame:
    clean = (
        df[df["condition"] == "clean"]
        .drop_duplicates(["sample_id", "severity"])
        .set_index(["sample_id", "severity"])
    )
    degraded = (
        df[df["condition"] == "degraded"]
        .drop_duplicates(["sample_id", "severity"])
        .set_index(["sample_id", "severity"])
    )
    patched = df[
        (df["condition"] == "patched")
        & (df["position_offset"] == position_offset)
    ].copy()

    layers = sorted(patched["layer"].dropna().astype(int).unique().tolist())
    rows: list[dict[str, Any]] = []
    clean_cache: dict[tuple[Any, Any], np.ndarray | None] = {}
    degraded_cache: dict[tuple[Any, Any], np.ndarray | None] = {}

    for layer in layers:
        js_clean_degraded: list[float] = []
        js_clean_patched: list[float] = []
        sub = patched[patched["layer"].astype(int) == layer]

        for _, patched_row in sub.iterrows():
            key = (patched_row["sample_id"], patched_row["severity"])
            if key not in clean.index or key not in degraded.index:
                continue

            if key not in clean_cache:
                clean_cache[key] = _softmax_vector(clean.loc[key])
            if key not in degraded_cache:
                degraded_cache[key] = _softmax_vector(degraded.loc[key])

            clean_vec = clean_cache[key]
            degraded_vec = degraded_cache[key]
            patched_vec = _softmax_vector(patched_row)
            if clean_vec is None or degraded_vec is None or patched_vec is None:
                continue

            js_clean_degraded.append(js_divergence(clean_vec, degraded_vec))
            js_clean_patched.append(js_divergence(clean_vec, patched_vec))

        rows.append(
            {
                "layer": int(layer),
                "display_layer": int(layer) + 1,
                "mean_js_clean_corrupted": (
                    float(np.mean(js_clean_degraded))
                    if js_clean_degraded else float("nan")
                ),
                "mean_js_clean_patched": (
                    float(np.mean(js_clean_patched))
                    if js_clean_patched else float("nan")
                ),
                "n_single_token_examples": int(len(js_clean_patched)),
                "position_offset": int(position_offset),
            }
        )

    return pd.DataFrame(rows)


def plot_layer_js(summary: pd.DataFrame, output_dir: Path, output_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5))

    ax.plot(
        summary["display_layer"],
        summary["mean_js_clean_corrupted"],
        "o-",
        color="#dc2626",
        linewidth=2,
        markersize=5,
        label="JS(clean, corrupted)",
    )
    ax.plot(
        summary["display_layer"],
        summary["mean_js_clean_patched"],
        "o-",
        color="#2563eb",
        linewidth=2,
        markersize=5,
        label="JS(clean, patched)",
    )

    ax.set_title(
        "VQA-v2 Layer Sweep - Single-Token Answer Softmax JS",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Mean JS divergence (nats)", fontsize=13)
    ax.set_xticks(summary["display_layer"].tolist())
    ax.set_xticklabels([str(v) for v in summary["display_layer"]], fontsize=8, rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=10)

    n_min = int(summary["n_single_token_examples"].min()) if not summary.empty else 0
    n_max = int(summary["n_single_token_examples"].max()) if not summary.empty else 0
    fig.suptitle(
        f"Single-token clean/corrupted/patched examples per layer: {n_min}-{n_max}",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext, kwargs in [("pdf", {}), ("png", {"dpi": 200})]:
        fig.savefig(output_dir / f"{output_name}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"Saved {output_dir / (output_name + '.pdf')} and .png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "layer_sweep_results.parquet",
        help="Layer-sweep parquet with stripped-answer softmax payloads.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for the plot and per-layer CSV. Defaults to parquet parent.",
    )
    parser.add_argument(
        "--output_name",
        default="vqa_layer_sweep_js_divergence_single_token",
        help="Base filename for PDF/PNG/CSV outputs.",
    )
    parser.add_argument(
        "--position_offset",
        type=int,
        default=-1,
        help="Patched prompt position offset to use; -1 matches standard layer plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.parquet.parent
    df = _load_layer_sweep(args.parquet)
    summary = compute_layer_js(df, position_offset=args.position_offset)

    csv_path = output_dir / f"{args.output_name}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    plot_layer_js(summary, output_dir, args.output_name)


if __name__ == "__main__":
    main()
