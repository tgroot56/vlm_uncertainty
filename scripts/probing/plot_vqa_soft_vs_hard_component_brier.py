#!/usr/bin/env python
"""Plot VQA-v2 soft-score probes against hard exact-score probes.

The hard condition uses probes trained with --binarize_one_score_labels, where
only original VQA scores of 1.0 are positive and all partial-credit scores are
negative.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("probe_results")
OUT_DIR = ROOT / "_plots" / "vqa_soft_vs_hard_exact"

FEATURES: Dict[str, List[tuple[str, str]]] = {
    "llava": [
        ("Vision\n(mean)", "vision_mean_layer_12"),
        ("LM Visual\n(mean)", "lm_visual_mean_layer_16"),
        ("LM Question\n(lasttok)", "lm_question_lasttok_layer_16"),
        ("LM Final Token\n(lasttok)", "lm_fullspan_lasttok_layer_16"),
        ("Entropy\n(mean)", "answer_gen_entropy_mean"),
        ("NegLogP\n(mean)", "answer_gen_neglogp_mean"),
    ],
    "qwen": [
        ("Vision\n(mean)", "vision_mean_layer_13"),
        ("LM Visual\n(mean)", "lm_visual_mean_layer_15"),
        ("LM Question\n(lasttok)", "lm_question_lasttok_layer_16"),
        ("LM Final Token\n(lasttok)", "lm_fullspan_lasttok_layer_16"),
        ("Entropy\n(mean)", "answer_gen_entropy_mean"),
        ("NegLogP\n(mean)", "answer_gen_neglogp_mean"),
    ],
}

MODEL_TITLES = {
    "llava": "LLaVA",
    "qwen": "Qwen",
}


def _brier_from_summary(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    pred = np.array([float(r["prediction"]) for r in rows], dtype=float)
    label = np.array([float(r["label"]) for r in rows], dtype=float)
    return float(np.mean((pred - label) ** 2))


def _run_dir(model: str, target: str, variant: str, feature: str) -> Path:
    if target == "soft" and variant == "trained":
        return ROOT / model / "vqa-v2" / "mlp" / feature
    if target == "soft" and variant == "shuffle":
        return ROOT / model / "vqa-v2" / "baselines" / "mlp" / feature
    if target == "hard" and variant == "trained":
        return ROOT / model / "vqa-v2-hard-exact" / "mlp" / feature
    if target == "hard" and variant == "shuffle":
        return ROOT / model / "vqa-v2-hard-exact" / "shuffle_train_labels" / "mlp" / feature
    raise ValueError(f"Unknown target={target!r}, variant={variant!r}")


def collect_rows() -> pd.DataFrame:
    rows = []
    for model, features in FEATURES.items():
        for label, feature in features:
            for target in ("soft", "hard"):
                for variant in ("trained", "shuffle"):
                    run_dir = _run_dir(model, target, variant, feature)
                    brier = _brier_from_summary(run_dir / "summary_predictions.csv")
                    rows.append(
                        {
                            "model": model,
                            "target": target,
                            "variant": variant,
                            "component": label.replace("\n", " "),
                            "feature": feature,
                            "brier": brier,
                            "path": str(run_dir),
                        }
                    )
    return pd.DataFrame(rows)


def _values_for(
    df: pd.DataFrame,
    *,
    model: str,
    target: str,
    variant: str,
    features: Iterable[tuple[str, str]],
) -> List[float]:
    vals: List[float] = []
    for _, feature in features:
        match = df[
            (df["model"] == model)
            & (df["target"] == target)
            & (df["variant"] == variant)
            & (df["feature"] == feature)
        ]
        val = match["brier"].iloc[0]
        vals.append(float(val) if pd.notna(val) else np.nan)
    return vals


def plot(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    targets = [("soft", "Original VQA soft scores"), ("hard", "Binarized exact-style scores")]
    models = ["llava", "qwen"]

    for row_idx, (target, target_title) in enumerate(targets):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            features = FEATURES[model]
            labels = [label for label, _ in features]
            x = np.arange(len(features))
            trained = _values_for(df, model=model, target=target, variant="trained", features=features)
            shuffle = _values_for(df, model=model, target=target, variant="shuffle", features=features)

            width = 0.38
            ax.bar(x - width / 2, trained, width=width, color="#3b82f6", label="Trained labels")
            ax.bar(
                x + width / 2,
                shuffle,
                width=width,
                color="#cbd5e1",
                edgecolor="#64748b",
                hatch="//",
                label="Shuffled train labels",
            )

            if np.all(np.isnan(trained)) and np.all(np.isnan(shuffle)):
                ax.text(0.5, 0.5, "Missing hard-label runs", ha="center", va="center", transform=ax.transAxes)

            ax.set_title(f"{MODEL_TITLES[model]} - {target_title}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel("Brier score")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("VQA-v2 Probe Brier Scores: Soft Accuracy vs Exact-Style Hard Labels", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = OUT_DIR / "vqa_soft_vs_hard_component_brier.png"
    pdf_path = OUT_DIR / "vqa_soft_vs_hard_component_brier.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    df = collect_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "vqa_soft_vs_hard_component_brier.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    missing = df[df["brier"].isna()]
    if not missing.empty:
        print("\nMissing runs:")
        print(missing[["model", "target", "variant", "feature", "path"]].to_string(index=False))

    plot(df)


if __name__ == "__main__":
    main()
