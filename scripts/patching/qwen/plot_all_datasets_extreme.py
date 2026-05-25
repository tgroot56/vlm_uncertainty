"""Combined Qwen extreme-severity layer + positional sweep plots across all
four datasets. Writes the same figure style as LLaVA's
/projects/prjs2014/patching/all_datasets/extreme/ — raw values, no filter.

Usage:
    python scripts/patching/qwen/plot_all_datasets_extreme.py
"""
from __future__ import annotations

from pathlib import Path

from src.cli.plot_sweep_filtered import (
    plot_all_datasets_layer_sweep,
    plot_all_datasets_positional_sweep,
    plot_layer_sweep_combined,
    plot_positional_sweep_combined,
)

VLM_SHORT = "Qwen_Qwen3-5-9B"
DATASETS = ["vqa-v2", "pope", "coco-qa-vi", "imagenet-r"]
SEVERITY = "extreme"

STORAGE_ROOT = Path("/projects/prjs2014/patching")
LAYER_ROOT = STORAGE_ROOT / "layer_sweep_results" / VLM_SHORT
POS_ROOT = STORAGE_ROOT / "positional_patching_results_including_visual" / VLM_SHORT
OUT_ROOT = STORAGE_ROOT / "qwen" / "all_datasets" / SEVERITY


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    layer_parquets: list[tuple[str, Path]] = []
    pos_parquets: list[tuple[str, Path]] = []
    for ds in DATASETS:
        layer_parquets.append(
            (ds, LAYER_ROOT / ds / SEVERITY / "layer_sweep_results.parquet")
        )
        pos_parquets.append(
            (ds, POS_ROOT / ds / SEVERITY / "positional_patching_results.parquet")
        )

    # Per-dataset plots go next to each parquet under its /extreme dir.
    for ds, parquet in layer_parquets:
        if parquet.exists():
            print(f"\n[layer] {ds}: {parquet}")
            plot_layer_sweep_combined(
                str(parquet), parquet.parent, raw=True, min_delta=0.0
            )
        else:
            print(f"[layer] {ds}: SKIP (missing {parquet})")

    for ds, parquet in pos_parquets:
        if parquet.exists():
            print(f"\n[pos]   {ds}: {parquet}")
            plot_positional_sweep_combined(
                str(parquet), parquet.parent, raw=True, min_delta=0.0
            )
        else:
            print(f"[pos]   {ds}: SKIP (missing {parquet})")

    print(f"\n=== All-datasets layer sweep (extreme, raw, nofilter) ===")
    plot_all_datasets_layer_sweep(
        [(ds, str(p)) for ds, p in layer_parquets if p.exists()],
        OUT_ROOT, raw=True, min_delta=0.0,
        title="Qwen3.5-9B — Final Prompt Token Activation Patching: Layer Sweep (σ=50)",
    )

    print(f"=== All-datasets positional sweep (extreme, raw, nofilter) ===")
    plot_all_datasets_positional_sweep(
        [(ds, str(p)) for ds, p in pos_parquets if p.exists()],
        OUT_ROOT, raw=True, min_delta=0.0,
        title="Qwen3.5-9B — Final Prompt Token Activation Patching: Positional Sweep (σ=50)",
    )

    print(f"\nWrote combined figures to {OUT_ROOT}")


if __name__ == "__main__":
    main()
