"""
CLI entry point for generating patching analysis plots.

Reads ``patching_results.parquet`` produced by ``src/patching/run_patching.py``
and generates Figures 1-6.

Usage::

    # Generate all figures from an existing results file:
    python -m src.cli.plot_patching_results \\
        --results_parquet /projects/prjs2014/patching/patching_results/\\
                          llava-hf_llava-1-5-7b-hf/vqa-v2/patching_results.parquet \\
        --output_dir /projects/prjs2014/patching/figures/llava/vqa-v2/

    # Limit bootstrap iterations for faster iteration:
    python -m src.cli.plot_patching_results \\
        --results_parquet <path> \\
        --output_dir <outdir> \\
        --n_bootstrap 200

    # Only run specific figures:
    python -m src.cli.plot_patching_results \\
        --results_parquet <path> \\
        --output_dir <outdir> \\
        --figures 1 4 5

    # Pass the peak probing layer for Figure 5 annotation:
    python -m src.cli.plot_patching_results \\
        --results_parquet <path> \\
        --output_dir <outdir> \\
        --probe_peak_layer 24
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figures 1-6 from activation patching results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_parquet", type=str, required=True,
        help="Path to patching_results.parquet from run_patching.py",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for figures. Defaults to <results_dir>/figures/",
    )
    parser.add_argument(
        "--figures", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
        choices=[1, 2, 3, 4, 5, 6],
        metavar="N",
        help="Which figures to generate (default: 1 2 3 4 5 6)",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=1000,
        help="Bootstrap iterations for CI (Figure 3, default 1000). Use 200 for fast iteration.",
    )
    parser.add_argument(
        "--severity_fig4", type=str, default="moderate",
        choices=["mild", "moderate", "severe"],
        help="Which severity to use for Figure 4 dissociation plot (default: moderate)",
    )
    parser.add_argument(
        "--probe_peak_layer", type=int, default=None,
        help="Layer index where probing found peak signal; shown as dashed line in Figure 5",
    )
    parser.add_argument(
        "--metric", type=str, default="output_entropy",
        choices=["output_entropy", "top1_probability", "score"],
        help="Primary recovery metric for Figures 1, 3, 5 (default: output_entropy)",
    )
    parser.add_argument(
        "--fig5_span", type=str, default="lm_fullspan_lasttok",
        help="Span to sweep over layers in Figure 5 (default: lm_fullspan_lasttok)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parquet_path = args.results_parquet
    if not os.path.exists(parquet_path):
        print(f"ERROR: results file not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = str(Path(parquet_path).parent / "figures")

    from src.patching.plot_patching_results import (
        generate_all_figures,
        load_data,
        figure_1_heatmap,
        figure_2_dose_response,
        figure_3_causal_specificity,
        figure_4_dissociation,
        figure_5_layer_sweep,
        figure_6_probe_propagation,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    figures = sorted(set(args.figures))
    print(f"Loading {parquet_path}")
    df = load_data(parquet_path)
    print(f"  {len(df):,} rows · {df['sample_id'].nunique()} samples · "
          f"{df['severity'].nunique()} severities · "
          f"{df[df['span_type'].notna()]['condition'].nunique()} patching conditions")
    print(f"Output → {out}/\n")

    if 1 in figures:
        print("[1/6] Patching Effect Matrix ...")
        figure_1_heatmap(df, out, metric=args.metric)

    if 2 in figures:
        print("[2/6] Dose-Response Curves ...")
        figure_2_dose_response(df, out, metric=args.metric)

    if 3 in figures:
        print(f"[3/6] Causal Specificity (n_bootstrap={args.n_bootstrap}) ...")
        figure_3_causal_specificity(df, out, n_bootstrap=args.n_bootstrap, metric=args.metric)

    if 4 in figures:
        print(f"[4/6] Accuracy-Confidence Dissociation (severity={args.severity_fig4}) ...")
        figure_4_dissociation(df, out, severity=args.severity_fig4)

    if 5 in figures:
        print(f"[5/6] Layer Sweep (span={args.fig5_span}) ...")
        figure_5_layer_sweep(
            df, out,
            span=args.fig5_span,
            metric=args.metric,
            probe_peak_layer=args.probe_peak_layer,
        )

    if 6 in figures:
        print("[6/6] Cross-Span Probe Propagation ...")
        figure_6_probe_propagation(df, out)

    print(f"\nDone. Figures saved to {out}/")


if __name__ == "__main__":
    main()
