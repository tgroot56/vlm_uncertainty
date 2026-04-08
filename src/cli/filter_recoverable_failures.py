"""Filter a patching dataset to Group 1 (recoverable failures) samples only.

Recoverable failure = clean correct AND degraded incorrect for a given severity.

For each sample we:
  1. Require clean_model_correct == True.
  2. Keep only the visual_perturbation entries where model_correct == False.
  3. Drop the sample entirely if no perturbation remains after filtering.

The output JSON follows the same schema as the input (patching_dataset_<id>.json)
so that run_patching.job works on it without modification.

Usage:
  python -m src.cli.filter_recoverable_failures \\
      --patching_dataset /path/to/patching_dataset_vqa-v2.json \\
      [--output /path/to/patching_dataset_vqa-v2_recoverable.json]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def filter_recoverable_failures(
    data: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Return (filtered_records, stats_dict).

    Filters samples to those where:
      - clean_model_correct is True
      - at least one visual_perturbation has model_correct == False

    For kept samples the visual_perturbations list is pruned to only the
    failing severities.
    """
    records = data["samples"]
    stats: Dict[str, int] = {
        "total_in": len(records),
        "clean_incorrect_dropped": 0,
        "no_failure_dropped": 0,
        "kept": 0,
    }

    per_severity: Dict[str, int] = {}
    filtered: List[Dict[str, Any]] = []

    for rec in records:
        if not rec.get("clean_model_correct", False):
            stats["clean_incorrect_dropped"] += 1
            continue

        failing_perts = [
            p for p in rec.get("visual_perturbations", [])
            if not p.get("model_correct", True)
        ]
        if not failing_perts:
            stats["no_failure_dropped"] += 1
            continue

        for p in failing_perts:
            sev = str(p.get("severity", "unknown"))
            per_severity[sev] = per_severity.get(sev, 0) + 1

        new_rec = {**rec, "visual_perturbations": failing_perts}
        filtered.append(new_rec)
        stats["kept"] += 1

    stats["per_severity"] = per_severity  # type: ignore[assignment]
    return filtered, stats


def _print_stats(stats: Dict[str, int], original_total: int) -> None:
    kept = stats["kept"]
    total = stats["total_in"]
    print(f"\nFilter summary (threshold: model_correct field, baked-in at score >= 1.0)")
    print(f"{'─'*55}")
    print(f"  Input samples                   : {total:>5d}")
    print(f"  Dropped – clean incorrect        : {stats['clean_incorrect_dropped']:>5d}")
    print(f"  Dropped – no degraded failure    : {stats['no_failure_dropped']:>5d}")
    print(f"  Kept (recoverable failures)      : {kept:>5d}  ({100*kept/max(total,1):.1f}%)")
    per_sev = stats.get("per_severity", {})
    if per_sev:
        print(f"\n  Failing (severity, N perturbations kept):")
        for sev in sorted(per_sev):
            print(f"    {sev:<12s}: {per_sev[sev]:>5d}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter patching dataset to recoverable-failure samples (Group 1)"
    )
    parser.add_argument(
        "--patching_dataset", type=str, required=True,
        help="Path to patching_dataset_<id>.json",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path. Defaults to <input_stem>_recoverable.json next to the input.",
    )
    args = parser.parse_args()

    print(f"Loading {args.patching_dataset} …")
    with open(args.patching_dataset) as f:
        data = json.load(f)

    filtered_records, stats = filter_recoverable_failures(data)
    _print_stats(stats, len(data["samples"]))

    # Build output path
    if args.output is None:
        base, ext = os.path.splitext(args.patching_dataset)
        out_path = f"{base}_recoverable{ext}"
    else:
        out_path = args.output

    # Write new dataset with same schema
    out_data: Dict[str, Any] = {
        "metadata": {
            **data["metadata"],
            "num_samples": len(filtered_records),
            "filter": "recoverable_failures",
            "filter_description": (
                "clean_model_correct=True AND model_correct=False "
                "for the retained visual_perturbations"
            ),
        },
        "samples": filtered_records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved {len(filtered_records)} samples → {out_path}")


if __name__ == "__main__":
    main()
