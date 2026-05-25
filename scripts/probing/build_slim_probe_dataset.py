#!/usr/bin/env python
"""Create a smaller supervision dataset with selected feature blocks only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.probe.data import build_feature_slice_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source supervision_dataset.pt")
    parser.add_argument("--output", required=True, help="Output slim .pt path")
    parser.add_argument("--features", nargs="+", required=True, help="Feature blocks to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading source dataset: {input_path}")
    payload = torch.load(input_path, map_location="cpu", weights_only=False)

    x_all = payload["X"].float()
    y = payload["y"].float()
    all_names = payload["feature_names"]
    all_dims = payload.get("feature_dims", None)
    name_to_slice = build_feature_slice_map(all_names, all_dims, x_all.shape[1])

    missing = [feature for feature in args.features if feature not in name_to_slice]
    if missing:
        raise ValueError(f"Missing requested features {missing}; available={all_names}")

    selected = [x_all[:, name_to_slice[feature]] for feature in args.features]
    x = torch.cat(selected, dim=1).contiguous()
    feature_dims = [name_to_slice[feature].stop - name_to_slice[feature].start for feature in args.features]

    slim = {
        "X": x,
        "y": y,
        "feature_names": list(args.features),
        "feature_dims": feature_dims,
        "source_path": str(input_path),
    }
    if "rows" in payload:
        slim["rows"] = payload["rows"]

    print(f"Saving slim dataset: {output_path}")
    print(f"  samples: {len(y)}")
    print(f"  X shape: {tuple(x.shape)}")
    print(f"  features: {list(args.features)}")

    y_min = float(y.min())
    y_max = float(y.max())
    y_mean = float(y.mean())
    num_eq_1 = int((y == 1.0).sum())
    num_lt_1 = int(len(y) - num_eq_1)
    print(
        "\n[LABEL VERIFY] Soft-score y stats: "
        f"min={y_min:.4f}, max={y_max:.4f}, mean={y_mean:.4f}"
    )
    print(
        "[LABEL VERIFY] Binarised (y==1.0 -> 1, else 0): "
        f"num_pos={num_eq_1} ({num_eq_1 / max(len(y), 1):.2%}), "
        f"num_neg={num_lt_1} ({num_lt_1 / max(len(y), 1):.2%})"
    )
    if num_eq_1 == 0 or num_lt_1 == 0:
        raise RuntimeError(
            "Binarised labels collapsed to a single class - aborting slim build."
        )

    torch.save(slim, output_path)


if __name__ == "__main__":
    main()
