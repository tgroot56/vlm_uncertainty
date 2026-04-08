"""Finalize a generated UQ dataset from a checkpoint without rerunning a VLM."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading checkpoint: {path}")
    print(f"Checkpoint size: {path.stat().st_size / (1024 ** 3):.2f} GiB")
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _validate_checkpoint(checkpoint: dict[str, Any], expected_samples: int | None) -> None:
    required_keys = ["X_parts", "y_list", "rows", "feature_names", "processed_indices"]
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {missing}")

    counts = {
        "X_parts": len(checkpoint["X_parts"]),
        "y_list": len(checkpoint["y_list"]),
        "rows": len(checkpoint["rows"]),
        "processed_indices": len(checkpoint["processed_indices"]),
    }
    print("Checkpoint counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    if len(set(counts.values())) != 1:
        raise ValueError(f"Checkpoint count mismatch: {counts}")

    if expected_samples is not None and counts["X_parts"] != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, but checkpoint has {counts['X_parts']}."
        )

    if counts["X_parts"] == 0:
        raise ValueError("Checkpoint contains no samples.")


def finalize_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path,
    dataset_id: str,
    model_id: str,
    seed_offset: int,
    max_samples: int | None,
    expected_samples: int | None,
) -> None:
    checkpoint = _load_checkpoint(checkpoint_path)
    _validate_checkpoint(checkpoint, expected_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = checkpoint["feature_names"]
    feature_dims = checkpoint.get("feature_dims", [])
    x_parts = checkpoint["X_parts"]

    print("Building final feature tensor with torch.cat(...).")
    print("This is the memory-heavy recovery step; no VLM/model forward passes are run.")
    X = torch.cat(x_parts, dim=0).cpu().float()

    # Release the per-sample tensor references before serializing the final payload.
    checkpoint["X_parts"] = []
    del x_parts
    gc.collect()

    y = torch.tensor(checkpoint["y_list"], dtype=torch.float32)
    rows = checkpoint["rows"]

    metadata = {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "num_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "feature_names": feature_names,
        "feature_dims": feature_dims,
        "seed_offset": seed_offset,
        "max_samples": max_samples,
        "label_mean": float(y.mean().item()),
        "label_variance": float(y.var(unbiased=False).item()),
        "recovered_from_checkpoint": str(checkpoint_path),
    }

    print(f"Saving finalized dataset: {output_path}")
    torch.save(
        {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "feature_dims": feature_dims,
            "rows": rows,
            "metadata": metadata,
        },
        output_path,
    )

    metadata_path = output_path.parent / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print("Finalized supervised dataset:")
    print(f"  dataset:  {output_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  N={X.shape[0]} D={X.shape[1]} label_mean={metadata['label_mean']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a supervision_dataset.pt from a completed checkpoint.pt."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dataset-id", type=str, default="imagenet-r")
    parser.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--seed-offset", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_path}. Pass --overwrite to replace it."
        )

    finalize_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        seed_offset=args.seed_offset,
        max_samples=args.max_samples,
        expected_samples=args.expected_samples,
    )


if __name__ == "__main__":
    main()
