"""Select a shared eligible cohort from existing Qwen sweep results.

The selected samples are clean-correct and corrupted-incorrect in both the
older positional and layer sweeps. Their original patching-dataset records are
written unchanged and in their original order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/projects/prjs2014/patching")
MODEL = "Qwen_Qwen3-5-9B"
OLD_TAG = "extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking"
DATASETS = ("vqa-v2", "coco-qa-vi", "pope", "imagenet-r")


def dataset_component(dataset: str) -> Path:
    return Path("pope/random") if dataset == "pope" else Path(dataset)


def dataset_filename(dataset: str) -> str:
    return f"patching_dataset_{dataset}.json"


def result_path(kind: str, dataset: str) -> Path:
    base = (
        "layer_sweep_results"
        if kind == "layer"
        else "positional_patching_results_including_visual"
    )
    filename = (
        "layer_sweep_results.parquet"
        if kind == "layer"
        else "positional_patching_results.parquet"
    )
    return ROOT / base / MODEL / dataset_component(dataset) / OLD_TAG / filename


def source_dataset_path(dataset: str) -> Path:
    return (
        ROOT
        / MODEL
        / dataset_component(dataset)
        / dataset_filename(dataset)
    )


def eligible_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["sample_id", "severity", "condition", "score"],
    )
    frame = frame[frame["severity"].eq("extreme")]
    scores = (
        frame[frame["condition"].isin(["clean", "degraded"])]
        .groupby(["sample_id", "condition"], as_index=False)["score"]
        .first()
        .pivot(index="sample_id", columns="condition", values="score")
    )
    scores.index = scores.index.map(str)
    return scores[(scores["clean"] > 0) & scores["degraded"].eq(0)]


def select_dataset(dataset: str, count: int, output_name: str) -> dict[str, Any]:
    layer_path = result_path("layer", dataset)
    positional_path = result_path("positional", dataset)
    layer = eligible_scores(layer_path)
    positional = eligible_scores(positional_path)
    shared_ids = set(layer.index) & set(positional.index)
    if len(shared_ids) < count:
        raise RuntimeError(
            f"{dataset}: only {len(shared_ids)} samples are eligible in both "
            f"sweeps; requested {count}"
        )

    source_path = source_dataset_path(dataset)
    source = json.loads(source_path.read_text())
    selected = [
        sample for sample in source["samples"] if str(sample["sample_id"]) in shared_ids
    ][:count]
    selected_ids = [str(sample["sample_id"]) for sample in selected]
    if len(selected_ids) != count or len(set(selected_ids)) != count:
        raise RuntimeError(
            f"{dataset}: selected {len(selected_ids)} records "
            f"({len(set(selected_ids))} unique), expected {count}"
        )

    output_dir = source_path.parent / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / source_path.name
    metadata = dict(source.get("metadata", {}))
    metadata["selection"] = {
        "filter": "clean accuracy > 0 and corrupted accuracy = 0",
        "required_in": ["layer", "positional"],
        "severity": "extreme",
        "count": count,
        "source_dataset": str(source_path),
        "source_layer_results": str(layer_path),
        "source_positional_results": str(positional_path),
        "old_layer_eligible": len(layer),
        "old_positional_eligible": len(positional),
        "shared_eligible": len(shared_ids),
    }
    output_path.write_text(
        json.dumps({"metadata": metadata, "samples": selected}, indent=2) + "\n"
    )

    manifest = {
        "dataset": dataset,
        "count": count,
        "output_dataset": str(output_path),
        "selected_sample_ids": selected_ids,
        "old_layer_eligible": len(layer),
        "old_positional_eligible": len(positional),
        "shared_eligible": len(shared_ids),
        "layer_only": len(set(layer.index) - set(positional.index)),
        "positional_only": len(set(positional.index) - set(layer.index)),
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=350)
    parser.add_argument(
        "--output_name",
        default="selected_clean_correct_corrupted_incorrect_350",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in DATASETS:
        manifest = select_dataset(dataset, args.count, args.output_name)
        print(
            f"{dataset:10s}: selected {manifest['count']} from "
            f"{manifest['shared_eligible']} shared eligible -> "
            f"{manifest['output_dataset']}"
        )


if __name__ == "__main__":
    main()
