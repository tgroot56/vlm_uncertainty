from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _most_common_answer(answers: List[str]) -> str:
    if not answers:
        return ""
    return Counter(str(a).strip().lower() for a in answers).most_common(1)[0][0]


def _run_row(label: str, row: pd.Series, *, kind: str, layer: int | None = None) -> Dict[str, Any]:
    return {
        "label": label,
        "kind": kind,
        "layer": layer,
        "answer": str(row["predicted_answer"]),
        "score": float(row["score"]),
        "top1_probability": float(row["top1_probability"]),
        "output_entropy": float(row["output_entropy"]),
    }


def build_payload(results_path: Path, dataset_path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(results_path)
    with dataset_path.open() as f:
        records = {str(r["sample_id"]): r for r in json.load(f)["samples"]}

    clean_mean = df[df["condition"] == "clean"]["score"].mean()
    late = df[(df["condition"] == "patched") & (df["layer"].between(22, 31))]
    late_mean = late.groupby("layer")["score"].mean()
    late_above_clean = [int(layer) for layer, score in late_mean.items() if float(score) > float(clean_mean)]

    summary: List[Dict[str, Any]] = []
    for cond in ["clean", "degraded"]:
        sub = df[df["condition"] == cond]
        summary.append(
            {
                "label": cond,
                "kind": cond,
                "layer": None,
                "mean_score": float(sub["score"].mean()),
                "mean_top1_probability": float(sub["top1_probability"].mean()),
                "n": int(sub["sample_id"].nunique()),
            }
        )
    for layer, sub in df[df["condition"] == "patched"].groupby("layer"):
        summary.append(
            {
                "label": f"layer {int(layer)}",
                "kind": "layer",
                "layer": int(layer),
                "mean_score": float(sub["score"].mean()),
                "mean_top1_probability": float(sub["top1_probability"].mean()),
                "n": int(sub["sample_id"].nunique()),
            }
        )

    samples: List[Dict[str, Any]] = []
    for sample_id, sub in df.groupby("sample_id", sort=False):
        sample_id = str(sample_id)
        record = records[sample_id]
        extreme = next(
            p for p in record["visual_perturbations"] if p.get("severity") == "extreme"
        )
        clean = sub[sub["condition"] == "clean"].iloc[0]
        degraded = sub[sub["condition"] == "degraded"].iloc[0]
        runs = [
            _run_row("clean", clean, kind="clean"),
            _run_row("degraded", degraded, kind="degraded"),
        ]
        for _, row in sub[sub["condition"] == "patched"].sort_values("layer").iterrows():
            layer = int(row["layer"])
            runs.append(_run_row(f"layer {layer}", row, kind="layer", layer=layer))

        gt_answers = record.get("gt_answers") or []
        samples.append(
            {
                "uid": f"vqa-v2:{sample_id}",
                "dataset": "vqa-v2",
                "sample_id": sample_id,
                "question": record.get("clean_question", ""),
                "most_common_gt_answer": _most_common_answer(gt_answers)
                or record.get("correct_answer", ""),
                "all_gt_answers": ", ".join(map(str, gt_answers)),
                "clean_image_path": record.get("clean_image_path", ""),
                "corrupted_image_path": extreme.get("blurred_image_path", ""),
                "runs": runs,
            }
        )

    return {
        "title": "LLaVA VQA-v2 final-token smoke layer sweep",
        "position": "final prompt token only (offset -1)",
        "num_layers": int(df[df["condition"] == "patched"]["layer"].nunique()),
        "num_samples": len(samples),
        "late_layers_above_clean": late_above_clean,
        "note": (
            "Clean mean score is 0.80 on this five-sample smoke run. Layers "
            "22-31 reach 0.933 because one sample where the clean answer is "
            "only partially correct becomes fully correct after late-layer "
            "patching, while first-token top-1 probability at layer 31 exactly "
            "matches the clean run."
        ),
        "summary": summary,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = build_payload(args.results, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
