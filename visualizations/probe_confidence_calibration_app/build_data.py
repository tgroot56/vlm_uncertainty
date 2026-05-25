from __future__ import annotations

import json
import math
import random
import shutil
import string
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from PIL import ImageOps
from torch import nn


RESULTS_ROOT = Path("/projects/prjs2014/probe_results/generalization/task_gen/llava")
COMBINED_DATASET_PATH = Path(
    "/projects/prjs2014/probe_results/generalization/datasets/llava/generalizability_supervision_dataset.pt"
)
HF_CACHE = Path("/projects/prjs2014/tgroot/hf_datasets")
OUTPUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUTPUT_DIR / "assets"

DATASETS = [
    ("vqa-v2", "VQA-v2"),
    ("coco-qa-vi", "COCO-QA-VI"),
    ("imagenet-r", "ImageNet-R"),
    ("pope/random", "POPE random"),
]

PROBE_FEATURE = "lm_fullspan_lasttok_layer_16"
SEED = 7
SAMPLES_PER_DATASET = 100
FEATURED_PER_DATASET = 4
NUM_BINS = 10


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.2):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x)).squeeze(-1)


def feature_slices(feature_names: list[str], feature_dims: list[int]) -> dict[str, slice]:
    out: dict[str, slice] = {}
    cursor = 0
    for name, dim in zip(feature_names, feature_dims):
        out[name] = slice(cursor, cursor + int(dim))
        cursor += int(dim)
    return out


def probe_checkpoint_path(dataset_id: str) -> Path:
    return (
        RESULTS_ROOT
        / dataset_id
        / PROBE_FEATURE
        / "seed_runs"
        / "seed_42"
        / "in_domain"
        / "mlp"
        / PROBE_FEATURE
        / "best_model.pt"
    )


def predict_probe(dataset_id: str, features: torch.Tensor) -> tuple[list[float], str]:
    checkpoint_path = probe_checkpoint_path(dataset_id)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MLPProbe(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=checkpoint.get("hidden_dims") or [256, 128],
        dropout=0.2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = features.float()
    if checkpoint.get("normalize"):
        x = (x - checkpoint["mean"].float()) / (checkpoint["std"].float() + 1e-8)

    with torch.no_grad():
        preds = model(x).float().tolist()
    return [float(p) for p in preds], str(checkpoint_path)


def answer_confidence(payload: dict[str, Any], slices: dict[str, slice]) -> tuple[list[float], str]:
    x = payload["X"].float()
    if "answer_gen_geom_mean_probability" in slices:
        values = x[:, slices["answer_gen_geom_mean_probability"]].squeeze(-1).clamp(0, 1)
        return [float(v) for v in values.tolist()], "answer_gen_geom_mean_probability"
    if "answer_gen_entropy_mean" in slices:
        values = torch.exp(-x[:, slices["answer_gen_entropy_mean"]].squeeze(-1)).clamp(0, 1)
        return [float(v) for v in values.tolist()], "exp(-answer_gen_entropy_mean)"
    raise ValueError("No answer confidence feature found in combined supervision dataset.")


def brier(preds: list[float], labels: list[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(preds)


def calibration_bins(preds: list[float], labels: list[float]) -> list[dict[str, float]]:
    bins: list[dict[str, float]] = []
    for idx in range(NUM_BINS):
        lo = idx / NUM_BINS
        hi = (idx + 1) / NUM_BINS
        members = [
            (p, y)
            for p, y in zip(preds, labels)
            if (lo <= p < hi) or (idx == NUM_BINS - 1 and p == 1.0)
        ]
        if members:
            conf = sum(p for p, _ in members) / len(members)
            acc = sum(y for _, y in members) / len(members)
        else:
            conf = (lo + hi) / 2
            acc = float("nan")
        bins.append({"bin": idx, "confidence": conf, "accuracy": acc, "count": len(members)})
    return bins


def ece(preds: list[float], labels: list[float]) -> float:
    total = len(preds)
    error = 0.0
    for item in calibration_bins(preds, labels):
        if item["count"] == 0 or math.isnan(item["accuracy"]):
            continue
        error += item["count"] / total * abs(item["confidence"] - item["accuracy"])
    return error


def option_keys(k: int) -> list[str]:
    alphabet = list(string.ascii_uppercase)
    keys: list[str] = []
    i = 0
    while len(keys) < k:
        n = i
        s = ""
        while True:
            s = alphabet[n % 26] + s
            n = n // 26 - 1
            if n < 0:
                break
        keys.append(s)
        i += 1
    return keys


def build_imagenet_option_map(raw_dataset: Dataset, source_idx: int, k_options: int = 16) -> dict[str, str]:
    all_class_names = sorted(set(raw_dataset["class_name"]))
    gt_class = raw_dataset[source_idx]["class_name"]
    rng = random.Random(42 + source_idx)
    distractors = rng.sample([name for name in all_class_names if name != gt_class], k_options - 1)
    options = [gt_class] + distractors
    rng.shuffle(options)
    return dict(zip(option_keys(k_options), options))


def arrow_dataset(dataset_id: str) -> Dataset:
    if dataset_id == "vqa-v2":
        return Dataset.from_file(
            str(
                HF_CACHE
                / "HuggingFaceM4___vq_av2/default/1.0.0/"
                / "e4d008385143be7a6bd81e99483e671d5096942bcb987542217121a5ac2cb420"
                / "vq_av2-validation.arrow"
            )
        )
    if dataset_id == "coco-qa-vi":
        root = (
            HF_CACHE
            / "ThucPD___coco-qa-vi/default/0.0.0/"
            / "dd8103c605c5c0a9a7bcf5a938ce608417912eb2"
        )
        return concatenate_datasets([Dataset.from_file(str(path)) for path in sorted(root.glob("coco-qa-vi-test-*.arrow"))])
    if dataset_id == "imagenet-r":
        root = (
            HF_CACHE
            / "axiong___imagenet-r/test/0.0.0/"
            / "762a23f169e080bef0be7c9d96bf73a126cbb660"
        )
        return concatenate_datasets([Dataset.from_file(str(path)) for path in sorted(root.glob("imagenet-r-test-*.arrow"))])
    if dataset_id == "pope/random":
        return Dataset.from_file(
            str(
                HF_CACHE
                / "lmms-lab___pope/Full/0.0.0/4db1276663dfa5eb8ad16a52d24c31a09e470896"
                / "pope-random.arrow"
            )
        )
    raise ValueError(f"Unknown dataset id: {dataset_id}")


def save_thumbnail(image, dataset_id: str, source_idx: int) -> str:
    safe_dataset = dataset_id.replace("/", "_")
    out_dir = ASSET_DIR / safe_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_idx}.jpg"
    if out_path.exists():
        return out_path.relative_to(OUTPUT_DIR).as_posix()

    img = ImageOps.exif_transpose(image).convert("RGB")
    img.thumbnail((520, 390))
    img.save(out_path, quality=86, optimize=True)
    return out_path.relative_to(OUTPUT_DIR).as_posix()


def raw_metadata(dataset_id: str, raw_dataset: Dataset, source_idx: int, row: dict[str, Any]) -> dict[str, Any]:
    raw = raw_dataset[source_idx]
    image_path = save_thumbnail(raw["image"], dataset_id, source_idx)

    if dataset_id == "vqa-v2":
        answers = row.get("gt_answers") or []
        gt = ", ".join(str(x) for x in answers[:4])
        return {
            "question": raw.get("question", ""),
            "ground_truth": gt,
            "image_path": image_path,
        }

    if dataset_id == "coco-qa-vi":
        return {
            "question": raw.get("question", ""),
            "ground_truth": str(row.get("gt_answer") or raw.get("answer") or ""),
            "image_path": image_path,
        }

    if dataset_id == "pope/random":
        return {
            "question": raw.get("question", ""),
            "ground_truth": str(row.get("gt_answer") or raw.get("answer") or ""),
            "image_path": image_path,
        }

    if dataset_id == "imagenet-r":
        option_map = build_imagenet_option_map(raw_dataset, source_idx)
        pred_key = str(row.get("pred_key") or row.get("raw_text") or "")
        gt_key = str(row.get("gt_key") or "")
        pred_label = option_map.get(pred_key, pred_key)
        gt_label = option_map.get(gt_key, str(row.get("gt_class") or raw.get("class_name") or ""))
        return {
            "question": "Which object is shown in the image?",
            "ground_truth": f"{gt_key}: {gt_label}" if gt_key else gt_label,
            "predicted_answer": f"{pred_key}: {pred_label}" if pred_key else pred_label,
            "image_path": image_path,
        }

    raise ValueError(dataset_id)


def case_label(label: float, answer_conf: float, probe_conf: float) -> str:
    if label < 0.5 and answer_conf >= 0.55 and probe_conf < answer_conf:
        return "Answer overconfident; probe lowers confidence"
    if label >= 0.5 and answer_conf <= 0.55 and probe_conf > answer_conf:
        return "Answer hesitant; probe raises confidence"
    if label < 0.5 and probe_conf <= 0.5:
        return "Probe catches an incorrect answer"
    if label >= 0.5 and probe_conf >= 0.5:
        return "Probe supports a correct answer"
    return "Calibration contrast"


def select_qualitative_indices(
    labels: list[float],
    answer_preds: list[float],
    probe_preds: list[float],
    count: int,
    seed: int,
) -> list[int]:
    candidates = list(range(len(labels)))
    wrong_answer_high = [
        idx for idx in candidates if labels[idx] < 0.5 and answer_preds[idx] >= 0.5 and probe_preds[idx] < answer_preds[idx]
    ]
    correct_probe_raises = [
        idx for idx in candidates if labels[idx] >= 0.5 and answer_preds[idx] <= 0.7 and probe_preds[idx] > answer_preds[idx]
    ]
    wrong_answer_high.sort(key=lambda idx: (answer_preds[idx] - probe_preds[idx], answer_preds[idx]), reverse=True)
    correct_probe_raises.sort(key=lambda idx: (probe_preds[idx] - answer_preds[idx], probe_preds[idx]), reverse=True)

    selected: list[int] = []
    for pool in (wrong_answer_high, correct_probe_raises):
        for idx in pool:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= count:
                return selected

    rng = random.Random(seed)
    rest = [idx for idx in candidates if idx not in selected]
    rng.shuffle(rest)
    selected.extend(rest[: count - len(selected)])
    return selected


def paired_dataset(
    *,
    payload: dict[str, Any],
    slices: dict[str, slice],
    raw_dataset: Dataset,
    all_answer_confidence: list[float],
    answer_feature: str,
    dataset_id: str,
    display_name: str,
    dataset_rank: int,
) -> dict[str, Any]:
    rows = payload["rows"]
    labels_all = payload["y"].float().tolist()
    dataset_indices = [idx for idx, row in enumerate(rows) if row.get("dataset_name") == dataset_id]

    probe_features = payload["X"][dataset_indices, slices[PROBE_FEATURE]].float()
    probe_preds, probe_path = predict_probe(dataset_id, probe_features)
    answer_preds = [all_answer_confidence[idx] for idx in dataset_indices]
    labels = [float(labels_all[idx]) for idx in dataset_indices]

    selected = select_qualitative_indices(labels, answer_preds, probe_preds, SAMPLES_PER_DATASET, SEED + dataset_rank)

    samples: list[dict[str, Any]] = []
    for rank, local_idx in enumerate(selected):
        global_idx = dataset_indices[local_idx]
        row = rows[global_idx]
        source_idx = int(row.get("source_row_index", row.get("idx", local_idx)))
        meta = raw_metadata(dataset_id, raw_dataset, source_idx, row)
        answer = str(meta.get("predicted_answer") or row.get("pred_answer") or row.get("raw_text") or "")
        label = labels[local_idx]
        answer_conf = answer_preds[local_idx]
        probe_conf = probe_preds[local_idx]
        samples.append(
            {
                "rank": rank + 1,
                "test_index": local_idx,
                "source_row_index": source_idx,
                "image_path": meta["image_path"],
                "question": meta["question"],
                "answer": answer,
                "ground_truth": meta["ground_truth"],
                "answer_confidence": answer_conf,
                "probe_confidence": probe_conf,
                "label": label,
                "correct": label >= 0.5,
                "case": case_label(label, answer_conf, probe_conf),
                "confidence_delta": probe_conf - answer_conf,
            }
        )

    return {
        "id": dataset_id,
        "name": display_name,
        "num_test": len(labels),
        "num_visible_samples": len(samples),
        "label_mean": sum(labels) / len(labels),
        "answer_feature": answer_feature,
        "probe_feature": PROBE_FEATURE,
        "featured_samples": samples[:FEATURED_PER_DATASET],
        "samples": samples,
        "answer": {
            "brier": brier(answer_preds, labels),
            "ece": ece(answer_preds, labels),
            "bins": calibration_bins(answer_preds, labels),
        },
        "probe": {
            "brier": brier(probe_preds, labels),
            "ece": ece(probe_preds, labels),
            "bins": calibration_bins(probe_preds, labels),
        },
        "source_paths": {"answer": str(COMBINED_DATASET_PATH), "probe": probe_path},
    }


def main() -> None:
    payload = torch.load(COMBINED_DATASET_PATH, map_location="cpu", weights_only=False)
    slices = feature_slices(payload["feature_names"], payload["feature_dims"])
    all_answer_confidence, answer_feature = answer_confidence(payload, slices)

    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    datasets = []
    for rank, (dataset_id, display_name) in enumerate(DATASETS):
        raw_dataset = arrow_dataset(dataset_id)
        datasets.append(
            paired_dataset(
                payload=payload,
                slices=slices,
                raw_dataset=raw_dataset,
                all_answer_confidence=all_answer_confidence,
                answer_feature=answer_feature,
                dataset_id=dataset_id,
                display_name=display_name,
                dataset_rank=rank,
            )
        )

    payload_out = {
        "title": "Individual Samples: Probe Confidence Tracks Correctness Better",
        "subtitle": "Qualitative examples selected from 100 samples per dataset.",
        "model": "LLaVA 1.5 7B",
        "probe_feature": PROBE_FEATURE,
        "answer_confidence_feature": answer_feature,
        "answer_confidence_note": (
            "Uses answer_gen_geom_mean_probability when present. This saved all-dataset run lacks "
            "that feature for all datasets, so this build uses exp(-answer_gen_entropy_mean) as the "
            "available answer-side confidence proxy."
        ),
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "featured_per_dataset": FEATURED_PER_DATASET,
        "seed": SEED,
        "summary": {
            "avg_answer_ece": sum(ds["answer"]["ece"] for ds in datasets) / len(datasets),
            "avg_probe_ece": sum(ds["probe"]["ece"] for ds in datasets) / len(datasets),
            "avg_answer_brier": sum(ds["answer"]["brier"] for ds in datasets) / len(datasets),
            "avg_probe_brier": sum(ds["probe"]["brier"] for ds in datasets) / len(datasets),
        },
        "datasets": datasets,
    }

    json_text = json.dumps(payload_out, indent=2)
    (OUTPUT_DIR / "data.json").write_text(json_text + "\n", encoding="utf-8")
    (OUTPUT_DIR / "data.js").write_text("window.CALIBRATION_DATA = " + json_text + ";\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'data.json'}")
    print(f"Wrote {OUTPUT_DIR / 'data.js'}")
    print(f"Wrote thumbnails to {ASSET_DIR}")


if __name__ == "__main__":
    main()
