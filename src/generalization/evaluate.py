from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping
from statistics import fmean, pstdev


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return json.load(handle)


def load_run_summary(run_dir: str) -> Dict[str, Any]:
    run_path = Path(run_dir)
    metrics = _load_json(run_path / "metrics_test.json")
    config = _load_json(run_path / "config.json")
    return {
        "run_dir": str(run_path),
        "metrics": metrics,
        "config": config,
    }


def save_comparison_summary(
    *,
    output_path: str,
    model_name: str,
    held_out_dataset: str,
    data_path: str,
    variants: Mapping[str, str],
) -> Dict[str, Any]:
    summary = {
        "model_name": model_name,
        "held_out_dataset": held_out_dataset,
        "data_path": data_path,
        "variants": {},
    }

    for variant_name, run_dir in variants.items():
        summary["variants"][variant_name] = load_run_summary(run_dir)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _aggregate_metrics(seed_runs: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    metric_keys = ["test_loss", "test_ece", "test_auroc", "best_val_loss", "best_epoch"]
    mean_metrics: Dict[str, Any] = {}
    std_metrics: Dict[str, Any] = {}

    for key in metric_keys:
        values = [run["metrics"].get(key) for run in seed_runs if run.get("metrics", {}).get(key) is not None]
        if not values:
            mean_metrics[key] = None
            std_metrics[key] = None
            continue
        mean_metrics[key] = float(fmean(values))
        std_metrics[key] = float(pstdev(values)) if len(values) > 1 else 0.0

    return {
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }


def save_aggregate_comparison_summary(
    *,
    output_path: str,
    model_name: str,
    held_out_dataset: str,
    data_path: str,
    seed_variant_runs: Mapping[int, Mapping[str, str]],
) -> Dict[str, Any]:
    summary = {
        "model_name": model_name,
        "held_out_dataset": held_out_dataset,
        "data_path": data_path,
        "seeds": [int(seed) for seed in seed_variant_runs.keys()],
        "variants": {},
    }

    variant_names = sorted({variant for runs in seed_variant_runs.values() for variant in runs.keys()})
    for variant_name in variant_names:
        seed_runs = []
        for seed, runs in seed_variant_runs.items():
            if variant_name not in runs:
                continue
            run_summary = load_run_summary(runs[variant_name])
            run_summary["seed"] = int(seed)
            seed_runs.append(run_summary)

        aggregated = _aggregate_metrics(seed_runs)
        summary["variants"][variant_name] = {
            "seed_runs": seed_runs,
            **aggregated,
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def save_multitest_summary(
    *,
    output_path: str,
    model_name: str,
    experiment_name: str,
    data_path: str,
    variants: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    summary = {
        "model_name": model_name,
        "experiment_name": experiment_name,
        "data_path": data_path,
        "variants": {},
    }

    for variant_name, test_dataset_runs in variants.items():
        summary["variants"][variant_name] = {
            "test_datasets": {
                test_dataset: load_run_summary(run_dir)
                for test_dataset, run_dir in test_dataset_runs.items()
            }
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def save_aggregate_multitest_summary(
    *,
    output_path: str,
    model_name: str,
    experiment_name: str,
    data_path: str,
    seed_variant_test_runs: Mapping[int, Mapping[str, Mapping[str, str]]],
) -> Dict[str, Any]:
    summary = {
        "model_name": model_name,
        "experiment_name": experiment_name,
        "data_path": data_path,
        "seeds": [int(seed) for seed in seed_variant_test_runs.keys()],
        "variants": {},
    }

    variant_names = sorted(
        {variant for seed_runs in seed_variant_test_runs.values() for variant in seed_runs.keys()}
    )
    for variant_name in variant_names:
        test_dataset_names = sorted(
            {
                test_dataset
                for seed_runs in seed_variant_test_runs.values()
                for test_dataset in seed_runs.get(variant_name, {}).keys()
            }
        )
        variant_summary = {"test_datasets": {}}
        for test_dataset in test_dataset_names:
            seed_runs = []
            for seed, variant_runs in seed_variant_test_runs.items():
                run_dir = variant_runs.get(variant_name, {}).get(test_dataset)
                if run_dir is None:
                    continue
                run_summary = load_run_summary(run_dir)
                run_summary["seed"] = int(seed)
                seed_runs.append(run_summary)
            variant_summary["test_datasets"][test_dataset] = {
                "seed_runs": seed_runs,
                **_aggregate_metrics(seed_runs),
            }
        summary["variants"][variant_name] = variant_summary

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary
