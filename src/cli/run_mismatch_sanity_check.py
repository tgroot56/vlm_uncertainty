"""2x2 mismatch sanity check.

Joins per-sample probe predictions with output-level confidence and ground-truth
correctness, then bins samples into a 2x2 grid (confidence x probe). Reports
counts and error rates per cell, with extra emphasis on the "overconfident
hallucination" cell (high output confidence, low probe score).

Usage:
    python -m src.cli.run_mismatch_sanity_check \\
        --model llava --dataset vqa-v2 [--probe_type mlp] [--feature ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

from src.probe.data import build_feature_slice_map
from src.storage_paths import storage_root


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PROBE_RESULTS_ROOT = os.path.join(REPO_ROOT, "probe_results")

# Maps (model_alias, dataset_alias) -> directory (relative to storage root) that
# contains run_<hash>/supervision_dataset.pt for the supervision dataset the probe
# was actually trained on.
#
# Layout is inconsistent across datasets: vqa-v2 / pope / imagenet-r were moved
# into supervised_datasets/, but coco-qa-vi (LLaVA) and Qwen runs live at the
# top-level legacy paths. Each entry below points to whichever path the probe
# trainer found via its own search logic — DO NOT change without verifying that
# y[test_indices] still matches the probe CSV labels.
SUPERVISION_LAYOUT: Dict[Tuple[str, str], str] = {
    ("llava", "vqa-v2"): "supervised_datasets/vqa-v2/llava-hf/llava-1.5-7b-hf",
    ("llava", "pope"): "supervised_datasets/pope/llava-hf/llava-1.5-7b-hf",
    ("llava", "coco-qa-vi"): "supervised_datasets/coco-qa-vi/llava-hf/llava-1.5-7b-hf",
    ("llava", "imagenet-r"): "supervised_datasets/imagenet_r_16mc/imagenet-r/llava-hf/llava-1.5-7b-hf",
    ("qwen", "pope"): "qwen/pope/Qwen/Qwen3.5-9B",
    ("qwen", "vqa-v2"): "qwen/vqa-v2/Qwen/Qwen3.5-9B",
    ("qwen", "imagenet-r"): "qwen/imagenet-r/Qwen/Qwen3.5-9B",
    ("qwen", "coco-qa-vi"): "qwen/coco-qa-vi/Qwen/Qwen3.5-9B",
}

# POPE is laid out per-split (adversarial/random). Maps (model_alias, pope_split)
# -> directory relative to storage root. The "random" split happens to live at
# the legacy top-level pope/ dir; adversarial sits one level deeper.
POPE_SUPERVISION_LAYOUT: Dict[Tuple[str, str], str] = {
    ("llava", "random"): "supervised_datasets/pope/llava-hf/llava-1.5-7b-hf",
    ("llava", "adversarial"): "supervised_datasets/pope/adversarial/llava-hf/llava-1.5-7b-hf",
    ("qwen", "random"): "qwen/pope/random/Qwen/Qwen3.5-9B",
    ("qwen", "adversarial"): "qwen/pope/adversarial/Qwen/Qwen3.5-9B",
}

POPE_SPLIT_CHOICES = ("adversarial", "random")

MID_LAYER_PROBE_CANDIDATES = [
    "lm_prompt_lasttok_layer_16",
    "lm_answer_lasttok_layer_16",
    "lm_answer_mean_layer_16",
    "lm_fullspan_lasttok_layer_16",
    "lm_fullspan_mean_layer_16",
]

# Output-confidence features whose direction must be flipped so that
# higher = more confident.
NEGATIVELY_ORIENTED_CONFIDENCE_FEATURES = {
    "answer_gen_entropy_max",
    "answer_gen_entropy_mean",
    "answer_gen_entropy_min",
    "answer_gen_entropy_std",
    "answer_gen_neglogp_mean",
    "answer_gen_neglogp_std",
    "answer_gen_negp_max",
    "answer_gen_negp_mean",
    "answer_gen_negp_min",
    "answer_gen_negp_std",
}


def find_supervision_pt(
    storage_root_path: str,
    model: str,
    dataset: str,
    run_hash: Optional[str] = None,
    pope_split: Optional[str] = None,
) -> str:
    if dataset == "pope" and pope_split is not None:
        key = (model, pope_split)
        if key not in POPE_SUPERVISION_LAYOUT:
            raise ValueError(
                f"No POPE supervised-dataset layout registered for "
                f"(model={model!r}, pope_split={pope_split!r}). "
                f"Add an entry to POPE_SUPERVISION_LAYOUT."
            )
        rel_path = POPE_SUPERVISION_LAYOUT[key]
    else:
        key = (model, dataset)
        if key not in SUPERVISION_LAYOUT:
            raise ValueError(
                f"No supervised-dataset layout registered for {key!r}. "
                f"Add an entry to SUPERVISION_LAYOUT."
            )
        rel_path = SUPERVISION_LAYOUT[key]
    base = os.path.join(storage_root_path, rel_path)
    pattern = os.path.join(base, run_hash if run_hash else "run_*", "supervision_dataset.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No supervision_dataset.pt found at {pattern!r}")
    if len(matches) > 1 and not run_hash:
        raise RuntimeError(
            f"Multiple supervision runs at {base}: {matches}. "
            f"Pass --run_hash to disambiguate."
        )
    return matches[0]


def auroc_from_predictions_csv(csv_path: str, threshold: float) -> Optional[float]:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    y_bin = (df["label"].values >= threshold).astype(int)
    if y_bin.min() == y_bin.max():
        return None
    return float(roc_auc_score(y_bin, df["prediction"].values))


def brier_from_results_csv(
    results_csv_path: str,
    feature: str,
    probe_type: str,
) -> Optional[float]:
    """Look up a probe's Brier score in the per-(model,dataset) results_brier.csv.

    The file has a `feature_label` column plus one column per probe_type.
    Returns None if the file or row is missing.
    """
    if not os.path.exists(results_csv_path):
        return None
    df = pd.read_csv(results_csv_path)
    if "feature_label" not in df.columns or probe_type not in df.columns:
        return None
    row = df[df["feature_label"] == feature]
    if len(row) == 0:
        return None
    val = row[probe_type].iloc[0]
    if pd.isna(val):
        return None
    return float(val)


def pick_best_probe(
    probe_dir: str,
    dataset_dir: str,
    probe_type: str,
    candidates: List[str],
    threshold: float,
    selection_metric: str,
) -> Tuple[str, float]:
    """Pick the candidate probe that scores best on `selection_metric`.

    `selection_metric == "auroc"` recomputes AUROC from each candidate's
    summary_predictions.csv (since metrics_test.json has test_auroc = null).
    `selection_metric == "brier"` reads the aggregated results_brier.csv,
    which is populated for every (model, dataset) and indexed by feature_label.
    Returns (feature, score) with score being whatever metric was used.
    """
    scored: List[Tuple[str, float]] = []
    if selection_metric == "auroc":
        for name in candidates:
            csv_path = os.path.join(probe_dir, name, "summary_predictions.csv")
            v = auroc_from_predictions_csv(csv_path, threshold)
            if v is not None:
                scored.append((name, v))
        if not scored:
            raise FileNotFoundError(
                f"None of the candidate probes had a usable summary_predictions.csv "
                f"under {probe_dir!r}: {candidates}"
            )
        scored.sort(key=lambda x: x[1], reverse=True)  # higher AUROC = better
    elif selection_metric == "brier":
        results_csv = os.path.join(dataset_dir, "results_brier.csv")
        for name in candidates:
            v = brier_from_results_csv(results_csv, name, probe_type)
            if v is not None:
                scored.append((name, v))
        if not scored:
            raise FileNotFoundError(
                f"No candidate features {candidates} found in {results_csv!r} "
                f"under column {probe_type!r}."
            )
        scored.sort(key=lambda x: x[1])  # lower Brier = better
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric!r}")
    return scored[0]


def replicate_test_indices(num_samples: int, seed: int = 42) -> List[int]:
    """Reproduce the test split that src/probe/data.py:create_dataloaders uses.

    The probe trainer feeds the test loader with shuffle=False, so the row
    order in summary_predictions.csv matches the order yielded by iterating
    the Subset returned by random_split.
    """
    n_train = int(num_samples * 0.7)
    n_val = int(num_samples * 0.15)
    n_test = num_samples - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    _, _, test_subset = torch.utils.data.random_split(
        range(num_samples), [n_train, n_val, n_test], generator=generator
    )
    return list(test_subset)


def extract_sample_id(row: dict) -> str:
    for key in ("question_id", "idx", "image_id", "sample_id", "id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def percentile_rank(values: np.ndarray) -> np.ndarray:
    # Average rank handles ties gracefully; divide by N to get [1/N, 1].
    return scipy_stats.rankdata(values, method="average") / len(values)


def cell_label_for(hi_conf: bool, hi_probe: bool) -> str:
    if hi_conf and hi_probe:
        return "A_aligned_confident"
    if hi_conf and not hi_probe:
        return "B_overconfident"
    if not hi_conf and hi_probe:
        return "C_undercommitted"
    return "D_aligned_uncertain"


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def fmt_pval(p: float) -> str:
    if p == 0.0:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3f}"


def format_cell_block(label: str, count: int, total: int, n_incorrect: int) -> str:
    pct_total = count / total if total else 0.0
    err_rate = n_incorrect / count if count else 0.0
    return (
        f"{label}: {count} ({fmt_pct(pct_total)})\n"
        f"err: {fmt_pct(err_rate)}"
    )


def chi2_cell_vs_rest(cell_mask: np.ndarray, incorrect_mask: np.ndarray) -> Tuple[float, float]:
    """Chi-squared on the 2x2 [cell vs not-cell] x [incorrect vs correct] table."""
    a = int((cell_mask & incorrect_mask).sum())
    b = int((cell_mask & ~incorrect_mask).sum())
    c = int((~cell_mask & incorrect_mask).sum())
    d = int((~cell_mask & ~incorrect_mask).sum())
    table = np.array([[a, b], [c, d]])
    if table.min() < 0 or table.sum(axis=0).min() == 0 or table.sum(axis=1).min() == 0:
        return float("nan"), float("nan")
    chi2 = scipy_stats.chi2_contingency(table, correction=False)
    fisher = scipy_stats.fisher_exact(table, alternative="greater")
    return float(chi2.pvalue), float(fisher.pvalue)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="llava")
    p.add_argument("--dataset", default="vqa-v2")
    p.add_argument("--probe_type", default="mlp")
    p.add_argument(
        "--feature",
        default=None,
        help="Probe feature directory name. If unset, picks best by --selection_metric "
             "from middle-to-late-layer candidates (lm_*_layer_16).",
    )
    p.add_argument(
        "--selection_metric",
        default="auroc",
        choices=("auroc", "brier"),
        help="Metric used to pick the best middle-layer probe when --feature is unset. "
             "auroc: recomputed from summary_predictions.csv (higher is better). "
             "brier: read from results_brier.csv (lower is better).",
    )
    p.add_argument(
        "--confidence_feature",
        default="answer_gen_entropy_mean",
        help="Feature name to read from supervised dataset's X for output confidence.",
    )
    p.add_argument("--correct_threshold", type=float, default=0.5)
    p.add_argument("--strict_pct", type=float, default=0.8)
    p.add_argument(
        "--output_csv",
        default=None,
        help="Per-sample output CSV path. Defaults to mismatch_sanity_check_<model>_<dataset>.csv in cwd.",
    )
    p.add_argument(
        "--probe_results_root",
        default=DEFAULT_PROBE_RESULTS_ROOT,
    )
    p.add_argument(
        "--storage_root",
        default=storage_root(),
        help="Base storage root (e.g. /projects/prjs2014). The supervision dataset "
             "is located via SUPERVISION_LAYOUT[(model, dataset)] joined under this root.",
    )
    p.add_argument("--run_hash", default=None, help="e.g. run_7f57ad86ba (optional)")
    p.add_argument(
        "--pope_split",
        default=None,
        choices=POPE_SPLIT_CHOICES,
        help="POPE benchmark split (only valid when --dataset=pope). "
             "Selects the probe-results subdir and the matching supervision dataset.",
    )
    return p


def run(args: argparse.Namespace) -> None:
    if args.pope_split is not None and args.dataset != "pope":
        raise ValueError(
            f"--pope_split={args.pope_split!r} is only valid when --dataset=pope "
            f"(got --dataset={args.dataset!r})."
        )
    if args.dataset == "pope" and args.pope_split is None:
        raise ValueError(
            f"--dataset=pope requires --pope_split (one of {POPE_SPLIT_CHOICES}). "
            f"Probe results and supervision datasets are stored per split."
        )

    dataset_dir = os.path.join(args.probe_results_root, args.model, args.dataset)
    if args.pope_split is not None:
        dataset_dir = os.path.join(dataset_dir, args.pope_split)
    probe_dir = os.path.join(dataset_dir, args.probe_type)
    if not os.path.isdir(probe_dir):
        raise FileNotFoundError(f"Probe directory not found: {probe_dir!r}")

    if args.feature is not None:
        feature = args.feature
        selection_score = None
    else:
        feature, selection_score = pick_best_probe(
            probe_dir,
            dataset_dir,
            args.probe_type,
            MID_LAYER_PROBE_CANDIDATES,
            args.correct_threshold,
            args.selection_metric,
        )

    # Always also report AUROC + Brier for the selected probe so the printed
    # output is complete regardless of which metric drove selection.
    feature_auroc = auroc_from_predictions_csv(
        os.path.join(probe_dir, feature, "summary_predictions.csv"),
        args.correct_threshold,
    )
    feature_brier = brier_from_results_csv(
        os.path.join(dataset_dir, "results_brier.csv"),
        feature,
        args.probe_type,
    )

    sup_path = find_supervision_pt(
        args.storage_root, args.model, args.dataset, args.run_hash,
        pope_split=args.pope_split,
    )

    payload = torch.load(sup_path, map_location="cpu", weights_only=False)
    X = payload["X"].float()
    y = payload["y"].float()
    rows = payload["rows"]
    feature_names = payload["feature_names"]
    feature_dims = payload.get("feature_dims")

    test_indices = replicate_test_indices(X.shape[0])
    probe_csv = os.path.join(probe_dir, feature, "summary_predictions.csv")
    probe_df = pd.read_csv(probe_csv)
    if len(probe_df) != len(test_indices):
        raise RuntimeError(
            f"Test split mismatch: probe CSV has {len(probe_df)} rows but "
            f"replicated test split has {len(test_indices)} indices. "
            f"Check that src/probe/data.py:create_dataloaders still uses seed=42, 70/15/15."
        )

    y_test = y[test_indices].numpy()
    label_diff = np.abs(probe_df["label"].values - y_test)
    n_mismatch = (label_diff > 1e-4).sum()
    mismatch_rate = n_mismatch / len(y_test)
    if mismatch_rate > 0.005:
        raise AssertionError(
            f"Probe-CSV labels do not match y[test_indices]; the test-set order has drifted. "
            f"Mismatched elements: {n_mismatch} / {len(y_test)} ({100*mismatch_rate:.2f}%). "
            f"Check that src/probe/data.py:create_dataloaders still uses seed=42, 70/15/15."
        )
    if n_mismatch > 0:
        print(
            f"[WARNING] {n_mismatch}/{len(y_test)} label(s) in probe CSV differ from "
            f"y[test_indices] by >1e-4 (max diff {label_diff.max():.4f}). "
            f"Mismatch rate {100*mismatch_rate:.3f}% is below the 0.5% tolerance — continuing."
        )

    slice_map = build_feature_slice_map(feature_names, feature_dims, X.shape[1])
    if args.confidence_feature not in slice_map:
        raise KeyError(
            f"Confidence feature {args.confidence_feature!r} not in supervised dataset. "
            f"Available answer_gen_* features: "
            f"{[n for n in slice_map if n.startswith('answer_gen_')]}"
        )
    conf_slice = slice_map[args.confidence_feature]
    conf_block = X[:, conf_slice].numpy()
    if conf_block.shape[1] != 1:
        raise ValueError(
            f"Confidence feature {args.confidence_feature!r} has width "
            f"{conf_block.shape[1]}, expected scalar (width 1)."
        )
    conf_raw_test = conf_block[test_indices, 0]

    flip = args.confidence_feature in NEGATIVELY_ORIENTED_CONFIDENCE_FEATURES
    confidence = -conf_raw_test if flip else conf_raw_test

    probe_score = probe_df["prediction"].values.astype(float)
    correct = (y_test >= args.correct_threshold).astype(int)
    incorrect = (correct == 0)

    sample_id = [extract_sample_id(rows[i]) for i in test_indices]
    pred_answer = [rows[i].get("pred_answer", "") for i in test_indices]
    gt_answers = [rows[i].get("gt_answers", "") for i in test_indices]

    conf_pct = percentile_rank(confidence)
    probe_pct = percentile_rank(probe_score)

    hi_conf = conf_pct >= 0.5
    hi_probe = probe_pct >= 0.5
    cell = np.array([
        cell_label_for(bool(hc), bool(hp)) for hc, hp in zip(hi_conf, hi_probe)
    ])

    n_total = len(test_indices)
    base_err = float(incorrect.mean())

    confidence_label = (
        f"-{args.confidence_feature}" if flip else args.confidence_feature
    )

    auroc_str = f"{feature_auroc:.3f}" if feature_auroc is not None else "n/a"
    brier_str = f"{feature_brier:.4f}" if feature_brier is not None else "n/a"
    if args.feature is not None:
        sel_str = "manual"
    else:
        sel_str = (
            f"{args.selection_metric}={selection_score:.4f}"
            if selection_score is not None else args.selection_metric
        )
    dataset_label = (
        f"{args.dataset}/{args.pope_split}" if args.pope_split else args.dataset
    )
    print("=== 2x2 Mismatch Sanity Check ===")
    print(
        f"Model: {args.model} | Dataset: {dataset_label} | "
        f"Probe: {feature} (selected by {sel_str}) | "
        f"AUROC: {auroc_str} | Brier: {brier_str}"
    )
    print(f"Confidence signal: {confidence_label}")
    print(
        f"Total samples: {n_total} | Correct threshold: y >= "
        f"{args.correct_threshold} | Base error rate: {fmt_pct(base_err)}"
    )

    print("\n--- Median split (50th percentile) ---")
    cells = {
        "A_aligned_confident": (hi_conf & hi_probe),
        "B_overconfident": (hi_conf & ~hi_probe),
        "C_undercommitted": (~hi_conf & hi_probe),
        "D_aligned_uncertain": (~hi_conf & ~hi_probe),
    }
    cell_stats = {}
    for name, mask in cells.items():
        n = int(mask.sum())
        n_inc = int((mask & incorrect).sum())
        cell_stats[name] = dict(
            count=n,
            n_incorrect=n_inc,
            err_rate=(n_inc / n) if n else float("nan"),
            pct_of_total=n / n_total,
        )

    def cell_line(name: str, marker: str = "") -> str:
        s = cell_stats[name]
        return (
            f"{name}: {s['count']} ({fmt_pct(s['pct_of_total'])})"
            f"  err: {fmt_pct(s['err_rate'])}{marker}"
        )

    print("                          Probe >= median       |   Probe < median")
    print(
        "Output confident   "
        f"  {cell_stats['A_aligned_confident']['count']:>6} "
        f"({fmt_pct(cell_stats['A_aligned_confident']['pct_of_total'])}) "
        f"err {fmt_pct(cell_stats['A_aligned_confident']['err_rate'])}"
        f"  | {cell_stats['B_overconfident']['count']:>6} "
        f"({fmt_pct(cell_stats['B_overconfident']['pct_of_total'])}) "
        f"err {fmt_pct(cell_stats['B_overconfident']['err_rate'])}  <- TARGET"
    )
    print(
        "Output uncertain   "
        f"  {cell_stats['C_undercommitted']['count']:>6} "
        f"({fmt_pct(cell_stats['C_undercommitted']['pct_of_total'])}) "
        f"err {fmt_pct(cell_stats['C_undercommitted']['err_rate'])}"
        f"  | {cell_stats['D_aligned_uncertain']['count']:>6} "
        f"({fmt_pct(cell_stats['D_aligned_uncertain']['pct_of_total'])}) "
        f"err {fmt_pct(cell_stats['D_aligned_uncertain']['err_rate'])}"
    )

    b_mask = cells["B_overconfident"]
    chi2_p, fisher_p = chi2_cell_vs_rest(b_mask, incorrect)
    b_err = cell_stats["B_overconfident"]["err_rate"]
    ratio = (b_err / base_err) if base_err > 0 else float("nan")
    print(
        f"\nCell B error rate vs base: {ratio:.2f}x  "
        f"(chi2 p = {fmt_pval(chi2_p)}, fisher p = {fmt_pval(fisher_p)})"
    )

    print(
        f"\n--- Strict split (>= {args.strict_pct:.0%} conf, "
        f"<= {1 - args.strict_pct:.0%} probe) ---"
    )
    extreme = (conf_pct >= args.strict_pct) & (probe_pct <= 1 - args.strict_pct)
    n_ext = int(extreme.sum())
    ext_err = float(incorrect[extreme].mean()) if n_ext else float("nan")
    ext_chi2_p, ext_fisher_p = chi2_cell_vs_rest(extreme, incorrect)
    ext_ratio = (ext_err / base_err) if (n_ext and base_err > 0) else float("nan")
    print(
        f"Extreme mismatch: {n_ext} samples ({fmt_pct(n_ext / n_total)} of test set)"
    )
    print(
        f"Error rate: {fmt_pct(ext_err)} ({ext_ratio:.2f}x base rate, "
        f"chi2 p = {fmt_pval(ext_chi2_p)}, fisher p = {fmt_pval(ext_fisher_p)})"
    )

    default_csv_dataset = (
        f"{args.dataset}_{args.pope_split}" if args.pope_split else args.dataset
    )
    out_csv = args.output_csv or f"mismatch_sanity_check_{args.model}_{default_csv_dataset}.csv"
    out_df = pd.DataFrame({
        "sample_id": sample_id,
        "confidence": confidence,
        "confidence_percentile": conf_pct,
        "probe_score": probe_score,
        "probe_percentile": probe_pct,
        "cell_label": cell,
        "correct": correct,
        "y_soft": y_test,
        "pred_answer": pred_answer,
        "gt_answers": [json.dumps(a) if not isinstance(a, str) else a for a in gt_answers],
    })
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved per-sample CSV: {out_csv}")


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
