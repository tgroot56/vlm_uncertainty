"""
Evaluate a trained probe checkpoint on a (potentially different) dataset.

Example:
python -m src.probe.eval_probe \
  --model-path probe_results/imagenet-r/16mc-clip_answertotext/mlp/answer_gen_entropy_max/best_model.pt \
  --data-path  supervision_data/vqa-v2.pkl \
  --output-base probe_results_cross

Or with dataset-name (optional convenience):
python -m src.probe.eval_probe \
  --model-path .../best_model.pt \
  --dataset-name vqa-v2 \
  --data-root supervision_data \
  --output-base probe_results_cross
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import pandas as pd

from .train import evaluate, print_predictions, make_feature_outdir
from ..probe.models import LinearProbe, MLPProbe, BrierScoreLoss
from ..probe.data import create_dataloaders


class InputNormWrapper(torch.nn.Module):
    """Wrap a probe so evaluate() can be reused unchanged, while applying fixed normalization."""
    def __init__(self, base_model: torch.nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / (self.std + 1e-12)
        return self.base_model(x)


def _infer_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_adjacent_config(model_path: str) -> Optional[Dict[str, Any]]:
    """Try to load config.json from the same folder as best_model.pt."""
    cfg_path = Path(model_path).resolve().parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    return None


def _resolve_data_path(dataset_name: Optional[str], data_path: Optional[str], data_root: Optional[str]) -> str:
    if data_path:
        return data_path
    if not dataset_name:
        raise ValueError("Provide either --data-path or --dataset-name.")
    root = data_root or "."
    # Adjust extension here if you use .pt / .pkl consistently
    candidate = Path(root) / f"{dataset_name}.pkl"
    if not candidate.exists():
        # fallback: try .pt
        candidate_pt = Path(root) / f"{dataset_name}.pt"
        if candidate_pt.exists():
            return str(candidate_pt)
        raise FileNotFoundError(f"Could not find dataset at {candidate} (or .pt). Provide --data-path explicitly.")
    return str(candidate)


def evaluate_checkpoint_on_dataset(
    model_path: str,
    data_path: str,
    output_base: str = "probe_results_cross",
    dataset_name: Optional[str] = None,
    seed: int = 42,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    device: Optional[str] = None,
    print_test_predictions: bool = True,
) -> Dict[str, Any]:
    """
    Load a trained probe from `model_path`, build dataloaders for `data_path` using the SAME feature_names,
    evaluate on the target test split, and save outputs similarly to train.py.

    Returns the test metrics dict (loss/ece/auroc/means, + preds/labels if requested).
    """
    device_t = _infer_device(device)

    ckpt = torch.load(model_path, weights_only=False)
    feature_names = ckpt.get("feature_names", None)
    input_dim = ckpt.get("input_dim", None)
    normalize = bool(ckpt.get("normalize", False))
    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)

    if feature_names is None or input_dim is None:
        raise ValueError("Checkpoint is missing 'feature_names' or 'input_dim'. Re-save checkpoints with these fields.")

    # Prefer config.json (adjacent) to reconstruct MLP settings reliably
    cfg = _load_adjacent_config(model_path) or {}
    model_type = (cfg.get("model_type") or "linear").lower()

    # Build the same architecture that was trained
    if model_type == "linear":
        model = LinearProbe(input_dim=input_dim).to(device_t)
    elif model_type == "mlp":
        hidden_dims = cfg.get("hidden_dims", [256, 128])
        dropout = cfg.get("dropout", 0.2)
        activation = cfg.get("activation", "relu")
        model = MLPProbe(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        ).to(device_t)
    else:
        raise ValueError(f"Unknown model_type in config: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])

    # Create target dataloaders with the SAME feature set.
    # Important: we set normalize=False here and (optionally) wrap the model with training-set mean/std.
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        data_path=data_path,
        feature_names=feature_names,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size,
        normalize=False,
        seed=seed,
    )

    # Wrap model to apply the *source* normalization parameters (from training checkpoint)
    # so cross-dataset eval is consistent with how the probe was trained.
    model_for_eval = model
    if normalize:
        if mean is None or std is None:
            raise ValueError("Checkpoint says normalize=True but mean/std are missing.")
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device_t)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=device_t)
        model_for_eval = InputNormWrapper(model, mean_t, std_t).to(device_t)

    # Loss: use unweighted Brier by default.
    # (If you want to reproduce weighted-loss exactly, store neg_weight in checkpoint/config too.)
    criterion = BrierScoreLoss(neg_weight=1.0)

    # Output dir structure: output_base / <target_dataset> / <model_type> / <feature_hash_dir>
    # You can also include source-dataset in the path if you want; here we keep it simple.
    tgt_name = dataset_name or Path(data_path).stem
    out_root = os.path.join(output_base, tgt_name)
    output_dir = make_feature_outdir(model_type, feature_names, base_dir=out_root)

    os.makedirs(output_dir, exist_ok=True)

    # Evaluate on target test split
    test_metrics = evaluate(
        model_for_eval,
        test_loader,
        criterion,
        device_t,
        return_preds=True,
        save_reliability_path=os.path.join(output_dir, "reliability_diagram.png"),
        reliability_title=f"Reliability diagram ({tgt_name} test)",
    )

    # Save metrics_test.json (same style)
    metrics_out = {
        "test_loss": float(test_metrics["loss"]),
        "test_auroc": float(test_metrics["auroc"]) if "auroc" in test_metrics else None,
        "test_ece": float(test_metrics["ece"]),
        "mean_label": float(test_metrics["mean_label"]),
        "mean_prediction": float(test_metrics["mean_prediction"]),
        "model_path": str(model_path),
        "data_path": str(data_path),
        "target_dataset": tgt_name,
        "feature_names": feature_names,
        "model_type": model_type,
        "normalize_applied": bool(normalize),
    }
    with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Save summary_predictions.csv + sample csv (same style)
    preds = test_metrics["predictions"].numpy()
    labs = test_metrics["labels"].numpy()

    df = pd.DataFrame({"prediction": preds, "label": labs})
    df.to_csv(os.path.join(output_dir, "summary_predictions.csv"), index=False)

    df0 = df[df["label"] == 0].sort_values("prediction", ascending=False).head(50)
    df1 = df[df["label"] == 1].sort_values("prediction", ascending=True).head(50)
    df_sample = pd.concat([df0, df1], axis=0)
    df_sample.to_csv(os.path.join(output_dir, "summary_predictions_sample.csv"), index=False)

    # Optional: raw JSON preds (reuses your function)
    if print_test_predictions:
        print_predictions(
            model_for_eval,
            test_loader,
            device_t,
            num_samples=50,
            save_path=os.path.join(output_dir, "test_predictions.json"),
        )

    # Save eval config
    eval_cfg = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "target_dataset": tgt_name,
        "seed": seed,
        "batch_size": batch_size,
        "train_split": train_split,
        "val_split": val_split,
        "feature_names": feature_names,
        "model_type": model_type,
        "normalize_applied": bool(normalize),
        "source_norm_from_checkpoint": bool(normalize),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(eval_cfg, f, indent=2)

    print(f"\nCross-dataset evaluation saved to: {output_dir}")
    print(f"  Loss: {metrics_out['test_loss']:.4f} | ECE: {metrics_out['test_ece']:.4f} | AUROC: {metrics_out['test_auroc']}")
    return test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True, help="Path to best_model.pt")
    ap.add_argument("--data-path", type=str, default=None, help="Path to target supervision dataset file")
    ap.add_argument("--dataset-name", type=str, default=None, help="Target dataset name (used with --data-root)")
    ap.add_argument("--data-root", type=str, default=None, help="Root folder containing <dataset-name>.pkl (or .pt)")
    ap.add_argument("--output-base", type=str, default="probe_results_cross")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-split", type=float, default=0.7)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-print", action="store_true", help="Disable saving test_predictions.json")
    args = ap.parse_args()

    data_path = _resolve_data_path(args.dataset_name, args.data_path, args.data_root)

    evaluate_checkpoint_on_dataset(
        model_path=args.model_path,
        data_path=data_path,
        output_base=args.output_base,
        dataset_name=args.dataset_name,
        seed=args.seed,
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        device=args.device,
        print_test_predictions=not args.no_print,
    )


if __name__ == "__main__":
    main()
