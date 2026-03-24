# utils/tables_and_plots/dataset_size_sweep.py
"""
Dataset-size sweep experiment for probe performance.

Trains probes on subsets of increasing size and plots test Brier score.

Example:
  python -m utils.tables_and_plots.dataset_size_sweep \
    --data-path /projects/prjs2014/vqa-v2/llava-hf/llava-1.5-7b-hf/run_cc1adbca16/supervision_dataset.pt \
    --output-dir probe_results/_plots/dataset_size_sweep \
    --seed 42

Notes:
- Tests each feature separately: answer_gen_negp_max and lm_answer_mean_layer_16
- Tests both linear and mlp probes
- Fractions: 0.1, 0.3, 0.5, 0.7, 0.9, 1.0
- Creates comparison plots showing linear vs mlp for each feature
- Keeps val/test size fixed as fractions of the *full* dataset by default:
    val_split=0.15, test_split=(1-train_split-val_split)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import from your package
from src.probe.models import LinearProbe, MLPProbe, BrierScoreLoss
from src.probe.data import create_dataloaders
from src.probe.train import evaluate  # reuse your evaluate()


FEATURES = ["answer_gen_negp_max", "lm_answer_mean_layer_16"]
FRACTIONS = [0.10, 0.30, 0.50, 0.70, 0.90, 1.00]


def train_one(
    data_path: str,
    feature_names: List[str],
    model_type: str,
    fraction: float,
    seed: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    activation: str,
    hidden_dims: List[int],
    normalize: bool,
    device: torch.device,
    train_split: float,
    val_split: float,
) -> Dict[str, Any]:
    """
    Train a probe on a fraction of the dataset and return test metrics.
    """
    torch.manual_seed(seed)

    # Create loaders from your existing helper
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        data_path=data_path,
        feature_names=feature_names,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size,
        normalize=normalize,
        seed=seed,
    )

    # Subsample TRAIN ONLY to desired fraction of *full dataset used by dataloaders*.
    # We do this by taking a subset of the train subset indices.
    # This keeps val/test the same and only changes how much supervision you have.
    train_subset = train_loader.dataset  # torch.utils.data.Subset
    n_train_full = len(train_subset)
    n_train_use = max(1, int(round(fraction * n_train_full)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_train_full, generator=g)[:n_train_use].tolist()
    # Build a new Subset over the Subset: indices map into the Subset's indices list.
    # torch Subset supports nesting.
    from torch.utils.data import Subset, DataLoader
    train_loader = DataLoader(
        Subset(train_subset, perm),
        batch_size=batch_size,
        shuffle=True,
        num_workers=getattr(train_loader, "num_workers", 0),
        pin_memory=True,
    )

    input_dim = dataset.get_feature_dim()

    if model_type.lower() == "linear":
        model = LinearProbe(input_dim=input_dim).to(device)
    elif model_type.lower() == "mlp":
        model = MLPProbe(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    criterion = BrierScoreLoss(neg_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        nb = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            p = model(X)
            loss = criterion(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            nb += 1

        # Evaluate on val
        val_metrics = evaluate(model, val_loader, criterion, device)
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Load best and evaluate test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)
    out = {
        "fraction": float(fraction),
        "train_used": int(n_train_use),
        "train_full": int(n_train_full),
        "best_epoch": int(best_epoch),
        "best_val_brier": float(best_val),
        "test_brier": float(test_metrics["loss"]),
        "test_ece": float(test_metrics["ece"]),
        "mean_label": float(test_metrics["mean_label"]),
        "mean_pred": float(test_metrics["mean_prediction"]),
    }
    if "auroc" in test_metrics:
        out["test_auroc"] = float(test_metrics["auroc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default="probe_results/_plots/dataset_size_sweep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    # MLP params (ignored for linear)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--activation", type=str, default="relu")

    # Dataloader splits + normalization (same as your defaults)
    ap.add_argument("--train-split", type=float, default=0.7)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--no-normalize", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = not args.no_normalize

    # Test both probe types
    probe_types = ["linear", "mlp"]
    
    # Test each feature separately
    all_results = {}
    
    for probe_type in probe_types:
        for feature_name in FEATURES:
            results = []
            for frac in FRACTIONS:
                print("\n" + "=" * 80)
                print(f"Fraction={frac:.2f} | probe={probe_type} | feature={feature_name}")
                print("=" * 80)
                r = train_one(
                    data_path=args.data_path,
                    feature_names=[feature_name],  # Single feature
                    model_type=probe_type,
                    fraction=frac,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    activation=args.activation,
                    hidden_dims=args.hidden_dims,
                    normalize=normalize,
                    device=device,
                    train_split=args.train_split,
                    val_split=args.val_split,
                )
                print("Result:", r)
                results.append(r)
            
            # Store results
            key = f"{probe_type}_{feature_name}"
            all_results[key] = results
            
            # Save individual results
            import pandas as pd
            df = pd.DataFrame(results).sort_values("fraction")
            csv_path = outdir / f"sweep_{probe_type}__{feature_name}__brier.csv"
            df.to_csv(csv_path, index=False)
            
            json_path = outdir / f"sweep_{probe_type}__{feature_name}__brier.json"
            with open(json_path, "w") as f:
                json.dump({"feature": feature_name, "probe_type": probe_type, 
                          "fractions": FRACTIONS, "results": results}, f, indent=2)
            
            print(f"\nSaved: {csv_path}")
            print(f"Saved: {json_path}")

    # Create comparison plots for each probe type (features compared)
    import pandas as pd
    for probe_type in probe_types:
        plt.figure(figsize=(8, 5))
        
        for feature_name in FEATURES:
            key = f"{probe_type}_{feature_name}"
            df = pd.DataFrame(all_results[key]).sort_values("fraction")
            x = df["fraction"].to_numpy() * 100.0
            y = df["test_brier"].to_numpy()
            plt.plot(x, y, marker="o", label=feature_name, linewidth=2)
        
        plt.xlabel("Training set size (% of available train split)", fontsize=12)
        plt.ylabel("Test Brier score (lower is better)", fontsize=12)
        plt.title(f"Dataset-size sweep: {probe_type.upper()} probe", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        fig_path = outdir / f"sweep_comparison__{probe_type}__brier.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nSaved comparison plot: {fig_path}")

    print("\n" + "=" * 80)
    print("All experiments complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
