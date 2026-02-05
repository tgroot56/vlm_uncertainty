"""Training script for linear probe."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
from typing import List, Optional
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Dict, Any, Optional
import pandas as pd


from ..probe.models import (LinearProbe, MLPProbe, BrierScoreLoss)
from ..probe.metrics import compute_ece, compute_auroc, plot_reliability_diagram

from ..probe.data import create_dataloaders, SupervisionDataset

import hashlib

def make_feature_outdir(model_type, feature_names, base_dir="probe_results"):

    feat_str = "__".join(feature_names)
    if len(feat_str) > 120:
        h = hashlib.md5(feat_str.encode()).hexdigest()[:8]
        feat_str = feat_str[:120] + f"__{h}"
    out_dir = os.path.join(base_dir, model_type, feat_str)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir



def plot_training_history(history: dict, output_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 7))

    # Top: Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Brier Score Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Bottom: AUROC
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history["train_auroc"], label="Train AUROC", linewidth=2)
    plt.plot(epochs, history["val_auroc"], label="Val AUROC", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("Training and Validation AUROC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.0, 1.0])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {plot_path}")




@torch.no_grad()
def print_predictions(
    model: LinearProbe,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 50,
    save_path: Optional[str] = None,
):
    """
    Print predictions vs ground truth labels.
    
    Args:
        model: Trained probe model
        data_loader: DataLoader to get predictions for
        device: Device to run on
        num_samples: Number of samples to display (default: 50, set to -1 for all)
        save_path: Optional path to save predictions to file
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        predictions = model(features)
        
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
    
    # Concatenate all
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    
    if num_samples == -1:
        selected_indices = torch.arange(len(predictions))
    else:
        k_per_class = num_samples // 2

        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

        # If one class is too small, reduce k_per_class
        k_per_class = min(k_per_class, len(pos_idx), len(neg_idx))

        # Shuffle indices so you don't always see the same examples
        if len(pos_idx) > 0:
            pos_idx = pos_idx[torch.randperm(len(pos_idx))[:k_per_class]]
        if len(neg_idx) > 0:
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:k_per_class]]

        selected_indices = torch.cat([pos_idx, neg_idx])

    # ---------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PREDICTIONS VS GROUND TRUTH (balanced by GT label)")
    print("=" * 80)
    print(f"{'Index':<8} {'Prediction':<15} {'Ground Truth':<15} {'Binary Pred':<12} {'Match':<8}")
    print("-" * 80)

    for i in selected_indices.tolist():
        pred = predictions[i].item()
        label = labels[i].item()
        binary_pred = 1 if pred > 0.5 else 0
        match = "✓" if binary_pred == int(label) else "✗"
        print(f"{i:<8} {pred:<15.4f} {int(label):<15} {binary_pred:<12} {match:<8}")

    # Summary statistics (still over FULL set, not just printed subset)
    binary_preds = (predictions > 0.5).float()
    accuracy = (binary_preds == labels).float().mean().item()

    print("-" * 80)
    print(f"Total samples: {len(predictions)}")
    print(f"Mean prediction: {predictions.mean():.4f}")
    print(f"Mean label: {labels.mean():.4f}")
    print(f"Std prediction: {predictions.std():.4f}")
    print("=" * 80)

    if save_path:
        results = {
            "predictions": predictions.numpy().tolist(),
            "labels": labels.numpy().tolist(),
            "accuracy": accuracy,
        }
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nPredictions saved to: {save_path}")


def train_epoch(
    model: LinearProbe,
    train_loader: DataLoader,
    criterion: BrierScoreLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        predictions = model(features)
        loss = criterion(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device: torch.device,
    num_ece_bins: int = 10,
    return_preds: bool = False,
    save_reliability_path: Optional[str] = None,
    reliability_title: str = "Reliability diagram",
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        preds = model(features)
        loss = criterion(preds, labels)

        total_loss += loss.item()
        all_predictions.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    preds = torch.cat(all_predictions, dim=0).float()
    labs = torch.cat(all_labels, dim=0).float()

    avg_loss = total_loss / max(len(loader), 1)

    ece_dict = compute_ece(preds, labs, num_bins=num_ece_bins)
    auroc = compute_auroc(preds, labs)

    if save_reliability_path is not None:
        plot_reliability_diagram(
            ece_dict=ece_dict,
            output_path=save_reliability_path,
            title=reliability_title,
        )

    out: Dict[str, Any] = {
        "loss": float(avg_loss),
        "ece": float(ece_dict["ece"]),
        "auroc": float(auroc),
        "mean_prediction": float(preds.mean().item()),
        "mean_label": float(labs.mean().item()),
    }

    if return_preds:
        out["predictions"] = preds
        out["labels"] = labs

    return out




def train_probe(
    data_path: str,
    feature_names: List[str],
    output_dir: str,    model_type: str = 'linear',
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.2,
    activation: str = 'relu',    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    normalize: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
    print_test_predictions: bool = True,
    use_class_weights: bool = True,
    shuffle_train_labels: bool = False
):
    """
    Train a linear probe on the extracted features.
    
    Args:
        data_path: Path to supervision dataset pickle file
        feature_names: List of feature names to use
        output_dir: Directory to save model and results
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization weight
        train_split: Fraction of data for training (default: 0.7)
        val_split: Fraction of data for validation (default: 0.15, test gets remaining 0.15)
        normalize: Whether to normalize features
        seed: Random seed
        device: Device to train on
        print_test_predictions: Whether to print test set predictions
        use_class_weights: Whether to use class weighting for imbalanced data
        shuffle_train_labels: Whether to shuffle training data for baseline
    """
    print(f"\nStarting training of probe model...")
    # Set random seed
    torch.manual_seed(seed)
    
    # Create output directory
    output_dir = make_feature_outdir(model_type, feature_names, base_dir="probe_results")

    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print(f"\nLoading data from: {data_path}")
    print(f"Selected features: {feature_names}")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        data_path=data_path,
        feature_names=feature_names,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size,
        normalize=normalize,
        seed=seed,
    )

    if shuffle_train_labels:
        print("\n[CONTROL] Shuffling TRAIN labels (features unchanged)")

        train_subset = train_loader.dataset                  # torch.utils.data.Subset
        base_dataset = train_subset.dataset                  # SupervisionDataset
        train_indices = torch.as_tensor(train_subset.indices, dtype=torch.long)

        train_labels = base_dataset.y[train_indices]
        perm = torch.randperm(len(train_indices))
        base_dataset.y[train_indices] = train_labels[perm]

    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Feature dimension: {stats['feature_dim']}")
    print(f"  Correct: {stats['num_correct']} ({stats['accuracy']:.2%})")
    print(f"  Incorrect: {stats['num_incorrect']} ({1-stats['accuracy']:.2%})")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    input_dim = dataset.get_feature_dim()
    
    if model_type.lower() == 'linear':
        model = LinearProbe(input_dim=input_dim).to(device)
        print(f"\nModel: Linear probe with {input_dim} input features")
    elif model_type.lower() == 'mlp':
        if hidden_dims is None:
            hidden_dims = [256, 128]
        model = MLPProbe(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation
        ).to(device)
        print(f"\nModel: MLP probe with {input_dim} input features")
        print(f"  Hidden layers: {hidden_dims}")
        print(f"  Activation: {activation}")
        print(f"  Dropout: {dropout}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'linear' or 'mlp'")
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Calculate class weights to handle imbalance
    if use_class_weights:
        # Weight for positive class (correct predictions)
        # neg_weight = num_negative / num_positive
        num_pos = stats['num_correct']
        num_neg = stats['num_incorrect']
        neg_weight = num_pos / num_neg if num_neg > 0 else 1.0
        print(f"\nUsing class weighting:")
        print(f"  Negative weight (upweights incorrect class): {neg_weight:.4f}")
        print(f"  This upweights errors on minority class (incorrect) by {neg_weight:.2f}x")
    else:
        neg_weight = 1.0
        print(f"\nNot using class weighting (neg_weight=1.0)")
    
    # Loss and optimizer with class weighting
    criterion = BrierScoreLoss(neg_weight=neg_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("="*80)
    
    best_val_loss = float("inf")
    best_epoch = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_auroc": [],
        "val_auroc": [],
    }

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Val metrics every epoch (cheap enough)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Train AUROC only occasionally to save compute (and reuse last value)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            train_metrics = evaluate(model, train_loader, criterion, device)
            train_auroc = train_metrics["auroc"]
        else:
            train_auroc = history["train_auroc"][-1] if len(history["train_auroc"]) > 0 else float("nan")

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["train_auroc"].append(float(train_auroc))
        history["val_auroc"].append(float(val_metrics["auroc"]))

        # Print progress occasionally
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"TrainLoss: {train_loss:.4f} | "
                f"ValLoss: {val_metrics['loss']:.4f} | "
                f"ValAUROC: {val_metrics['auroc']:.4f} | "
                f"ValECE: {val_metrics['ece']:.4f}"
            )

        # Save best model by validation loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_auroc": float(val_metrics["auroc"]),
                    "val_ece": float(val_metrics["ece"]),
                    "feature_names": feature_names,
                    "input_dim": input_dim,
                    "normalize": normalize,
                    "mean": dataset.mean if normalize else None,
                    "std": dataset.std if normalize else None,
                },
                os.path.join(output_dir, "best_model.pt"),
            )

    print("=" * 80)
    print(f"Training complete! Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}")

    
    # Evaluate on test set with best model
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        return_preds=True,
        save_reliability_path=os.path.join(output_dir, "reliability_diagram.png"),
        reliability_title="Test-set reliability diagram",
    )

    # metrics_test.json (as requested)
    metrics_out = {
        "test_loss": test_metrics["loss"],
        "test_auroc": test_metrics["auroc"],
        "test_ece": test_metrics["ece"],
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
    }
    with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nTest Set Results:")
    print(f"  Loss (Brier Score): {test_metrics['loss']:.4f}")
    print(f"  AUROC:              {test_metrics['auroc']:.4f}")
    print(f"  ECE:                {test_metrics['ece']:.4f}")
    print(f"  Mean label:         {test_metrics['mean_label']:.4f}")
    print(f"  Mean prediction:    {test_metrics['mean_prediction']:.4f}")

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # training_curves.png
    plot_training_history(history, output_dir)

    # summary_predictions.csv + balanced sample

    preds = test_metrics["predictions"].numpy()
    labs = test_metrics["labels"].numpy()

    df = pd.DataFrame({"prediction": preds, "label": labs})
    df.to_csv(os.path.join(output_dir, "summary_predictions.csv"), index=False)

    df0 = df[df["label"] == 0].sort_values("prediction", ascending=False).head(50)
    df1 = df[df["label"] == 1].sort_values("prediction", ascending=True).head(50)
    df_sample = pd.concat([df0, df1], axis=0)
    df_sample.to_csv(os.path.join(output_dir, "summary_predictions_sample.csv"), index=False)

    # Optional: also save raw JSON predictions (you already have print_predictions)
    if print_test_predictions:
        print_predictions(
            model,
            test_loader,
            device,
            num_samples=50,
            save_path=os.path.join(output_dir, "test_predictions.json"),
        )

    # config.json (only keys that exist)
    config = {
        "data_path": data_path,
        "feature_names": feature_names,
        "model_type": model_type,
        "hidden_dims": hidden_dims if model_type.lower() == "mlp" else None,
        "dropout": dropout if model_type.lower() == "mlp" else None,
        "activation": activation if model_type.lower() == "mlp" else None,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_split": train_split,
        "val_split": val_split,
        "normalize": normalize,
        "seed": seed,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_auroc": test_metrics["auroc"],
        "test_ece": test_metrics["ece"],
        "dataset_stats": stats,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel and results saved to: {output_dir}")
    return model, history

    


