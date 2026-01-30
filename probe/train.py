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

from probe.models import (LinearProbe, MLPProbe, BrierScoreLoss)
from probe.metrics import (compute_calibration_metrics, compute_classification_metrics, 
        find_optimal_threshold, plot_reliability_diagram)
from probe.data import create_dataloaders, SupervisionDataset


def plot_training_history(history: dict, output_dir: str):
    """
    Plot and save training history (losses, accuracy, ECE).
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save plots
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot 1: Losses
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Brier Score Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: F1 Score
    axes[1].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Plot 3: AUROC and Balanced Accuracy
    axes[2].plot(epochs, history['val_auroc'], 'purple', label='AUROC', linewidth=2, marker='o', markersize=3)
    axes[2].plot(epochs, history['val_balanced_accuracy'], 'orange', label='Balanced Acc', linewidth=2, marker='s', markersize=3)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Score', fontsize=12)
    axes[2].set_title('AUROC & Balanced Accuracy', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    # Plot 4: ECE
    axes[3].plot(epochs, history['val_ece'], 'm-', label='Val ECE', linewidth=2)
    axes[3].set_xlabel('Epoch', fontsize=12)
    axes[3].set_ylabel('Expected Calibration Error', fontsize=12)
    axes[3].set_title('Validation ECE', fontsize=14, fontweight='bold')
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
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
    print(f"Accuracy (0.5 threshold): {accuracy:.2%}")
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
    model: LinearProbe,
    output_dir: str,
    val_loader: DataLoader,
    criterion: BrierScoreLoss,
    device: torch.device,
    find_threshold: bool = False,
    test: bool = False,
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for features, labels in val_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        predictions = model(features)
        loss = criterion(predictions, labels)
        
        total_loss += loss.item()
        all_predictions.append(predictions)
        all_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    
    # Calibration metrics
    calibration = compute_calibration_metrics(all_predictions, all_labels)

    # Plot reliability diagram
    if test:
        plot_reliability_diagram(
            calibration_metrics=calibration,
            output_path=os.path.join(output_dir, "reliability_diagram.png"),
            title="Test-set Reliability Diagram"
        )
    
    # Find optimal threshold if requested
    if find_threshold:
        optimal_thresh, best_f1, _, _ = find_optimal_threshold(
            all_predictions, all_labels, metric='f1'
        )
    else:
        optimal_thresh = 0.5

    preds = all_predictions.detach().float().cpu()
    labs  = all_labels.detach().float().cpu()
    
    # Classification metrics at optimal threshold
    clf_metrics = compute_classification_metrics(
        preds, labs, threshold=optimal_thresh
    )
    
    # Also get metrics at 0.5 threshold for comparison
    clf_metrics_05 = compute_classification_metrics(
        preds, labs, threshold=0.5
    )
    
    return {
        'loss': avg_loss,
        'ece': calibration['ece'],
        'mean_prediction': preds.mean().item(),
        'mean_label': labs.mean().item(),
        # Metrics at optimal threshold
        'optimal_threshold': optimal_thresh,
        'auroc': clf_metrics['auroc'],
        'f1': clf_metrics['f1'],
        'balanced_accuracy': clf_metrics['balanced_accuracy'],
        'precision': clf_metrics['precision'],
        'recall': clf_metrics['recall'],
        'accuracy': clf_metrics['accuracy'],
        # Metrics at 0.5 threshold for reference
        'accuracy_at_05': clf_metrics_05['accuracy'],
        'f1_at_05': clf_metrics_05['f1'],
    }


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

        # Pull labels for train indices
        train_labels = base_dataset.labels[train_indices]

        # Permute them
        perm = torch.randperm(len(train_indices))
        shuffled_train_labels = train_labels[perm]

        # Write back ONLY into the positions used by train split
        base_dataset.labels[train_indices] = shuffled_train_labels
    
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
    
    best_val_loss = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auroc': [],
        'val_balanced_accuracy': [],
        'val_ece': [],
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, output_dir=output_dir, val_loader=val_loader, criterion=criterion, device=device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_balanced_accuracy'].append(val_metrics['balanced_accuracy'])
        history['val_ece'].append(val_metrics['ece'])
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"ValLoss: {val_metrics['loss']:.4f} | "
                  f"F1: {val_metrics['f1']:.3f} | "
                  f"AUROC: {val_metrics['auroc']:.3f} | "
                  f"BalAcc: {val_metrics['balanced_accuracy']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'feature_names': feature_names,
                'input_dim': input_dim,
                'normalize': normalize,
                'mean': dataset.mean if normalize else None,
                'std': dataset.std if normalize else None,
            }, os.path.join(output_dir, 'best_model.pt'))
    
    print("="*80)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Evaluate on test set with best model
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, output_dir=output_dir, val_loader=test_loader, criterion=criterion, device=device, find_threshold=True, test=True)
    
    print(f"\nTest Set Results:")
    print(f"  Loss (Brier Score): {test_metrics['loss']:.4f}")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  ECE: {test_metrics['ece']:.4f}")
    print(f"\n  Metrics at threshold=0.5:")
    print(f"    Accuracy: {test_metrics['accuracy_at_05']:.4f}")
    print(f"    F1 Score: {test_metrics['f1_at_05']:.4f}")
    print(f"\n  Metrics at optimal threshold={test_metrics['optimal_threshold']:.3f}:")
    print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"    F1 Score: {test_metrics['f1']:.4f}")
    print(f"    Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall: {test_metrics['recall']:.4f}")
    
    print(f"\n  Baseline (always predict 1): {test_metrics['mean_label']:.4f}")
    
    # Print predictions on test set
    if print_test_predictions:
        print_predictions(
            model, 
            test_loader, 
            device, 
            num_samples=50,
            save_path=os.path.join(output_dir, 'test_predictions.json')
        )
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot and save training curves
    plot_training_history(history, output_dir)
    
    # Save configuration
    config = {
        'data_path': data_path,
        'feature_names': feature_names,
        'model_type': model_type,
        'hidden_dims': hidden_dims if model_type.lower() == 'mlp' else None,
        'dropout': dropout if model_type.lower() == 'mlp' else None,
        'activation': activation if model_type.lower() == 'mlp' else None,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'train_split': train_split,
        'val_split': val_split,
        'normalize': normalize,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_ece': test_metrics['ece'],
        'dataset_stats': stats,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel and results saved to: {output_dir}")
    
    return model, history


def main():
    print("In main of probe.train.py")
    parser = argparse.ArgumentParser(description="Train linear probe for correctness prediction")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to supervision dataset pickle file')
    parser.add_argument('--features', type=str, nargs='+', required=True,
                        help='Feature names to use (space-separated)')
    parser.add_argument('--output_dir', type=str, default='probe_results',
                        help='Directory to save results')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='linear', choices=['linear', 'mlp'],
                        help='Model type: linear or mlp (default: linear)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None,
                        help='Hidden layer dimensions for MLP (e.g., 256 128)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for MLP (default: 0.2)')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu', 'elu'],
                        help='Activation function for MLP (default: relu)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 regularization weight')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Fraction of data for training (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data for validation (default: 0.15, test gets remaining)')
    
    # Other arguments
    parser.add_argument('--shuffle_train_labels',
                        action='store_true',
                        help='Shuffle labels in training set (control experiment)')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Do not normalize features')
    parser.add_argument('--no_print_test', action='store_true',
                        help='Disable printing test predictions')
    parser.add_argument('--no_class_weights', action='store_true',
                        help='Disable class weighting for imbalanced data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Parse device
    if args.device:
        device = torch.device(args.device)
    else:
        device = None
    
    # Train probe
    train_probe(
        data_path=args.data_path,
        feature_names=args.features,
        output_dir=args.output_dir,
        model_type=args.model_type,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        activation=args.activation,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        val_split=args.val_split,
        normalize=not args.no_normalize,
        seed=args.seed,
        device=device,
        print_test_predictions=not args.no_print_test,
        use_class_weights=not args.no_class_weights,
        shuffle_train_labels=args.shuffle_train_labels,
    )


if __name__ == '__main__':
    main()
