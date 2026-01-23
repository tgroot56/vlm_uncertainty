"""Training script for linear probe."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
from typing import List, Optional
import json

from probe.models import LinearProbe, BrierScoreLoss, compute_calibration_metrics
from probe.data import create_dataloaders, SupervisionDataset


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
    val_loader: DataLoader,
    criterion: BrierScoreLoss,
    device: torch.device,
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
    
    # Accuracy (using 0.5 threshold)
    binary_preds = (all_predictions > 0.5).float()
    accuracy = (binary_preds == all_labels).float().mean().item()
    
    # Calibration metrics
    calibration = compute_calibration_metrics(all_predictions, all_labels)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'ece': calibration['ece'],
        'mean_prediction': all_predictions.mean().item(),
        'mean_label': all_labels.mean().item(),
    }


def train_probe(
    data_path: str,
    feature_names: List[str],
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    train_split: float = 0.8,
    normalize: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
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
        train_split: Fraction of data for training
        normalize: Whether to normalize features
        seed: Random seed
        device: Device to train on
    """
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
    train_loader, val_loader, dataset = create_dataloaders(
        data_path=data_path,
        feature_names=feature_names,
        train_split=train_split,
        batch_size=batch_size,
        normalize=normalize,
        seed=seed,
    )
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Feature dimension: {stats['feature_dim']}")
    print(f"  Correct: {stats['num_correct']} ({stats['accuracy']:.2%})")
    print(f"  Incorrect: {stats['num_incorrect']} ({1-stats['accuracy']:.2%})")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Initialize model
    input_dim = dataset.get_feature_dim()
    model = LinearProbe(input_dim=input_dim).to(device)
    print(f"\nModel: Linear probe with {input_dim} input features")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = BrierScoreLoss()
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
        'val_ece': [],
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_ece'].append(val_metrics['ece'])
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"ECE: {val_metrics['ece']:.4f}")
        
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
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save configuration
    config = {
        'data_path': data_path,
        'feature_names': feature_names,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'train_split': train_split,
        'normalize': normalize,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'dataset_stats': stats,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel and results saved to: {output_dir}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train linear probe for correctness prediction")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to supervision dataset pickle file')
    parser.add_argument('--features', type=str, nargs='+', required=True,
                        help='Feature names to use (space-separated)')
    parser.add_argument('--output_dir', type=str, default='probe_results',
                        help='Directory to save results')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 regularization weight')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data for training')
    
    # Other arguments
    parser.add_argument('--no_normalize', action='store_true',
                        help='Do not normalize features')
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
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        normalize=not args.no_normalize,
        seed=args.seed,
        device=device,
    )


if __name__ == '__main__':
    main()
