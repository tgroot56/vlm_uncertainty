# """Evaluate a trained probe and print predictions on test set."""

# import torch
# import torch.utils.data
# from torch.utils.data import DataLoader
# import argparse
# import os
# import json
# from pathlib import Path

# from probe.models import LinearProbe, BrierScoreLoss
# from probe.data import create_dataloaders, SupervisionDataset
# from probe.train import evaluate, print_predictions


# def load_model(checkpoint_path: str, device: torch.device) -> LinearProbe:
#     """Load a trained probe model from checkpoint."""
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     # Create model
#     model = LinearProbe(input_dim=checkpoint['input_dim']).to(device)
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     return model, checkpoint


# def main():
#     parser = argparse.ArgumentParser(description="Evaluate trained probe on test set")
#     parser.add_argument('--model_dir', type=str, default='probe_results',
#                         help='Directory containing trained model')
#     parser.add_argument('--num_samples', type=int, default=50,
#                         help='Number of predictions to display (-1 for all)')
#     parser.add_argument('--device', type=str, default=None,
#                         help='Device to use (cuda/cpu)')
    
#     args = parser.parse_args()
    
#     # Setup device
#     if args.device:
#         device = torch.device(args.device)
#     else:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Load config
#     config_path = os.path.join(args.model_dir, 'config.json')
#     with open(config_path, 'r') as f:
#         config = json.load(f)
    
#     print(f"\nModel configuration:")
#     print(f"  Features: {config['feature_names']}")
#     print(f"  Best epoch: {config['best_epoch']}")
#     print(f"  Best val loss: {config['best_val_loss']:.4f}")
    
#     # Load model
#     checkpoint_path = os.path.join(args.model_dir, 'best_model.pt')
#     model, checkpoint = load_model(checkpoint_path, device)
#     print(f"\nModel loaded from: {checkpoint_path}")
    
#     # Create dataloaders
#     print(f"\nLoading data from: {config['data_path']}")
    
#     # Check if data_path needs updating
#     data_path = config['data_path']
#     if not os.path.exists(data_path):
#         # Try current directory
#         if os.path.exists(os.path.basename(data_path)):
#             data_path = os.path.basename(data_path)
#         else:
#             print(f"WARNING: Could not find data file: {data_path}")
#             return
    
#     # Get splits from config (use old default if not present)
#     train_split = config.get('train_split', 0.8)
#     val_split = config.get('val_split', None)
    
#     if val_split is None:
#         # Old style: only train/val split
#         print("Note: Using old 2-way split (train/val). Retraining with new code will create a separate test set.")
        
#         # Load dataset
#         dataset = SupervisionDataset(
#             data_path=data_path,
#             feature_names=config['feature_names'],
#             normalize=config['normalize']
#         )
        
#         # Apply same normalization as training
#         if config['normalize'] and 'mean' in checkpoint and checkpoint['mean'] is not None:
#             dataset.mean = checkpoint['mean'].to(device)
#             dataset.std = checkpoint['std'].to(device)
        
#         # Split into train and val (using same seed)
#         num_samples = len(dataset)
#         num_train = int(num_samples * train_split)
#         num_val = num_samples - num_train
        
#         generator = torch.Generator().manual_seed(config['seed'])
#         train_dataset, val_dataset = torch.utils.data.random_split(
#             dataset, [num_train, num_val], generator=generator
#         )
        
#         test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#         print(f"Using validation set as test set ({len(val_dataset)} samples)")
        
#     else:
#         # New style: train/val/test split
#         train_loader, val_loader, test_loader, dataset = create_dataloaders(
#             data_path=data_path,
#             feature_names=config['feature_names'],
#             train_split=train_split,
#             val_split=val_split,
#             batch_size=config['batch_size'],
#             normalize=config['normalize'],
#             seed=config['seed'],
#         )
#         print(f"Test set: {len(test_loader.dataset)} samples")
    
#     # Evaluate and print predictions
#     criterion = BrierScoreLoss()
    
#     print("\nEvaluating on test set...")
#     test_metrics = evaluate(model, test_loader, criterion, device)
    
#     print(f"\nTest Set Metrics:")
#     print(f"  Loss (Brier Score): {test_metrics['loss']:.4f}")
#     print(f"  Accuracy (0.5 threshold): {test_metrics['accuracy']:.2%}")
#     print(f"  ECE: {test_metrics['ece']:.4f}")
#     print(f"  Mean prediction: {test_metrics['mean_prediction']:.4f}")
#     print(f"  Mean label: {test_metrics['mean_label']:.4f}")
    
#     # Print predictions
#     print_predictions(
#         model,
#         test_loader,
#         device,
#         num_samples=args.num_samples,
#         save_path=os.path.join(args.model_dir, 'test_predictions.json')
#     )


# if __name__ == '__main__':
#     main()
