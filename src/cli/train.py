"""
CLI entrypoint for training a probe model on a supervised UQ dataset.

End-to-end pipeline:
- Train probe model
- Evaluate probe model
- Save best checkpoint + plots + metrics
"""

import argparse
from ..pipelines.train_probe import run_train_probe

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a probe model on a supervised UQ dataset"
    )

    # ---- Core identifiers ----
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
   
    # )
    return parser.parse_args()



def main():
    print("Starting supervised UQ dataset generation pipeline...")
    args = parse_args()

    run_train_probe(args)

if __name__ == "__main__":
    main()

