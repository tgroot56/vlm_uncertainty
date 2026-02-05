import torch
from src.probe.train import train_probe


def run_train_probe(args):
    # Parse device
    device = torch.device(args.device) if args.device else None

    # Run training + evaluation
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
