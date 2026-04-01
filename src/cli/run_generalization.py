"""
CLI entrypoint for the cross-dataset generalization experiment.
"""

from __future__ import annotations

import argparse
import torch

from ..generalization.build_dataset import (
    DEFAULT_FEATURE_NAME,
    DEFAULT_FEATURE_NAMES,
    DEFAULT_MAIN_PROBE_SPLIT_SEED,
    DEFAULT_MAIN_PROBE_TRAIN_FRACTION,
    DEFAULT_MAIN_PROBE_VAL_FRACTION,
    DEFAULT_MAX_PER_DATASET,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_TRAIN_FRACTION,
    DEFAULT_SOURCE_VAL_FRACTION,
    build_generalization_dataset,
    canonicalize_model_name,
)
from ..generalization.paths import generalization_dataset_path
from ..generalization.train_leave_one_out import train_leave_one_out
from ..generalization.train_leave_one_out import train_all_datasets
from ..generalization.train_leave_one_out import train_model_generalization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and run leave-one-dataset-out probe generalization experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-dataset",
        help="Build a single combined generalization supervision dataset for one model.",
    )
    build_parser.add_argument("--model_name", type=str, required=True, help="Model name: qwen or llava.")
    build_parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    build_parser.add_argument("--feature_name", type=str, default=None)
    build_parser.add_argument("--feature_names", type=str, nargs="+", default=None)
    build_parser.add_argument("--max_per_dataset", type=int, default=DEFAULT_MAX_PER_DATASET)
    build_parser.add_argument("--sampling_seed", type=int, default=42)
    build_parser.add_argument("--split_seed", type=int, default=42)
    build_parser.add_argument("--binarize_threshold", type=float, default=1.0)
    build_parser.add_argument("--train_fraction", type=float, default=DEFAULT_SOURCE_TRAIN_FRACTION)
    build_parser.add_argument("--val_fraction", type=float, default=DEFAULT_SOURCE_VAL_FRACTION)
    build_parser.add_argument("--main_probe_split_seed", type=int, default=DEFAULT_MAIN_PROBE_SPLIT_SEED)
    build_parser.add_argument("--main_probe_train_fraction", type=float, default=DEFAULT_MAIN_PROBE_TRAIN_FRACTION)
    build_parser.add_argument("--main_probe_val_fraction", type=float, default=DEFAULT_MAIN_PROBE_VAL_FRACTION)
    build_parser.add_argument("--vqa_source_path", type=str, default=None)
    build_parser.add_argument("--coco_source_path", type=str, default=None)
    build_parser.add_argument("--imagenet_r_source_path", type=str, default=None)
    build_parser.add_argument("--pope_source_path", type=str, default=None)

    train_parser = subparsers.add_parser(
        "train-leave-one-out",
        help="Train and evaluate leave-one-out, shuffled-label, and in-domain probes.",
    )
    train_parser.add_argument("--model_name", type=str, required=True, help="Model name: qwen or llava.")
    train_parser.add_argument("--held_out_dataset", type=str, required=True)
    train_parser.add_argument("--data_path", type=str, default=None)
    train_parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    train_parser.add_argument("--feature_name", type=str, default=DEFAULT_FEATURE_NAME)
    train_parser.add_argument("--model_type", type=str, default="mlp", choices=["linear", "mlp"])
    train_parser.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "elu"])
    train_parser.add_argument("--num_epochs", type=int, default=100)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--learning_rate", type=float, default=0.01)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--seeds", type=int, nargs="+", default=None)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--no_normalize", action="store_true")
    train_parser.add_argument("--no_print_test", action="store_true")
    train_parser.add_argument("--use_class_weights", action="store_true")

    train_all_parser = subparsers.add_parser(
        "train-all-datasets",
        help="Train one probe on all datasets and evaluate it on every dataset test split.",
    )
    train_all_parser.add_argument("--model_name", type=str, required=True, help="Model name: qwen or llava.")
    train_all_parser.add_argument("--data_path", type=str, default=None)
    train_all_parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    train_all_parser.add_argument("--feature_name", type=str, default=DEFAULT_FEATURE_NAME)
    train_all_parser.add_argument("--model_type", type=str, default="mlp", choices=["linear", "mlp"])
    train_all_parser.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    train_all_parser.add_argument("--dropout", type=float, default=0.2)
    train_all_parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "elu"])
    train_all_parser.add_argument("--num_epochs", type=int, default=100)
    train_all_parser.add_argument("--batch_size", type=int, default=32)
    train_all_parser.add_argument("--learning_rate", type=float, default=0.01)
    train_all_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_all_parser.add_argument("--seed", type=int, default=42)
    train_all_parser.add_argument("--seeds", type=int, nargs="+", default=None)
    train_all_parser.add_argument("--device", type=str, default=None)
    train_all_parser.add_argument("--no_normalize", action="store_true")
    train_all_parser.add_argument("--no_print_test", action="store_true")
    train_all_parser.add_argument("--use_class_weights", action="store_true")
    train_all_parser.add_argument("--no_shuffled_label_control", action="store_true")

    model_gen_parser = subparsers.add_parser(
        "train-model-gen",
        help="Train on one model's dataset split and evaluate on the other model's test split for the same dataset.",
    )
    model_gen_parser.add_argument("--source_model_name", type=str, required=True, help="Source model: qwen or llava.")
    model_gen_parser.add_argument("--target_model_name", type=str, required=True, help="Target model: qwen or llava.")
    model_gen_parser.add_argument("--dataset_name", type=str, required=True)
    model_gen_parser.add_argument("--source_data_path", type=str, default=None)
    model_gen_parser.add_argument("--target_data_path", type=str, default=None)
    model_gen_parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    model_gen_parser.add_argument("--feature_name", type=str, default=DEFAULT_FEATURE_NAME)
    model_gen_parser.add_argument("--model_type", type=str, default="mlp", choices=["linear", "mlp"])
    model_gen_parser.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    model_gen_parser.add_argument("--dropout", type=float, default=0.2)
    model_gen_parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "elu"])
    model_gen_parser.add_argument("--num_epochs", type=int, default=100)
    model_gen_parser.add_argument("--batch_size", type=int, default=32)
    model_gen_parser.add_argument("--learning_rate", type=float, default=0.01)
    model_gen_parser.add_argument("--weight_decay", type=float, default=1e-4)
    model_gen_parser.add_argument("--seed", type=int, default=42)
    model_gen_parser.add_argument("--seeds", type=int, nargs="+", default=None)
    model_gen_parser.add_argument("--device", type=str, default=None)
    model_gen_parser.add_argument("--no_normalize", action="store_true")
    model_gen_parser.add_argument("--no_print_test", action="store_true")
    model_gen_parser.add_argument("--use_class_weights", action="store_true")
    model_gen_parser.add_argument("--no_shuffled_label_control", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "build-dataset":
        build_generalization_dataset(
            model_name=args.model_name,
            output_root=args.output_root,
            feature_name=args.feature_name or DEFAULT_FEATURE_NAME,
            feature_names=(
                args.feature_names
                if args.feature_names is not None
                else ([args.feature_name] if args.feature_name is not None else list(DEFAULT_FEATURE_NAMES))
            ),
            max_per_dataset=args.max_per_dataset,
            sampling_seed=args.sampling_seed,
            split_seed=args.split_seed,
            binarize_threshold=args.binarize_threshold,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            main_probe_split_seed=args.main_probe_split_seed,
            main_probe_train_fraction=args.main_probe_train_fraction,
            main_probe_val_fraction=args.main_probe_val_fraction,
            source_overrides={
                "vqa-v2": args.vqa_source_path,
                "coco-qa-vi": args.coco_source_path,
                "imagenet-r": args.imagenet_r_source_path,
                "pope/random": args.pope_source_path,
            },
        )
        return

    device = torch.device(args.device) if args.device else None

    if args.command == "train-leave-one-out":
        data_path = args.data_path
        if data_path is None:
            canonical_model_name = canonicalize_model_name(args.model_name)
            data_path = str(generalization_dataset_path(args.output_root, canonical_model_name))
        train_leave_one_out(
            data_path=data_path,
            model_name=args.model_name,
            held_out_dataset=args.held_out_dataset,
            output_root=args.output_root,
            model_type=args.model_type,
            feature_name=args.feature_name,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            activation=args.activation,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            normalize=not args.no_normalize,
            seed=args.seed,
            seeds=args.seeds,
            device=device,
            print_test_predictions=not args.no_print_test,
            use_class_weights=args.use_class_weights,
        )
        return

    if args.command == "train-all-datasets":
        data_path = args.data_path
        if data_path is None:
            canonical_model_name = canonicalize_model_name(args.model_name)
            data_path = str(generalization_dataset_path(args.output_root, canonical_model_name))
        train_all_datasets(
            data_path=data_path,
            model_name=args.model_name,
            output_root=args.output_root,
            model_type=args.model_type,
            feature_name=args.feature_name,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            activation=args.activation,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            normalize=not args.no_normalize,
            seed=args.seed,
            seeds=args.seeds,
            device=device,
            print_test_predictions=not args.no_print_test,
            use_class_weights=args.use_class_weights,
            include_shuffled_label_control=not args.no_shuffled_label_control,
        )
        return

    source_data_path = args.source_data_path
    if source_data_path is None:
        canonical_source_model = canonicalize_model_name(args.source_model_name)
        source_data_path = str(generalization_dataset_path(args.output_root, canonical_source_model))

    target_data_path = args.target_data_path
    if target_data_path is None:
        canonical_target_model = canonicalize_model_name(args.target_model_name)
        target_data_path = str(generalization_dataset_path(args.output_root, canonical_target_model))

    train_model_generalization(
        source_data_path=source_data_path,
        target_data_path=target_data_path,
        source_model_name=args.source_model_name,
        target_model_name=args.target_model_name,
        dataset_name=args.dataset_name,
        output_root=args.output_root,
        model_type=args.model_type,
        feature_name=args.feature_name,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        activation=args.activation,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        normalize=not args.no_normalize,
        seed=args.seed,
        seeds=args.seeds,
        device=device,
        print_test_predictions=not args.no_print_test,
        use_class_weights=args.use_class_weights,
        include_shuffled_label_control=not args.no_shuffled_label_control,
    )


if __name__ == "__main__":
    main()
