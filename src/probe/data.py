"""Data loading and preprocessing for probe training."""

import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np


class SupervisionDataset(Dataset):
    def __init__(self, data_path: str, feature_names: list[str], normalize: bool = True):
        payload = torch.load(data_path, map_location="cpu", weights_only=False)

        X_all = payload["X"].float()          # [N, D_total]
        y = payload["y"].float()              # [N]
        all_names = payload["feature_names"]  # length = number of feature blocks (24)

        # ---- infer per-block dimensions (works for your current feature set) ----
        def infer_dim(name: str) -> int:
            if name.startswith("vision_"):
                return 1024
            if name.startswith("lm_"):
                return 4096
            if name.startswith("answer_gen_"):
                return 1
            raise ValueError(f"Unknown feature name pattern: {name}")

        # Build name -> slice over X_all columns, in the order blocks were concatenated
        offset = 0
        name_to_slice = {}
        for n in all_names:
            d = infer_dim(n)
            name_to_slice[n] = slice(offset, offset + d)
            offset += d

        # Sanity: does this reconstruction match X_all width?
        if offset != X_all.shape[1]:
            raise ValueError(
                f"Feature dim inference mismatch: inferred total {offset} "
                f"but X has {X_all.shape[1]} columns. "
                f"Fix infer_dim() or store feature_dims in the dataset."
            )

        # Validate requested feature names
        missing = [n for n in feature_names if n not in name_to_slice]
        if missing:
            raise ValueError(
                f"Missing features: {missing}\n"
                f"Available: {list(name_to_slice.keys())}"
            )

        # Select blocks and concatenate into [N, D_selected]
        X = torch.cat([X_all[:, name_to_slice[n]] for n in feature_names], dim=1)

        # Compute stats for logging + optional normalization
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0).clamp_min(1e-6)

        if normalize:
            X = (X - self.mean) / self.std

        self.X = X
        self.y = y
        self.selected_feature_names = feature_names
        self.all_feature_names = all_names

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


    def get_feature_dim(self) -> int:
        return self.X.shape[1]

    def get_statistics(self) -> Dict:
        return {
            "num_samples": len(self),
            "feature_dim": self.get_feature_dim(),
            "num_correct": (self.y == 1.0).sum().item(),
            "num_incorrect": (self.y == 0.0).sum().item(),
            "accuracy": self.y.mean().item(),
        }



def create_dataloaders(
    data_path: str,
    feature_names: List[str],
    train_split: float = 0.7,
    val_split: float = 0.15,
    batch_size: int = 32,
    normalize: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SupervisionDataset]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_path: Path to the supervision dataset
        feature_names: List of feature names to use
        train_split: Fraction of data to use for training (default: 0.7)
        val_split: Fraction of data to use for validation (default: 0.15, test gets remaining 0.15)
        batch_size: Batch size for dataloaders
        normalize: Whether to normalize features
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset)
    """
    # Load full dataset
    dataset = SupervisionDataset(data_path, feature_names, normalize=normalize)
    
    # Split into train, validation, and test
    num_samples = len(dataset)
    num_train = int(num_samples * train_split)
    num_val = int(num_samples * val_split)
    num_test = num_samples - num_train - num_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_val, num_test], generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, dataset
