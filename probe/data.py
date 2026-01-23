"""Data loading and preprocessing for probe training."""

import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np


class SupervisionDataset(Dataset):
    """Dataset for loading extracted features and correctness labels."""
    
    AVAILABLE_FEATURES = [
        'vision_middle_layer_features',
        'vision_final_layer_features',
        'lm_middle_visual_features',
        'lm_final_visual_features',
        'lm_middle_prompt_features',
        'lm_final_prompt_features',
        'lm_middle_answer_features',
        'lm_final_answer_features',
    ]
    
    def __init__(
        self,
        data_path: str,
        feature_names: List[str],
        normalize: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the supervision dataset pickle file
            feature_names: List of feature names to use (e.g., ['vision_final_layer_features'])
            normalize: Whether to normalize features (z-score normalization)
        """
        # Load data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Validate feature names
        for name in feature_names:
            if name not in self.AVAILABLE_FEATURES:
                raise ValueError(f"Unknown feature: {name}. Available: {self.AVAILABLE_FEATURES}")
        
        self.feature_names = feature_names
        self.normalize = normalize
        
        # Extract features and labels
        self.features, self.labels = self._prepare_features()
        
        # Compute normalization statistics if needed
        if self.normalize:
            self.mean = self.features.mean(dim=0)
            self.std = self.features.std(dim=0) + 1e-8  # Add small epsilon to avoid division by zero
            self.features = (self.features - self.mean) / self.std
    
    def _prepare_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and concatenate specified features from the dataset."""
        all_features = []
        all_labels = []
        
        for sample in self.data:
            # Concatenate selected features
            feature_list = []
            for feat_name in self.feature_names:
                feat = sample.get(feat_name)
                if feat is None:
                    raise ValueError(f"Feature {feat_name} not found in sample {sample['idx']}")
                
                # Flatten if needed (should be shape [1, hidden_dim] or [hidden_dim])
                if feat.dim() == 2:
                    feat = feat.squeeze(0)
                
                feature_list.append(feat)
            
            # Concatenate all selected features
            combined_features = torch.cat(feature_list, dim=0)
            all_features.append(combined_features)
            
            # Extract label (1 if correct, 0 if incorrect)
            label = 1.0 if sample['is_correct'] else 0.0
            all_labels.append(label)
        
        # Stack into tensors
        features = torch.stack(all_features)  # Shape: [N, total_feature_dim]
        labels = torch.tensor(all_labels, dtype=torch.float32)  # Shape: [N]
        
        return features, labels
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
    
    def get_feature_dim(self) -> int:
        """Return the total dimension of concatenated features."""
        return self.features.shape[1]
    
    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        return {
            'num_samples': len(self),
            'feature_dim': self.get_feature_dim(),
            'num_correct': (self.labels == 1.0).sum().item(),
            'num_incorrect': (self.labels == 0.0).sum().item(),
            'accuracy': self.labels.mean().item(),
        }


def create_dataloaders(
    data_path: str,
    feature_names: List[str],
    train_split: float = 0.8,
    batch_size: int = 32,
    normalize: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_path: Path to the supervision dataset
        feature_names: List of feature names to use
        train_split: Fraction of data to use for training
        batch_size: Batch size for dataloaders
        normalize: Whether to normalize features
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load full dataset
    dataset = SupervisionDataset(data_path, feature_names, normalize=normalize)
    
    # Split into train and validation
    num_samples = len(dataset)
    num_train = int(num_samples * train_split)
    num_val = num_samples - num_train
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_val], generator=generator
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
    
    return train_loader, val_loader, dataset
