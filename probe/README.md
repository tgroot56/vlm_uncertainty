# Linear Probe Training Pipeline

This module provides a complete pipeline for training linear probes to predict model correctness from extracted features.

## Overview

The pipeline consists of:
- **Data loading**: Flexible feature selection from extracted supervision dataset
- **Model**: Simple linear probe with sigmoid activation
- **Loss**: Brier score for calibrated probability estimates
- **Evaluation**: Accuracy, Brier score, and Expected Calibration Error (ECE)

## Structure

```
probe/
├── __init__.py
├── data.py        # Dataset loader with feature selection
├── models.py      # Linear probe and Brier score loss
└── train.py       # Training script
```

## Available Features

The following features can be used (extracted from the supervision dataset):

- `vision_middle_layer_features` - CLIP vision tower middle layer
- `vision_final_layer_features` - CLIP vision tower final layer
- `lm_middle_visual_features` - LM middle layer, visual token span
- `lm_final_visual_features` - LM final layer, visual token span
- `lm_middle_prompt_features` - LM middle layer, prompt token span
- `lm_final_prompt_features` - LM final layer, prompt token span
- `lm_middle_answer_features` - LM middle layer, answer token span
- `lm_final_answer_features` - LM final layer, answer token span

## Usage

### Command Line

Train a probe using specific features:

```bash
# Single feature
python -m probe.train \
    --data_path supervision_dataset_with_features.pkl \
    --features vision_final_layer_features \
    --output_dir probe_results/vision_final \
    --num_epochs 100

# Multiple features
python -m probe.train \
    --data_path supervision_dataset_with_features.pkl \
    --features vision_final_layer_features lm_final_answer_features \
    --output_dir probe_results/combined \
    --num_epochs 100 \
    --learning_rate 0.01
```

### Python API

```python
from probe.train import train_probe

# Train probe
model, history = train_probe(
    data_path='supervision_dataset_with_features.pkl',
    feature_names=['vision_final_layer_features', 'lm_final_answer_features'],
    output_dir='probe_results',
    num_epochs=100,
    learning_rate=0.01,
)
```

### Example Experiments

Run multiple experiments with different feature combinations:

```bash
python train_probes_example.py
```

## Arguments

### Required
- `--data_path`: Path to supervision dataset pickle file
- `--features`: Feature names to use (space-separated)

### Optional
- `--output_dir`: Directory to save results (default: `probe_results`)
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.01)
- `--weight_decay`: L2 regularization (default: 1e-4)
- `--train_split`: Fraction for training (default: 0.8)
- `--no_normalize`: Disable feature normalization
- `--seed`: Random seed (default: 42)
- `--device`: Device to use (cuda/cpu)

## Output

The training script saves:
- `best_model.pt`: Best model checkpoint with metadata
- `training_history.json`: Loss and metrics per epoch
- `config.json`: Experiment configuration and results

## Brier Score

The Brier score is used as the loss function:

```
BS = (1/N) * Σ(p - y)²
```

where `p` is predicted probability and `y` is true label (0 or 1).

**Properties:**
- Proper scoring rule (encourages accurate probability estimates)
- Differentiable (suitable for gradient-based optimization)
- Interpretable (mean squared error of probabilities)
- Encourages calibration

## Calibration Metrics

The pipeline also computes:
- **ECE (Expected Calibration Error)**: Measures calibration quality
- **Accuracy**: Classification accuracy using 0.5 threshold
- **Mean predictions vs labels**: Checks for bias

## Example Output

```
Using device: cuda
Loading data from: supervision_dataset_with_features.pkl
Selected features: ['vision_final_layer_features', 'lm_final_answer_features']

Dataset statistics:
  Total samples: 10
  Feature dimension: 5120
  Correct: 7 (70.00%)
  Incorrect: 3 (30.00%)
  Train samples: 8
  Val samples: 2

Training for 100 epochs...
================================================================================
Epoch   1/100 | Train Loss: 0.2145 | Val Loss: 0.1987 | Val Acc: 0.7500 | ECE: 0.0432
Epoch  10/100 | Train Loss: 0.1823 | Val Loss: 0.1756 | Val Acc: 0.8000 | ECE: 0.0321
...
```
