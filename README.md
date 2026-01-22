# Probe Experiments

Code structure for running ImageNet-R experiments with LLaVA.

## Project Structure

```
Probe_Experiments/
├── run.py                   # Main entry point (argument parsing & orchestration)
├── data_loader.py           # Dataset loading & prompt building
├── analyze_results.py       # Result analysis & calibration plots
├── environment.yaml         # Conda environment specification
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── model_loader.py      # Model loading utilities
│   ├── inference.py         # Prediction & logit extraction
│   └── experiment.py        # Experiment running & results saving
└── jobs/                    # SLURM job scripts
    ├── environment.job
    └── run.job
```

## Usage

### Running Experiments

Basic usage:
```bash
python run.py
```

With custom arguments:
```bash
python run.py \
    --model_id llava-hf/llava-1.5-7b-hf \
    --dataset axiong/imagenet-r \
    --output my_results.pkl \
    --save_logits \
    --seed_offset 42 \
    --progress_interval 100
```

### Arguments

- `--model_id`: HuggingFace model identifier (default: `llava-hf/llava-1.5-7b-hf`)
- `--dataset`: Dataset name from HuggingFace (default: `axiong/imagenet-r`)
- `--output`: Output file for results (default: `imagenet_r_results.pkl`)
- `--save_logits`: Save logits in results (default: enabled)
- `--no_save_logits`: Disable saving logits to reduce file size
- `--seed_offset`: Starting seed for reproducibility (default: 42)
- `--progress_interval`: Print progress every N samples (default: 100)

### Analyzing Results

After running experiments:
```bash
python analyze_results.py
```

This will:
- Load results from `imagenet_r_results.pkl`
- Calculate Expected Calibration Error (ECE)
- Generate calibration plots
- Print detailed statistics

### Running on Snellius

Submit a job:
```bash
sbatch jobs/run.job
```

## Modules

### `data_loader.py`
- `load_dataset_by_name()`: Load datasets from HuggingFace
- `build_mc_prompt_imagenet_r()`: Build multiple-choice prompts

### `utils/model_loader.py`
- `load_llava_model()`: Load LLaVA model with optimal device settings

### `utils/inference.py`
- `predict_letter_and_logits()`: Run inference and extract predictions/logits

### `utils/experiment.py`
- `run_imagenet_r_experiment()`: Run full experiment loop
- `save_results()`: Save results to pickle file

### `analyze_results.py`
- Load and analyze experiment results
- Calculate calibration metrics (ECE)
- Generate calibration plots

## Output Format

Results are saved as pickle files containing:
```python
{
    "results": [
        {
            "idx": int,
            "gt_letter": str,
            "pred_letter": str,
            "is_correct": bool,
            "option_probs": Dict[str, float],
            "option_logits": Dict[str, float],  # if save_logits=True
            "gt_class": str,
            "option_map": Dict[str, str],
            "raw_text": str
        },
        ...
    ],
    "accuracy": float,
    "total_samples": int,
    "correct_predictions": int,
    "model_id": str,
    "dataset": str,
    "seed_offset": int
}
```
