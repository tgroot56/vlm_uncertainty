# VLM Uncertainty Experiments

This repository studies uncertainty and correctness in vision-language models
with two main experiment families:

- supervised probing of hidden representations for correctness prediction;
- activation patching of clean versus corrupted image runs, including layer
  sweeps, positional sweeps, and gradient-guided token patching.

The main production runs target LLaVA-1.5-7B and Qwen3.5-9B on VQA-v2,
COCO-QA-VI, ImageNet-R, and POPE random.

## Environment

Create and activate the project environment:

```bash
conda env create -f environment.yaml
conda activate llava-experiments
```

Set the storage and Hugging Face cache roots before running jobs:

```bash
export VLM_UQ_STORAGE_ROOT=/projects/prjs2014
export VLM_UQ_USER_STORAGE_ROOT="$VLM_UQ_STORAGE_ROOT/$USER"
export HF_HOME="$VLM_UQ_USER_STORAGE_ROOT/hf_home"
export HF_DATASETS_CACHE="$VLM_UQ_USER_STORAGE_ROOT/hf_datasets"
export TRANSFORMERS_CACHE="$VLM_UQ_USER_STORAGE_ROOT/hf_transformers"
export VLM_UQ_PREP_CACHE="$VLM_UQ_USER_STORAGE_ROOT/vlm_uq_prepared_cache"
```

All Python CLIs are run as modules from the repository root:

```bash
python -m src.cli.<module> [args]
```

On the cluster, prefer the SLURM scripts in `scripts/`. They set the same
environment variables and write logs under `slurm_outputs/`.

## Supervised Probing

The supervised probing pipeline has four parts:

1. generate a supervision dataset with VLM outputs and hidden-state features;
2. train linear or MLP probes on selected features;
3. train shuffled-label controls with `--shuffle_train_labels`;
4. optionally run layer-sweep and generalisation variants.

### Generate Supervision Datasets

Direct CLI template:

```bash
python -m src.cli.generate \
  --vlm llava-hf/llava-1.5-7b-hf \
  --dataset vqa-v2 \
  --output_dir "$VLM_UQ_STORAGE_ROOT" \
  --seed 42 \
  --device cuda
```

Production jobs are organised by model and dataset:

```bash
sbatch scripts/llava/vqa-v2/generate_dataset.job
sbatch scripts/llava/coco-qa-vi/generate_dataset.job
sbatch scripts/llava/imagenet-r/generate_dataset.job
sbatch scripts/llava/pope/random/generate_dataset.job

sbatch scripts/qwen/vqa-v2/generate_dataset.job
sbatch scripts/qwen/coco-qa-vi/generate_dataset.job
sbatch scripts/qwen/imagenet-r/generate_dataset.job
sbatch scripts/qwen/pope/random/generate_dataset.job
```

Qwen has separate vision-only feature jobs when you need Qwen vision features:

```bash
sbatch scripts/qwen/vqa-v2/generate_visual_dataset.job
sbatch scripts/qwen/coco-qa-vi/generate_visual_dataset.job
sbatch scripts/qwen/imagenet-r/generate_visual_dataset.job
sbatch scripts/qwen/pope/random/generate_visual_dataset.job
```

Generated datasets are saved as `supervision_dataset.pt` under the configured
output root, grouped by dataset and model.

### Train Main Probes

Direct CLI template for one feature:

```bash
python -m src.cli.train \
  --data_path /path/to/supervision_dataset.pt \
  --features lm_fullspan_lasttok_layer_16 \
  --model_type mlp \
  --num_epochs 20 \
  --batch_size 32 \
  --output_dir probe_results/llava/vqa-v2 \
  --seed 42 \
  --no_class_weights
```

Model/dataset jobs loop over the configured feature list:

```bash
sbatch scripts/llava/vqa-v2/train_probes_linear.job
sbatch scripts/llava/vqa-v2/train_probes_mlp.job
sbatch scripts/llava/vqa-v2/train_probes_linear_shuffle_labels.job
sbatch scripts/llava/vqa-v2/train_probes_mlp_shuffle_labels.job
```

Use the matching dataset directory for the rest of the matrix, for example
`scripts/llava/coco-qa-vi/`, `scripts/llava/imagenet-r/`,
`scripts/llava/pope/random/`, and the corresponding `scripts/qwen/...`
directories.

### Probing Layer Sweep

The probing layer sweep generates all-layer feature families, then trains one
probe per layer feature.

For Qwen all-dataset layer sweeps plus a dependent plotting job:

```bash
MAX_SAMPLES=1000 bash scripts/qwen/submit_layer_experiment_all_datasets.sh
```

For one LLaVA dataset:

```bash
sbatch scripts/llava/vqa-v2/layer_experiment.job
```

For one Qwen dataset:

```bash
sbatch scripts/qwen/vqa-v2/layer_experiment.job
```

The layer-experiment jobs write probe results under
`/projects/prjs2014/<model>/layer_experiment/probe_results/...`.

### Hold-One-Dataset-Out Generalisation

Build the combined generalisation dataset:

```bash
MODEL_NAME=qwen sbatch scripts/generalization/generate_dataset.job
MODEL_NAME=llava sbatch scripts/generalization/generate_dataset.job
```

Train leave-one-dataset-out probes by selecting the held-out dataset:

```bash
MODEL_NAME=qwen HELD_OUT_DATASET=vqa-v2 sbatch scripts/generalization/train_leave_one_out_probe.job
MODEL_NAME=qwen HELD_OUT_DATASET=coco-qa-vi sbatch scripts/generalization/train_leave_one_out_probe.job
MODEL_NAME=qwen HELD_OUT_DATASET=imagenet-r sbatch scripts/generalization/train_leave_one_out_probe.job
MODEL_NAME=qwen HELD_OUT_DATASET=pope/random sbatch scripts/generalization/train_leave_one_out_probe.job
```

Repeat with `MODEL_NAME=llava` for LLaVA. The all-dataset in-domain reference
can be trained with:

```bash
MODEL_NAME=qwen sbatch scripts/generalization/train_all_datasets_probe.job
MODEL_NAME=llava sbatch scripts/generalization/train_all_datasets_probe.job
```

Cross-model generalisation is handled by:

```bash
sbatch scripts/generalization/train_model_generalization.job
```

## Activation Patching

The activation-patching pipeline has three stages:

1. build a clean/corrupted patching dataset;
2. extend the dataset with the `extreme` corruption when needed;
3. run layer, positional, or gradient-guided patching.

### Build Patching Datasets

Direct CLI template:

```bash
python -m src.cli.build_patching_dataset \
  --vlm llava-hf/llava-1.5-7b-hf \
  --dataset vqa-v2 \
  --max_samples 2000 \
  --calibration_samples 200 \
  --seed 42 \
  --device cuda
```

SLURM wrappers:

```bash
DATASET=vqa-v2 sbatch scripts/patching/llava/build_dataset.job
DATASET=coco-qa-vi sbatch scripts/patching/llava/build_dataset.job
DATASET=imagenet-r sbatch scripts/patching/llava/build_dataset.job
DATASET=pope DATASET_SPLIT=random sbatch scripts/patching/llava/build_dataset.job
```

For Qwen, use the model-specific or dataset-specific scripts under
`scripts/patching/qwen/`, for example:

```bash
sbatch scripts/patching/qwen/vqa-v2/build_dataset.job
sbatch scripts/patching/qwen/coco-qa-vi/build_dataset.job
sbatch scripts/patching/qwen/imagenet-r/build_dataset.job
sbatch scripts/patching/qwen/pope/build_dataset.job
```

To reuse an existing calibration file, use `build_dataset_from_calibration.job`.

### Add Extreme Corruptions

The final sweeps usually use `SEVERITY=extreme`. Add that severity to an
existing patching dataset with:

```bash
DATASET=vqa-v2 sbatch scripts/patching/llava/extend_dataset_extreme.job
DATASET=pope DATASET_SPLIT=random sbatch scripts/patching/qwen/pope/extend_dataset_extreme.job
```

Use the matching `extend_dataset_extreme.job` under
`scripts/patching/{llava,qwen}/<dataset>/` when present.

### Main Layer-Sweep Patching

Layer sweeps patch final prompt-token activations across LM layers and write
`layer_sweep_results.parquet`.

Direct CLI template:

```bash
python -m src.cli.run_layer_sweep_patching \
  --patching_dataset /projects/prjs2014/patching/llava-hf_llava-1-5-7b-hf/vqa-v2/patching_dataset_vqa-v2.json \
  --vlm llava-hf/llava-1.5-7b-hf \
  --output_dir /projects/prjs2014/patching/layer_sweep_results/llava-hf_llava-1-5-7b-hf/vqa-v2/extreme_clean_answer_confidence_stripped_softmax \
  --n_prompt_tokens 4 \
  --severity extreme \
  --max_samples 1000 \
  --checkpoint_every 10 \
  --save_clean_answer_probability \
  --device cuda
```

SLURM wrappers:

```bash
SAVE_CLEAN_ANSWER_PROBABILITY=1 OUTPUT_TAG=extreme_clean_answer_confidence_stripped_softmax \
  sbatch scripts/patching/llava/vqa-v2/run_layer_sweep_patching_extreme.job

SAVE_CLEAN_ANSWER_PROBABILITY=1 OUTPUT_TAG=extreme_clean_answer_confidence_stripped_softmax_finaltoken \
  sbatch scripts/patching/qwen/vqa-v2/run_layer_sweep_patching_extreme.job
```

Use the corresponding dataset directories for COCO-QA-VI, ImageNet-R, and
POPE. Qwen jobs default to `N_PROMPT_TOKENS=1` and
`QWEN_PREFILL_EMPTY_THINKING=1`; LLaVA jobs default to `N_PROMPT_TOKENS=4`.
The tag names above match the defaults used by
`scripts/plot_filtered_clean_answer_cross_model_sweeps.py`. If you choose a
different tag, pass the matching tag environment variables when plotting.

### Main Positional Patching

Positional sweeps patch one prompt position at a fixed layer and write
`positional_patching_results.parquet`.

Direct CLI template:

```bash
python -m src.cli.run_positional_patching \
  --patching_dataset /projects/prjs2014/patching/llava-hf_llava-1-5-7b-hf/vqa-v2/patching_dataset_vqa-v2.json \
  --vlm llava-hf/llava-1.5-7b-hf \
  --output_dir /projects/prjs2014/patching/positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf/vqa-v2/extreme_clean_answer_confidence \
  --layer 17 \
  --severity extreme \
  --sample_visual_every 10 \
  --include_visual_last 10 \
  --max_samples 1000 \
  --checkpoint_every 10 \
  --save_clean_answer_probability \
  --device cuda
```

SLURM wrappers:

```bash
SAVE_CLEAN_ANSWER_PROBABILITY=1 OUTPUT_TAG=extreme_clean_answer_confidence \
  sbatch scripts/patching/llava/vqa-v2/run_positional_patching_extreme.job

SAVE_CLEAN_ANSWER_PROBABILITY=1 OUTPUT_TAG=extreme_clean_answer_confidence_stripped_softmax_finaltoken \
  sbatch scripts/patching/qwen/vqa-v2/run_positional_patching_extreme.job
```

Again, use the matching dataset folders for the full matrix.

### Gradient-Guided Token Patching

Gradient-guided patching runs on POPE random samples. It ranks candidate tokens
with a gradient score, exactly evaluates a reduced candidate pool, and greedily
adds the best patch until `max_patches` or early stopping.

Direct CLI template:

```bash
python -m src.cli.run_gradient_guided_token_patching \
  --patching_dataset /projects/prjs2014/patching/llava-hf_llava-1-5-7b-hf/pope/random/patching_dataset_pope.json \
  --vlm llava-hf/llava-1.5-7b-hf \
  --output_dir /projects/prjs2014/patching/gradient_guided_token_patching/llava-hf_llava-1-5-7b-hf/pope/random/sample_795 \
  --sample_id 795 \
  --severity extreme \
  --layers $(seq -s ' ' 1 32) \
  --max_patches 10 \
  --visual_top_k 64 \
  --objectives maximize_p_yes \
  --early_stop_window 3 \
  --early_stop_abs_sum_threshold 0.002 \
  --require_corrupted_no \
  --device cuda
```

Ten-sample cross-model jobs:

```bash
sbatch scripts/patching/llava/pope/run_gradient_guided_objectives_10_samples.job
sbatch scripts/patching/qwen/pope/run_gradient_guided_p_yes_10_samples.job
```

Qwen uses the same core setup with `--qwen_prefill_empty_thinking`; the Qwen
job adds this automatically.

## Plotting

### Main Thesis Plot Wrapper

For the editable thesis main plots, run:

```bash
./run_thesis_main_plots.sh
```

This executes `reproduce_thesis_main_plots.ipynb` and writes outputs under
`thesis_plot_outputs/`. It stages the required input data under
`thesis_plot_notebook_data/`.

### Probing: Semantic Token-Span Comparison

Regenerate the configured probe component and semantic-token-span comparison
figures:

```bash
python -m utils.tables_and_plots.probe_results_plot_vs_baselines_2
```

For one model/dataset combination:

```bash
python -m utils.tables_and_plots.probe_results_plot_vs_baselines_2 \
  --combo llava/vqa-v2
```

Outputs go to:

```text
probe_results/_plots/brier_grouped_clean/
```

The main cross-model semantic token-span comparison is:

```text
probe_results/_plots/brier_grouped_clean/all_models_all_datasets/model_comparison_component_comparison_mlp.pdf
```

### Probing: Layer Sweep

For Qwen all-dataset layer-sweep plots:

```bash
python -m utils.tables_and_plots.plot_layer_experiment_brier \
  --model qwen \
  --combined-all-datasets \
  --layer-run-suffix _1k \
  --combined-output-pdf probe_results/layer_experiment/qwen_all_datasets_mlp_layer_brier_1k.pdf
```

For LLaVA:

```bash
python -m utils.tables_and_plots.plot_layer_experiment_brier \
  --model llava \
  --combined-all-datasets \
  --combined-output-pdf probe_results/layer_experiment/llava_all_datasets_mlp_layer_brier.pdf
```

Individual dataset plots can be produced by adding `--dataset vqa-v2`,
`--dataset coco-qa-vi`, `--dataset imagenet-r`, or `--dataset pope/random`.

### Probing: Hold-One-Dataset-Out Generalisation

After running the generalisation jobs, render the leave-one-dataset-out grid:

```bash
python scripts/probing/plot_generalizability_grid.py \
  --root /projects/prjs2014/probe_results \
  --feature-name lm_fullspan_lasttok_layer_16 \
  --output /projects/prjs2014/probe_results/generalization/task_gen/plots/generalizability_grid_lm_final_token_middle_layer_auroc.pdf
```

The script also writes a PNG next to the PDF.

### Activation Patching: Layer And Positional Sweeps

For the final clean-answer confidence plus accuracy recovery figures:

```bash
python scripts/plot_filtered_clean_answer_cross_model_sweeps.py
```

This writes layer plots under:

```text
/projects/prjs2014/patching/layer_sweep_results/
```

and positional plots under:

```text
/projects/prjs2014/patching/positional_patching_results_including_visual/
```

Run only one plot family with:

```bash
PLOT_KIND=layer python scripts/plot_filtered_clean_answer_cross_model_sweeps.py
PLOT_KIND=positional python scripts/plot_filtered_clean_answer_cross_model_sweeps.py
```

The older no-filter two-metric plots are:

```bash
python scripts/plot_two_metrics_layer_sweep.py
python scripts/plot_two_metrics_positional_sweep.py
```

### Gradient-Guided Patching: Confidence And Mean Patches

First aggregate the per-sample gradient-guided runs. For LLaVA:

```bash
ROOT=/projects/prjs2014/patching/gradient_guided_token_patching/llava-hf_llava-1-5-7b-hf/pope/random

python scripts/plot_gradient_guided_objective_contributions.py \
  --root "$ROOT" \
  --sample_ids 795 791 1941 2273 1309 1269 2347 25 429 161 \
  --objectives maximize_p_yes \
  --output_dir "$ROOT/objective_summary_10samples_crossmodel"
```

For Qwen:

```bash
ROOT=/projects/prjs2014/patching/gradient_guided_token_patching/Qwen_Qwen3-5-9B/pope/random

python scripts/plot_gradient_guided_objective_contributions.py \
  --root "$ROOT" \
  --sample_ids 795 791 429 25 2347 2273 1941 161 1309 1269 \
  --objectives maximize_p_yes \
  --output_dir "$ROOT/objective_summary_10samples_crossmodel"
```

Then compose the cross-model figure:

```bash
python scripts/plot_gradient_guided_crossmodel_confidence_recovery.py
```

The combined output is:

```text
/projects/prjs2014/patching/gradient_guided_token_patching/gradient_guided_confidence_recovery_crossmodel.pdf
```

The editable notebook version is
`reproduce_gradient_guided_confidence_recovery_crossmodel_editable.ipynb`. It
produces separate confidence-recovery, mean-patches, and combined figures, and
can write to an alternate output root with `GGP_OUTPUT_ROOT=/path/to/output`.

## Qualitative Viewer

Run the Streamlit viewer with:

```bash
qualitative_viewer/run_app.sh 8501
```

The viewer can inspect layer-sweep, positional-sweep, and LLaVA gradient-guided
patching examples, and can export PNG/PDF figures for selected samples.

## Important Paths

```text
Supervision datasets:
  /projects/prjs2014/<dataset>/<model>/run_*/supervision_dataset.pt
  /projects/prjs2014/qwen/<dataset>/<model>/run_*/supervision_dataset.pt

Probe results:
  probe_results/<model>/<dataset>/
  /projects/prjs2014/<model>/layer_experiment/probe_results/
  /projects/prjs2014/probe_results/generalization/

Patching datasets:
  /projects/prjs2014/patching/<model>/<dataset>/patching_dataset_<dataset>.json
  /projects/prjs2014/patching/<model>/pope/random/patching_dataset_pope.json

Layer-sweep patching results:
  /projects/prjs2014/patching/layer_sweep_results/<model>/<dataset>/<tag>/layer_sweep_results.parquet

Positional patching results:
  /projects/prjs2014/patching/positional_patching_results_including_visual/<model>/<dataset>/<tag>/positional_patching_results.parquet

Gradient-guided patching results:
  /projects/prjs2014/patching/gradient_guided_token_patching/<model>/pope/random/sample_<id>/
```
