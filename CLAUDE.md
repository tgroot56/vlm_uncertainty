# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **Vision Language Model (VLM) Uncertainty Quantification** — probing model confidence and correctness prediction via hidden representations, and performing activation patching experiments across different VLMs (LLaVA, Qwen, Molmo).

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate llava-experiments
```

**Key environment variables** (set before running):
```bash
export VLM_UQ_STORAGE_ROOT=/projects/prjs2014     # Root storage
export VLM_UQ_USER_STORAGE_ROOT=$VLM_UQ_STORAGE_ROOT/$USER
export HF_HOME=...                                  # HuggingFace model cache
```

All path defaults and derived paths are configured in `src/storage_paths.py`.

## Running Code

All CLI modules are invoked as:
```bash
python -m src.cli.<module> [args]
```

Main CLI modules in `src/cli/`:
- `generate` — generate supervised UQ datasets from VLM forward passes
- `train` — train probe models on supervision datasets
- `build_patching_dataset` — build paired clean/degraded image datasets
- `collect_patching_activations` — pre-cache activations for patching
- `run_patching` — execute activation patching experiments
- `run_positional_patching` — single-token positional sweep: patch one non-visual token at a time at a fixed layer, measuring recovery vs position (args: `--patching_dataset`, `--vlm`, `--layer` default 17, `--severity`)
- `plot_patching_results` — generate Figures 1–6 from `patching_results.parquet` (args: `--results_parquet`, `--output_dir`, `--figures`, `--n_bootstrap`, `--probe_peak_layer`)
- `run_generalization` — cross-dataset generalization studies

Production runs use SLURM job scripts in `scripts/` (organized by model: `llava/`, `qwen/`, `molmo/`):
```bash
sbatch scripts/llava/vqa-v2/generate_dataset.job
```

There are no automated tests; experiments are validated by examining outputs and metrics.

## Architecture

### Typical Experiment Pipeline

1. **Generate dataset**: Load VLM + dataset → run forward passes → extract features (hidden states, logits) → save supervision dataset (`.pt` file)
2. **Train probe**: Load supervision dataset → train linear/MLP probe to predict correctness from features → evaluate with AUROC/Brier score
3. **Patching** (optional): Build clean/degraded image pairs → cache activations → swap activations between pairs → measure effect on model confidence

### Key Module Responsibilities

**`src/vlm/adapters.py`** — Unified VLM interface for LLaVA, Qwen, Molmo, BLIP2. Handles model loading, preprocessing, forward passes, and hidden state extraction from specific layers. This is the main entry point for model interaction.

**`src/uq_dataset_generation/`** — Core generation pipeline. `SupervisionGenConfig` controls what dataset/model/features to use. Task-specific logic (answer normalization, label extraction) lives in `tasks/`. Extracts vision encoder states, LM hidden states (by span: visual tokens, question tokens, answer tokens), and generation statistics.

**`src/data/`** — Registry-based dataset loading for VQA-v2, POPE, COCO-QA-VI, ImageNet-R with caching. `registry.py` maps dataset IDs to implementations.

**`src/probe/`** — Probe model definitions (Linear, MLP, ComponentAttentionMLP), training loop with class weighting for imbalanced data, and evaluation metrics.

**`src/patching/`** — Activation patching: build datasets of clean/blurred image pairs (`build_dataset.py`), auto-calibrate blur levels (`calibrate_blur.py`), collect and cache activations (`collect_activations.py`), swap activations and measure correctness/confidence changes (`run_patching.py`). `run_positional_patching.py` iterates over every non-visual token position at a single layer. `plot_patching_results.py` generates Figures 1–6 from any `patching_results.parquet`; recovery formula: `(patched − degraded) / (clean − degraded + ε)`.

**`src/labeling/`** — Correctness scoring: VQA soft accuracy (`vqa.py`, handles contractions/normalization), multiple-choice binary accuracy.

**`src/generalization/`** — Cross-dataset transfer: train on N-1 datasets, evaluate on held-out (leave-one-out).

**`utils/`** — Lower-level utilities: model loading (`model_loader.py`), inference and feature extraction (`inference.py`, `extract_features.py`), image loading/caching (`image_resolver.py`).

### Supported VLMs and Datasets

- **VLMs**: LLaVA-1.5/1.6 (7B), Qwen-VL, Molmo, BLIP2-Flan-T5
- **Datasets**: VQA-v2, POPE (always use random split), COCO-QA-VI, ImageNet-R

### Feature Types Extracted

Hidden states are extracted by span type and layer index, e.g.:
- `vision_mean_layer_12` — vision encoder hidden state at layer 12
- `lm_visual_mean_layer_16` — LM hidden states over visual token span at layer 16
- `lm_question_mean_layer_24` — LM hidden states over question token span
- Answer generation statistics: token probabilities, entropy, logits

# Project Summary

## Purpose

This repository studies uncertainty and correctness in vision-language models
(VLMs). The main models are LLaVA-1.5-7B and Qwen3.5-9B, with support elsewhere
for Molmo and BLIP2. The work connects hidden-state probing, activation
patching, output confidence, and qualitative inspection across VQA-v2,
COCO-QA-VI, ImageNet-R, and POPE.

The Streamlit qualitative visualizer lives in `qualitative_viewer/`. It makes
sample-level patching behavior inspectable and exports thesis-ready figures.

## Main Probing Experiments

- Linear, MLP, and component-attention probes predict answer correctness from
  hidden representations.
- Features include vision states and language-model states pooled over visual,
  question, final-prompt-token, and answer spans.
- Probe quality is compared with output-level confidence baselines using AUROC,
  Brier score, calibration, and mismatch analyses.
- The main finding being tested is that middle-layer hidden states encode
  correctness better than the expressed output distribution.

Key code:

- `src/probe/`: probe models, training, data, and metrics.
- `src/pipelines/train_probe.py`: probe training pipeline.
- `src/cli/evaluate_msts_probes.py`: MSTS/OOD evaluation.
- `scripts/probing/`, `scripts/generalization/`, and model-specific scripts:
  production SLURM jobs and analysis plots.

## Main Activation-Patching Experiments

- Clean/degraded patching datasets pair a clean image with blurred
  perturbations.
- Layer sweeps patch the clean final-prompt-token activation into a degraded
  run across hidden-state layers.
- Positional sweeps patch individual prompt positions at a fixed layer,
  including visual tokens.
- Gradient-guided patching selects prompt tokens by their gradient-estimated
  contribution to a target objective and applies them in ranked order.
- Recovery is measured relative to clean and degraded baselines:
  `(patched - degraded) / (clean - degraded + epsilon)`.

Key code:

- `src/patching/build_dataset.py`
- `src/patching/collect_activations.py`
- `src/patching/run_layer_sweep_patching.py`
- `src/patching/run_positional_patching.py`
- `src/patching/run_gradient_guided_token_patching.py`
- `src/cli/plot_sweep_filtered.py`
- `scripts/plot_filtered_clean_answer_cross_model_sweeps.py`

## Confidence And Accuracy Tracking

The current sweep pipeline stores and visualizes dataset-specific correctness,
generated-answer confidence, and clean-answer confidence. Clean-answer
confidence is the probability assigned to the clean run's answer under clean,
corrupted, and patched conditions. The qualitative viewer exposes accuracy
recovery, generated-answer confidence recovery, and clean-answer confidence
recovery for the selected intervention.

The clean-answer fields are produced by
`src/patching/run_patching.py::_compute_clean_answer_probability_stats` and are
enabled by the layer/positional runner option
`--save_clean_answer_probability`.

## Qualitative Visualizer

Run the app with:

```bash
qualitative_viewer/run_app.sh 8501
```

Implemented views:

- Positional sweep: clean, corrupted, and patched three-panel comparison.
- Layer sweep: clean, corrupted, and patched three-panel comparison.
- Gradient-guided LLaVA POPE view: four aligned panels for layers 1, 11, 15,
  and 18.
- Prompt token reconstruction and highlighting.
- Visual-token-to-image-patch projection for LLaVA and Qwen.
- Clean image, corrupted image, patched output, confidence, correctness,
  recovery metrics, prompt filtering, and PNG/PDF exports.

The gradient-guided view reads:

`/projects/prjs2014/patching/gradient_guided_token_patching/llava-hf_llava-1-5-7b-hf/pope/random/sample_<id>/obj_yes`

It overlays numbered visual-token patches on the clean image, numbers patched
text tokens in the prompt, shows clean/corrupted/patched answers and
confidences, and summarizes incremental patch increases.

## Final Sweep Sources

The qualitative layer and positional views now use the same parquet families
and filtering logic as `scripts/plot_filtered_clean_answer_cross_model_sweeps.py`.
They take the first 100 source sample IDs per model/dataset, keep
clean-correct/corrupted-incorrect pairs, and use a minimum positional support
of 35.

Final layer figure:

`/projects/prjs2014/patching/layer_sweep_results/two_metrics_layer_sweep_clean_answer_confidence_clean_correct_corrupted_incorrect_unclipped.pdf`

Final positional figure:

`/projects/prjs2014/patching/positional_patching_results_including_visual/two_metrics_positional_sweep_clean_answer_confidence_clean_correct_corrupted_incorrect_n35_unclipped.pdf`

Important cohort note: a separate Qwen 350-sample parquet family exists under
tags containing
`extreme_clean_answer_confidence_stripped_softmax_filtered350_finaltoken_qwen_prefill_empty_thinking`.
It generated the differently named
`two_metrics_*_clean_answer_confidence_qwen_filtered350_llava1000_*` figures.
The exact final figures named above use the non-`filtered350` tags and the
plotting script's 100-sample defaults.

## Top-5 Softmax Status

The configured LLaVA positional parquets and LLaVA POPE layer parquet do not
contain stripped-answer softmax distributions. Most other configured sweep
parquets contain raw `stripped_answer_softmax_*` matrices, but not materialized
top-5 token IDs/text/probabilities. These raw distributions are stored in
single-row-group files as large as tens of GiB, so they are not suitable for
interactive per-sample decoding.

The visualizer documents this below each three-panel view. To enable a real
compact top-5 table, add first stripped-answer-step top-5 IDs, decoded token
text, and probabilities to
`src/patching/run_patching.py::_compute_stripped_answer_softmax_stats`, then rerun
layer and positional sweeps with `--save_stripped_answer_softmax`.

### Softmax Changes POC (layer sweep)
A lightweight standalone experiment that *does* materialize compact top-5 softmax
for browsing — without touching the production parquets.

- **Runner**: `src/patching/run_softmax_changes.py` (core) + `src/cli/run_softmax_changes.py` (CLI).
  Reuses the exact layer-sweep machinery: `_extract_all_layer_activations`,
  `_get_final_prompt_positions`, `_get_layer_index_pairs` (from `run_layer_sweep_patching`)
  and `_run_baseline_forward`, `_patched_generate`, `_compute_output_stats`,
  `_answer_token_indices` (from `run_patching`). It patches the **final prompt token**
  (`position_offset == -1`) at **every LM layer** and records the output softmax top-5 at
  the first stripped answer-token step for the clean baseline, corrupted baseline, and each
  patched layer.
- **Cohort filter** is identical to the main layer-sweep plot: keep only samples with
  `clean_score > 0 and corrupted_score == 0` (same as `_final_plot_cohort` in
  `qualitative_viewer/data_utils.py`). The CLI scans a shuffled pool (`--seed`, default 0)
  up to `--max_candidates` (default 100) and retains `--n_samples` (default 10).
- **Severity = `extreme`** for coco-qa-vi (matches the coco-qa-vi layer sweep; the Qwen
  dataset only carries `extreme`). Qwen must use `--qwen_prefill_empty_thinking`.
- **Output** (one self-contained JSON per model+dataset, ~small):
  `/projects/prjs2014/patching/softmax_changes/<model_slug>/<dataset>/softmax_changes.json`.
  Schema (v2): `metadata{schema_version, ...}` + `samples[{sample_id, question, ground_truth,
  clean_image_path, degraded_image_path,
  clean{answer,score,top1_probability,clean_answer_probability,top5[]},
  corrupted{...}, layers[{layer, patched{...}}]}]`. `clean_answer_probability` (schema v2) is
  the probability that run assigned to the **clean run's answer tokens** (reuses
  `run_patching._compute_clean_answer_probability_stats`), powering the viewer's
  "Clean answer confidence" panel line.
- **SLURM**: `scripts/patching/{llava,qwen}/coco-qa-vi/run_softmax_changes.job`
  (submit with plain `sbatch`, no `--export=ALL`).
- **Viewer**: a new **"Softmax changes"** option in the `Sweep` radio
  (`qualitative_viewer/app.py::show_softmax_changes_view`). It loads the JSON via
  `softmax_changes_path` (config) → `softmax_changes_available` / `softmax_changes_data`
  (data_utils). It **reuses the Layer-sweep three-panel layout** (`show_three_panel_layout`)
  for the selected model/sample/layer — Clean | Corrupted | Patched@L panels with image,
  full prompt, answer+confidence, and the patched-token marker on the final prompt token —
  then renders the three **top-5 softmax tables** (token, softmax probability) below, plus a
  per-layer patched top-1 scan table. A sidebar layer selector and a header line show the
  current model / sample / layer / severity / run type.
- **Sample alignment**: the LLaVA and Qwen runs were drawn from different random samples and
  currently share **0 sample_ids** (though the two coco-qa-vi datasets share 800 ids with
  matching question+image, so alignment is achievable by re-running on a common pool).
  Browsing is therefore **model-specific** (one model at a time); the view reports the
  cross-model overlap. If the outputs were aligned, the same view is the place to add a
  two-row (LLaVA / Qwen) same-sample comparison.
- **Cross-tab "clean answer confidence" line** (`app.py::_panel_answer_block`, shared by
  Positional sweep, Layer sweep, Softmax changes — on-screen panels and PDF/PNG export):
  line 1 is the run's own expressed answer + confidence; line 2 is the probability that run
  assigned to the **clean run's answer**, shown only when it adds information:
  - **non-POPE**: `Clean answer confidence: <clean answer> (P)` using
    `<cond>_clean_answer_probability`, only when the run's answer differs from the clean
    answer (NaN → omitted).
  - **POPE** (binary): always show the complementary answer's probability, e.g.
    `Answer: no (0.89)` / `yes (0.11)`, using per-run `pope_yes_probability` /
    `pope_no_probability` (threaded through `data_utils.py`). Falls back to the non-POPE line
    where those columns are absent (e.g. the LLaVA positional-POPE parquet).
  Scope of the Softmax changes tab is the coco-qa-vi POC (`SOFTMAX_CHANGES_DATASETS`); this
  is **not** wired into the production parquet sweeps.

## Important Data Paths

- Patching datasets:
  `/projects/prjs2014/patching/<model>/<dataset>/patching_dataset_<dataset>.json`
- Layer sweeps:
  `/projects/prjs2014/patching/layer_sweep_results/<model>/<dataset>/<tag>/layer_sweep_results.parquet`
- Positional sweeps:
  `/projects/prjs2014/patching/positional_patching_results_including_visual/<model>/<dataset>/<tag>/positional_patching_results.parquet`
- Softmax-changes POC:
  `/projects/prjs2014/patching/softmax_changes/<model>/<dataset>/softmax_changes.json`
- Visualizer exports:
  `/projects/prjs2014/patching/qualitative_viewer/`
- Viewer source configuration:
  `qualitative_viewer/config.py`
- Viewer data preparation:
  `qualitative_viewer/data_utils.py`
- Viewer UI and exports:
  `qualitative_viewer/app.py`

For POPE, the dataset path component is `pope/random`. Cached Hugging Face
processors must remain available for exact prompt-token reconstruction and
model-specific visual patch geometry.


### SLURM Submission Gotcha
Do **not** pass `--export=ALL,VAR=value` when submitting patching jobs from a programmatic shell — `ALL` inherits the caller's env and breaks the job's `module load Anaconda3` + `source activate llava-experiments` flow (observed failure: `ModuleNotFoundError: No module named 'torch'`). Use `--export=VAR=value` (without `ALL`) instead; slurm then sets only the minimal env plus the overrides, and `source activate` works as in an interactive submission.

### Patching Sweep Severity Gotcha (MUST CHECK)
Before submitting *any* layer-sweep or positional-patching job, **explicitly verify the patching dataset contains the severity the job will request**. The sweep CLI silently skips perturbations whose severity is not in `--severity`, but the outer per-sample loop still marks each sample "completed" and the tqdm bar reaches 100% — producing an **empty parquet (0 rows, ~636 B)** and then crashing the plot step with `KeyError: 'condition'`. Observed: the Qwen datasets only carry `extreme`, the default `SEVERITY=moderate` in `scripts/patching/qwen/run_{layer_sweep,positional}_patching.job` produced zero data and wasted 2×4 GPU-hours (jobs 22217833/22217834).

**How to apply**:
- When the user says "run extreme", pass `SEVERITY=extreme` (or use the dedicated `*_extreme.job` scripts under `scripts/patching/<vlm>/<dataset>/`). Never rely on the script default without checking.
- Before submitting, sanity-check severities in the patching dataset (`samples[*].visual_perturbations[*].severity`) and confirm they include the requested `SEVERITY`. If the dataset only has `extreme` (`calibration_num_samples: 0`, `selected_sigmas: {'extreme': 50.0}` in metadata), running with `moderate` is a guaranteed no-op.
- If the user asks you to submit without naming a severity, stop and ask — do not assume the job-script default.