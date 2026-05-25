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
- **Datasets**: VQA-v2, POPE, COCO-QA-VI, ImageNet-R

### Feature Types Extracted

Hidden states are extracted by span type and layer index, e.g.:
- `vision_mean_layer_12` — vision encoder hidden state at layer 12
- `lm_visual_mean_layer_16` — LM hidden states over visual token span at layer 16
- `lm_question_mean_layer_24` — LM hidden states over question token span
- Answer generation statistics: token probabilities, entropy, logits

## Thesis Storyline

The thesis tells a three-act story: **discovery → mechanism → verification**.

### Act 1: Probing — Hidden states encode correctness better than outputs (MAIN CLAIM)
MLP probes on hidden-state activations (final prompt token, answer span) outperform output-level baselines (softmax entropy, top-1 probability) at predicting model correctness. Correctness is maximally decodable in middle layers. This establishes an **information gap**: the model internally "knows" more about its own reliability than it expresses in the output distribution.

### Act 2: Activation Patching — Two distinct signals explain the gap (MECHANISM)
Layer sweep on clean-vs-blurred pairs reveals why the gap exists:
- **Grounded correctness** (middle-to-late layers ~15–28): patching restores both accuracy AND confidence together. Genuine task-relevant processing.
- **Commitment confidence** (final layer ~31): patching restores confidence WITHOUT accuracy. The model "commits" regardless of evidence quality.

The final layer's commitment operation compresses and distorts the grounded signal, creating the information gap observed in Act 1. Effect is localized entirely to the final prompt token (offset −1).

### Act 3: Mismatch Analysis — Verifying the two-signal account (VERIFICATION)
The two-signal dissociation predicts that overconfident errors (hallucinations) should occur specifically when commitment confidence exceeds grounded correctness. The mismatch analysis tests this falsifiable prediction: if high-mismatch samples have elevated error rates, the mechanistic account is validated. This is a **verification experiment**, not a deployment-ready tool.

## Current Experiment Plan

### Immediate: Mismatch Sanity Check
Split existing evaluation samples into a 2×2 grid (output confident vs. uncertain × probe predicts correct vs. incorrect). Check if the "overconfident hallucination" cell (high output confidence, low probe score, incorrect answer) has elevated error rates vs. the base rate. This uses existing probe results and output confidence data — no new model runs needed.

- Implemented: `src.cli.run_mismatch_sanity_check` joins the best middle-layer probe (by AUROC or Brier via `--selection_metric`) with raw `answer_gen_entropy_mean` per test sample, median-splits both into A/B/C/D cells, reports chi²/Fisher p-values, and writes a per-sample CSV. Plots: `plot_signed_mismatch` (signed mismatch `m = conf_pct − probe_pct` with Wilson CIs) and `plot_mismatch_strata` (within-stratum curves that remove A↔D confounding).
- Heatmap plot (`plot_mismatch_heatmap`): 10×10 error-rate grid over (probe_percentile × confidence_percentile), `origin="lower"` so the four cells sit in expected corners — **A** top-right (both high, lowest error), **B** top-left (output confident, probe disagrees → overconfident hallucination region), **C** bottom-right (output uncertain, probe disagrees → undercommitted), **D** bottom-left (both low, highest error). RdYlGn_r colormap (red = high error), cells annotated with error rate + n, median-split reference lines, sibling `*_heatmap_cells.csv`.

### Priority TODO (ordered by narrative impact)
1. **Mismatch analysis (full)** — Percentile-rank mismatch metric, AUROC comparison (mismatch vs. probe-only vs. confidence-only), error rate by mismatch decile. Compute recovery ratio: (AUROC_mismatch − AUROC_output) / (AUROC_probe − AUROC_output) to quantify how much of the information gap the mismatch recovers. ~1–2 weeks.
2. **Patching on Qwen3.5-VL** — Replicate the two-signal finding on a second model. Critical for generalization of the central mechanistic claim. ~1 week.
3. **MSTS out-of-distribution evaluation** — Supervisor request. Tests probe generalization on unseen distribution. ~3–5 days.
4. **Probing on Molmo-2** — Nice-to-have third model for probing. Cut first if time is short. ~3–5 days.
5. **Writing** — Remaining time.

### Key Conceptual Distinctions
- **Correctness ≠ confidence.** Probes predict binary correctness (will the model be right?), not confidence (how peaked is the output?).
- **Grounded correctness ≠ commitment confidence.** The former is evidence-based (mid layers), the latter is the model's decision to commit (final layer). Their divergence predicts hallucinations.
- **The mismatch analysis is verification, not a tool.** It tests whether the two-signal mechanistic account makes correct predictions about real model failures.

### Positional Sweep Plotting Notes (`src/cli/plot_sweep_filtered.py`)
- Positional sweep plots use **span-relative positions** (`token_position − per-sample span start`) so that e.g. all assistant_marker token 0s align across samples regardless of question length; positions with fewer than 10 samples per severity are dropped.
- The x-axis shows span names ("Text pre", "Visual tokens", "Question", "Assistant marker") centered over each shaded region instead of numeric token indices, with vertical dividers at span boundaries.
- Supported severities: `mild`, `moderate`, `severe`, `extreme`. Missing a severity from `SEVERITY_ORDER` / `SEVERITY_COLORS` silently produces empty plots (filtered out as "not in severity list"); add any new severity to both maps.
- The runner modules (`src/patching/run_{layer_sweep,positional}_patching.py`) now call `plot_{layer_sweep,positional_sweep}_combined` from `plot_sweep_filtered.py` at the end of each job, emitting both `raw` (clean/degraded baselines + patched means) and recovery-ratio variants. `--plot_only` mode in the two CLIs does the same.

### Extreme-Sigma (σ=50) Patching Tests
A fourth severity level `extreme` (fixed `sigma=50.0` = `src.patching.calibrate_blur.EXTREME_BLUR_SIGMA`) tests the patching effect when the visual signal is collapsed to near-uniform colour. It is an *uncalibrated* level, appended to existing patching datasets rather than re-built:
- **Extend existing datasets**: `python -m src.cli.extend_patching_dataset_extreme` (idempotent; skips samples that already have an `extreme` entry). Job file: `scripts/patching/llava/extend_dataset_extreme.job` (env override `EXTREME_SIGMA`, default 50.0). Writes blurred PNGs to `<dataset_dir>/blurred_images/<sid>_sigma50.0.png` and patches metadata + sidecar `calibration.json`.
- **Run sweeps**: `scripts/patching/llava/run_{layer_sweep,positional}_patching_extreme.job` with `SEVERITY=extreme`; outputs land in `.../{layer_sweep_results,positional_patching_results_including_visual}/<vlm>/<dataset>/extreme/` subdirs.
- **Per-dataset job copies** live under `scripts/patching/llava/<dataset>/{extend_dataset_extreme,run_layer_sweep_patching_extreme,run_positional_patching_extreme}.job` with `DATASET` baked in and slurm outputs namespaced to `slurm_outputs/patching/llava/<dataset>/`.

**Qualitative helper**: `python -m src.cli.plot_qualitative_extreme` (defaults to coco-qa-vi/extreme) produces `correct_under_extreme_blur.{pdf,png}` and `top1_score_decoupling.{pdf,png}` in the `extreme/qualitative/` subdir.

### SLURM Submission Gotcha
Do **not** pass `--export=ALL,VAR=value` when submitting patching jobs from a programmatic shell — `ALL` inherits the caller's env and breaks the job's `module load Anaconda3` + `source activate llava-experiments` flow (observed failure: `ModuleNotFoundError: No module named 'torch'`). Use `--export=VAR=value` (without `ALL`) instead; slurm then sets only the minimal env plus the overrides, and `source activate` works as in an interactive submission.

### Patching Sweep Severity Gotcha (MUST CHECK)
Before submitting *any* layer-sweep or positional-patching job, **explicitly verify the patching dataset contains the severity the job will request**. The sweep CLI silently skips perturbations whose severity is not in `--severity`, but the outer per-sample loop still marks each sample "completed" and the tqdm bar reaches 100% — producing an **empty parquet (0 rows, ~636 B)** and then crashing the plot step with `KeyError: 'condition'`. Observed: the Qwen datasets only carry `extreme`, the default `SEVERITY=moderate` in `scripts/patching/qwen/run_{layer_sweep,positional}_patching.job` produced zero data and wasted 2×4 GPU-hours (jobs 22217833/22217834).

**How to apply**:
- When the user says "run extreme", pass `SEVERITY=extreme` (or use the dedicated `*_extreme.job` scripts under `scripts/patching/<vlm>/<dataset>/`). Never rely on the script default without checking.
- Before submitting, sanity-check severities in the patching dataset (`samples[*].visual_perturbations[*].severity`) and confirm they include the requested `SEVERITY`. If the dataset only has `extreme` (`calibration_num_samples: 0`, `selected_sigmas: {'extreme': 50.0}` in metadata), running with `moderate` is a guaranteed no-op.
- If the user asks you to submit without naming a severity, stop and ask — do not assume the job-script default.