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

## Important Data Paths

- Patching datasets:
  `/projects/prjs2014/patching/<model>/<dataset>/patching_dataset_<dataset>.json`
- Layer sweeps:
  `/projects/prjs2014/patching/layer_sweep_results/<model>/<dataset>/<tag>/layer_sweep_results.parquet`
- Positional sweeps:
  `/projects/prjs2014/patching/positional_patching_results_including_visual/<model>/<dataset>/<tag>/positional_patching_results.parquet`
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
