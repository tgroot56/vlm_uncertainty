# Qualitative positional-patching viewer

This directory contains a Streamlit viewer and export scripts for the sample-level positional activation-patching results behind:

`/projects/prjs2014/patching/filtered_plots/cross_model/sample_cutoff/two_metrics_positional_sweep_nofilter_100_alltokenspans.pdf`

## Files loaded

The viewer uses the same positional-sweep parquet files as `scripts/plot_two_metrics_positional_sweep.py`:

- LLaVA-1.5-7B: `/projects/prjs2014/patching/positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf/{dataset}/extreme_answer_confidence/positional_patching_results.parquet`
- Qwen3.5-9B: `/projects/prjs2014/patching/positional_patching_results_including_visual/Qwen_Qwen3-5-9B/{dataset}/extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking/positional_patching_results.parquet`
- For POPE, `{dataset}` is `pope/random`.

It also loads the matching `patching_dataset_<dataset>.json` files for image paths, questions, and ground-truth answers.

## Run validation

```bash
cd /home/tgroot/vlm_uncertainty
PYTHONDONTWRITEBYTECODE=1 /gpfs/home4/tgroot/.conda/envs/llava-experiments/bin/python qualitative_viewer/validate_sources.py
```

This writes:

- `validation/validation_report.md`
- `validation/loaded_parquets_summary.csv`
- `validation/aggregate_recovery_values.csv`

The aggregate values are recomputed with the same preparation and recovery functions used by the plotting pipeline.

## Run the app

Streamlit is required in the active environment.

```bash
cd /home/tgroot/vlm_uncertainty
qualitative_viewer/run_app.sh 8501
```

The launcher treats this repository as the canonical app copy. To run a custom
copy, override `PATCHING_QUAL_VIEWER_APP`:

```bash
PATCHING_QUAL_VIEWER_APP=/path/to/app.py qualitative_viewer/run_app.sh 8501
```

## SSH port forwarding

From your local machine:

```bash
ssh -L 8501:localhost:8501 <user>@<cluster-login-node>
```

Then open `http://localhost:8501` locally after starting Streamlit on the cluster node. If port `8501` is busy, use another port in both commands.

## Selected-sample layouts

The original visualization code is backed up in:

- `qualitative_viewer/backups/app_original_2026-05-26.py`
- `qualitative_viewer/backups/data_utils_original_2026-05-26.py`

The default selected-sample view is the three-panel layout implemented by
`show_three_panel_layout()` in `qualitative_viewer/app.py`. It renders clean,
corrupted, and patched runs side by side, with the image, full prompt, answer,
and answer-level geometric confidence for each run.

The same selected sample can be exported with the `Download three-panel PDF`
and `Download three-panel PNG` buttons under the figure. The PDF keeps text and
panel borders vector-sharp for thesis use; images are embedded at high DPI.

To switch back without editing files, use the sidebar control
`Selected sample layout` and choose `Original`. To permanently restore the old
code, replace `qualitative_viewer/app.py` and `qualitative_viewer/data_utils.py`
with the corresponding files in `qualitative_viewer/backups/`.

## Export paper examples

```bash
cd /home/tgroot/vlm_uncertainty
PYTHONDONTWRITEBYTECODE=1 /gpfs/home4/tgroot/.conda/envs/llava-experiments/bin/python /projects/prjs2014/patching/qualitative_viewer/export_paper_examples.py --per-category 4
```

Outputs are written to:

`/projects/prjs2014/patching/qualitative_viewer/paper_examples/`

Each selected example gets:

- high-resolution PNG figure
- PDF figure
- Markdown file with caption, LaTeX table row, and interpretation paragraph
- one combined `selected_qualitative_examples.csv`

## Field meanings

- `clean`, `degraded`, `patched`: model outputs on the clean image, degraded image, and degraded image with one clean activation patched into a prompt token position.
- `Answer (0.70)`: stripped generated answer plus `answer_geom_mean_probability` rounded to two decimals when available.
- `correctness`: dataset-specific score stored in the parquet.
- `accuracy_recovery`: per-sample `score` recovery from the plotting pipeline.
- `confidence_recovery`: per-sample `top1_probability` recovery from the plotting pipeline.
- `span_rel_pos`: token position relative to the first plotted token in that span.
- `canonical_x`: x-axis position used by the aggregate positional-sweep plot.

## Visual-token highlights

For visual-token interventions, the viewer reconstructs model-specific image geometry from the cached Hugging Face processors:

- LLaVA-1.5-7B uses the CLIP image processor crop geometry. In the current setup this is a `336x336` center crop with patch size `14`, giving a `24x24` visual-token grid.
- Qwen3.5-9B uses the processor-returned `image_grid_thw` and `merge_size` to derive the merged image-token grid. Qwen visual boundary tokens, such as `[VIS_START]` and `[VIS_END]`, are shown in the prompt but are not mapped to pixel regions.

The highlighted region is a projection from the model input grid back onto the displayed image. It should be interpreted as the approximate spatial location of the patched visual token, not as an exact neuron-level receptive field.

## Logged limitations

The parquet files store prompt token positions and span labels, but not prompt token strings. The app reconstructs exact token strings from cached Hugging Face processors when available. If that fails, it displays explicit placeholder tokens and a warning.

If exact processor reconstruction fails, the app does not draw a visual-token patch highlight. It shows the clean and degraded images and reports that no defensible pixel mapping is available.
