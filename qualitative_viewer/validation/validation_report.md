# Qualitative viewer validation report

Target plot: `/projects/prjs2014/patching/filtered_plots/cross_model/sample_cutoff/two_metrics_positional_sweep_nofilter_100_alltokenspans.pdf`

The viewer uses the same eight positional-sweep parquet files and applies the same no-filter, `min_pos_samples=100` positional preparation used by `scripts/plot_two_metrics_positional_sweep.py`.

## Loaded parquet files

| model        | dataset    | exists | clean_samples | plot_samples | plot_rows | parquet                                                                                                                                                                                                   |
| ------------ | ---------- | ------ | ------------- | ------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLaVA-1.5-7B | vqa-v2     | True   | 1000          | 1000         | 96781     | /projects/prjs2014/patching/positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf/vqa-v2/extreme_answer_confidence/positional_patching_results.parquet                                    |
| LLaVA-1.5-7B | coco-qa-vi | True   | 1000          | 1000         | 98609     | /projects/prjs2014/patching/positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf/coco-qa-vi/extreme_answer_confidence/positional_patching_results.parquet                                |
| LLaVA-1.5-7B | imagenet-r | True   | 1000          | 1000         | 133273    | /projects/prjs2014/patching/positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf/imagenet-r/extreme_answer_confidence/positional_patching_results.parquet                                |
| LLaVA-1.5-7B | pope       | True   | 1000          | 1000         | 100747    | /projects/prjs2014/patching/positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf/pope/random/extreme_answer_confidence/positional_patching_results.parquet                               |
| Qwen3.5-9B   | vqa-v2     | True   | 1000          | 1000         | 61045     | /projects/prjs2014/patching/positional_patching_results_including_visual/Qwen_Qwen3-5-9B/vqa-v2/extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking/positional_patching_results.parquet      |
| Qwen3.5-9B   | coco-qa-vi | True   | 1000          | 1000         | 62086     | /projects/prjs2014/patching/positional_patching_results_including_visual/Qwen_Qwen3-5-9B/coco-qa-vi/extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking/positional_patching_results.parquet  |
| Qwen3.5-9B   | imagenet-r | True   | 1000          | 1000         | 83937     | /projects/prjs2014/patching/positional_patching_results_including_visual/Qwen_Qwen3-5-9B/imagenet-r/extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking/positional_patching_results.parquet  |
| Qwen3.5-9B   | pope       | True   | 1000          | 1000         | 66543     | /projects/prjs2014/patching/positional_patching_results_including_visual/Qwen_Qwen3-5-9B/pope/random/extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking/positional_patching_results.parquet |

## Available token spans

- LLaVA-1.5-7B / VQA-v2: assistant_marker, question, text_pre, visual
- LLaVA-1.5-7B / COCO-QA-VI: assistant_marker, question, text_pre, visual
- LLaVA-1.5-7B / ImageNet-R: assistant_marker, question, text_pre, visual
- LLaVA-1.5-7B / POPE: assistant_marker, question, text_pre, visual
- Qwen3.5-9B / VQA-v2: assistant_marker, question, text_pre, visual
- Qwen3.5-9B / COCO-QA-VI: assistant_marker, question, text_pre, visual
- Qwen3.5-9B / ImageNet-R: assistant_marker, question, text_pre, visual
- Qwen3.5-9B / POPE: assistant_marker, question, text_pre, visual

## Available columns

- LLaVA-1.5-7B / vqa-v2: `sample_id, severity, condition, token_position, position_index, span_label, predicted_answer, score, top1_probability, output_entropy, answer_geom_mean_probability, answer_mean_logprob, answer_min_token_probability, answer_first_token_probability, answer_token_count, answer_token_ids, answer_token_text, first_generated_token_top1_probability, first_generated_token_entropy`
- LLaVA-1.5-7B / coco-qa-vi: `sample_id, severity, condition, token_position, position_index, span_label, predicted_answer, score, top1_probability, output_entropy, answer_geom_mean_probability, answer_mean_logprob, answer_min_token_probability, answer_first_token_probability, answer_token_count, answer_token_ids, answer_token_text, first_generated_token_top1_probability, first_generated_token_entropy`
- LLaVA-1.5-7B / imagenet-r: `sample_id, severity, condition, token_position, position_index, span_label, predicted_answer, score, top1_probability, output_entropy, answer_geom_mean_probability, answer_mean_logprob, answer_min_token_probability, answer_first_token_probability, answer_token_count, answer_token_ids, answer_token_text, first_generated_token_top1_probability, first_generated_token_entropy`
- LLaVA-1.5-7B / pope: `sample_id, severity, condition, token_position, position_index, span_label, predicted_answer, score, top1_probability, output_entropy, answer_geom_mean_probability, answer_mean_logprob, answer_min_token_probability, answer_first_token_probability, answer_token_count, answer_token_ids, answer_token_text, first_generated_token_top1_probability, first_generated_token_entropy`
- Qwen3.5-9B / vqa-v2: `sample_id, severity, condition, token_position, position_index, span_label, predicted_answer, score, top1_probability, output_entropy, answer_geom_mean_probability, answer_mean_logprob, answer_min_token_probability, answer_first_token_probability, answer_token_count, answer_token_ids, answer_token_text, first_generated_token_top1_probability, first_generated_token_entropy`
- Qwen3.5-9B / coco-qa-vi: `sample_id, severity, condition, token_position, position_index, span_label, predicted_answer, score, top1_probability, output_entropy, answer_geom_mean_probability, answer_mean_logprob, answer_min_token_probability, answer_first_token_probability, answer_token_count, answer_token_ids, answer_token_text, first_generated_token_top1_probability, first_generated_token_entropy`
- Qwen3.5-9B / imagenet-r: `sample_id, severity, condition, token_position, position_index, span_label, predicted_answer, score, top1_probability, output_entropy, answer_geom_mean_probability, answer_mean_logprob, answer_min_token_probability, answer_first_token_probability, answer_token_count, answer_token_ids, answer_token_text, first_generated_token_top1_probability, first_generated_token_entropy`
- Qwen3.5-9B / pope: `sample_id, severity, condition, token_position, position_index, span_label, predicted_answer, score, top1_probability, output_entropy, answer_geom_mean_probability, answer_mean_logprob, answer_min_token_probability, answer_first_token_probability, answer_token_count, answer_token_ids, answer_token_text, first_generated_token_top1_probability, first_generated_token_entropy`

## Aggregate recovery validation

Aggregate confidence and accuracy recovery values were recomputed with `src.cli.plot_sweep_filtered._prepare_positional_sweep` and `aggregate_recovery_ci`, the same functions used by the target plot. The numerical values are saved in `aggregate_recovery_values.csv`. No separate numeric dump from the original PDF was found, so the match is by exact source pipeline equivalence rather than by reading values back from the rendered PDF.

| model        | dataset    | min_samples_per_position | max_samples_per_position |
| ------------ | ---------- | ------------------------ | ------------------------ |
| LLaVA-1.5-7B | coco-qa-vi | 132                      | 1000                     |
| LLaVA-1.5-7B | imagenet-r | 109                      | 1000                     |
| LLaVA-1.5-7B | pope       | 205                      | 1000                     |
| LLaVA-1.5-7B | vqa-v2     | 121                      | 1000                     |
| Qwen3.5-9B   | coco-qa-vi | 103                      | 1000                     |
| Qwen3.5-9B   | imagenet-r | 101                      | 1000                     |
| Qwen3.5-9B   | pope       | 241                      | 1000                     |
| Qwen3.5-9B   | vqa-v2     | 123                      | 1000                     |

## Known limitations logged for requested qualitative features

- The parquet stores prompt token positions and span labels, but not prompt token strings. The app reconstructs exact token strings from cached Hugging Face processors when available and otherwise displays an explicit fallback warning.
- The parquet does not store a guaranteed visual-token-to-original-pixel map. Visual-token highlights are therefore row-major grid visualizations derived from the visual token count; the app labels this explicitly.
- Displayed confidence uses `answer_geom_mean_probability` when present. Recovery values use `top1_probability`, matching the positional-sweep plotting pipeline; in the inspected parquets these columns carry the same answer-level geometric confidence values.
