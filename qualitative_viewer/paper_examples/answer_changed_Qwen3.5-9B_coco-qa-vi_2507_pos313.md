# answer_changed_Qwen3.5-9B_coco-qa-vi_2507_pos313

## Caption

Qwen3.5-9B on COCO-QA-VI, sample 2507: patching the question at token position 313 changes the degraded output from red (0.23) to gray (0.27). The clean output is white (0.26). Selection category: answer changed.

## LaTeX table row

```latex
Qwen3.5-9B & COCO-QA-VI & 2507 & Question & 313 & white & 0.26 & 0.00 & red & 0.23 & 0.00 & gray & 0.27 & 1.00 & 5.00 & 1.79 \\
```

## Interpretation

This example shows that patching the selected question position can alter the degraded model behavior at the sample level. In this case, both answer correctness and expressed confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
