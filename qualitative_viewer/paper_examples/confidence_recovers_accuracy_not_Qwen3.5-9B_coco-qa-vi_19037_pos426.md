# confidence_recovers_accuracy_not_Qwen3.5-9B_coco-qa-vi_19037_pos426

## Caption

Qwen3.5-9B on COCO-QA-VI, sample 19037: patching the template suffix at token position 426 changes the degraded output from bat (0.94) to bat (0.97). The clean output is bat (0.94). Selection category: confidence recovers accuracy not.

## LaTeX table row

```latex
Qwen3.5-9B & COCO-QA-VI & 19037 & Template suffix & 426 & bat & 0.94 & 1.00 & bat & 0.94 & 1.00 & bat & 0.97 & 1.00 & 0.00 & 5.00 \\
```

## Interpretation

This example shows that patching the selected template suffix position can alter the degraded model behavior at the sample level. In this case, expressed confidence increases while the correctness score does not improve. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
