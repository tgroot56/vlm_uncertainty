# confidence_recovers_accuracy_not_LLaVA-1.5-7B_imagenet-r_408_pos623

## Caption

LLaVA-1.5-7B on ImageNet-R, sample 408: patching the question at token position 623 changes the degraded output from A (0.99) to A (0.99). The clean output is B (0.99). Selection category: confidence recovers accuracy not.

## LaTeX table row

```latex
LLaVA-1.5-7B & ImageNet-R & 408 & Question & 623 & B & 0.99 & 1.00 & A & 0.99 & 0.00 & A & 0.99 & 0.00 & 0.00 & 5.00 \\
```

## Interpretation

This example shows that patching the selected question position can alter the degraded model behavior at the sample level. In this case, expressed confidence increases while the correctness score does not improve. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
