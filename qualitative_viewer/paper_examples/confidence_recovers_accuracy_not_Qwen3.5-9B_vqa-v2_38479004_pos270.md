# confidence_recovers_accuracy_not_Qwen3.5-9B_vqa-v2_38479004_pos270

## Caption

Qwen3.5-9B on VQA-v2, sample 38479004: patching the question at token position 270 changes the degraded output from 0 (0.99) to 0 (0.97). The clean output is 1 (0.99). Selection category: confidence recovers accuracy not.

## LaTeX table row

```latex
Qwen3.5-9B & VQA-v2 & 38479004 & Question & 270 & 1 & 0.99 & 1.00 & 0 & 0.99 & 0.00 & 0 & 0.97 & 0.00 & 0.00 & 5.00 \\
```

## Interpretation

This example shows that patching the selected question position can alter the degraded model behavior at the sample level. In this case, the patch changes the output pattern without improving the stored recovery metrics. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
