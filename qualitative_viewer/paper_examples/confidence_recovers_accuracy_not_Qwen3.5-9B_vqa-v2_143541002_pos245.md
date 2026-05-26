# confidence_recovers_accuracy_not_Qwen3.5-9B_vqa-v2_143541002_pos245

## Caption

Qwen3.5-9B on VQA-v2, sample 143541002: patching the template suffix at token position 245 changes the degraded output from 0 (1.00) to 0 (0.98). The clean output is 1 (1.00). Selection category: confidence recovers accuracy not.

## LaTeX table row

```latex
Qwen3.5-9B & VQA-v2 & 143541002 & Template suffix & 245 & 1 & 1.00 & 1.00 & 0 & 1.00 & 0.33 & 0 & 0.98 & 0.33 & 0.00 & 5.00 \\
```

## Interpretation

This example shows that patching the selected template suffix position can alter the degraded model behavior at the sample level. In this case, the patch changes the output pattern without improving the stored recovery metrics. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
