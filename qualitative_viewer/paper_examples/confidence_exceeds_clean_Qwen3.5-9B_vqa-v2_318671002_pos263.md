# confidence_exceeds_clean_Qwen3.5-9B_vqa-v2_318671002_pos263

## Caption

Qwen3.5-9B on VQA-v2, sample 318671002: patching the visual tokens at token position 263 changes the degraded output from 0 (1.00) to 0 (1.00). The clean output is 1 (0.99). Selection category: confidence exceeds clean.

## LaTeX table row

```latex
Qwen3.5-9B & VQA-v2 & 318671002 & Visual tokens & 263 & 1 & 0.99 & 1.00 & 0 & 1.00 & 0.00 & 0 & 1.00 & 0.00 & 0.00 & -0.00 \\
```

## Interpretation

This example shows that patching the selected visual tokens position can alter the degraded model behavior at the sample level. In this case, expressed confidence increases while the correctness score does not improve. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
