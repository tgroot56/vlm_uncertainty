# visual_token_accuracy_recovery_LLaVA-1.5-7B_vqa-v2_66072000_pos125

## Caption

LLaVA-1.5-7B on VQA-v2, sample 66072000: patching the visual tokens at token position 125 changes the degraded output from White (0.18) to Blue (0.19). The clean output is Black (0.66). Selection category: visual token accuracy recovery.

## LaTeX table row

```latex
LLaVA-1.5-7B & VQA-v2 & 66072000 & Visual tokens & 125 & Black & 0.66 & 0.00 & White & 0.18 & 0.00 & Blue & 0.19 & 1.00 & 5.00 & 0.01 \\
```

## Interpretation

This example shows that patching the selected visual tokens position can alter the degraded model behavior at the sample level. In this case, both answer correctness and expressed confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
