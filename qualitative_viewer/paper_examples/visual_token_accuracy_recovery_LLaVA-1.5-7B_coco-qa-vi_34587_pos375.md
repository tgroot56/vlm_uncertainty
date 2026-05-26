# visual_token_accuracy_recovery_LLaVA-1.5-7B_coco-qa-vi_34587_pos375

## Caption

LLaVA-1.5-7B on COCO-QA-VI, sample 34587: patching the visual tokens at token position 375 changes the degraded output from Stove (0.27) to Oven (0.33). The clean output is Microwave (0.79). Selection category: visual token accuracy recovery.

## LaTeX table row

```latex
LLaVA-1.5-7B & COCO-QA-VI & 34587 & Visual tokens & 375 & Microwave & 0.79 & 0.00 & Stove & 0.27 & 0.00 & Oven & 0.33 & 1.00 & 5.00 & 0.12 \\
```

## Interpretation

This example shows that patching the selected visual tokens position can alter the degraded model behavior at the sample level. In this case, both answer correctness and expressed confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
