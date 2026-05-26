# accuracy_recovers_confidence_not_LLaVA-1.5-7B_coco-qa-vi_28748_pos592

## Caption

LLaVA-1.5-7B on COCO-QA-VI, sample 28748: patching the question at token position 592 changes the degraded output from Nothing (0.12) to Tree (0.12). The clean output is Branch (0.20). Selection category: accuracy recovers confidence not.

## LaTeX table row

```latex
LLaVA-1.5-7B & COCO-QA-VI & 28748 & Question & 592 & Branch & 0.20 & 0.00 & Nothing & 0.12 & 0.00 & Tree & 0.12 & 1.00 & 5.00 & -0.08 \\
```

## Interpretation

This example shows that patching the selected question position can alter the degraded model behavior at the sample level. In this case, answer correctness increases without a corresponding confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
