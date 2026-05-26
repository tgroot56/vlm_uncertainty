# final_prompt_accuracy_recovery_LLaVA-1.5-7B_coco-qa-vi_2908_pos606

## Caption

LLaVA-1.5-7B on COCO-QA-VI, sample 2908: patching the template suffix at token position 606 changes the degraded output from Nothing (0.37) to Airplanes (0.53). The clean output is Cars (0.39). Selection category: final prompt accuracy recovery.

## LaTeX table row

```latex
LLaVA-1.5-7B & COCO-QA-VI & 2908 & Template suffix & 606 & Cars & 0.39 & 0.00 & Nothing & 0.37 & 0.00 & Airplanes & 0.53 & 1.00 & 5.00 & 5.00 \\
```

## Interpretation

This example shows that patching the selected template suffix position can alter the degraded model behavior at the sample level. In this case, both answer correctness and expressed confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
