# final_prompt_accuracy_recovery_LLaVA-1.5-7B_vqa-v2_507975003_pos608

## Caption

LLaVA-1.5-7B on VQA-v2, sample 507975003: patching the template suffix at token position 608 changes the degraded output from No hat (0.63) to Black (0.49). The clean output is Red (0.53). Selection category: final prompt accuracy recovery.

## LaTeX table row

```latex
LLaVA-1.5-7B & VQA-v2 & 507975003 & Template suffix & 608 & Red & 0.53 & 0.00 & No hat & 0.63 & 0.00 & Black & 0.49 & 1.00 & 5.00 & 1.33 \\
```

## Interpretation

This example shows that patching the selected template suffix position can alter the degraded model behavior at the sample level. In this case, answer correctness increases without a corresponding confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
