# final_prompt_accuracy_recovery_LLaVA-1.5-7B_vqa-v2_188891002_pos603

## Caption

LLaVA-1.5-7B on VQA-v2, sample 188891002: patching the template suffix at token position 603 changes the degraded output from No plane (0.53) to White (0.32). The clean output is Red (0.37). Selection category: final prompt accuracy recovery.

## LaTeX table row

```latex
LLaVA-1.5-7B & VQA-v2 & 188891002 & Template suffix & 603 & Red & 0.37 & 0.00 & No plane & 0.53 & 0.00 & White & 0.32 & 1.00 & 5.00 & 1.32 \\
```

## Interpretation

This example shows that patching the selected template suffix position can alter the degraded model behavior at the sample level. In this case, answer correctness increases without a corresponding confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
