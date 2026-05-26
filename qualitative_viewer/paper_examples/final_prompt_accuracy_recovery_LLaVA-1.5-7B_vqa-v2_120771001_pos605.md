# final_prompt_accuracy_recovery_LLaVA-1.5-7B_vqa-v2_120771001_pos605

## Caption

LLaVA-1.5-7B on VQA-v2, sample 120771001: patching the template suffix at token position 605 changes the degraded output from No one (0.74) to Woman (0.26). The clean output is Boy (0.23). Selection category: final prompt accuracy recovery.

## LaTeX table row

```latex
LLaVA-1.5-7B & VQA-v2 & 120771001 & Template suffix & 605 & Boy & 0.23 & 0.00 & No one & 0.74 & 0.00 & Woman & 0.26 & 1.00 & 5.00 & 0.93 \\
```

## Interpretation

This example shows that patching the selected template suffix position can alter the degraded model behavior at the sample level. In this case, answer correctness increases without a corresponding confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
