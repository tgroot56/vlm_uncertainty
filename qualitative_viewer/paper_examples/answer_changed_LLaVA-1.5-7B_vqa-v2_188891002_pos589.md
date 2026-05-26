# answer_changed_LLaVA-1.5-7B_vqa-v2_188891002_pos589

## Caption

LLaVA-1.5-7B on VQA-v2, sample 188891002: patching the question at token position 589 changes the degraded output from No plane (0.53) to White (0.26). The clean output is Red (0.37). Selection category: answer changed.

## LaTeX table row

```latex
LLaVA-1.5-7B & VQA-v2 & 188891002 & Question & 589 & Red & 0.37 & 0.00 & No plane & 0.53 & 0.00 & White & 0.26 & 1.00 & 5.00 & 1.74 \\
```

## Interpretation

This example shows that patching the selected question position can alter the degraded model behavior at the sample level. In this case, answer correctness increases without a corresponding confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
