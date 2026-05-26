# answer_changed_LLaVA-1.5-7B_vqa-v2_507975003_pos607

## Caption

LLaVA-1.5-7B on VQA-v2, sample 507975003: patching the template suffix at token position 607 changes the degraded output from No hat (0.63) to Black (0.31). The clean output is Red (0.53). Selection category: answer changed.

## LaTeX table row

```latex
LLaVA-1.5-7B & VQA-v2 & 507975003 & Template suffix & 607 & Red & 0.53 & 0.00 & No hat & 0.63 & 0.00 & Black & 0.31 & 1.00 & 5.00 & 3.04 \\
```

## Interpretation

This example shows that patching the selected template suffix position can alter the degraded model behavior at the sample level. In this case, answer correctness increases without a corresponding confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
