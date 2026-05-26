# accuracy_recovers_confidence_not_LLaVA-1.5-7B_vqa-v2_148614006_pos405

## Caption

LLaVA-1.5-7B on VQA-v2, sample 148614006: patching the visual tokens at token position 405 changes the degraded output from Lamp (0.48) to Plant (0.30). The clean output is Clock (0.87). Selection category: accuracy recovers confidence not.

## LaTeX table row

```latex
LLaVA-1.5-7B & VQA-v2 & 148614006 & Visual tokens & 405 & Clock & 0.87 & 0.00 & Lamp & 0.48 & 0.00 & Plant & 0.30 & 1.00 & 5.00 & -0.47 \\
```

## Interpretation

This example shows that patching the selected visual tokens position can alter the degraded model behavior at the sample level. In this case, answer correctness increases without a corresponding confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
