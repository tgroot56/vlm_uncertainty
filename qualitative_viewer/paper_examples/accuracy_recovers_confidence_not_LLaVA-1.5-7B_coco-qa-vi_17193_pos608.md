# accuracy_recovers_confidence_not_LLaVA-1.5-7B_coco-qa-vi_17193_pos608

## Caption

LLaVA-1.5-7B on COCO-QA-VI, sample 17193: patching the template suffix at token position 608 changes the degraded output from Building (0.24) to Car (0.15). The clean output is Red car (0.45). Selection category: accuracy recovers confidence not.

## LaTeX table row

```latex
LLaVA-1.5-7B & COCO-QA-VI & 17193 & Template suffix & 608 & Red car & 0.45 & 0.00 & Building & 0.24 & 0.00 & Car & 0.15 & 1.00 & 5.00 & -0.41 \\
```

## Interpretation

This example shows that patching the selected template suffix position can alter the degraded model behavior at the sample level. In this case, answer correctness increases without a corresponding confidence increase. The qualitative pattern should be interpreted as evidence about this specific intervention and sample, with aggregate trends assessed by the positional-sweep recovery curves.
