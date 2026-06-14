# Phase 109 Cross-Model Support/Suppressor Decomposition Summary

## Category Decomposition
| category | qwen3 | glm4 | deepseek7b | objective reading |
|---|---|---|---|---|
| number | cos=0.17 frac=0.17; parT=-0.05; orthT=-3.05; orthRel=fruit+0.56; fullT=-3.06; orthogonal_target_support | cos=0.02 frac=0.02; parT=-0.01; orthT=-0.01; orthRel=material+0.45; fullT=-0.00; weak_or_mixed | cos=0.13 frac=0.13; parT=-0.08; orthT=-4.95; orthRel=none+0.00; fullT=-4.75; orthogonal_target_support | target support is mostly orthogonal to readout direction |
| time | cos=0.20 frac=0.20; parT=-0.18; orthT=-0.64; orthRel=animal+0.96; fullT=-1.35; mixed_target_down | cos=-0.02 frac=0.02; parT=-0.01; orthT=-0.23; orthRel=material+0.33; fullT=-0.24; weak_or_mixed | cos=0.10 frac=0.10; parT=-0.02; orthT=-0.05; orthRel=clothing+0.42; fullT=-0.05; weak_or_mixed | weak or mixed |
| container | cos=0.17 frac=0.17; parT=0.01; orthT=-0.33; orthRel=shape+2.81; fullT=-0.34; orthogonal_competition_release | cos=-0.01 frac=0.01; parT=0.00; orthT=-0.04; orthRel=event+0.25; fullT=-0.05; weak_or_mixed | cos=0.10 frac=0.10; parT=0.06; orthT=-3.15; orthRel=none+0.00; fullT=-3.21; orthogonal_target_support | target support is mostly orthogonal to readout direction |
| clothing | cos=0.17 frac=0.17; parT=-0.37; orthT=-0.57; orthRel=tool+1.46; fullT=-0.45; orthogonal_competition_release | cos=0.00 frac=0.00; parT=-0.01; orthT=-0.13; orthRel=property+0.17; fullT=-0.13; weak_or_mixed | cos=0.09 frac=0.09; parT=-0.87; orthT=0.40; orthRel=tool+2.24; fullT=0.39; readout_parallel_support | readout-parallel support exists |
| furniture | cos=0.15 frac=0.15; parT=0.02; orthT=1.00; orthRel=number+3.30; fullT=0.72; orthogonal_competition_release | cos=0.03 frac=0.03; parT=-0.00; orthT=-0.08; orthRel=material+0.22; fullT=-0.08; weak_or_mixed | cos=0.07 frac=0.07; parT=-1.11; orthT=0.16; orthRel=tool+1.09; fullT=0.31; readout_parallel_support | readout-parallel support exists |
| plant | cos=0.19 frac=0.19; parT=0.11; orthT=-0.41; orthRel=color+0.13; fullT=-0.37; weak_or_mixed | cos=0.00 frac=0.00; parT=-0.01; orthT=-0.06; orthRel=shape+0.39; fullT=-0.06; weak_or_mixed | cos=0.10 frac=0.10; parT=-0.19; orthT=0.28; orthRel=animal+1.59; fullT=0.33; orthogonal_competition_release | orthogonal component releases competitors |

## Objective Facts
- Across Qwen3 and DS7B, strong target-down for number comes from the orthogonal component, not the readout-parallel component.
- Qwen3 time also has larger orthogonal target-down and orthogonal competitor release than readout-parallel removal.
- DS7B container target-down is almost entirely orthogonal: orthogonal target_delta=-3.15, full=-3.21, readout_parallel does not reduce target.
- DS7B clothing/furniture show readout-parallel target-down but orthogonal competitor release and full-boundary target-up, confirming component conflict.
- Boundary-readout cos is small in all models; the category-causal boundary is mostly not aligned with direct output readout words.
- GLM4 remains weak; its boundary-readout cos is near zero and effects are small.
