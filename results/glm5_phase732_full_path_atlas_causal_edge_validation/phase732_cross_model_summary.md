# Phase 732 Full-Path Atlas Causal Edge Validation

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: prompt-type site replacement + candidate-head ablation + DS7B full-head-to-MLP delta.

| model | best prompt transfer | delta | changed | strongest head edge | head delta | mediation target delta |
|---|---|---:|---:|---|---:|---:|
| qwen3 | explicit<-commonsense|hidden_35 | -3.698 | 0.056 | L24H29/taste/explicit_profile | -0.099 | 0.000 |
| glm4 | explicit<-commonsense|hidden_39 | -2.387 | 0.167 | L24H19/taste/commonsense | -0.027 | 0.000 |
| deepseek7b | explicit<-commonsense|hidden_27 | -4.559 | 0.889 | L20H17/category/conflict_profile | -1.957 | 32.777 |

## Strict Interpretation

- Positive prompt transfer delta means donor site replacement improved recipient target likelihood.
- Negative prompt transfer delta means replacement hurt target likelihood or caused distribution shift.
- Head ablation is coarse and tests necessity, not semantic purity.
- DS7B full-head-to-MLP delta tests propagation of perturbation, not full closure.

Atlas graph: nodes=55 edges=52
