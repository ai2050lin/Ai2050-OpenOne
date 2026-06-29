# Phase 740 Natural Readout Boost Source Backtrace (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: final-state threshold fraction and late-component raw projections.

| model | target site | top audit | threshold | patched final fraction | donor final fraction | top component | component patched fraction |
|---|---|---|---:|---:|---:|---|---:|
| qwen3 | hidden_36 | source_plus_all_mlp|L35H0<-self_last|L34:mlp[85:128]+L28:mlp[299:341] explicit<-conflict | 25.916 | -0.005 | 0.905 | L34:attn_out | 0.014 |
| glm4 | hidden_40 | source_plus_all_mlp|L23H17<-instruction|L38:mlp[2597:2665]+L38:mlp[3007:3075] conflict<-explicit | 8.357 | 0.060 | 2.932 | L38:mlp_out | 0.063 |
| deepseek7b | hidden_28 | source_plus_all_mlp|L22H24<-all_pre_answer|L27:mlp[2872:2932]+L22:mlp[957:1017] conflict<-explicit | 8.096 | 0.031 | 1.990 | L26:attn_out | 0.280 |

## Strict Interpretation

- Final-state projection fractions are comparable to Phase739 alpha thresholds.
- Component projections are pre-final-norm raw signals and should be treated as backtrace candidates, not causal closure proof.

Atlas graph: nodes=36 edges=33
