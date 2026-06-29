# Phase 740 Natural Readout Boost Source Backtrace (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: final-state threshold fraction and late-component raw projections.

| model | target site | top audit | threshold | patched final fraction | donor final fraction | top component | component patched fraction |
|---|---|---|---:|---:|---:|---|---:|
| qwen3 | hidden_36 | source_plus_all_mlp|L35H0<-self_last|L34:mlp[85:128]+L28:mlp[299:341] explicit<-conflict | 17.986 | 0.004 | 1.292 | L34:attn_out | 0.009 |
| glm4 | hidden_40 | source_plus_all_mlp|L39H21<-self_last|L38:mlp[2597:2665]+L38:mlp[3007:3075] conflict<-explicit | 12.291 | 0.029 | 1.892 | L38:mlp_out | 0.042 |
| deepseek7b | hidden_28 | source_plus_all_mlp|L22H24<-all_pre_answer|L27:mlp[2872:2932]+L22:mlp[957:1017] conflict<-explicit | 11.654 | 0.020 | 1.101 | L26:attn_out | 0.057 |

## Strict Interpretation

- Final-state projection fractions are comparable to Phase739 alpha thresholds.
- Component projections are pre-final-norm raw signals and should be treated as backtrace candidates, not causal closure proof.

Atlas graph: nodes=39 edges=36
