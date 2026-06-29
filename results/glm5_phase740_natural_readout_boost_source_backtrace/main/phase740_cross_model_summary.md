# Phase 740 Natural Readout Boost Source Backtrace (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: final-state threshold fraction and late-component raw projections.

| model | target site | top audit | threshold | patched final fraction | donor final fraction | top component | component patched fraction |
|---|---|---|---:|---:|---:|---|---:|
| qwen3 | hidden_36 | source_plus_all_mlp|L35H0<-self_last|L34:mlp[85:128]+L28:mlp[299:341] explicit<-conflict | 19.231 | 0.005 | 1.274 | L34:mlp_out | 0.013 |
| glm4 | hidden_40 | source_plus_all_mlp|L23H17<-instruction|L38:mlp[2597:2665]+L38:mlp[3007:3075] conflict<-explicit | 12.805 | 0.029 | 1.891 | L38:mlp_out | 0.038 |
| deepseek7b | hidden_28 | source_plus_all_mlp|L22H24<-all_pre_answer|L27:mlp[2872:2932]+L22:mlp[957:1017] conflict<-explicit | 10.577 | 0.012 | 1.239 | L26:attn_out | 0.060 |

## Strict Interpretation

- Final-state projection fractions are comparable to Phase739 alpha thresholds.
- Component projections are pre-final-norm raw signals and should be treated as backtrace candidates, not causal closure proof.

Atlas graph: nodes=36 edges=33
