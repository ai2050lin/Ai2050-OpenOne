# Phase 738 Readout Margin and Token Continuation Gate Audit (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: token0 candidate competition and forced donor-token continuation.

| model | target site | top audit | margin delta | patched margin | donor top rate | top patched competitor counts | token1 counts |
|---|---|---|---:|---:|---:|---|---|
| qwen3 | hidden_36 | source_plus_all_mlp|L35H0<-self_last|L34:mlp[85:128]+L28:mlp[299:341] explicit<-conflict | -0.375 | -19.250 | 0.000 | {'recipient_answer': 1} | {'cont_is': 1} |
| glm4 | hidden_40 | source_plus_all_mlp|L23H17<-all_pre_answer|L38:mlp[2597:2665]+L38:mlp[3007:3075] conflict<-explicit | 0.375 | -5.938 | 0.000 | {'recipient_answer': 1} | {'relation_echo': 1} |
| deepseek7b | hidden_28 | mlp_only|no_source|L27:mlp[2872:2932] conflict<-explicit | 0.000 | -7.812 | 0.000 | {'recipient_answer': 1} | {'cont_is': 1} |

## Strict Interpretation

- Positive margin delta only means donor answer improved against recipient answer.
- Negative patched margin means donor answer still loses readout competition.
- Token1 counts show what continuation route is preferred after forcing donor token0.

Atlas graph: nodes=18 edges=18
