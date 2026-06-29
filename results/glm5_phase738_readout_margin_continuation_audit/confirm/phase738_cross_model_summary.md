# Phase 738 Readout Margin and Token Continuation Gate Audit (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: token0 candidate competition and forced donor-token continuation.

| model | target site | top audit | margin delta | patched margin | donor top rate | top patched competitor counts | token1 counts |
|---|---|---|---:|---:|---:|---|---|
| qwen3 | hidden_36 | source_plus_all_mlp|L35H0<-self_last|L34:mlp[85:128]+L28:mlp[299:341] explicit<-conflict | 0.099 | -15.651 | 0.000 | {'recipient_answer': 12} | {'cont_is': 6, 'cont_stop_newline': 6} |
| glm4 | hidden_40 | source_plus_all_mlp|L39H21<-self_last|L38:mlp[2597:2665]+L38:mlp[3007:3075] conflict<-explicit | 0.310 | -8.244 | 0.000 | {'recipient_answer': 12} | {'cont_is': 3, 'cont_of': 1, 'object_echo': 1, 'relation_echo': 7} |
| deepseek7b | hidden_28 | source_plus_all_mlp|L22H24<-target_record_line|L27:mlp[2872:2932]+L22:mlp[957:1017] explicit<-conflict | 0.411 | -11.932 | 0.000 | {'object_echo': 1, 'recipient_answer': 10, 'relation_echo': 1} | {'cont_is': 11, 'cont_of': 1} |

## Strict Interpretation

- Positive margin delta only means donor answer improved against recipient answer.
- Negative patched margin means donor answer still loses readout competition.
- Token1 counts show what continuation route is preferred after forcing donor token0.

Atlas graph: nodes=30 edges=46
