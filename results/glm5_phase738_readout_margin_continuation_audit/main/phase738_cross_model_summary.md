# Phase 738 Readout Margin and Token Continuation Gate Audit (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: token0 candidate competition and forced donor-token continuation.

| model | target site | top audit | margin delta | patched margin | donor top rate | top patched competitor counts | token1 counts |
|---|---|---|---:|---:|---:|---|---|
| qwen3 | hidden_36 | source_plus_all_mlp|L35H0<-self_last|L34:mlp[85:128]+L28:mlp[299:341] explicit<-conflict | 0.125 | -17.016 | 0.000 | {'recipient_answer': 8} | {'cont_is': 4, 'cont_stop_newline': 4} |
| glm4 | hidden_40 | source_plus_all_mlp|L23H17<-all_pre_answer|L38:mlp[2597:2665]+L38:mlp[3007:3075] conflict<-explicit | 0.351 | -8.867 | 0.000 | {'recipient_answer': 8} | {'cont_of': 1, 'relation_echo': 7} |
| deepseek7b | hidden_28 | source_plus_all_mlp|L22H24<-target_record_line|L27:mlp[2872:2932]+L22:mlp[957:1017] explicit<-conflict | 0.520 | -12.933 | 0.000 | {'object_echo': 1, 'recipient_answer': 6, 'relation_echo': 1} | {'cont_is': 7, 'cont_of': 1} |

## Strict Interpretation

- Positive margin delta only means donor answer improved against recipient answer.
- Negative patched margin means donor answer still loses readout competition.
- Token1 counts show what continuation route is preferred after forcing donor token0.

Atlas graph: nodes=30 edges=45
