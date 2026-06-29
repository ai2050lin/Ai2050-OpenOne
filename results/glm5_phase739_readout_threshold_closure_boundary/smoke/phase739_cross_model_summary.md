# Phase 739 Readout Threshold and Closure Boundary Test (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: minimum final-readout boost and boosted generation closure.

| model | target site | top audit | mean alpha* vocab | vocab flip found | boosted donor hit | generation classes |
|---|---|---|---:|---:|---:|---|
| qwen3 | hidden_36 | source_plus_all_mlp|L35H0<-self_last|L34:mlp[85:128]+L28:mlp[299:341] explicit<-conflict | 12.958 | 1.000 | 1.000 | {'answer_stop': 1} |
| glm4 | hidden_40 | source_plus_all_mlp|L39H21<-self_last|L38:mlp[2597:2665]+L38:mlp[3007:3075] conflict<-explicit | 6.828 | 1.000 | 1.000 | {'answer_stop': 1} |
| deepseek7b | hidden_28 | source_plus_all_mlp|L22H24<-target_record_line|L27:mlp[2872:2932]+L22:mlp[957:1017] explicit<-conflict | 11.125 | 0.000 | 0.000 | {} |

## Strict Interpretation

- This phase applies an artificial final readout boost along donor-vs-current-top direction.
- If donor token0 flips only after a large alpha, the natural writer/rewriter path is far from closure.
- If boosted generation still leaves the answer route, continuation closure remains a separate bottleneck.

Atlas graph: nodes=9 edges=6
