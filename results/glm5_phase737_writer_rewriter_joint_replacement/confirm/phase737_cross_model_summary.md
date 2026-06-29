# Phase 737 Writer-Rewriter Joint Replacement (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: joint replacement of source writer and MLP rewriter candidates.

| model | target site | top intervention | restore | donor logprob | margin delta | donor hit gain | changed | role |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | hidden_36 | source_plus_all_mlp|L35H0<-self_last|L34:mlp[85:128]+L28:mlp[299:341] explicit<-conflict | 1.091 | 0.119 | 0.125 | 0.000 | 0.000 | joint_readout_transfer_candidate |
| glm4 | hidden_40 | source_plus_all_mlp|L23H17<-all_pre_answer|L38:mlp[2597:2665]+L38:mlp[3007:3075] conflict<-explicit | 2.832 | 0.333 | 0.351 | 0.000 | 0.000 | joint_readout_transfer_candidate |
| deepseek7b | hidden_28 | mlp_only|no_source|L27:mlp[2872:2932] conflict<-explicit | 8.444 | 0.108 | 0.231 | 0.000 | 0.250 | joint_readout_transfer_candidate |

## Strict Interpretation

- A positive restore projection means the target hidden state moved toward the donor state.
- A positive margin delta means donor answer readout improved against the recipient answer, not just in isolation.
- Generation hit gain remains the strictest closure criterion.

Atlas graph: nodes=72 edges=150
