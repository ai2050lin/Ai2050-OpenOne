# Phase 737 Writer-Rewriter Joint Replacement (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: joint replacement of source writer and MLP rewriter candidates.

| model | target site | top intervention | restore | donor logprob | margin delta | donor hit gain | changed | role |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | hidden_36 | source_plus_all_mlp|L28H28<-instruction|L34:mlp[85:128]+L28:mlp[299:341] conflict<-explicit | 0.991 | 0.078 | 0.078 | 0.000 | 0.000 | joint_readout_transfer_candidate |
| glm4 | hidden_40 | source_plus_all_mlp|L23H17<-instruction|L38:mlp[2597:2665]+L38:mlp[3007:3075] conflict<-explicit | 2.937 | 0.266 | 0.270 | 0.000 | 0.000 | joint_readout_transfer_candidate |
| deepseek7b | hidden_28 | mlp_all|no_source|L27:mlp[2872:2932]+L22:mlp[957:1017] conflict<-explicit | 11.891 | 0.056 | 0.180 | 0.000 | 0.500 | joint_readout_transfer_candidate |

## Strict Interpretation

- A positive restore projection means the target hidden state moved toward the donor state.
- A positive margin delta means donor answer readout improved against the recipient answer, not just in isolation.
- Generation hit gain remains the strictest closure criterion.

Atlas graph: nodes=72 edges=155
