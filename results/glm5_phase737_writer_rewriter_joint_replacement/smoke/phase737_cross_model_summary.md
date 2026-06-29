# Phase 737 Writer-Rewriter Joint Replacement (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: joint replacement of source writer and MLP rewriter candidates.

| model | target site | top intervention | restore | donor logprob | margin delta | donor hit gain | changed | role |
|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | hidden_36 | mlp_only|no_source|L34:mlp[85:128] conflict<-explicit | 6.550 | 0.254 | 0.250 | 0.000 | 0.000 | joint_readout_transfer_candidate |
| glm4 | hidden_40 | source_plus_top_mlp|L39H21<-self_last|L38:mlp[2597:2665] conflict<-explicit | -0.196 | 0.180 | 0.188 | 0.000 | 0.000 | readout_transfer_only |
| deepseek7b | hidden_28 | source_only|L22H24<-records_all|no_mlp explicit<-conflict | 6.179 | 0.022 | -0.078 | 0.000 | 1.000 | joint_state_likelihood_transfer |

## Strict Interpretation

- A positive restore projection means the target hidden state moved toward the donor state.
- A positive margin delta means donor answer readout improved against the recipient answer, not just in isolation.
- Generation hit gain remains the strictest closure criterion.

Atlas graph: nodes=48 edges=78
