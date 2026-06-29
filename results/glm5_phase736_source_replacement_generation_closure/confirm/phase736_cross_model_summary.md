# Phase 736 Source-Restricted Replacement and Generation Closure (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: source contribution replacement from donor prompt to recipient prompt, with likelihood and greedy generation checks.

| model | target site | top replacement path | restore | donor logprob | donor hit gain | changed | role |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | hidden_36 | L35H0<-self_last conflict<-explicit | 1.803 | -0.151 | 0.000 | 0.000 | state_transfer_only |
| glm4 | hidden_40 | L23H17<-all_pre_answer explicit<-conflict | 0.805 | 0.041 | 0.000 | 0.000 | content_transfer_candidate |
| deepseek7b | hidden_28 | L22H24<-all_pre_answer conflict<-explicit | 15.831 | 0.286 | 0.000 | 0.167 | content_transfer_candidate |

## Strict Interpretation

- Positive restore projection means the recipient hidden state moved toward the donor hidden state after source contribution replacement.
- Positive donor logprob delta means the donor answer became more supported in the recipient context.
- Generation hit gain is the strictest metric and can remain sparse even when hidden/readout effects are present.

Atlas graph: nodes=38 edges=54
