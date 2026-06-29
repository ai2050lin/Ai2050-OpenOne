# Phase 736 Source-Restricted Replacement and Generation Closure (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: source contribution replacement from donor prompt to recipient prompt, with likelihood and greedy generation checks.

| model | target site | top replacement path | restore | donor logprob | donor hit gain | changed | role |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | hidden_36 | L35H0<-self_last explicit<-conflict | 1.349 | 0.023 | 0.000 | 0.000 | content_transfer_candidate |
| glm4 | hidden_40 | L23H17<-all_pre_answer explicit<-conflict | 1.284 | 0.057 | 0.000 | 0.000 | content_transfer_candidate |
| deepseek7b | hidden_28 | L22H24<-all_pre_answer conflict<-explicit | 19.917 | 0.074 | 0.000 | 0.000 | content_transfer_candidate |

## Strict Interpretation

- Positive restore projection means the recipient hidden state moved toward the donor hidden state after source contribution replacement.
- Positive donor logprob delta means the donor answer became more supported in the recipient context.
- Generation hit gain is the strictest metric and can remain sparse even when hidden/readout effects are present.

Atlas graph: nodes=26 edges=30
