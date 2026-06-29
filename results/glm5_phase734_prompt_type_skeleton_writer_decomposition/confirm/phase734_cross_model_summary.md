# Phase 734 Prompt-Type Skeleton Writer Decomposition (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: component ablation -> downstream prompt-type skeleton loss.

| model | target site | top attention | attn loss | attn logprob | top MLP group | MLP loss | MLP logprob |
|---|---|---|---:|---:|---|---:|---:|
| qwen3 | hidden_36 | L35H0 | 7.007 | -0.001 | L35:mlp[256:512] | 6.486 | 0.005 |
| glm4 | hidden_40 | L39H21 | 9.445 | -0.008 | L38:mlp[2870:3280] | 8.267 | -0.003 |
| deepseek7b | hidden_28 | L22H24 | 24.410 | -0.132 | L27:mlp[2872:3231] | 108.435 | -0.698 |

## Strict Interpretation

- Positive skeleton loss means ablation moved the explicit path away from the explicit-vs-commonsense downstream direction.
- A component is only a writer candidate when skeleton loss is positive and target likelihood is hurt.
- This is component-level v0, not neuron-level proof.

Atlas graph: nodes=57 edges=54
