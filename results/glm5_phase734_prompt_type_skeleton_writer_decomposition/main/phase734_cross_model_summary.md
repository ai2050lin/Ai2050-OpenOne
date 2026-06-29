# Phase 734 Prompt-Type Skeleton Writer Decomposition (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: component ablation -> downstream prompt-type skeleton loss.

| model | target site | top attention | attn loss | attn logprob | top MLP group | MLP loss | MLP logprob |
|---|---|---|---:|---:|---|---:|---:|
| qwen3 | hidden_36 | L26H4 | 6.811 | -0.003 | L35:mlp[320:640] | 7.439 | 0.011 |
| glm4 | hidden_40 | L23H13 | 1.344 | 0.010 | L38:mlp[2048:2560] | 12.041 | 0.018 |
| deepseek7b | hidden_28 | L21H19 | 17.643 | -0.022 | L27:mlp[2688:3136] | 100.977 | -0.401 |

## Strict Interpretation

- Positive skeleton loss means ablation moved the explicit path away from the explicit-vs-commonsense downstream direction.
- A component is only a writer candidate when skeleton loss is positive and target likelihood is hurt.
- This is component-level v0, not neuron-level proof.

Atlas graph: nodes=57 edges=54
