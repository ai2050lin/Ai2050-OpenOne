# Phase 734 Prompt-Type Skeleton Writer Decomposition (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: component ablation -> downstream prompt-type skeleton loss.

| model | target site | top attention | attn loss | attn logprob | top MLP group | MLP loss | MLP logprob |
|---|---|---|---:|---:|---|---:|---:|
| qwen3 | hidden_36 | L35H10 | 1.402 | 0.003 | L28:mlp[0:640] | 12.422 | -0.005 |
| glm4 | hidden_40 | L39H21 | 9.076 | -0.006 | L23:mlp[0:1024] | 5.511 | -0.012 |
| deepseek7b | hidden_28 | L21H0 | 8.082 | 0.044 | L27:mlp[2688:3584] | 85.372 | -1.218 |

## Strict Interpretation

- Positive skeleton loss means ablation moved the explicit path away from the explicit-vs-commonsense downstream direction.
- A component is only a writer candidate when skeleton loss is positive and target likelihood is hurt.
- This is component-level v0, not neuron-level proof.

Atlas graph: nodes=57 edges=54
