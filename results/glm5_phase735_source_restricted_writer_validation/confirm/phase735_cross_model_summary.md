# Phase 735 Source-Restricted Writer Validation (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: source-token-group contribution erasure for candidate attention heads; fine MLP subgroup ablation.

| model | target site | top source path | source loss | source logprob | attention mass | top MLP fine | MLP loss | MLP logprob |
|---|---|---|---:|---:|---:|---|---:|---:|
| qwen3 | hidden_36 | L35H0<-self_last | 6.537 | -0.002 | 0.947 | L34:mlp[43:85] | 3.681 | 0.005 |
| glm4 | hidden_40 | L39H21<-self_last | 8.997 | -0.005 | 1.000 | L38:mlp[3212:3280] | 3.267 | 0.003 |
| deepseek7b | hidden_28 | L22H24<-all_pre_answer | 24.993 | -0.187 | 0.981 | L27:mlp[2872:2932] | 66.368 | -0.141 |

## Strict Interpretation

- Source-restricted erasure shows which source token group contributed through a candidate head to the downstream skeleton direction.
- This is stronger than a head ranking, but it is still not a full neuron-level proof.
- MLP fine decomposition narrows output-channel groups; it does not yet identify individual hidden neurons.

Atlas graph: nodes=87 edges=114
