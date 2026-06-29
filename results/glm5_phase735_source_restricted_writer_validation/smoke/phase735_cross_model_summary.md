# Phase 735 Source-Restricted Writer Validation (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: source-token-group contribution erasure for candidate attention heads; fine MLP subgroup ablation.

| model | target site | top source path | source loss | source logprob | attention mass | top MLP fine | MLP loss | MLP logprob |
|---|---|---|---:|---:|---:|---|---:|---:|
| qwen3 | hidden_36 | L35H0<-self_last | 11.225 | -0.014 | 0.951 | L28:mlp[256:384] | 7.985 | 0.007 |
| glm4 | hidden_40 | L39H21<-self_last | 8.068 | -0.004 | 1.000 | L38:mlp[3075:3280] | 5.068 | -0.027 |
| deepseek7b | hidden_28 | L22H24<-all_pre_answer | 16.094 | -0.312 | 0.992 | L27:mlp[2872:3052] | 76.638 | 0.105 |

## Strict Interpretation

- Source-restricted erasure shows which source token group contributed through a candidate head to the downstream skeleton direction.
- This is stronger than a head ranking, but it is still not a full neuron-level proof.
- MLP fine decomposition narrows output-channel groups; it does not yet identify individual hidden neurons.

Atlas graph: nodes=51 edges=78
