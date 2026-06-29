# Phase 735 Source-Restricted Writer Validation (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: source-token-group contribution erasure for candidate attention heads; fine MLP subgroup ablation.

| model | target site | top source path | source loss | source logprob | attention mass | top MLP fine | MLP loss | MLP logprob |
|---|---|---|---:|---:|---:|---|---:|---:|
| qwen3 | hidden_36 | L35H0<-self_last | 10.046 | -0.002 | 0.949 | L34:mlp[64:128] | 7.812 | 0.001 |
| glm4 | hidden_40 | L39H21<-self_last | 8.934 | -0.007 | 1.000 | L38:mlp[3178:3280] | 3.863 | 0.009 |
| deepseek7b | hidden_28 | L22H24<-all_pre_answer | 28.170 | -0.187 | 0.989 | L27:mlp[2872:2962] | 84.004 | -0.229 |

## Strict Interpretation

- Source-restricted erasure shows which source token group contributed through a candidate head to the downstream skeleton direction.
- This is stronger than a head ranking, but it is still not a full neuron-level proof.
- MLP fine decomposition narrows output-channel groups; it does not yet identify individual hidden neurons.

Atlas graph: nodes=75 edges=102
