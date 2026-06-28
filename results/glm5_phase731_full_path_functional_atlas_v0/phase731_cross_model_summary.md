# Phase 731 Full-Path Functional Atlas v0

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: absolute trajectory + factor mean-difference + candidate-head attention.

| model | cases | category hit | color hit | taste hit | top factor effect | effect norm |
|---|---:|---:|---:|---:|---|---:|
| qwen3 | 66 | 0.955 | 1.000 | 0.909 | prompt_type/commonsense@hidden_35 | 190.465 |
| glm4 | 66 | 0.955 | 1.000 | 1.000 | prompt_type/commonsense@hidden_39 | 89.866 |
| deepseek7b | 66 | 0.500 | 0.682 | 0.727 | prompt_type/commonsense@L27_mlp_out | 394.751 |

## Strict Interpretation

- This is a v0 full-path descriptive atlas, not a causal closure proof.
- Factor effects are mean-vector differences against the global centroid.
- Candidate-head attention is observational; causal edge validation must follow.

Atlas graph: nodes=70 edges=78
