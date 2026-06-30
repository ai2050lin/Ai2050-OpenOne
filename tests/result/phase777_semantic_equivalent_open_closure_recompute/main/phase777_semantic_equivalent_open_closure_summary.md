# Phase 777 Semantic-Equivalent Open Closure Recompute (main)

- Source: `tests/result/phase776_readout_bridge_competition_audit/main`
- Model loading: not used; this is a Phase 776 result recomputation.
- Semantic-equivalent classes: `target_value`, `case_variant_target_value`.

## By Prompt

| model | variant | n | strict open | semantic-equivalent open | gain | pool top1 | latent hit | hard readout after equiv |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `constrained_free_prompt` | 7 | 0.000 | 0.857 | 0.857 | 0.857 | 0.857 | 0.000 |
| qwen3 | `with_candidate_list` | 7 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| qwen3 | `without_candidate_list` | 7 | 0.000 | 0.714 | 0.714 | 0.857 | 0.857 | 0.143 |
| glm4 | `constrained_free_prompt` | 8 | 0.000 | 0.750 | 0.750 | 0.625 | 0.625 | 0.125 |
| glm4 | `with_candidate_list` | 8 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| glm4 | `without_candidate_list` | 8 | 0.000 | 0.750 | 0.750 | 1.000 | 1.000 | 0.250 |
| deepseek7b | `constrained_free_prompt` | 8 | 0.125 | 0.750 | 0.625 | 0.750 | 0.625 | 0.125 |
| deepseek7b | `with_candidate_list` | 8 | 0.375 | 1.000 | 0.625 | 0.750 | 0.375 | 0.000 |
| deepseek7b | `without_candidate_list` | 8 | 0.125 | 0.750 | 0.625 | 0.625 | 0.500 | 0.000 |

## Latent-Hit Reclassification

| model | variant | latent n | latent semantic-equivalent open | rate |
|---|---|---:|---:|---:|
| qwen3 | `constrained_free_prompt` | 6 | 6 | 1.000 |
| qwen3 | `without_candidate_list` | 6 | 5 | 0.833 |
| glm4 | `constrained_free_prompt` | 5 | 4 | 0.800 |
| glm4 | `without_candidate_list` | 8 | 6 | 0.750 |
| deepseek7b | `constrained_free_prompt` | 5 | 4 | 0.800 |
| deepseek7b | `with_candidate_list` | 3 | 3 | 1.000 |
| deepseek7b | `without_candidate_list` | 4 | 4 | 1.000 |

## By Domain

| model | domain | n | strict open | semantic-equivalent open | gain | pool top1 | hard readout after equiv |
|---|---|---:|---:|---:|---:|---:|---:|
| deepseek7b | `abstract` | 6 | 0.000 | 0.667 | 0.667 | 0.000 | 0.000 |
| deepseek7b | `animal` | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| deepseek7b | `fruit` | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| deepseek7b | `object` | 3 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| deepseek7b | `plant` | 6 | 0.333 | 0.667 | 0.333 | 0.833 | 0.167 |
| deepseek7b | `tool` | 3 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| glm4 | `abstract` | 6 | 0.333 | 0.333 | 0.000 | 0.833 | 0.500 |
| glm4 | `animal` | 3 | 0.333 | 1.000 | 0.667 | 0.667 | 0.000 |
| glm4 | `object` | 9 | 0.333 | 1.000 | 0.667 | 0.889 | 0.000 |
| glm4 | `plant` | 6 | 0.333 | 1.000 | 0.667 | 1.000 | 0.000 |
| qwen3 | `abstract` | 3 | 0.333 | 0.333 | 0.000 | 0.333 | 0.000 |
| qwen3 | `fruit` | 9 | 0.333 | 1.000 | 0.667 | 1.000 | 0.000 |
| qwen3 | `object` | 6 | 0.333 | 0.833 | 0.500 | 1.000 | 0.167 |
| qwen3 | `plant` | 3 | 0.333 | 1.000 | 0.667 | 1.000 | 0.000 |

## Strict Interpretation

- This is an evaluation correction, not a new causal intervention.
- A semantic-equivalent open hit means the top1 token matches the target value after surface-form normalization.
- Remaining hard readout failures are cases where the value pool is correct but top1 is not target-equivalent.
