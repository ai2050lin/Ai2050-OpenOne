# Phase 803 Semantic New-Blocker Source Localization (confirm)

- Status: `complete`
- Boundary: tracks the same semantic new blockers released at alpha=0 across target-direction doses.
- A lower new-blocker rate is not counted as true suppression unless matched semantic blocker logits drop.

## Matched Semantic New Blockers By Alpha

| model | alpha | rows | cases | target gain | target gain vs a0 | old suppress | new rate | matched delta vs a0 | true suppress vs a0 | still above | label counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 12 | 6 | -0.542 | 0.000 | 0.747 | 0.392 | 0.000 | 0.000 | 1.000 | `{"semantic_new_blockers_persist": 12}` |
| qwen3 | 0.500 | 12 | 6 | 1.156 | 1.698 | 0.716 | 0.136 | 0.046 | -0.046 | 0.475 | `{"semantic_new_blockers_persist": 5, "threshold_cover_not_true_suppression": 7}` |
| qwen3 | 0.750 | 12 | 6 | 2.018 | 2.560 | 0.699 | 0.071 | 0.071 | -0.071 | 0.225 | `{"semantic_new_blockers_persist": 3, "threshold_cover_not_true_suppression": 9}` |
| qwen3 | 1.000 | 12 | 6 | 2.852 | 3.393 | 0.691 | 0.041 | 0.092 | -0.092 | 0.153 | `{"semantic_new_blockers_persist": 2, "threshold_cover_not_true_suppression": 10}` |
| qwen3 | 1.250 | 12 | 6 | 3.690 | 4.232 | 0.678 | 0.027 | 0.113 | -0.113 | 0.072 | `{"threshold_cover_not_true_suppression": 12}` |
| glm4 | 0.000 | 12 | 6 | 0.174 | 0.000 | 0.418 | 0.193 | 0.000 | 0.000 | 1.000 | `{"semantic_blockers_below_target_no_logit_suppression": 3, "semantic_new_blockers_persist": 9}` |
| glm4 | 0.500 | 12 | 6 | 0.678 | 0.505 | 0.421 | 0.105 | -0.001 | 0.001 | 0.548 | `{"semantic_blockers_below_target_no_logit_suppression": 3, "semantic_new_blockers_persist": 7, "threshold_cover_not_true_suppression": 2}` |
| glm4 | 0.750 | 12 | 6 | 0.931 | 0.757 | 0.418 | 0.083 | 0.003 | -0.003 | 0.417 | `{"semantic_blockers_below_target_no_logit_suppression": 3, "semantic_new_blockers_persist": 6, "threshold_cover_not_true_suppression": 2, "weak_or_mixed": 1}` |
| glm4 | 1.000 | 12 | 6 | 1.146 | 0.972 | 0.419 | 0.068 | 0.003 | -0.003 | 0.310 | `{"semantic_blockers_below_target_no_logit_suppression": 3, "semantic_new_blockers_persist": 5, "threshold_cover_not_true_suppression": 3, "weak_or_mixed": 1}` |
| glm4 | 1.250 | 12 | 6 | 1.355 | 1.182 | 0.420 | 0.059 | 0.006 | -0.006 | 0.253 | `{"semantic_blockers_below_target_no_logit_suppression": 4, "semantic_new_blockers_persist": 2, "threshold_cover_not_true_suppression": 5, "weak_or_mixed": 1}` |
| deepseek7b | 0.000 | 12 | 6 | 0.702 | 0.000 | -0.424 | 0.427 | 0.000 | 0.000 | 1.000 | `{"semantic_new_blockers_persist": 12}` |
| deepseek7b | 0.500 | 12 | 6 | 1.859 | 1.156 | -0.474 | 0.198 | 0.041 | -0.041 | 0.831 | `{"semantic_new_blockers_persist": 10, "threshold_cover_not_true_suppression": 2}` |
| deepseek7b | 0.750 | 12 | 6 | 2.436 | 1.734 | -0.501 | 0.143 | 0.061 | -0.061 | 0.711 | `{"semantic_new_blockers_persist": 9, "threshold_cover_not_true_suppression": 3}` |
| deepseek7b | 1.000 | 12 | 6 | 3.016 | 2.313 | -0.530 | 0.116 | 0.080 | -0.080 | 0.641 | `{"semantic_new_blockers_persist": 8, "threshold_cover_not_true_suppression": 4}` |
| deepseek7b | 1.250 | 12 | 6 | 3.591 | 2.888 | -0.556 | 0.100 | 0.101 | -0.101 | 0.473 | `{"semantic_new_blockers_persist": 6, "threshold_cover_not_true_suppression": 6}` |

## Top Component Sources

| model | component | rows | cases | semantic new | overlap | jaccard | gap | logit delta | release score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `mlp:L35` | 12 | 6 | 49.083 | 19.750 | 0.265 | 1.222 | 1.239 | 71.055 |
| qwen3 | `mlp:L34` | 12 | 6 | 18.417 | 5.917 | 0.078 | 0.763 | 1.420 | 20.849 |
| qwen3 | `mlp:L33` | 6 | 6 | 10.167 | 4.000 | 0.050 | 0.258 | 1.412 | 2.865 |
| qwen3 | `attn:L34` | 6 | 6 | 23.667 | 3.333 | 0.040 | 0.480 | 0.981 | 14.573 |
| qwen3 | `attn:L35` | 6 | 6 | 6.833 | 3.333 | 0.049 | 0.406 | 1.335 | 1.958 |
| qwen3 | `attn:L31` | 6 | 6 | 13.500 | 2.000 | 0.019 | 0.381 | 1.387 | 7.516 |
| glm4 | `mlp:L38` | 12 | 6 | 26.833 | 21.833 | 0.458 | 0.410 | 0.857 | 13.461 |
| glm4 | `mlp:L39` | 6 | 6 | 34.167 | 21.667 | 0.350 | 0.377 | 0.472 | 16.068 |
| glm4 | `mlp:L27` | 6 | 6 | 21.833 | 11.333 | 0.191 | 0.205 | 0.253 | 3.266 |
| glm4 | `mlp:L34` | 6 | 6 | 16.167 | 7.500 | 0.138 | 0.106 | 0.144 | 1.815 |
| glm4 | `attn:L33` | 6 | 6 | 11.167 | 5.000 | 0.070 | 0.193 | 0.535 | 2.099 |
| glm4 | `attn:L35` | 6 | 6 | 8.167 | 2.333 | 0.039 | 0.119 | 0.224 | 0.600 |
| glm4 | `attn:L29` | 6 | 6 | 8.667 | 2.167 | 0.041 | 0.138 | 0.349 | 1.099 |
| deepseek7b | `attn:L19` | 6 | 6 | 38.667 | 22.167 | 0.392 | 1.193 | 3.691 | 55.630 |
| deepseek7b | `mlp:L27` | 12 | 6 | 43.583 | 17.917 | 0.205 | 1.557 | 2.404 | 73.451 |
| deepseek7b | `mlp:L26` | 6 | 6 | 39.667 | 16.000 | 0.217 | 0.546 | 1.493 | 29.531 |
| deepseek7b | `mlp:L24` | 6 | 6 | 41.500 | 11.333 | 0.138 | 0.614 | 0.879 | 29.951 |
| deepseek7b | `attn:L26` | 6 | 6 | 46.000 | 9.667 | 0.131 | 0.354 | 0.710 | 19.027 |
| deepseek7b | `attn:L27` | 6 | 6 | 26.500 | 8.333 | 0.165 | 0.461 | 0.701 | 21.734 |
| deepseek7b | `attn:L25` | 6 | 6 | 40.000 | 6.167 | 0.115 | 0.194 | 0.478 | 7.804 |
