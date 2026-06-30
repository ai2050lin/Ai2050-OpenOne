# Phase 803 Semantic New-Blocker Source Localization (main)

- Status: `complete`
- Boundary: tracks the same semantic new blockers released at alpha=0 across target-direction doses.
- A lower new-blocker rate is not counted as true suppression unless matched semantic blocker logits drop.

## Matched Semantic New Blockers By Alpha

| model | alpha | rows | cases | target gain | target gain vs a0 | old suppress | new rate | matched delta vs a0 | true suppress vs a0 | still above | label counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 8 | 4 | -0.703 | 0.000 | 0.756 | 0.418 | 0.000 | 0.000 | 1.000 | `{"semantic_new_blockers_persist": 8}` |
| qwen3 | 0.500 | 8 | 4 | 1.039 | 1.742 | 0.745 | 0.139 | 0.026 | -0.026 | 0.611 | `{"semantic_new_blockers_persist": 4, "threshold_cover_not_true_suppression": 4}` |
| qwen3 | 0.750 | 8 | 4 | 1.926 | 2.629 | 0.734 | 0.070 | 0.047 | -0.047 | 0.299 | `{"semantic_new_blockers_persist": 2, "threshold_cover_not_true_suppression": 6}` |
| qwen3 | 1.000 | 8 | 4 | 2.785 | 3.488 | 0.734 | 0.035 | 0.057 | -0.057 | 0.209 | `{"semantic_new_blockers_persist": 2, "threshold_cover_not_true_suppression": 6}` |
| glm4 | 0.000 | 8 | 4 | 0.043 | 0.000 | 0.395 | 0.238 | 0.000 | 0.000 | 1.000 | `{"semantic_blockers_below_target_no_logit_suppression": 1, "semantic_new_blockers_persist": 7}` |
| glm4 | 0.500 | 8 | 4 | 0.453 | 0.410 | 0.399 | 0.148 | -0.004 | 0.004 | 0.623 | `{"semantic_blockers_below_target_no_logit_suppression": 1, "semantic_new_blockers_persist": 6, "threshold_cover_not_true_suppression": 1}` |
| glm4 | 0.750 | 8 | 4 | 0.664 | 0.621 | 0.395 | 0.119 | -0.001 | 0.001 | 0.541 | `{"semantic_blockers_below_target_no_logit_suppression": 1, "semantic_new_blockers_persist": 5, "threshold_cover_not_true_suppression": 1, "weak_or_mixed": 1}` |
| glm4 | 1.000 | 8 | 4 | 0.834 | 0.791 | 0.400 | 0.098 | -0.003 | 0.003 | 0.421 | `{"semantic_blockers_below_target_no_logit_suppression": 1, "semantic_new_blockers_persist": 4, "threshold_cover_not_true_suppression": 2, "weak_or_mixed": 1}` |
| deepseek7b | 0.000 | 8 | 4 | 0.938 | 0.000 | -0.415 | 0.367 | 0.000 | 0.000 | 1.000 | `{"semantic_new_blockers_persist": 8}` |
| deepseek7b | 0.500 | 8 | 4 | 2.241 | 1.303 | -0.474 | 0.156 | 0.050 | -0.050 | 0.750 | `{"semantic_new_blockers_persist": 6, "threshold_cover_not_true_suppression": 2}` |
| deepseek7b | 0.750 | 8 | 4 | 2.896 | 1.958 | -0.513 | 0.129 | 0.080 | -0.080 | 0.589 | `{"semantic_new_blockers_persist": 5, "threshold_cover_not_true_suppression": 3}` |
| deepseek7b | 1.000 | 8 | 4 | 3.557 | 2.618 | -0.549 | 0.124 | 0.101 | -0.101 | 0.500 | `{"semantic_new_blockers_persist": 4, "threshold_cover_not_true_suppression": 4}` |

## Top Component Sources

| model | component | rows | cases | semantic new | overlap | jaccard | gap | logit delta | release score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `mlp:L35` | 8 | 4 | 46.500 | 20.125 | 0.296 | 1.621 | 1.421 | 76.973 |
| qwen3 | `mlp:L34` | 8 | 4 | 21.250 | 6.500 | 0.097 | 0.806 | 1.380 | 24.887 |
| qwen3 | `mlp:L33` | 4 | 4 | 15.000 | 5.500 | 0.081 | 0.324 | 1.507 | 4.281 |
| qwen3 | `attn:L35` | 4 | 4 | 10.250 | 5.000 | 0.090 | 0.406 | 1.335 | 2.938 |
| qwen3 | `attn:L34` | 4 | 4 | 24.250 | 3.000 | 0.036 | 0.527 | 1.231 | 16.773 |
| qwen3 | `attn:L31` | 4 | 4 | 15.250 | 2.000 | 0.023 | 0.454 | 1.655 | 8.977 |
| glm4 | `mlp:L39` | 4 | 4 | 27.250 | 19.500 | 0.451 | 0.443 | 0.354 | 17.023 |
| glm4 | `mlp:L38` | 8 | 4 | 22.250 | 18.625 | 0.544 | 0.320 | 0.756 | 7.301 |
| glm4 | `mlp:L27` | 4 | 4 | 16.000 | 10.750 | 0.256 | 0.192 | 0.227 | 2.184 |
| glm4 | `attn:L33` | 4 | 4 | 16.250 | 6.750 | 0.101 | 0.199 | 0.273 | 3.059 |
| glm4 | `mlp:L34` | 4 | 4 | 8.250 | 5.000 | 0.160 | 0.102 | 0.120 | 0.777 |
| glm4 | `attn:L35` | 4 | 4 | 3.000 | 1.500 | 0.041 | 0.140 | 0.251 | 0.371 |
| glm4 | `attn:L29` | 4 | 4 | 3.500 | 1.250 | 0.040 | 0.143 | 0.355 | 0.516 |
| deepseek7b | `attn:L19` | 4 | 4 | 22.000 | 12.500 | 0.403 | 1.382 | 4.030 | 44.809 |
| deepseek7b | `mlp:L27` | 8 | 4 | 25.375 | 10.500 | 0.177 | 1.819 | 2.604 | 53.240 |
| deepseek7b | `attn:L27` | 4 | 4 | 30.750 | 9.250 | 0.220 | 0.628 | 0.573 | 26.514 |
| deepseek7b | `mlp:L26` | 4 | 4 | 23.500 | 8.250 | 0.188 | 0.549 | 1.559 | 21.033 |
| deepseek7b | `attn:L26` | 4 | 4 | 29.000 | 8.000 | 0.159 | 0.461 | 0.848 | 18.474 |
| deepseek7b | `mlp:L24` | 4 | 4 | 26.250 | 6.000 | 0.119 | 0.696 | 0.903 | 24.079 |
| deepseek7b | `attn:L25` | 4 | 4 | 24.000 | 4.750 | 0.141 | 0.206 | 0.518 | 5.226 |
