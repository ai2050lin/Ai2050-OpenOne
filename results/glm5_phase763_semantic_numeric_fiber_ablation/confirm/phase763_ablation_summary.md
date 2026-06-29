# Phase 763 Semantic Numeric Fiber Ablation (confirm)

- Status: `complete`
- Input: Phase 762 confirm rows; no model was loaded.
- Purpose: identify which causal-fiber components carry same-domain object structure.

## Ablation Results

| model | config | features | NN | same | diff | sep |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `phase762_exact` | 2960 | 0.556 | 0.531 | -0.122 | 0.653 |
| qwen3 | `all_direct4_relation_specific` | 3200 | 0.722 | 0.386 | -0.115 | 0.501 |
| qwen3 | `target_drop_only` | 120 | 0.167 | 0.113 | -0.069 | 0.182 |
| qwen3 | `attention_mass_only` | 120 | 0.667 | 0.497 | -0.086 | 0.583 |
| qwen3 | `direct2_scores_only` | 240 | 0.611 | 0.649 | -0.128 | 0.777 |
| qwen3 | `direct4_scores_only` | 480 | 0.722 | 0.395 | -0.116 | 0.511 |
| qwen3 | `route_release_only` | 1240 | 0.333 | 0.084 | -0.075 | 0.159 |
| qwen3 | `margin_drop_only` | 1240 | 0.389 | 0.159 | -0.080 | 0.240 |
| qwen3 | `no_attention_mass` | 2840 | 0.556 | 0.531 | -0.122 | 0.653 |
| qwen3 | `no_direct_scores` | 2720 | 0.389 | 0.129 | -0.079 | 0.208 |
| qwen3 | `no_route_features` | 480 | 0.611 | 0.640 | -0.128 | 0.768 |
| qwen3 | `records_only` | 1776 | 0.611 | 0.568 | -0.123 | 0.691 |
| qwen3 | `object_relation_sources_only` | 1184 | 0.444 | 0.119 | -0.079 | 0.198 |
| qwen3 | `no_target_value_tokens` | 2368 | 0.556 | 0.515 | -0.122 | 0.636 |
| qwen3 | `no_object_tokens` | 2368 | 0.611 | 0.557 | -0.124 | 0.681 |
| qwen3 | `no_relation_tokens` | 2368 | 0.556 | 0.540 | -0.121 | 0.661 |
| qwen3 | `all_relation_collapsed` | 560 | 0.611 | 0.399 | -0.110 | 0.509 |
| qwen3 | `target_drop_relation_collapsed` | 20 | 0.111 | 0.126 | -0.046 | 0.172 |
| qwen3 | `route_release_relation_collapsed` | 240 | 0.333 | 0.113 | -0.069 | 0.182 |
| glm4 | `phase762_exact` | 3200 | 0.611 | 0.411 | -0.090 | 0.501 |
| glm4 | `all_direct4_relation_specific` | 3440 | 0.889 | 0.279 | -0.103 | 0.381 |
| glm4 | `target_drop_only` | 120 | 0.500 | 0.330 | -0.090 | 0.420 |
| glm4 | `attention_mass_only` | 120 | 0.667 | 0.569 | -0.104 | 0.673 |
| glm4 | `direct2_scores_only` | 240 | 0.722 | 0.802 | -0.029 | 0.831 |
| glm4 | `direct4_scores_only` | 480 | 0.889 | 0.278 | -0.103 | 0.381 |
| glm4 | `route_release_only` | 1360 | 0.444 | 0.227 | -0.089 | 0.316 |
| glm4 | `margin_drop_only` | 1360 | 0.667 | 0.283 | -0.086 | 0.369 |
| glm4 | `no_attention_mass` | 3080 | 0.667 | 0.398 | -0.088 | 0.485 |
| glm4 | `no_direct_scores` | 2960 | 0.667 | 0.297 | -0.091 | 0.388 |
| glm4 | `no_route_features` | 480 | 0.667 | 0.674 | -0.069 | 0.742 |
| glm4 | `records_only` | 1920 | 0.722 | 0.503 | -0.085 | 0.588 |
| glm4 | `object_relation_sources_only` | 1280 | 0.056 | -0.042 | -0.056 | 0.013 |
| glm4 | `no_target_value_tokens` | 2560 | 0.611 | 0.375 | -0.090 | 0.466 |
| glm4 | `no_object_tokens` | 2560 | 0.611 | 0.458 | -0.089 | 0.547 |
| glm4 | `no_relation_tokens` | 2560 | 0.611 | 0.445 | -0.088 | 0.533 |
| glm4 | `all_relation_collapsed` | 560 | 0.556 | 0.377 | -0.083 | 0.460 |
| glm4 | `target_drop_relation_collapsed` | 20 | 0.444 | 0.255 | -0.081 | 0.336 |
| glm4 | `route_release_relation_collapsed` | 240 | 0.389 | 0.172 | -0.082 | 0.254 |
| deepseek7b | `phase762_exact` | 3120 | 0.556 | 0.214 | -0.093 | 0.307 |
| deepseek7b | `all_direct4_relation_specific` | 3360 | 0.833 | 0.280 | -0.102 | 0.381 |
| deepseek7b | `target_drop_only` | 120 | 0.389 | 0.104 | -0.079 | 0.183 |
| deepseek7b | `attention_mass_only` | 120 | 0.667 | 0.431 | -0.119 | 0.550 |
| deepseek7b | `direct2_scores_only` | 240 | 0.667 | 0.301 | -0.100 | 0.402 |
| deepseek7b | `direct4_scores_only` | 480 | 0.833 | 0.291 | -0.103 | 0.394 |
| deepseek7b | `route_release_only` | 1320 | 0.667 | 0.043 | -0.072 | 0.115 |
| deepseek7b | `margin_drop_only` | 1320 | 0.611 | 0.155 | -0.085 | 0.239 |
| deepseek7b | `no_attention_mass` | 3000 | 0.556 | 0.212 | -0.093 | 0.305 |
| deepseek7b | `no_direct_scores` | 2880 | 0.611 | 0.103 | -0.079 | 0.183 |
| deepseek7b | `no_route_features` | 480 | 0.611 | 0.294 | -0.100 | 0.394 |
| deepseek7b | `records_only` | 1872 | 0.667 | 0.236 | -0.096 | 0.331 |
| deepseek7b | `object_relation_sources_only` | 1248 | 0.278 | 0.039 | -0.071 | 0.110 |
| deepseek7b | `no_target_value_tokens` | 2496 | 0.611 | 0.205 | -0.092 | 0.297 |
| deepseek7b | `no_object_tokens` | 2496 | 0.611 | 0.228 | -0.095 | 0.323 |
| deepseek7b | `no_relation_tokens` | 2496 | 0.722 | 0.220 | -0.094 | 0.313 |
| deepseek7b | `all_relation_collapsed` | 560 | 0.389 | 0.157 | -0.085 | 0.241 |
| deepseek7b | `target_drop_relation_collapsed` | 20 | 0.278 | 0.116 | -0.070 | 0.186 |
| deepseek7b | `route_release_relation_collapsed` | 240 | 0.444 | 0.085 | -0.076 | 0.161 |

## Cross-Model Correlations

| config | pair | common pairs | pearson |
|---|---|---:|---:|
| `phase762_exact` | `deepseek7b__glm4` | 153 | 0.287 |
| `phase762_exact` | `deepseek7b__qwen3` | 153 | 0.292 |
| `phase762_exact` | `glm4__qwen3` | 153 | 0.344 |
| `all_direct4_relation_specific` | `deepseek7b__glm4` | 153 | 0.548 |
| `all_direct4_relation_specific` | `deepseek7b__qwen3` | 153 | 0.624 |
| `all_direct4_relation_specific` | `glm4__qwen3` | 153 | 0.574 |
| `target_drop_only` | `deepseek7b__glm4` | 153 | 0.225 |
| `target_drop_only` | `deepseek7b__qwen3` | 153 | 0.114 |
| `target_drop_only` | `glm4__qwen3` | 153 | 0.204 |
| `attention_mass_only` | `deepseek7b__glm4` | 153 | 0.213 |
| `attention_mass_only` | `deepseek7b__qwen3` | 153 | 0.468 |
| `attention_mass_only` | `glm4__qwen3` | 153 | 0.334 |
| `direct2_scores_only` | `deepseek7b__glm4` | 153 | 0.151 |
| `direct2_scores_only` | `deepseek7b__qwen3` | 153 | 0.229 |
| `direct2_scores_only` | `glm4__qwen3` | 153 | 0.262 |
| `direct4_scores_only` | `deepseek7b__glm4` | 153 | 0.539 |
| `direct4_scores_only` | `deepseek7b__qwen3` | 153 | 0.621 |
| `direct4_scores_only` | `glm4__qwen3` | 153 | 0.572 |
| `route_release_only` | `deepseek7b__glm4` | 153 | 0.339 |
| `route_release_only` | `deepseek7b__qwen3` | 153 | 0.173 |
| `route_release_only` | `glm4__qwen3` | 153 | 0.333 |
| `margin_drop_only` | `deepseek7b__glm4` | 153 | 0.348 |
| `margin_drop_only` | `deepseek7b__qwen3` | 153 | 0.129 |
| `margin_drop_only` | `glm4__qwen3` | 153 | 0.357 |
| `no_attention_mass` | `deepseek7b__glm4` | 153 | 0.282 |
| `no_attention_mass` | `deepseek7b__qwen3` | 153 | 0.290 |
| `no_attention_mass` | `glm4__qwen3` | 153 | 0.331 |
| `no_direct_scores` | `deepseek7b__glm4` | 153 | 0.395 |
| `no_direct_scores` | `deepseek7b__qwen3` | 153 | 0.229 |
| `no_direct_scores` | `glm4__qwen3` | 153 | 0.398 |
| `no_route_features` | `deepseek7b__glm4` | 153 | 0.186 |
| `no_route_features` | `deepseek7b__qwen3` | 153 | 0.234 |
| `no_route_features` | `glm4__qwen3` | 153 | 0.304 |
| `records_only` | `deepseek7b__glm4` | 153 | 0.275 |
| `records_only` | `deepseek7b__qwen3` | 153 | 0.264 |
| `records_only` | `glm4__qwen3` | 153 | 0.318 |
| `object_relation_sources_only` | `deepseek7b__glm4` | 153 | -0.165 |
| `object_relation_sources_only` | `deepseek7b__qwen3` | 153 | 0.105 |
| `object_relation_sources_only` | `glm4__qwen3` | 153 | 0.024 |
| `no_target_value_tokens` | `deepseek7b__glm4` | 153 | 0.290 |
| `no_target_value_tokens` | `deepseek7b__qwen3` | 153 | 0.314 |
| `no_target_value_tokens` | `glm4__qwen3` | 153 | 0.334 |
| `no_object_tokens` | `deepseek7b__glm4` | 153 | 0.282 |
| `no_object_tokens` | `deepseek7b__qwen3` | 153 | 0.283 |
| `no_object_tokens` | `glm4__qwen3` | 153 | 0.334 |
| `no_relation_tokens` | `deepseek7b__glm4` | 153 | 0.283 |
| `no_relation_tokens` | `deepseek7b__qwen3` | 153 | 0.274 |
| `no_relation_tokens` | `glm4__qwen3` | 153 | 0.330 |
| `all_relation_collapsed` | `deepseek7b__glm4` | 153 | 0.068 |
| `all_relation_collapsed` | `deepseek7b__qwen3` | 153 | 0.090 |
| `all_relation_collapsed` | `glm4__qwen3` | 153 | 0.287 |
| `target_drop_relation_collapsed` | `deepseek7b__glm4` | 153 | 0.018 |
| `target_drop_relation_collapsed` | `deepseek7b__qwen3` | 153 | 0.001 |
| `target_drop_relation_collapsed` | `glm4__qwen3` | 153 | 0.167 |
| `route_release_relation_collapsed` | `deepseek7b__glm4` | 153 | 0.245 |
| `route_release_relation_collapsed` | `deepseek7b__qwen3` | 153 | 0.216 |
| `route_release_relation_collapsed` | `glm4__qwen3` | 153 | 0.289 |

## Strict Interpretation

- A valid semantic-numeric interface should survive feature ablation and not depend on a single metric family.
- If a feature family has high separation but poor nearest-neighbor accuracy, it is a weak topology signal rather than a solved semantic code.
- This phase is an offline audit of Phase 762, not a new causal intervention.
