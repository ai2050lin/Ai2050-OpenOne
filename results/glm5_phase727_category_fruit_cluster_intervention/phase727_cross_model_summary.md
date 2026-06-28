# Phase 727 Category/Fruit Route Cluster Intervention

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: cluster-level likelihood and greedy generation intervention.

| model | intervention | mean_delta | hit_rate | changed | hit_drop | rank_delta |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | baseline | 0.0000 | 0.955 | 0.000 | 0.000 | 0.00 |
| qwen3 | category_cluster | 0.0184 | 0.955 | 0.000 | 0.000 | -0.05 |
| qwen3 | category_full_head | -0.0159 | 0.955 | 0.000 | 0.000 | 0.00 |
| qwen3 | category_plus_fruit_cluster | 0.0190 | 0.955 | 0.000 | 0.000 | 0.00 |
| qwen3 | category_single | 0.0039 | 0.955 | 0.000 | 0.000 | 0.05 |
| qwen3 | fruit_cluster | 0.0160 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | baseline | 0.0000 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | category_cluster | 0.0006 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | category_full_head | 0.0016 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | category_plus_fruit_cluster | 0.0014 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | category_single | -0.0035 | 0.955 | 0.000 | 0.000 | 0.00 |
| glm4 | fruit_cluster | 0.0070 | 0.955 | 0.000 | 0.000 | 0.00 |
| deepseek7b | baseline | 0.0000 | 0.500 | 0.000 | 0.000 | 0.00 |
| deepseek7b | category_cluster | -0.2613 | 0.500 | 0.000 | 0.000 | 0.68 |
| deepseek7b | category_full_head | -1.0575 | 0.500 | 0.409 | 0.091 | 4.50 |
| deepseek7b | category_plus_fruit_cluster | -0.3463 | 0.500 | 0.000 | 0.000 | 1.00 |
| deepseek7b | category_single | -0.2271 | 0.545 | 0.045 | 0.000 | 3.14 |
| deepseek7b | fruit_cluster | -0.0582 | 0.500 | 0.000 | 0.000 | 0.05 |

## Strict Interpretation

- Cluster likelihood drops without generation hit drops indicate downstream compensation or a generation gate.
- Full-head effects are stronger but less localized.
- This phase still uses greedy category answers only.
