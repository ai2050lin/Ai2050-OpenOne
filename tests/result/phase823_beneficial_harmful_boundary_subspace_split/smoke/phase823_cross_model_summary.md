# Phase 823 Beneficial / Harmful Boundary-Transition Subspace Split (smoke)

- Boundary: Phase 820 answer-boundary standard v1.
- Source: Phase 822 successful/improved/degraded components.
- Intervention: split donor-recipient component deltas into readout-positive and readout-negative dimensions, then patch selected subspaces.

## Model Summary

| model | source comps | rows | improved | target | degraded | mean delta | positive better pairs | paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 2 | 10 | 2 | 2 | 0 | 0.400 | 0 | 2 |
| glm4 | 2 | 10 | 2 | 2 | 0 | 1.000 | 0 | 2 |
| deepseek7b | 2 | 10 | 6 | 6 | 0 | 3.000 | 1 | 2 |

## Mode Summary

| model | mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `abs_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `all` | 2 | 2 | 2 | 0 | 2.000 | `{"target_equivalent": 2}` |
| qwen3 | `negative_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `positive_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `random_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| glm4 | `abs_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `all` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| glm4 | `negative_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `positive_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `random_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| deepseek7b | `abs_topk` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| deepseek7b | `all` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| deepseek7b | `negative_topk` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `positive_topk` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `random_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"object_echo": 2}` |

## Component / Mode Summary

| model | component/mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `layer_residual/abs_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` |
| qwen3 | `layer_residual/all` | 1 | 1 | 1 | 0 | 2.000 | `{"target_equivalent": 1}` |
| qwen3 | `layer_residual/negative_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` |
| qwen3 | `layer_residual/positive_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` |
| qwen3 | `layer_residual/random_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` |
| qwen3 | `mlp_output/abs_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` |
| qwen3 | `mlp_output/all` | 1 | 1 | 1 | 0 | 2.000 | `{"target_equivalent": 1}` |
| qwen3 | `mlp_output/negative_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` |
| qwen3 | `mlp_output/positive_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` |
| qwen3 | `mlp_output/random_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` |
| glm4 | `layer_residual/abs_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `layer_residual/all` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| glm4 | `layer_residual/negative_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `layer_residual/positive_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `layer_residual/random_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| deepseek7b | `layer_residual/abs_topk` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `layer_residual/all` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `layer_residual/negative_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"object_echo": 1}` |
| deepseek7b | `layer_residual/positive_topk` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `layer_residual/random_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"object_echo": 1}` |
| deepseek7b | `mlp_channel_group/abs_topk` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `mlp_channel_group/all` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `mlp_channel_group/negative_topk` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `mlp_channel_group/positive_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"object_echo": 1}` |
| deepseek7b | `mlp_channel_group/random_topk` | 1 | 0 | 0 | 0 | 0.000 | `{"object_echo": 1}` |

## Best Case Rows

| model | case | baseline | best mode | budget | component | class | delta | generated |
|---|---|---|---|---:|---|---|---:|---|
| qwen3 | p816_heart_body_organ | `broad_near_miss` | `all` | 2560 | `mlp_output/whole_mlp_output` | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | p816_winter_cold_season | `broad_near_miss` | `all` | 2560 | `layer_residual/whole_layer_residual` | `target_equivalent` | 2 | `Winter Season` |
| glm4 | p816_cactus_desert_plant | `unknown_other` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `Desert plant` |
| glm4 | p816_winter_cold_season | `unknown_other` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `cold season` |
| deepseek7b | p816_triangle_geometric_shape | `object_echo` | `all` | 3584 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `geometric shape` |
