# Phase 823 Beneficial / Harmful Boundary-Transition Subspace Split (main)

- Boundary: Phase 820 answer-boundary standard v1.
- Source: Phase 822 successful/improved/degraded components.
- Intervention: split donor-recipient component deltas into readout-positive and readout-negative dimensions, then patch selected subspaces.

## Model Summary

| model | source comps | rows | improved | target | degraded | mean delta | positive better pairs | paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 8 | 72 | 16 | 16 | 0 | 0.444 | 1 | 16 |
| glm4 | 4 | 36 | 7 | 7 | 1 | 0.333 | 1 | 8 |
| deepseek7b | 5 | 45 | 14 | 12 | 0 | 1.378 | 2 | 10 |

## Mode Summary

| model | mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `abs_topk` | 16 | 2 | 2 | 0 | 0.250 | `{"broad_near_miss": 14, "target_equivalent": 2}` |
| qwen3 | `all` | 8 | 8 | 8 | 0 | 2.000 | `{"target_equivalent": 8}` |
| qwen3 | `negative_topk` | 16 | 3 | 3 | 0 | 0.375 | `{"broad_near_miss": 13, "target_equivalent": 3}` |
| qwen3 | `positive_topk` | 16 | 2 | 2 | 0 | 0.250 | `{"broad_near_miss": 14, "target_equivalent": 2}` |
| qwen3 | `random_topk` | 16 | 1 | 1 | 0 | 0.125 | `{"broad_near_miss": 15, "target_equivalent": 1}` |
| glm4 | `abs_topk` | 8 | 2 | 2 | 0 | 0.250 | `{"broad_near_miss": 2, "target_equivalent": 2, "unknown_other": 4}` |
| glm4 | `all` | 4 | 4 | 4 | 0 | 3.250 | `{"target_equivalent": 4}` |
| glm4 | `negative_topk` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `positive_topk` | 8 | 1 | 1 | 1 | -0.375 | `{"broad_near_miss": 2, "target_equivalent": 1, "unknown_other": 5}` |
| glm4 | `random_topk` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| deepseek7b | `abs_topk` | 10 | 4 | 4 | 0 | 2.000 | `{"format_echo": 6, "target_equivalent": 4}` |
| deepseek7b | `all` | 5 | 5 | 3 | 0 | 3.400 | `{"generic_blocker": 2, "target_equivalent": 3}` |
| deepseek7b | `negative_topk` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 4, "object_echo": 2, "target_equivalent": 2, "unknown_other": 2}` |
| deepseek7b | `positive_topk` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 4, "object_echo": 2, "target_equivalent": 2, "unknown_other": 2}` |
| deepseek7b | `random_topk` | 10 | 1 | 1 | 0 | 0.500 | `{"format_echo": 6, "object_echo": 3, "target_equivalent": 1}` |

## Component / Mode Summary

| model | component/mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `attention_head/abs_topk` | 8 | 2 | 2 | 0 | 0.500 | `{"broad_near_miss": 6, "target_equivalent": 2}` |
| qwen3 | `attention_head/all` | 4 | 4 | 4 | 0 | 2.000 | `{"target_equivalent": 4}` |
| qwen3 | `attention_head/negative_topk` | 8 | 2 | 2 | 0 | 0.500 | `{"broad_near_miss": 6, "target_equivalent": 2}` |
| qwen3 | `attention_head/positive_topk` | 8 | 1 | 1 | 0 | 0.250 | `{"broad_near_miss": 7, "target_equivalent": 1}` |
| qwen3 | `attention_head/random_topk` | 8 | 1 | 1 | 0 | 0.250 | `{"broad_near_miss": 7, "target_equivalent": 1}` |
| qwen3 | `attention_output/abs_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `attention_output/all` | 1 | 1 | 1 | 0 | 2.000 | `{"target_equivalent": 1}` |
| qwen3 | `attention_output/negative_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `attention_output/positive_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `attention_output/random_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `layer_residual/abs_topk` | 4 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 4}` |
| qwen3 | `layer_residual/all` | 2 | 2 | 2 | 0 | 2.000 | `{"target_equivalent": 2}` |
| qwen3 | `layer_residual/negative_topk` | 4 | 1 | 1 | 0 | 0.500 | `{"broad_near_miss": 3, "target_equivalent": 1}` |
| qwen3 | `layer_residual/positive_topk` | 4 | 1 | 1 | 0 | 0.500 | `{"broad_near_miss": 3, "target_equivalent": 1}` |
| qwen3 | `layer_residual/random_topk` | 4 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 4}` |
| qwen3 | `mlp_output/abs_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `mlp_output/all` | 1 | 1 | 1 | 0 | 2.000 | `{"target_equivalent": 1}` |
| qwen3 | `mlp_output/negative_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `mlp_output/positive_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `mlp_output/random_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| glm4 | `layer_residual/abs_topk` | 8 | 2 | 2 | 0 | 0.250 | `{"broad_near_miss": 2, "target_equivalent": 2, "unknown_other": 4}` |
| glm4 | `layer_residual/all` | 4 | 4 | 4 | 0 | 3.250 | `{"target_equivalent": 4}` |
| glm4 | `layer_residual/negative_topk` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `layer_residual/positive_topk` | 8 | 1 | 1 | 1 | -0.375 | `{"broad_near_miss": 2, "target_equivalent": 1, "unknown_other": 5}` |
| glm4 | `layer_residual/random_topk` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| deepseek7b | `layer_residual/abs_topk` | 8 | 2 | 2 | 0 | 1.250 | `{"format_echo": 6, "target_equivalent": 2}` |
| deepseek7b | `layer_residual/all` | 4 | 4 | 2 | 0 | 3.000 | `{"generic_blocker": 2, "target_equivalent": 2}` |
| deepseek7b | `layer_residual/negative_topk` | 8 | 0 | 0 | 0 | 0.000 | `{"format_echo": 4, "object_echo": 2, "unknown_other": 2}` |
| deepseek7b | `layer_residual/positive_topk` | 8 | 2 | 2 | 0 | 1.250 | `{"format_echo": 4, "target_equivalent": 2, "unknown_other": 2}` |
| deepseek7b | `layer_residual/random_topk` | 8 | 0 | 0 | 0 | 0.000 | `{"format_echo": 6, "object_echo": 2}` |
| deepseek7b | `mlp_channel_group/abs_topk` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| deepseek7b | `mlp_channel_group/all` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `mlp_channel_group/negative_topk` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| deepseek7b | `mlp_channel_group/positive_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"object_echo": 2}` |
| deepseek7b | `mlp_channel_group/random_topk` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |

## Best Case Rows

| model | case | baseline | best mode | budget | component | class | delta | generated |
|---|---|---|---|---:|---|---|---:|---|
| qwen3 | p816_heart_body_organ | `broad_near_miss` | `all` | 2560 | `mlp_output/whole_mlp_output` | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | p816_winter_cold_season | `broad_near_miss` | `all` | 2560 | `layer_residual/whole_layer_residual` | `target_equivalent` | 2 | `Winter Season` |
| glm4 | p816_cactus_desert_plant | `unknown_other` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `Desert plant` |
| glm4 | p816_carrot_root_vegetable | `close_near_miss` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 1 | `root vegetable` |
| glm4 | p816_red_warm_color | `broad_near_miss` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 2 | `red color` |
| glm4 | p816_winter_cold_season | `unknown_other` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `cold season` |
| deepseek7b | p816_carrot_root_vegetable | `format_echo` | `all` | 3584 | `layer_residual/whole_layer_residual` | `generic_blocker` | 1 | `The answer should be a single word or` |
| deepseek7b | p816_cat_living_thing | `format_echo` | `all` | 3584 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `living organism` |
| deepseek7b | p816_doctor_medical_worker | `format_echo` | `all` | 3584 | `layer_residual/whole_layer_residual` | `generic_blocker` | 1 | `The answer should be a single word or` |
| deepseek7b | p816_triangle_geometric_shape | `object_echo` | `all` | 3584 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `geometric shape` |
