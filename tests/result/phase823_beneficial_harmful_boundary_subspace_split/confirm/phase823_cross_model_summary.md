# Phase 823 Beneficial / Harmful Boundary-Transition Subspace Split (confirm)

- Boundary: Phase 820 answer-boundary standard v1.
- Source: Phase 822 successful/improved/degraded components.
- Intervention: split donor-recipient component deltas into readout-positive and readout-negative dimensions, then patch selected subspaces.

## Model Summary

| model | source comps | rows | improved | target | degraded | mean delta | positive better pairs | paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 14 | 182 | 43 | 43 | 3 | 0.374 | 8 | 42 |
| glm4 | 4 | 52 | 9 | 9 | 1 | 0.269 | 2 | 12 |
| deepseek7b | 5 | 65 | 19 | 17 | 0 | 1.338 | 4 | 15 |

## Mode Summary

| model | mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `abs_topk` | 42 | 12 | 12 | 0 | 0.500 | `{"broad_near_miss": 24, "close_near_miss": 6, "target_equivalent": 12}` |
| qwen3 | `all` | 14 | 11 | 11 | 0 | 1.357 | `{"broad_near_miss": 3, "target_equivalent": 11}` |
| qwen3 | `negative_topk` | 42 | 5 | 5 | 2 | 0.095 | `{"broad_near_miss": 26, "close_near_miss": 9, "target_equivalent": 5, "unknown_other": 2}` |
| qwen3 | `positive_topk` | 42 | 10 | 10 | 1 | 0.333 | `{"broad_near_miss": 25, "close_near_miss": 6, "target_equivalent": 10, "unknown_other": 1}` |
| qwen3 | `random_topk` | 42 | 5 | 5 | 0 | 0.238 | `{"broad_near_miss": 28, "close_near_miss": 9, "target_equivalent": 5}` |
| glm4 | `abs_topk` | 12 | 3 | 3 | 0 | 0.250 | `{"broad_near_miss": 3, "target_equivalent": 3, "unknown_other": 6}` |
| glm4 | `all` | 4 | 4 | 4 | 0 | 3.250 | `{"target_equivalent": 4}` |
| glm4 | `negative_topk` | 12 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "close_near_miss": 3, "unknown_other": 6}` |
| glm4 | `positive_topk` | 12 | 2 | 2 | 1 | -0.167 | `{"broad_near_miss": 3, "target_equivalent": 2, "unknown_other": 7}` |
| glm4 | `random_topk` | 12 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "close_near_miss": 3, "unknown_other": 6}` |
| deepseek7b | `abs_topk` | 15 | 5 | 5 | 0 | 1.667 | `{"format_echo": 10, "target_equivalent": 5}` |
| deepseek7b | `all` | 5 | 5 | 3 | 0 | 3.400 | `{"generic_blocker": 2, "target_equivalent": 3}` |
| deepseek7b | `negative_topk` | 15 | 3 | 3 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 3, "target_equivalent": 3, "unknown_other": 3}` |
| deepseek7b | `positive_topk` | 15 | 4 | 4 | 0 | 1.333 | `{"format_echo": 5, "object_echo": 3, "target_equivalent": 4, "unknown_other": 3}` |
| deepseek7b | `random_topk` | 15 | 2 | 2 | 0 | 0.667 | `{"format_echo": 8, "object_echo": 3, "target_equivalent": 2, "unknown_other": 2}` |

## Component / Mode Summary

| model | component/mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `attention_head/abs_topk` | 21 | 7 | 7 | 0 | 0.667 | `{"broad_near_miss": 14, "target_equivalent": 7}` |
| qwen3 | `attention_head/all` | 7 | 4 | 4 | 0 | 1.143 | `{"broad_near_miss": 3, "target_equivalent": 4}` |
| qwen3 | `attention_head/negative_topk` | 21 | 3 | 3 | 0 | 0.286 | `{"broad_near_miss": 18, "target_equivalent": 3}` |
| qwen3 | `attention_head/positive_topk` | 21 | 4 | 4 | 0 | 0.381 | `{"broad_near_miss": 17, "target_equivalent": 4}` |
| qwen3 | `attention_head/random_topk` | 21 | 5 | 5 | 0 | 0.476 | `{"broad_near_miss": 16, "target_equivalent": 5}` |
| qwen3 | `attention_output/abs_topk` | 3 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3}` |
| qwen3 | `attention_output/all` | 1 | 1 | 1 | 0 | 2.000 | `{"target_equivalent": 1}` |
| qwen3 | `attention_output/negative_topk` | 3 | 0 | 0 | 1 | -1.000 | `{"broad_near_miss": 2, "unknown_other": 1}` |
| qwen3 | `attention_output/positive_topk` | 3 | 1 | 1 | 0 | 0.667 | `{"broad_near_miss": 2, "target_equivalent": 1}` |
| qwen3 | `attention_output/random_topk` | 3 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3}` |
| qwen3 | `layer_residual/abs_topk` | 12 | 4 | 4 | 0 | 0.417 | `{"broad_near_miss": 5, "close_near_miss": 3, "target_equivalent": 4}` |
| qwen3 | `layer_residual/all` | 4 | 4 | 4 | 0 | 1.500 | `{"target_equivalent": 4}` |
| qwen3 | `layer_residual/negative_topk` | 12 | 1 | 1 | 1 | -0.083 | `{"broad_near_miss": 4, "close_near_miss": 6, "target_equivalent": 1, "unknown_other": 1}` |
| qwen3 | `layer_residual/positive_topk` | 12 | 4 | 4 | 1 | 0.167 | `{"broad_near_miss": 4, "close_near_miss": 3, "target_equivalent": 4, "unknown_other": 1}` |
| qwen3 | `layer_residual/random_topk` | 12 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 6, "close_near_miss": 6}` |
| qwen3 | `mlp_output/abs_topk` | 6 | 1 | 1 | 0 | 0.333 | `{"broad_near_miss": 2, "close_near_miss": 3, "target_equivalent": 1}` |
| qwen3 | `mlp_output/all` | 2 | 2 | 2 | 0 | 1.500 | `{"target_equivalent": 2}` |
| qwen3 | `mlp_output/negative_topk` | 6 | 1 | 1 | 0 | 0.333 | `{"broad_near_miss": 2, "close_near_miss": 3, "target_equivalent": 1}` |
| qwen3 | `mlp_output/positive_topk` | 6 | 1 | 1 | 0 | 0.333 | `{"broad_near_miss": 2, "close_near_miss": 3, "target_equivalent": 1}` |
| qwen3 | `mlp_output/random_topk` | 6 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "close_near_miss": 3}` |
| glm4 | `layer_residual/abs_topk` | 12 | 3 | 3 | 0 | 0.250 | `{"broad_near_miss": 3, "target_equivalent": 3, "unknown_other": 6}` |
| glm4 | `layer_residual/all` | 4 | 4 | 4 | 0 | 3.250 | `{"target_equivalent": 4}` |
| glm4 | `layer_residual/negative_topk` | 12 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "close_near_miss": 3, "unknown_other": 6}` |
| glm4 | `layer_residual/positive_topk` | 12 | 2 | 2 | 1 | -0.167 | `{"broad_near_miss": 3, "target_equivalent": 2, "unknown_other": 7}` |
| glm4 | `layer_residual/random_topk` | 12 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "close_near_miss": 3, "unknown_other": 6}` |
| deepseek7b | `layer_residual/abs_topk` | 12 | 2 | 2 | 0 | 0.833 | `{"format_echo": 10, "target_equivalent": 2}` |
| deepseek7b | `layer_residual/all` | 4 | 4 | 2 | 0 | 3.000 | `{"generic_blocker": 2, "target_equivalent": 2}` |
| deepseek7b | `layer_residual/negative_topk` | 12 | 0 | 0 | 0 | 0.000 | `{"format_echo": 6, "object_echo": 3, "unknown_other": 3}` |
| deepseek7b | `layer_residual/positive_topk` | 12 | 4 | 4 | 0 | 1.667 | `{"format_echo": 5, "target_equivalent": 4, "unknown_other": 3}` |
| deepseek7b | `layer_residual/random_topk` | 12 | 0 | 0 | 0 | 0.000 | `{"format_echo": 8, "object_echo": 2, "unknown_other": 2}` |
| deepseek7b | `mlp_channel_group/abs_topk` | 3 | 3 | 3 | 0 | 5.000 | `{"target_equivalent": 3}` |
| deepseek7b | `mlp_channel_group/all` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `mlp_channel_group/negative_topk` | 3 | 3 | 3 | 0 | 5.000 | `{"target_equivalent": 3}` |
| deepseek7b | `mlp_channel_group/positive_topk` | 3 | 0 | 0 | 0 | 0.000 | `{"object_echo": 3}` |
| deepseek7b | `mlp_channel_group/random_topk` | 3 | 2 | 2 | 0 | 3.333 | `{"object_echo": 1, "target_equivalent": 2}` |

## Best Case Rows

| model | case | baseline | best mode | budget | component | class | delta | generated |
|---|---|---|---|---:|---|---|---:|---|
| qwen3 | p816_carrot_root_vegetable | `close_near_miss` | `all` | 2560 | `layer_residual/whole_layer_residual` | `target_equivalent` | 1 | `root vegetable` |
| qwen3 | p816_heart_body_organ | `broad_near_miss` | `all` | 2560 | `mlp_output/whole_mlp_output` | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | p816_laptop_electronic_device | `close_near_miss` | `all` | 2560 | `mlp_output/whole_mlp_output` | `target_equivalent` | 1 | `Electronic Devices` |
| qwen3 | p816_winter_cold_season | `broad_near_miss` | `all` | 2560 | `layer_residual/whole_layer_residual` | `target_equivalent` | 2 | `Winter Season` |
| glm4 | p816_cactus_desert_plant | `unknown_other` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `Desert plant` |
| glm4 | p816_carrot_root_vegetable | `close_near_miss` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 1 | `root vegetable` |
| glm4 | p816_red_warm_color | `broad_near_miss` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 2 | `red color` |
| glm4 | p816_winter_cold_season | `unknown_other` | `all` | 4096 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `cold season` |
| deepseek7b | p816_carrot_root_vegetable | `format_echo` | `positive_topk` | 256 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `Root vegetable` |
| deepseek7b | p816_cat_living_thing | `format_echo` | `all` | 3584 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `living organism` |
| deepseek7b | p816_doctor_medical_worker | `format_echo` | `all` | 3584 | `layer_residual/whole_layer_residual` | `generic_blocker` | 1 | `The answer should be a single word or` |
| deepseek7b | p816_triangle_geometric_shape | `object_echo` | `all` | 3584 | `layer_residual/whole_layer_residual` | `target_equivalent` | 5 | `geometric shape` |
