# Phase 824 Boundary-Objective Sparse Subspace Search (main)

- Boundary: Phase 820 answer-boundary standard v1.
- Source: Phase 822 signal-bearing components.
- Search: candidate chunks are scored by actual generated boundary class; greedy subsets are compared with readout-positive, abs, random, and all-component controls.

## Model Summary

| model | source comps | rows | improved | target | degraded | mean delta | greedy better | greedy >= control | paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 8 | 72 | 18 | 18 | 0 | 0.500 | 4 | 14 | 16 |
| glm4 | 4 | 36 | 9 | 9 | 1 | 0.389 | 0 | 8 | 8 |
| deepseek7b | 5 | 45 | 18 | 16 | 0 | 1.822 | 0 | 10 | 10 |

## Mode Summary

| model | mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `abs_topk` | 16 | 2 | 2 | 0 | 0.250 | `{"broad_near_miss": 14, "target_equivalent": 2}` |
| qwen3 | `all` | 8 | 7 | 7 | 0 | 1.750 | `{"broad_near_miss": 1, "target_equivalent": 7}` |
| qwen3 | `causal_greedy` | 16 | 6 | 6 | 0 | 0.750 | `{"broad_near_miss": 10, "target_equivalent": 6}` |
| qwen3 | `positive_topk` | 16 | 2 | 2 | 0 | 0.250 | `{"broad_near_miss": 14, "target_equivalent": 2}` |
| qwen3 | `random_topk` | 16 | 1 | 1 | 0 | 0.125 | `{"broad_near_miss": 15, "target_equivalent": 1}` |
| glm4 | `abs_topk` | 8 | 2 | 2 | 0 | 0.250 | `{"broad_near_miss": 2, "target_equivalent": 2, "unknown_other": 4}` |
| glm4 | `all` | 4 | 4 | 4 | 0 | 3.250 | `{"target_equivalent": 4}` |
| glm4 | `causal_greedy` | 8 | 2 | 2 | 0 | 0.250 | `{"broad_near_miss": 2, "target_equivalent": 2, "unknown_other": 4}` |
| glm4 | `positive_topk` | 8 | 1 | 1 | 1 | -0.375 | `{"broad_near_miss": 2, "target_equivalent": 1, "unknown_other": 5}` |
| glm4 | `random_topk` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| deepseek7b | `abs_topk` | 10 | 4 | 4 | 0 | 2.000 | `{"format_echo": 6, "target_equivalent": 4}` |
| deepseek7b | `all` | 5 | 5 | 3 | 0 | 3.400 | `{"generic_blocker": 2, "target_equivalent": 3}` |
| deepseek7b | `causal_greedy` | 10 | 4 | 4 | 0 | 2.000 | `{"format_echo": 6, "target_equivalent": 4}` |
| deepseek7b | `positive_topk` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 4, "object_echo": 2, "target_equivalent": 2, "unknown_other": 2}` |
| deepseek7b | `random_topk` | 10 | 3 | 3 | 0 | 1.500 | `{"format_echo": 6, "object_echo": 1, "target_equivalent": 3}` |

## Best Source Components

| model | source | baseline | best mode | budget | class | delta | generated |
|---|---|---|---|---:|---|---:|---|
| qwen3 | `p816_heart_body_organ|14|attention_head|head_21` | `broad_near_miss` | `causal_greedy` | 16 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_3` | `broad_near_miss` | `all` | 128 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_4` | `broad_near_miss` | `all` | 128 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_8` | `broad_near_miss` | `all` | 128 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_output|whole_attention_output` | `broad_near_miss` | `all` | 2560 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|layer_residual|whole_layer_residual` | `broad_near_miss` | `all` | 2560 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|mlp_output|whole_mlp_output` | `broad_near_miss` | `all` | 2560 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_winter_cold_season|21|layer_residual|whole_layer_residual` | `broad_near_miss` | `all` | 2560 | `target_equivalent` | 2 | `Winter Season` |
| glm4 | `p816_cactus_desert_plant|8|layer_residual|whole_layer_residual` | `unknown_other` | `all` | 4096 | `target_equivalent` | 5 | `Desert plant` |
| glm4 | `p816_carrot_root_vegetable|39|layer_residual|whole_layer_residual` | `close_near_miss` | `all` | 4096 | `target_equivalent` | 1 | `root vegetable` |
| glm4 | `p816_red_warm_color|23|layer_residual|whole_layer_residual` | `broad_near_miss` | `all` | 4096 | `target_equivalent` | 2 | `red color` |
| glm4 | `p816_winter_cold_season|8|layer_residual|whole_layer_residual` | `unknown_other` | `all` | 4096 | `target_equivalent` | 5 | `cold season` |
| deepseek7b | `p816_carrot_root_vegetable|22|layer_residual|whole_layer_residual` | `format_echo` | `all` | 3584 | `generic_blocker` | 1 | `The answer should be a single word or` |
| deepseek7b | `p816_cat_living_thing|27|layer_residual|whole_layer_residual` | `format_echo` | `all` | 3584 | `target_equivalent` | 5 | `living organism` |
| deepseek7b | `p816_doctor_medical_worker|5|layer_residual|whole_layer_residual` | `format_echo` | `all` | 3584 | `generic_blocker` | 1 | `The answer should be a single word or` |
| deepseek7b | `p816_triangle_geometric_shape|27|layer_residual|whole_layer_residual` | `object_echo` | `all` | 3584 | `target_equivalent` | 5 | `geometric shape` |
| deepseek7b | `p816_triangle_geometric_shape|27|mlp_channel_group|mlp_topdiff_32` | `object_echo` | `all` | 32 | `target_equivalent` | 5 | `polygon` |
