# Phase 824 Boundary-Objective Sparse Subspace Search (confirm)

- Boundary: Phase 820 answer-boundary standard v1.
- Source: Phase 822 signal-bearing components.
- Search: candidate chunks are scored by actual generated boundary class; greedy subsets are compared with readout-positive, abs, random, and all-component controls.

## Model Summary

| model | source comps | rows | improved | target | degraded | mean delta | greedy better | greedy >= control | paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 14 | 294 | 61 | 61 | 1 | 0.364 | 7 | 36 | 42 |
| glm4 | 4 | 84 | 13 | 13 | 1 | 0.214 | 0 | 12 | 12 |
| deepseek7b | 5 | 101 | 23 | 21 | 0 | 1.059 | 0 | 13 | 14 |

## Mode Summary

| model | mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `abs_topk` | 42 | 12 | 12 | 0 | 0.500 | `{"broad_near_miss": 24, "close_near_miss": 6, "target_equivalent": 12}` |
| qwen3 | `all` | 14 | 11 | 11 | 0 | 1.357 | `{"broad_near_miss": 3, "target_equivalent": 11}` |
| qwen3 | `candidate_probe` | 112 | 3 | 3 | 0 | 0.054 | `{"broad_near_miss": 85, "close_near_miss": 24, "target_equivalent": 3}` |
| qwen3 | `causal_greedy` | 42 | 18 | 18 | 0 | 0.786 | `{"broad_near_miss": 18, "close_near_miss": 6, "target_equivalent": 18}` |
| qwen3 | `positive_topk` | 42 | 10 | 10 | 1 | 0.333 | `{"broad_near_miss": 25, "close_near_miss": 6, "target_equivalent": 10, "unknown_other": 1}` |
| qwen3 | `random_topk` | 42 | 7 | 7 | 0 | 0.333 | `{"broad_near_miss": 26, "close_near_miss": 9, "target_equivalent": 7}` |
| glm4 | `abs_topk` | 12 | 3 | 3 | 0 | 0.250 | `{"broad_near_miss": 3, "target_equivalent": 3, "unknown_other": 6}` |
| glm4 | `all` | 4 | 4 | 4 | 0 | 3.250 | `{"target_equivalent": 4}` |
| glm4 | `candidate_probe` | 32 | 1 | 1 | 0 | 0.031 | `{"broad_near_miss": 8, "close_near_miss": 7, "target_equivalent": 1, "unknown_other": 16}` |
| glm4 | `causal_greedy` | 12 | 3 | 3 | 0 | 0.250 | `{"broad_near_miss": 3, "target_equivalent": 3, "unknown_other": 6}` |
| glm4 | `positive_topk` | 12 | 2 | 2 | 1 | -0.167 | `{"broad_near_miss": 3, "target_equivalent": 2, "unknown_other": 7}` |
| glm4 | `random_topk` | 12 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "close_near_miss": 3, "unknown_other": 6}` |
| deepseek7b | `abs_topk` | 14 | 4 | 4 | 0 | 1.429 | `{"format_echo": 10, "target_equivalent": 4}` |
| deepseek7b | `all` | 5 | 5 | 3 | 0 | 3.400 | `{"generic_blocker": 2, "target_equivalent": 3}` |
| deepseek7b | `candidate_probe` | 40 | 2 | 2 | 0 | 0.250 | `{"format_echo": 24, "object_echo": 13, "target_equivalent": 2, "unknown_other": 1}` |
| deepseek7b | `causal_greedy` | 14 | 5 | 5 | 0 | 1.786 | `{"format_echo": 9, "target_equivalent": 5}` |
| deepseek7b | `positive_topk` | 14 | 4 | 4 | 0 | 1.429 | `{"format_echo": 5, "object_echo": 2, "target_equivalent": 4, "unknown_other": 3}` |
| deepseek7b | `random_topk` | 14 | 3 | 3 | 0 | 1.071 | `{"format_echo": 9, "object_echo": 2, "target_equivalent": 3}` |

## Best Source Components

| model | source | baseline | best mode | budget | class | delta | generated |
|---|---|---|---|---:|---|---:|---|
| qwen3 | `p816_carrot_root_vegetable|35|layer_residual|whole_layer_residual` | `close_near_miss` | `all` | 2560 | `target_equivalent` | 1 | `root vegetable` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_16` | `broad_near_miss` | `all` | 128 | `broad_near_miss` | 0 | `Circulatory System` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_21` | `broad_near_miss` | `causal_greedy` | 16 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_22` | `broad_near_miss` | `positive_topk` | 64 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_25` | `broad_near_miss` | `all` | 128 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_3` | `broad_near_miss` | `all` | 128 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_4` | `broad_near_miss` | `all` | 128 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_head|head_8` | `broad_near_miss` | `all` | 128 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|attention_output|whole_attention_output` | `broad_near_miss` | `all` | 2560 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|layer_residual|whole_layer_residual` | `broad_near_miss` | `all` | 2560 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_heart_body_organ|14|mlp_output|whole_mlp_output` | `broad_near_miss` | `all` | 2560 | `target_equivalent` | 2 | `Body Organ` |
| qwen3 | `p816_laptop_electronic_device|7|layer_residual|whole_layer_residual` | `close_near_miss` | `all` | 2560 | `target_equivalent` | 1 | `Electronic Devices` |
| qwen3 | `p816_laptop_electronic_device|7|mlp_output|whole_mlp_output` | `close_near_miss` | `all` | 2560 | `target_equivalent` | 1 | `Electronic Devices` |
| qwen3 | `p816_winter_cold_season|21|layer_residual|whole_layer_residual` | `broad_near_miss` | `all` | 2560 | `target_equivalent` | 2 | `Winter Season` |
| glm4 | `p816_cactus_desert_plant|8|layer_residual|whole_layer_residual` | `unknown_other` | `all` | 4096 | `target_equivalent` | 5 | `Desert plant` |
| glm4 | `p816_carrot_root_vegetable|39|layer_residual|whole_layer_residual` | `close_near_miss` | `all` | 4096 | `target_equivalent` | 1 | `root vegetable` |
| glm4 | `p816_red_warm_color|23|layer_residual|whole_layer_residual` | `broad_near_miss` | `all` | 4096 | `target_equivalent` | 2 | `red color` |
| glm4 | `p816_winter_cold_season|8|layer_residual|whole_layer_residual` | `unknown_other` | `all` | 4096 | `target_equivalent` | 5 | `cold season` |
| deepseek7b | `p816_carrot_root_vegetable|22|layer_residual|whole_layer_residual` | `format_echo` | `positive_topk` | 256 | `target_equivalent` | 5 | `Root vegetable` |
| deepseek7b | `p816_cat_living_thing|27|layer_residual|whole_layer_residual` | `format_echo` | `all` | 3584 | `target_equivalent` | 5 | `living organism` |
| deepseek7b | `p816_doctor_medical_worker|5|layer_residual|whole_layer_residual` | `format_echo` | `all` | 3584 | `generic_blocker` | 1 | `The answer should be a single word or` |
| deepseek7b | `p816_triangle_geometric_shape|27|layer_residual|whole_layer_residual` | `object_echo` | `all` | 3584 | `target_equivalent` | 5 | `geometric shape` |
| deepseek7b | `p816_triangle_geometric_shape|27|mlp_channel_group|mlp_topdiff_32` | `object_echo` | `all` | 32 | `target_equivalent` | 5 | `polygon` |
