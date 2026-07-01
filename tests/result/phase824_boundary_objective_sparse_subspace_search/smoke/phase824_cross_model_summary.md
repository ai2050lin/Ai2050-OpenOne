# Phase 824 Boundary-Objective Sparse Subspace Search (smoke)

- Boundary: Phase 820 answer-boundary standard v1.
- Source: Phase 822 signal-bearing components.
- Search: candidate chunks are scored by actual generated boundary class; greedy subsets are compared with readout-positive, abs, random, and all-component controls.

## Model Summary

| model | source comps | rows | improved | target | degraded | mean delta | greedy better | greedy >= control | paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 2 | 10 | 1 | 1 | 0 | 0.200 | 0 | 2 | 2 |
| glm4 | 2 | 10 | 2 | 2 | 0 | 1.000 | 0 | 2 | 2 |
| deepseek7b | 2 | 10 | 5 | 4 | 0 | 2.100 | 0 | 2 | 2 |

## Mode Summary

| model | mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `abs_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `all` | 2 | 1 | 1 | 0 | 1.000 | `{"broad_near_miss": 1, "target_equivalent": 1}` |
| qwen3 | `causal_greedy` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `positive_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `random_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| glm4 | `abs_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `all` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| glm4 | `causal_greedy` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `positive_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `random_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| deepseek7b | `abs_topk` | 2 | 1 | 1 | 0 | 2.500 | `{"format_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `all` | 2 | 2 | 1 | 0 | 3.000 | `{"generic_blocker": 1, "target_equivalent": 1}` |
| deepseek7b | `causal_greedy` | 2 | 1 | 1 | 0 | 2.500 | `{"format_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `positive_topk` | 2 | 1 | 1 | 0 | 2.500 | `{"format_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `random_topk` | 2 | 0 | 0 | 0 | 0.000 | `{"format_echo": 1, "object_echo": 1}` |

## Best Source Components

| model | source | baseline | best mode | budget | class | delta | generated |
|---|---|---|---|---:|---|---:|---|
| qwen3 | `p816_heart_body_organ|14|attention_head|head_21` | `broad_near_miss` | `all` | 128 | `broad_near_miss` | 0 | `Circulatory System` |
| qwen3 | `p816_winter_cold_season|21|layer_residual|whole_layer_residual` | `broad_near_miss` | `all` | 2560 | `target_equivalent` | 2 | `Winter Season` |
| glm4 | `p816_cactus_desert_plant|8|layer_residual|whole_layer_residual` | `unknown_other` | `all` | 4096 | `target_equivalent` | 5 | `Desert plant` |
| glm4 | `p816_winter_cold_season|8|layer_residual|whole_layer_residual` | `unknown_other` | `all` | 4096 | `target_equivalent` | 5 | `cold season` |
| deepseek7b | `p816_doctor_medical_worker|5|layer_residual|whole_layer_residual` | `format_echo` | `all` | 3584 | `generic_blocker` | 1 | `The answer should be a single` |
| deepseek7b | `p816_triangle_geometric_shape|27|layer_residual|whole_layer_residual` | `object_echo` | `all` | 3584 | `target_equivalent` | 5 | `geometric shape` |
