# Phase 822 Boundary-Transition Head / MLP Decomposition (main)

- Boundary: Phase 820 answer-boundary standard v1, with Phase 821 source rows.
- Intervention: decompose a successful/improved layer residual transition into whole attention, whole MLP, attention-head o-proj slices, and MLP top-difference channel groups.

## Model Summary

| model | cases | rows | improved | target transitions | protocol repairs | degraded | mean delta | roles |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 3 | 54 | 10 | 10 | 0 | 1 | 0.278 | `{"category_writer_or_refiner": 10, "harmful_mixer": 1, "neutral": 43}` |
| glm4 | 3 | 54 | 3 | 3 | 2 | 0 | 0.222 | `{"category_writer_or_refiner": 3, "neutral": 51}` |
| deepseek7b | 3 | 54 | 4 | 3 | 3 | 0 | 0.296 | `{"neutral": 50, "protocol_plus_category_repair": 3, "protocol_verbalizer_not_answer_writer": 1}` |

## Component Kind Summary

| model | component kind | n | improved | target | protocol | degraded | mean delta | patched classes | roles |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| qwen3 | attention_head | 36 | 4 | 4 | 0 | 1 | 0.139 | `{"broad_near_miss": 19, "close_near_miss": 12, "target_equivalent": 4, "unknown_other": 1}` | `{"category_writer_or_refiner": 4, "harmful_mixer": 1, "neutral": 31}` |
| qwen3 | attention_output | 3 | 1 | 1 | 0 | 0 | 0.667 | `{"broad_near_miss": 1, "close_near_miss": 1, "target_equivalent": 1}` | `{"category_writer_or_refiner": 1, "neutral": 2}` |
| qwen3 | layer_residual | 3 | 3 | 3 | 0 | 0 | 1.667 | `{"target_equivalent": 3}` | `{"category_writer_or_refiner": 3}` |
| qwen3 | mlp_channel_group | 9 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 6, "close_near_miss": 3}` | `{"neutral": 9}` |
| qwen3 | mlp_output | 3 | 2 | 2 | 0 | 0 | 1.000 | `{"broad_near_miss": 1, "target_equivalent": 2}` | `{"category_writer_or_refiner": 2, "neutral": 1}` |
| glm4 | attention_head | 36 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 12, "unknown_other": 24}` | `{"neutral": 36}` |
| glm4 | attention_output | 3 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1, "unknown_other": 2}` | `{"neutral": 3}` |
| glm4 | layer_residual | 3 | 3 | 3 | 2 | 0 | 4.000 | `{"target_equivalent": 3}` | `{"category_writer_or_refiner": 3}` |
| glm4 | mlp_channel_group | 9 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "unknown_other": 6}` | `{"neutral": 9}` |
| glm4 | mlp_output | 3 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1, "unknown_other": 2}` | `{"neutral": 3}` |
| deepseek7b | attention_head | 36 | 0 | 0 | 0 | 0 | 0.000 | `{"format_echo": 24, "object_echo": 12}` | `{"neutral": 36}` |
| deepseek7b | attention_output | 3 | 0 | 0 | 0 | 0 | 0.000 | `{"format_echo": 2, "object_echo": 1}` | `{"neutral": 3}` |
| deepseek7b | layer_residual | 3 | 3 | 2 | 2 | 0 | 3.667 | `{"generic_blocker": 1, "target_equivalent": 2}` | `{"protocol_plus_category_repair": 2, "protocol_verbalizer_not_answer_writer": 1}` |
| deepseek7b | mlp_channel_group | 9 | 1 | 1 | 1 | 0 | 0.556 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 1}` | `{"neutral": 8, "protocol_plus_category_repair": 1}` |
| deepseek7b | mlp_output | 3 | 0 | 0 | 0 | 0 | 0.000 | `{"format_echo": 2, "object_echo": 1}` | `{"neutral": 3}` |

## Best Case Components

| model | case | baseline | best kind | best component | best class | delta | generated | role |
|---|---|---|---|---|---|---:|---|---|
| qwen3 | p816_heart_body_organ | `broad_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 2 | `Body Organ` | `category_writer_or_refiner` |
| qwen3 | p816_laptop_electronic_device | `close_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 1 | `Electronic Devices` | `category_writer_or_refiner` |
| qwen3 | p816_winter_cold_season | `broad_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 2 | `Winter Season` | `category_writer_or_refiner` |
| glm4 | p816_cactus_desert_plant | `unknown_other` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `Desert plant` | `category_writer_or_refiner` |
| glm4 | p816_red_warm_color | `broad_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 2 | `red color` | `category_writer_or_refiner` |
| glm4 | p816_winter_cold_season | `unknown_other` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `cold season` | `category_writer_or_refiner` |
| deepseek7b | p816_cat_living_thing | `format_echo` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `living organism` | `protocol_plus_category_repair` |
| deepseek7b | p816_doctor_medical_worker | `format_echo` | `layer_residual` | `whole_layer_residual` | `generic_blocker` | 1 | `The answer should be a single word or` | `protocol_verbalizer_not_answer_writer` |
| deepseek7b | p816_triangle_geometric_shape | `object_echo` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `geometric shape` | `protocol_plus_category_repair` |
