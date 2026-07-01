# Phase 822 Boundary-Transition Head / MLP Decomposition (confirm)

- Boundary: Phase 820 answer-boundary standard v1, with Phase 821 source rows.
- Intervention: decompose a successful/improved layer residual transition into whole attention, whole MLP, attention-head o-proj slices, and MLP top-difference channel groups.

## Model Summary

| model | cases | rows | improved | target transitions | protocol repairs | degraded | mean delta | roles |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 4 | 152 | 13 | 13 | 0 | 1 | 0.132 | `{"category_writer_or_refiner": 13, "harmful_mixer": 1, "neutral": 138}` |
| glm4 | 4 | 152 | 4 | 4 | 2 | 0 | 0.086 | `{"category_writer_or_refiner": 4, "neutral": 148}` |
| deepseek7b | 4 | 136 | 5 | 3 | 3 | 0 | 0.125 | `{"neutral": 131, "protocol_plus_category_repair": 3, "protocol_verbalizer_not_answer_writer": 2}` |

## Component Kind Summary

| model | component kind | n | improved | target | protocol | degraded | mean delta | patched classes | roles |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| qwen3 | attention_head | 128 | 6 | 6 | 0 | 1 | 0.070 | `{"broad_near_miss": 57, "close_near_miss": 64, "target_equivalent": 6, "unknown_other": 1}` | `{"category_writer_or_refiner": 6, "harmful_mixer": 1, "neutral": 121}` |
| qwen3 | attention_output | 4 | 1 | 1 | 0 | 0 | 0.500 | `{"broad_near_miss": 1, "close_near_miss": 2, "target_equivalent": 1}` | `{"category_writer_or_refiner": 1, "neutral": 3}` |
| qwen3 | layer_residual | 4 | 4 | 4 | 0 | 0 | 1.500 | `{"target_equivalent": 4}` | `{"category_writer_or_refiner": 4}` |
| qwen3 | mlp_channel_group | 12 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 6, "close_near_miss": 6}` | `{"neutral": 12}` |
| qwen3 | mlp_output | 4 | 2 | 2 | 0 | 0 | 0.750 | `{"broad_near_miss": 1, "close_near_miss": 1, "target_equivalent": 2}` | `{"category_writer_or_refiner": 2, "neutral": 2}` |
| glm4 | attention_head | 128 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 32, "close_near_miss": 32, "unknown_other": 64}` | `{"neutral": 128}` |
| glm4 | attention_output | 4 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1, "close_near_miss": 1, "unknown_other": 2}` | `{"neutral": 4}` |
| glm4 | layer_residual | 4 | 4 | 4 | 2 | 0 | 3.250 | `{"target_equivalent": 4}` | `{"category_writer_or_refiner": 4}` |
| glm4 | mlp_channel_group | 12 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "close_near_miss": 3, "unknown_other": 6}` | `{"neutral": 12}` |
| glm4 | mlp_output | 4 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1, "close_near_miss": 1, "unknown_other": 2}` | `{"neutral": 4}` |
| deepseek7b | attention_head | 112 | 0 | 0 | 0 | 0 | 0.000 | `{"format_echo": 84, "object_echo": 28}` | `{"neutral": 112}` |
| deepseek7b | attention_output | 4 | 0 | 0 | 0 | 0 | 0.000 | `{"format_echo": 3, "object_echo": 1}` | `{"neutral": 4}` |
| deepseek7b | layer_residual | 4 | 4 | 2 | 2 | 0 | 3.000 | `{"generic_blocker": 2, "target_equivalent": 2}` | `{"protocol_plus_category_repair": 2, "protocol_verbalizer_not_answer_writer": 2}` |
| deepseek7b | mlp_channel_group | 12 | 1 | 1 | 1 | 0 | 0.417 | `{"format_echo": 9, "object_echo": 2, "target_equivalent": 1}` | `{"neutral": 11, "protocol_plus_category_repair": 1}` |
| deepseek7b | mlp_output | 4 | 0 | 0 | 0 | 0 | 0.000 | `{"format_echo": 3, "object_echo": 1}` | `{"neutral": 4}` |

## Best Case Components

| model | case | baseline | best kind | best component | best class | delta | generated | role |
|---|---|---|---|---|---|---:|---|---|
| qwen3 | p816_carrot_root_vegetable | `close_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 1 | `root vegetable` | `category_writer_or_refiner` |
| qwen3 | p816_heart_body_organ | `broad_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 2 | `Body Organ` | `category_writer_or_refiner` |
| qwen3 | p816_laptop_electronic_device | `close_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 1 | `Electronic Devices` | `category_writer_or_refiner` |
| qwen3 | p816_winter_cold_season | `broad_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 2 | `Winter Season` | `category_writer_or_refiner` |
| glm4 | p816_cactus_desert_plant | `unknown_other` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `Desert plant` | `category_writer_or_refiner` |
| glm4 | p816_carrot_root_vegetable | `close_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 1 | `root vegetable` | `category_writer_or_refiner` |
| glm4 | p816_red_warm_color | `broad_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 2 | `red color` | `category_writer_or_refiner` |
| glm4 | p816_winter_cold_season | `unknown_other` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `cold season` | `category_writer_or_refiner` |
| deepseek7b | p816_carrot_root_vegetable | `format_echo` | `layer_residual` | `whole_layer_residual` | `generic_blocker` | 1 | `The answer should be a single word or` | `protocol_verbalizer_not_answer_writer` |
| deepseek7b | p816_cat_living_thing | `format_echo` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `living organism` | `protocol_plus_category_repair` |
| deepseek7b | p816_doctor_medical_worker | `format_echo` | `layer_residual` | `whole_layer_residual` | `generic_blocker` | 1 | `The answer should be a single word or` | `protocol_verbalizer_not_answer_writer` |
| deepseek7b | p816_triangle_geometric_shape | `object_echo` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `geometric shape` | `protocol_plus_category_repair` |
