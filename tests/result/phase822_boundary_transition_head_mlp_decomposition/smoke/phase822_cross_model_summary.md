# Phase 822 Boundary-Transition Head / MLP Decomposition (smoke)

- Boundary: Phase 820 answer-boundary standard v1, with Phase 821 source rows.
- Intervention: decompose a successful/improved layer residual transition into whole attention, whole MLP, attention-head o-proj slices, and MLP top-difference channel groups.

## Model Summary

| model | cases | rows | improved | target transitions | protocol repairs | degraded | mean delta | roles |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 1 | 11 | 1 | 1 | 0 | 0 | 0.182 | `{"category_writer_or_refiner": 1, "neutral": 10}` |
| glm4 | 1 | 11 | 1 | 1 | 1 | 0 | 0.455 | `{"category_writer_or_refiner": 1, "neutral": 10}` |
| deepseek7b | 1 | 11 | 1 | 1 | 1 | 0 | 0.455 | `{"neutral": 10, "protocol_plus_category_repair": 1}` |

## Component Kind Summary

| model | component kind | n | improved | target | protocol | degraded | mean delta | patched classes | roles |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| qwen3 | attention_head | 6 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 6}` | `{"neutral": 6}` |
| qwen3 | attention_output | 1 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` | `{"neutral": 1}` |
| qwen3 | layer_residual | 1 | 1 | 1 | 0 | 0 | 2.000 | `{"target_equivalent": 1}` | `{"category_writer_or_refiner": 1}` |
| qwen3 | mlp_channel_group | 2 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` | `{"neutral": 2}` |
| qwen3 | mlp_output | 1 | 0 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 1}` | `{"neutral": 1}` |
| glm4 | attention_head | 6 | 0 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 6}` | `{"neutral": 6}` |
| glm4 | attention_output | 1 | 0 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 1}` | `{"neutral": 1}` |
| glm4 | layer_residual | 1 | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` | `{"category_writer_or_refiner": 1}` |
| glm4 | mlp_channel_group | 2 | 0 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` | `{"neutral": 2}` |
| glm4 | mlp_output | 1 | 0 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 1}` | `{"neutral": 1}` |
| deepseek7b | attention_head | 6 | 0 | 0 | 0 | 0 | 0.000 | `{"object_echo": 6}` | `{"neutral": 6}` |
| deepseek7b | attention_output | 1 | 0 | 0 | 0 | 0 | 0.000 | `{"object_echo": 1}` | `{"neutral": 1}` |
| deepseek7b | layer_residual | 1 | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` | `{"protocol_plus_category_repair": 1}` |
| deepseek7b | mlp_channel_group | 2 | 0 | 0 | 0 | 0 | 0.000 | `{"object_echo": 2}` | `{"neutral": 2}` |
| deepseek7b | mlp_output | 1 | 0 | 0 | 0 | 0 | 0.000 | `{"object_echo": 1}` | `{"neutral": 1}` |

## Best Case Components

| model | case | baseline | best kind | best component | best class | delta | generated | role |
|---|---|---|---|---|---|---:|---|---|
| qwen3 | p816_winter_cold_season | `broad_near_miss` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 2 | `Winter Season` | `category_writer_or_refiner` |
| glm4 | p816_winter_cold_season | `unknown_other` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `cold season` | `category_writer_or_refiner` |
| deepseek7b | p816_triangle_geometric_shape | `object_echo` | `layer_residual` | `whole_layer_residual` | `target_equivalent` | 5 | `geometric shape` | `protocol_plus_category_repair` |
