# Phase 828 Cross-Component Consistency Fiber Composition (confirm)

- Source: Phase 827 selected subspaces.
- Objective: simultaneous two-component patching, then exact/natural consistency audit.

## Model Summary

| model | groups | pairs | exact target | natural target | natural degraded | pair exact+multi | new exact+multi | preserve exact+multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 10 | 12 | 11 | 18 | 6 | 7 | 0 | 7 |
| glm4 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 10 | 2 | 2 | 5 | 0 | 2 | 0 | 2 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 12 | 11 | 11 | 0 | 1.833 | `{"broad_near_miss": 1, "target_equivalent": 11}` |
| qwen3 | `natural_category` | 12 | 0 | 0 | 5 | -1.250 | `{"broad_near_miss": 7, "unknown_other": 5}` |
| qwen3 | `natural_question` | 12 | 7 | 7 | 1 | 0.917 | `{"broad_near_miss": 4, "target_equivalent": 7, "unknown_other": 1}` |
| qwen3 | `object_only` | 12 | 11 | 11 | 0 | 1.833 | `{"broad_near_miss": 1, "target_equivalent": 11}` |
| deepseek7b | `exact_choices` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| deepseek7b | `natural_category` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `natural_question` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| deepseek7b | `object_only` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |

## Pair Records

| model | pair | exact | natural count | exact+multi | degraded | single had exact+multi |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `L14:layer_residual:whole_layer_residual:B32 + L14:attention_output:whole_attention_output:B32` | 1 | 2 | 1 | 1 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B32 + L14:attention_head:head_3:B32` | 1 | 2 | 1 | 1 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_output:whole_attention_output:B16` | 1 | 1 | 0 | 1 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B32 + L14:attention_head:head_4:B32` | 1 | 2 | 1 | 1 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B16 + L14:mlp_output:whole_mlp_output:B16` | 1 | 2 | 1 | 0 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_head:head_4:B16` | 1 | 2 | 1 | 0 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B32 + L14:mlp_output:whole_mlp_output:B32` | 1 | 2 | 1 | 1 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B32 + L14:attention_head:head_8:B32` | 1 | 2 | 1 | 1 | 1 |
| qwen3 | `L14:attention_output:whole_attention_output:B32 + L14:attention_head:head_3:B32` | 1 | 1 | 0 | 0 | 0 |
| qwen3 | `L14:attention_output:whole_attention_output:B32 + L14:attention_head:head_4:B32` | 0 | 0 | 0 | 0 | 0 |
| qwen3 | `L14:attention_head:head_3:B32 + L14:attention_head:head_4:B32` | 1 | 1 | 0 | 0 | 0 |
| qwen3 | `L14:attention_output:whole_attention_output:B32 + L14:mlp_output:whole_mlp_output:B32` | 1 | 1 | 0 | 0 | 0 |
| deepseek7b | `L27:mlp_channel_group:mlp_topdiff_32:B32 + L27:layer_residual:whole_layer_residual:B32` | 1 | 3 | 1 | 0 | 1 |
| deepseek7b | `L27:mlp_channel_group:mlp_topdiff_32:B16 + L27:layer_residual:whole_layer_residual:B16` | 1 | 2 | 1 | 0 | 1 |
