# Phase 828 Cross-Component Consistency Fiber Composition (main)

- Source: Phase 827 selected subspaces.
- Objective: simultaneous two-component patching, then exact/natural consistency audit.

## Model Summary

| model | groups | pairs | exact target | natural target | natural degraded | pair exact+multi | new exact+multi | preserve exact+multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 6 | 7 | 6 | 6 | 3 | 0 | 0 | 0 |
| glm4 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 6 | 2 | 2 | 3 | 0 | 1 | 0 | 1 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 7 | 6 | 6 | 0 | 1.714 | `{"broad_near_miss": 1, "target_equivalent": 6}` |
| qwen3 | `natural_category` | 7 | 0 | 0 | 3 | -1.286 | `{"broad_near_miss": 4, "unknown_other": 3}` |
| qwen3 | `object_only` | 7 | 6 | 6 | 0 | 1.714 | `{"broad_near_miss": 1, "target_equivalent": 6}` |
| deepseek7b | `exact_choices` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| deepseek7b | `natural_category` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `object_only` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |

## Pair Records

| model | pair | exact | natural count | exact+multi | degraded | single had exact+multi |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `L14:layer_residual:whole_layer_residual:B32 + L14:attention_output:whole_attention_output:B32` | 1 | 1 | 0 | 1 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B32 + L14:attention_head:head_3:B32` | 1 | 1 | 0 | 1 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_output:whole_attention_output:B16` | 1 | 1 | 0 | 0 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B32 + L14:attention_head:head_4:B32` | 1 | 1 | 0 | 1 | 1 |
| qwen3 | `L14:attention_output:whole_attention_output:B32 + L14:attention_head:head_3:B32` | 1 | 1 | 0 | 0 | 0 |
| qwen3 | `L14:attention_output:whole_attention_output:B32 + L14:attention_head:head_4:B32` | 0 | 0 | 0 | 0 | 0 |
| qwen3 | `L14:attention_head:head_3:B32 + L14:attention_head:head_4:B32` | 1 | 1 | 0 | 0 | 0 |
| deepseek7b | `L27:mlp_channel_group:mlp_topdiff_32:B32 + L27:layer_residual:whole_layer_residual:B32` | 1 | 2 | 1 | 0 | 1 |
| deepseek7b | `L27:mlp_channel_group:mlp_topdiff_32:B16 + L27:layer_residual:whole_layer_residual:B16` | 1 | 1 | 0 | 0 | 1 |
