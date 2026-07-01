# Phase 828 Cross-Component Consistency Fiber Composition (smoke)

- Source: Phase 827 selected subspaces.
- Objective: simultaneous two-component patching, then exact/natural consistency audit.

## Model Summary

| model | groups | pairs | exact target | natural target | natural degraded | pair exact+multi | new exact+multi | preserve exact+multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 4 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| glm4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 3 | 3 | 3 | 0 | 2.000 | `{"target_equivalent": 3}` |
| qwen3 | `natural_category` | 3 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3}` |
| deepseek7b | `exact_choices` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `natural_category` | 1 | 0 | 0 | 0 | 0.000 | `{"object_echo": 1}` |

## Pair Records

| model | pair | exact | natural count | exact+multi | degraded | single had exact+multi |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_output:whole_attention_output:B16` | 1 | 0 | 0 | 0 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B16 + L14:mlp_output:whole_mlp_output:B16` | 1 | 0 | 0 | 0 | 1 |
| qwen3 | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_head:head_4:B16` | 1 | 0 | 0 | 0 | 1 |
| deepseek7b | `L27:mlp_channel_group:mlp_topdiff_32:B16 + L27:layer_residual:whole_layer_residual:B16` | 1 | 0 | 0 | 0 | 1 |
