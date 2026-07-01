# Phase 829 Non-Interference Constrained Component Composition (main)

- Source: Phase 827 selected subspaces, with Phase 828 composition logic.
- Objective: exclude unsafe single-component groups before two-component composition.

## Model Summary

| model | groups | eligible | pairs | exact target | natural target | natural degraded | natural_category degraded | pair exact+multi | new exact+multi | preserve exact+multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 16 | 15 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| glm4 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 10 | 10 | 2 | 2 | 3 | 0 | 0 | 1 | 0 | 1 |

## Constraint Diagnostics

| model | raw pairs | safe pairs before anchor | anchor excluded | selected pairs | excluded groups |
|---|---:|---:|---:|---:|---|
| qwen3 | 42 | 36 | 30 | 6 | `{"single_degraded": 1}` |
| glm4 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | 2 | 2 | 0 | 2 | `{}` |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 6 | 6 | 6 | 0 | 2.000 | `{"target_equivalent": 6}` |
| qwen3 | `natural_category` | 6 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 6}` |
| qwen3 | `object_only` | 6 | 6 | 6 | 0 | 2.000 | `{"target_equivalent": 6}` |
| deepseek7b | `exact_choices` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |
| deepseek7b | `natural_category` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `object_only` | 2 | 2 | 2 | 0 | 5.000 | `{"target_equivalent": 2}` |

## Pair Records

| model | case | pair | exact | natural count | exact+multi | degraded | single had exact+multi |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `p816_heart_body_organ` | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_output:whole_attention_output:B16` | 1 | 1 | 0 | 0 | 1 |
| qwen3 | `p816_heart_body_organ` | `L14:layer_residual:whole_layer_residual:B16 + L14:mlp_output:whole_mlp_output:B16` | 1 | 1 | 0 | 0 | 1 |
| qwen3 | `p816_heart_body_organ` | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_head:head_4:B16` | 1 | 1 | 0 | 0 | 1 |
| qwen3 | `p816_heart_body_organ` | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_head:head_3:B16` | 1 | 1 | 0 | 0 | 1 |
| qwen3 | `p816_heart_body_organ` | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_head:head_25:B16` | 1 | 1 | 0 | 0 | 1 |
| qwen3 | `p816_heart_body_organ` | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_head:head_8:B16` | 1 | 1 | 0 | 0 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `L27:mlp_channel_group:mlp_topdiff_32:B32 + L27:layer_residual:whole_layer_residual:B32` | 1 | 2 | 1 | 0 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `L27:mlp_channel_group:mlp_topdiff_32:B16 + L27:layer_residual:whole_layer_residual:B16` | 1 | 1 | 0 | 0 | 1 |
