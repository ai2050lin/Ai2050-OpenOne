# Phase 830 Donor-Gated Non-Interference Composition (smoke)

- Source: Phase 829 selected non-interfering pairs.
- Objective: activate only donor-compatible components inside each pair.

## Model Summary

| model | pairs | exact target | natural target | natural degraded | natural_category degraded | pair exact+multi | new exact+multi | preserve exact+multi | active components |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 3 | 3 | 2 | 0 | 0 | 2 | 0 | 2 | `{"0": 1, "1": 5}` |
| glm4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | `{"1": 2}` |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 3 | 3 | 3 | 0 | 2.000 | `{"target_equivalent": 3}` |
| qwen3 | `natural_category` | 3 | 2 | 2 | 0 | 1.333 | `{"broad_near_miss": 1, "target_equivalent": 2}` |
| deepseek7b | `exact_choices` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |
| deepseek7b | `natural_category` | 1 | 1 | 1 | 0 | 5.000 | `{"target_equivalent": 1}` |

## Pair Records

| model | case | pair | exact | natural count | exact+multi | degraded | single had exact+multi |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `p816_heart_body_organ` | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_output:whole_attention_output:B16` | 1 | 1 | 1 | 0 | 1 |
| qwen3 | `p816_heart_body_organ` | `L14:layer_residual:whole_layer_residual:B16 + L14:mlp_output:whole_mlp_output:B16` | 1 | 0 | 0 | 0 | 1 |
| qwen3 | `p816_heart_body_organ` | `L14:layer_residual:whole_layer_residual:B16 + L14:attention_head:head_4:B16` | 1 | 1 | 1 | 0 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `L27:mlp_channel_group:mlp_topdiff_32:B16 + L27:layer_residual:whole_layer_residual:B16` | 1 | 1 | 1 | 0 | 1 |
