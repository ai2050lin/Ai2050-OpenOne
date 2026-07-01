# Phase 839 Gear Interaction Edge and Minimal Set (main)

- Source: Phase 838 top gear components, tested on held-out cases.
- Boundary: patch-intervention interaction test; not natural mechanism proof.

## Model Summary

| model | rows | components | cases | target | object_echo | format_echo | degraded | positive interaction | minimal candidates | mean quality | mean echo risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 84 | 3 | 4 | 42 | 0 | 0 | 0 | 3 | 0 | 0.5085 | 0.0000 |
| glm4 | 84 | 3 | 4 | 84 | 0 | 0 | 0 | 0 | 0 | 1.0028 | 0.0000 |
| deepseek7b | 84 | 3 | 4 | 30 | 29 | 21 | 0 | 6 | 2 | -0.0175 | 0.4702 |

## Combo Kind Summary

| model | combo kind | n | target | object_echo | format_echo | positive interaction | minimal | mean quality | mean echo risk | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `pair` | 36 | 18 | 0 | 0 | 1 | 0 | 0.5110 | 0.0000 | `{"broad_near_miss": 18, "target_equivalent": 18}` |
| qwen3 | `set3` | 12 | 6 | 0 | 0 | 2 | 0 | 0.5367 | 0.0000 | `{"broad_near_miss": 6, "target_equivalent": 6}` |
| qwen3 | `single` | 36 | 18 | 0 | 0 | 0 | 0 | 0.4967 | 0.0000 | `{"broad_near_miss": 18, "target_equivalent": 18}` |
| glm4 | `pair` | 36 | 36 | 0 | 0 | 0 | 0 | 1.0033 | 0.0000 | `{"target_equivalent": 36}` |
| glm4 | `set3` | 12 | 12 | 0 | 0 | 0 | 0 | 1.0143 | 0.0000 | `{"target_equivalent": 12}` |
| glm4 | `single` | 36 | 36 | 0 | 0 | 0 | 0 | 0.9984 | 0.0000 | `{"target_equivalent": 36}` |
| deepseek7b | `pair` | 36 | 13 | 12 | 9 | 4 | 2 | -0.0144 | 0.4583 | `{"format_echo": 9, "object_echo": 12, "target_equivalent": 13, "unknown_other": 2}` |
| deepseek7b | `set3` | 12 | 5 | 3 | 3 | 2 | 0 | 0.0704 | 0.3750 | `{"format_echo": 3, "object_echo": 3, "target_equivalent": 5, "unknown_other": 1}` |
| deepseek7b | `single` | 36 | 12 | 14 | 9 | 0 | 0 | -0.0498 | 0.5139 | `{"format_echo": 9, "object_echo": 14, "target_equivalent": 12, "unknown_other": 1}` |

## Top Interaction Rows

| model | case | donor | kind | combo | class | output | quality | gain | echo risk | echo gain | minimal |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `p816_oxygen_chemical_element` | `object_only` | `set3` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_output:whole_attention_output:B16 + p816_heart_body_organ::L14:attention_head:head_4:B32` | `broad_near_miss` | Gas | 0.1279 | 0.0832 | 0.0000 | 0.0000 | 0 |
| qwen3 | `p816_oxygen_chemical_element` | `natural_category` | `pair` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_head:head_4:B32` | `broad_near_miss` | Gas | 0.1633 | 0.0601 | 0.0000 | 0.0000 | 0 |
| qwen3 | `p816_oxygen_chemical_element` | `natural_category` | `set3` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_output:whole_attention_output:B16 + p816_heart_body_organ::L14:attention_head:head_4:B32` | `broad_near_miss` | Gas | 0.1630 | 0.0597 | 0.0000 | 0.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `unknown_other` | Oxygen is a gas | -0.1109 | 0.5736 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `unknown_other` | Oxygen is a gas | -0.1109 | 0.5736 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `set3` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `unknown_other` | Oxygen is a gas | -0.1109 | 0.5736 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 1.0730 | 0.0843 | 0.0000 | 0.0000 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 1.0730 | 0.0843 | 0.0000 | 0.0000 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `set3` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 1.0730 | 0.0843 | 0.0000 | 0.0000 | 0 |
