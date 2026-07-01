# Phase 839 Gear Interaction Edge and Minimal Set (confirm)

- Source: Phase 838 top gear components, tested on held-out cases.
- Boundary: patch-intervention interaction test; not natural mechanism proof.

## Model Summary

| model | rows | components | cases | target | object_echo | format_echo | degraded | positive interaction | minimal candidates | mean quality | mean echo risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 240 | 4 | 4 | 120 | 0 | 0 | 0 | 8 | 0 | 0.4994 | 0.0000 |
| glm4 | 240 | 4 | 4 | 240 | 0 | 0 | 0 | 0 | 0 | 1.0010 | 0.0000 |
| deepseek7b | 240 | 4 | 4 | 93 | 55 | 60 | 0 | 13 | 4 | 0.0928 | 0.3542 |

## Combo Kind Summary

| model | combo kind | n | target | object_echo | format_echo | positive interaction | minimal | mean quality | mean echo risk | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `pair` | 96 | 48 | 0 | 0 | 2 | 0 | 0.4986 | 0.0000 | `{"broad_near_miss": 48, "target_equivalent": 48}` |
| qwen3 | `set3` | 64 | 32 | 0 | 0 | 4 | 0 | 0.5036 | 0.0000 | `{"broad_near_miss": 32, "target_equivalent": 32}` |
| qwen3 | `set4` | 16 | 8 | 0 | 0 | 2 | 0 | 0.5146 | 0.0000 | `{"broad_near_miss": 8, "target_equivalent": 8}` |
| qwen3 | `single` | 64 | 32 | 0 | 0 | 0 | 0 | 0.4926 | 0.0000 | `{"broad_near_miss": 32, "target_equivalent": 32}` |
| glm4 | `pair` | 96 | 96 | 0 | 0 | 0 | 0 | 1.0008 | 0.0000 | `{"target_equivalent": 96}` |
| glm4 | `set3` | 64 | 64 | 0 | 0 | 0 | 0 | 1.0011 | 0.0000 | `{"target_equivalent": 64}` |
| glm4 | `set4` | 16 | 16 | 0 | 0 | 0 | 0 | 1.0013 | 0.0000 | `{"target_equivalent": 16}` |
| glm4 | `single` | 64 | 64 | 0 | 0 | 0 | 0 | 1.0011 | 0.0000 | `{"target_equivalent": 64}` |
| deepseek7b | `pair` | 96 | 36 | 24 | 24 | 7 | 4 | 0.0690 | 0.3750 | `{"format_echo": 24, "object_echo": 24, "target_equivalent": 36, "unknown_other": 12}` |
| deepseek7b | `set3` | 64 | 28 | 8 | 16 | 5 | 0 | 0.1932 | 0.2500 | `{"format_echo": 16, "object_echo": 8, "target_equivalent": 28, "unknown_other": 12}` |
| deepseek7b | `set4` | 16 | 8 | 0 | 4 | 1 | 0 | 0.3180 | 0.1250 | `{"format_echo": 4, "target_equivalent": 8, "unknown_other": 4}` |
| deepseek7b | `single` | 64 | 21 | 23 | 16 | 0 | 0 | -0.0283 | 0.4844 | `{"format_echo": 16, "object_echo": 23, "target_equivalent": 21, "unknown_other": 4}` |

## Top Interaction Rows

| model | case | donor | kind | combo | class | output | quality | gain | echo risk | echo gain | minimal |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | `p816_oxygen_chemical_element` | `object_only` | `set3` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_output:whole_attention_output:B16 + p816_heart_body_organ::L14:attention_head:head_4:B32` | `broad_near_miss` | Gas | 0.1279 | 0.0832 | 0.0000 | 0.0000 | 0 |
| qwen3 | `p816_oxygen_chemical_element` | `object_only` | `set3` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_output:whole_attention_output:B16 + p816_heart_body_organ::L14:attention_head:head_3:B32` | `broad_near_miss` | Gas | 0.1323 | 0.0756 | 0.0000 | 0.0000 | 0 |
| qwen3 | `p816_oxygen_chemical_element` | `object_only` | `set4` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_output:whole_attention_output:B16 + p816_heart_body_organ::L14:attention_head:head_4:B32 + p816_heart_body_organ::L14:attention_head:head_3:B32` | `broad_near_miss` | Gas | 0.1263 | 0.0697 | 0.0000 | 0.0000 | 0 |
| qwen3 | `p816_oxygen_chemical_element` | `object_only` | `pair` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_head:head_3:B32` | `broad_near_miss` | Gas | 0.1234 | 0.0667 | 0.0000 | 0.0000 | 0 |
| qwen3 | `p816_triangle_geometric_shape` | `object_only` | `set4` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_output:whole_attention_output:B16 + p816_heart_body_organ::L14:attention_head:head_4:B32 + p816_heart_body_organ::L14:attention_head:head_3:B32` | `broad_near_miss` | Geometry | 0.0701 | 0.0637 | 0.0000 | 0.0000 | 0 |
| qwen3 | `p816_oxygen_chemical_element` | `natural_category` | `pair` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_head:head_4:B32` | `broad_near_miss` | Gas | 0.1633 | 0.0601 | 0.0000 | 0.0000 | 0 |
| qwen3 | `p816_oxygen_chemical_element` | `natural_category` | `set3` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_output:whole_attention_output:B16 + p816_heart_body_organ::L14:attention_head:head_4:B32` | `broad_near_miss` | Gas | 0.1630 | 0.0597 | 0.0000 | 0.0000 | 0 |
| qwen3 | `p816_oxygen_chemical_element` | `natural_category` | `set3` | `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16 + p816_heart_body_organ::L14:attention_head:head_4:B32 + p816_heart_body_organ::L14:attention_head:head_3:B32` | `broad_near_miss` | Gas | 0.1597 | 0.0565 | 0.0000 | 0.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `exact_choices` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B32` | `unknown_other` | Oxygen is a gas | 0.0302 | 0.6892 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `exact_choices` | `set3` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B32` | `unknown_other` | Oxygen is a gas | 0.0302 | 0.6892 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `exact_choices` | `set3` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B32` | `unknown_other` | Oxygen is a gas | 0.0302 | 0.6892 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `exact_choices` | `set4` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B32` | `unknown_other` | Oxygen is a gas | 0.0302 | 0.6892 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `unknown_other` | Oxygen is a gas | -0.1109 | 0.5736 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `unknown_other` | Oxygen is a gas | -0.1109 | 0.5736 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_oxygen_chemical_element` | `natural_category` | `set3` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `unknown_other` | Oxygen is a gas | -0.1109 | 0.5736 | 0.0000 | 1.0000 | 0 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 1.0730 | 0.0843 | 0.0000 | 0.0000 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 1.0730 | 0.0843 | 0.0000 | 0.0000 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `set3` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 1.0730 | 0.0843 | 0.0000 | 0.0000 | 0 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 0.9458 | 0.0590 | 0.0000 | 0.0000 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 0.9458 | 0.0590 | 0.0000 | 0.0000 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `set3` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 0.9458 | 0.0590 | 0.0000 | 0.0000 | 0 |
