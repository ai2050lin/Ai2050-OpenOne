# Phase 840 Strict Target Interaction and Natural Co-activation (confirm)

- Source: Phase 839 confirm strict target-positive interaction rows only.
- Boundary: patch expansion plus natural donor-state co-activation audit; not natural causal ablation.

## Model Summary

| model | skipped | candidates | rows | cases | target | strict positive | natural-supported strict | minimal | object_echo | format_echo | mean quality | mean echo risk | mean natural positive ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | NA |
| glm4 | 1 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | NA |
| deepseek7b | 0 | 2 | 80 | 4 | 30 | 4 | 0 | 4 | 24 | 20 | 0.0415 | 0.4250 | 0.3000 |

## Combo Records

| model | kind | combo | rows | target | strict | natural-supported | minimal | mean quality | mean echo | natural ratio | classes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | 16 | 7 | 2 | 0 | 2 | 0.1524 | 0.3125 | 0.2812 | `{"format_echo": 4, "object_echo": 3, "target_equivalent": 7, "unknown_other": 2}` |
| deepseek7b | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | 16 | 7 | 2 | 0 | 2 | 0.1524 | 0.3125 | 0.2812 | `{"format_echo": 4, "object_echo": 3, "target_equivalent": 7, "unknown_other": 2}` |
| deepseek7b | `single_control` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16` | 16 | 8 | 0 | 0 | 0 | 0.2580 | 0.2500 | 0.1875 | `{"format_echo": 4, "object_echo": 2, "target_equivalent": 8, "unknown_other": 2}` |
| deepseek7b | `single_control` | `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | 16 | 4 | 0 | 0 | 0 | -0.1777 | 0.6250 | 0.3750 | `{"format_echo": 4, "object_echo": 8, "target_equivalent": 4}` |
| deepseek7b | `single_control` | `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | 16 | 4 | 0 | 0 | 0 | -0.1777 | 0.6250 | 0.3750 | `{"format_echo": 4, "object_echo": 8, "target_equivalent": 4}` |

## Top Strict Rows

| model | case | donor | kind | combo | class | output | quality | gain | echo | natural ratio | natural all+ | natural-supported | minimal |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 1.0730 | 0.0843 | 0.0000 | 0.5000 | 0 | 0 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 1.0730 | 0.0843 | 0.0000 | 0.5000 | 0 | 0 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 0.9458 | 0.0590 | 0.0000 | 0.5000 | 0 | 0 | 1 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 0.9458 | 0.0590 | 0.0000 | 0.5000 | 0 | 0 | 1 |
