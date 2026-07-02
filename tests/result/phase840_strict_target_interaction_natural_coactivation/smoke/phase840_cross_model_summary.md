# Phase 840 Strict Target Interaction and Natural Co-activation (smoke)

- Source: Phase 839 confirm strict target-positive interaction rows only.
- Boundary: patch expansion plus natural donor-state co-activation audit; not natural causal ablation.

## Model Summary

| model | skipped | candidates | rows | cases | target | strict positive | natural-supported strict | minimal | object_echo | format_echo | mean quality | mean echo risk | mean natural positive ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | NA |
| glm4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | NA |
| deepseek7b | 0 | 1 | 3 | 1 | 2 | 1 | 0 | 1 | 1 | 0 | 0.3916 | 0.3333 | 0.5000 |

## Combo Records

| model | kind | combo | rows | target | strict | natural-supported | minimal | mean quality | mean echo | natural ratio | classes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | 1 | 1 | 1 | 0 | 1 | 0.9510 | 0.0000 | 0.5000 | `{"target_equivalent": 1}` |
| deepseek7b | `single_control` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16` | 1 | 1 | 0 | 0 | 0 | 0.8943 | 0.0000 | 0.0000 | `{"target_equivalent": 1}` |
| deepseek7b | `single_control` | `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | 1 | 0 | 0 | 0 | 0 | -0.6706 | 1.0000 | 1.0000 | `{"object_echo": 1}` |

## Top Strict Rows

| model | case | donor | kind | combo | class | output | quality | gain | echo | natural ratio | natural all+ | natural-supported | minimal |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `pair` | `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16 + p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `target_equivalent` | polygon | 0.9510 | 0.0567 | 0.0000 | 0.5000 | 0 | 0 | 1 |
