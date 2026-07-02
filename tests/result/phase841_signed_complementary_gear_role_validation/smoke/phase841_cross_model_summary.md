# Phase 841 Signed Complementary Gear Role Validation (smoke)

- Source: Phase 840 strict pair rows and inferred role signs.
- Boundary: patch-mode perturbation evidence; not full natural ablation.

## Model Summary

| model | skipped | candidates | roles | rows | cases | pair-original target | target lost vs original | negative-role needed | positive-role needed | object_echo | format_echo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| glm4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 0 | 1 | 3 | 5 | 1 | 1 | 2 | 2 | 0 | 2 | 0 |

## Role Signs

### deepseek7b

| component | role | mean signed sum | observations |
|---|---|---:|---:|
| `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | `positive_carrier` | 1.2119 | 2 |
| `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | `positive_carrier` | 1.2119 | 2 |
| `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16` | `negative_suppressor_or_rewriter` | -3.1581 | 4 |

## Mode Summary

| model | mode | n | target | lost vs original | negative needed | positive needed | mean quality | mean delta quality | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `flip_negative` | 1 | 0 | 1 | 1 | 0 | -0.5765 | -1.5276 | `{"object_echo": 1}` |
| deepseek7b | `negative_only` | 1 | 1 | 0 | 0 | 0 | 0.8943 | -0.0567 | `{"target_equivalent": 1}` |
| deepseek7b | `pair_original` | 1 | 1 | 0 | 0 | 0 | 0.9510 | 0.0000 | `{"target_equivalent": 1}` |
| deepseek7b | `positive_only` | 1 | 0 | 1 | 1 | 0 | -0.6706 | -1.6216 | `{"object_echo": 1}` |
| deepseek7b | `zero_negative` | 1 | 1 | 0 | 0 | 0 | 0.9545 | 0.0035 | `{"target_equivalent": 1}` |

## Top Mode Rows

| model | case | donor | mode | class | output | target | original target | lost | neg needed | pos needed | quality | delta quality |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.6706 | -1.6216 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_negative` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5765 | -1.5276 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `negative_only` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.8943 | -0.0567 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_negative` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9545 | 0.0035 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `pair_original` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9510 | 0.0000 |
