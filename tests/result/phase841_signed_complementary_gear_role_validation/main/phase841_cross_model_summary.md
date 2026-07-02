# Phase 841 Signed Complementary Gear Role Validation (main)

- Source: Phase 840 strict pair rows and inferred role signs.
- Boundary: patch-mode perturbation evidence; not full natural ablation.

## Model Summary

| model | skipped | candidates | roles | rows | cases | pair-original target | target lost vs original | negative-role needed | positive-role needed | object_echo | format_echo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| glm4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 0 | 2 | 3 | 28 | 1 | 4 | 8 | 8 | 0 | 8 | 0 |

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
| deepseek7b | `flip_negative` | 4 | 0 | 4 | 4 | 0 | -0.5366 | -1.5459 | `{"object_echo": 4}` |
| deepseek7b | `flip_positive` | 4 | 4 | 0 | 0 | 0 | 0.8639 | -0.1454 | `{"target_equivalent": 4}` |
| deepseek7b | `negative_only` | 4 | 4 | 0 | 0 | 0 | 0.9377 | -0.0716 | `{"target_equivalent": 4}` |
| deepseek7b | `pair_original` | 4 | 4 | 0 | 0 | 0 | 1.0094 | 0.0000 | `{"target_equivalent": 4}` |
| deepseek7b | `positive_only` | 4 | 0 | 4 | 4 | 0 | -0.5919 | -1.6013 | `{"object_echo": 4}` |
| deepseek7b | `zero_negative` | 4 | 4 | 0 | 0 | 0 | 0.9562 | -0.0532 | `{"target_equivalent": 4}` |
| deepseek7b | `zero_positive` | 4 | 4 | 0 | 0 | 0 | 1.0304 | 0.0210 | `{"target_equivalent": 4}` |

## Top Mode Rows

| model | case | donor | mode | class | output | target | original target | lost | neg needed | pos needed | quality | delta quality |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5691 | -1.6421 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5691 | -1.6421 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_negative` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5601 | -1.6331 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_negative` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5601 | -1.6331 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.6147 | -1.5604 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `positive_only` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.6147 | -1.5604 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_negative` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5131 | -1.4588 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_negative` | `object_echo` | Triangle | 0 | 1 | 1 | 1 | 0 | -0.5131 | -1.4588 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.8873 | -0.1857 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `flip_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.8873 | -0.1857 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.8405 | -0.1052 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `flip_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.8405 | -0.1052 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_negative` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9796 | -0.0934 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_negative` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9796 | -0.0934 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `negative_only` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9887 | -0.0843 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `negative_only` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9887 | -0.0843 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `negative_only` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.8868 | -0.0590 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `negative_only` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.8868 | -0.0590 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 1.1047 | 0.0317 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `zero_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 1.1047 | 0.0317 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_negative` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9329 | -0.0129 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_negative` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9329 | -0.0129 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9561 | 0.0104 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `zero_positive` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9561 | 0.0104 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `pair_original` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 1.0730 | 0.0000 |
| deepseek7b | `p816_triangle_geometric_shape` | `object_only` | `pair_original` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 1.0730 | 0.0000 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `pair_original` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9458 | 0.0000 |
| deepseek7b | `p816_triangle_geometric_shape` | `natural_question` | `pair_original` | `target_equivalent` | polygon | 1 | 1 | 0 | 0 | 0 | 0.9458 | 0.0000 |
