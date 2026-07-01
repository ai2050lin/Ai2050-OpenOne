# Phase 837 Global Gear Response Atlas Pilot (smoke)

- Objective: collect standardized gear-like response fingerprints across components, cases, donors, and output metrics.
- Boundary: pilot atlas only; no final gear decomposition yet.

## Model Summary

| model | rows | groups | cases | target | improved | degraded | object_echo | contrast_cleared | echo_cleared | rank_improved | top response types |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 32 | 4 | 4 | 32 | 0 | 0 | 0 | 32 | 0 | 13 | `{"target_writer_candidate": 32}` |
| glm4 | 32 | 4 | 4 | 24 | 0 | 0 | 0 | 32 | 0 | 13 | `{"echo_amplifier_or_unsuppressed": 8, "target_writer_candidate": 24}` |
| deepseek7b | 32 | 4 | 4 | 0 | 0 | 0 | 0 | 32 | 0 | 12 | `{"echo_amplifier_or_unsuppressed": 32}` |

## Reuse Candidates

### qwen3

| component | target cases | contrast-cleared cases | echo-cleared cases | response types |
|---|---:|---:|---:|---|
| `p816_heart_body_organ::L14:mlp_output:whole_mlp_output:B16` | 4 | 4 | 0 | `{"target_writer_candidate": 8}` |
| `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16` | 4 | 4 | 0 | `{"target_writer_candidate": 8}` |
| `p816_heart_body_organ::L14:attention_output:whole_attention_output:B16` | 4 | 4 | 0 | `{"target_writer_candidate": 8}` |
| `p816_heart_body_organ::L14:attention_head:head_4:B16` | 4 | 4 | 0 | `{"target_writer_candidate": 8}` |

### glm4

| component | target cases | contrast-cleared cases | echo-cleared cases | response types |
|---|---:|---:|---:|---|
| `p816_winter_cold_season::L8:layer_residual:whole_layer_residual:B16` | 3 | 4 | 0 | `{"echo_amplifier_or_unsuppressed": 2, "target_writer_candidate": 6}` |
| `p816_red_warm_color::L23:layer_residual:whole_layer_residual:B16` | 3 | 4 | 0 | `{"echo_amplifier_or_unsuppressed": 2, "target_writer_candidate": 6}` |
| `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B16` | 3 | 4 | 0 | `{"echo_amplifier_or_unsuppressed": 2, "target_writer_candidate": 6}` |
| `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B16` | 3 | 4 | 0 | `{"echo_amplifier_or_unsuppressed": 2, "target_writer_candidate": 6}` |

### deepseek7b

| component | target cases | contrast-cleared cases | echo-cleared cases | response types |
|---|---:|---:|---:|---|
| `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16` | 0 | 4 | 0 | `{"echo_amplifier_or_unsuppressed": 8}` |
| `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | 0 | 4 | 0 | `{"echo_amplifier_or_unsuppressed": 8}` |
| `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B16` | 0 | 4 | 0 | `{"echo_amplifier_or_unsuppressed": 8}` |
| `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | 0 | 4 | 0 | `{"echo_amplifier_or_unsuppressed": 8}` |

