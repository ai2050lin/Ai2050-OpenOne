# Phase 837 Global Gear Response Atlas Pilot (main)

- Objective: collect standardized gear-like response fingerprints across components, cases, donors, and output metrics.
- Boundary: pilot atlas only; no final gear decomposition yet.

## Model Summary

| model | rows | groups | cases | target | improved | degraded | object_echo | contrast_cleared | echo_cleared | rank_improved | top response types |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 144 | 6 | 8 | 87 | 0 | 3 | 0 | 144 | 0 | 64 | `{"echo_amplifier_or_unsuppressed": 54, "harmful_mixer": 3, "target_writer_candidate": 87}` |
| glm4 | 144 | 6 | 8 | 90 | 0 | 0 | 0 | 144 | 0 | 60 | `{"echo_amplifier_or_unsuppressed": 54, "target_writer_candidate": 90}` |
| deepseek7b | 144 | 6 | 8 | 43 | 7 | 0 | 11 | 144 | 0 | 75 | `{"echo_amplifier_or_unsuppressed": 101, "target_writer_candidate": 43}` |

## Reuse Candidates

### qwen3

| component | target cases | contrast-cleared cases | echo-cleared cases | response types |
|---|---:|---:|---:|---|
| `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B32` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "harmful_mixer": 2, "target_writer_candidate": 13}` |
| `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "harmful_mixer": 1, "target_writer_candidate": 14}` |
| `p816_heart_body_organ::L14:attention_output:whole_attention_output:B32` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |
| `p816_heart_body_organ::L14:attention_output:whole_attention_output:B16` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |
| `p816_heart_body_organ::L14:attention_head:head_4:B32` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |
| `p816_heart_body_organ::L14:attention_head:head_3:B32` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |

### glm4

| component | target cases | contrast-cleared cases | echo-cleared cases | response types |
|---|---:|---:|---:|---|
| `p816_winter_cold_season::L8:layer_residual:whole_layer_residual:B32` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |
| `p816_winter_cold_season::L8:layer_residual:whole_layer_residual:B16` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |
| `p816_red_warm_color::L23:layer_residual:whole_layer_residual:B32` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |
| `p816_red_warm_color::L23:layer_residual:whole_layer_residual:B16` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |
| `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B32` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |
| `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B16` | 5 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 9, "target_writer_candidate": 15}` |

### deepseek7b

| component | target cases | contrast-cleared cases | echo-cleared cases | response types |
|---|---:|---:|---:|---|
| `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B32` | 3 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 15, "target_writer_candidate": 9}` |
| `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16` | 3 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 15, "target_writer_candidate": 9}` |
| `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B32` | 3 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 17, "target_writer_candidate": 7}` |
| `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | 2 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 18, "target_writer_candidate": 6}` |
| `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B32` | 2 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 18, "target_writer_candidate": 6}` |
| `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B16` | 2 | 8 | 0 | `{"echo_amplifier_or_unsuppressed": 18, "target_writer_candidate": 6}` |

