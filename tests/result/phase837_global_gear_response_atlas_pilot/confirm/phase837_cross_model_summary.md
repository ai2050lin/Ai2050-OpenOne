# Phase 837 Global Gear Response Atlas Pilot (confirm)

- Objective: collect standardized gear-like response fingerprints across components, cases, donors, and output metrics.
- Boundary: pilot atlas only; no final gear decomposition yet.

## Model Summary

| model | rows | groups | cases | target | improved | degraded | object_echo | contrast_cleared | echo_cleared | rank_improved | top response types |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 384 | 8 | 12 | 267 | 14 | 4 | 0 | 384 | 0 | 191 | `{"echo_amplifier_or_unsuppressed": 113, "harmful_mixer": 4, "target_writer_candidate": 267}` |
| glm4 | 384 | 8 | 12 | 320 | 0 | 12 | 0 | 384 | 0 | 117 | `{"echo_amplifier_or_unsuppressed": 52, "harmful_mixer": 12, "target_writer_candidate": 320}` |
| deepseek7b | 384 | 8 | 12 | 133 | 9 | 7 | 45 | 352 | 0 | 185 | `{"echo_amplifier_or_unsuppressed": 244, "harmful_mixer": 7, "target_writer_candidate": 133}` |

## Reuse Candidates

### qwen3

| component | target cases | contrast-cleared cases | echo-cleared cases | response types |
|---|---:|---:|---:|---|
| `p816_heart_body_organ::L14:mlp_output:whole_mlp_output:B32` | 9 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 15, "target_writer_candidate": 33}` |
| `p816_heart_body_organ::L14:mlp_output:whole_mlp_output:B16` | 9 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 15, "target_writer_candidate": 33}` |
| `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B32` | 9 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 12, "harmful_mixer": 3, "target_writer_candidate": 33}` |
| `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16` | 9 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 13, "harmful_mixer": 1, "target_writer_candidate": 34}` |
| `p816_heart_body_organ::L14:attention_output:whole_attention_output:B32` | 9 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 15, "target_writer_candidate": 33}` |
| `p816_heart_body_organ::L14:attention_output:whole_attention_output:B16` | 9 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 14, "target_writer_candidate": 34}` |
| `p816_heart_body_organ::L14:attention_head:head_4:B32` | 9 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 14, "target_writer_candidate": 34}` |
| `p816_heart_body_organ::L14:attention_head:head_3:B32` | 9 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 15, "target_writer_candidate": 33}` |

### glm4

| component | target cases | contrast-cleared cases | echo-cleared cases | response types |
|---|---:|---:|---:|---|
| `p816_winter_cold_season::L8:layer_residual:whole_layer_residual:B32` | 10 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 7, "harmful_mixer": 1, "target_writer_candidate": 40}` |
| `p816_winter_cold_season::L8:layer_residual:whole_layer_residual:B16` | 10 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 6, "harmful_mixer": 2, "target_writer_candidate": 40}` |
| `p816_red_warm_color::L23:layer_residual:whole_layer_residual:B32` | 10 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 6, "harmful_mixer": 2, "target_writer_candidate": 40}` |
| `p816_red_warm_color::L23:layer_residual:whole_layer_residual:B16` | 10 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 5, "harmful_mixer": 3, "target_writer_candidate": 40}` |
| `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B32` | 10 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 7, "harmful_mixer": 1, "target_writer_candidate": 40}` |
| `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B16` | 10 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 8, "target_writer_candidate": 40}` |
| `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B32` | 10 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 7, "harmful_mixer": 1, "target_writer_candidate": 40}` |
| `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B16` | 10 | 12 | 0 | `{"echo_amplifier_or_unsuppressed": 6, "harmful_mixer": 2, "target_writer_candidate": 40}` |

### deepseek7b

| component | target cases | contrast-cleared cases | echo-cleared cases | response types |
|---|---:|---:|---:|---|
| `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16` | 5 | 11 | 0 | `{"echo_amplifier_or_unsuppressed": 28, "target_writer_candidate": 20}` |
| `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B32` | 5 | 11 | 0 | `{"echo_amplifier_or_unsuppressed": 31, "target_writer_candidate": 17}` |
| `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B32` | 4 | 11 | 0 | `{"echo_amplifier_or_unsuppressed": 25, "harmful_mixer": 7, "target_writer_candidate": 16}` |
| `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16` | 4 | 11 | 0 | `{"echo_amplifier_or_unsuppressed": 32, "target_writer_candidate": 16}` |
| `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B32` | 4 | 11 | 0 | `{"echo_amplifier_or_unsuppressed": 32, "target_writer_candidate": 16}` |
| `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B16` | 4 | 11 | 0 | `{"echo_amplifier_or_unsuppressed": 32, "target_writer_candidate": 16}` |
| `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B32` | 4 | 11 | 0 | `{"echo_amplifier_or_unsuppressed": 32, "target_writer_candidate": 16}` |
| `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` | 4 | 11 | 0 | `{"echo_amplifier_or_unsuppressed": 32, "target_writer_candidate": 16}` |

