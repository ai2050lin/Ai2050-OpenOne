# Phase 838: gear response decomposition and prediction validation

- timestamp: 2026-07-01 23:58:08
- input: `tests/result/phase837_global_gear_response_atlas_pilot/confirm`
- boundary: offline analysis over Phase 837 confirm rows; no new model forward pass.

## qwen3

| rows | components | train cases | holdout cases | train->holdout pearson | train->holdout spearman | top-k lift quality | top-k lift target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 8 | 4 | -0.3670 | -0.0238 | 0.0130 | 0.0000 |

Family counts:
- broad_target_writer_family: 3
- conditional_target_writer_family: 5

Top train components:
- `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16`
- `p816_heart_body_organ::L14:attention_output:whole_attention_output:B16`

Top similarity edges:
- `p816_heart_body_organ::L14:mlp_output:whole_mlp_output:B16` <-> `p816_heart_body_organ::L14:mlp_output:whole_mlp_output:B32`: 0.9812
- `p816_heart_body_organ::L14:attention_head:head_4:B32` <-> `p816_heart_body_organ::L14:attention_output:whole_attention_output:B16`: 0.9760
- `p816_heart_body_organ::L14:attention_head:head_3:B32` <-> `p816_heart_body_organ::L14:attention_output:whole_attention_output:B32`: 0.9693
- `p816_heart_body_organ::L14:attention_head:head_4:B32` <-> `p816_heart_body_organ::L14:mlp_output:whole_mlp_output:B32`: 0.9608
- `p816_heart_body_organ::L14:attention_head:head_4:B32` <-> `p816_heart_body_organ::L14:mlp_output:whole_mlp_output:B16`: 0.9591

Clusters:
- cluster 1: `p816_heart_body_organ::L14:attention_head:head_3:B32`, `p816_heart_body_organ::L14:attention_head:head_4:B32`, `p816_heart_body_organ::L14:attention_output:whole_attention_output:B16`, `p816_heart_body_organ::L14:attention_output:whole_attention_output:B32`, `p816_heart_body_organ::L14:mlp_output:whole_mlp_output:B16`, `p816_heart_body_organ::L14:mlp_output:whole_mlp_output:B32`
- cluster 2: `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B16`, `p816_heart_body_organ::L14:layer_residual:whole_layer_residual:B32`

## glm4

| rows | components | train cases | holdout cases | train->holdout pearson | train->holdout spearman | top-k lift quality | top-k lift target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 8 | 4 | 0.3766 | 0.0000 | 0.0084 | 0.0000 |

Family counts:
- broad_target_writer_family: 8

Top train components:
- `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B16`
- `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B32`

Top similarity edges:
- `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B16` <-> `p816_winter_cold_season::L8:layer_residual:whole_layer_residual:B16`: 1.0000
- `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B32` <-> `p816_winter_cold_season::L8:layer_residual:whole_layer_residual:B32`: 1.0000
- `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B16` <-> `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B32`: 0.9768
- `p816_red_warm_color::L23:layer_residual:whole_layer_residual:B16` <-> `p816_red_warm_color::L23:layer_residual:whole_layer_residual:B32`: 0.9746
- `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B16` <-> `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B32`: 0.9718

Clusters:
- cluster 1: `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B16`, `p816_cactus_desert_plant::L8:layer_residual:whole_layer_residual:B32`, `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B16`, `p816_carrot_root_vegetable::L39:layer_residual:whole_layer_residual:B32`, `p816_red_warm_color::L23:layer_residual:whole_layer_residual:B16`, `p816_red_warm_color::L23:layer_residual:whole_layer_residual:B32`, `p816_winter_cold_season::L8:layer_residual:whole_layer_residual:B16`, `p816_winter_cold_season::L8:layer_residual:whole_layer_residual:B32`

## deepseek7b

| rows | components | train cases | holdout cases | train->holdout pearson | train->holdout spearman | top-k lift quality | top-k lift target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 8 | 8 | 4 | -0.3877 | -0.1566 | 0.0777 | 0.0547 |

Family counts:
- echo_dominated_family: 7
- weak_or_unresolved_family: 1

Top train components:
- `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16`
- `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16`

Top similarity edges:
- `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` <-> `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16`: 1.0000
- `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B16` <-> `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B32`: 0.9529
- `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` <-> `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B32`: 0.9379
- `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B32` <-> `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16`: 0.9379
- `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16` <-> `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B16`: 0.9308

Clusters:
- cluster 1: `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B16`, `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B16`, `p816_doctor_medical_worker::L5:layer_residual:whole_layer_residual:B32`, `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B16`
- cluster 2: `p816_cat_living_thing::L27:layer_residual:whole_layer_residual:B32`
- cluster 3: `p816_triangle_geometric_shape::L27:layer_residual:whole_layer_residual:B32`
- cluster 4: `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B16`
- cluster 5: `p816_triangle_geometric_shape::L27:mlp_channel_group:mlp_topdiff_32:B32`

