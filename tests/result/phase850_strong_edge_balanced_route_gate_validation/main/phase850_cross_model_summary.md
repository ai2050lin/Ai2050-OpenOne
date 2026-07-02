# Phase 850 Strong-edge Balanced Route Gate Validation (main)

- Source: Phase 849 feature rows.
- Method: calibrated strong-edge class predictors with raw and balanced metrics.
- Boundary: strong-edge validation, not closure.

## Raw Strong-edge Summary

| model | rows | strong rows | split | predictor | n | strong F1 | recall | precision | balanced acc | macro F1 |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 304 | 32 | `in_sample` | `blocker_field_combo` | 304 | 0.9412 | 1.0000 | 0.8889 | 0.9926 | 0.9402 |
| qwen3 | 304 | 32 | `in_sample` | `compact_joint_gate_combo` | 304 | 0.9697 | 1.0000 | 0.9412 | 0.9963 | 0.9779 |
| qwen3 | 304 | 32 | `in_sample` | `global_combo` | 304 | 0.5938 | 0.5938 | 0.5938 | 0.7730 | 0.6563 |
| qwen3 | 304 | 32 | `in_sample` | `internal_strength_combo` | 304 | 0.8125 | 0.8125 | 0.8125 | 0.8952 | 0.8304 |
| qwen3 | 304 | 32 | `in_sample` | `joint_gate_combo` | 304 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| qwen3 | 304 | 32 | `in_sample` | `model_default_gate` | 304 | 0.9552 | 1.0000 | 0.9143 | 0.9945 | 0.9286 |
| qwen3 | 304 | 32 | `in_sample` | `residual_projection_combo` | 304 | 0.9552 | 1.0000 | 0.9143 | 0.9945 | 0.9286 |
| qwen3 | 304 | 32 | `in_sample` | `route_competition_combo` | 304 | 0.7619 | 1.0000 | 0.6154 | 0.9632 | 0.8186 |
| qwen3 | 304 | 32 | `in_sample` | `train_selected_gate` | 304 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| qwen3 | 304 | 32 | `object_holdout` | `blocker_field_combo` | 304 | 0.3729 | 0.3438 | 0.4074 | 0.6425 | 0.4886 |
| qwen3 | 304 | 32 | `object_holdout` | `compact_joint_gate_combo` | 304 | 0.4688 | 0.4688 | 0.4688 | 0.7031 | 0.5145 |
| qwen3 | 304 | 32 | `object_holdout` | `global_combo` | 304 | 0.4054 | 0.4688 | 0.3571 | 0.6847 | 0.5639 |
| qwen3 | 304 | 32 | `object_holdout` | `internal_strength_combo` | 304 | 0.4839 | 0.4688 | 0.5000 | 0.7068 | 0.5562 |
| qwen3 | 304 | 32 | `object_holdout` | `joint_gate_combo` | 304 | 0.4194 | 0.4062 | 0.4333 | 0.6719 | 0.5264 |
| qwen3 | 304 | 32 | `object_holdout` | `model_default_gate` | 304 | 0.4483 | 0.4062 | 0.5000 | 0.6792 | 0.5720 |
| qwen3 | 304 | 32 | `object_holdout` | `residual_projection_combo` | 304 | 0.4483 | 0.4062 | 0.5000 | 0.6792 | 0.5720 |
| qwen3 | 304 | 32 | `object_holdout` | `route_competition_combo` | 304 | 0.4000 | 0.4062 | 0.3939 | 0.6664 | 0.5518 |
| qwen3 | 304 | 32 | `object_holdout` | `train_selected_gate` | 304 | 0.3793 | 0.3438 | 0.4231 | 0.6443 | 0.5266 |
| qwen3 | 304 | 32 | `prompt_holdout` | `blocker_field_combo` | 304 | 0.5172 | 0.4688 | 0.5769 | 0.7142 | 0.5588 |
| qwen3 | 304 | 32 | `prompt_holdout` | `compact_joint_gate_combo` | 304 | 0.6076 | 0.7500 | 0.5106 | 0.8327 | 0.6972 |
| qwen3 | 304 | 32 | `prompt_holdout` | `global_combo` | 304 | 0.5357 | 0.4688 | 0.6250 | 0.7178 | 0.6290 |
| qwen3 | 304 | 32 | `prompt_holdout` | `internal_strength_combo` | 304 | 0.6076 | 0.7500 | 0.5106 | 0.8327 | 0.7108 |
| qwen3 | 304 | 32 | `prompt_holdout` | `joint_gate_combo` | 304 | 0.6076 | 0.7500 | 0.5106 | 0.8327 | 0.6972 |
| qwen3 | 304 | 32 | `prompt_holdout` | `model_default_gate` | 304 | 0.5556 | 0.4688 | 0.6818 | 0.7215 | 0.6748 |
| qwen3 | 304 | 32 | `prompt_holdout` | `residual_projection_combo` | 304 | 0.5556 | 0.4688 | 0.6818 | 0.7215 | 0.6748 |
| qwen3 | 304 | 32 | `prompt_holdout` | `route_competition_combo` | 304 | 0.4304 | 0.5312 | 0.3617 | 0.7105 | 0.5434 |
| qwen3 | 304 | 32 | `prompt_holdout` | `train_selected_gate` | 304 | 0.5556 | 0.4688 | 0.6818 | 0.7215 | 0.6748 |
| glm4 | 304 | 0 | `in_sample` | `blocker_field_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `in_sample` | `compact_joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `in_sample` | `global_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `in_sample` | `internal_strength_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `in_sample` | `joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `in_sample` | `model_default_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `in_sample` | `residual_projection_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `in_sample` | `route_competition_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `in_sample` | `train_selected_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `object_holdout` | `blocker_field_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `object_holdout` | `compact_joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `object_holdout` | `global_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `object_holdout` | `internal_strength_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `object_holdout` | `joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `object_holdout` | `model_default_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `object_holdout` | `residual_projection_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `object_holdout` | `route_competition_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `object_holdout` | `train_selected_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `prompt_holdout` | `blocker_field_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `prompt_holdout` | `compact_joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `prompt_holdout` | `global_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `prompt_holdout` | `internal_strength_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `prompt_holdout` | `joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `prompt_holdout` | `model_default_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `prompt_holdout` | `residual_projection_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `prompt_holdout` | `route_competition_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 304 | 0 | `prompt_holdout` | `train_selected_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 304 | 3 | `in_sample` | `blocker_field_combo` | 304 | 0.6667 | 1.0000 | 0.5000 | 0.9950 | 0.5539 |
| deepseek7b | 304 | 3 | `in_sample` | `compact_joint_gate_combo` | 304 | 0.6667 | 1.0000 | 0.5000 | 0.9950 | 0.5539 |
| deepseek7b | 304 | 3 | `in_sample` | `global_combo` | 304 | 0.2222 | 1.0000 | 0.1250 | 0.9651 | 0.3954 |
| deepseek7b | 304 | 3 | `in_sample` | `internal_strength_combo` | 304 | 0.3333 | 0.3333 | 0.3333 | 0.6633 | 0.4422 |
| deepseek7b | 304 | 3 | `in_sample` | `joint_gate_combo` | 304 | 0.6667 | 1.0000 | 0.5000 | 0.9950 | 0.5539 |
| deepseek7b | 304 | 3 | `in_sample` | `model_default_gate` | 304 | 0.6667 | 1.0000 | 0.5000 | 0.9950 | 0.5539 |
| deepseek7b | 304 | 3 | `in_sample` | `residual_projection_combo` | 304 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | 304 | 3 | `in_sample` | `route_competition_combo` | 304 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | 304 | 3 | `in_sample` | `train_selected_gate` | 304 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | 304 | 3 | `object_holdout` | `blocker_field_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3317 |
| deepseek7b | 304 | 3 | `object_holdout` | `compact_joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4983 | 0.3311 |
| deepseek7b | 304 | 3 | `object_holdout` | `global_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4701 | 0.3214 |
| deepseek7b | 304 | 3 | `object_holdout` | `internal_strength_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4900 | 0.3283 |
| deepseek7b | 304 | 3 | `object_holdout` | `joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4983 | 0.3311 |
| deepseek7b | 304 | 3 | `object_holdout` | `model_default_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3317 |
| deepseek7b | 304 | 3 | `object_holdout` | `residual_projection_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3317 |
| deepseek7b | 304 | 3 | `object_holdout` | `route_competition_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4950 | 0.3300 |
| deepseek7b | 304 | 3 | `object_holdout` | `train_selected_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3317 |
| deepseek7b | 304 | 3 | `prompt_holdout` | `blocker_field_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4950 | 0.3300 |
| deepseek7b | 304 | 3 | `prompt_holdout` | `compact_joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4934 | 0.3295 |
| deepseek7b | 304 | 3 | `prompt_holdout` | `global_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4801 | 0.3249 |
| deepseek7b | 304 | 3 | `prompt_holdout` | `internal_strength_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4801 | 0.3249 |
| deepseek7b | 304 | 3 | `prompt_holdout` | `joint_gate_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4934 | 0.3295 |
| deepseek7b | 304 | 3 | `prompt_holdout` | `model_default_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4950 | 0.3300 |
| deepseek7b | 304 | 3 | `prompt_holdout` | `residual_projection_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3317 |
| deepseek7b | 304 | 3 | `prompt_holdout` | `route_competition_combo` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.4850 | 0.3266 |
| deepseek7b | 304 | 3 | `prompt_holdout` | `train_selected_gate` | 304 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3317 |

## Balanced Subset Summary

| model | split | predictor | n | strong F1 | recall | precision | balanced acc | macro F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `in_sample` | `blocker_field_combo` | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9790 |
| qwen3 | `in_sample` | `compact_joint_gate_combo` | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| qwen3 | `in_sample` | `global_combo` | 64 | 0.6909 | 0.5938 | 0.8261 | 0.7344 | 0.6290 |
| qwen3 | `in_sample` | `internal_strength_combo` | 64 | 0.8667 | 0.8125 | 0.9286 | 0.8750 | 0.8298 |
| qwen3 | `in_sample` | `joint_gate_combo` | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| qwen3 | `in_sample` | `model_default_gate` | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9577 |
| qwen3 | `in_sample` | `residual_projection_combo` | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9577 |
| qwen3 | `in_sample` | `route_competition_combo` | 64 | 0.9275 | 1.0000 | 0.8649 | 0.9219 | 0.9031 |
| qwen3 | `in_sample` | `train_selected_gate` | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| qwen3 | `object_holdout` | `blocker_field_combo` | 64 | 0.4681 | 0.3438 | 0.7333 | 0.6094 | 0.4356 |
| qwen3 | `object_holdout` | `compact_joint_gate_combo` | 64 | 0.6000 | 0.4688 | 0.8333 | 0.6875 | 0.4701 |
| qwen3 | `object_holdout` | `global_combo` | 64 | 0.5660 | 0.4688 | 0.7143 | 0.6406 | 0.5070 |
| qwen3 | `object_holdout` | `internal_strength_combo` | 64 | 0.6000 | 0.4688 | 0.8333 | 0.6875 | 0.5145 |
| qwen3 | `object_holdout` | `joint_gate_combo` | 64 | 0.5417 | 0.4062 | 0.8125 | 0.6562 | 0.4798 |
| qwen3 | `object_holdout` | `model_default_gate` | 64 | 0.5532 | 0.4062 | 0.8667 | 0.6719 | 0.5228 |
| qwen3 | `object_holdout` | `residual_projection_combo` | 64 | 0.5532 | 0.4062 | 0.8667 | 0.6719 | 0.5228 |
| qwen3 | `object_holdout` | `route_competition_combo` | 64 | 0.5306 | 0.4062 | 0.7647 | 0.6406 | 0.5030 |
| qwen3 | `object_holdout` | `train_selected_gate` | 64 | 0.4889 | 0.3438 | 0.8462 | 0.6406 | 0.4791 |
| qwen3 | `prompt_holdout` | `blocker_field_combo` | 64 | 0.6000 | 0.4688 | 0.8333 | 0.6875 | 0.4979 |
| qwen3 | `prompt_holdout` | `compact_joint_gate_combo` | 64 | 0.8136 | 0.7500 | 0.8889 | 0.8281 | 0.7783 |
| qwen3 | `prompt_holdout` | `global_combo` | 64 | 0.6000 | 0.4688 | 0.8333 | 0.6875 | 0.5812 |
| qwen3 | `prompt_holdout` | `internal_strength_combo` | 64 | 0.8000 | 0.7500 | 0.8571 | 0.8125 | 0.7654 |
| qwen3 | `prompt_holdout` | `joint_gate_combo` | 64 | 0.8136 | 0.7500 | 0.8889 | 0.8281 | 0.7783 |
| qwen3 | `prompt_holdout` | `model_default_gate` | 64 | 0.6250 | 0.4688 | 0.9375 | 0.7188 | 0.6440 |
| qwen3 | `prompt_holdout` | `residual_projection_combo` | 64 | 0.6250 | 0.4688 | 0.9375 | 0.7188 | 0.6440 |
| qwen3 | `prompt_holdout` | `route_competition_combo` | 64 | 0.6182 | 0.5312 | 0.7391 | 0.6719 | 0.5405 |
| qwen3 | `prompt_holdout` | `train_selected_gate` | 64 | 0.6250 | 0.4688 | 0.9375 | 0.7188 | 0.6440 |
| deepseek7b | `in_sample` | `blocker_field_combo` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `compact_joint_gate_combo` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `global_combo` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `internal_strength_combo` | 6 | 0.5000 | 0.3333 | 1.0000 | 0.6667 | 0.4167 |
| deepseek7b | `in_sample` | `joint_gate_combo` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `model_default_gate` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `residual_projection_combo` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `route_competition_combo` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `train_selected_gate` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `object_holdout` | `blocker_field_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `object_holdout` | `compact_joint_gate_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `object_holdout` | `global_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `object_holdout` | `internal_strength_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `object_holdout` | `joint_gate_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `object_holdout` | `model_default_gate` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `object_holdout` | `residual_projection_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `object_holdout` | `route_competition_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `object_holdout` | `train_selected_gate` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `prompt_holdout` | `blocker_field_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `prompt_holdout` | `compact_joint_gate_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `prompt_holdout` | `global_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `prompt_holdout` | `internal_strength_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `prompt_holdout` | `joint_gate_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `prompt_holdout` | `model_default_gate` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `prompt_holdout` | `residual_projection_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `prompt_holdout` | `route_competition_combo` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
| deepseek7b | `prompt_holdout` | `train_selected_gate` | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2222 |
