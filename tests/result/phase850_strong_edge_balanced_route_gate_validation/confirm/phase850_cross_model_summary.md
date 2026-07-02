# Phase 850 Strong-edge Balanced Route Gate Validation (confirm)

- Source: Phase 849 feature rows.
- Method: calibrated strong-edge class predictors with raw and balanced metrics.
- Boundary: strong-edge validation, not closure.

## Raw Strong-edge Summary

| model | rows | strong rows | split | predictor | n | strong F1 | recall | precision | balanced acc | macro F1 |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 570 | 34 | `in_sample` | `blocker_field_combo` | 570 | 0.8657 | 0.8529 | 0.8788 | 0.9227 | 0.8683 |
| qwen3 | 570 | 34 | `in_sample` | `compact_joint_gate_combo` | 570 | 0.8333 | 0.8824 | 0.7895 | 0.9337 | 0.8856 |
| qwen3 | 570 | 34 | `in_sample` | `global_combo` | 570 | 0.4688 | 0.4412 | 0.5000 | 0.7066 | 0.5006 |
| qwen3 | 570 | 34 | `in_sample` | `internal_strength_combo` | 570 | 0.6452 | 0.8824 | 0.5085 | 0.9141 | 0.7208 |
| qwen3 | 570 | 34 | `in_sample` | `joint_gate_combo` | 570 | 0.8831 | 1.0000 | 0.7907 | 0.9916 | 0.9214 |
| qwen3 | 570 | 34 | `in_sample` | `model_default_gate` | 570 | 0.8923 | 0.8529 | 0.9355 | 0.9246 | 0.8906 |
| qwen3 | 570 | 34 | `in_sample` | `residual_projection_combo` | 570 | 0.8923 | 0.8529 | 0.9355 | 0.9246 | 0.8906 |
| qwen3 | 570 | 34 | `in_sample` | `route_competition_combo` | 570 | 0.6757 | 0.7353 | 0.6250 | 0.8537 | 0.7634 |
| qwen3 | 570 | 34 | `in_sample` | `train_selected_gate` | 570 | 0.8923 | 0.8529 | 0.9355 | 0.9246 | 0.8906 |
| qwen3 | 570 | 34 | `object_holdout` | `blocker_field_combo` | 570 | 0.4000 | 0.3824 | 0.4194 | 0.6744 | 0.4933 |
| qwen3 | 570 | 34 | `object_holdout` | `compact_joint_gate_combo` | 570 | 0.4211 | 0.3529 | 0.5217 | 0.6662 | 0.5613 |
| qwen3 | 570 | 34 | `object_holdout` | `global_combo` | 570 | 0.3947 | 0.4412 | 0.3571 | 0.6954 | 0.4967 |
| qwen3 | 570 | 34 | `object_holdout` | `internal_strength_combo` | 570 | 0.4051 | 0.4706 | 0.3556 | 0.7082 | 0.5291 |
| qwen3 | 570 | 34 | `object_holdout` | `joint_gate_combo` | 570 | 0.4590 | 0.4118 | 0.5185 | 0.6938 | 0.5612 |
| qwen3 | 570 | 34 | `object_holdout` | `model_default_gate` | 570 | 0.5312 | 0.5000 | 0.5667 | 0.7379 | 0.5145 |
| qwen3 | 570 | 34 | `object_holdout` | `residual_projection_combo` | 570 | 0.5312 | 0.5000 | 0.5667 | 0.7379 | 0.5145 |
| qwen3 | 570 | 34 | `object_holdout` | `route_competition_combo` | 570 | 0.2632 | 0.2941 | 0.2381 | 0.6172 | 0.4561 |
| qwen3 | 570 | 34 | `object_holdout` | `train_selected_gate` | 570 | 0.5000 | 0.4706 | 0.5333 | 0.7222 | 0.5254 |
| qwen3 | 570 | 34 | `prompt_holdout` | `blocker_field_combo` | 570 | 0.4333 | 0.3824 | 0.5000 | 0.6790 | 0.4752 |
| qwen3 | 570 | 34 | `prompt_holdout` | `compact_joint_gate_combo` | 570 | 0.4783 | 0.6471 | 0.3793 | 0.7899 | 0.5655 |
| qwen3 | 570 | 34 | `prompt_holdout` | `global_combo` | 570 | 0.4054 | 0.4412 | 0.3750 | 0.6973 | 0.4974 |
| qwen3 | 570 | 34 | `prompt_holdout` | `internal_strength_combo` | 570 | 0.4255 | 0.5882 | 0.3333 | 0.7568 | 0.5806 |
| qwen3 | 570 | 34 | `prompt_holdout` | `joint_gate_combo` | 570 | 0.4490 | 0.6471 | 0.3438 | 0.7844 | 0.5580 |
| qwen3 | 570 | 34 | `prompt_holdout` | `model_default_gate` | 570 | 0.4839 | 0.4412 | 0.5357 | 0.7085 | 0.6088 |
| qwen3 | 570 | 34 | `prompt_holdout` | `residual_projection_combo` | 570 | 0.4839 | 0.4412 | 0.5357 | 0.7085 | 0.6088 |
| qwen3 | 570 | 34 | `prompt_holdout` | `route_competition_combo` | 570 | 0.0317 | 0.0294 | 0.0345 | 0.4886 | 0.3145 |
| qwen3 | 570 | 34 | `prompt_holdout` | `train_selected_gate` | 570 | 0.4051 | 0.4706 | 0.3556 | 0.7082 | 0.5164 |
| glm4 | 570 | 0 | `in_sample` | `blocker_field_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `in_sample` | `compact_joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `in_sample` | `global_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `in_sample` | `internal_strength_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `in_sample` | `joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `in_sample` | `model_default_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `in_sample` | `residual_projection_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `in_sample` | `route_competition_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `in_sample` | `train_selected_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `object_holdout` | `blocker_field_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `object_holdout` | `compact_joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `object_holdout` | `global_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `object_holdout` | `internal_strength_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `object_holdout` | `joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `object_holdout` | `model_default_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `object_holdout` | `residual_projection_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `object_holdout` | `route_competition_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `object_holdout` | `train_selected_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `prompt_holdout` | `blocker_field_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `prompt_holdout` | `compact_joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `prompt_holdout` | `global_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `prompt_holdout` | `internal_strength_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `prompt_holdout` | `joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `prompt_holdout` | `model_default_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `prompt_holdout` | `residual_projection_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `prompt_holdout` | `route_competition_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 570 | 0 | `prompt_holdout` | `train_selected_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 570 | 3 | `in_sample` | `blocker_field_combo` | 570 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | 570 | 3 | `in_sample` | `compact_joint_gate_combo` | 570 | 0.8571 | 1.0000 | 0.7500 | 0.9991 | 0.6188 |
| deepseek7b | 570 | 3 | `in_sample` | `global_combo` | 570 | 0.1250 | 1.0000 | 0.0667 | 0.9630 | 0.3622 |
| deepseek7b | 570 | 3 | `in_sample` | `internal_strength_combo` | 570 | 0.2353 | 0.6667 | 0.1429 | 0.8228 | 0.4079 |
| deepseek7b | 570 | 3 | `in_sample` | `joint_gate_combo` | 570 | 0.8571 | 1.0000 | 0.7500 | 0.9991 | 0.6188 |
| deepseek7b | 570 | 3 | `in_sample` | `model_default_gate` | 570 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | 570 | 3 | `in_sample` | `residual_projection_combo` | 570 | 0.6667 | 1.0000 | 0.5000 | 0.9974 | 0.5547 |
| deepseek7b | 570 | 3 | `in_sample` | `route_competition_combo` | 570 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | 570 | 3 | `in_sample` | `train_selected_gate` | 570 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | 570 | 3 | `object_holdout` | `blocker_field_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3325 |
| deepseek7b | 570 | 3 | `object_holdout` | `compact_joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4991 | 0.3322 |
| deepseek7b | 570 | 3 | `object_holdout` | `global_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4683 | 0.3215 |
| deepseek7b | 570 | 3 | `object_holdout` | `internal_strength_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4956 | 0.3310 |
| deepseek7b | 570 | 3 | `object_holdout` | `joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4991 | 0.3322 |
| deepseek7b | 570 | 3 | `object_holdout` | `model_default_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3325 |
| deepseek7b | 570 | 3 | `object_holdout` | `residual_projection_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3325 |
| deepseek7b | 570 | 3 | `object_holdout` | `route_competition_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4947 | 0.3307 |
| deepseek7b | 570 | 3 | `object_holdout` | `train_selected_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3325 |
| deepseek7b | 570 | 3 | `prompt_holdout` | `blocker_field_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3325 |
| deepseek7b | 570 | 3 | `prompt_holdout` | `compact_joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4991 | 0.3322 |
| deepseek7b | 570 | 3 | `prompt_holdout` | `global_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4735 | 0.3234 |
| deepseek7b | 570 | 3 | `prompt_holdout` | `internal_strength_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4965 | 0.3313 |
| deepseek7b | 570 | 3 | `prompt_holdout` | `joint_gate_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4991 | 0.3322 |
| deepseek7b | 570 | 3 | `prompt_holdout` | `model_default_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3325 |
| deepseek7b | 570 | 3 | `prompt_holdout` | `residual_projection_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4974 | 0.3316 |
| deepseek7b | 570 | 3 | `prompt_holdout` | `route_competition_combo` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4974 | 0.3316 |
| deepseek7b | 570 | 3 | `prompt_holdout` | `train_selected_gate` | 570 | 0.0000 | 0.0000 | 0.0000 | 0.4974 | 0.3316 |

## Balanced Subset Summary

| model | split | predictor | n | strong F1 | recall | precision | balanced acc | macro F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `in_sample` | `blocker_field_combo` | 68 | 0.9206 | 0.8529 | 1.0000 | 0.9265 | 0.8824 |
| qwen3 | `in_sample` | `compact_joint_gate_combo` | 68 | 0.9231 | 0.8824 | 0.9677 | 0.9265 | 0.9257 |
| qwen3 | `in_sample` | `global_combo` | 68 | 0.5882 | 0.4412 | 0.8824 | 0.6912 | 0.5010 |
| qwen3 | `in_sample` | `internal_strength_combo` | 68 | 0.8824 | 0.8824 | 0.8824 | 0.8824 | 0.8064 |
| qwen3 | `in_sample` | `joint_gate_combo` | 68 | 0.9855 | 1.0000 | 0.9714 | 0.9853 | 0.9865 |
| qwen3 | `in_sample` | `model_default_gate` | 68 | 0.9206 | 0.8529 | 1.0000 | 0.9265 | 0.8846 |
| qwen3 | `in_sample` | `residual_projection_combo` | 68 | 0.9206 | 0.8529 | 1.0000 | 0.9265 | 0.8846 |
| qwen3 | `in_sample` | `route_competition_combo` | 68 | 0.8197 | 0.7353 | 0.9259 | 0.8382 | 0.8103 |
| qwen3 | `in_sample` | `train_selected_gate` | 68 | 0.9206 | 0.8529 | 1.0000 | 0.9265 | 0.8846 |
| qwen3 | `object_holdout` | `blocker_field_combo` | 68 | 0.5417 | 0.3824 | 0.9286 | 0.6765 | 0.4633 |
| qwen3 | `object_holdout` | `compact_joint_gate_combo` | 68 | 0.4898 | 0.3529 | 0.8000 | 0.6324 | 0.4756 |
| qwen3 | `object_holdout` | `global_combo` | 68 | 0.5882 | 0.4412 | 0.8824 | 0.6912 | 0.5090 |
| qwen3 | `object_holdout` | `internal_strength_combo` | 68 | 0.5926 | 0.4706 | 0.8000 | 0.6765 | 0.5020 |
| qwen3 | `object_holdout` | `joint_gate_combo` | 68 | 0.5490 | 0.4118 | 0.8235 | 0.6618 | 0.4812 |
| qwen3 | `object_holdout` | `model_default_gate` | 68 | 0.6296 | 0.5000 | 0.8500 | 0.7059 | 0.4671 |
| qwen3 | `object_holdout` | `residual_projection_combo` | 68 | 0.6296 | 0.5000 | 0.8500 | 0.7059 | 0.4671 |
| qwen3 | `object_holdout` | `route_competition_combo` | 68 | 0.4167 | 0.2941 | 0.7143 | 0.5882 | 0.4248 |
| qwen3 | `object_holdout` | `train_selected_gate` | 68 | 0.6038 | 0.4706 | 0.8421 | 0.6912 | 0.4789 |
| qwen3 | `prompt_holdout` | `blocker_field_combo` | 68 | 0.5417 | 0.3824 | 0.9286 | 0.6765 | 0.4475 |
| qwen3 | `prompt_holdout` | `compact_joint_gate_combo` | 68 | 0.7213 | 0.6471 | 0.8148 | 0.7500 | 0.5918 |
| qwen3 | `prompt_holdout` | `global_combo` | 68 | 0.5882 | 0.4412 | 0.8824 | 0.6912 | 0.5010 |
| qwen3 | `prompt_holdout` | `internal_strength_combo` | 68 | 0.6897 | 0.5882 | 0.8333 | 0.7353 | 0.6626 |
| qwen3 | `prompt_holdout` | `joint_gate_combo` | 68 | 0.7097 | 0.6471 | 0.7857 | 0.7353 | 0.5828 |
| qwen3 | `prompt_holdout` | `model_default_gate` | 68 | 0.6122 | 0.4412 | 1.0000 | 0.7206 | 0.5894 |
| qwen3 | `prompt_holdout` | `residual_projection_combo` | 68 | 0.6122 | 0.4412 | 1.0000 | 0.7206 | 0.5894 |
| qwen3 | `prompt_holdout` | `route_competition_combo` | 68 | 0.0571 | 0.0294 | 1.0000 | 0.5147 | 0.2244 |
| qwen3 | `prompt_holdout` | `train_selected_gate` | 68 | 0.6038 | 0.4706 | 0.8421 | 0.6912 | 0.5195 |
| deepseek7b | `in_sample` | `blocker_field_combo` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `compact_joint_gate_combo` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `global_combo` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| deepseek7b | `in_sample` | `internal_strength_combo` | 6 | 0.8000 | 0.6667 | 1.0000 | 0.8333 | 0.5524 |
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
