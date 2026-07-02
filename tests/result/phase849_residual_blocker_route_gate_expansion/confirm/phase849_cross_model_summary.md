# Phase 849 Residual-stream / Blocker-field Route Gate Expansion (confirm)

- Source: Phase 845 residual rows plus fresh natural residual/logit/activation captures.
- Method: compare MLP-only, residual-projection, blocker-field, route-competition, and joint gate predictors.
- Boundary: internal route gate expansion probe; not geometry closure.

## Model Summary

| model | feature rows | prompts | split | predictor | n | MAE | sign acc | strong F1 | MAE gain vs global |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| qwen3 | 570 | 15 | `in_sample` | `blocker_field_combo` | 570 | 0.0612 | 0.7935 | 0.8571 | 0.1000 |
| qwen3 | 570 | 15 | `in_sample` | `compact_joint_gate_combo` | 570 | 0.0543 | 0.8407 | 0.7586 | 0.1070 |
| qwen3 | 570 | 15 | `in_sample` | `global_combo` | 570 | 0.1612 | 0.5221 | 0.4688 | NA |
| qwen3 | 570 | 15 | `in_sample` | `internal_strength_combo` | 570 | 0.1113 | 0.7243 | 0.5926 | 0.0500 |
| qwen3 | 570 | 15 | `in_sample` | `joint_gate_combo` | 570 | 0.0303 | 0.9286 | 0.8571 | 0.1310 |
| qwen3 | 570 | 15 | `in_sample` | `residual_projection_combo` | 570 | 0.0425 | 0.8242 | 0.8070 | 0.1187 |
| qwen3 | 570 | 15 | `in_sample` | `route_competition_combo` | 570 | 0.1076 | 0.7168 | 0.6000 | 0.0537 |
| qwen3 | 570 | 15 | `object_holdout` | `blocker_field_combo` | 570 | 0.1647 | 0.4936 | 0.4333 | 0.0166 |
| qwen3 | 570 | 15 | `object_holdout` | `compact_joint_gate_combo` | 570 | 0.1396 | 0.5412 | 0.5000 | 0.0417 |
| qwen3 | 570 | 15 | `object_holdout` | `global_combo` | 570 | 0.1813 | 0.4491 | 0.4688 | NA |
| qwen3 | 570 | 15 | `object_holdout` | `internal_strength_combo` | 570 | 0.1761 | 0.5344 | 0.4688 | 0.0052 |
| qwen3 | 570 | 15 | `object_holdout` | `joint_gate_combo` | 570 | 0.1377 | 0.5403 | 0.5000 | 0.0436 |
| qwen3 | 570 | 15 | `object_holdout` | `residual_projection_combo` | 570 | 0.1462 | 0.4919 | 0.5556 | 0.0351 |
| qwen3 | 570 | 15 | `object_holdout` | `route_competition_combo` | 570 | 0.2116 | 0.4120 | 0.3667 | -0.0303 |
| qwen3 | 570 | 15 | `prompt_holdout` | `blocker_field_combo` | 570 | 0.1712 | 0.4810 | 0.4333 | 0.0090 |
| qwen3 | 570 | 15 | `prompt_holdout` | `compact_joint_gate_combo` | 570 | 0.1732 | 0.5112 | 0.5057 | 0.0070 |
| qwen3 | 570 | 15 | `prompt_holdout` | `global_combo` | 570 | 0.1802 | 0.4886 | 0.4688 | NA |
| qwen3 | 570 | 15 | `prompt_holdout` | `internal_strength_combo` | 570 | 0.1735 | 0.5127 | 0.4507 | 0.0068 |
| qwen3 | 570 | 15 | `prompt_holdout` | `joint_gate_combo` | 570 | 0.1733 | 0.5178 | 0.5057 | 0.0069 |
| qwen3 | 570 | 15 | `prompt_holdout` | `residual_projection_combo` | 570 | 0.1420 | 0.5383 | 0.5429 | 0.0382 |
| qwen3 | 570 | 15 | `prompt_holdout` | `route_competition_combo` | 570 | 0.2216 | 0.4830 | 0.0769 | -0.0413 |
| glm4 | 570 | 15 | `in_sample` | `blocker_field_combo` | 570 | 0.0210 | 0.7853 | 0.0000 | 0.0196 |
| glm4 | 570 | 15 | `in_sample` | `compact_joint_gate_combo` | 570 | 0.0181 | 0.8126 | 0.0000 | 0.0224 |
| glm4 | 570 | 15 | `in_sample` | `global_combo` | 570 | 0.0406 | 0.5702 | 0.0000 | NA |
| glm4 | 570 | 15 | `in_sample` | `internal_strength_combo` | 570 | 0.0315 | 0.6890 | 0.0000 | 0.0091 |
| glm4 | 570 | 15 | `in_sample` | `joint_gate_combo` | 570 | 0.0095 | 0.8889 | 0.0000 | 0.0311 |
| glm4 | 570 | 15 | `in_sample` | `residual_projection_combo` | 570 | 0.0315 | 0.6915 | 0.0000 | 0.0091 |
| glm4 | 570 | 15 | `in_sample` | `route_competition_combo` | 570 | 0.0303 | 0.7057 | 0.0000 | 0.0103 |
| glm4 | 570 | 15 | `object_holdout` | `blocker_field_combo` | 570 | 0.0534 | 0.4665 | 0.0000 | -0.0087 |
| glm4 | 570 | 15 | `object_holdout` | `compact_joint_gate_combo` | 570 | 0.0546 | 0.4806 | 0.0000 | -0.0100 |
| glm4 | 570 | 15 | `object_holdout` | `global_combo` | 570 | 0.0447 | 0.4474 | 0.0000 | NA |
| glm4 | 570 | 15 | `object_holdout` | `internal_strength_combo` | 570 | 0.0475 | 0.4903 | 0.0000 | -0.0028 |
| glm4 | 570 | 15 | `object_holdout` | `joint_gate_combo` | 570 | 0.0550 | 0.4754 | 0.0000 | -0.0104 |
| glm4 | 570 | 15 | `object_holdout` | `residual_projection_combo` | 570 | 0.0551 | 0.4394 | 0.0000 | -0.0104 |
| glm4 | 570 | 15 | `object_holdout` | `route_competition_combo` | 570 | 0.0534 | 0.4507 | 0.0000 | -0.0087 |
| glm4 | 570 | 15 | `prompt_holdout` | `blocker_field_combo` | 570 | 0.0500 | 0.4842 | 0.0000 | -0.0062 |
| glm4 | 570 | 15 | `prompt_holdout` | `compact_joint_gate_combo` | 570 | 0.0493 | 0.5088 | 0.0000 | -0.0055 |
| glm4 | 570 | 15 | `prompt_holdout` | `global_combo` | 570 | 0.0438 | 0.5175 | 0.0000 | NA |
| glm4 | 570 | 15 | `prompt_holdout` | `internal_strength_combo` | 570 | 0.0454 | 0.5431 | 0.0000 | -0.0016 |
| glm4 | 570 | 15 | `prompt_holdout` | `joint_gate_combo` | 570 | 0.0504 | 0.5070 | 0.0000 | -0.0067 |
| glm4 | 570 | 15 | `prompt_holdout` | `residual_projection_combo` | 570 | 0.0506 | 0.4560 | 0.0000 | -0.0069 |
| glm4 | 570 | 15 | `prompt_holdout` | `route_competition_combo` | 570 | 0.0560 | 0.4903 | 0.0000 | -0.0122 |
| deepseek7b | 570 | 15 | `in_sample` | `blocker_field_combo` | 570 | 0.0135 | 0.8242 | 1.0000 | 0.0268 |
| deepseek7b | 570 | 15 | `in_sample` | `compact_joint_gate_combo` | 570 | 0.0176 | 0.8257 | 0.8000 | 0.0228 |
| deepseek7b | 570 | 15 | `in_sample` | `global_combo` | 570 | 0.0403 | 0.4509 | 0.0000 | NA |
| deepseek7b | 570 | 15 | `in_sample` | `internal_strength_combo` | 570 | 0.0394 | 0.5009 | 0.0000 | 0.0010 |
| deepseek7b | 570 | 15 | `in_sample` | `joint_gate_combo` | 570 | 0.0143 | 0.8628 | 0.8000 | 0.0260 |
| deepseek7b | 570 | 15 | `in_sample` | `residual_projection_combo` | 570 | 0.0255 | 0.6917 | 0.0000 | 0.0149 |
| deepseek7b | 570 | 15 | `in_sample` | `route_competition_combo` | 570 | 0.0210 | 0.6728 | 1.0000 | 0.0193 |
| deepseek7b | 570 | 15 | `object_holdout` | `blocker_field_combo` | 570 | 0.0423 | 0.5785 | 0.0000 | 0.0009 |
| deepseek7b | 570 | 15 | `object_holdout` | `compact_joint_gate_combo` | 570 | 0.0543 | 0.5028 | 0.0000 | -0.0111 |
| deepseek7b | 570 | 15 | `object_holdout` | `global_combo` | 570 | 0.0432 | 0.4482 | 0.0000 | NA |
| deepseek7b | 570 | 15 | `object_holdout` | `internal_strength_combo` | 570 | 0.0473 | 0.4583 | 0.0000 | -0.0042 |
| deepseek7b | 570 | 15 | `object_holdout` | `joint_gate_combo` | 570 | 0.0559 | 0.4917 | 0.0000 | -0.0127 |
| deepseek7b | 570 | 15 | `object_holdout` | `residual_projection_combo` | 570 | 0.0563 | 0.3902 | 0.0000 | -0.0131 |
| deepseek7b | 570 | 15 | `object_holdout` | `route_competition_combo` | 570 | 0.0543 | 0.4356 | 0.0000 | -0.0111 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `blocker_field_combo` | 570 | 0.0423 | 0.5655 | 0.0000 | 0.0025 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `compact_joint_gate_combo` | 570 | 0.0477 | 0.4951 | 0.0000 | -0.0029 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `global_combo` | 570 | 0.0448 | 0.4208 | 0.0000 | NA |
| deepseek7b | 570 | 15 | `prompt_holdout` | `internal_strength_combo` | 570 | 0.0457 | 0.4557 | 0.0000 | -0.0009 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `joint_gate_combo` | 570 | 0.0482 | 0.5000 | 0.0000 | -0.0033 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `residual_projection_combo` | 570 | 0.0501 | 0.3931 | 0.0000 | -0.0053 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `route_competition_combo` | 570 | 0.0518 | 0.5328 | 0.0000 | -0.0070 |

## Feature Summary

| model | top1 roles | target rank buckets | residual classes | mean target-blocker logit | mean blocker pressure | mean residual target-blocker |
|---|---|---|---|---:|---:|---:|
| qwen3 | `{"blocker": 532, "other": 38}` | `{"tail": 532, "top100": 38}` | `{"additive": 536, "synergy": 19, "antagonistic": 15}` | -12.7458 | 12.7458 | -6.1895 |
| glm4 | `{"blocker": 570}` | `{"tail": 570}` | `{"additive": 570}` | -8.2802 | 8.2802 | 5.9103 |
| deepseek7b | `{"blocker": 532, "other": 38}` | `{"tail": 570}` | `{"additive": 567, "synergy": 3}` | -9.7000 | 9.7000 | 133.5860 |
