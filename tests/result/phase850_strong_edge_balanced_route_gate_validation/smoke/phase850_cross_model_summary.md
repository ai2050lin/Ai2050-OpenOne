# Phase 850 Strong-edge Balanced Route Gate Validation (smoke)

- Source: Phase 849 feature rows.
- Method: calibrated strong-edge class predictors with raw and balanced metrics.
- Boundary: strong-edge validation, not closure.

## Raw Strong-edge Summary

| model | rows | strong rows | split | predictor | n | strong F1 | recall | precision | balanced acc | macro F1 |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 12 | 1 | `in_sample` | `blocker_field_combo` | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | 12 | 1 | `in_sample` | `compact_joint_gate_combo` | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | 12 | 1 | `in_sample` | `global_combo` | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | 12 | 1 | `in_sample` | `internal_strength_combo` | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | 12 | 1 | `in_sample` | `joint_gate_combo` | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | 12 | 1 | `in_sample` | `model_default_gate` | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | 12 | 1 | `in_sample` | `residual_projection_combo` | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | 12 | 1 | `in_sample` | `route_competition_combo` | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | 12 | 1 | `in_sample` | `train_selected_gate` | 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| glm4 | 12 | 0 | `in_sample` | `blocker_field_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 12 | 0 | `in_sample` | `compact_joint_gate_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 12 | 0 | `in_sample` | `global_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 12 | 0 | `in_sample` | `internal_strength_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 12 | 0 | `in_sample` | `joint_gate_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 12 | 0 | `in_sample` | `model_default_gate` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 12 | 0 | `in_sample` | `residual_projection_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 12 | 0 | `in_sample` | `route_competition_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| glm4 | 12 | 0 | `in_sample` | `train_selected_gate` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 12 | 0 | `in_sample` | `blocker_field_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 12 | 0 | `in_sample` | `compact_joint_gate_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 12 | 0 | `in_sample` | `global_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 12 | 0 | `in_sample` | `internal_strength_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 12 | 0 | `in_sample` | `joint_gate_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 12 | 0 | `in_sample` | `model_default_gate` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 12 | 0 | `in_sample` | `residual_projection_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 12 | 0 | `in_sample` | `route_competition_combo` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |
| deepseek7b | 12 | 0 | `in_sample` | `train_selected_gate` | 12 | 0.0000 | 0.0000 | 0.0000 | NA | 0.3333 |

## Balanced Subset Summary

| model | split | predictor | n | strong F1 | recall | precision | balanced acc | macro F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `in_sample` | `blocker_field_combo` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | `in_sample` | `compact_joint_gate_combo` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | `in_sample` | `global_combo` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | `in_sample` | `internal_strength_combo` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | `in_sample` | `joint_gate_combo` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | `in_sample` | `model_default_gate` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | `in_sample` | `residual_projection_combo` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | `in_sample` | `route_competition_combo` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| qwen3 | `in_sample` | `train_selected_gate` | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
