# Phase 849 Residual-stream / Blocker-field Route Gate Expansion (smoke)

- Source: Phase 845 residual rows plus fresh natural residual/logit/activation captures.
- Method: compare MLP-only, residual-projection, blocker-field, route-competition, and joint gate predictors.
- Boundary: internal route gate expansion probe; not geometry closure.

## Model Summary

| model | feature rows | prompts | split | predictor | n | MAE | sign acc | strong F1 | MAE gain vs global |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| qwen3 | 12 | 1 | `in_sample` | `blocker_field_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | 12 | 1 | `in_sample` | `compact_joint_gate_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | 12 | 1 | `in_sample` | `global_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | NA |
| qwen3 | 12 | 1 | `in_sample` | `internal_strength_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | 12 | 1 | `in_sample` | `joint_gate_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | 12 | 1 | `in_sample` | `residual_projection_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | 12 | 1 | `in_sample` | `route_competition_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `blocker_field_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `compact_joint_gate_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `global_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | NA |
| glm4 | 12 | 1 | `in_sample` | `internal_strength_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `joint_gate_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `residual_projection_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `route_competition_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `blocker_field_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `compact_joint_gate_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `global_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | NA |
| deepseek7b | 12 | 1 | `in_sample` | `internal_strength_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `joint_gate_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `residual_projection_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `route_competition_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

## Feature Summary

| model | top1 roles | target rank buckets | residual classes | mean target-blocker logit | mean blocker pressure | mean residual target-blocker |
|---|---|---|---|---:|---:|---:|
| qwen3 | `{"blocker": 12}` | `{"tail": 12}` | `{"additive": 11, "synergy": 1}` | -14.2500 | 14.2500 | -10.4958 |
| glm4 | `{"blocker": 12}` | `{"tail": 12}` | `{"additive": 12}` | -7.4375 | 7.4375 | 3.6206 |
| deepseek7b | `{"blocker": 12}` | `{"tail": 12}` | `{"additive": 12}` | -7.5000 | 7.5000 | 355.1385 |
