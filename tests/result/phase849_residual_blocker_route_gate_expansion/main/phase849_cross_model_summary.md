# Phase 849 Residual-stream / Blocker-field Route Gate Expansion (main)

- Source: Phase 845 residual rows plus fresh natural residual/logit/activation captures.
- Method: compare MLP-only, residual-projection, blocker-field, route-competition, and joint gate predictors.
- Boundary: internal route gate expansion probe; not geometry closure.

## Model Summary

| model | feature rows | prompts | split | predictor | n | MAE | sign acc | strong F1 | MAE gain vs global |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| qwen3 | 304 | 8 | `in_sample` | `blocker_field_combo` | 304 | 0.0323 | 0.9167 | 0.9000 | 0.1477 |
| qwen3 | 304 | 8 | `in_sample` | `compact_joint_gate_combo` | 304 | 0.0319 | 0.9524 | 0.9677 | 0.1480 |
| qwen3 | 304 | 8 | `in_sample` | `global_combo` | 304 | 0.1799 | 0.5921 | 0.5417 | NA |
| qwen3 | 304 | 8 | `in_sample` | `internal_strength_combo` | 304 | 0.1144 | 0.7621 | 0.7119 | 0.0655 |
| qwen3 | 304 | 8 | `in_sample` | `joint_gate_combo` | 304 | 0.0000 | 1.0000 | 1.0000 | 0.1799 |
| qwen3 | 304 | 8 | `in_sample` | `residual_projection_combo` | 304 | 0.0335 | 0.9064 | 0.8772 | 0.1464 |
| qwen3 | 304 | 8 | `in_sample` | `route_competition_combo` | 304 | 0.1096 | 0.7807 | 0.6071 | 0.0704 |
| qwen3 | 304 | 8 | `object_holdout` | `blocker_field_combo` | 304 | 0.2677 | 0.4796 | 0.4151 | -0.0375 |
| qwen3 | 304 | 8 | `object_holdout` | `compact_joint_gate_combo` | 304 | 0.2582 | 0.4982 | 0.4561 | -0.0279 |
| qwen3 | 304 | 8 | `object_holdout` | `global_combo` | 304 | 0.2302 | 0.5050 | 0.5417 | NA |
| qwen3 | 304 | 8 | `object_holdout` | `internal_strength_combo` | 304 | 0.2232 | 0.5544 | 0.5556 | 0.0070 |
| qwen3 | 304 | 8 | `object_holdout` | `joint_gate_combo` | 304 | 0.2582 | 0.4982 | 0.4561 | -0.0279 |
| qwen3 | 304 | 8 | `object_holdout` | `residual_projection_combo` | 304 | 0.2569 | 0.4629 | 0.4815 | -0.0267 |
| qwen3 | 304 | 8 | `object_holdout` | `route_competition_combo` | 304 | 0.2407 | 0.4483 | 0.5000 | -0.0105 |
| qwen3 | 304 | 8 | `prompt_holdout` | `blocker_field_combo` | 304 | 0.1981 | 0.5458 | 0.5769 | -0.0021 |
| qwen3 | 304 | 8 | `prompt_holdout` | `compact_joint_gate_combo` | 304 | 0.2076 | 0.5780 | 0.5714 | -0.0116 |
| qwen3 | 304 | 8 | `prompt_holdout` | `global_combo` | 304 | 0.1960 | 0.5452 | 0.5417 | NA |
| qwen3 | 304 | 8 | `prompt_holdout` | `internal_strength_combo` | 304 | 0.1767 | 0.5903 | 0.5714 | 0.0193 |
| qwen3 | 304 | 8 | `prompt_holdout` | `joint_gate_combo` | 304 | 0.2076 | 0.5780 | 0.5714 | -0.0116 |
| qwen3 | 304 | 8 | `prompt_holdout` | `residual_projection_combo` | 304 | 0.1834 | 0.5102 | 0.5417 | 0.0126 |
| qwen3 | 304 | 8 | `prompt_holdout` | `route_competition_combo` | 304 | 0.2768 | 0.5243 | 0.4304 | -0.0808 |
| glm4 | 304 | 8 | `in_sample` | `blocker_field_combo` | 304 | 0.0159 | 0.8454 | 0.0000 | 0.0222 |
| glm4 | 304 | 8 | `in_sample` | `compact_joint_gate_combo` | 304 | 0.0121 | 0.8962 | 0.0000 | 0.0260 |
| glm4 | 304 | 8 | `in_sample` | `global_combo` | 304 | 0.0381 | 0.5888 | 0.0000 | NA |
| glm4 | 304 | 8 | `in_sample` | `internal_strength_combo` | 304 | 0.0297 | 0.7114 | 0.0000 | 0.0084 |
| glm4 | 304 | 8 | `in_sample` | `joint_gate_combo` | 304 | 0.0036 | 0.9751 | 0.0000 | 0.0345 |
| glm4 | 304 | 8 | `in_sample` | `residual_projection_combo` | 304 | 0.0000 | 1.0000 | 0.0000 | 0.0381 |
| glm4 | 304 | 8 | `in_sample` | `route_competition_combo` | 304 | 0.0198 | 0.8552 | 0.0000 | 0.0183 |
| glm4 | 304 | 8 | `object_holdout` | `blocker_field_combo` | 304 | 0.0640 | 0.4884 | 0.0000 | -0.0172 |
| glm4 | 304 | 8 | `object_holdout` | `compact_joint_gate_combo` | 304 | 0.0612 | 0.4323 | 0.0000 | -0.0143 |
| glm4 | 304 | 8 | `object_holdout` | `global_combo` | 304 | 0.0469 | 0.4112 | 0.0000 | NA |
| glm4 | 304 | 8 | `object_holdout` | `internal_strength_combo` | 304 | 0.0520 | 0.4040 | 0.0000 | -0.0051 |
| glm4 | 304 | 8 | `object_holdout` | `joint_gate_combo` | 304 | 0.0619 | 0.4323 | 0.0000 | -0.0150 |
| glm4 | 304 | 8 | `object_holdout` | `residual_projection_combo` | 304 | 0.0528 | 0.4112 | 0.0000 | -0.0060 |
| glm4 | 304 | 8 | `object_holdout` | `route_competition_combo` | 304 | 0.0627 | 0.4272 | 0.0000 | -0.0158 |
| glm4 | 304 | 8 | `prompt_holdout` | `blocker_field_combo` | 304 | 0.0446 | 0.4983 | 0.0000 | -0.0030 |
| glm4 | 304 | 8 | `prompt_holdout` | `compact_joint_gate_combo` | 304 | 0.0482 | 0.4669 | 0.0000 | -0.0065 |
| glm4 | 304 | 8 | `prompt_holdout` | `global_combo` | 304 | 0.0416 | 0.5296 | 0.0000 | NA |
| glm4 | 304 | 8 | `prompt_holdout` | `internal_strength_combo` | 304 | 0.0451 | 0.4934 | 0.0000 | -0.0035 |
| glm4 | 304 | 8 | `prompt_holdout` | `joint_gate_combo` | 304 | 0.0482 | 0.4669 | 0.0000 | -0.0065 |
| glm4 | 304 | 8 | `prompt_holdout` | `residual_projection_combo` | 304 | 0.0433 | 0.5197 | 0.0000 | -0.0016 |
| glm4 | 304 | 8 | `prompt_holdout` | `route_competition_combo` | 304 | 0.0495 | 0.5116 | 0.0000 | -0.0079 |
| deepseek7b | 304 | 8 | `in_sample` | `blocker_field_combo` | 304 | 0.0172 | 0.9071 | 0.0000 | 0.0364 |
| deepseek7b | 304 | 8 | `in_sample` | `compact_joint_gate_combo` | 304 | 0.0370 | 0.7073 | 0.0000 | 0.0165 |
| deepseek7b | 304 | 8 | `in_sample` | `global_combo` | 304 | 0.0536 | 0.5526 | 0.0000 | NA |
| deepseek7b | 304 | 8 | `in_sample` | `internal_strength_combo` | 304 | 0.0485 | 0.6033 | 0.0000 | 0.0051 |
| deepseek7b | 304 | 8 | `in_sample` | `joint_gate_combo` | 304 | 0.0283 | 0.7985 | 0.0000 | 0.0253 |
| deepseek7b | 304 | 8 | `in_sample` | `residual_projection_combo` | 304 | 0.0000 | 1.0000 | 1.0000 | 0.0536 |
| deepseek7b | 304 | 8 | `in_sample` | `route_competition_combo` | 304 | 0.0194 | 0.7794 | 1.0000 | 0.0341 |
| deepseek7b | 304 | 8 | `object_holdout` | `blocker_field_combo` | 304 | 0.0566 | 0.5631 | 0.0000 | 0.0023 |
| deepseek7b | 304 | 8 | `object_holdout` | `compact_joint_gate_combo` | 304 | 0.0704 | 0.4343 | 0.0000 | -0.0115 |
| deepseek7b | 304 | 8 | `object_holdout` | `global_combo` | 304 | 0.0589 | 0.5364 | 0.0000 | NA |
| deepseek7b | 304 | 8 | `object_holdout` | `internal_strength_combo` | 304 | 0.0621 | 0.5033 | 0.0000 | -0.0032 |
| deepseek7b | 304 | 8 | `object_holdout` | `joint_gate_combo` | 304 | 0.0704 | 0.4343 | 0.0000 | -0.0115 |
| deepseek7b | 304 | 8 | `object_holdout` | `residual_projection_combo` | 304 | 0.0796 | 0.3000 | 0.0000 | -0.0208 |
| deepseek7b | 304 | 8 | `object_holdout` | `route_competition_combo` | 304 | 0.0768 | 0.4698 | 0.0000 | -0.0179 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `blocker_field_combo` | 304 | 0.0744 | 0.5537 | 0.0000 | -0.0016 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `compact_joint_gate_combo` | 304 | 0.0825 | 0.4539 | 0.0000 | -0.0098 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `global_combo` | 304 | 0.0727 | 0.4503 | 0.0000 | NA |
| deepseek7b | 304 | 8 | `prompt_holdout` | `internal_strength_combo` | 304 | 0.0722 | 0.4633 | 0.0000 | 0.0006 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `joint_gate_combo` | 304 | 0.0867 | 0.4539 | 0.0000 | -0.0140 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `residual_projection_combo` | 304 | 0.0733 | 0.4485 | 0.0000 | -0.0006 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `route_competition_combo` | 304 | 0.1111 | 0.3434 | 0.0000 | -0.0384 |

## Feature Summary

| model | top1 roles | target rank buckets | residual classes | mean target-blocker logit | mean blocker pressure | mean residual target-blocker |
|---|---|---|---|---:|---:|---:|
| qwen3 | `{"blocker": 304}` | `{"tail": 266, "top100": 38}` | `{"additive": 272, "synergy": 17, "antagonistic": 15}` | -12.8750 | 12.8750 | -13.5242 |
| glm4 | `{"blocker": 304}` | `{"tail": 304}` | `{"additive": 304}` | -7.7188 | 7.7188 | -3.5822 |
| deepseek7b | `{"blocker": 304}` | `{"tail": 304}` | `{"additive": 301, "synergy": 3}` | -9.3242 | 9.3242 | 178.2775 |
