# Phase 848 Internal Route Gate Discovery (smoke)

- Source: Phase 845 residual rows plus fresh natural activation captures.
- Method: compare external and internal activation-gate residual predictors.
- Boundary: internal gate discovery probe; not geometry closure.

## Model Summary

| model | feature rows | prompts | split | predictor | n | MAE | sign acc | strong F1 | MAE gain vs global |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| qwen3 | 12 | 1 | `in_sample` | `global_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | NA |
| qwen3 | 12 | 1 | `in_sample` | `internal_count_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | 12 | 1 | `in_sample` | `internal_sign_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | 12 | 1 | `in_sample` | `internal_strength_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | 12 | 1 | `in_sample` | `object_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | 12 | 1 | `in_sample` | `prompt_combo` | 12 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `global_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | NA |
| glm4 | 12 | 1 | `in_sample` | `internal_count_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `internal_sign_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `internal_strength_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `object_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| glm4 | 12 | 1 | `in_sample` | `prompt_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `global_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | NA |
| deepseek7b | 12 | 1 | `in_sample` | `internal_count_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `internal_sign_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `internal_strength_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `object_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek7b | 12 | 1 | `in_sample` | `prompt_combo` | 12 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

## Feature Summary

| model | sign patterns | neg count | residual classes | mean abs sum |
|---|---|---|---|---:|
| qwen3 | `{"+-": 6, "++": 2, "-+": 2, "--": 2}` | `{"1": 8, "0": 2, "2": 2}` | `{"additive": 11, "synergy": 1}` | 29.2676 |
| glm4 | `{"++": 6, "+-": 4, "-+": 2}` | `{"0": 6, "1": 6}` | `{"additive": 12}` | 5.9944 |
| deepseek7b | `{"--": 6, "-+": 6}` | `{"2": 6, "1": 6}` | `{"additive": 12}` | 124.4250 |
