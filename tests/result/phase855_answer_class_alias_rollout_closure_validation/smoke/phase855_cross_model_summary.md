# Phase 855 Answer-Class Alias Field and Rollout Closure Validation (smoke)

- Source: Phase 854 full-combo rows.
- Boundary: short greedy rollout, not final language closure.

## Cross-Model Summary

| model | sources | full first-token class | full rollout class | full strict rollout | full object echo | full predictor F1 | labels |
|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 9 | 9 | 9 | 0 | 0 | 1.0000 | `{"answer_alias": 9}` |
| glm4 | 3 | 2 | 2 | 0 | 0 | 1.0000 | `{"other": 1, "answer_alias": 2}` |
| deepseek7b | 7 | 5 | 3 | 0 | 2 | 0.7500 | `{"answer_alias": 3, "other": 2, "object_echo": 2}` |

## Conditions

| model | condition | n | first-token class | rollout class | strict token | class blockers | class rank | predictor F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `original` | 9 | 7 | 7 | 0 | 0.2222 | 1.2222 | 1.0000 |
| qwen3 | `full_combo` | 9 | 9 | 9 | 0 | 0.0000 | 1.0000 | 1.0000 |
| qwen3 | `without_necessary` | 2 | 0 | 0 | 0 | 1.0000 | 2.0000 | 0.0000 |
| glm4 | `original` | 3 | 2 | 2 | 2 | 1.0000 | 2.0000 | 1.0000 |
| glm4 | `full_combo` | 3 | 2 | 2 | 0 | 1.0000 | 2.0000 | 1.0000 |
| glm4 | `without_necessary` | 1 | 0 | 0 | 0 | 3.0000 | 4.0000 | 0.0000 |
| deepseek7b | `original` | 7 | 2 | 0 | 0 | 0.7143 | 1.7143 | 0.0000 |
| deepseek7b | `full_combo` | 7 | 5 | 3 | 0 | 1.7143 | 2.7143 | 0.7500 |
| deepseek7b | `without_necessary` | 1 | 0 | 0 | 0 | 1.0000 | 2.0000 | 0.0000 |
