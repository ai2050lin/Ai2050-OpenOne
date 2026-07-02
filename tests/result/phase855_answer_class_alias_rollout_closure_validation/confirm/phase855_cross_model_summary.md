# Phase 855 Answer-Class Alias Field and Rollout Closure Validation (confirm)

- Source: Phase 854 full-combo rows.
- Boundary: short greedy rollout, not final language closure.

## Cross-Model Summary

| model | sources | full first-token class | full rollout class | full strict rollout | full object echo | full predictor F1 | labels |
|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 60 | 57 | 57 | 0 | 3 | 1.0000 | `{"answer_alias": 57, "object_echo": 3}` |
| glm4 | 13 | 12 | 12 | 7 | 0 | 1.0000 | `{"other": 1, "answer_alias": 5, "strict_canonical": 7}` |
| deepseek7b | 20 | 17 | 5 | 0 | 12 | 0.4545 | `{"answer_alias": 5, "other": 3, "object_echo": 12}` |

## Conditions

| model | condition | n | first-token class | rollout class | strict token | class blockers | class rank | predictor F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `original` | 60 | 51 | 51 | 0 | 0.1500 | 1.1500 | 1.0000 |
| qwen3 | `full_combo` | 60 | 57 | 57 | 0 | 0.0500 | 1.0500 | 1.0000 |
| qwen3 | `without_necessary` | 10 | 1 | 1 | 0 | 0.9000 | 1.9000 | 1.0000 |
| glm4 | `original` | 13 | 12 | 12 | 12 | 0.2308 | 1.2308 | 1.0000 |
| glm4 | `full_combo` | 13 | 12 | 12 | 7 | 0.2308 | 1.2308 | 1.0000 |
| glm4 | `without_necessary` | 1 | 0 | 0 | 0 | 3.0000 | 4.0000 | 0.0000 |
| deepseek7b | `original` | 20 | 12 | 0 | 0 | 0.4000 | 1.4000 | 0.0000 |
| deepseek7b | `full_combo` | 20 | 17 | 5 | 0 | 0.9000 | 1.9000 | 0.4545 |
| deepseek7b | `without_necessary` | 2 | 0 | 0 | 0 | 1.0000 | 2.0000 | 0.0000 |
