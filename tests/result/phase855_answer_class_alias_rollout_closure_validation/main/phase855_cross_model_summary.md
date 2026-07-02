# Phase 855 Answer-Class Alias Field and Rollout Closure Validation (main)

- Source: Phase 854 full-combo rows.
- Boundary: short greedy rollout, not final language closure.

## Cross-Model Summary

| model | sources | full first-token class | full rollout class | full strict rollout | full object echo | full predictor F1 | labels |
|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 30 | 30 | 30 | 0 | 0 | 1.0000 | `{"answer_alias": 30}` |
| glm4 | 7 | 6 | 6 | 1 | 0 | 1.0000 | `{"other": 1, "answer_alias": 5, "strict_canonical": 1}` |
| deepseek7b | 14 | 11 | 5 | 0 | 6 | 0.6250 | `{"answer_alias": 5, "other": 3, "object_echo": 6}` |

## Conditions

| model | condition | n | first-token class | rollout class | strict token | class blockers | class rank | predictor F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `original` | 30 | 24 | 24 | 0 | 0.2000 | 1.2000 | 1.0000 |
| qwen3 | `full_combo` | 30 | 30 | 30 | 0 | 0.0000 | 1.0000 | 1.0000 |
| qwen3 | `without_necessary` | 6 | 0 | 0 | 0 | 1.0000 | 2.0000 | 0.0000 |
| glm4 | `original` | 7 | 6 | 6 | 6 | 0.4286 | 1.4286 | 1.0000 |
| glm4 | `full_combo` | 7 | 6 | 6 | 1 | 0.4286 | 1.4286 | 1.0000 |
| glm4 | `without_necessary` | 1 | 0 | 0 | 0 | 3.0000 | 4.0000 | 0.0000 |
| deepseek7b | `original` | 14 | 6 | 0 | 0 | 0.5714 | 1.5714 | 0.0000 |
| deepseek7b | `full_combo` | 14 | 11 | 5 | 0 | 1.2857 | 2.2857 | 0.6250 |
| deepseek7b | `without_necessary` | 2 | 0 | 0 | 0 | 1.0000 | 2.0000 | 0.0000 |
