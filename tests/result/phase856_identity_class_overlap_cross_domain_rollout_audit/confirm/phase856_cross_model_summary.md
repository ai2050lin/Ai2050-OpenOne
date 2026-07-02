# Phase 856 Identity-Class Overlap and Cross-Domain Rollout Audit (confirm)

- Source: natural prompts across domains; no gear intervention.
- Boundary: first-token field vs short rollout field, not full causal closure.

## Cross-Model Summary

| model | rows | first class | clear first | rollout class | clear rollout | object echo | first->rollout F1 | clear F1 | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 114 | 41 | 34 | 41 | 37 | 10 | 0.9756 | 0.9296 | `{"other": 61, "answer_alias": 22, "object_echo": 10, "strict_canonical": 15, "format_or_empty": 2, "identity_class_overlap": 4}` |
| glm4 | 114 | 45 | 38 | 46 | 38 | 5 | 0.9670 | 1.0000 | `{"strict_canonical": 27, "answer_alias": 11, "other": 63, "identity_class_overlap": 8, "object_echo": 5}` |
| deepseek7b | 114 | 32 | 23 | 34 | 24 | 13 | 0.9697 | 0.9787 | `{"other": 52, "answer_alias": 6, "object_echo": 13, "format_or_empty": 15, "identity_class_overlap": 10, "strict_canonical": 18}` |

## Domain Summary

| model | domain | rows | first class | rollout class | clear rollout | object echo | clear F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `abstract` | 15 | 1 | 1 | 1 | 5 | 0.0000 |
| qwen3 | `animal` | 15 | 4 | 4 | 4 | 0 | 0.8571 |
| qwen3 | `color` | 15 | 10 | 10 | 8 | 1 | 1.0000 |
| qwen3 | `geometry` | 15 | 3 | 3 | 3 | 1 | 1.0000 |
| qwen3 | `material` | 15 | 4 | 4 | 3 | 2 | 1.0000 |
| qwen3 | `object` | 12 | 3 | 3 | 2 | 1 | 0.6667 |
| qwen3 | `plant` | 12 | 7 | 7 | 7 | 0 | 1.0000 |
| qwen3 | `tool` | 15 | 9 | 9 | 9 | 0 | 0.9412 |
| glm4 | `abstract` | 15 | 2 | 1 | 0 | 3 | 0.0000 |
| glm4 | `animal` | 15 | 5 | 5 | 5 | 0 | 1.0000 |
| glm4 | `color` | 15 | 9 | 9 | 7 | 1 | 1.0000 |
| glm4 | `geometry` | 15 | 10 | 10 | 10 | 0 | 1.0000 |
| glm4 | `material` | 15 | 3 | 4 | 2 | 1 | 1.0000 |
| glm4 | `object` | 12 | 2 | 3 | 2 | 0 | 1.0000 |
| glm4 | `plant` | 12 | 6 | 6 | 6 | 0 | 1.0000 |
| glm4 | `tool` | 15 | 8 | 8 | 6 | 0 | 1.0000 |
| deepseek7b | `abstract` | 15 | 2 | 2 | 0 | 3 | 0.0000 |
| deepseek7b | `animal` | 15 | 4 | 4 | 4 | 1 | 1.0000 |
| deepseek7b | `color` | 15 | 5 | 7 | 5 | 2 | 0.8889 |
| deepseek7b | `geometry` | 15 | 4 | 4 | 2 | 2 | 1.0000 |
| deepseek7b | `material` | 15 | 1 | 1 | 0 | 3 | 0.0000 |
| deepseek7b | `object` | 12 | 1 | 1 | 1 | 2 | 1.0000 |
| deepseek7b | `plant` | 12 | 2 | 2 | 2 | 0 | 1.0000 |
| deepseek7b | `tool` | 15 | 13 | 13 | 10 | 0 | 1.0000 |

## Overlap Summary

| model | overlap kind | rows | first overlap | rollout overlap | object echo | clear rollout | clear F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `attribute_value_not_alias` | 12 | 0 | 0 | 1 | 8 | 1.0000 |
| qwen3 | `member_not_alias` | 78 | 0 | 0 | 9 | 22 | 1.0000 |
| qwen3 | `object_is_answer_alias` | 24 | 16 | 4 | 0 | 7 | 0.5455 |
| glm4 | `attribute_value_not_alias` | 12 | 0 | 0 | 1 | 7 | 1.0000 |
| glm4 | `member_not_alias` | 78 | 0 | 0 | 4 | 25 | 1.0000 |
| glm4 | `object_is_answer_alias` | 24 | 15 | 8 | 0 | 6 | 1.0000 |
| deepseek7b | `attribute_value_not_alias` | 12 | 0 | 0 | 2 | 5 | 0.8889 |
| deepseek7b | `member_not_alias` | 78 | 0 | 0 | 11 | 17 | 1.0000 |
| deepseek7b | `object_is_answer_alias` | 24 | 17 | 10 | 0 | 2 | 1.0000 |
