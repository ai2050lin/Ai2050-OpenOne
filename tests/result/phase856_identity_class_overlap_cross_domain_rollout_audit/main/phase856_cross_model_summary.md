# Phase 856 Identity-Class Overlap and Cross-Domain Rollout Audit (main)

- Source: natural prompts across domains; no gear intervention.
- Boundary: first-token field vs short rollout field, not full causal closure.

## Cross-Model Summary

| model | rows | first class | clear first | rollout class | clear rollout | object echo | first->rollout F1 | clear F1 | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 48 | 24 | 24 | 24 | 24 | 3 | 1.0000 | 1.0000 | `{"other": 21, "answer_alias": 12, "strict_canonical": 12, "object_echo": 3}` |
| glm4 | 48 | 27 | 27 | 27 | 27 | 3 | 1.0000 | 1.0000 | `{"strict_canonical": 20, "answer_alias": 7, "other": 18, "object_echo": 3}` |
| deepseek7b | 48 | 15 | 15 | 16 | 16 | 3 | 0.9677 | 0.9677 | `{"other": 21, "answer_alias": 2, "format_or_empty": 8, "strict_canonical": 14, "object_echo": 3}` |

## Domain Summary

| model | domain | rows | first class | rollout class | clear rollout | object echo | clear F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `abstract` | 8 | 0 | 0 | 0 | 2 | 0.0000 |
| qwen3 | `animal` | 8 | 2 | 2 | 2 | 0 | 1.0000 |
| qwen3 | `color` | 8 | 8 | 8 | 8 | 0 | 1.0000 |
| qwen3 | `geometry` | 8 | 3 | 3 | 3 | 0 | 1.0000 |
| qwen3 | `material` | 8 | 3 | 3 | 3 | 1 | 1.0000 |
| qwen3 | `tool` | 8 | 8 | 8 | 8 | 0 | 1.0000 |
| glm4 | `abstract` | 8 | 0 | 0 | 0 | 2 | 0.0000 |
| glm4 | `animal` | 8 | 4 | 4 | 4 | 0 | 1.0000 |
| glm4 | `color` | 8 | 7 | 7 | 7 | 1 | 1.0000 |
| glm4 | `geometry` | 8 | 8 | 8 | 8 | 0 | 1.0000 |
| glm4 | `material` | 8 | 2 | 2 | 2 | 0 | 1.0000 |
| glm4 | `tool` | 8 | 6 | 6 | 6 | 0 | 1.0000 |
| deepseek7b | `abstract` | 8 | 0 | 0 | 0 | 2 | 0.0000 |
| deepseek7b | `animal` | 8 | 2 | 2 | 2 | 0 | 1.0000 |
| deepseek7b | `color` | 8 | 4 | 5 | 5 | 0 | 0.8889 |
| deepseek7b | `geometry` | 8 | 2 | 2 | 2 | 0 | 1.0000 |
| deepseek7b | `material` | 8 | 0 | 0 | 0 | 1 | 0.0000 |
| deepseek7b | `tool` | 8 | 7 | 7 | 7 | 0 | 1.0000 |

## Overlap Summary

| model | overlap kind | rows | first overlap | rollout overlap | object echo | clear rollout | clear F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `attribute_value_not_alias` | 8 | 0 | 0 | 0 | 8 | 1.0000 |
| qwen3 | `member_not_alias` | 40 | 0 | 0 | 3 | 16 | 1.0000 |
| glm4 | `attribute_value_not_alias` | 8 | 0 | 0 | 1 | 7 | 1.0000 |
| glm4 | `member_not_alias` | 40 | 0 | 0 | 2 | 20 | 1.0000 |
| deepseek7b | `attribute_value_not_alias` | 8 | 0 | 0 | 0 | 5 | 0.8889 |
| deepseek7b | `member_not_alias` | 40 | 0 | 0 | 3 | 11 | 1.0000 |
