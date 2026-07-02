# Phase 856 Identity-Class Overlap and Cross-Domain Rollout Audit (smoke)

- Source: natural prompts across domains; no gear intervention.
- Boundary: first-token field vs short rollout field, not full causal closure.

## Cross-Model Summary

| model | rows | first class | clear first | rollout class | clear rollout | object echo | first->rollout F1 | clear F1 | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 6 | 2 | 2 | 2 | 2 | 0 | 1.0000 | 1.0000 | `{"other": 4, "answer_alias": 2}` |
| glm4 | 6 | 4 | 4 | 4 | 4 | 0 | 1.0000 | 1.0000 | `{"strict_canonical": 4, "other": 2}` |
| deepseek7b | 6 | 1 | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | `{"other": 4, "format_or_empty": 1, "strict_canonical": 1}` |

## Domain Summary

| model | domain | rows | first class | rollout class | clear rollout | object echo | clear F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `animal` | 2 | 0 | 0 | 0 | 0 | 0.0000 |
| qwen3 | `geometry` | 2 | 0 | 0 | 0 | 0 | 0.0000 |
| qwen3 | `tool` | 2 | 2 | 2 | 2 | 0 | 1.0000 |
| glm4 | `animal` | 2 | 0 | 0 | 0 | 0 | 0.0000 |
| glm4 | `geometry` | 2 | 2 | 2 | 2 | 0 | 1.0000 |
| glm4 | `tool` | 2 | 2 | 2 | 2 | 0 | 1.0000 |
| deepseek7b | `animal` | 2 | 0 | 0 | 0 | 0 | 0.0000 |
| deepseek7b | `geometry` | 2 | 0 | 0 | 0 | 0 | 0.0000 |
| deepseek7b | `tool` | 2 | 1 | 1 | 1 | 0 | 1.0000 |

## Overlap Summary

| model | overlap kind | rows | first overlap | rollout overlap | object echo | clear rollout | clear F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `member_not_alias` | 6 | 0 | 0 | 0 | 2 | 1.0000 |
| glm4 | `member_not_alias` | 6 | 0 | 0 | 0 | 4 | 1.0000 |
| deepseek7b | `member_not_alias` | 6 | 0 | 0 | 0 | 1 | 1.0000 |
