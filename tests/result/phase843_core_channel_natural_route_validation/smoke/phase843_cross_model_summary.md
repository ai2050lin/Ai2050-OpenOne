# Phase 843 Core Channel Natural Route Validation (smoke)

- Source: Phase 842 core channel candidate.
- Boundary: natural activation + first-step channel edit; not global closure.

## Model Summary

| model | skipped | rows | cases | original target | target | lost vs original | gained vs original |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| glm4 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| deepseek7b | 0 | 3 | 1 | 0 | 2 | 0 | 2 |

## Mode Summary

| model | mode | n | target | lost | gained | object_echo | unknown | mean act | mean target-object | classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `flip` | 1 | 1 | 0 | 1 | 0 | 0 | -18.6250 | -0.9375 | `{"target_equivalent": 1}` |
| deepseek7b | `original` | 1 | 0 | 0 | 0 | 0 | 1 | -18.6250 | -1.1875 | `{"unknown_other": 1}` |
| deepseek7b | `zero` | 1 | 1 | 0 | 1 | 0 | 0 | -18.6250 | -1.0625 | `{"target_equivalent": 1}` |

## Object Summary

| model | object | n | target | lost | gained | mean act | mean abs act | classes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| deepseek7b | `triangle` | 3 | 2 | 0 | 2 | -18.6250 | 18.6250 | `{"target_equivalent": 2, "unknown_other": 1}` |

## Top Rows

| model | object | prompt | mode | act | class | output | orig class | lost | gained | target-object | target rank | object rank | top tokens |
|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| deepseek7b | `triangle` | `natural_question` | `zero` | -18.6250 | `target_equivalent` | Polygon | `unknown_other` | 0 | 1 | -1.0625 | 176 | 89 | `[" Polygon", " polygon", " ", " triangle", " ["]` |
| deepseek7b | `triangle` | `natural_question` | `flip` | -18.6250 | `target_equivalent` | polygon | `unknown_other` | 0 | 1 | -0.9375 | 130 | 73 | `[" polygon", " Polygon", " polygons", " triangle", " "]` |
| deepseek7b | `triangle` | `natural_question` | `original` | -18.6250 | `unknown_other` | 1 | `unknown_other` | 0 | 0 | -1.1875 | 245 | 121 | `[" ", " [", " geometry", " Polygon", " A"]` |
