# Phase 826 Multi-Donor Natural-Objective Sparse Search (confirm)

- Source: Phase 822 signal-bearing components.
- Objective: exact donor and natural donors are scored during greedy search.

## Model Summary

| model | source comps | rows | exact target | natural target | natural improved | natural degraded | multi-natural pairs | exact+multi-natural |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 8 | 64 | 1 | 13 | 13 | 3 | 3 | 0 |
| glm4 | 4 | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 5 | 40 | 2 | 6 | 6 | 0 | 2 | 2 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 16 | 1 | 1 | 0 | 0.125 | `{"broad_near_miss": 15, "target_equivalent": 1}` |
| qwen3 | `natural_category` | 16 | 4 | 4 | 2 | 0.125 | `{"broad_near_miss": 10, "target_equivalent": 4, "unknown_other": 2}` |
| qwen3 | `natural_question` | 16 | 3 | 3 | 1 | 0.188 | `{"broad_near_miss": 12, "target_equivalent": 3, "unknown_other": 1}` |
| qwen3 | `object_only` | 16 | 6 | 6 | 0 | 0.750 | `{"broad_near_miss": 10, "target_equivalent": 6}` |
| glm4 | `exact_choices` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `natural_category` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `natural_question` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `object_only` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| deepseek7b | `exact_choices` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |
| deepseek7b | `natural_category` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |
| deepseek7b | `natural_question` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |
| deepseek7b | `object_only` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |
