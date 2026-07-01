# Phase 826 Multi-Donor Natural-Objective Sparse Search (main)

- Source: Phase 822 signal-bearing components.
- Objective: exact donor and natural donors are scored during greedy search.

## Model Summary

| model | source comps | rows | exact target | natural target | natural improved | natural degraded | multi-natural pairs | exact+multi-natural |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 4 | 24 | 2 | 6 | 6 | 0 | 0 | 0 |
| glm4 | 4 | 24 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 4 | 24 | 2 | 4 | 4 | 0 | 2 | 2 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 8 | 2 | 2 | 0 | 0.500 | `{"broad_near_miss": 6, "target_equivalent": 2}` |
| qwen3 | `natural_category` | 8 | 2 | 2 | 0 | 0.500 | `{"broad_near_miss": 6, "target_equivalent": 2}` |
| qwen3 | `object_only` | 8 | 4 | 4 | 0 | 1.000 | `{"broad_near_miss": 4, "target_equivalent": 4}` |
| glm4 | `exact_choices` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `natural_category` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `object_only` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| deepseek7b | `exact_choices` | 8 | 2 | 2 | 0 | 1.250 | `{"format_echo": 4, "object_echo": 2, "target_equivalent": 2}` |
| deepseek7b | `natural_category` | 8 | 2 | 2 | 0 | 1.250 | `{"format_echo": 4, "object_echo": 2, "target_equivalent": 2}` |
| deepseek7b | `object_only` | 8 | 2 | 2 | 0 | 1.250 | `{"format_echo": 4, "object_echo": 2, "target_equivalent": 2}` |
