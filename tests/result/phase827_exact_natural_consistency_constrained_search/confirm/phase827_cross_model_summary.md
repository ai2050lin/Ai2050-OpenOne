# Phase 827 Exact-Natural Consistency Constrained Search (confirm)

- Source: Phase 822 signal-bearing components.
- Objective: exact+multi-natural consistency before natural-only target gains.

## Model Summary

| model | source comps | rows | exact target | natural target | natural degraded | exact+any natural | exact+multi natural | multi-natural |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 8 | 64 | 4 | 15 | 1 | 2 | 2 | 4 |
| glm4 | 4 | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 5 | 40 | 3 | 6 | 0 | 2 | 2 | 2 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 16 | 4 | 4 | 0 | 0.500 | `{"broad_near_miss": 12, "target_equivalent": 4}` |
| qwen3 | `natural_category` | 16 | 3 | 3 | 1 | 0.188 | `{"broad_near_miss": 12, "target_equivalent": 3, "unknown_other": 1}` |
| qwen3 | `natural_question` | 16 | 3 | 3 | 0 | 0.375 | `{"broad_near_miss": 13, "target_equivalent": 3}` |
| qwen3 | `object_only` | 16 | 9 | 9 | 0 | 1.125 | `{"broad_near_miss": 7, "target_equivalent": 9}` |
| glm4 | `exact_choices` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `natural_category` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `natural_question` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `object_only` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| deepseek7b | `exact_choices` | 10 | 3 | 3 | 0 | 1.500 | `{"format_echo": 6, "object_echo": 1, "target_equivalent": 3}` |
| deepseek7b | `natural_category` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |
| deepseek7b | `natural_question` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |
| deepseek7b | `object_only` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |
