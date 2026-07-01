# Phase 827 Exact-Natural Consistency Constrained Search (smoke)

- Source: Phase 822 signal-bearing components.
- Objective: exact+multi-natural consistency before natural-only target gains.

## Model Summary

| model | source comps | rows | exact target | natural target | natural degraded | exact+any natural | exact+multi natural | multi-natural |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 2 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| glm4 | 2 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| deepseek7b | 2 | 4 | 1 | 1 | 0 | 1 | 1 | 1 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| qwen3 | `natural_category` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| glm4 | `exact_choices` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `natural_category` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| deepseek7b | `exact_choices` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `natural_category` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |
