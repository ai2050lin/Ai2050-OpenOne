# Phase 825 Candidate-Pool Expansion and Natural-Route Validation (main)

- Source: Phase 822 signal-bearing components plus Phase 823/824 successful sparse indices.
- Search donor: exact_choices.
- Validation donors: no-choice natural prompt rewrites without explicit choices.

## Model Summary

| model | source comps | rows | exact target | natural target | natural improved | preserve target | transfer pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 8 | 48 | 7 | 2 | 2 | 2 | 32 |
| glm4 | 4 | 24 | 0 | 0 | 0 | 0 | 16 |
| deepseek7b | 5 | 30 | 2 | 3 | 3 | 3 | 20 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 16 | 7 | 7 | 0 | 0.875 | `{"broad_near_miss": 9, "target_equivalent": 7}` |
| qwen3 | `natural_category` | 16 | 2 | 2 | 1 | 0.062 | `{"broad_near_miss": 13, "target_equivalent": 2, "unknown_other": 1}` |
| qwen3 | `natural_question` | 16 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 16}` |
| glm4 | `exact_choices` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `natural_category` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `natural_question` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| deepseek7b | `exact_choices` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |
| deepseek7b | `natural_category` | 10 | 1 | 1 | 0 | 0.500 | `{"format_echo": 6, "object_echo": 3, "target_equivalent": 1}` |
| deepseek7b | `natural_question` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |

## Mode Summary

| model | mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `expanded_greedy_exact` | 16 | 7 | 7 | 0 | 0.875 | `{"broad_near_miss": 9, "target_equivalent": 7}` |
| qwen3 | `natural_route_validation` | 32 | 2 | 2 | 1 | 0.031 | `{"broad_near_miss": 29, "target_equivalent": 2, "unknown_other": 1}` |
| glm4 | `expanded_greedy_exact` | 8 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2, "close_near_miss": 2, "unknown_other": 4}` |
| glm4 | `natural_route_validation` | 16 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 4, "close_near_miss": 4, "unknown_other": 8}` |
| deepseek7b | `expanded_greedy_exact` | 10 | 2 | 2 | 0 | 1.000 | `{"format_echo": 6, "object_echo": 2, "target_equivalent": 2}` |
| deepseek7b | `natural_route_validation` | 20 | 3 | 3 | 0 | 0.750 | `{"format_echo": 12, "object_echo": 5, "target_equivalent": 3}` |
