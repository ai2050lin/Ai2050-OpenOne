# Phase 825 Candidate-Pool Expansion and Natural-Route Validation (confirm)

- Source: Phase 822 signal-bearing components plus Phase 823/824 successful sparse indices.
- Search donor: exact_choices.
- Validation donors: no-choice natural prompt rewrites without explicit choices.

## Model Summary

| model | source comps | rows | exact target | natural target | natural improved | preserve target | transfer pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 14 | 168 | 18 | 8 | 8 | 6 | 126 |
| glm4 | 4 | 48 | 3 | 0 | 0 | 0 | 36 |
| deepseek7b | 5 | 56 | 5 | 9 | 9 | 9 | 42 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 42 | 18 | 18 | 0 | 0.810 | `{"broad_near_miss": 17, "close_near_miss": 7, "target_equivalent": 18}` |
| qwen3 | `natural_category` | 42 | 3 | 3 | 3 | -0.071 | `{"broad_near_miss": 27, "close_near_miss": 9, "target_equivalent": 3, "unknown_other": 3}` |
| qwen3 | `natural_question` | 42 | 0 | 0 | 2 | -0.143 | `{"broad_near_miss": 31, "close_near_miss": 9, "unknown_other": 2}` |
| qwen3 | `object_only` | 42 | 5 | 5 | 0 | 0.238 | `{"broad_near_miss": 28, "close_near_miss": 9, "target_equivalent": 5}` |
| glm4 | `exact_choices` | 12 | 3 | 3 | 0 | 0.250 | `{"broad_near_miss": 3, "target_equivalent": 3, "unknown_other": 6}` |
| glm4 | `natural_category` | 12 | 0 | 0 | 3 | -1.000 | `{"broad_near_miss": 3, "unknown_other": 9}` |
| glm4 | `natural_question` | 12 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "close_near_miss": 3, "unknown_other": 6}` |
| glm4 | `object_only` | 12 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 3, "close_near_miss": 3, "unknown_other": 6}` |
| deepseek7b | `exact_choices` | 14 | 5 | 5 | 0 | 1.786 | `{"format_echo": 9, "target_equivalent": 5}` |
| deepseek7b | `natural_category` | 14 | 2 | 2 | 0 | 0.714 | `{"format_echo": 9, "object_echo": 3, "target_equivalent": 2}` |
| deepseek7b | `natural_question` | 14 | 2 | 2 | 0 | 0.714 | `{"format_echo": 9, "object_echo": 3, "target_equivalent": 2}` |
| deepseek7b | `object_only` | 14 | 5 | 5 | 0 | 1.786 | `{"format_echo": 9, "target_equivalent": 5}` |

## Mode Summary

| model | mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `expanded_greedy_exact` | 42 | 18 | 18 | 0 | 0.810 | `{"broad_near_miss": 17, "close_near_miss": 7, "target_equivalent": 18}` |
| qwen3 | `natural_route_validation` | 126 | 8 | 8 | 5 | 0.008 | `{"broad_near_miss": 86, "close_near_miss": 27, "target_equivalent": 8, "unknown_other": 5}` |
| glm4 | `expanded_greedy_exact` | 12 | 3 | 3 | 0 | 0.250 | `{"broad_near_miss": 3, "target_equivalent": 3, "unknown_other": 6}` |
| glm4 | `natural_route_validation` | 36 | 0 | 0 | 3 | -0.333 | `{"broad_near_miss": 9, "close_near_miss": 6, "unknown_other": 21}` |
| deepseek7b | `expanded_greedy_exact` | 14 | 5 | 5 | 0 | 1.786 | `{"format_echo": 9, "target_equivalent": 5}` |
| deepseek7b | `natural_route_validation` | 42 | 9 | 9 | 0 | 1.071 | `{"format_echo": 27, "object_echo": 6, "target_equivalent": 9}` |
