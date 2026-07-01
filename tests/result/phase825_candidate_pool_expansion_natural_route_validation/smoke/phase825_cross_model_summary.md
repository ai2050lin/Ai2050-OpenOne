# Phase 825 Candidate-Pool Expansion and Natural-Route Validation (smoke)

- Source: Phase 822 signal-bearing components plus Phase 823/824 successful sparse indices.
- Search donor: exact_choices.
- Validation donors: no-choice natural prompt rewrites without explicit choices.

## Model Summary

| model | source comps | rows | exact target | natural target | natural improved | preserve target | transfer pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 2 | 4 | 1 | 0 | 0 | 0 | 2 |
| glm4 | 2 | 4 | 0 | 0 | 0 | 0 | 2 |
| deepseek7b | 2 | 4 | 1 | 0 | 0 | 0 | 2 |

## Donor Summary

| model | donor | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `exact_choices` | 2 | 1 | 1 | 0 | 1.000 | `{"broad_near_miss": 1, "target_equivalent": 1}` |
| qwen3 | `natural_category` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| glm4 | `exact_choices` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `natural_category` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| deepseek7b | `exact_choices` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `natural_category` | 2 | 0 | 0 | 0 | 0.000 | `{"object_echo": 2}` |

## Mode Summary

| model | mode | n | improved | target | degraded | mean delta | classes |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `expanded_greedy_exact` | 2 | 1 | 1 | 0 | 1.000 | `{"broad_near_miss": 1, "target_equivalent": 1}` |
| qwen3 | `natural_route_validation` | 2 | 0 | 0 | 0 | 0.000 | `{"broad_near_miss": 2}` |
| glm4 | `expanded_greedy_exact` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| glm4 | `natural_route_validation` | 2 | 0 | 0 | 0 | 0.000 | `{"unknown_other": 2}` |
| deepseek7b | `expanded_greedy_exact` | 2 | 1 | 1 | 0 | 2.500 | `{"object_echo": 1, "target_equivalent": 1}` |
| deepseek7b | `natural_route_validation` | 2 | 0 | 0 | 0 | 0.000 | `{"object_echo": 2}` |
