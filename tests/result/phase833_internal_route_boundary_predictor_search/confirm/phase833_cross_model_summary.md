# Phase 833 Internal Route-Boundary Predictor Search (confirm)

- Source: Phase 829 pairs, Phase 831 internal features, and Phase 832 route-boundary upper bound.
- Objective: test whether simple internal/protocol-structural proxies can replace behavioral route labels.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `category_nonresidual_else_gain_positive` | 0.792 | 22 | 16 | 8 | 2 | 6 | 13 | 0 | 1 | 0 | `{"0": 4, "1": 10, "2": 10}` |
| qwen3 | `category_nonresidual_else_sum_positive` | 0.792 | 22 | 16 | 8 | 2 | 6 | 13 | 0 | 1 | 0 | `{"0": 4, "1": 10, "2": 10}` |
| qwen3 | `donor_full_gain_positive` | 0.583 | 22 | 6 | 18 | 2 | 5 | 11 | 1 | 0 | 0 | `{"1": 8, "2": 16}` |
| qwen3 | `donor_selected_gain_positive` | 0.667 | 22 | 10 | 14 | 2 | 6 | 12 | 0 | 0 | 0 | `{"1": 12, "2": 12}` |
| qwen3 | `full_count_majority` | 0.438 | 15 | 6 | 18 | 9 | 0 | 6 | 0 | 0 | 0 | `{"0": 2, "1": 11, "2": 11}` |
| qwen3 | `full_sum_positive` | 0.583 | 22 | 6 | 18 | 2 | 5 | 11 | 1 | 0 | 0 | `{"1": 8, "2": 16}` |
| qwen3 | `non_residual_count_nonnegative` | 0.333 | 5 | 11 | 13 | 19 | 0 | 5 | 0 | 2 | 0 | `{"0": 6, "1": 18}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 24 | 24 | 0 | 0 | 6 | 14 | 0 | 2 | 0 | `{"0": 4, "1": 16, "2": 4}` |
| qwen3 | `selected_count_majority` | 0.688 | 23 | 10 | 14 | 1 | 5 | 11 | 1 | 0 | 0 | `{"1": 11, "2": 13}` |
| qwen3 | `selected_count_nonnegative` | 0.583 | 23 | 5 | 19 | 1 | 5 | 11 | 1 | 0 | 0 | `{"1": 6, "2": 18}` |
| qwen3 | `selected_energy_ratio_55` | 0.667 | 22 | 10 | 14 | 2 | 6 | 12 | 0 | 0 | 0 | `{"1": 12, "2": 12}` |
| qwen3 | `selected_sum_positive` | 0.667 | 22 | 10 | 14 | 2 | 6 | 12 | 0 | 0 | 0 | `{"1": 12, "2": 12}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_gain_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_sum_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `donor_full_gain_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `donor_selected_gain_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `full_count_majority` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `full_sum_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `non_residual_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `selected_count_majority` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `selected_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `selected_energy_ratio_55` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `selected_sum_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.812 | 9 | 4 | 3 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 4, "2": 4}` |
| deepseek7b | `category_nonresidual_else_gain_positive` | 0.250 | 2 | 2 | 5 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 5, "2": 1}` |
| deepseek7b | `category_nonresidual_else_sum_positive` | 0.250 | 2 | 2 | 5 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 5, "2": 1}` |
| deepseek7b | `donor_full_gain_positive` | 0.188 | 3 | 0 | 7 | 6 | 0 | 2 | 0 | 0 | 0 | `{"1": 6, "2": 2}` |
| deepseek7b | `donor_selected_gain_positive` | 0.250 | 2 | 2 | 5 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 5, "2": 1}` |
| deepseek7b | `full_count_majority` | 0.562 | 9 | 0 | 7 | 0 | 2 | 5 | 0 | 1 | 0 | `{"2": 8}` |
| deepseek7b | `full_sum_positive` | 0.188 | 3 | 0 | 7 | 6 | 0 | 2 | 0 | 0 | 0 | `{"1": 6, "2": 2}` |
| deepseek7b | `non_residual_count_nonnegative` | 0.938 | 8 | 7 | 0 | 1 | 2 | 6 | 0 | 2 | 0 | `{"1": 8}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 9 | 7 | 0 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 7, "2": 1}` |
| deepseek7b | `selected_count_majority` | 0.688 | 8 | 3 | 4 | 1 | 2 | 5 | 0 | 1 | 0 | `{"1": 4, "2": 4}` |
| deepseek7b | `selected_count_nonnegative` | 0.688 | 9 | 2 | 5 | 0 | 2 | 5 | 0 | 1 | 0 | `{"1": 2, "2": 6}` |
| deepseek7b | `selected_energy_ratio_55` | 0.250 | 1 | 3 | 4 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 3, "1": 5}` |
| deepseek7b | `selected_sum_positive` | 0.250 | 2 | 2 | 5 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 5, "2": 1}` |

## Boundary

The non-oracle modes do not use single-component behavioral classes as gate inputs. If they approach oracle_route_target_only, route boundary has an internal/protocol-structural proxy; if not, Phase 832's route boundary remains behavioral rather than endogenous.
