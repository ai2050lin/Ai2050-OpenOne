# Phase 833 Internal Route-Boundary Predictor Search (main)

- Source: Phase 829 pairs, Phase 831 internal features, and Phase 832 route-boundary upper bound.
- Objective: test whether simple internal/protocol-structural proxies can replace behavioral route labels.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `category_nonresidual_else_sum_positive` | 0.778 | 16 | 12 | 6 | 2 | 1 | 7 | 0 | 1 | 0 | `{"0": 4, "1": 6, "2": 8}` |
| qwen3 | `donor_selected_gain_positive` | 0.611 | 16 | 6 | 12 | 2 | 0 | 6 | 0 | 0 | 0 | `{"1": 8, "2": 10}` |
| qwen3 | `full_count_majority` | 0.528 | 15 | 4 | 14 | 3 | 0 | 6 | 0 | 0 | 0 | `{"1": 7, "2": 11}` |
| qwen3 | `full_sum_positive` | 0.556 | 16 | 4 | 14 | 2 | 0 | 6 | 0 | 0 | 0 | `{"1": 6, "2": 12}` |
| qwen3 | `non_residual_count_nonnegative` | 0.389 | 5 | 9 | 9 | 13 | 0 | 5 | 0 | 2 | 0 | `{"0": 4, "1": 14}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 18 | 18 | 0 | 0 | 2 | 8 | 0 | 2 | 0 | `{"0": 4, "1": 10, "2": 4}` |
| qwen3 | `selected_count_majority` | 0.639 | 17 | 6 | 12 | 1 | 0 | 6 | 0 | 0 | 0 | `{"1": 7, "2": 11}` |
| qwen3 | `selected_count_nonnegative` | 0.556 | 17 | 3 | 15 | 1 | 0 | 6 | 0 | 0 | 0 | `{"1": 4, "2": 14}` |
| qwen3 | `selected_sum_positive` | 0.611 | 16 | 6 | 12 | 2 | 0 | 6 | 0 | 0 | 0 | `{"1": 8, "2": 10}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_sum_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `donor_selected_gain_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `full_count_majority` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `full_sum_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `non_residual_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `selected_count_majority` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `selected_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `selected_sum_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.917 | 7 | 4 | 1 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `category_nonresidual_else_sum_positive` | 0.333 | 2 | 2 | 3 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 3, "2": 1}` |
| deepseek7b | `donor_selected_gain_positive` | 0.333 | 2 | 2 | 3 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 3, "2": 1}` |
| deepseek7b | `full_count_majority` | 0.583 | 7 | 0 | 5 | 0 | 1 | 3 | 0 | 1 | 0 | `{"2": 6}` |
| deepseek7b | `full_sum_positive` | 0.250 | 3 | 0 | 5 | 4 | 0 | 2 | 0 | 0 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `non_residual_count_nonnegative` | 0.917 | 6 | 5 | 0 | 1 | 2 | 4 | 0 | 2 | 0 | `{"1": 6}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 7 | 5 | 0 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 5, "2": 1}` |
| deepseek7b | `selected_count_majority` | 0.750 | 6 | 3 | 2 | 1 | 1 | 3 | 0 | 1 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `selected_count_nonnegative` | 0.750 | 7 | 2 | 3 | 0 | 1 | 3 | 0 | 1 | 0 | `{"1": 2, "2": 4}` |
| deepseek7b | `selected_sum_positive` | 0.333 | 2 | 2 | 3 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 3, "2": 1}` |

## Boundary

The non-oracle modes do not use single-component behavioral classes as gate inputs. If they approach oracle_route_target_only, route boundary has an internal/protocol-structural proxy; if not, Phase 832's route boundary remains behavioral rather than endogenous.
