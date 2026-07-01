# Phase 833 Internal Route-Boundary Predictor Search (smoke)

- Source: Phase 829 pairs, Phase 831 internal features, and Phase 832 route-boundary upper bound.
- Objective: test whether simple internal/protocol-structural proxies can replace behavioral route labels.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.750 | 5 | 4 | 3 | 0 | 2 | 2 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 5 | 7 | 0 | 0 | 2 | 2 | 0 | 2 | 0 | `{"0": 1, "1": 5}` |
| qwen3 | `selected_count_nonnegative` | 0.500 | 5 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | `{"1": 1, "2": 5}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `selected_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.750 | 2 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 1, "2": 1}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 2 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 2}` |
| deepseek7b | `selected_count_nonnegative` | 0.500 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | `{"2": 2}` |

## Boundary

The non-oracle modes do not use single-component behavioral classes as gate inputs. If they approach oracle_route_target_only, route boundary has an internal/protocol-structural proxy; if not, Phase 832's route boundary remains behavioral rather than endogenous.
