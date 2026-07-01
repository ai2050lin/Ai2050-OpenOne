# Phase 834 Blocker-Aware Internal Route-Boundary Predictor (smoke)

- Source: Phase 833 protocol-structure proxy plus first-step full-vocabulary blocker profile.
- Objective: test whether target-rank / above-target signals can remove qwen3 route interference.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_count_rank_improved` | 1.000 | 5 | 7 | 0 | 0 | 2 | 2 | 0 | 2 | 0 | `{"0": 1, "1": 5}` |
| qwen3 | `category_count_rank_le500` | 0.583 | 0 | 7 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | `{"0": 6}` |
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.750 | 5 | 4 | 3 | 0 | 2 | 2 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 5 | 7 | 0 | 0 | 2 | 2 | 0 | 2 | 0 | `{"0": 1, "1": 5}` |
| glm4 | `category_count_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_rank_le500` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_count_rank_improved` | 0.250 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 1, "1": 1}` |
| deepseek7b | `category_count_rank_le500` | 0.500 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 2}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.750 | 2 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 1, "2": 1}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 2 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 2}` |

## Boundary

First-step blocker awareness is still a proxy. It is useful only if it improves natural_category recovery without adding natural degradation, and only if the same rule works across qwen3 and DS7B.
