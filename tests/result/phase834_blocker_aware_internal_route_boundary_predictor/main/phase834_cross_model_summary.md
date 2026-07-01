# Phase 834 Blocker-Aware Internal Route-Boundary Predictor (main)

- Source: Phase 833 protocol-structure proxy plus first-step full-vocabulary blocker profile.
- Objective: test whether target-rank / above-target signals can remove qwen3 route interference.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_count_above_decreased` | 0.917 | 17 | 16 | 2 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 3, "1": 11, "2": 4}` |
| qwen3 | `category_count_rank_improved` | 0.917 | 17 | 16 | 2 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 3, "1": 11, "2": 4}` |
| qwen3 | `category_count_rank_le2000` | 0.500 | 0 | 18 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | `{"0": 18}` |
| qwen3 | `category_count_rank_le50` | 0.500 | 0 | 18 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | `{"0": 18}` |
| qwen3 | `category_count_rank_le500` | 0.500 | 0 | 18 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | `{"0": 18}` |
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `count_rank_improved` | 0.750 | 17 | 10 | 8 | 1 | 0 | 6 | 0 | 0 | 0 | `{"1": 11, "2": 7}` |
| qwen3 | `count_rank_le500` | 0.500 | 0 | 18 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | `{"0": 18}` |
| qwen3 | `nonresidual_count_rank_le500` | 0.500 | 0 | 18 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | `{"0": 18}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 18 | 18 | 0 | 0 | 2 | 8 | 0 | 2 | 0 | `{"0": 4, "1": 10, "2": 4}` |
| glm4 | `category_count_above_decreased` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_rank_le2000` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_rank_le50` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_rank_le500` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `count_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `count_rank_le500` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `nonresidual_count_rank_le500` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_count_above_decreased` | 0.500 | 2 | 4 | 1 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 3}` |
| deepseek7b | `category_count_rank_improved` | 0.500 | 2 | 4 | 1 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 3}` |
| deepseek7b | `category_count_rank_le2000` | 0.583 | 3 | 4 | 1 | 4 | 0 | 2 | 0 | 0 | 0 | `{"0": 2, "1": 4}` |
| deepseek7b | `category_count_rank_le50` | 0.417 | 0 | 5 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | `{"0": 6}` |
| deepseek7b | `category_count_rank_le500` | 0.417 | 0 | 5 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | `{"0": 6}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.917 | 7 | 4 | 1 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `count_rank_improved` | 0.417 | 2 | 3 | 2 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 4}` |
| deepseek7b | `count_rank_le500` | 0.417 | 0 | 5 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | `{"0": 6}` |
| deepseek7b | `nonresidual_count_rank_le500` | 0.417 | 0 | 5 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | `{"0": 6}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 7 | 5 | 0 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 5, "2": 1}` |

## Boundary

First-step blocker awareness is still a proxy. It is useful only if it improves natural_category recovery without adding natural degradation, and only if the same rule works across qwen3 and DS7B.
