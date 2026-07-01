# Phase 834 Blocker-Aware Internal Route-Boundary Predictor (confirm)

- Source: Phase 833 protocol-structure proxy plus first-step full-vocabulary blocker profile.
- Objective: test whether target-rank / above-target signals can remove qwen3 route interference.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_count_above_decreased` | 0.896 | 23 | 20 | 4 | 1 | 6 | 14 | 0 | 2 | 0 | `{"0": 3, "1": 15, "2": 6}` |
| qwen3 | `category_count_rank_improved` | 0.896 | 23 | 20 | 4 | 1 | 6 | 14 | 0 | 2 | 0 | `{"0": 3, "1": 15, "2": 6}` |
| qwen3 | `category_count_rank_le2000` | 0.500 | 0 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | `{"0": 24}` |
| qwen3 | `category_count_rank_le50` | 0.500 | 0 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | `{"0": 24}` |
| qwen3 | `category_count_rank_le500` | 0.500 | 0 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | `{"0": 24}` |
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `count_rank_improved` | 0.771 | 23 | 14 | 10 | 1 | 6 | 12 | 0 | 0 | 0 | `{"1": 15, "2": 9}` |
| qwen3 | `count_rank_le500` | 0.500 | 0 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | `{"0": 24}` |
| qwen3 | `nonresidual_count_rank_le500` | 0.500 | 0 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | `{"0": 24}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 24 | 24 | 0 | 0 | 6 | 14 | 0 | 2 | 0 | `{"0": 4, "1": 16, "2": 4}` |
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
| deepseek7b | `category_count_above_decreased` | 0.375 | 2 | 4 | 3 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 5}` |
| deepseek7b | `category_count_rank_improved` | 0.375 | 2 | 4 | 3 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 5}` |
| deepseek7b | `category_count_rank_le2000` | 0.438 | 3 | 4 | 3 | 6 | 0 | 2 | 0 | 0 | 0 | `{"0": 2, "1": 6}` |
| deepseek7b | `category_count_rank_le50` | 0.438 | 0 | 7 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | `{"0": 8}` |
| deepseek7b | `category_count_rank_le500` | 0.438 | 0 | 7 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | `{"0": 8}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.812 | 9 | 4 | 3 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 4, "2": 4}` |
| deepseek7b | `count_rank_improved` | 0.312 | 2 | 3 | 4 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 6}` |
| deepseek7b | `count_rank_le500` | 0.438 | 0 | 7 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | `{"0": 8}` |
| deepseek7b | `nonresidual_count_rank_le500` | 0.438 | 0 | 7 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | `{"0": 8}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 9 | 7 | 0 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 7, "2": 1}` |

## Boundary

First-step blocker awareness is still a proxy. It is useful only if it improves natural_category recovery without adding natural degradation, and only if the same rule works across qwen3 and DS7B.
