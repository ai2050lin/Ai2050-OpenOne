# Phase 836 Full-Span Causal Blocker Profile (smoke)

- Source: Phase 835 plus stepwise patched answer-span scoring.
- Objective: test whether DS7B needs multi-token span evidence rather than first-token evidence.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_count_fullspan_contrast_cleared` | 0.750 | 3 | 3 | 2 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 3, "2": 1}` |
| qwen3 | `category_count_fullspan_margin_improved` | 0.625 | 1 | 4 | 1 | 2 | 0 | 1 | 0 | 1 | 0 | `{"0": 2, "1": 2}` |
| qwen3 | `category_count_rank_improved` | 1.000 | 3 | 5 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | `{"0": 1, "1": 3}` |
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.750 | 3 | 3 | 2 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 3, "2": 1}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 3 | 5 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | `{"0": 1, "1": 3}` |
| glm4 | `category_count_fullspan_contrast_cleared` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_margin_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_count_fullspan_contrast_cleared` | 0.750 | 2 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 1, "2": 1}` |
| deepseek7b | `category_count_fullspan_margin_improved` | 0.250 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 1, "1": 1}` |
| deepseek7b | `category_count_rank_improved` | 0.250 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 1, "1": 1}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.750 | 2 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 1, "2": 1}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 2 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 2}` |

## Boundary

Full-span scoring is still a teacher-forced diagnostic; it is not identical to free generation closure.
