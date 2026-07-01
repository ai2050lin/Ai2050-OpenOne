# Phase 836 Full-Span Causal Blocker Profile (main)

- Source: Phase 835 plus stepwise patched answer-span scoring.
- Objective: test whether DS7B needs multi-token span evidence rather than first-token evidence.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_count_fullspan_closure` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `category_count_fullspan_contrast_cleared` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `category_count_fullspan_generic_cleared` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `category_count_fullspan_margin_improved` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `category_count_fullspan_or_rank_improved` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `category_count_fullspan_rank_improved` | 0.500 | 0 | 18 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | `{"0": 18}` |
| qwen3 | `category_count_rank_improved` | 0.917 | 17 | 16 | 2 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 3, "1": 11, "2": 4}` |
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 18 | 18 | 0 | 0 | 2 | 8 | 0 | 2 | 0 | `{"0": 4, "1": 10, "2": 4}` |
| glm4 | `category_count_fullspan_closure` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_contrast_cleared` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_generic_cleared` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_margin_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_or_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_count_fullspan_closure` | 0.417 | 1 | 4 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 2}` |
| deepseek7b | `category_count_fullspan_contrast_cleared` | 0.917 | 7 | 4 | 1 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `category_count_fullspan_generic_cleared` | 0.417 | 1 | 4 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 2}` |
| deepseek7b | `category_count_fullspan_margin_improved` | 0.417 | 1 | 4 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 2}` |
| deepseek7b | `category_count_fullspan_or_rank_improved` | 0.500 | 2 | 4 | 1 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 3}` |
| deepseek7b | `category_count_fullspan_rank_improved` | 0.417 | 1 | 4 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 2}` |
| deepseek7b | `category_count_rank_improved` | 0.500 | 2 | 4 | 1 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 3}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.917 | 7 | 4 | 1 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 7 | 5 | 0 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 5, "2": 1}` |

## Boundary

Full-span scoring is still a teacher-forced diagnostic; it is not identical to free generation closure.
