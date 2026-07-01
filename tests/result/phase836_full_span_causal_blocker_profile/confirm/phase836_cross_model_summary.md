# Phase 836 Full-Span Causal Blocker Profile (confirm)

- Source: Phase 835 plus stepwise patched answer-span scoring.
- Objective: test whether DS7B needs multi-token span evidence rather than first-token evidence.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_count_fullspan_and_rank_improved` | 0.896 | 23 | 20 | 4 | 1 | 6 | 14 | 0 | 2 | 0 | `{"0": 3, "1": 15, "2": 6}` |
| qwen3 | `category_count_fullspan_closure` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `category_count_fullspan_contrast_cleared` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `category_count_fullspan_generic_cleared` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `category_count_fullspan_margin_improved` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `category_count_fullspan_or_rank_improved` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `category_count_fullspan_rank_improved` | 0.500 | 0 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | `{"0": 24}` |
| qwen3 | `category_count_rank_improved` | 0.896 | 23 | 20 | 4 | 1 | 6 | 14 | 0 | 2 | 0 | `{"0": 3, "1": 15, "2": 6}` |
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 24 | 24 | 0 | 0 | 6 | 14 | 0 | 2 | 0 | `{"0": 4, "1": 16, "2": 4}` |
| glm4 | `category_count_fullspan_and_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_closure` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_contrast_cleared` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_generic_cleared` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_margin_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_or_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_fullspan_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_count_fullspan_and_rank_improved` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_count_fullspan_closure` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_count_fullspan_contrast_cleared` | 0.812 | 9 | 4 | 3 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 4, "2": 4}` |
| deepseek7b | `category_count_fullspan_generic_cleared` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_count_fullspan_margin_improved` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_count_fullspan_or_rank_improved` | 0.375 | 2 | 4 | 3 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 5}` |
| deepseek7b | `category_count_fullspan_rank_improved` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_count_rank_improved` | 0.375 | 2 | 4 | 3 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 5}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.812 | 9 | 4 | 3 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 4, "2": 4}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 9 | 7 | 0 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 7, "2": 1}` |

## Boundary

Full-span scoring is still a teacher-forced diagnostic; it is not identical to free generation closure.
