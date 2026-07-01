# Phase 835 Span-Aware / Protocol-Aware Blocker Profile (main)

- Source: Phase 834 first-token blocker profile plus Phase 816 answer-span scoring.
- Objective: test whether span margins explain cases where first-token rank fails.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_count_rank_improved` | 0.917 | 17 | 16 | 2 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 3, "1": 11, "2": 4}` |
| qwen3 | `category_count_span_closure` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `category_count_span_contrast_cleared` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `category_count_span_generic_cleared` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `category_count_span_margin_improved` | 0.750 | 17 | 10 | 8 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 2, "1": 7, "2": 9}` |
| qwen3 | `category_count_span_or_rank_improved` | 0.750 | 17 | 10 | 8 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 2, "1": 7, "2": 9}` |
| qwen3 | `category_count_span_rank_improved` | 0.500 | 0 | 18 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | `{"0": 18}` |
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.722 | 17 | 9 | 9 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 1, "1": 8, "2": 9}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 18 | 18 | 0 | 0 | 2 | 8 | 0 | 2 | 0 | `{"0": 4, "1": 10, "2": 4}` |
| glm4 | `category_count_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_closure` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_contrast_cleared` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_generic_cleared` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_margin_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_or_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_count_rank_improved` | 0.500 | 2 | 4 | 1 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 3}` |
| deepseek7b | `category_count_span_closure` | 0.417 | 1 | 4 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 2}` |
| deepseek7b | `category_count_span_contrast_cleared` | 0.917 | 7 | 4 | 1 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `category_count_span_generic_cleared` | 0.417 | 1 | 4 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 2}` |
| deepseek7b | `category_count_span_margin_improved` | 0.417 | 1 | 4 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 2}` |
| deepseek7b | `category_count_span_or_rank_improved` | 0.500 | 2 | 4 | 1 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 3}` |
| deepseek7b | `category_count_span_rank_improved` | 0.417 | 1 | 4 | 1 | 6 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 2}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.917 | 7 | 4 | 1 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 7 | 5 | 0 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 5, "2": 1}` |

## Boundary

Span-aware features are still target-aware probes. The phase is successful only if they explain DS7B without breaking qwen3.
