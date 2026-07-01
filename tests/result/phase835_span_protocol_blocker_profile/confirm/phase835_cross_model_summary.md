# Phase 835 Span-Aware / Protocol-Aware Blocker Profile (confirm)

- Source: Phase 834 first-token blocker profile plus Phase 816 answer-span scoring.
- Objective: test whether span margins explain cases where first-token rank fails.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_count_rank_improved` | 0.896 | 23 | 20 | 4 | 1 | 6 | 14 | 0 | 2 | 0 | `{"0": 3, "1": 15, "2": 6}` |
| qwen3 | `category_count_span_and_rank_improved` | 0.896 | 23 | 20 | 4 | 1 | 6 | 14 | 0 | 2 | 0 | `{"0": 3, "1": 15, "2": 6}` |
| qwen3 | `category_count_span_closure` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `category_count_span_contrast_cleared` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `category_count_span_generic_cleared` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `category_count_span_margin_improved` | 0.750 | 23 | 13 | 11 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 2, "1": 10, "2": 12}` |
| qwen3 | `category_count_span_or_rank_improved` | 0.750 | 23 | 13 | 11 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 2, "1": 10, "2": 12}` |
| qwen3 | `category_count_span_rank_improved` | 0.500 | 0 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | `{"0": 24}` |
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.708 | 23 | 11 | 13 | 1 | 6 | 13 | 1 | 2 | 0 | `{"0": 1, "1": 10, "2": 13}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 24 | 24 | 0 | 0 | 6 | 14 | 0 | 2 | 0 | `{"0": 4, "1": 16, "2": 4}` |
| glm4 | `category_count_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_and_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_closure` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_contrast_cleared` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_generic_cleared` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_margin_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_or_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_count_rank_improved` | 0.375 | 2 | 4 | 3 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 5}` |
| deepseek7b | `category_count_span_and_rank_improved` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_count_span_closure` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_count_span_contrast_cleared` | 0.812 | 9 | 4 | 3 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 4, "2": 4}` |
| deepseek7b | `category_count_span_generic_cleared` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_count_span_margin_improved` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_count_span_or_rank_improved` | 0.375 | 2 | 4 | 3 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 3, "1": 5}` |
| deepseek7b | `category_count_span_rank_improved` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.812 | 9 | 4 | 3 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 4, "2": 4}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 9 | 7 | 0 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 7, "2": 1}` |

## Boundary

Span-aware features are still target-aware probes. The phase is successful only if they explain DS7B without breaking qwen3.
