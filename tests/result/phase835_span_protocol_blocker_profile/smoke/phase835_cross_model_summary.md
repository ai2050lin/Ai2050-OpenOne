# Phase 835 Span-Aware / Protocol-Aware Blocker Profile (smoke)

- Source: Phase 834 first-token blocker profile plus Phase 816 answer-span scoring.
- Objective: test whether span margins explain cases where first-token rank fails.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `category_count_rank_improved` | 1.000 | 5 | 7 | 0 | 0 | 2 | 2 | 0 | 2 | 0 | `{"0": 1, "1": 5}` |
| qwen3 | `category_count_span_margin_improved` | 0.750 | 5 | 4 | 3 | 0 | 2 | 2 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| qwen3 | `category_count_span_or_rank_improved` | 0.750 | 5 | 4 | 3 | 0 | 2 | 2 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| qwen3 | `category_nonresidual_else_count_nonnegative` | 0.750 | 5 | 4 | 3 | 0 | 2 | 2 | 0 | 2 | 0 | `{"1": 4, "2": 2}` |
| qwen3 | `oracle_route_target_only` | 1.000 | 5 | 7 | 0 | 0 | 2 | 2 | 0 | 2 | 0 | `{"0": 1, "1": 5}` |
| glm4 | `category_count_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_margin_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_count_span_or_rank_improved` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `category_nonresidual_else_count_nonnegative` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `oracle_route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `category_count_rank_improved` | 0.250 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 1, "1": 1}` |
| deepseek7b | `category_count_span_margin_improved` | 0.500 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 2}` |
| deepseek7b | `category_count_span_or_rank_improved` | 0.250 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 1, "1": 1}` |
| deepseek7b | `category_nonresidual_else_count_nonnegative` | 0.750 | 2 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 1, "2": 1}` |
| deepseek7b | `oracle_route_target_only` | 1.000 | 2 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 2}` |

## Boundary

Span-aware features are still target-aware probes. The phase is successful only if they explain DS7B without breaking qwen3.
