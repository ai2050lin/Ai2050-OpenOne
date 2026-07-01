# Phase 831 Internal Gate Predictor Search (confirm)

- Source: Phase 829 pairs and Phase 830 target-only gates.
- Objective: test whether simple internal readout-alignment signals can predict useful donor gates.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `positive_count_majority` | 0.688 | 23 | 10 | 14 | 1 | 5 | 11 | 1 | 0 | 0 | `{"1": 11, "2": 13}` |
| qwen3 | `signed_sum_positive` | 0.667 | 22 | 10 | 14 | 2 | 6 | 12 | 0 | 0 | 0 | `{"1": 12, "2": 12}` |
| qwen3 | `top_abs_positive` | 0.667 | 22 | 10 | 14 | 2 | 6 | 12 | 0 | 0 | 0 | `{"1": 12, "2": 12}` |
| glm4 | `positive_count_majority` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `signed_sum_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `top_abs_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `positive_count_majority` | 0.688 | 8 | 3 | 4 | 1 | 2 | 5 | 0 | 1 | 0 | `{"1": 4, "2": 4}` |
| deepseek7b | `signed_sum_positive` | 0.250 | 2 | 2 | 5 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 5, "2": 1}` |
| deepseek7b | `top_abs_positive` | 0.312 | 1 | 4 | 3 | 8 | 0 | 0 | 0 | 0 | 0 | `{"0": 4, "1": 4}` |

## Boundary

This is not a learned classifier; it is a basic readout-alignment probe. It is useful only if the same signal both approximates the Phase 830 gate and preserves generation quality.
