# Phase 831 Internal Gate Predictor Search (main)

- Source: Phase 829 pairs and Phase 830 target-only gates.
- Objective: test whether simple internal readout-alignment signals can predict useful donor gates.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `positive_count_majority` | 0.639 | 17 | 6 | 12 | 1 | 0 | 6 | 0 | 0 | 0 | `{"1": 7, "2": 11}` |
| qwen3 | `signed_sum_positive` | 0.611 | 16 | 6 | 12 | 2 | 0 | 6 | 0 | 0 | 0 | `{"1": 8, "2": 10}` |
| qwen3 | `top_abs_positive` | 0.611 | 16 | 6 | 12 | 2 | 0 | 6 | 0 | 0 | 0 | `{"1": 8, "2": 10}` |
| glm4 | `positive_count_majority` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `signed_sum_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `top_abs_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `positive_count_majority` | 0.750 | 6 | 3 | 2 | 1 | 1 | 3 | 0 | 1 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `signed_sum_positive` | 0.333 | 2 | 2 | 3 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 3, "2": 1}` |
| deepseek7b | `top_abs_positive` | 0.250 | 1 | 2 | 3 | 6 | 0 | 0 | 0 | 0 | 0 | `{"0": 2, "1": 4}` |

## Boundary

This is not a learned classifier; it is a basic readout-alignment probe. It is useful only if the same signal both approximates the Phase 830 gate and preserves generation quality.
