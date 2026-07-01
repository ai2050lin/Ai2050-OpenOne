# Phase 831 Internal Gate Predictor Search (smoke)

- Source: Phase 829 pairs and Phase 830 target-only gates.
- Objective: test whether simple internal readout-alignment signals can predict useful donor gates.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `positive_count_majority` | 0.583 | 5 | 2 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | `{"1": 2, "2": 4}` |
| qwen3 | `signed_sum_positive` | 0.583 | 4 | 3 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | `{"1": 4, "2": 2}` |
| qwen3 | `top_abs_positive` | 0.583 | 4 | 3 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | `{"1": 4, "2": 2}` |
| glm4 | `positive_count_majority` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `signed_sum_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `top_abs_positive` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `positive_count_majority` | 0.250 | 1 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | `{"1": 1, "2": 1}` |
| deepseek7b | `signed_sum_positive` | 0.250 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 1, "1": 1}` |
| deepseek7b | `top_abs_positive` | 0.250 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 1, "1": 1}` |

## Boundary

This is not a learned classifier; it is a basic readout-alignment probe. It is useful only if the same signal both approximates the Phase 830 gate and preserves generation quality.
