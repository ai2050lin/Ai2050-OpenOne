# Phase 832 Gate Signal Expansion Beyond Readout (main)

- Source: Phase 829 pairs, Phase 831 readout features, and component donor-route classes.
- Objective: compare pure readout gates with route-boundary augmented gates.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `readout_count` | 0.639 | 17 | 6 | 12 | 1 | 0 | 6 | 0 | 0 | 0 | `{"1": 7, "2": 11}` |
| qwen3 | `readout_signed_sum` | 0.611 | 16 | 6 | 12 | 2 | 0 | 6 | 0 | 0 | 0 | `{"1": 8, "2": 10}` |
| qwen3 | `route_clean_count` | 0.972 | 17 | 18 | 0 | 1 | 2 | 8 | 0 | 2 | 0 | `{"0": 4, "1": 11, "2": 3}` |
| qwen3 | `route_clean_signed_sum` | 0.944 | 16 | 18 | 0 | 2 | 1 | 7 | 0 | 1 | 0 | `{"0": 5, "1": 10, "2": 3}` |
| qwen3 | `route_target_else_count` | 0.861 | 18 | 13 | 5 | 0 | 2 | 8 | 0 | 2 | 0 | `{"1": 13, "2": 5}` |
| qwen3 | `route_target_else_signed_sum` | 0.861 | 18 | 13 | 5 | 0 | 2 | 8 | 0 | 2 | 0 | `{"1": 13, "2": 5}` |
| qwen3 | `route_target_only` | 1.000 | 18 | 18 | 0 | 0 | 2 | 8 | 0 | 2 | 0 | `{"0": 4, "1": 10, "2": 4}` |
| glm4 | `readout_count` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `readout_signed_sum` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_clean_count` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_clean_signed_sum` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_target_else_count` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_target_else_signed_sum` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `readout_count` | 0.750 | 6 | 3 | 2 | 1 | 1 | 3 | 0 | 1 | 0 | `{"1": 4, "2": 2}` |
| deepseek7b | `readout_signed_sum` | 0.333 | 2 | 2 | 3 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 3, "2": 1}` |
| deepseek7b | `route_clean_count` | 0.917 | 6 | 5 | 0 | 1 | 1 | 3 | 0 | 1 | 0 | `{"0": 1, "1": 4, "2": 1}` |
| deepseek7b | `route_clean_signed_sum` | 0.583 | 2 | 5 | 0 | 5 | 0 | 1 | 0 | 0 | 0 | `{"0": 4, "1": 2}` |
| deepseek7b | `route_target_else_count` | 1.000 | 7 | 5 | 0 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 5, "2": 1}` |
| deepseek7b | `route_target_else_signed_sum` | 1.000 | 7 | 5 | 0 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 5, "2": 1}` |
| deepseek7b | `route_target_only` | 1.000 | 7 | 5 | 0 | 0 | 2 | 4 | 0 | 2 | 0 | `{"1": 5, "2": 1}` |

## Boundary

Route-boundary augmented gates may restore missing natural routes, but they use single-component behavioral boundary classes. They are therefore a diagnostic bridge, not a fully internal gate mechanism.
