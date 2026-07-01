# Phase 832 Gate Signal Expansion Beyond Readout (confirm)

- Source: Phase 829 pairs, Phase 831 readout features, and component donor-route classes.
- Objective: compare pure readout gates with route-boundary augmented gates.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `readout_count` | 0.688 | 23 | 10 | 14 | 1 | 5 | 11 | 1 | 0 | 0 | `{"1": 11, "2": 13}` |
| qwen3 | `readout_signed_sum` | 0.667 | 22 | 10 | 14 | 2 | 6 | 12 | 0 | 0 | 0 | `{"1": 12, "2": 12}` |
| qwen3 | `route_clean_count` | 0.979 | 23 | 24 | 0 | 1 | 6 | 14 | 0 | 2 | 0 | `{"0": 4, "1": 17, "2": 3}` |
| qwen3 | `route_clean_signed_sum` | 0.958 | 22 | 24 | 0 | 2 | 6 | 13 | 0 | 1 | 0 | `{"0": 5, "1": 16, "2": 3}` |
| qwen3 | `route_target_else_count` | 0.896 | 24 | 19 | 5 | 0 | 6 | 14 | 0 | 2 | 0 | `{"1": 19, "2": 5}` |
| qwen3 | `route_target_else_signed_sum` | 0.896 | 24 | 19 | 5 | 0 | 6 | 14 | 0 | 2 | 0 | `{"1": 19, "2": 5}` |
| qwen3 | `route_target_only` | 1.000 | 24 | 24 | 0 | 0 | 6 | 14 | 0 | 2 | 0 | `{"0": 4, "1": 16, "2": 4}` |
| glm4 | `readout_count` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `readout_signed_sum` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_clean_count` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_clean_signed_sum` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_target_else_count` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_target_else_signed_sum` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `readout_count` | 0.688 | 8 | 3 | 4 | 1 | 2 | 5 | 0 | 1 | 0 | `{"1": 4, "2": 4}` |
| deepseek7b | `readout_signed_sum` | 0.250 | 2 | 2 | 5 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 2, "1": 5, "2": 1}` |
| deepseek7b | `route_clean_count` | 0.938 | 8 | 7 | 0 | 1 | 2 | 5 | 0 | 1 | 0 | `{"0": 1, "1": 6, "2": 1}` |
| deepseek7b | `route_clean_signed_sum` | 0.562 | 2 | 7 | 0 | 7 | 0 | 1 | 0 | 0 | 0 | `{"0": 6, "1": 2}` |
| deepseek7b | `route_target_else_count` | 1.000 | 9 | 7 | 0 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 7, "2": 1}` |
| deepseek7b | `route_target_else_signed_sum` | 1.000 | 9 | 7 | 0 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 7, "2": 1}` |
| deepseek7b | `route_target_only` | 1.000 | 9 | 7 | 0 | 0 | 2 | 6 | 0 | 2 | 0 | `{"1": 7, "2": 1}` |

## Boundary

Route-boundary augmented gates may restore missing natural routes, but they use single-component behavioral boundary classes. They are therefore a diagnostic bridge, not a fully internal gate mechanism.
