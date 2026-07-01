# Phase 832 Gate Signal Expansion Beyond Readout (smoke)

- Source: Phase 829 pairs, Phase 831 readout features, and component donor-route classes.
- Objective: compare pure readout gates with route-boundary augmented gates.

## Model Summary

| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `readout_count` | 0.583 | 5 | 2 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | `{"1": 2, "2": 4}` |
| qwen3 | `readout_signed_sum` | 0.583 | 4 | 3 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | `{"1": 4, "2": 2}` |
| qwen3 | `route_target_else_count` | 0.917 | 5 | 6 | 1 | 0 | 2 | 2 | 0 | 2 | 0 | `{"1": 6}` |
| qwen3 | `route_target_else_signed_sum` | 0.917 | 5 | 6 | 1 | 0 | 2 | 2 | 0 | 2 | 0 | `{"1": 6}` |
| qwen3 | `route_target_only` | 1.000 | 5 | 7 | 0 | 0 | 2 | 2 | 0 | 2 | 0 | `{"0": 1, "1": 5}` |
| glm4 | `readout_count` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `readout_signed_sum` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_target_else_count` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_target_else_signed_sum` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| glm4 | `route_target_only` |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `{}` |
| deepseek7b | `readout_count` | 0.250 | 1 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | `{"1": 1, "2": 1}` |
| deepseek7b | `readout_signed_sum` | 0.250 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | `{"0": 1, "1": 1}` |
| deepseek7b | `route_target_else_count` | 1.000 | 2 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 2}` |
| deepseek7b | `route_target_else_signed_sum` | 1.000 | 2 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 2}` |
| deepseek7b | `route_target_only` | 1.000 | 2 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | `{"1": 2}` |

## Boundary

Route-boundary augmented gates may restore missing natural routes, but they use single-component behavioral boundary classes. They are therefore a diagnostic bridge, not a fully internal gate mechanism.
