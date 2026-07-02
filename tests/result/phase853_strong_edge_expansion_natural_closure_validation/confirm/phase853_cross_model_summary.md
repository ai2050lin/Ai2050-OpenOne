# Phase 853 Strong-edge Expansion and Natural Closure Validation (confirm)

- Source: new BF16 forward passes over expanded Phase 844/851 gear sets.
- Boundary: strong-edge expansion + natural closure audit, not final language closure.

## Expansion / Closure

| model | rows | interaction rows | strong rows | target in strong | exact natural in strong | classes | strong boundaries |
|---|---:|---:|---:|---:|---:|---|---|
| qwen3 | 1395 | 1080 | 62 | 38 | 0 | `{"additive": 1018, "synergy": 33, "antagonistic": 29}` | `{"broad_near_miss": 2, "target_equivalent": 38, "unknown_other": 19, "object_echo": 3}` |
| glm4 | 1395 | 1080 | 1 | 0 | 0 | `{"additive": 1079, "antagonistic": 1}` | `{"unknown_other": 1}` |
| deepseek7b | 1425 | 1110 | 8 | 5 | 0 | `{"additive": 1102, "synergy": 8}` | `{"target_equivalent": 5, "unknown_other": 3}` |

## Gate Holdout Summary

| model | predictor | in F1 | object F1 | prompt F1 | object balanced F1 | prompt balanced F1 |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `global_combo` | 0.4098 | 0.3151 | 0.3684 | 0.5227 | 0.6022 |
| qwen3 | `residual_projection_combo` | 0.8500 | 0.4167 | 0.4615 | 0.5376 | 0.5581 |
| qwen3 | `blocker_field_combo` | 0.8462 | 0.3276 | 0.3654 | 0.4634 | 0.4634 |
| qwen3 | `model_default_gate` | 0.8500 | 0.4167 | 0.4615 | 0.5376 | 0.5581 |
| qwen3 | `train_selected_gate` | 0.8986 | 0.4466 | 0.4133 | 0.5227 | 0.6263 |
| glm4 | `global_combo` | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| glm4 | `residual_projection_combo` | 0.0133 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| glm4 | `blocker_field_combo` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| glm4 | `model_default_gate` | 0.0133 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| glm4 | `train_selected_gate` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `global_combo` | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `residual_projection_combo` | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `blocker_field_combo` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `model_default_gate` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `train_selected_gate` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
