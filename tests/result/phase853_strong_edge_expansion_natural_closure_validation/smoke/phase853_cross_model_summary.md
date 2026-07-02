# Phase 853 Strong-edge Expansion and Natural Closure Validation (smoke)

- Source: new BF16 forward passes over expanded Phase 844/851 gear sets.
- Boundary: strong-edge expansion + natural closure audit, not final language closure.

## Expansion / Closure

| model | rows | interaction rows | strong rows | target in strong | exact natural in strong | classes | strong boundaries |
|---|---:|---:|---:|---:|---:|---|---|
| qwen3 | 23 | 14 | 2 | 0 | 0 | `{"additive": 12, "synergy": 2}` | `{"broad_near_miss": 2}` |
| glm4 | 21 | 12 | 0 | 0 | 0 | `{"additive": 12}` | `{}` |
| deepseek7b | 23 | 12 | 0 | 0 | 0 | `{"additive": 12}` | `{}` |

## Gate Holdout Summary

| model | predictor | in F1 | object F1 | prompt F1 | object balanced F1 | prompt balanced F1 |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `global_combo` | 1.0000 | NA | NA | NA | NA |
| qwen3 | `residual_projection_combo` | 1.0000 | NA | NA | NA | NA |
| qwen3 | `blocker_field_combo` | 1.0000 | NA | NA | NA | NA |
| qwen3 | `model_default_gate` | 1.0000 | NA | NA | NA | NA |
| qwen3 | `train_selected_gate` | 1.0000 | NA | NA | NA | NA |
| glm4 | `global_combo` | 0.0000 | NA | NA | NA | NA |
| glm4 | `residual_projection_combo` | 0.0000 | NA | NA | NA | NA |
| glm4 | `blocker_field_combo` | 0.0000 | NA | NA | NA | NA |
| glm4 | `model_default_gate` | 0.0000 | NA | NA | NA | NA |
| glm4 | `train_selected_gate` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | `global_combo` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | `residual_projection_combo` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | `blocker_field_combo` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | `model_default_gate` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | `train_selected_gate` | 0.0000 | NA | NA | NA | NA |
