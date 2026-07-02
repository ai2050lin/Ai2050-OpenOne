# Phase 853 Strong-edge Expansion and Natural Closure Validation (main)

- Source: new BF16 forward passes over expanded Phase 844/851 gear sets.
- Boundary: strong-edge expansion + natural closure audit, not final language closure.

## Expansion / Closure

| model | rows | interaction rows | strong rows | target in strong | exact natural in strong | classes | strong boundaries |
|---|---:|---:|---:|---:|---:|---|---|
| qwen3 | 1035 | 780 | 46 | 26 | 0 | `{"additive": 734, "synergy": 20, "antagonistic": 26}` | `{"broad_near_miss": 2, "target_equivalent": 26, "unknown_other": 15, "object_echo": 3}` |
| glm4 | 1035 | 780 | 1 | 0 | 0 | `{"additive": 779, "antagonistic": 1}` | `{"unknown_other": 1}` |
| deepseek7b | 1065 | 810 | 6 | 4 | 0 | `{"additive": 804, "synergy": 6}` | `{"target_equivalent": 4, "unknown_other": 2}` |

## Gate Holdout Summary

| model | predictor | in F1 | object F1 | prompt F1 | object balanced F1 | prompt balanced F1 |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `global_combo` | 0.4615 | 0.4078 | 0.4158 | 0.6087 | 0.6087 |
| qwen3 | `residual_projection_combo` | 0.8764 | 0.4719 | 0.4938 | 0.5915 | 0.6061 |
| qwen3 | `blocker_field_combo` | 0.8511 | 0.3448 | 0.3750 | 0.4839 | 0.4839 |
| qwen3 | `model_default_gate` | 0.8764 | 0.4719 | 0.4938 | 0.5915 | 0.6061 |
| qwen3 | `train_selected_gate` | 0.8846 | 0.4783 | 0.4151 | 0.6027 | 0.6111 |
| glm4 | `global_combo` | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| glm4 | `residual_projection_combo` | 0.0182 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| glm4 | `blocker_field_combo` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| glm4 | `model_default_gate` | 0.0182 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| glm4 | `train_selected_gate` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `global_combo` | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `residual_projection_combo` | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `blocker_field_combo` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `model_default_gate` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | `train_selected_gate` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
