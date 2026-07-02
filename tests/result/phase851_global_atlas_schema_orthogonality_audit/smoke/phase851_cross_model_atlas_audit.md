# Phase 851 Global Atlas Schema and Orthogonality Audit (smoke)

- Source: Phase 849 feature rows and Phase 850 strong-edge summaries.
- Boundary: schema audit and candidate ranking, not new forward-pass mechanism discovery.

## Gate Evidence

| model | rows | strong rows | predictor | evidence | in F1 | object F1 | prompt F1 | object balanced F1 | prompt balanced F1 |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| qwen3 | 12 | 1 | `global_combo` | `L3_in_sample_only` | 1.0000 | NA | NA | NA | NA |
| qwen3 | 12 | 1 | `residual_projection_combo` | `L3_in_sample_only` | 1.0000 | NA | NA | NA | NA |
| qwen3 | 12 | 1 | `blocker_field_combo` | `L3_in_sample_only` | 1.0000 | NA | NA | NA | NA |
| qwen3 | 12 | 1 | `model_default_gate` | `L3_in_sample_only` | 1.0000 | NA | NA | NA | NA |
| qwen3 | 12 | 1 | `train_selected_gate` | `L3_in_sample_only` | 1.0000 | NA | NA | NA | NA |
| glm4 | 12 | 0 | `global_combo` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |
| glm4 | 12 | 0 | `residual_projection_combo` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |
| glm4 | 12 | 0 | `blocker_field_combo` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |
| glm4 | 12 | 0 | `model_default_gate` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |
| glm4 | 12 | 0 | `train_selected_gate` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | 12 | 0 | `global_combo` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | 12 | 0 | `residual_projection_combo` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | 12 | 0 | `blocker_field_combo` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | 12 | 0 | `model_default_gate` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |
| deepseek7b | 12 | 0 | `train_selected_gate` | `L0_untriggered` | 0.0000 | NA | NA | NA | NA |

## Orthogonality Audit

| model | class | feature | object eta | prompt eta | mean |
|---|---|---|---:|---:|---:|

## Counterfactual Min-Cut Pre-Candidates

| model | gear | total | strong | strong rate | lift | status |
|---|---|---:|---:|---:|---:|---|
| qwen3 | `L30C2848` | 6 | 1 | 0.1667 | 2.0000 | `low_support` |
| qwen3 | `L27C2767` | 6 | 1 | 0.1667 | 2.0000 | `low_support` |
| qwen3 | `L29C1532` | 6 | 0 | 0.0000 | 0.0000 | `low_support` |
| qwen3 | `L30C1349` | 6 | 0 | 0.0000 | 0.0000 | `low_support` |
| glm4 | `L28C2777` | 6 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L30C6115` | 6 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L26C6031` | 6 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L28C8036` | 6 | 0 | 0.0000 | NA | `low_support` |
| deepseek7b | `L27C15791` | 6 | 0 | 0.0000 | NA | `low_support` |
| deepseek7b | `L27C1106` | 6 | 0 | 0.0000 | NA | `low_support` |
| deepseek7b | `L27C15305` | 6 | 0 | 0.0000 | NA | `low_support` |
| deepseek7b | `L25C4036` | 6 | 0 | 0.0000 | NA | `low_support` |

## Atlas Edges

| model | source | target | type | evidence | object F1 | prompt F1 |
|---|---|---|---|---|---:|---:|
| qwen3 | `qwen3:gate:global_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | NA | NA |
| qwen3 | `qwen3:gate:internal_strength_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | NA | NA |
| qwen3 | `qwen3:gate:residual_projection_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | NA | NA |
| qwen3 | `qwen3:gate:blocker_field_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | NA | NA |
| qwen3 | `qwen3:gate:route_competition_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | NA | NA |
| qwen3 | `qwen3:gate:compact_joint_gate_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | NA | NA |
| qwen3 | `qwen3:gate:joint_gate_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | NA | NA |
| qwen3 | `qwen3:gate:model_default_gate` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | NA | NA |
| qwen3 | `qwen3:gate:train_selected_gate` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | NA | NA |
