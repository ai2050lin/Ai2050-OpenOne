# Phase 851 Global Atlas Schema and Orthogonality Audit (main)

- Source: Phase 849 feature rows and Phase 850 strong-edge summaries.
- Boundary: schema audit and candidate ranking, not new forward-pass mechanism discovery.

## Gate Evidence

| model | rows | strong rows | predictor | evidence | in F1 | object F1 | prompt F1 | object balanced F1 | prompt balanced F1 |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| qwen3 | 304 | 32 | `global_combo` | `L3_in_sample_only` | 0.5938 | 0.4054 | 0.5357 | 0.5660 | 0.6000 |
| qwen3 | 304 | 32 | `residual_projection_combo` | `L4_partial_holdout_candidate` | 0.9552 | 0.4483 | 0.5556 | 0.5532 | 0.6250 |
| qwen3 | 304 | 32 | `blocker_field_combo` | `L3_in_sample_only` | 0.9412 | 0.3729 | 0.5172 | 0.4681 | 0.6000 |
| qwen3 | 304 | 32 | `model_default_gate` | `L4_partial_holdout_candidate` | 0.9552 | 0.4483 | 0.5556 | 0.5532 | 0.6250 |
| qwen3 | 304 | 32 | `train_selected_gate` | `L3_in_sample_only` | 1.0000 | 0.3793 | 0.5556 | 0.4889 | 0.6250 |
| glm4 | 304 | 0 | `global_combo` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| glm4 | 304 | 0 | `residual_projection_combo` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| glm4 | 304 | 0 | `blocker_field_combo` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| glm4 | 304 | 0 | `model_default_gate` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| glm4 | 304 | 0 | `train_selected_gate` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| deepseek7b | 304 | 3 | `global_combo` | `L2_weak_strong_edge_signal` | 0.2222 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | 304 | 3 | `residual_projection_combo` | `L3_in_sample_only` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | 304 | 3 | `blocker_field_combo` | `L3_in_sample_only` | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | 304 | 3 | `model_default_gate` | `L3_in_sample_only` | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | 304 | 3 | `train_selected_gate` | `L3_in_sample_only` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Orthogonality Audit

| model | class | feature | object eta | prompt eta | mean |
|---|---|---|---:|---:|---:|
| qwen3 | `protocol_like` | `topk_entropy` | 0.0519 | 0.8351 | 1.3917 |
| qwen3 | `protocol_like` | `best_target_rank` | 0.2149 | 0.6376 | 161.5000 |
| qwen3 | `protocol_like` | `resid_target_blocker_span` | 0.1949 | 0.6097 | 32.3546 |
| qwen3 | `semantic_like` | `best_target_object_resid_final` | 0.9104 | 0.0796 | 10.5356 |
| qwen3 | `semantic_like` | `object_minus_blocker_logit` | 0.8332 | 0.0276 | -16.7617 |
| qwen3 | `semantic_like` | `original_margin` | 0.7763 | 0.2158 | 3.8867 |
| qwen3 | `semantic_like` | `target_minus_object_logit` | 0.7763 | 0.2158 | 3.8867 |
| qwen3 | `semantic_like` | `object_echo_pressure` | 0.7763 | 0.2158 | -3.8867 |
| qwen3 | `entangled_or_shared` | `blocker_pressure` | 0.3833 | 0.4118 | 12.8750 |
| qwen3 | `entangled_or_shared` | `route_gap` | 0.3833 | 0.4118 | 12.8750 |
| qwen3 | `entangled_or_shared` | `target_minus_blocker_logit` | 0.3833 | 0.4118 | -12.8750 |
| qwen3 | `entangled_or_shared` | `resid_polygon_blocker_span` | 0.2635 | 0.1818 | 32.6684 |
| glm4 | `protocol_like` | `topk_entropy` | 0.1481 | 0.6185 | 2.5285 |
| glm4 | `protocol_like` | `abs_mean` | 0.0223 | 0.0713 | 2.3187 |
| glm4 | `protocol_like` | `abs_sum` | 0.0227 | 0.0650 | 5.1481 |
| glm4 | `protocol_like` | `max_abs` | 0.0213 | 0.0563 | 3.8924 |
| glm4 | `protocol_like` | `signed_mean` | 0.0246 | 0.0515 | 2.1981 |
| glm4 | `semantic_like` | `object_minus_blocker_logit` | 0.9361 | 0.0401 | -9.3799 |
| glm4 | `semantic_like` | `original_margin` | 0.9023 | 0.0083 | 1.6611 |
| glm4 | `semantic_like` | `target_minus_object_logit` | 0.9023 | 0.0083 | 1.6611 |
| glm4 | `semantic_like` | `object_echo_pressure` | 0.9023 | 0.0083 | -1.6611 |
| glm4 | `semantic_like` | `object_rank` | 0.8506 | 0.1247 | 2760.3750 |
| glm4 | `entangled_or_shared` | `blocker_pressure` | 0.2769 | 0.2138 | 7.7188 |
| glm4 | `entangled_or_shared` | `route_gap` | 0.2769 | 0.2138 | 7.7188 |
| glm4 | `entangled_or_shared` | `target_minus_blocker_logit` | 0.2769 | 0.2138 | -7.7188 |
| deepseek7b | `protocol_like` | `object_minus_blocker_logit` | 0.1933 | 0.7465 | -8.8008 |
| deepseek7b | `protocol_like` | `object_blocker_resid_final` | 0.0899 | 0.7362 | 149.0520 |
| deepseek7b | `protocol_like` | `topk_entropy` | 0.0642 | 0.6932 | 2.4128 |
| deepseek7b | `protocol_like` | `target_blocker_resid_final` | 0.1660 | 0.6689 | 31.1483 |
| deepseek7b | `protocol_like` | `resid_polygon_blocker_span` | 0.1785 | 0.6671 | 221.9915 |
| deepseek7b | `semantic_like` | `best_target_object_resid_final` | 0.8677 | 0.0729 | 29.2256 |
| deepseek7b | `semantic_like` | `original_margin` | 0.5276 | 0.3406 | -0.5234 |
| deepseek7b | `semantic_like` | `target_minus_object_logit` | 0.5276 | 0.3406 | -0.5234 |
| deepseek7b | `semantic_like` | `object_echo_pressure` | 0.5276 | 0.3406 | 0.5234 |
| deepseek7b | `semantic_like` | `expected_additive_delta` | 0.2948 | 0.0007 | 0.0446 |
| deepseek7b | `entangled_or_shared` | `blocker_pressure` | 0.3902 | 0.5249 | 9.3242 |
| deepseek7b | `entangled_or_shared` | `route_gap` | 0.3902 | 0.5249 | 9.3242 |
| deepseek7b | `entangled_or_shared` | `target_minus_blocker_logit` | 0.3902 | 0.5249 | -9.3242 |
| deepseek7b | `entangled_or_shared` | `best_target_rank` | 0.4220 | 0.4076 | 884.6250 |
| deepseek7b | `entangled_or_shared` | `actual_residual` | 0.1120 | 0.1453 | 0.0324 |

## Counterfactual Min-Cut Pre-Candidates

| model | gear | total | strong | strong rate | lift | status |
|---|---|---:|---:|---:|---:|---|
| qwen3 | `L29C1532` | 144 | 28 | 0.1944 | 1.6540 | `counterfactual_min_cut_candidate` |
| qwen3 | `L27C2767` | 96 | 18 | 0.1875 | 1.5949 | `counterfactual_min_cut_candidate` |
| qwen3 | `L30C2848` | 144 | 17 | 0.1181 | 1.0042 | `weak_candidate` |
| qwen3 | `L30C1349` | 96 | 7 | 0.0729 | 0.6203 | `weak_candidate` |
| qwen3 | `L30C5558` | 96 | 5 | 0.0521 | 0.4430 | `weak_candidate` |
| qwen3 | `L29C4588` | 96 | 4 | 0.0417 | 0.3544 | `weak_candidate` |
| glm4 | `L28C2777` | 144 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L30C6115` | 144 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L26C6031` | 96 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L28C8036` | 96 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L29C10031` | 96 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L27C10905` | 96 | 0 | 0.0000 | NA | `low_support` |
| deepseek7b | `L27C2295` | 96 | 3 | 0.0312 | 3.0000 | `counterfactual_min_cut_candidate` |
| deepseek7b | `L27C15791` | 144 | 2 | 0.0139 | 1.3333 | `weak_candidate` |
| deepseek7b | `L27C13360` | 96 | 1 | 0.0104 | 1.0000 | `weak_candidate` |
| deepseek7b | `L27C1106` | 144 | 1 | 0.0069 | 0.6667 | `weak_candidate` |
| deepseek7b | `L27C15305` | 96 | 0 | 0.0000 | 0.0000 | `low_support` |
| deepseek7b | `L25C4036` | 96 | 0 | 0.0000 | 0.0000 | `low_support` |

## Atlas Edges

| model | source | target | type | evidence | object F1 | prompt F1 |
|---|---|---|---|---|---:|---:|
| qwen3 | `qwen3:gate:global_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.4054 | 0.5357 |
| qwen3 | `qwen3:gate:internal_strength_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L5_strong_edge_holdout_candidate` | 0.4839 | 0.6076 |
| qwen3 | `qwen3:gate:residual_projection_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L4_partial_holdout_candidate` | 0.4483 | 0.5556 |
| qwen3 | `qwen3:gate:blocker_field_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.3729 | 0.5172 |
| qwen3 | `qwen3:gate:route_competition_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.4000 | 0.4304 |
| qwen3 | `qwen3:gate:compact_joint_gate_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L5_strong_edge_holdout_candidate` | 0.4688 | 0.6076 |
| qwen3 | `qwen3:gate:joint_gate_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L4_partial_holdout_candidate` | 0.4194 | 0.6076 |
| qwen3 | `qwen3:gate:model_default_gate` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L4_partial_holdout_candidate` | 0.4483 | 0.5556 |
| qwen3 | `qwen3:gate:train_selected_gate` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.3793 | 0.5556 |
| deepseek7b | `deepseek7b:gate:residual_projection_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:blocker_field_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:route_competition_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:compact_joint_gate_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:joint_gate_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:model_default_gate` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:train_selected_gate` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
