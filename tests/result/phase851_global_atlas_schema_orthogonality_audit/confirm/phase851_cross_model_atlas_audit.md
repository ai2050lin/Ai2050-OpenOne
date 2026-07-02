# Phase 851 Global Atlas Schema and Orthogonality Audit (confirm)

- Source: Phase 849 feature rows and Phase 850 strong-edge summaries.
- Boundary: schema audit and candidate ranking, not new forward-pass mechanism discovery.

## Gate Evidence

| model | rows | strong rows | predictor | evidence | in F1 | object F1 | prompt F1 | object balanced F1 | prompt balanced F1 |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| qwen3 | 570 | 34 | `global_combo` | `L2_weak_strong_edge_signal` | 0.4688 | 0.3947 | 0.4054 | 0.5882 | 0.5882 |
| qwen3 | 570 | 34 | `residual_projection_combo` | `L5_strong_edge_holdout_candidate` | 0.8923 | 0.5312 | 0.4839 | 0.6296 | 0.6122 |
| qwen3 | 570 | 34 | `blocker_field_combo` | `L4_partial_holdout_candidate` | 0.8657 | 0.4000 | 0.4333 | 0.5417 | 0.5417 |
| qwen3 | 570 | 34 | `model_default_gate` | `L5_strong_edge_holdout_candidate` | 0.8923 | 0.5312 | 0.4839 | 0.6296 | 0.6122 |
| qwen3 | 570 | 34 | `train_selected_gate` | `L4_partial_holdout_candidate` | 0.8923 | 0.5000 | 0.4051 | 0.6038 | 0.6038 |
| glm4 | 570 | 0 | `global_combo` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| glm4 | 570 | 0 | `residual_projection_combo` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| glm4 | 570 | 0 | `blocker_field_combo` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| glm4 | 570 | 0 | `model_default_gate` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| glm4 | 570 | 0 | `train_selected_gate` | `L0_untriggered` | 0.0000 | 0.0000 | 0.0000 | NA | NA |
| deepseek7b | 570 | 3 | `global_combo` | `L2_weak_strong_edge_signal` | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | 570 | 3 | `residual_projection_combo` | `L3_in_sample_only` | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | 570 | 3 | `blocker_field_combo` | `L3_in_sample_only` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | 570 | 3 | `model_default_gate` | `L3_in_sample_only` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deepseek7b | 570 | 3 | `train_selected_gate` | `L3_in_sample_only` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Orthogonality Audit

| model | class | feature | object eta | prompt eta | mean |
|---|---|---|---:|---:|---:|
| qwen3 | `protocol_like` | `resid_target_blocker_span` | 0.0961 | 0.8284 | 26.7492 |
| qwen3 | `protocol_like` | `topk_entropy` | 0.0478 | 0.8008 | 1.4083 |
| qwen3 | `protocol_like` | `best_target_object_resid_final` | 0.1432 | 0.6394 | 0.6501 |
| qwen3 | `protocol_like` | `original_margin` | 0.1803 | 0.5513 | 2.0271 |
| qwen3 | `protocol_like` | `target_minus_object_logit` | 0.1803 | 0.5513 | 2.0271 |
| qwen3 | `semantic_like` | `best_target_blocker_resid_final` | 0.4819 | 0.1769 | -6.1895 |
| qwen3 | `semantic_like` | `neg_count` | 0.1126 | 0.0039 | 1.1789 |
| qwen3 | `semantic_like` | `signed_sum` | 0.0878 | 0.0053 | -0.1093 |
| qwen3 | `semantic_like` | `signed_mean` | 0.0813 | 0.0031 | -0.4068 |
| qwen3 | `semantic_like` | `expected_additive_delta` | 0.0807 | 0.0335 | -0.7215 |
| qwen3 | `entangled_or_shared` | `blocker_pressure` | 0.3428 | 0.3550 | 12.7458 |
| qwen3 | `entangled_or_shared` | `route_gap` | 0.3428 | 0.3550 | 12.7458 |
| qwen3 | `entangled_or_shared` | `target_minus_blocker_logit` | 0.3428 | 0.3550 | -12.7458 |
| qwen3 | `entangled_or_shared` | `object_rank` | 0.3018 | 0.2282 | 1746.6000 |
| qwen3 | `entangled_or_shared` | `resid_polygon_blocker_span` | 0.2734 | 0.2880 | 34.8293 |
| glm4 | `protocol_like` | `target_blocker_resid_final` | 0.0162 | 0.9614 | -9.7204 |
| glm4 | `protocol_like` | `object_blocker_resid_final` | 0.0301 | 0.9579 | 4.5572 |
| glm4 | `protocol_like` | `best_target_blocker_resid_final` | 0.0580 | 0.9016 | 5.9103 |
| glm4 | `protocol_like` | `resid_polygon_blocker_span` | 0.0224 | 0.8786 | 12.2941 |
| glm4 | `protocol_like` | `resid_target_blocker_span` | 0.1255 | 0.7374 | 16.8662 |
| glm4 | `semantic_like` | `object_minus_blocker_logit` | 0.5931 | 0.1615 | -9.2057 |
| glm4 | `semantic_like` | `object_rank` | 0.4569 | 0.2576 | 2144.0000 |
| glm4 | `semantic_like` | `min_abs` | 0.1428 | 0.0254 | 0.9200 |
| glm4 | `semantic_like` | `actual_residual` | 0.1424 | 0.0474 | 0.0056 |
| glm4 | `semantic_like` | `expected_additive_delta` | 0.1258 | 0.0496 | 0.0901 |
| glm4 | `entangled_or_shared` | `abs_sum` | 0.0609 | 0.0492 | 5.2971 |
| glm4 | `entangled_or_shared` | `abs_mean` | 0.0607 | 0.0537 | 2.3821 |
| deepseek7b | `protocol_like` | `object_minus_blocker_logit` | 0.1956 | 0.7163 | -8.8187 |
| deepseek7b | `protocol_like` | `blocker_pressure` | 0.3210 | 0.5705 | 9.7000 |
| deepseek7b | `protocol_like` | `route_gap` | 0.3210 | 0.5705 | 9.7000 |
| deepseek7b | `protocol_like` | `target_minus_blocker_logit` | 0.3210 | 0.5705 | -9.7000 |
| deepseek7b | `protocol_like` | `object_rank` | 0.2420 | 0.5409 | 913.0667 |
| deepseek7b | `semantic_like` | `best_target_object_resid_final` | 0.8508 | 0.0724 | 19.7024 |
| deepseek7b | `semantic_like` | `actual_delta` | 0.2628 | 0.0116 | 0.0889 |
| deepseek7b | `semantic_like` | `expected_additive_delta` | 0.2452 | 0.0117 | 0.0695 |
| deepseek7b | `entangled_or_shared` | `original_margin` | 0.4568 | 0.3605 | -0.8812 |
| deepseek7b | `entangled_or_shared` | `target_minus_object_logit` | 0.4568 | 0.3605 | -0.8812 |
| deepseek7b | `entangled_or_shared` | `object_echo_pressure` | 0.4568 | 0.3605 | 0.8812 |
| deepseek7b | `entangled_or_shared` | `resid_polygon_blocker_span` | 0.2995 | 0.3924 | 173.4350 |
| deepseek7b | `entangled_or_shared` | `target_blocker_resid_final` | 0.3545 | 0.3490 | -20.9379 |

## Counterfactual Min-Cut Pre-Candidates

| model | gear | total | strong | strong rate | lift | status |
|---|---|---:|---:|---:|---:|---|
| qwen3 | `L29C1532` | 270 | 30 | 0.1111 | 1.6667 | `counterfactual_min_cut_candidate` |
| qwen3 | `L27C2767` | 180 | 20 | 0.1111 | 1.6667 | `counterfactual_min_cut_candidate` |
| qwen3 | `L30C2848` | 270 | 18 | 0.0667 | 1.0000 | `weak_candidate` |
| qwen3 | `L30C1349` | 180 | 7 | 0.0389 | 0.5833 | `weak_candidate` |
| qwen3 | `L30C5558` | 180 | 5 | 0.0278 | 0.4167 | `weak_candidate` |
| qwen3 | `L29C4588` | 180 | 4 | 0.0222 | 0.3333 | `weak_candidate` |
| glm4 | `L28C2777` | 270 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L30C6115` | 270 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L26C6031` | 180 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L28C8036` | 180 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L29C10031` | 180 | 0 | 0.0000 | NA | `low_support` |
| glm4 | `L27C10905` | 180 | 0 | 0.0000 | NA | `low_support` |
| deepseek7b | `L27C2295` | 180 | 3 | 0.0167 | 3.0000 | `counterfactual_min_cut_candidate` |
| deepseek7b | `L27C15791` | 270 | 2 | 0.0074 | 1.3333 | `weak_candidate` |
| deepseek7b | `L27C13360` | 180 | 1 | 0.0056 | 1.0000 | `weak_candidate` |
| deepseek7b | `L27C1106` | 270 | 1 | 0.0037 | 0.6667 | `weak_candidate` |
| deepseek7b | `L27C15305` | 180 | 0 | 0.0000 | 0.0000 | `low_support` |
| deepseek7b | `L25C4036` | 180 | 0 | 0.0000 | 0.0000 | `low_support` |

## Atlas Edges

| model | source | target | type | evidence | object F1 | prompt F1 |
|---|---|---|---|---|---:|---:|
| qwen3 | `qwen3:gate:internal_strength_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L5_strong_edge_holdout_candidate` | 0.4051 | 0.4255 |
| qwen3 | `qwen3:gate:residual_projection_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L5_strong_edge_holdout_candidate` | 0.5312 | 0.4839 |
| qwen3 | `qwen3:gate:blocker_field_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L4_partial_holdout_candidate` | 0.4000 | 0.4333 |
| qwen3 | `qwen3:gate:route_competition_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.2632 | 0.0317 |
| qwen3 | `qwen3:gate:compact_joint_gate_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L4_partial_holdout_candidate` | 0.4211 | 0.4783 |
| qwen3 | `qwen3:gate:joint_gate_combo` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L4_partial_holdout_candidate` | 0.4590 | 0.4490 |
| qwen3 | `qwen3:gate:model_default_gate` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L5_strong_edge_holdout_candidate` | 0.5312 | 0.4839 |
| qwen3 | `qwen3:gate:train_selected_gate` | `qwen3:boundary:strong_edge` | `interaction_prediction` | `L4_partial_holdout_candidate` | 0.5000 | 0.4051 |
| deepseek7b | `deepseek7b:gate:residual_projection_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:blocker_field_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:route_competition_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:compact_joint_gate_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:joint_gate_combo` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:model_default_gate` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
| deepseek7b | `deepseek7b:gate:train_selected_gate` | `deepseek7b:boundary:strong_edge` | `interaction_prediction` | `L3_in_sample_only` | 0.0000 | 0.0000 |
