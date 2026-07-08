# Pattern Family Atlas Data Contract

schema_version: 1.0.0
phase: Phase235
families: 9
modes: 72
test_cases: 36

## Required Files

- schema: `schema.json`
- client_index: `client_index.json`
- families: `families.jsonl`
- modes: `modes.jsonl`
- test_cases: `test_cases.jsonl`
- runs: `runs.jsonl`
- observations: `observations.jsonl`
- metrics: `metrics.jsonl`
- graph_nodes: `graph_nodes.jsonl`
- graph_edges: `graph_edges.jsonl`
- progress: `progress.json`
- summary: `summary.md`

## Progress

- pattern_family_atlas: 0.34
- model_internal_closure: 0.46
- general_language_mechanism_confidence: 0.43

## Known Evidence Edges

| edge | model | type | status | confidence |
| --- | --- | --- | --- | ---: |
| edge_glm4_no_answer_anchor_for_continuation | glm4 | prompt_anchor_to_regime_switch | hook_supported | 0.72 |
| edge_glm4_explain_instruction_because | glm4 | instruction_to_competitor_pressure | hook_supported | 0.60 |
| edge_qwen3_no_answer_anchor_because_period | qwen3 | prompt_anchor_to_competitor_pressure | hook_supported | 0.56 |
| edge_qwen3_period_second_takeover | qwen3 | suppression_to_takeover | readout_mapped | 0.50 |
| edge_deepseek7b_be_continuation_candidate | deepseek7b | weak_product_coupling | source_candidate | 0.32 |

## Phase236 Behavior Benchmark Update

- models: qwen3, glm4, deepseek7b
- case_rows: 132
- observation_rows: 1056
- mean_behavior_score: 0.6462
- pattern_match_rate: 0.6288
- drift_types: {'none': 83, 'wrong_or_missing_target': 30, 'over_generation': 19}

## Phase238 Scoring Calibration Update

- case_rows: 132
- calibrated_observation_rows: 792
- mean_original_behavior_score: 0.6462
- mean_calibrated_behavior_score: 0.8133
- ambiguous_rows: 13
- semantic_mismatch_rows: 13
- stable_failure_candidates: 12

## Phase239 Prompt Trigger Update

- variant_rows: 264
- observation_rows: 2376
- mean_score: 0.6152
- protocol_match_rate: 0.0038
- best_variants: [{'variant_id': 'colon_removed', 'rows': 24, 'mean_score': 0.725, 'protocol_match_rate': 0.0, 'over_generation_rate': 0.9583, 'mean_score_delta': 0.0125, 'winner_regimes': {'the_continuation': 12, 'be_continuation': 12}}, {'variant_id': 'full', 'rows': 24, 'mean_score': 0.7125, 'protocol_match_rate': 0.0, 'over_generation_rate': 0.9583, 'mean_score_delta': 0.0, 'winner_regimes': {'the_continuation': 16, 'newline_boundary': 8}}, {'variant_id': 'short_answer_instruction', 'rows': 24, 'mean_score': 0.6937, 'protocol_match_rate': 0.0, 'over_generation_rate': 0.9167, 'mean_score_delta': -0.0187, 'winner_regimes': {'the_continuation': 15, 'newline_boundary': 8, 'answer_boundary': 1}}, {'variant_id': 'one_word_strict', 'rows': 24, 'mean_score': 0.6625, 'protocol_match_rate': 0.0, 'over_generation_rate': 0.9583, 'mean_score_delta': -0.05, 'winner_regimes': {'the_continuation': 13, 'answer_boundary': 9, 'newline_boundary': 2}}, {'variant_id': 'explain_instruction', 'rows': 24, 'mean_score': 0.6437, 'protocol_match_rate': 0.0, 'over_generation_rate': 0.9583, 'mean_score_delta': -0.0687, 'winner_regimes': {'the_continuation': 12, 'answer_boundary': 8, 'newline_boundary': 4}}]

## Phase240 Gate/Product Protocol Trace Update

- behavior_rows: 108
- gate_product_trace_rows: 540
- residual_trace_rows: 324
- mean_behavior_score: 0.6278
- protocol_match_rate: 0.0
- model_decisions: {'qwen3': {'decision': 'protocol_state_written_but_readout_competition_failed', 'strict_mean_product_down_relative_delta': 0.172966, 'strict_mean_margin_delta': -0.066, 'strict_protocol_match_rate': 0.0, 'strict_over_generation_rate': 1.0}, 'glm4': {'decision': 'protocol_state_written_but_readout_competition_failed', 'strict_mean_product_down_relative_delta': 0.604032, 'strict_mean_margin_delta': -1.0608, 'strict_protocol_match_rate': 0.0, 'strict_over_generation_rate': 1.0}, 'deepseek7b': {'decision': 'protocol_state_written_but_readout_competition_failed', 'strict_mean_product_down_relative_delta': 0.682197, 'strict_mean_margin_delta': -2.6493, 'strict_protocol_match_rate': 0.0, 'strict_over_generation_rate': 0.6667}}
- top_component_deltas: [{'component': 'product', 'created_at': '2026-07-07T23:57:15.109395+00:00', 'family_id': 'output_protocol', 'layer_idx': 24, 'mean_cosine_vs_full': 0.37122, 'mean_delta_norm_vs_full': 137.304766, 'metric_id': 'phase240:deepseek7b:target_seeded:gate_up_product:l24:product', 'metric_name': 'mean_relative_delta_vs_full', 'metric_value': 1.101602, 'mode_id': 'short_answer', 'model': 'deepseek7b', 'phase_id': 'Phase240', 'rows': 6, 'schema_version': '1.0.0', 'scope': 'variant_component', 'trace_level': 'gate_up_product', 'variant_id': 'target_seeded'}, {'component': 'recomputed_product', 'created_at': '2026-07-07T23:57:15.109431+00:00', 'family_id': 'output_protocol', 'layer_idx': 24, 'mean_cosine_vs_full': 0.371307, 'mean_delta_norm_vs_full': 137.309387, 'metric_id': 'phase240:deepseek7b:target_seeded:gate_up_product:l24:recomputed_product', 'metric_name': 'mean_relative_delta_vs_full', 'metric_value': 1.101519, 'mode_id': 'short_answer', 'model': 'deepseek7b', 'phase_id': 'Phase240', 'rows': 6, 'schema_version': '1.0.0', 'scope': 'variant_component', 'trace_level': 'gate_up_product', 'variant_id': 'target_seeded'}, {'component': 'recomputed_product', 'created_at': '2026-07-07T23:56:22.676290+00:00', 'family_id': 'output_protocol', 'layer_idx': 29, 'mean_cosine_vs_full': 0.137354, 'mean_delta_norm_vs_full': 72.802993, 'metric_id': 'phase240:qwen3:target_seeded:gate_up_product:l29:recomputed_product', 'metric_name': 'mean_relative_delta_vs_full', 'metric_value': 1.042922, 'mode_id': 'short_answer', 'model': 'qwen3', 'phase_id': 'Phase240', 'rows': 6, 'schema_version': '1.0.0', 'scope': 'variant_component', 'trace_level': 'gate_up_product', 'variant_id': 'target_seeded'}, {'component': 'product', 'created_at': '2026-07-07T23:56:22.676251+00:00', 'family_id': 'output_protocol', 'layer_idx': 29, 'mean_cosine_vs_full': 0.137357, 'mean_delta_norm_vs_full': 72.806269, 'metric_id': 'phase240:qwen3:target_seeded:gate_up_product:l29:product', 'metric_name': 'mean_relative_delta_vs_full', 'metric_value': 1.042874, 'mode_id': 'short_answer', 'model': 'qwen3', 'phase_id': 'Phase240', 'rows': 6, 'schema_version': '1.0.0', 'scope': 'variant_component', 'trace_level': 'gate_up_product', 'variant_id': 'target_seeded'}, {'component': 'down_out', 'created_at': '2026-07-07T23:56:22.676271+00:00', 'family_id': 'output_protocol', 'layer_idx': 29, 'mean_cosine_vs_full': 0.238086, 'mean_delta_norm_vs_full': 92.795578, 'metric_id': 'phase240:qwen3:target_seeded:gate_up_product:l29:down_out', 'metric_name': 'mean_relative_delta_vs_full', 'metric_value': 1.009176, 'mode_id': 'short_answer', 'model': 'qwen3', 'phase_id': 'Phase240', 'rows': 6, 'schema_version': '1.0.0', 'scope': 'variant_component', 'trace_level': 'gate_up_product', 'variant_id': 'target_seeded'}]

## Phase241 Large-Scale Pattern Atlas Update

- case_count: 288
- behavior_rows: 5184
- readout_rows: 5184
- negative_rows: 4223
- mean_score: 0.5386
- semantic_match_rate: 0.7355
- protocol_match_rate: 0.1854
- negative_rate: 0.8146
- negative_categories: {'rollout_negative': 1863, 'semantic_failure': 1371, 'closure_negative': 398, 'readout_negative': 363, 'protocol_negative': 228}
- top_negative_modes: [{'created_at': '2026-07-08T01:11:36.814973+00:00', 'family_id': 'content_knowledge', 'mean_target_margin_vs_winner': -10.1013, 'metric_id': 'phase241:deepseek7b:content_knowledge:location_fact:large_scale_behavior', 'metric_name': 'large_scale_behavior_signature', 'metric_value': 0.1125, 'mode_id': 'location_fact', 'model': 'deepseek7b', 'negative_categories': {'rollout_negative': 4, 'semantic_failure': 20}, 'negative_rate': 1.0, 'over_generation_rate': 0.1667, 'phase_id': 'Phase241', 'protocol_match_rate': 0.0, 'rows': 24, 'schema_version': '1.0.0', 'scope': 'mode', 'semantic_match_rate': 0.1667, 'winner_regimes': {'comma_repeat': 12, 'period_stop': 8, 'the_continuation': 4}}, {'created_at': '2026-07-08T01:06:42.429963+00:00', 'family_id': 'content_knowledge', 'mean_target_margin_vs_winner': -9.5254, 'metric_id': 'phase241:qwen3:content_knowledge:causal_fact:large_scale_behavior', 'metric_name': 'large_scale_behavior_signature', 'metric_value': 0.1167, 'mode_id': 'causal_fact', 'model': 'qwen3', 'negative_categories': {'rollout_negative': 4, 'semantic_failure': 20}, 'negative_rate': 1.0, 'over_generation_rate': 0.1667, 'phase_id': 'Phase241', 'protocol_match_rate': 0.0, 'rows': 24, 'schema_version': '1.0.0', 'scope': 'mode', 'semantic_match_rate': 0.1667, 'winner_regimes': {'answer_boundary': 2, 'be_continuation': 1, 'comma_repeat': 7, 'newline_boundary': 4, 'period_stop': 8, 'the_continuation': 2}}, {'created_at': '2026-07-08T01:09:28.268621+00:00', 'family_id': 'content_knowledge', 'mean_target_margin_vs_winner': -8.7844, 'metric_id': 'phase241:glm4:content_knowledge:location_fact:large_scale_behavior', 'metric_name': 'large_scale_behavior_signature', 'metric_value': 0.1167, 'mode_id': 'location_fact', 'model': 'glm4', 'negative_categories': {'rollout_negative': 4, 'semantic_failure': 20}, 'negative_rate': 1.0, 'over_generation_rate': 0.1667, 'phase_id': 'Phase241', 'protocol_match_rate': 0.0, 'rows': 24, 'schema_version': '1.0.0', 'scope': 'mode', 'semantic_match_rate': 0.1667, 'winner_regimes': {'answer_boundary': 4, 'be_continuation': 4, 'newline_boundary': 1, 'period_stop': 4, 'the_continuation': 11}}, {'created_at': '2026-07-08T01:06:42.429963+00:00', 'family_id': 'content_knowledge', 'mean_target_margin_vs_winner': -14.4437, 'metric_id': 'phase241:qwen3:content_knowledge:location_fact:large_scale_behavior', 'metric_name': 'large_scale_behavior_signature', 'metric_value': 0.125, 'mode_id': 'location_fact', 'model': 'qwen3', 'negative_categories': {'rollout_negative': 4, 'semantic_failure': 20}, 'negative_rate': 1.0, 'over_generation_rate': 0.1667, 'phase_id': 'Phase241', 'protocol_match_rate': 0.0, 'rows': 24, 'schema_version': '1.0.0', 'scope': 'mode', 'semantic_match_rate': 0.1667, 'winner_regimes': {'answer_boundary': 2, 'be_continuation': 8, 'comma_repeat': 4, 'period_stop': 8, 'the_continuation': 2}}, {'created_at': '2026-07-08T01:11:36.814973+00:00', 'family_id': 'content_knowledge', 'mean_target_margin_vs_winner': -8.6527, 'metric_id': 'phase241:deepseek7b:content_knowledge:causal_fact:large_scale_behavior', 'metric_name': 'large_scale_behavior_signature', 'metric_value': 0.125, 'mode_id': 'causal_fact', 'model': 'deepseek7b', 'negative_categories': {'rollout_negative': 4, 'semantic_failure': 20}, 'negative_rate': 1.0, 'over_generation_rate': 0.1667, 'phase_id': 'Phase241', 'protocol_match_rate': 0.0, 'rows': 24, 'schema_version': '1.0.0', 'scope': 'mode', 'semantic_match_rate': 0.1667, 'winner_regimes': {'answer_boundary': 3, 'be_continuation': 1, 'comma_repeat': 9, 'period_stop': 7, 'the_continuation': 4}}]

## Phase242 Negative Multilabel And Trace Selection Update

- source_behavior_rows: 5184
- multilabel_rows: 5184
- high_value_candidates: 300
- hook_ready_candidates: 300
- case_bank_review_rows: 288
- manual_review_cases: 95
- multilabel_counts: {'semantic': 1371, 'protocol': 2852, 'readout': 4644, 'rollout': 3476, 'closure': 3345, 'scoring': 930}

## Phase243 Candidate Clustering And Case Bank V2 Update

- dedup_candidates: 300
- cluster_count: 157
- trace_selection_rows: 100
- case_bank_v2_rows: 288
- manual_review_cases: 95
- trace_selection_by_test: {'readout_competitor_trace': 40, 'protocol_gate_product_residual_trace': 25, 'stepwise_rollout_trace': 20, 'rollout_closure_trace': 10, 'cross_model_structure_comparison': 5}
- data_split_counts: {'explore': 168, 'validate': 70, 'frozen': 62}
