# Phase274 Pattern Family Atlas v2 Gap Queue

- generated_at: 2026-07-09T04:11:21.456555+00:00
- source_path_signatures: 972
- gap_rows: 972
- selected_batch_rows: 54
- model_test_status: not_run_gap_queue_only

This phase does not claim new causal evidence. It turns Phase273 v2 into an explicit gap matrix and first batch queue for physical path completion.

## Highest Pressure Family-Model Cells

- cross_lingual / deepseek7b: pressure=0.994444, physical=0.126093, closure_readiness=0.018519, gaps={'need_component_path': 36, 'need_causal_audit': 36, 'need_layer_path': 33, 'good_behavior_low_path': 6, 'good_readout_low_causal': 10, 'need_closure_quality': 34, 'need_readout_competition': 8}
- syntax_structure / glm4: pressure=0.987605, physical=0.095341, closure_readiness=0.023585, gaps={'need_component_path': 35, 'need_causal_audit': 36, 'need_layer_path': 33, 'need_readout_competition': 18, 'good_behavior_low_path': 31, 'need_closure_quality': 34}
- content_knowledge / deepseek7b: pressure=0.984665, physical=0.076952, closure_readiness=0.01406, gaps={'need_readout_competition': 32, 'candidate_not_closed': 1, 'need_component_path': 35, 'need_causal_audit': 35, 'need_closure_quality': 35, 'need_layer_path': 33, 'good_behavior_low_path': 2}
- closure / deepseek7b: pressure=0.977778, physical=0.097354, closure_readiness=0.074074, gaps={'need_component_path': 36, 'need_causal_audit': 36, 'need_layer_path': 33, 'need_readout_competition': 24, 'good_behavior_low_path': 12, 'need_closure_quality': 28, 'good_readout_low_causal': 2}
- language_action / deepseek7b: pressure=0.977778, physical=0.123576, closure_readiness=0.074074, gaps={'need_component_path': 36, 'need_causal_audit': 36, 'need_layer_path': 33, 'good_behavior_low_path': 18, 'good_readout_low_causal': 16, 'need_closure_quality': 28, 'need_readout_competition': 20}
- readout_competition / qwen3: pressure=0.977778, physical=0.145859, closure_readiness=0.074074, gaps={'need_component_path': 36, 'need_causal_audit': 36, 'need_layer_path': 33, 'good_behavior_low_path': 20, 'good_readout_low_causal': 14, 'need_closure_quality': 28, 'need_readout_competition': 6}
- content_knowledge / glm4: pressure=0.969001, physical=0.12792, closure_readiness=0.067973, gaps={'need_component_path': 35, 'need_causal_audit': 35, 'need_layer_path': 33, 'good_behavior_low_path': 20, 'good_readout_low_causal': 6, 'need_closure_quality': 30, 'need_readout_competition': 6, 'candidate_not_closed': 1}
- readout_competition / glm4: pressure=0.966785, physical=0.180759, closure_readiness=0.084018, gaps={'need_component_path': 35, 'need_causal_audit': 35, 'need_layer_path': 33, 'good_behavior_low_path': 24, 'good_readout_low_causal': 20, 'need_closure_quality': 28, 'candidate_not_closed': 1, 'need_readout_competition': 6}

## First Batch Queue

- rank=1 glm4 closure phase265_closure_answer_correct_000_structured_json kind=candidate_closure_path_fill priority=11.810741 missing=['need_readout_competition', 'candidate_closure_verification']
- rank=2 glm4 output_protocol phase265_output_protocol_explain_answer_000_structured_json kind=candidate_closure_path_fill priority=11.761509 missing=['need_readout_competition', 'candidate_closure_verification']
- rank=3 glm4 cross_lingual phase265_cross_lingual_cross_lingual_attribute_000_structured_json kind=candidate_closure_path_fill priority=9.242447 missing=['candidate_closure_verification']
- rank=4 deepseek7b output_protocol phase265_output_protocol_explain_answer_001_structured_json kind=candidate_closure_path_fill priority=8.560989 missing=['need_readout_competition', 'candidate_closure_verification']
- rank=5 glm4 content_knowledge phase265_content_knowledge_category_membership_000_structured_json kind=candidate_closure_path_fill priority=8.383919 missing=['need_closure_quality', 'need_readout_competition', 'candidate_closure_verification']
- rank=6 qwen3 syntax_structure phase265_syntax_structure_clause_scope_000_boundary_period kind=candidate_closure_path_fill priority=8.378405 missing=['need_readout_competition', 'candidate_closure_verification']
- rank=7 glm4 readout_competition phase265_readout_competition_target_vs_because_000_explain_pressure kind=candidate_closure_path_fill priority=8.189886 missing=['need_closure_quality', 'candidate_closure_verification']
- rank=8 deepseek7b content_knowledge phase265_content_knowledge_category_membership_000_explain_pressure kind=candidate_closure_path_fill priority=7.949855 missing=['need_readout_competition', 'candidate_closure_verification']
- rank=9 deepseek7b reasoning_constraint phase265_reasoning_constraint_comparison_000_boundary_period kind=candidate_closure_path_fill priority=7.183584 missing=['need_closure_quality', 'need_readout_competition', 'candidate_closure_verification']
- rank=10 glm4 cross_lingual phase265_cross_lingual_en_to_zh_001_answer_only kind=high_signal_missing_mechanism priority=9.650991 missing=['need_component_path', 'need_causal_audit', 'need_layer_path']
- rank=11 glm4 closure phase265_closure_answer_correct_001_base kind=high_signal_missing_mechanism priority=9.586865 missing=['need_component_path', 'need_causal_audit', 'need_layer_path']
- rank=12 qwen3 closure phase265_closure_answer_correct_001_base kind=high_signal_missing_mechanism priority=9.56988 missing=['need_component_path', 'need_causal_audit', 'need_layer_path']

## Caution

Scores are coverage and prioritization signals. They are not closure proof. Small local models may have coarse internal structure, so selected gaps should be retested across qwen3, GLM4, and DS7B before theoretical claims are upgraded.
