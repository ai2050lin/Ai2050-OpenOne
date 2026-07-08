# Phase242 Negative Multilabel And Trace Selection

source_behavior_rows: 5184
multilabel_rows: 5184
high_value_candidates: 300
hook_ready_candidates: 300
case_bank_review_rows: 288
manual_review_cases: 95

## Multilabel Counts

- semantic: 1371
- protocol: 2852
- readout: 4644
- rollout: 3476
- closure: 3345
- scoring: 930

## Candidate Reasons

- semantic_correct_rollout_failure: 286
- cross_model_divergence: 14
- semantic_correct_closure_failure: 209
- cross_model_stable_failure: 286
- semantic_correct_protocol_failure: 194
- stable_readout_competitor: 216
- high_target_pressure_protocol_failure: 38

## Next Tests

- stepwise_rollout_trace: 29
- rollout_closure_trace: 13
- readout_competitor_trace: 216
- protocol_gate_product_residual_trace: 36
- cross_model_structure_comparison: 6

## Top Hook Candidates

| score | family | mode | variant | reasons | next | winner | margin |
| ---: | --- | --- | --- | --- | --- | --- | ---: |
| 0.8667 | reasoning_constraint | scope_binding | explain_instruction | semantic_correct_rollout_failure,cross_model_divergence | stepwise_rollout_trace | the_continuation | 1.0 |
| 0.8667 | cross_lingual | ZH_to_EN | explain_instruction | semantic_correct_closure_failure,cross_model_divergence | rollout_closure_trace | the_continuation | 0.2083 |
| 0.85 | content_knowledge | object_relation_value | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | 1.8125 |
| 0.85 | reasoning_constraint | comparison | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | -0.2917 |
| 0.85 | reasoning_constraint | comparison | no_answer_anchor | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,stable_readout_competitor,high_target_pressure_protocol_failure | readout_competitor_trace | comma_repeat | -0.8333 |
| 0.85 | reasoning_constraint | scope_binding | full | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | newline_boundary | 4.4896 |
| 0.85 | reasoning_constraint | scope_binding | full | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | newline_boundary | 2.9167 |
| 0.85 | reasoning_constraint | scope_binding | full | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | newline_boundary | 4.0521 |
| 0.85 | reasoning_constraint | scope_binding | short_answer_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | comma_repeat | -0.7917 |
| 0.85 | reasoning_constraint | scope_binding | full | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | newline_boundary | -0.1771 |
| 0.85 | syntax_structure | answer_anchor | short_answer_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | comma_repeat | 1.6667 |
| 0.85 | syntax_structure | answer_anchor | short_answer_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | comma_repeat | -0.8125 |
| 0.85 | readout_competition | target_answer | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure | rollout_closure_trace | the_continuation | 0.4167 |
| 0.85 | readout_competition | the_continuation | short_answer_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | the_continuation | 3.2708 |
| 0.85 | state_drift | over_generation | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | -0.3333 |
| 0.85 | state_drift | continuation_takeover | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | -0.1667 |
| 0.85 | closure | eos_pressure | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | the_continuation | 5.6146 |
| 0.8167 | reasoning_constraint | if_then | explain_instruction | semantic_correct_rollout_failure,cross_model_divergence | stepwise_rollout_trace | answer_boundary | 2.8438 |
| 0.8 | content_knowledge | object_attribute | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure | rollout_closure_trace | answer_boundary | -0.2917 |
| 0.8 | content_knowledge | category_membership | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | answer_boundary | -0.4792 |
| 0.8 | content_knowledge | category_membership | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | 3.0625 |
| 0.8 | output_protocol | one_word | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | the_continuation | 0.1875 |
| 0.8 | output_protocol | list_answer | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,stable_readout_competitor | readout_competitor_trace | the_continuation | 0.6458 |
| 0.8 | output_protocol | table_answer | full | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure | rollout_closure_trace | comma_repeat | 0.4583 |
| 0.8 | output_protocol | stop_after_answer | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | the_continuation | 3.7708 |
| 0.8 | reasoning_constraint | if_then | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | 3.1146 |
| 0.8 | reasoning_constraint | if_then | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | 3.9271 |
| 0.8 | reasoning_constraint | negation | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | answer_boundary | 4.9688 |
| 0.8 | reasoning_constraint | negation | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | answer_boundary | 4.3646 |
| 0.8 | reasoning_constraint | negation | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | answer_boundary | 4.9583 |
| 0.8 | reasoning_constraint | comparison | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | 1.5833 |
| 0.8 | reasoning_constraint | comparison | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | 1.8542 |
| 0.8 | reasoning_constraint | comparison | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | answer_boundary | 1.9375 |
| 0.8 | reasoning_constraint | counterfactual | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | answer_boundary | 3.2812 |
| 0.8 | reasoning_constraint | counterfactual | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | answer_boundary | 3.0938 |
| 0.8 | reasoning_constraint | counterfactual | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | answer_boundary | 3.5 |
| 0.8 | reasoning_constraint | scope_binding | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | the_continuation | 0.8542 |
| 0.8 | syntax_structure | period_stop | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | the_continuation | 0.9792 |
| 0.8 | syntax_structure | newline_boundary | short_answer_instruction | cross_model_stable_failure,semantic_correct_rollout_failure,semantic_correct_closure_failure,semantic_correct_protocol_failure,high_target_pressure_protocol_failure | protocol_gate_product_residual_trace | comma_repeat | -0.4583 |
| 0.8 | syntax_structure | question_form | explain_instruction | cross_model_stable_failure,semantic_correct_rollout_failure | stepwise_rollout_trace | answer_boundary | 4.0833 |

## Case Bank Review Hotspots

| family | mode | review cases |
| --- | --- | ---: |
| content_knowledge | function_answer | 4 |
| content_knowledge | part_whole | 4 |
| content_knowledge | material_answer | 4 |
| content_knowledge | location_fact | 4 |
| content_knowledge | causal_fact | 4 |
| language_action | summarize | 4 |
| language_action | translate | 4 |
| language_action | classify | 4 |
| language_action | rewrite | 4 |
| language_action | compare | 4 |
| cross_lingual | ZH_to_ZH | 4 |
| cross_lingual | EN_to_FR | 4 |
| cross_lingual | FR_to_EN | 4 |
| cross_lingual | cross_lingual_reasoning | 4 |
| cross_lingual | ZH_to_EN | 3 |
| content_knowledge | object_relation_value | 2 |
| content_knowledge | category_membership | 2 |
| output_protocol | one_word | 2 |
| cross_lingual | EN_to_ZH | 2 |
| state_drift | format_drift | 2 |
| closure | pattern_matched | 2 |
| closure | boundary_stable | 2 |
| closure | done_state_stable | 2 |
| output_protocol | explain_answer | 1 |
| output_protocol | repeat_answer | 1 |
| output_protocol | list_answer | 1 |
| output_protocol | json_answer | 1 |
| output_protocol | table_answer | 1 |
| reasoning_constraint | multi_hop_reasoning | 1 |
| syntax_structure | answer_anchor | 1 |

## Caution

This phase does not run models or hooks. It upgrades Phase241 observations into multilabel negatives, case-bank calibration targets, and internal-trace candidates.
