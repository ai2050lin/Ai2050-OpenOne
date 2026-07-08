# Phase243 Candidate Clustering And Case Bank V2

input_candidates: 300
dedup_candidates: 300
cluster_count: 157
trace_selection_rows: 100
case_bank_v2_rows: 288
manual_review_cases: 95

## Trace Selection By Test

- readout_competitor_trace: 40
- protocol_gate_product_residual_trace: 25
- stepwise_rollout_trace: 20
- rollout_closure_trace: 10
- cross_model_structure_comparison: 5

## Data Splits

- explore: 168
- validate: 70
- frozen: 62

## Top Clusters

| count | hook | score | family | mode | test | winner | group | margin |
| ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 8 | 8 | 0.75 | reasoning_constraint | if_then | readout_competitor_trace | be_continuation | stable_readout | margin_negative |
| 8 | 8 | 0.75 | reasoning_constraint | counterfactual | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 7 | 7 | 0.75 | state_drift | early_correct_late_drift | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 6 | 6 | 0.75 | readout_competition | because_reason | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 6 | 6 | 0.7333 | reasoning_constraint | if_then | readout_competitor_trace | period_stop | stable_readout | margin_negative |
| 6 | 6 | 0.7 | content_knowledge | object_attribute | readout_competitor_trace | period_stop | stable_readout | margin_negative |
| 5 | 5 | 0.75 | output_protocol | explain_answer | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.7875 | reasoning_constraint | negation | stepwise_rollout_trace | answer_boundary | rollout_failure | margin_positive |
| 4 | 4 | 0.75 | content_knowledge | function_answer | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.75 | reasoning_constraint | negation | readout_competitor_trace | comma_repeat | stable_readout | margin_negative |
| 4 | 4 | 0.75 | reasoning_constraint | negation | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.75 | reasoning_constraint | double_negation | readout_competitor_trace | be_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.75 | reasoning_constraint | double_negation | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.75 | reasoning_constraint | comparison | readout_competitor_trace | be_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.75 | syntax_structure | colon_boundary | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.75 | syntax_structure | newline_boundary | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.75 | closure | model_stop_executed | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.7125 | reasoning_constraint | comparison | readout_competitor_trace | comma_repeat | stable_readout | margin_negative |
| 4 | 4 | 0.7 | content_knowledge | material_answer | readout_competitor_trace | comma_repeat | stable_readout | margin_negative |
| 4 | 4 | 0.7 | output_protocol | stop_after_answer | readout_competitor_trace | comma_repeat | stable_readout | margin_negative |
| 4 | 4 | 0.7 | output_protocol | stop_after_answer | readout_competitor_trace | period_stop | stable_readout | margin_negative |
| 4 | 4 | 0.7 | reasoning_constraint | because_reason | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 4 | 4 | 0.7 | reasoning_constraint | negation | readout_competitor_trace | period_stop | stable_readout | margin_negative |
| 3 | 3 | 0.85 | reasoning_constraint | scope_binding | protocol_gate_product_residual_trace | newline_boundary | protocol_failure | margin_positive |
| 3 | 3 | 0.75 | output_protocol | repeat_answer | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.75 | reasoning_constraint | multi_hop_reasoning | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.75 | syntax_structure | question_form | readout_competitor_trace | period_stop | stable_readout | margin_negative |
| 3 | 3 | 0.75 | language_action | explain | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.75 | cross_lingual | cross_lingual_negation | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.75 | readout_competition | newline_boundary | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.75 | state_drift | next_task_drift | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.75 | state_drift | explain_takeover | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.75 | state_drift | boundary_takeover | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.75 | closure | eos_pressure | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.7333 | output_protocol | json_answer | readout_competitor_trace | the_continuation | stable_readout | margin_negative |
| 3 | 3 | 0.7 | content_knowledge | object_attribute | readout_competitor_trace | comma_repeat | stable_readout | margin_negative |
| 3 | 3 | 0.7 | output_protocol | short_answer | readout_competitor_trace | period_stop | stable_readout | margin_negative |
| 3 | 3 | 0.7 | output_protocol | repeat_answer | readout_competitor_trace | period_stop | stable_readout | margin_negative |
| 2 | 2 | 0.825 | closure | eos_pressure | protocol_gate_product_residual_trace | the_continuation | protocol_failure | margin_positive |
| 2 | 2 | 0.8 | readout_competition | target_answer | rollout_closure_trace | the_continuation | rollout_failure | margin_near |

## Selected Internal Trace Rows

| rank | score | family | mode | variant | test | split | winner |
| ---: | ---: | --- | --- | --- | --- | --- | --- |
| 1 | 0.85 | reasoning_constraint | comparison | no_answer_anchor | readout_competitor_trace | explore | comma_repeat |
| 2 | 0.8 | output_protocol | list_answer | explain_instruction | readout_competitor_trace | explore | the_continuation |
| 3 | 0.8 | readout_competition | for_continuation | explain_instruction | readout_competitor_trace | frozen | the_continuation |
| 4 | 0.8 | state_drift | continuation_takeover | explain_instruction | readout_competitor_trace | validate | the_continuation |
| 5 | 0.7667 | syntax_structure | clause_embedding | explain_instruction | readout_competitor_trace | explore | the_continuation |
| 6 | 0.75 | content_knowledge | object_relation_value | full | readout_competitor_trace | validate | the_continuation |
| 7 | 0.75 | content_knowledge | object_relation_value | explain_instruction | readout_competitor_trace | explore | the_continuation |
| 8 | 0.75 | content_knowledge | category_membership | full | readout_competitor_trace | explore | comma_repeat |
| 9 | 0.75 | content_knowledge | category_membership | one_word_strict | readout_competitor_trace | explore | period_stop |
| 10 | 0.75 | content_knowledge | function_answer | full | readout_competitor_trace | validate | the_continuation |
| 11 | 0.75 | content_knowledge | function_answer | full | readout_competitor_trace | frozen | the_continuation |
| 12 | 0.75 | content_knowledge | function_answer | full | readout_competitor_trace | validate | the_continuation |
| 13 | 0.75 | content_knowledge | function_answer | full | readout_competitor_trace | explore | the_continuation |
| 14 | 0.75 | content_knowledge | part_whole | full | readout_competitor_trace | explore | the_continuation |
| 15 | 0.75 | content_knowledge | part_whole | full | readout_competitor_trace | validate | the_continuation |
| 16 | 0.75 | output_protocol | short_answer | explain_instruction | readout_competitor_trace | explore | the_continuation |
| 17 | 0.75 | output_protocol | short_answer | explain_instruction | readout_competitor_trace | explore | the_continuation |
| 18 | 0.75 | output_protocol | one_word | full | readout_competitor_trace | validate | the_continuation |
| 19 | 0.75 | output_protocol | one_word | short_answer_instruction | readout_competitor_trace | explore | period_stop |
| 20 | 0.75 | output_protocol | explain_answer | full | readout_competitor_trace | explore | the_continuation |
| 21 | 0.75 | output_protocol | explain_answer | no_answer_anchor | readout_competitor_trace | explore | the_continuation |
| 22 | 0.75 | output_protocol | explain_answer | explain_instruction | readout_competitor_trace | frozen | the_continuation |
| 23 | 0.75 | output_protocol | explain_answer | full | readout_competitor_trace | validate | the_continuation |
| 24 | 0.75 | output_protocol | explain_answer | explain_instruction | readout_competitor_trace | explore | the_continuation |
| 25 | 0.75 | output_protocol | repeat_answer | explain_instruction | readout_competitor_trace | frozen | the_continuation |
| 26 | 0.75 | output_protocol | repeat_answer | full | readout_competitor_trace | explore | the_continuation |
| 27 | 0.75 | output_protocol | repeat_answer | explain_instruction | readout_competitor_trace | explore | the_continuation |
| 28 | 0.75 | output_protocol | list_answer | no_answer_anchor | readout_competitor_trace | explore | the_continuation |
| 29 | 0.75 | output_protocol | json_answer | full | readout_competitor_trace | explore | the_continuation |
| 30 | 0.75 | output_protocol | json_answer | one_word_strict | readout_competitor_trace | explore | period_stop |
| 31 | 0.75 | output_protocol | json_answer | explain_instruction | readout_competitor_trace | validate | the_continuation |
| 32 | 0.75 | output_protocol | table_answer | explain_instruction | readout_competitor_trace | explore | the_continuation |
| 33 | 0.75 | reasoning_constraint | if_then | full | readout_competitor_trace | explore | be_continuation |
| 34 | 0.75 | reasoning_constraint | if_then | no_answer_anchor | readout_competitor_trace | explore | be_continuation |
| 35 | 0.75 | reasoning_constraint | if_then | one_word_strict | readout_competitor_trace | frozen | period_stop |
| 36 | 0.75 | reasoning_constraint | if_then | short_answer_instruction | readout_competitor_trace | frozen | period_stop |
| 37 | 0.75 | reasoning_constraint | if_then | full | readout_competitor_trace | explore | be_continuation |
| 38 | 0.75 | reasoning_constraint | if_then | no_answer_anchor | readout_competitor_trace | frozen | be_continuation |
| 39 | 0.75 | reasoning_constraint | if_then | short_answer_instruction | readout_competitor_trace | explore | period_stop |
| 40 | 0.75 | reasoning_constraint | if_then | full | readout_competitor_trace | frozen | be_continuation |
| 41 | 0.85 | reasoning_constraint | scope_binding | full | protocol_gate_product_residual_trace | validate | newline_boundary |
| 42 | 0.85 | reasoning_constraint | scope_binding | full | protocol_gate_product_residual_trace | explore | newline_boundary |
| 43 | 0.85 | reasoning_constraint | scope_binding | full | protocol_gate_product_residual_trace | validate | newline_boundary |
| 44 | 0.85 | reasoning_constraint | scope_binding | short_answer_instruction | protocol_gate_product_residual_trace | explore | comma_repeat |
| 45 | 0.85 | reasoning_constraint | scope_binding | full | protocol_gate_product_residual_trace | validate | newline_boundary |
| 46 | 0.85 | syntax_structure | answer_anchor | short_answer_instruction | protocol_gate_product_residual_trace | explore | comma_repeat |
| 47 | 0.85 | syntax_structure | answer_anchor | short_answer_instruction | protocol_gate_product_residual_trace | explore | comma_repeat |
| 48 | 0.85 | readout_competition | the_continuation | short_answer_instruction | protocol_gate_product_residual_trace | frozen | the_continuation |
| 49 | 0.85 | closure | eos_pressure | explain_instruction | protocol_gate_product_residual_trace | frozen | the_continuation |
| 50 | 0.8 | content_knowledge | category_membership | explain_instruction | protocol_gate_product_residual_trace | explore | answer_boundary |
| 51 | 0.8 | output_protocol | one_word | explain_instruction | protocol_gate_product_residual_trace | frozen | the_continuation |
| 52 | 0.8 | output_protocol | stop_after_answer | explain_instruction | protocol_gate_product_residual_trace | frozen | the_continuation |
| 53 | 0.8 | reasoning_constraint | counterfactual | explain_instruction | protocol_gate_product_residual_trace | explore | answer_boundary |
| 54 | 0.8 | reasoning_constraint | counterfactual | explain_instruction | protocol_gate_product_residual_trace | explore | answer_boundary |
| 55 | 0.8 | syntax_structure | period_stop | explain_instruction | protocol_gate_product_residual_trace | explore | the_continuation |
| 56 | 0.8 | syntax_structure | newline_boundary | short_answer_instruction | protocol_gate_product_residual_trace | explore | comma_repeat |
| 57 | 0.8 | readout_competition | period_stop | explain_instruction | protocol_gate_product_residual_trace | frozen | answer_boundary |
| 58 | 0.8 | readout_competition | the_continuation | short_answer_instruction | protocol_gate_product_residual_trace | explore | newline_boundary |
| 59 | 0.8 | state_drift | continuation_takeover | short_answer_instruction | protocol_gate_product_residual_trace | validate | the_continuation |
| 60 | 0.8 | state_drift | continuation_takeover | short_answer_instruction | protocol_gate_product_residual_trace | explore | the_continuation |
| 61 | 0.8 | state_drift | boundary_takeover | short_answer_instruction | protocol_gate_product_residual_trace | validate | comma_repeat |
| 62 | 0.8 | closure | pattern_matched | explain_instruction | protocol_gate_product_residual_trace | validate | the_continuation |
| 63 | 0.8 | closure | boundary_stable | explain_instruction | protocol_gate_product_residual_trace | explore | the_continuation |
| 64 | 0.8 | closure | no_drift | explain_instruction | protocol_gate_product_residual_trace | frozen | the_continuation |
| 65 | 0.8 | closure | no_drift | explain_instruction | protocol_gate_product_residual_trace | validate | answer_boundary |
| 66 | 0.8667 | reasoning_constraint | scope_binding | explain_instruction | stepwise_rollout_trace | validate | the_continuation |
| 67 | 0.85 | content_knowledge | object_relation_value | explain_instruction | stepwise_rollout_trace | explore | the_continuation |
| 68 | 0.85 | reasoning_constraint | comparison | explain_instruction | stepwise_rollout_trace | frozen | the_continuation |
| 69 | 0.85 | state_drift | over_generation | explain_instruction | stepwise_rollout_trace | explore | the_continuation |
| 70 | 0.85 | state_drift | continuation_takeover | explain_instruction | stepwise_rollout_trace | explore | the_continuation |
| 71 | 0.8167 | reasoning_constraint | if_then | explain_instruction | stepwise_rollout_trace | frozen | answer_boundary |
| 72 | 0.8 | content_knowledge | category_membership | explain_instruction | stepwise_rollout_trace | explore | the_continuation |
| 73 | 0.8 | reasoning_constraint | if_then | explain_instruction | stepwise_rollout_trace | frozen | the_continuation |
| 74 | 0.8 | reasoning_constraint | if_then | explain_instruction | stepwise_rollout_trace | explore | the_continuation |
| 75 | 0.8 | reasoning_constraint | negation | explain_instruction | stepwise_rollout_trace | explore | answer_boundary |
| 76 | 0.8 | reasoning_constraint | negation | explain_instruction | stepwise_rollout_trace | explore | answer_boundary |
| 77 | 0.8 | reasoning_constraint | negation | explain_instruction | stepwise_rollout_trace | explore | answer_boundary |
| 78 | 0.8 | reasoning_constraint | comparison | explain_instruction | stepwise_rollout_trace | explore | the_continuation |
| 79 | 0.8 | reasoning_constraint | comparison | explain_instruction | stepwise_rollout_trace | frozen | the_continuation |
| 80 | 0.8 | reasoning_constraint | comparison | explain_instruction | stepwise_rollout_trace | explore | answer_boundary |

## Case Bank V2 Review

manual_review_cases: 95

Phase243 does not run model hooks. It prepares a balanced first internal-trace batch and data splits.
