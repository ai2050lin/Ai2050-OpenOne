# Phase 874 State Transition Decomposition

- Boundary: offline decomposition from Phase 872 rows; no new model run.
- Goal: separate output state, observed state transition, and clean causal edge.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `intervened_output_state_open` | `intervened_rollout_clear_answer_class` | 792 | 233 | 0 | 12 | 547 | 1.000 | 0.951 | 0.985 |
| `intervened_output_state_open` | `target_output_clean_transition` | 792 | 33 | 200 | 0 | 559 | 0.142 | 1.000 | 0.747 |
| `intervened_output_state_open` | `target_clean_transition` | 792 | 9 | 224 | 0 | 559 | 0.039 | 1.000 | 0.717 |
| `latent_output_gate_transition_rule` | `intervened_rollout_clear_answer_class` | 792 | 36 | 0 | 209 | 547 | 1.000 | 0.147 | 0.736 |
| `latent_output_gate_transition_rule` | `target_output_clean_transition` | 792 | 33 | 3 | 0 | 756 | 0.917 | 1.000 | 0.996 |
| `latent_output_gate_transition_rule` | `target_clean_transition` | 792 | 9 | 27 | 0 | 756 | 0.250 | 1.000 | 0.966 |
| `observed_clear_transition_rule` | `intervened_rollout_clear_answer_class` | 792 | 33 | 0 | 212 | 547 | 1.000 | 0.135 | 0.732 |
| `observed_clear_transition_rule` | `target_output_clean_transition` | 792 | 33 | 0 | 0 | 759 | 1.000 | 1.000 | 1.000 |
| `observed_clear_transition_rule` | `target_clean_transition` | 792 | 9 | 24 | 0 | 759 | 0.273 | 1.000 | 0.970 |
| `observed_strict_transition_rule` | `intervened_rollout_clear_answer_class` | 792 | 47 | 0 | 198 | 547 | 1.000 | 0.192 | 0.750 |
| `observed_strict_transition_rule` | `target_output_clean_transition` | 792 | 33 | 14 | 0 | 745 | 0.702 | 1.000 | 0.982 |
| `observed_strict_transition_rule` | `target_clean_transition` | 792 | 9 | 38 | 0 | 745 | 0.191 | 1.000 | 0.952 |
| `clean_causal_edge_rule` | `intervened_rollout_clear_answer_class` | 792 | 5 | 0 | 240 | 547 | 1.000 | 0.020 | 0.697 |
| `clean_causal_edge_rule` | `target_output_clean_transition` | 792 | 5 | 0 | 28 | 759 | 1.000 | 0.152 | 0.965 |
| `clean_causal_edge_rule` | `target_clean_transition` | 792 | 5 | 0 | 4 | 783 | 1.000 | 0.556 | 0.995 |

## Summary

- Transition class counts: `{'answer_class_stable_closed': 519, 'answer_class_stable_open': 212, 'clean_causal_transition': 9, 'answer_class_loss': 28, 'nonclean_output_transition': 24}`
- Nonclean transition reasons: `{'field_not_strict_admissible': 22, 'not_phase866_pair_rule': 24, 'original_blocker_not_reduced': 24, 'field_tag:semantic_other_pressure': 16, 'field_not_base_admissible': 7, 'field_tag:format_dominates': 2, 'field_tag:object_dominates_class': 6, 'field_tag:object_echo_pressure': 6, 'field_tag:protocol_pressure': 2}`
- Observed transition labels: `{'object_echo->strict_canonical': 6, 'other->strict_canonical': 16, 'format_or_empty->strict_canonical': 5, 'format_or_empty->answer_alias': 3, 'other->answer_alias': 3}`

## Output Transitions

| round | model | domain | object | prompt | mode | class | labels | clear-rule | strict-rule | clean-edge | field tags | reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `holdout_phase867` | qwen3 | material | rubber | `holdout_kind_phrase` | `flip` | `clean_causal_transition` | `object_echo->strict_canonical` | True | True | True | `['field_low_pressure']` | `[]` |
| `holdout_phase867` | deepseek7b | animal | whale | `holdout_category_short` | `flip` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `holdout_phase867` | deepseek7b | animal | whale | `holdout_category_short` | `zero` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `holdout_phase867` | deepseek7b | color | purple | `holdout_category_short` | `flip` | `clean_causal_transition` | `format_or_empty->strict_canonical` | True | True | True | `['field_low_pressure']` | `[]` |
| `holdout_phase867` | deepseek7b | color | purple | `holdout_category_short` | `half` | `clean_causal_transition` | `format_or_empty->strict_canonical` | True | True | True | `['field_low_pressure']` | `[]` |
| `holdout_phase867` | deepseek7b | color | purple | `holdout_category_short` | `zero` | `clean_causal_transition` | `format_or_empty->strict_canonical` | True | True | True | `['field_low_pressure']` | `[]` |
| `holdout_phase867` | deepseek7b | color | orange | `holdout_category_short` | `flip` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `holdout_phase867` | deepseek7b | color | orange | `holdout_category_short` | `half` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `holdout_phase867` | deepseek7b | color | orange | `holdout_category_short` | `zero` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `holdout_phase867` | deepseek7b | color | orange | `holdout_kind_phrase` | `flip` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `holdout_phase867` | deepseek7b | color | black | `holdout_category_short` | `flip` | `nonclean_output_transition` | `format_or_empty->strict_canonical` | True | True | False | `['format_dominates']` | `['field_not_base_admissible', 'field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:format_dominates']` |
| `holdout_phase867` | deepseek7b | color | white | `holdout_category_short` | `flip` | `nonclean_output_transition` | `format_or_empty->strict_canonical` | True | True | False | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | `['field_not_base_admissible', 'field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:object_dominates_class', 'field_tag:format_dominates', 'field_tag:object_echo_pressure']` |
| `validation_phase871` | deepseek7b | color | cyan | `validation_direct` | `flip` | `clean_causal_transition` | `other->strict_canonical` | True | True | True | `['field_low_pressure']` | `[]` |
| `validation_phase871` | deepseek7b | color | brown | `validation_direct` | `flip` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `validation_phase871` | deepseek7b | color | brown | `validation_table` | `flip` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `validation_phase871` | deepseek7b | color | gray | `validation_direct` | `flip` | `nonclean_output_transition` | `object_echo->strict_canonical` | True | True | False | `['object_dominates_class', 'object_echo_pressure']` | `['field_not_base_admissible', 'field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:object_dominates_class', 'field_tag:object_echo_pressure']` |
| `validation_phase871` | deepseek7b | color | gray | `validation_direct` | `half` | `nonclean_output_transition` | `object_echo->strict_canonical` | True | True | False | `['object_dominates_class', 'object_echo_pressure']` | `['field_not_base_admissible', 'field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:object_dominates_class', 'field_tag:object_echo_pressure']` |
| `validation_phase871` | deepseek7b | color | gray | `validation_direct` | `zero` | `nonclean_output_transition` | `object_echo->strict_canonical` | True | True | False | `['object_dominates_class', 'object_echo_pressure']` | `['field_not_base_admissible', 'field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:object_dominates_class', 'field_tag:object_echo_pressure']` |
| `validation_phase871` | deepseek7b | color | gray | `validation_table` | `flip` | `nonclean_output_transition` | `object_echo->strict_canonical` | True | True | False | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | `['field_not_base_admissible', 'field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:object_dominates_class', 'field_tag:object_echo_pressure', 'field_tag:semantic_other_pressure']` |
| `replication_phase873` | qwen3 | material | plastic | `replication_sentence` | `flip` | `answer_class_stable_open` | `strict_canonical->strict_canonical` | False | False | False | `['field_low_pressure']` | `[]` |
| `replication_phase873` | qwen3 | material | plastic | `replication_sentence` | `half` | `answer_class_stable_open` | `strict_canonical->strict_canonical` | False | False | False | `['field_low_pressure']` | `[]` |
| `replication_phase873` | qwen3 | material | plastic | `replication_sentence` | `zero` | `answer_class_stable_open` | `strict_canonical->strict_canonical` | False | False | False | `['field_low_pressure']` | `[]` |
| `replication_phase873` | deepseek7b | animal | goat | `replication_direct` | `flip` | `nonclean_output_transition` | `object_echo->strict_canonical` | True | True | False | `['object_dominates_class', 'object_echo_pressure']` | `['field_not_base_admissible', 'field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:object_dominates_class', 'field_tag:object_echo_pressure']` |
| `replication_phase873` | deepseek7b | color | red | `replication_direct` | `flip` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `validation_phase876` | deepseek7b | animal | seal | `format_pressure` | `scale_up` | `clean_causal_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `[]` |
| `validation_phase876` | deepseek7b | animal | bat | `nonclean_direct` | `flip` | `clean_causal_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `[]` |
| `validation_phase876` | deepseek7b | animal | sheep | `echo_pressure` | `flip` | `nonclean_output_transition` | `format_or_empty->answer_alias` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `validation_phase876` | deepseek7b | animal | sheep | `echo_pressure` | `half` | `nonclean_output_transition` | `format_or_empty->answer_alias` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `validation_phase876` | deepseek7b | animal | sheep | `echo_pressure` | `zero` | `nonclean_output_transition` | `format_or_empty->answer_alias` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `validation_phase876` | deepseek7b | animal | wolf | `echo_pressure` | `flip` | `nonclean_output_transition` | `other->answer_alias` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `validation_phase876` | deepseek7b | animal | wolf | `echo_pressure` | `half` | `nonclean_output_transition` | `other->answer_alias` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `validation_phase876` | deepseek7b | animal | wolf | `echo_pressure` | `zero` | `nonclean_output_transition` | `other->answer_alias` | True | True | False | `['semantic_other_pressure']` | `['field_not_strict_admissible', 'not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:semantic_other_pressure']` |
| `validation_phase876` | deepseek7b | animal | wolf | `format_pressure` | `flip` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['protocol_pressure']` | `['not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:protocol_pressure']` |
| `validation_phase876` | deepseek7b | animal | wolf | `format_pressure` | `zero` | `nonclean_output_transition` | `other->strict_canonical` | True | True | False | `['protocol_pressure']` | `['not_phase866_pair_rule', 'original_blocker_not_reduced', 'field_tag:protocol_pressure']` |
| `validation_phase876` | deepseek7b | color | navy | `nonclean_direct` | `flip` | `clean_causal_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `[]` |
| `validation_phase876` | deepseek7b | color | navy | `nonclean_direct` | `zero` | `clean_causal_transition` | `other->strict_canonical` | True | True | False | `['semantic_other_pressure']` | `[]` |
