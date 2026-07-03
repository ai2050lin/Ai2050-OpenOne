# Phase 874 State Transition Decomposition

- Boundary: offline decomposition from Phase 872 rows; no new model run.
- Goal: separate output state, observed state transition, and clean causal edge.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `intervened_output_state_open` | `intervened_rollout_clear_answer_class` | 504 | 170 | 0 | 0 | 334 | 1.000 | 1.000 | 1.000 |
| `intervened_output_state_open` | `target_output_clean_transition` | 504 | 21 | 149 | 0 | 334 | 0.124 | 1.000 | 0.704 |
| `intervened_output_state_open` | `target_clean_transition` | 504 | 5 | 165 | 0 | 334 | 0.029 | 1.000 | 0.673 |
| `latent_output_gate_transition_rule` | `intervened_rollout_clear_answer_class` | 504 | 24 | 0 | 146 | 334 | 1.000 | 0.141 | 0.710 |
| `latent_output_gate_transition_rule` | `target_output_clean_transition` | 504 | 21 | 3 | 0 | 480 | 0.875 | 1.000 | 0.994 |
| `latent_output_gate_transition_rule` | `target_clean_transition` | 504 | 5 | 19 | 0 | 480 | 0.208 | 1.000 | 0.962 |
| `observed_clear_transition_rule` | `intervened_rollout_clear_answer_class` | 504 | 21 | 0 | 149 | 334 | 1.000 | 0.124 | 0.704 |
| `observed_clear_transition_rule` | `target_output_clean_transition` | 504 | 21 | 0 | 0 | 483 | 1.000 | 1.000 | 1.000 |
| `observed_clear_transition_rule` | `target_clean_transition` | 504 | 5 | 16 | 0 | 483 | 0.238 | 1.000 | 0.968 |
| `observed_strict_transition_rule` | `intervened_rollout_clear_answer_class` | 504 | 24 | 0 | 146 | 334 | 1.000 | 0.141 | 0.710 |
| `observed_strict_transition_rule` | `target_output_clean_transition` | 504 | 21 | 3 | 0 | 480 | 0.875 | 1.000 | 0.994 |
| `observed_strict_transition_rule` | `target_clean_transition` | 504 | 5 | 19 | 0 | 480 | 0.208 | 1.000 | 0.962 |
| `clean_causal_edge_rule` | `intervened_rollout_clear_answer_class` | 504 | 5 | 0 | 165 | 334 | 1.000 | 0.029 | 0.673 |
| `clean_causal_edge_rule` | `target_output_clean_transition` | 504 | 5 | 0 | 16 | 483 | 1.000 | 0.238 | 0.968 |
| `clean_causal_edge_rule` | `target_clean_transition` | 504 | 5 | 0 | 0 | 499 | 1.000 | 1.000 | 1.000 |

## Summary

- Transition class counts: `{'answer_class_stable_closed': 311, 'answer_class_stable_open': 149, 'clean_causal_transition': 5, 'answer_class_loss': 23, 'nonclean_output_transition': 16}`
- Nonclean transition reasons: `{'field_not_strict_admissible': 16, 'not_phase866_pair_rule': 16, 'original_blocker_not_reduced': 16, 'field_tag:semantic_other_pressure': 10, 'field_not_base_admissible': 7, 'field_tag:format_dominates': 2, 'field_tag:object_dominates_class': 6, 'field_tag:object_echo_pressure': 6}`
- Observed transition labels: `{'object_echo->strict_canonical': 6, 'other->strict_canonical': 10, 'format_or_empty->strict_canonical': 5}`

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
