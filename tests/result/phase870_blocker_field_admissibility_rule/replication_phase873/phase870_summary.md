# Phase 870 Blocker Field Admissibility Rule (replication)

- Source: Phase 867 paired original/intervened rows.
- Boundary: single-context rule audit, not model training and not closure.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_predict_clean_mixed` | `target_clean_transition` | 216 | 0 | 108 | 0 | 108 | 0.000 | 0.000 | 0.500 |
| `source_predict_clean_mixed` | `target_output_clean_transition` | 216 | 1 | 107 | 1 | 107 | 0.009 | 0.500 | 0.500 |
| `phase866_pair_rule` | `target_clean_transition` | 216 | 0 | 14 | 0 | 202 | 0.000 | 0.000 | 0.935 |
| `phase866_pair_rule` | `target_output_clean_transition` | 216 | 0 | 14 | 2 | 200 | 0.000 | 0.000 | 0.926 |
| `field_base_admissible` | `target_clean_transition` | 216 | 0 | 112 | 0 | 104 | 0.000 | 0.000 | 0.481 |
| `field_base_admissible` | `target_output_clean_transition` | 216 | 1 | 111 | 1 | 103 | 0.009 | 0.500 | 0.481 |
| `field_plus_effect_rule` | `target_clean_transition` | 216 | 0 | 3 | 0 | 213 | 0.000 | 0.000 | 0.986 |
| `field_plus_effect_rule` | `target_output_clean_transition` | 216 | 0 | 3 | 2 | 211 | 0.000 | 0.000 | 0.977 |
| `field_strict_admissible` | `target_clean_transition` | 216 | 0 | 84 | 0 | 132 | 0.000 | 0.000 | 0.611 |
| `field_strict_admissible` | `target_output_clean_transition` | 216 | 0 | 84 | 2 | 130 | 0.000 | 0.000 | 0.602 |
| `field_strict_plus_effect_rule` | `target_clean_transition` | 216 | 0 | 0 | 0 | 216 | 0.000 | 0.000 | 1.000 |
| `field_strict_plus_effect_rule` | `target_output_clean_transition` | 216 | 0 | 0 | 2 | 214 | 0.000 | 0.000 | 0.991 |

## Summary

- Transfer status counts: `{'source_clean_failed': 108, 'stable_nonclean': 108}`
- Field tag counts: `{'object_dominates_class': 72, 'format_dominates': 64, 'object_echo_pressure': 48, 'semantic_other_pressure': 116, 'protocol_pressure': 52, 'field_low_pressure': 80, 'too_many_blockers': 52}`

## Pair Rows

| model | domain | object | prompt | mode | status | field tags | field ok | phase866 | field+effect | target clean | clear gain/loss | ans | block red. | orig block |
|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| qwen3 | material | glass | `replication_direct` | `flip` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 0.000 | 0.031 |
| qwen3 | material | glass | `replication_direct` | `half` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.019 |
| qwen3 | material | glass | `replication_direct` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -1.000 | -0.006 |
| qwen3 | material | glass | `replication_direct` | `zero` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 0.000 | 0.025 |
| qwen3 | material | glass | `replication_sentence` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.500 | 0.000 | 0.000 |
| qwen3 | material | glass | `replication_sentence` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | glass | `replication_sentence` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.500 | 0.000 | 0.000 |
| qwen3 | material | glass | `replication_sentence` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | 0.000 |
| qwen3 | material | glass | `replication_form` | `flip` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.312 | 1.000 | 0.000 |
| qwen3 | material | glass | `replication_form` | `half` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.000 |
| qwen3 | material | glass | `replication_form` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.188 | 0.000 | 0.016 |
| qwen3 | material | glass | `replication_form` | `zero` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 1.000 | 0.000 |
| qwen3 | material | steel | `replication_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_sentence` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.250 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_sentence` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_sentence` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_sentence` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_form` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_form` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_form` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | steel | `replication_form` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | wool | `replication_direct` | `flip` | `source_clean_failed` | `['protocol_pressure']` | True | False | False | False | 0/0 | 0.125 | 1.000 | 0.031 |
| qwen3 | material | wool | `replication_direct` | `half` | `source_clean_failed` | `['protocol_pressure']` | True | False | False | False | 0/0 | 0.062 | 1.000 | 0.000 |
| qwen3 | material | wool | `replication_direct` | `scale_up` | `stable_nonclean` | `['protocol_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.031 |
| qwen3 | material | wool | `replication_direct` | `zero` | `source_clean_failed` | `['protocol_pressure']` | True | False | False | False | 0/0 | 0.125 | 1.000 | 0.031 |
| qwen3 | material | wool | `replication_sentence` | `flip` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.750 | 4.000 | -0.087 |
| qwen3 | material | wool | `replication_sentence` | `half` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.250 | 1.000 | -0.050 |
| qwen3 | material | wool | `replication_sentence` | `scale_up` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.375 | -4.000 | -0.025 |
| qwen3 | material | wool | `replication_sentence` | `zero` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.375 | 2.000 | -0.075 |
| qwen3 | material | wool | `replication_form` | `flip` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.562 | 2.000 | 0.045 |
| qwen3 | material | wool | `replication_form` | `half` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.188 | 1.000 | 0.071 |
| qwen3 | material | wool | `replication_form` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.250 | -1.000 | 0.062 |
| qwen3 | material | wool | `replication_form` | `zero` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.250 | 1.000 | 0.009 |
| qwen3 | material | clay | `replication_direct` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 1.125 | 21.000 | -0.013 |
| qwen3 | material | clay | `replication_direct` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.312 | 9.000 | 0.000 |
| qwen3 | material | clay | `replication_direct` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.562 | -19.000 | -0.013 |
| qwen3 | material | clay | `replication_direct` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.562 | 11.000 | 0.013 |
| qwen3 | material | clay | `replication_sentence` | `flip` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 1.500 | 15.000 | -0.025 |
| qwen3 | material | clay | `replication_sentence` | `half` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.500 | 7.000 | 0.000 |
| qwen3 | material | clay | `replication_sentence` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.938 | -14.000 | 0.025 |
| qwen3 | material | clay | `replication_sentence` | `zero` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.875 | 12.000 | 0.013 |
| qwen3 | material | clay | `replication_form` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.125 | -11.000 | -0.006 |
| qwen3 | material | clay | `replication_form` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | -10.000 | 0.013 |
| qwen3 | material | clay | `replication_form` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.019 |
| qwen3 | material | clay | `replication_form` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | -8.000 | -0.013 |
| qwen3 | material | plastic | `replication_direct` | `flip` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.500 | 5.000 | 0.037 |
| qwen3 | material | plastic | `replication_direct` | `half` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 2.000 | 0.013 |
| qwen3 | material | plastic | `replication_direct` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.250 | -3.000 | 0.000 |
| qwen3 | material | plastic | `replication_direct` | `zero` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 2.000 | 0.019 |
| qwen3 | material | plastic | `replication_sentence` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.000 | 0.000 | 0.000 |
| qwen3 | material | plastic | `replication_sentence` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | 0.000 |
| qwen3 | material | plastic | `replication_sentence` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -0.625 | -6.000 | 0.000 |
| qwen3 | material | plastic | `replication_sentence` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.500 | 0.000 | 0.000 |
| qwen3 | material | plastic | `replication_form` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.500 | 0.000 | 0.000 |
| qwen3 | material | plastic | `replication_form` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | plastic | `replication_form` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.250 | 0.000 | 0.000 |
| qwen3 | material | plastic | `replication_form` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.312 | 0.000 | 0.000 |
| qwen3 | material | sand | `replication_direct` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.312 | 19.000 | 0.100 |
| qwen3 | material | sand | `replication_direct` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 1.000 | 0.125 |
| qwen3 | material | sand | `replication_direct` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -21.000 | 0.087 |
| qwen3 | material | sand | `replication_direct` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 9.000 | 0.175 |
| qwen3 | material | sand | `replication_sentence` | `flip` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 1.125 | 8.000 | -0.125 |
| qwen3 | material | sand | `replication_sentence` | `half` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.375 | 4.000 | -0.037 |
| qwen3 | material | sand | `replication_sentence` | `scale_up` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.750 | -4.000 | 0.037 |
| qwen3 | material | sand | `replication_sentence` | `zero` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.625 | 5.000 | -0.050 |
| qwen3 | material | sand | `replication_form` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 5.000 | -0.037 |
| qwen3 | material | sand | `replication_form` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 4.000 | -0.037 |
| qwen3 | material | sand | `replication_form` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.006 |
| qwen3 | material | sand | `replication_form` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 2.000 | -0.013 |
| deepseek7b | animal | tiger | `replication_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.875 | 0.000 | 0.000 |
| deepseek7b | animal | tiger | `replication_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.188 | 0.000 | 0.000 |
| deepseek7b | animal | tiger | `replication_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -2.562 | -5.000 | 0.000 |
| deepseek7b | animal | tiger | `replication_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.438 | 0.000 | 0.000 |
| deepseek7b | animal | tiger | `replication_sentence` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | True | True | False | 0/0 | 0.562 | 1.000 | -0.031 |
| deepseek7b | animal | tiger | `replication_sentence` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| deepseek7b | animal | tiger | `replication_sentence` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.250 | -3.000 | 0.031 |
| deepseek7b | animal | tiger | `replication_sentence` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | -0.031 |
| deepseek7b | animal | tiger | `replication_form` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.500 | 0.000 | 0.000 |
| deepseek7b | animal | tiger | `replication_form` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | tiger | `replication_form` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | tiger | `replication_form` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| deepseek7b | animal | dolphin | `replication_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | dolphin | `replication_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | dolphin | `replication_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | dolphin | `replication_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | dolphin | `replication_sentence` | `flip` | `source_clean_failed` | `['format_dominates']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.016 |
| deepseek7b | animal | dolphin | `replication_sentence` | `half` | `source_clean_failed` | `['format_dominates']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.016 |
| deepseek7b | animal | dolphin | `replication_sentence` | `scale_up` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 0/0 | -0.062 | 0.000 | -0.047 |
| deepseek7b | animal | dolphin | `replication_sentence` | `zero` | `source_clean_failed` | `['format_dominates']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | dolphin | `replication_form` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.062 | 0.000 | -0.062 |
| deepseek7b | animal | dolphin | `replication_form` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | dolphin | `replication_form` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | -0.062 |
| deepseek7b | animal | dolphin | `replication_form` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | -0.062 |
| deepseek7b | animal | sparrow | `replication_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_sentence` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_sentence` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_sentence` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_sentence` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_form` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.062 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_form` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_form` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | sparrow | `replication_form` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | turtle | `replication_direct` | `flip` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -1.000 | 0.000 |
| deepseek7b | animal | turtle | `replication_direct` | `half` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.031 |
| deepseek7b | animal | turtle | `replication_direct` | `scale_up` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.006 |
| deepseek7b | animal | turtle | `replication_direct` | `zero` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -1.000 | 0.000 |
| deepseek7b | animal | turtle | `replication_sentence` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.375 | 0.000 | 0.188 |
| deepseek7b | animal | turtle | `replication_sentence` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.188 |
| deepseek7b | animal | turtle | `replication_sentence` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | -0.062 |
| deepseek7b | animal | turtle | `replication_sentence` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.188 | 0.000 | 0.125 |
| deepseek7b | animal | turtle | `replication_form` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.438 | 0.000 | 0.000 |
| deepseek7b | animal | turtle | `replication_form` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| deepseek7b | animal | turtle | `replication_form` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | 0.000 |
| deepseek7b | animal | turtle | `replication_form` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.188 | 0.000 | 0.000 |
| deepseek7b | animal | goat | `replication_direct` | `flip` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure']` | False | False | False | False | 1/0 | 1.875 | 1.000 | 0.125 |
| deepseek7b | animal | goat | `replication_direct` | `half` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure']` | False | False | False | False | 0/0 | 0.500 | 0.000 | 0.062 |
| deepseek7b | animal | goat | `replication_direct` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure']` | False | False | False | False | 0/0 | -0.875 | -1.000 | -0.062 |
| deepseek7b | animal | goat | `replication_direct` | `zero` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure']` | False | False | False | False | 0/0 | 0.938 | 0.000 | 0.062 |
| deepseek7b | animal | goat | `replication_sentence` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.688 | 0.000 | 0.000 |
| deepseek7b | animal | goat | `replication_sentence` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.375 | 0.000 | 0.000 |
| deepseek7b | animal | goat | `replication_sentence` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -0.812 | -1.000 | 0.000 |
| deepseek7b | animal | goat | `replication_sentence` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.875 | 0.000 | 0.000 |
| deepseek7b | animal | goat | `replication_form` | `flip` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.250 | -1.000 | -0.062 |
| deepseek7b | animal | goat | `replication_form` | `half` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | -0.031 |
| deepseek7b | animal | goat | `replication_form` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.062 | 0.000 | -0.031 |
| deepseek7b | animal | goat | `replication_form` | `zero` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.188 | -1.000 | -0.094 |
| deepseek7b | animal | bear | `replication_direct` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.500 | -11.000 | 0.083 |
| deepseek7b | animal | bear | `replication_direct` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| deepseek7b | animal | bear | `replication_direct` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | True | True | False | 0/0 | 0.188 | 1.000 | -0.062 |
| deepseek7b | animal | bear | `replication_direct` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.219 | -5.000 | 0.028 |
| deepseek7b | animal | bear | `replication_sentence` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 3.438 | 0.000 | 0.000 |
| deepseek7b | animal | bear | `replication_sentence` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.875 | 0.000 | 0.000 |
| deepseek7b | animal | bear | `replication_sentence` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.750 | -2.000 | 0.000 |
| deepseek7b | animal | bear | `replication_sentence` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.688 | 0.000 | 0.000 |
| deepseek7b | animal | bear | `replication_form` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.188 | 0.000 | 0.000 |
| deepseek7b | animal | bear | `replication_form` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | bear | `replication_form` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.188 | 0.000 | 0.000 |
| deepseek7b | animal | bear | `replication_form` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| deepseek7b | color | red | `replication_direct` | `flip` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 0.688 | 2.000 | 0.062 |
| deepseek7b | color | red | `replication_direct` | `half` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 1.000 | 0.000 |
| deepseek7b | color | red | `replication_direct` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.188 | -1.000 | -0.031 |
| deepseek7b | color | red | `replication_direct` | `zero` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.250 | 1.000 | 0.031 |
| deepseek7b | color | red | `replication_sentence` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.875 | 0.000 | 0.000 |
| deepseek7b | color | red | `replication_sentence` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.500 | 0.000 | 0.000 |
| deepseek7b | color | red | `replication_sentence` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.062 | -2.000 | 0.000 |
| deepseek7b | color | red | `replication_sentence` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.125 | 0.000 | 0.000 |
| deepseek7b | color | red | `replication_form` | `flip` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.125 | 0.000 | -0.131 |
| deepseek7b | color | red | `replication_form` | `half` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.013 |
| deepseek7b | color | red | `replication_form` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | -0.081 |
| deepseek7b | color | red | `replication_form` | `zero` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.006 |
| deepseek7b | color | green | `replication_direct` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.719 | 76.000 | 0.031 |
| deepseek7b | color | green | `replication_direct` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.219 | 24.000 | 0.006 |
| deepseek7b | color | green | `replication_direct` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.438 | -75.000 | -0.013 |
| deepseek7b | color | green | `replication_direct` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.406 | 46.000 | 0.000 |
| deepseek7b | color | green | `replication_sentence` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.062 | 1.000 | 0.188 |
| deepseek7b | color | green | `replication_sentence` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.188 | 0.000 | 0.000 |
| deepseek7b | color | green | `replication_sentence` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.312 | 0.000 | -0.094 |
| deepseek7b | color | green | `replication_sentence` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.375 | 0.000 | -0.031 |
| deepseek7b | color | green | `replication_form` | `flip` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | -1.500 | -13.000 | -0.562 |
| deepseek7b | color | green | `replication_form` | `half` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | -1.688 | -14.000 | -0.688 |
| deepseek7b | color | green | `replication_form` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | -0.042 |
| deepseek7b | color | green | `replication_form` | `zero` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | -1.562 | -13.000 | -0.625 |
| deepseek7b | color | yellow | `replication_direct` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.812 | 17.000 | -0.013 |
| deepseek7b | color | yellow | `replication_direct` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.188 | 4.000 | 0.013 |
| deepseek7b | color | yellow | `replication_direct` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.500 | -11.000 | -0.006 |
| deepseek7b | color | yellow | `replication_direct` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.375 | 9.000 | 0.000 |
| deepseek7b | color | yellow | `replication_sentence` | `flip` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.500 | 4.000 | 0.019 |
| deepseek7b | color | yellow | `replication_sentence` | `half` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.250 | 2.000 | 0.006 |
| deepseek7b | color | yellow | `replication_sentence` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.625 | -2.000 | -0.031 |
| deepseek7b | color | yellow | `replication_sentence` | `zero` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.438 | 3.000 | 0.006 |
| deepseek7b | color | yellow | `replication_form` | `flip` | `stable_nonclean` | `['semantic_other_pressure']` | True | True | True | False | 0/0 | 0.125 | 2.000 | -0.014 |
| deepseek7b | color | yellow | `replication_form` | `half` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.062 | -1.000 | 0.000 |
| deepseek7b | color | yellow | `replication_form` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.062 | 1.000 | 0.014 |
| deepseek7b | color | yellow | `replication_form` | `zero` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | -1.000 | 0.035 |
| deepseek7b | color | pink | `replication_direct` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.125 | 0.000 | 0.000 |
| deepseek7b | color | pink | `replication_direct` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.500 | 0.000 | 0.000 |
| deepseek7b | color | pink | `replication_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -0.875 | -1.000 | 0.000 |
| deepseek7b | color | pink | `replication_direct` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.000 | 0.000 | 0.000 |
| deepseek7b | color | pink | `replication_sentence` | `flip` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 0/0 | -0.375 | 0.000 | -0.031 |
| deepseek7b | color | pink | `replication_sentence` | `half` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | color | pink | `replication_sentence` | `scale_up` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.031 |
| deepseek7b | color | pink | `replication_sentence` | `zero` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| deepseek7b | color | pink | `replication_form` | `flip` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.062 | -1.000 | 0.069 |
| deepseek7b | color | pink | `replication_form` | `half` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.062 | 0.000 | 0.006 |
| deepseek7b | color | pink | `replication_form` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | -1.000 | -0.006 |
| deepseek7b | color | pink | `replication_form` | `zero` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.062 | 0.000 | 0.019 |
| deepseek7b | color | silver | `replication_direct` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.156 | -29.000 | 0.000 |
| deepseek7b | color | silver | `replication_direct` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.156 | -30.000 | 0.025 |
| deepseek7b | color | silver | `replication_direct` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.281 | 32.000 | 0.031 |
| deepseek7b | color | silver | `replication_direct` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.156 | -30.000 | 0.019 |
| deepseek7b | color | silver | `replication_sentence` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.375 | 66.000 | -0.006 |
| deepseek7b | color | silver | `replication_sentence` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.094 | 20.000 | 0.000 |
| deepseek7b | color | silver | `replication_sentence` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.219 | -37.000 | 0.006 |
| deepseek7b | color | silver | `replication_sentence` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.188 | 38.000 | -0.006 |
| deepseek7b | color | silver | `replication_form` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.125 | 5.000 | 0.069 |
| deepseek7b | color | silver | `replication_form` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | -2.000 | 0.019 |
| deepseek7b | color | silver | `replication_form` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | -5.000 | 0.025 |
| deepseek7b | color | silver | `replication_form` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.125 | 5.000 | 0.081 |
| deepseek7b | color | beige | `replication_direct` | `flip` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | 0.013 |
| deepseek7b | color | beige | `replication_direct` | `half` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 1.000 | -0.031 |
| deepseek7b | color | beige | `replication_direct` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 2.000 | -0.056 |
| deepseek7b | color | beige | `replication_direct` | `zero` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.006 |
| deepseek7b | color | beige | `replication_sentence` | `flip` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.250 | -2.000 | -0.006 |
| deepseek7b | color | beige | `replication_sentence` | `half` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | -0.006 |
| deepseek7b | color | beige | `replication_sentence` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.125 | 4.000 | 0.000 |
| deepseek7b | color | beige | `replication_sentence` | `zero` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| deepseek7b | color | beige | `replication_form` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| deepseek7b | color | beige | `replication_form` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.125 | -1.000 | -0.006 |
| deepseek7b | color | beige | `replication_form` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.025 |
| deepseek7b | color | beige | `replication_form` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.013 |
