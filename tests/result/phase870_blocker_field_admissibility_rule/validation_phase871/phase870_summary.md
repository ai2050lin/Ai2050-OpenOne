# Phase 870 Blocker Field Admissibility Rule (validation)

- Source: Phase 867 paired original/intervened rows.
- Boundary: single-context rule audit, not model training and not closure.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_predict_clean_mixed` | `target_clean_transition` | 144 | 0 | 72 | 1 | 71 | 0.000 | 0.000 | 0.493 |
| `source_predict_clean_mixed` | `target_output_clean_transition` | 144 | 0 | 72 | 7 | 65 | 0.000 | 0.000 | 0.451 |
| `phase866_pair_rule` | `target_clean_transition` | 144 | 1 | 11 | 0 | 132 | 0.083 | 1.000 | 0.924 |
| `phase866_pair_rule` | `target_output_clean_transition` | 144 | 1 | 11 | 6 | 126 | 0.083 | 0.143 | 0.882 |
| `field_base_admissible` | `target_clean_transition` | 144 | 1 | 87 | 0 | 56 | 0.011 | 1.000 | 0.396 |
| `field_base_admissible` | `target_output_clean_transition` | 144 | 3 | 85 | 4 | 52 | 0.034 | 0.429 | 0.382 |
| `field_plus_effect_rule` | `target_clean_transition` | 144 | 1 | 1 | 0 | 142 | 0.500 | 1.000 | 0.993 |
| `field_plus_effect_rule` | `target_output_clean_transition` | 144 | 1 | 1 | 6 | 136 | 0.500 | 0.143 | 0.951 |
| `field_strict_admissible` | `target_clean_transition` | 144 | 1 | 63 | 0 | 80 | 0.016 | 1.000 | 0.562 |
| `field_strict_admissible` | `target_output_clean_transition` | 144 | 1 | 63 | 6 | 74 | 0.016 | 0.143 | 0.521 |
| `field_strict_plus_effect_rule` | `target_clean_transition` | 144 | 1 | 1 | 0 | 142 | 0.500 | 1.000 | 0.993 |
| `field_strict_plus_effect_rule` | `target_output_clean_transition` | 144 | 1 | 1 | 6 | 136 | 0.500 | 0.143 | 0.951 |

## Summary

- Transfer status counts: `{'source_clean_failed': 72, 'stable_nonclean': 71, 'emergent_clean': 1}`
- Field tag counts: `{'format_dominates': 48, 'too_many_blockers': 40, 'object_dominates_class': 28, 'semantic_other_pressure': 72, 'protocol_pressure': 48, 'field_low_pressure': 64, 'object_echo_pressure': 12}`

## Pair Rows

| model | domain | object | prompt | mode | status | field tags | field ok | phase866 | field+effect | target clean | clear gain/loss | ans | block red. | orig block |
|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| qwen3 | material | ceramic | `validation_direct` | `flip` | `source_clean_failed` | `['format_dominates']` | False | True | False | False | 0/0 | 0.500 | 1.000 | -0.042 |
| qwen3 | material | ceramic | `validation_direct` | `half` | `source_clean_failed` | `['format_dominates']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | ceramic | `validation_direct` | `scale_up` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 0/0 | -0.500 | -2.000 | 0.000 |
| qwen3 | material | ceramic | `validation_direct` | `zero` | `source_clean_failed` | `['format_dominates']` | False | False | False | False | 0/0 | 0.250 | 1.000 | 0.000 |
| qwen3 | material | ceramic | `validation_question` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 14.000 | 0.075 |
| qwen3 | material | ceramic | `validation_question` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 3.000 | 0.075 |
| qwen3 | material | ceramic | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -10.000 | 0.025 |
| qwen3 | material | ceramic | `validation_question` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.075 |
| qwen3 | material | ceramic | `validation_table` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.438 | 0.000 | 0.000 |
| qwen3 | material | ceramic | `validation_table` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | ceramic | `validation_table` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -0.250 | -1.000 | 0.000 |
| qwen3 | material | ceramic | `validation_table` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.188 | 0.000 | 0.000 |
| qwen3 | material | leather | `validation_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | 0.000 |
| qwen3 | material | leather | `validation_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | leather | `validation_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -0.125 | 0.000 | 0.000 |
| qwen3 | material | leather | `validation_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | leather | `validation_question` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 5.000 | 0.013 |
| qwen3 | material | leather | `validation_question` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 7.000 | 0.000 |
| qwen3 | material | leather | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 3.000 | 0.013 |
| qwen3 | material | leather | `validation_question` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 4.000 | 0.000 |
| qwen3 | material | leather | `validation_table` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | leather | `validation_table` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | leather | `validation_table` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| qwen3 | material | leather | `validation_table` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | silk | `validation_direct` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | silk | `validation_direct` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| qwen3 | material | silk | `validation_direct` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | silk | `validation_direct` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | silk | `validation_question` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 14.000 | 0.025 |
| qwen3 | material | silk | `validation_question` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 7.000 | -0.031 |
| qwen3 | material | silk | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.125 | -3.000 | -0.100 |
| qwen3 | material | silk | `validation_question` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -3.000 | 0.037 |
| qwen3 | material | silk | `validation_table` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | silk | `validation_table` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | silk | `validation_table` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | silk | `validation_table` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | concrete | `validation_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | True | True | False | 0/0 | 0.625 | 2.000 | -0.062 |
| qwen3 | material | concrete | `validation_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.250 | 1.000 | 0.000 |
| qwen3 | material | concrete | `validation_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.375 | -1.000 | 0.062 |
| qwen3 | material | concrete | `validation_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.375 | 1.000 | 0.000 |
| qwen3 | material | concrete | `validation_question` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 41.000 | -0.025 |
| qwen3 | material | concrete | `validation_question` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 16.000 | -0.037 |
| qwen3 | material | concrete | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -5.000 | -0.013 |
| qwen3 | material | concrete | `validation_question` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 35.000 | -0.037 |
| qwen3 | material | concrete | `validation_table` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.625 | 0.000 | 0.000 |
| qwen3 | material | concrete | `validation_table` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.062 |
| qwen3 | material | concrete | `validation_table` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.375 | 0.000 | 0.062 |
| qwen3 | material | concrete | `validation_table` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.375 | 0.000 | 0.062 |
| deepseek7b | animal | elephant | `validation_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 8.062 | 0.000 | 0.000 |
| deepseek7b | animal | elephant | `validation_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.062 | 0.000 | 0.000 |
| deepseek7b | animal | elephant | `validation_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -3.500 | -3.000 | 0.000 |
| deepseek7b | animal | elephant | `validation_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.188 | 0.000 | 0.000 |
| deepseek7b | animal | elephant | `validation_question` | `flip` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.625 | 3.000 | 0.000 |
| deepseek7b | animal | elephant | `validation_question` | `half` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| deepseek7b | animal | elephant | `validation_question` | `scale_up` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.312 | -3.000 | 0.006 |
| deepseek7b | animal | elephant | `validation_question` | `zero` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 0.000 | -0.025 |
| deepseek7b | animal | elephant | `validation_table` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.188 | 0.000 | 0.000 |
| deepseek7b | animal | elephant | `validation_table` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.062 | 0.000 | 0.000 |
| deepseek7b | animal | elephant | `validation_table` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -2.062 | 0.000 | 0.000 |
| deepseek7b | animal | elephant | `validation_table` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.062 | 0.000 | 0.000 |
| deepseek7b | animal | rabbit | `validation_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 8.000 | 0.000 | 0.000 |
| deepseek7b | animal | rabbit | `validation_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.125 | 0.000 | 0.000 |
| deepseek7b | animal | rabbit | `validation_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -4.125 | -27.000 | 0.000 |
| deepseek7b | animal | rabbit | `validation_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.125 | 0.000 | 0.000 |
| deepseek7b | animal | rabbit | `validation_question` | `flip` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.812 | 10.000 | 0.031 |
| deepseek7b | animal | rabbit | `validation_question` | `half` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 7.000 | 0.050 |
| deepseek7b | animal | rabbit | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.375 | -11.000 | -0.006 |
| deepseek7b | animal | rabbit | `validation_question` | `zero` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.438 | 9.000 | 0.025 |
| deepseek7b | animal | rabbit | `validation_table` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 2.375 | 1.000 | 0.156 |
| deepseek7b | animal | rabbit | `validation_table` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.562 | 1.000 | 0.031 |
| deepseek7b | animal | rabbit | `validation_table` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -1.188 | -5.000 | -0.156 |
| deepseek7b | animal | rabbit | `validation_table` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 1.250 | 1.000 | 0.094 |
| deepseek7b | animal | shark | `validation_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 5.438 | 0.000 | 0.000 |
| deepseek7b | animal | shark | `validation_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.375 | 0.000 | 0.000 |
| deepseek7b | animal | shark | `validation_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -2.812 | -6.000 | 0.000 |
| deepseek7b | animal | shark | `validation_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.688 | 0.000 | 0.000 |
| deepseek7b | animal | shark | `validation_question` | `flip` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.125 | 6.000 | -0.006 |
| deepseek7b | animal | shark | `validation_question` | `half` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 2.000 | -0.006 |
| deepseek7b | animal | shark | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | -0.019 |
| deepseek7b | animal | shark | `validation_question` | `zero` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 4.000 | -0.019 |
| deepseek7b | animal | shark | `validation_table` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.312 | 0.000 | 0.000 |
| deepseek7b | animal | shark | `validation_table` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.375 | 0.000 | 0.000 |
| deepseek7b | animal | shark | `validation_table` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.688 | 0.000 | 0.000 |
| deepseek7b | animal | shark | `validation_table` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.688 | 0.000 | 0.000 |
| deepseek7b | animal | eagle | `validation_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 7.562 | 0.000 | 0.000 |
| deepseek7b | animal | eagle | `validation_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.938 | 0.000 | 0.000 |
| deepseek7b | animal | eagle | `validation_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.062 | -1.000 | 0.000 |
| deepseek7b | animal | eagle | `validation_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 3.812 | 0.000 | 0.000 |
| deepseek7b | animal | eagle | `validation_question` | `flip` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.037 |
| deepseek7b | animal | eagle | `validation_question` | `half` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.025 |
| deepseek7b | animal | eagle | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.025 |
| deepseek7b | animal | eagle | `validation_question` | `zero` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.050 |
| deepseek7b | animal | eagle | `validation_table` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.250 | 0.000 | 0.000 |
| deepseek7b | animal | eagle | `validation_table` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.562 | 0.000 | 0.000 |
| deepseek7b | animal | eagle | `validation_table` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.562 | 0.000 | 0.000 |
| deepseek7b | animal | eagle | `validation_table` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.125 | 0.000 | 0.000 |
| deepseek7b | color | cyan | `validation_direct` | `flip` | `emergent_clean` | `['field_low_pressure']` | True | True | True | True | 1/0 | 2.188 | 1.000 | -0.062 |
| deepseek7b | color | cyan | `validation_direct` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.438 | 0.000 | 0.000 |
| deepseek7b | color | cyan | `validation_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.875 | -6.000 | 0.000 |
| deepseek7b | color | cyan | `validation_direct` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.875 | 0.000 | -0.062 |
| deepseek7b | color | cyan | `validation_question` | `flip` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.019 |
| deepseek7b | color | cyan | `validation_question` | `half` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -1.000 | 0.019 |
| deepseek7b | color | cyan | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.188 | 6.000 | -0.025 |
| deepseek7b | color | cyan | `validation_question` | `zero` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -1.000 | 0.019 |
| deepseek7b | color | cyan | `validation_table` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.500 | -2.000 | 0.000 |
| deepseek7b | color | cyan | `validation_table` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -0.250 | -1.000 | 0.000 |
| deepseek7b | color | cyan | `validation_table` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.750 | 0.000 | 0.000 |
| deepseek7b | color | cyan | `validation_table` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -0.625 | -2.000 | 0.000 |
| deepseek7b | color | magenta | `validation_direct` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 5.188 | 0.000 | 0.000 |
| deepseek7b | color | magenta | `validation_direct` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.188 | 0.000 | 0.000 |
| deepseek7b | color | magenta | `validation_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.812 | -8.000 | 0.000 |
| deepseek7b | color | magenta | `validation_direct` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.438 | 0.000 | 0.000 |
| deepseek7b | color | magenta | `validation_question` | `flip` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.375 | 7.000 | -0.050 |
| deepseek7b | color | magenta | `validation_question` | `half` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.006 |
| deepseek7b | color | magenta | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | 1.000 | -0.081 |
| deepseek7b | color | magenta | `validation_question` | `zero` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 2.000 | -0.031 |
| deepseek7b | color | magenta | `validation_table` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 3.750 | 0.000 | 0.000 |
| deepseek7b | color | magenta | `validation_table` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.562 | 0.000 | 0.000 |
| deepseek7b | color | magenta | `validation_table` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.625 | 0.000 | 0.000 |
| deepseek7b | color | magenta | `validation_table` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.375 | 0.000 | 0.000 |
| deepseek7b | color | brown | `validation_direct` | `flip` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 2.500 | 4.000 | 0.016 |
| deepseek7b | color | brown | `validation_direct` | `half` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.562 | 3.000 | 0.000 |
| deepseek7b | color | brown | `validation_direct` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -1.062 | -1.000 | -0.031 |
| deepseek7b | color | brown | `validation_direct` | `zero` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 1.188 | 3.000 | 0.000 |
| deepseek7b | color | brown | `validation_question` | `flip` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | 0.188 | 0.000 | 0.134 |
| deepseek7b | color | brown | `validation_question` | `half` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.054 |
| deepseek7b | color | brown | `validation_question` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | 0.116 |
| deepseek7b | color | brown | `validation_question` | `zero` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.107 |
| deepseek7b | color | brown | `validation_table` | `flip` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 1.688 | 4.000 | 0.047 |
| deepseek7b | color | brown | `validation_table` | `half` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.375 | 2.000 | 0.031 |
| deepseek7b | color | brown | `validation_table` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.375 | -4.000 | -0.047 |
| deepseek7b | color | brown | `validation_table` | `zero` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.688 | 2.000 | 0.031 |
| deepseek7b | color | gray | `validation_direct` | `flip` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure']` | False | False | False | False | 1/0 | 1.875 | 1.000 | 0.062 |
| deepseek7b | color | gray | `validation_direct` | `half` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure']` | False | False | False | False | 1/0 | 0.438 | 1.000 | 0.062 |
| deepseek7b | color | gray | `validation_direct` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure']` | False | False | False | False | 0/0 | -0.750 | -2.000 | 0.000 |
| deepseek7b | color | gray | `validation_direct` | `zero` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure']` | False | False | False | False | 1/0 | 0.938 | 1.000 | 0.000 |
| deepseek7b | color | gray | `validation_question` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.562 | -8.000 | -0.031 |
| deepseek7b | color | gray | `validation_question` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.125 | -3.000 | 0.013 |
| deepseek7b | color | gray | `validation_question` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.312 | 3.000 | -0.019 |
| deepseek7b | color | gray | `validation_question` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.250 | -4.000 | 0.006 |
| deepseek7b | color | gray | `validation_table` | `flip` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 1/0 | 0.375 | 3.000 | 0.042 |
| deepseek7b | color | gray | `validation_table` | `half` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.125 | 1.000 | 0.000 |
| deepseek7b | color | gray | `validation_table` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.125 | 0.000 | 0.042 |
| deepseek7b | color | gray | `validation_table` | `zero` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.188 | 1.000 | 0.021 |
