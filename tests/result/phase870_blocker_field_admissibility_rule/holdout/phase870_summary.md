# Phase 870 Blocker Field Admissibility Rule (holdout)

- Source: Phase 867 paired original/intervened rows.
- Boundary: single-context rule audit, not model training and not closure.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_predict_clean_mixed` | `target_clean_transition` | 144 | 1 | 71 | 3 | 69 | 0.014 | 0.250 | 0.486 |
| `source_predict_clean_mixed` | `target_output_clean_transition` | 144 | 3 | 69 | 9 | 63 | 0.042 | 0.250 | 0.458 |
| `phase866_pair_rule` | `target_clean_transition` | 144 | 4 | 16 | 0 | 124 | 0.200 | 1.000 | 0.889 |
| `phase866_pair_rule` | `target_output_clean_transition` | 144 | 4 | 16 | 8 | 116 | 0.200 | 0.333 | 0.833 |
| `field_base_admissible` | `target_clean_transition` | 144 | 4 | 64 | 0 | 76 | 0.059 | 1.000 | 0.556 |
| `field_base_admissible` | `target_output_clean_transition` | 144 | 10 | 58 | 2 | 74 | 0.147 | 0.833 | 0.583 |
| `field_plus_effect_rule` | `target_clean_transition` | 144 | 4 | 1 | 0 | 139 | 0.800 | 1.000 | 0.993 |
| `field_plus_effect_rule` | `target_output_clean_transition` | 144 | 4 | 1 | 8 | 131 | 0.800 | 0.333 | 0.938 |
| `field_strict_admissible` | `target_clean_transition` | 144 | 4 | 48 | 0 | 92 | 0.077 | 1.000 | 0.667 |
| `field_strict_admissible` | `target_output_clean_transition` | 144 | 4 | 48 | 8 | 84 | 0.077 | 0.333 | 0.611 |
| `field_strict_plus_effect_rule` | `target_clean_transition` | 144 | 4 | 0 | 0 | 140 | 1.000 | 1.000 | 1.000 |
| `field_strict_plus_effect_rule` | `target_output_clean_transition` | 144 | 4 | 0 | 8 | 132 | 1.000 | 0.333 | 0.944 |

## Summary

- Transfer status counts: `{'source_clean_failed': 71, 'stable_nonclean': 69, 'stable_clean': 1, 'emergent_clean': 3}`
- Field tag counts: `{'too_many_blockers': 40, 'object_dominates_class': 60, 'format_dominates': 68, 'object_echo_pressure': 60, 'semantic_other_pressure': 68, 'protocol_pressure': 48, 'field_low_pressure': 52}`

## Pair Rows

| model | domain | object | prompt | mode | status | field tags | field ok | phase866 | field+effect | target clean | clear gain/loss | ans | block red. | orig block |
|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| qwen3 | material | stone | `holdout_category_short` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.562 | 26.000 | 0.062 |
| qwen3 | material | stone | `holdout_category_short` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.250 | 10.000 | 0.050 |
| qwen3 | material | stone | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.188 | -19.000 | 0.075 |
| qwen3 | material | stone | `holdout_category_short` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.312 | 16.000 | 0.025 |
| qwen3 | material | stone | `holdout_kind_phrase` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | True | True | False | 0/0 | 1.000 | 4.000 | -0.075 |
| qwen3 | material | stone | `holdout_kind_phrase` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.250 | 2.000 | 0.025 |
| qwen3 | material | stone | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.625 | -1.000 | 0.025 |
| qwen3 | material | stone | `holdout_kind_phrase` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.625 | 2.000 | 0.050 |
| qwen3 | material | stone | `holdout_label` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.006 |
| qwen3 | material | stone | `holdout_label` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.013 |
| qwen3 | material | stone | `holdout_label` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -1.000 | 0.031 |
| qwen3 | material | stone | `holdout_label` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -1.000 | 0.006 |
| qwen3 | material | paper | `holdout_category_short` | `flip` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | False | False | False | 0/0 | 0.750 | 4.000 | 0.037 |
| qwen3 | material | paper | `holdout_category_short` | `half` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | True | False | False | 0/0 | 0.125 | 1.000 | -0.025 |
| qwen3 | material | paper | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | False | False | False | 0/0 | -0.562 | -4.000 | -0.075 |
| qwen3 | material | paper | `holdout_category_short` | `zero` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | True | False | False | 0/0 | 0.375 | 4.000 | -0.075 |
| qwen3 | material | paper | `holdout_kind_phrase` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.000 | 0.000 | 0.000 |
| qwen3 | material | paper | `holdout_kind_phrase` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.375 | 0.000 | 0.000 |
| qwen3 | material | paper | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.750 | 0.000 | 0.000 |
| qwen3 | material | paper | `holdout_kind_phrase` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.625 | 0.000 | 0.000 |
| qwen3 | material | paper | `holdout_label` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.188 | 46.000 | -0.044 |
| qwen3 | material | paper | `holdout_label` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 20.000 | -0.031 |
| qwen3 | material | paper | `holdout_label` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -14.000 | -0.025 |
| qwen3 | material | paper | `holdout_label` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 20.000 | -0.031 |
| qwen3 | material | cotton | `holdout_category_short` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 1.062 | 19.000 | 0.100 |
| qwen3 | material | cotton | `holdout_category_short` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.250 | 7.000 | 0.000 |
| qwen3 | material | cotton | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.500 | -13.000 | 0.013 |
| qwen3 | material | cotton | `holdout_category_short` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.500 | 13.000 | -0.037 |
| qwen3 | material | cotton | `holdout_kind_phrase` | `flip` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 1.438 | 13.000 | -0.075 |
| qwen3 | material | cotton | `holdout_kind_phrase` | `half` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.312 | 2.000 | -0.050 |
| qwen3 | material | cotton | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -1.000 | -12.000 | 0.025 |
| qwen3 | material | cotton | `holdout_kind_phrase` | `zero` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.812 | 6.000 | -0.025 |
| qwen3 | material | cotton | `holdout_label` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 24.000 | 0.000 |
| qwen3 | material | cotton | `holdout_label` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 9.000 | -0.019 |
| qwen3 | material | cotton | `holdout_label` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.188 | -20.000 | -0.013 |
| qwen3 | material | cotton | `holdout_label` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.125 | 14.000 | -0.006 |
| qwen3 | material | rubber | `holdout_category_short` | `flip` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | False | False | False | 0/0 | 0.625 | 3.000 | 0.125 |
| qwen3 | material | rubber | `holdout_category_short` | `half` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | False | False | False | 0/0 | 0.125 | 1.000 | 0.062 |
| qwen3 | material | rubber | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | False | False | False | 0/0 | -0.312 | -4.000 | -0.025 |
| qwen3 | material | rubber | `holdout_category_short` | `zero` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | False | False | False | 0/0 | 0.250 | 2.000 | 0.075 |
| qwen3 | material | rubber | `holdout_kind_phrase` | `flip` | `stable_clean` | `['field_low_pressure']` | True | True | True | True | 1/0 | 1.000 | 1.000 | -0.125 |
| qwen3 | material | rubber | `holdout_kind_phrase` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.375 | 0.000 | 0.000 |
| qwen3 | material | rubber | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.625 | -2.000 | 0.000 |
| qwen3 | material | rubber | `holdout_kind_phrase` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.625 | 0.000 | 0.000 |
| qwen3 | material | rubber | `holdout_label` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 2.000 | -0.013 |
| qwen3 | material | rubber | `holdout_label` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -2.000 | 0.000 |
| qwen3 | material | rubber | `holdout_label` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -1.000 | -0.006 |
| qwen3 | material | rubber | `holdout_label` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 2.000 | 0.013 |
| deepseek7b | animal | horse | `holdout_category_short` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 3.062 | 0.000 | 0.000 |
| deepseek7b | animal | horse | `holdout_category_short` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.750 | 0.000 | 0.000 |
| deepseek7b | animal | horse | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.625 | -1.000 | 0.000 |
| deepseek7b | animal | horse | `holdout_category_short` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.562 | 0.000 | 0.000 |
| deepseek7b | animal | horse | `holdout_kind_phrase` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 3.000 | 0.000 | 0.000 |
| deepseek7b | animal | horse | `holdout_kind_phrase` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.750 | 0.000 | 0.000 |
| deepseek7b | animal | horse | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -1.438 | 0.000 | 0.000 |
| deepseek7b | animal | horse | `holdout_kind_phrase` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.500 | 0.000 | 0.000 |
| deepseek7b | animal | horse | `holdout_label` | `flip` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.500 | -5.000 | -0.042 |
| deepseek7b | animal | horse | `holdout_label` | `half` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.125 | 0.000 | -0.049 |
| deepseek7b | animal | horse | `holdout_label` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.312 | 4.000 | -0.007 |
| deepseek7b | animal | horse | `holdout_label` | `zero` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.250 | -2.000 | -0.028 |
| deepseek7b | animal | cow | `holdout_category_short` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.812 | 0.000 | 0.000 |
| deepseek7b | animal | cow | `holdout_category_short` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.188 | 0.000 | 0.000 |
| deepseek7b | animal | cow | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -2.375 | -3.000 | 0.000 |
| deepseek7b | animal | cow | `holdout_category_short` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.438 | 0.000 | 0.000 |
| deepseek7b | animal | cow | `holdout_kind_phrase` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.250 | 0.000 | 0.000 |
| deepseek7b | animal | cow | `holdout_kind_phrase` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.125 | 0.000 | 0.000 |
| deepseek7b | animal | cow | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -2.188 | 0.000 | 0.000 |
| deepseek7b | animal | cow | `holdout_kind_phrase` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.125 | 0.000 | 0.000 |
| deepseek7b | animal | cow | `holdout_label` | `flip` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.188 | 1.000 | 0.181 |
| deepseek7b | animal | cow | `holdout_label` | `half` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 0.000 | 0.081 |
| deepseek7b | animal | cow | `holdout_label` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | -0.037 |
| deepseek7b | animal | cow | `holdout_label` | `zero` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.069 |
| deepseek7b | animal | lion | `holdout_category_short` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 5.250 | 0.000 | 0.000 |
| deepseek7b | animal | lion | `holdout_category_short` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.312 | 0.000 | 0.000 |
| deepseek7b | animal | lion | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -2.688 | -1.000 | 0.000 |
| deepseek7b | animal | lion | `holdout_category_short` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.625 | 0.000 | 0.000 |
| deepseek7b | animal | lion | `holdout_kind_phrase` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.250 | 0.000 | 0.000 |
| deepseek7b | animal | lion | `holdout_kind_phrase` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.625 | 0.000 | 0.000 |
| deepseek7b | animal | lion | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.188 | -1.000 | 0.000 |
| deepseek7b | animal | lion | `holdout_kind_phrase` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.125 | 0.000 | 0.000 |
| deepseek7b | animal | lion | `holdout_label` | `flip` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.500 | 8.000 | 0.006 |
| deepseek7b | animal | lion | `holdout_label` | `half` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 1.000 | 0.013 |
| deepseek7b | animal | lion | `holdout_label` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.188 | -2.000 | 0.006 |
| deepseek7b | animal | lion | `holdout_label` | `zero` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 3.000 | 0.000 |
| deepseek7b | animal | whale | `holdout_category_short` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 3.438 | 1.000 | 0.062 |
| deepseek7b | animal | whale | `holdout_category_short` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.750 | 0.000 | 0.000 |
| deepseek7b | animal | whale | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | -0.062 |
| deepseek7b | animal | whale | `holdout_category_short` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 1.625 | 1.000 | 0.062 |
| deepseek7b | animal | whale | `holdout_kind_phrase` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.312 | 0.000 | 0.000 |
| deepseek7b | animal | whale | `holdout_kind_phrase` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.562 | 0.000 | 0.000 |
| deepseek7b | animal | whale | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.125 | -2.000 | 0.000 |
| deepseek7b | animal | whale | `holdout_kind_phrase` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.125 | 0.000 | 0.000 |
| deepseek7b | animal | whale | `holdout_label` | `flip` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 0.000 | 0.094 |
| deepseek7b | animal | whale | `holdout_label` | `half` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 2.000 | 0.275 |
| deepseek7b | animal | whale | `holdout_label` | `scale_up` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.188 | 1.000 | 0.231 |
| deepseek7b | animal | whale | `holdout_label` | `zero` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 0.000 | 0.094 |
| deepseek7b | color | purple | `holdout_category_short` | `flip` | `emergent_clean` | `['field_low_pressure']` | True | True | True | True | 1/0 | 7.125 | 2.000 | -0.062 |
| deepseek7b | color | purple | `holdout_category_short` | `half` | `emergent_clean` | `['field_low_pressure']` | True | True | True | True | 1/0 | 1.188 | 2.000 | -0.031 |
| deepseek7b | color | purple | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -1.375 | -1.000 | -0.031 |
| deepseek7b | color | purple | `holdout_category_short` | `zero` | `emergent_clean` | `['field_low_pressure']` | True | True | True | True | 1/0 | 2.688 | 2.000 | -0.062 |
| deepseek7b | color | purple | `holdout_kind_phrase` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.312 | 0.000 | 0.000 |
| deepseek7b | color | purple | `holdout_kind_phrase` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.062 | 0.000 | 0.000 |
| deepseek7b | color | purple | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.938 | -5.000 | 0.000 |
| deepseek7b | color | purple | `holdout_kind_phrase` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.062 | 0.000 | 0.000 |
| deepseek7b | color | purple | `holdout_label` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.031 | 0.000 | -0.013 |
| deepseek7b | color | purple | `holdout_label` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -2.000 | -0.031 |
| deepseek7b | color | purple | `holdout_label` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | -1.000 | 0.075 |
| deepseek7b | color | purple | `holdout_label` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.156 | -1.000 | -0.125 |
| deepseek7b | color | orange | `holdout_category_short` | `flip` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 4.188 | 3.000 | 0.042 |
| deepseek7b | color | orange | `holdout_category_short` | `half` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 0.750 | 3.000 | 0.021 |
| deepseek7b | color | orange | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.938 | -2.000 | 0.042 |
| deepseek7b | color | orange | `holdout_category_short` | `zero` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 1.688 | 3.000 | 0.042 |
| deepseek7b | color | orange | `holdout_kind_phrase` | `flip` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 2.625 | 3.000 | 0.000 |
| deepseek7b | color | orange | `holdout_kind_phrase` | `half` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.500 | 1.000 | 0.000 |
| deepseek7b | color | orange | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.750 | -3.000 | 0.000 |
| deepseek7b | color | orange | `holdout_kind_phrase` | `zero` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 1.125 | 2.000 | 0.000 |
| deepseek7b | color | orange | `holdout_label` | `flip` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | -0.019 |
| deepseek7b | color | orange | `holdout_label` | `half` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | -0.050 |
| deepseek7b | color | orange | `holdout_label` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | -0.025 |
| deepseek7b | color | orange | `holdout_label` | `zero` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -1.000 | 0.037 |
| deepseek7b | color | black | `holdout_category_short` | `flip` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 1/0 | 5.062 | 3.000 | 0.000 |
| deepseek7b | color | black | `holdout_category_short` | `half` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 0/0 | 0.875 | 0.000 | 0.000 |
| deepseek7b | color | black | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 0/0 | -1.188 | -8.000 | -0.042 |
| deepseek7b | color | black | `holdout_category_short` | `zero` | `stable_nonclean` | `['format_dominates']` | False | False | False | False | 0/0 | 1.938 | 1.000 | 0.000 |
| deepseek7b | color | black | `holdout_kind_phrase` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.625 | 0.000 | 0.000 |
| deepseek7b | color | black | `holdout_kind_phrase` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.625 | 0.000 | 0.000 |
| deepseek7b | color | black | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.938 | 0.000 | 0.000 |
| deepseek7b | color | black | `holdout_kind_phrase` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.250 | 0.000 | 0.000 |
| deepseek7b | color | black | `holdout_label` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.069 |
| deepseek7b | color | black | `holdout_label` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.062 |
| deepseek7b | color | black | `holdout_label` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.031 | 2.000 | 0.000 |
| deepseek7b | color | black | `holdout_label` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.069 |
| deepseek7b | color | white | `holdout_category_short` | `flip` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | False | False | False | 1/0 | 3.312 | 5.000 | 0.037 |
| deepseek7b | color | white | `holdout_category_short` | `half` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | True | False | False | 0/0 | 0.562 | 2.000 | -0.013 |
| deepseek7b | color | white | `holdout_category_short` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | False | False | False | 0/0 | -0.312 | -5.000 | -0.025 |
| deepseek7b | color | white | `holdout_category_short` | `zero` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure']` | False | True | False | False | 0/0 | 1.312 | 3.000 | -0.013 |
| deepseek7b | color | white | `holdout_kind_phrase` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.688 | 0.000 | 0.000 |
| deepseek7b | color | white | `holdout_kind_phrase` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.562 | 0.000 | 0.000 |
| deepseek7b | color | white | `holdout_kind_phrase` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -0.812 | -1.000 | 0.000 |
| deepseek7b | color | white | `holdout_kind_phrase` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.188 | 0.000 | 0.000 |
| deepseek7b | color | white | `holdout_label` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 1.000 | 0.006 |
| deepseek7b | color | white | `holdout_label` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -1.000 | 0.019 |
| deepseek7b | color | white | `holdout_label` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 1.000 | 0.006 |
| deepseek7b | color | white | `holdout_label` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.019 |
