# Phase 870 Blocker Field Admissibility Rule (validation)

- Source: Phase 867 paired original/intervened rows.
- Boundary: single-context rule audit, not model training and not closure.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_predict_clean_mixed` | `target_clean_transition` | 288 | 1 | 143 | 3 | 141 | 0.007 | 0.250 | 0.493 |
| `source_predict_clean_mixed` | `target_output_clean_transition` | 288 | 9 | 135 | 3 | 141 | 0.062 | 0.750 | 0.521 |
| `phase866_pair_rule` | `target_clean_transition` | 288 | 4 | 40 | 0 | 244 | 0.091 | 1.000 | 0.861 |
| `phase866_pair_rule` | `target_output_clean_transition` | 288 | 4 | 40 | 8 | 236 | 0.091 | 0.333 | 0.833 |
| `field_base_admissible` | `target_clean_transition` | 288 | 4 | 136 | 0 | 148 | 0.029 | 1.000 | 0.528 |
| `field_base_admissible` | `target_output_clean_transition` | 288 | 12 | 128 | 0 | 148 | 0.086 | 1.000 | 0.556 |
| `field_plus_effect_rule` | `target_clean_transition` | 288 | 4 | 8 | 0 | 276 | 0.333 | 1.000 | 0.972 |
| `field_plus_effect_rule` | `target_output_clean_transition` | 288 | 4 | 8 | 8 | 268 | 0.333 | 0.333 | 0.944 |
| `field_strict_admissible` | `target_clean_transition` | 288 | 0 | 72 | 4 | 212 | 0.000 | 0.000 | 0.736 |
| `field_strict_admissible` | `target_output_clean_transition` | 288 | 2 | 70 | 10 | 206 | 0.028 | 0.167 | 0.722 |
| `field_strict_plus_effect_rule` | `target_clean_transition` | 288 | 0 | 0 | 4 | 284 | 0.000 | 0.000 | 0.986 |
| `field_strict_plus_effect_rule` | `target_output_clean_transition` | 288 | 0 | 0 | 12 | 276 | 0.000 | 0.000 | 0.958 |

## Summary

- Transfer status counts: `{'source_clean_failed': 143, 'stable_nonclean': 141, 'emergent_clean': 3, 'stable_clean': 1}`
- Field tag counts: `{'field_low_pressure': 68, 'too_many_blockers': 124, 'format_dominates': 92, 'semantic_other_pressure': 196, 'protocol_pressure': 100, 'object_dominates_class': 80, 'object_echo_pressure': 28}`

## Pair Rows

| model | domain | object | prompt | mode | status | field tags | field ok | phase866 | field+effect | target clean | clear gain/loss | ans | block red. | orig block |
|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| qwen3 | material | wood | `nonclean_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.500 | 0.000 | 0.000 |
| qwen3 | material | wood | `nonclean_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | wood | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.625 | 0.000 | 0.000 |
| qwen3 | material | wood | `nonclean_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | 0.000 |
| qwen3 | material | wood | `semantic_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.688 | 15.000 | -0.037 |
| qwen3 | material | wood | `semantic_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.188 | 4.000 | -0.037 |
| qwen3 | material | wood | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.312 | -7.000 | 0.025 |
| qwen3 | material | wood | `semantic_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.375 | 7.000 | -0.013 |
| qwen3 | material | wood | `echo_pressure` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.500 | 0.000 | 0.125 |
| qwen3 | material | wood | `echo_pressure` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | wood | `echo_pressure` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.250 | 0.000 | -0.062 |
| qwen3 | material | wood | `echo_pressure` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | 0.000 |
| qwen3 | material | wood | `format_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | -0.080 |
| qwen3 | material | wood | `format_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.062 | 0.000 | -0.018 |
| qwen3 | material | wood | `format_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.188 | -4.000 | -0.018 |
| qwen3 | material | wood | `format_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | -0.018 |
| qwen3 | material | brick | `nonclean_direct` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.625 | 0.000 | -0.125 |
| qwen3 | material | brick | `nonclean_direct` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | -0.042 |
| qwen3 | material | brick | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.250 | 0.000 | -0.042 |
| qwen3 | material | brick | `nonclean_direct` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | -0.083 |
| qwen3 | material | brick | `semantic_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.125 | 3.000 | -0.013 |
| qwen3 | material | brick | `semantic_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.050 |
| qwen3 | material | brick | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -1.000 | 0.050 |
| qwen3 | material | brick | `semantic_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 0.000 | 0.087 |
| qwen3 | material | brick | `echo_pressure` | `flip` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.375 | 3.000 | -0.037 |
| qwen3 | material | brick | `echo_pressure` | `half` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.125 | 3.000 | -0.062 |
| qwen3 | material | brick | `echo_pressure` | `scale_up` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.125 | 0.000 | 0.037 |
| qwen3 | material | brick | `echo_pressure` | `zero` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 3.000 | 0.000 |
| qwen3 | material | brick | `format_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | 0.000 |
| qwen3 | material | brick | `format_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.025 |
| qwen3 | material | brick | `format_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.125 | -2.000 | 0.000 |
| qwen3 | material | brick | `format_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | copper | `nonclean_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| qwen3 | material | copper | `nonclean_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `nonclean_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `semantic_pressure` | `flip` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.025 |
| qwen3 | material | copper | `semantic_pressure` | `half` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| qwen3 | material | copper | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `semantic_pressure` | `zero` | `source_clean_failed` | `['format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `echo_pressure` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.062 |
| qwen3 | material | copper | `echo_pressure` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `echo_pressure` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | copper | `echo_pressure` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `format_pressure` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `format_pressure` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `format_pressure` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | copper | `format_pressure` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | nylon | `nonclean_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | nylon | `nonclean_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | nylon | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| qwen3 | material | nylon | `nonclean_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | nylon | `semantic_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.875 | 21.000 | 0.000 |
| qwen3 | material | nylon | `semantic_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.312 | 6.000 | 0.013 |
| qwen3 | material | nylon | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.500 | -13.000 | 0.000 |
| qwen3 | material | nylon | `semantic_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.500 | 11.000 | 0.000 |
| qwen3 | material | nylon | `echo_pressure` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | nylon | `echo_pressure` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | nylon | `echo_pressure` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | nylon | `echo_pressure` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | nylon | `format_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.188 | 1.000 | 0.031 |
| qwen3 | material | nylon | `format_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.062 | 0.000 | 0.000 |
| qwen3 | material | nylon | `format_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.062 | -1.000 | 0.000 |
| qwen3 | material | nylon | `format_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.062 | 0.000 | 0.010 |
| qwen3 | material | granite | `nonclean_direct` | `flip` | `source_clean_failed` | `['too_many_blockers', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 1.750 | 11.000 | -0.075 |
| qwen3 | material | granite | `nonclean_direct` | `half` | `source_clean_failed` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.500 | 4.000 | 0.025 |
| qwen3 | material | granite | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.875 | -17.000 | 0.087 |
| qwen3 | material | granite | `nonclean_direct` | `zero` | `source_clean_failed` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 1.000 | 8.000 | 0.013 |
| qwen3 | material | granite | `semantic_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.562 | 142.000 | 0.000 |
| qwen3 | material | granite | `semantic_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.125 | 42.000 | -0.037 |
| qwen3 | material | granite | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.188 | -43.000 | -0.025 |
| qwen3 | material | granite | `semantic_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.250 | 73.000 | -0.013 |
| qwen3 | material | granite | `echo_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure', 'protocol_pressure']` | True | True | True | False | 0/0 | 0.500 | 4.000 | -0.196 |
| qwen3 | material | granite | `echo_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | 0.250 | 1.000 | 0.054 |
| qwen3 | material | granite | `echo_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | -0.375 | -3.000 | -0.071 |
| qwen3 | material | granite | `echo_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure', 'protocol_pressure']` | True | True | True | False | 0/0 | 0.250 | 2.000 | -0.089 |
| qwen3 | material | granite | `format_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 1.250 | 17.000 | -0.019 |
| qwen3 | material | granite | `format_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.312 | 6.000 | -0.006 |
| qwen3 | material | granite | `format_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.625 | -9.000 | 0.000 |
| qwen3 | material | granite | `format_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.625 | 7.000 | 0.000 |
| qwen3 | material | wax | `nonclean_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.250 | 0.000 | 0.000 |
| qwen3 | material | wax | `nonclean_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | wax | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.000 |
| qwen3 | material | wax | `nonclean_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | 0.000 |
| qwen3 | material | wax | `semantic_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.625 | 16.000 | 0.000 |
| qwen3 | material | wax | `semantic_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 1.000 | 0.025 |
| qwen3 | material | wax | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.375 | -20.000 | 0.000 |
| qwen3 | material | wax | `semantic_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.312 | 9.000 | 0.013 |
| qwen3 | material | wax | `echo_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 1.000 | 0.000 |
| qwen3 | material | wax | `echo_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.083 |
| qwen3 | material | wax | `echo_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | -0.042 |
| qwen3 | material | wax | `echo_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| qwen3 | material | wax | `format_pressure` | `flip` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.312 | 3.000 | -0.013 |
| qwen3 | material | wax | `format_pressure` | `half` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.125 | 2.000 | -0.006 |
| qwen3 | material | wax | `format_pressure` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.188 | 0.000 | -0.025 |
| qwen3 | material | wax | `format_pressure` | `zero` | `source_clean_failed` | `['object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.125 | 2.000 | -0.006 |
| deepseek7b | animal | seal | `nonclean_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.188 | 0.000 | 0.000 |
| deepseek7b | animal | seal | `nonclean_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.562 | 0.000 | 0.000 |
| deepseek7b | animal | seal | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.062 | -1.000 | 0.000 |
| deepseek7b | animal | seal | `nonclean_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.188 | 0.000 | 0.000 |
| deepseek7b | animal | seal | `semantic_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.750 | 10.000 | 0.044 |
| deepseek7b | animal | seal | `semantic_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 1.000 | 0.000 |
| deepseek7b | animal | seal | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.375 | -6.000 | -0.019 |
| deepseek7b | animal | seal | `semantic_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.312 | 5.000 | 0.013 |
| deepseek7b | animal | seal | `echo_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | 1.000 | 5.000 | 0.009 |
| deepseek7b | animal | seal | `echo_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure', 'protocol_pressure']` | True | True | True | False | 0/0 | 0.250 | 2.000 | -0.009 |
| deepseek7b | animal | seal | `echo_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure', 'protocol_pressure']` | True | False | False | False | 0/0 | -0.500 | -2.000 | -0.027 |
| deepseek7b | animal | seal | `echo_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure', 'protocol_pressure']` | True | True | True | False | 0/0 | 0.500 | 4.000 | -0.009 |
| deepseek7b | animal | seal | `format_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -2.625 | 0.000 | -0.062 |
| deepseek7b | animal | seal | `format_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.688 | 0.000 | -0.062 |
| deepseek7b | animal | seal | `format_pressure` | `scale_up` | `emergent_clean` | `['semantic_other_pressure']` | True | True | True | True | 1/0 | 1.312 | 1.000 | -0.062 |
| deepseek7b | animal | seal | `format_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -1.312 | 0.000 | -0.062 |
| deepseek7b | animal | bat | `nonclean_direct` | `flip` | `stable_clean` | `['semantic_other_pressure']` | True | True | True | True | 1/0 | 1.438 | 2.000 | -0.062 |
| deepseek7b | animal | bat | `nonclean_direct` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | True | True | False | 0/0 | 0.312 | 1.000 | -0.031 |
| deepseek7b | animal | bat | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.812 | -3.000 | -0.062 |
| deepseek7b | animal | bat | `nonclean_direct` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | True | True | False | 0/0 | 0.625 | 1.000 | -0.062 |
| deepseek7b | animal | bat | `semantic_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.812 | 10.000 | -0.006 |
| deepseek7b | animal | bat | `semantic_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.188 | 3.000 | 0.019 |
| deepseek7b | animal | bat | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.438 | -2.000 | -0.013 |
| deepseek7b | animal | bat | `semantic_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.375 | 5.000 | -0.019 |
| deepseek7b | animal | bat | `echo_pressure` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.438 | 0.000 | 0.000 |
| deepseek7b | animal | bat | `echo_pressure` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.625 | 0.000 | 0.000 |
| deepseek7b | animal | bat | `echo_pressure` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.250 | -3.000 | 0.000 |
| deepseek7b | animal | bat | `echo_pressure` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.188 | 0.000 | 0.000 |
| deepseek7b | animal | bat | `format_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.562 | 0.000 | -0.031 |
| deepseek7b | animal | bat | `format_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.188 | 0.000 | -0.031 |
| deepseek7b | animal | bat | `format_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.125 | 0.000 | -0.062 |
| deepseek7b | animal | bat | `format_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.312 | 0.000 | -0.031 |
| deepseek7b | animal | salmon | `nonclean_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| deepseek7b | animal | salmon | `nonclean_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | salmon | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | salmon | `nonclean_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | salmon | `semantic_pressure` | `flip` | `source_clean_failed` | `['format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| deepseek7b | animal | salmon | `semantic_pressure` | `half` | `source_clean_failed` | `['format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.013 |
| deepseek7b | animal | salmon | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.025 |
| deepseek7b | animal | salmon | `semantic_pressure` | `zero` | `source_clean_failed` | `['format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.013 |
| deepseek7b | animal | salmon | `echo_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.031 |
| deepseek7b | animal | salmon | `echo_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | salmon | `echo_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | salmon | `echo_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | salmon | `format_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.125 | 0.000 | 0.013 |
| deepseek7b | animal | salmon | `format_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| deepseek7b | animal | salmon | `format_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.062 | 1.000 | 0.000 |
| deepseek7b | animal | salmon | `format_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.062 | 0.000 | 0.025 |
| deepseek7b | animal | turkey | `nonclean_direct` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.375 | -2.000 | 0.062 |
| deepseek7b | animal | turkey | `nonclean_direct` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.125 | -1.000 | 0.000 |
| deepseek7b | animal | turkey | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | True | True | False | 0/0 | 0.125 | 1.000 | -0.042 |
| deepseek7b | animal | turkey | `nonclean_direct` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.188 | -2.000 | 0.021 |
| deepseek7b | animal | turkey | `semantic_pressure` | `flip` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 4.000 | 0.050 |
| deepseek7b | animal | turkey | `semantic_pressure` | `half` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -1.000 | 0.050 |
| deepseek7b | animal | turkey | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -1.000 | 0.000 |
| deepseek7b | animal | turkey | `semantic_pressure` | `zero` | `source_clean_failed` | `['object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.044 |
| deepseek7b | animal | turkey | `echo_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.312 | 8.000 | 0.044 |
| deepseek7b | animal | turkey | `echo_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.062 | 1.000 | 0.013 |
| deepseek7b | animal | turkey | `echo_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.188 | -1.000 | 0.013 |
| deepseek7b | animal | turkey | `echo_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 5.000 | 0.019 |
| deepseek7b | animal | turkey | `format_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -1.375 | -52.000 | 0.050 |
| deepseek7b | animal | turkey | `format_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.312 | -7.000 | 0.037 |
| deepseek7b | animal | turkey | `format_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.750 | 9.000 | -0.019 |
| deepseek7b | animal | turkey | `format_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.688 | -14.000 | 0.006 |
| deepseek7b | animal | sheep | `nonclean_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 3.188 | 0.000 | 0.000 |
| deepseek7b | animal | sheep | `nonclean_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.812 | 0.000 | 0.000 |
| deepseek7b | animal | sheep | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -1.688 | -5.000 | 0.000 |
| deepseek7b | animal | sheep | `nonclean_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.625 | 0.000 | 0.000 |
| deepseek7b | animal | sheep | `semantic_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.438 | 8.000 | 0.100 |
| deepseek7b | animal | sheep | `semantic_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.062 | 2.000 | -0.013 |
| deepseek7b | animal | sheep | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.188 | -3.000 | 0.000 |
| deepseek7b | animal | sheep | `semantic_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 5.000 | 0.062 |
| deepseek7b | animal | sheep | `echo_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 3.188 | 2.000 | 0.000 |
| deepseek7b | animal | sheep | `echo_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 0.875 | 2.000 | 0.031 |
| deepseek7b | animal | sheep | `echo_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -1.562 | -1.000 | -0.031 |
| deepseek7b | animal | sheep | `echo_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 1.625 | 2.000 | 0.000 |
| deepseek7b | animal | sheep | `format_pressure` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.062 | 0.000 | 0.000 |
| deepseek7b | animal | sheep | `format_pressure` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | animal | sheep | `format_pressure` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| deepseek7b | animal | sheep | `format_pressure` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.062 | 0.000 | 0.000 |
| deepseek7b | animal | wolf | `nonclean_direct` | `flip` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.812 | 0.000 | 0.000 |
| deepseek7b | animal | wolf | `nonclean_direct` | `half` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.312 | 0.000 | 0.000 |
| deepseek7b | animal | wolf | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -2.562 | -6.000 | 0.000 |
| deepseek7b | animal | wolf | `nonclean_direct` | `zero` | `source_clean_failed` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.438 | 0.000 | 0.000 |
| deepseek7b | animal | wolf | `semantic_pressure` | `flip` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.562 | 8.000 | 0.031 |
| deepseek7b | animal | wolf | `semantic_pressure` | `half` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.125 | 2.000 | -0.006 |
| deepseek7b | animal | wolf | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.312 | -3.000 | -0.056 |
| deepseek7b | animal | wolf | `semantic_pressure` | `zero` | `source_clean_failed` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.250 | 3.000 | -0.006 |
| deepseek7b | animal | wolf | `echo_pressure` | `flip` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 4.750 | 1.000 | 0.125 |
| deepseek7b | animal | wolf | `echo_pressure` | `half` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 1.188 | 1.000 | 0.062 |
| deepseek7b | animal | wolf | `echo_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -2.438 | -4.000 | -0.125 |
| deepseek7b | animal | wolf | `echo_pressure` | `zero` | `source_clean_failed` | `['semantic_other_pressure']` | True | False | False | False | 1/0 | 2.375 | 1.000 | 0.125 |
| deepseek7b | animal | wolf | `format_pressure` | `flip` | `source_clean_failed` | `['protocol_pressure']` | True | False | False | False | 1/0 | 0.812 | 1.000 | 0.062 |
| deepseek7b | animal | wolf | `format_pressure` | `half` | `source_clean_failed` | `['protocol_pressure']` | True | False | False | False | 0/0 | 0.188 | 0.000 | 0.062 |
| deepseek7b | animal | wolf | `format_pressure` | `scale_up` | `stable_nonclean` | `['protocol_pressure']` | True | False | False | False | 0/0 | -0.375 | -1.000 | 0.000 |
| deepseek7b | animal | wolf | `format_pressure` | `zero` | `source_clean_failed` | `['protocol_pressure']` | True | False | False | False | 1/0 | 0.438 | 1.000 | 0.062 |
| deepseek7b | color | gold | `nonclean_direct` | `flip` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.125 | -1.000 | 0.000 |
| deepseek7b | color | gold | `nonclean_direct` | `half` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| deepseek7b | color | gold | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.125 | 4.000 | 0.000 |
| deepseek7b | color | gold | `nonclean_direct` | `zero` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | 0.000 | 0.000 |
| deepseek7b | color | gold | `semantic_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -4.000 | 0.000 |
| deepseek7b | color | gold | `semantic_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -2.000 | 0.000 |
| deepseek7b | color | gold | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | 0.000 |
| deepseek7b | color | gold | `semantic_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | -1.000 | 0.000 |
| deepseek7b | color | gold | `echo_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.125 | 26.000 | 0.006 |
| deepseek7b | color | gold | `echo_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.031 | 8.000 | -0.006 |
| deepseek7b | color | gold | `echo_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -22.000 | -0.013 |
| deepseek7b | color | gold | `echo_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.094 | 23.000 | 0.000 |
| deepseek7b | color | gold | `format_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.125 | 22.000 | -0.031 |
| deepseek7b | color | gold | `format_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.013 |
| deepseek7b | color | gold | `format_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.125 | -23.000 | -0.025 |
| deepseek7b | color | gold | `format_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.062 | 14.000 | -0.025 |
| deepseek7b | color | violet | `nonclean_direct` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 6.438 | 0.000 | 0.000 |
| deepseek7b | color | violet | `nonclean_direct` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.562 | 0.000 | 0.000 |
| deepseek7b | color | violet | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/1 | -2.812 | -2.000 | 0.000 |
| deepseek7b | color | violet | `nonclean_direct` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 3.062 | 0.000 | 0.000 |
| deepseek7b | color | violet | `semantic_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.812 | -57.000 | 0.006 |
| deepseek7b | color | violet | `semantic_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.188 | -13.000 | -0.006 |
| deepseek7b | color | violet | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.406 | 11.000 | -0.013 |
| deepseek7b | color | violet | `semantic_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'object_echo_pressure', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.406 | -27.000 | 0.000 |
| deepseek7b | color | violet | `echo_pressure` | `flip` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 1.125 | 2.000 | 0.042 |
| deepseek7b | color | violet | `echo_pressure` | `half` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.250 | 1.000 | 0.000 |
| deepseek7b | color | violet | `echo_pressure` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.438 | -3.000 | -0.021 |
| deepseek7b | color | violet | `echo_pressure` | `zero` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | 0.500 | 2.000 | 0.000 |
| deepseek7b | color | violet | `format_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.375 | 17.000 | 0.044 |
| deepseek7b | color | violet | `format_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.062 | 5.000 | 0.013 |
| deepseek7b | color | violet | `format_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.188 | -10.000 | -0.025 |
| deepseek7b | color | violet | `format_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.125 | 11.000 | 0.006 |
| deepseek7b | color | teal | `nonclean_direct` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.625 | 0.000 | 0.000 |
| deepseek7b | color | teal | `nonclean_direct` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 0.938 | 0.000 | 0.000 |
| deepseek7b | color | teal | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -1.500 | 0.000 | 0.000 |
| deepseek7b | color | teal | `nonclean_direct` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.125 | 0.000 | 0.000 |
| deepseek7b | color | teal | `semantic_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.438 | -127.000 | -0.006 |
| deepseek7b | color | teal | `semantic_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.125 | -38.000 | 0.006 |
| deepseek7b | color | teal | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.250 | 37.000 | 0.006 |
| deepseek7b | color | teal | `semantic_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.219 | -68.000 | 0.006 |
| deepseek7b | color | teal | `echo_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.500 | 7.000 | -0.031 |
| deepseek7b | color | teal | `echo_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.188 | 1.000 | 0.056 |
| deepseek7b | color | teal | `echo_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.188 | -5.000 | -0.019 |
| deepseek7b | color | teal | `echo_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.188 | 1.000 | -0.025 |
| deepseek7b | color | teal | `format_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.188 | 4.000 | 0.081 |
| deepseek7b | color | teal | `format_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.062 | 2.000 | -0.013 |
| deepseek7b | color | teal | `format_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.250 | -8.000 | -0.037 |
| deepseek7b | color | teal | `format_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.188 | 6.000 | 0.025 |
| deepseek7b | color | maroon | `nonclean_direct` | `flip` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 4.375 | 0.000 | 0.000 |
| deepseek7b | color | maroon | `nonclean_direct` | `half` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 1.125 | 0.000 | 0.000 |
| deepseek7b | color | maroon | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | -2.125 | 0.000 | 0.000 |
| deepseek7b | color | maroon | `nonclean_direct` | `zero` | `stable_nonclean` | `['field_low_pressure']` | True | False | False | False | 0/0 | 2.250 | 0.000 | 0.000 |
| deepseek7b | color | maroon | `semantic_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.438 | -7.000 | 0.037 |
| deepseek7b | color | maroon | `semantic_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.062 | -1.000 | 0.044 |
| deepseek7b | color | maroon | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.312 | 13.000 | 0.056 |
| deepseek7b | color | maroon | `semantic_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.250 | -4.000 | 0.013 |
| deepseek7b | color | maroon | `echo_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.625 | 7.000 | -0.025 |
| deepseek7b | color | maroon | `echo_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.188 | 2.000 | 0.006 |
| deepseek7b | color | maroon | `echo_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.250 | -9.000 | 0.006 |
| deepseek7b | color | maroon | `echo_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.312 | 4.000 | -0.006 |
| deepseek7b | color | maroon | `format_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.844 | 90.000 | 0.163 |
| deepseek7b | color | maroon | `format_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.156 | 25.000 | 0.006 |
| deepseek7b | color | maroon | `format_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.438 | -47.000 | -0.138 |
| deepseek7b | color | maroon | `format_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.312 | 50.000 | 0.013 |
| deepseek7b | color | navy | `nonclean_direct` | `flip` | `emergent_clean` | `['semantic_other_pressure']` | True | True | True | True | 1/0 | 1.438 | 3.000 | -0.042 |
| deepseek7b | color | navy | `nonclean_direct` | `half` | `stable_nonclean` | `['semantic_other_pressure']` | True | True | True | False | 0/0 | 0.312 | 2.000 | -0.021 |
| deepseek7b | color | navy | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['semantic_other_pressure']` | True | False | False | False | 0/0 | -0.562 | -1.000 | -0.042 |
| deepseek7b | color | navy | `nonclean_direct` | `zero` | `emergent_clean` | `['semantic_other_pressure']` | True | True | True | True | 1/0 | 0.688 | 3.000 | -0.042 |
| deepseek7b | color | navy | `semantic_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 3.000 | -0.006 |
| deepseek7b | color | navy | `semantic_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.016 | 18.000 | 0.000 |
| deepseek7b | color | navy | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.016 | 17.000 | 0.000 |
| deepseek7b | color | navy | `semantic_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.016 | 18.000 | 0.000 |
| deepseek7b | color | navy | `echo_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.031 | -2.000 | -0.019 |
| deepseek7b | color | navy | `echo_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 1.000 | -0.019 |
| deepseek7b | color | navy | `echo_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | 0.000 | 0.000 | -0.006 |
| deepseek7b | color | navy | `echo_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.031 | -2.000 | -0.006 |
| deepseek7b | color | navy | `format_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 2.000 | -0.025 |
| deepseek7b | color | navy | `format_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | -1.000 | 0.019 |
| deepseek7b | color | navy | `format_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | -6.000 | 0.000 |
| deepseek7b | color | navy | `format_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'object_echo_pressure', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.000 | 3.000 | -0.006 |
| deepseek7b | color | ivory | `nonclean_direct` | `flip` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -1.125 | -11.000 | -0.019 |
| deepseek7b | color | ivory | `nonclean_direct` | `half` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.312 | -1.000 | -0.013 |
| deepseek7b | color | ivory | `nonclean_direct` | `scale_up` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.500 | 4.000 | 0.000 |
| deepseek7b | color | ivory | `nonclean_direct` | `zero` | `stable_nonclean` | `['format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.562 | -3.000 | -0.013 |
| deepseek7b | color | ivory | `semantic_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.500 | -315.000 | -0.013 |
| deepseek7b | color | ivory | `semantic_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.125 | -65.000 | -0.006 |
| deepseek7b | color | ivory | `semantic_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | True | False | False | 0/0 | 0.250 | 110.000 | -0.013 |
| deepseek7b | color | ivory | `semantic_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure', 'protocol_pressure']` | False | False | False | False | 0/0 | -0.219 | -151.000 | 0.006 |
| deepseek7b | color | ivory | `echo_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.188 | 144.000 | 0.044 |
| deepseek7b | color | ivory | `echo_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.031 | 26.000 | 0.000 |
| deepseek7b | color | ivory | `echo_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | -57.000 | 0.031 |
| deepseek7b | color | ivory | `echo_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'format_dominates', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.094 | 69.000 | 0.037 |
| deepseek7b | color | ivory | `format_pressure` | `flip` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.094 | 29.000 | -0.013 |
| deepseek7b | color | ivory | `format_pressure` | `half` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | 0.031 | 5.000 | 0.013 |
| deepseek7b | color | ivory | `format_pressure` | `scale_up` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | False | False | False | 0/0 | -0.062 | -23.000 | 0.013 |
| deepseek7b | color | ivory | `format_pressure` | `zero` | `stable_nonclean` | `['too_many_blockers', 'object_dominates_class', 'semantic_other_pressure']` | False | True | False | False | 0/0 | 0.031 | 12.000 | -0.006 |
