# Phase 878 Full-Vocabulary Blocker Displacement Audit

- Boundary: offline audit using saved Phase876 top-token/top-blocker rows; no new model run.
- Goal: check whether original_blocker_not_reduced hides top blocker-field displacement.

## Summary

- Rows: `12`
- Pair found: `12`
- Transition classes: `{'clean_causal_transition': 4, 'nonclean_output_transition': 8}`
- Routes: `{'clean_causal_transition': 4, 'format_recovery': 3, 'semantic_pressure_transition': 3, 'protocol_pressure_transition': 2}`
- Nonclean displacement: `{'n': 8, 'blocker_set_changed': 8, 'top_set_changed': 4, 'count_reduced_without_original_blocker_reduction': 8, 'target_rank_reached_top1': 8, 'mean_blocker_count_reduction_raw': 1.375, 'mean_target_rank_improvement': 1.375, 'mean_target_logit_delta_raw': 1.90625, 'mean_original_blocker_delta': 0.05859375}`
- Clean reference: `{'n': 4, 'blocker_set_changed': 4, 'top_set_changed': 1, 'target_rank_reached_top1': 4, 'mean_blocker_count_reduction_raw': 2.25, 'mean_target_rank_improvement': 2.25, 'mean_target_logit_delta_raw': 1.21875, 'mean_original_blocker_delta': -0.05208333333333333}`

## By Route

| route | n | candidates | objects | prompts | mean blocker red. | mean rank improve | mean target logit delta | mean orig blocker | blocker set changed | top set changed | target top1 |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `clean_causal_transition` | 4 | `{'L27C16651+L24C3875:scale_up': 1, 'L27C16651+L24C3875:flip': 1, 'L27C15369+L26C8587:flip': 1, 'L27C15369+L26C8587:zero': 1}` | `['bat', 'navy', 'seal']` | `['format_pressure', 'nonclean_direct']` | 2.250 | 2.250 | 1.219 | -0.052 | 4 | 1 | 4 |
| `format_recovery` | 3 | `{'L27C16651+L24C3875:flip': 1, 'L27C16651+L24C3875:half': 1, 'L27C16651+L24C3875:zero': 1}` | `['sheep']` | `['echo_pressure']` | 2.000 | 2.000 | 1.896 | 0.010 | 3 | 1 | 3 |
| `protocol_pressure_transition` | 2 | `{'L27C16651+L24C3875:flip': 1, 'L27C16651+L24C3875:zero': 1}` | `['wolf']` | `['format_pressure']` | 1.000 | 1.000 | 0.625 | 0.062 | 2 | 2 | 2 |
| `semantic_pressure_transition` | 3 | `{'L27C16651+L24C3875:flip': 1, 'L27C16651+L24C3875:half': 1, 'L27C16651+L24C3875:zero': 1}` | `['wolf']` | `['echo_pressure']` | 1.000 | 1.000 | 2.771 | 0.104 | 3 | 1 | 3 |

## Rows

| class | route | object | prompt | candidate | label | count red. | rank improve | logit delta | orig blocker | top1 raw | blocker changed | top changed | target top1 |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| `clean_causal_transition` | `clean_causal_transition` | seal | `format_pressure` | `L27C16651+L24C3875:scale_up` | `other->strict_canonical` | 1.000 | 1.000 | 1.312 | -0.062 | `other_blocker->strict_target` | True | True | True |
| `clean_causal_transition` | `clean_causal_transition` | bat | `nonclean_direct` | `L27C16651+L24C3875:flip` | `other->strict_canonical` | 2.000 | 2.000 | 1.438 | -0.062 | `other_blocker->strict_target` | True | False | True |
| `nonclean_output_transition` | `format_recovery` | sheep | `echo_pressure` | `L27C16651+L24C3875:flip` | `format_or_empty->answer_alias` | 2.000 | 2.000 | 3.188 | 0.000 | `other_blocker->strict_target` | True | True | True |
| `nonclean_output_transition` | `format_recovery` | sheep | `echo_pressure` | `L27C16651+L24C3875:half` | `format_or_empty->answer_alias` | 2.000 | 2.000 | 0.875 | 0.031 | `other_blocker->strict_target` | True | False | True |
| `nonclean_output_transition` | `format_recovery` | sheep | `echo_pressure` | `L27C16651+L24C3875:zero` | `format_or_empty->answer_alias` | 2.000 | 2.000 | 1.625 | 0.000 | `other_blocker->strict_target` | True | False | True |
| `nonclean_output_transition` | `semantic_pressure_transition` | wolf | `echo_pressure` | `L27C16651+L24C3875:flip` | `other->answer_alias` | 1.000 | 1.000 | 4.750 | 0.125 | `other_blocker->strict_target` | True | True | True |
| `nonclean_output_transition` | `semantic_pressure_transition` | wolf | `echo_pressure` | `L27C16651+L24C3875:half` | `other->answer_alias` | 1.000 | 1.000 | 1.188 | 0.062 | `other_blocker->strict_target` | True | False | True |
| `nonclean_output_transition` | `semantic_pressure_transition` | wolf | `echo_pressure` | `L27C16651+L24C3875:zero` | `other->answer_alias` | 1.000 | 1.000 | 2.375 | 0.125 | `other_blocker->strict_target` | True | False | True |
| `nonclean_output_transition` | `protocol_pressure_transition` | wolf | `format_pressure` | `L27C16651+L24C3875:flip` | `other->strict_canonical` | 1.000 | 1.000 | 0.812 | 0.062 | `protocol_word->strict_target` | True | True | True |
| `nonclean_output_transition` | `protocol_pressure_transition` | wolf | `format_pressure` | `L27C16651+L24C3875:zero` | `other->strict_canonical` | 1.000 | 1.000 | 0.438 | 0.062 | `protocol_word->strict_target` | True | True | True |
| `clean_causal_transition` | `clean_causal_transition` | navy | `nonclean_direct` | `L27C15369+L26C8587:flip` | `other->strict_canonical` | 3.000 | 3.000 | 1.438 | -0.042 | `other_blocker->strict_target` | True | False | True |
| `clean_causal_transition` | `clean_causal_transition` | navy | `nonclean_direct` | `L27C15369+L26C8587:zero` | `other->strict_canonical` | 3.000 | 3.000 | 0.688 | -0.042 | `other_blocker->strict_target` | True | False | True |
