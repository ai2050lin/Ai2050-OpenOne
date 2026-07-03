# Phase 877 Effective Transition Source-Gate Audit

- Boundary: offline audit over Phase875/876 transition rows; no new model run.
- Goal: identify source gear, prompt gate, field gate, and blocker diagnostics for Phase876 effective transitions.

## Summary

- Target round: `validation_phase876`
- Effective transitions in target round: `12`
- Transition classes: `{'clean_causal_transition': 4, 'nonclean_output_transition': 8}`
- Routes: `{'clean_causal_transition': 4, 'format_recovery': 3, 'semantic_pressure_transition': 3, 'protocol_pressure_transition': 2}`
- Model/domains: `{'deepseek7b:animal': 10, 'deepseek7b:color': 2}`
- Source gates: `{'clean_blocker_weakening_gate': 4, 'format_recovery_gate': 3, 'semantic_answer_lift_gate': 3, 'protocol_pressure_gate': 2}`
- Prompt gates: `{'format_prompt_gate': 3, 'direct_prompt_gate': 3, 'echo_prompt_gate': 6}`
- Field gates: `{'semantic_field_gate': 10, 'protocol_field_gate': 2}`
- Same-source multi-route candidates: `['L27C16651+L24C3875:flip', 'L27C16651+L24C3875:half', 'L27C16651+L24C3875:zero']`

## Source Candidates

| candidate | n | classes | routes | source gates | prompts | labels | mean ans | mean blocker red. | mean orig blocker | mean margin |
|---|---:|---|---|---|---|---|---:|---:|---:|---:|
| `L27C15369+L26C8587:flip` | 1 | `{'clean_causal_transition': 1}` | `{'clean_causal_transition': 1}` | `{'clean_blocker_weakening_gate': 1}` | `{'direct_prompt_gate': 1}` | `{'other->strict_canonical': 1}` | 1.438 | 3.000 | -0.042 | 0.812 |
| `L27C15369+L26C8587:zero` | 1 | `{'clean_causal_transition': 1}` | `{'clean_causal_transition': 1}` | `{'clean_blocker_weakening_gate': 1}` | `{'direct_prompt_gate': 1}` | `{'other->strict_canonical': 1}` | 0.688 | 3.000 | -0.042 | 0.062 |
| `L27C16651+L24C3875:flip` | 4 | `{'clean_causal_transition': 1, 'nonclean_output_transition': 3}` | `{'clean_causal_transition': 1, 'format_recovery': 1, 'semantic_pressure_transition': 1, 'protocol_pressure_transition': 1}` | `{'clean_blocker_weakening_gate': 1, 'format_recovery_gate': 1, 'semantic_answer_lift_gate': 1, 'protocol_pressure_gate': 1}` | `{'direct_prompt_gate': 1, 'echo_prompt_gate': 2, 'format_prompt_gate': 1}` | `{'other->strict_canonical': 2, 'format_or_empty->answer_alias': 1, 'other->answer_alias': 1}` | 2.547 | 1.500 | 0.031 | 1.969 |
| `L27C16651+L24C3875:half` | 2 | `{'nonclean_output_transition': 2}` | `{'format_recovery': 1, 'semantic_pressure_transition': 1}` | `{'format_recovery_gate': 1, 'semantic_answer_lift_gate': 1}` | `{'echo_prompt_gate': 2}` | `{'format_or_empty->answer_alias': 1, 'other->answer_alias': 1}` | 1.031 | 1.500 | 0.047 | 0.469 |
| `L27C16651+L24C3875:scale_up` | 1 | `{'clean_causal_transition': 1}` | `{'clean_causal_transition': 1}` | `{'clean_blocker_weakening_gate': 1}` | `{'format_prompt_gate': 1}` | `{'other->strict_canonical': 1}` | 1.312 | 1.000 | -0.062 | 1.125 |
| `L27C16651+L24C3875:zero` | 3 | `{'nonclean_output_transition': 3}` | `{'format_recovery': 1, 'semantic_pressure_transition': 1, 'protocol_pressure_transition': 1}` | `{'format_recovery_gate': 1, 'semantic_answer_lift_gate': 1, 'protocol_pressure_gate': 1}` | `{'echo_prompt_gate': 2, 'format_prompt_gate': 1}` | `{'format_or_empty->answer_alias': 1, 'other->answer_alias': 1, 'other->strict_canonical': 1}` | 1.479 | 1.333 | 0.062 | 1.021 |

## Object Prompt Entrances

| object::prompt | n | routes | candidates | modes | labels | field gates | mean ans | mean blocker red. | mean orig blocker | mean margin |
|---|---:|---|---|---|---|---|---:|---:|---:|---:|
| `bat::nonclean_direct` | 1 | `{'clean_causal_transition': 1}` | `{'L27C16651+L24C3875:flip': 1}` | `['flip']` | `{'other->strict_canonical': 1}` | `{'semantic_field_gate': 1}` | 1.438 | 2.000 | -0.062 | 0.562 |
| `navy::nonclean_direct` | 2 | `{'clean_causal_transition': 2}` | `{'L27C15369+L26C8587:flip': 1, 'L27C15369+L26C8587:zero': 1}` | `['flip', 'zero']` | `{'other->strict_canonical': 2}` | `{'semantic_field_gate': 2}` | 1.062 | 3.000 | -0.042 | 0.438 |
| `seal::format_pressure` | 1 | `{'clean_causal_transition': 1}` | `{'L27C16651+L24C3875:scale_up': 1}` | `['scale_up']` | `{'other->strict_canonical': 1}` | `{'semantic_field_gate': 1}` | 1.312 | 1.000 | -0.062 | 1.125 |
| `sheep::echo_pressure` | 3 | `{'format_recovery': 3}` | `{'L27C16651+L24C3875:flip': 1, 'L27C16651+L24C3875:half': 1, 'L27C16651+L24C3875:zero': 1}` | `['flip', 'half', 'zero']` | `{'format_or_empty->answer_alias': 3}` | `{'semantic_field_gate': 3}` | 1.896 | 2.000 | 0.010 | 1.604 |
| `wolf::echo_pressure` | 3 | `{'semantic_pressure_transition': 3}` | `{'L27C16651+L24C3875:flip': 1, 'L27C16651+L24C3875:half': 1, 'L27C16651+L24C3875:zero': 1}` | `['flip', 'half', 'zero']` | `{'other->answer_alias': 3}` | `{'semantic_field_gate': 3}` | 2.771 | 1.000 | 0.104 | 1.917 |
| `wolf::format_pressure` | 2 | `{'protocol_pressure_transition': 2}` | `{'L27C16651+L24C3875:flip': 1, 'L27C16651+L24C3875:zero': 1}` | `['flip', 'zero']` | `{'other->strict_canonical': 2}` | `{'protocol_field_gate': 2}` | 0.625 | 1.000 | 0.062 | 0.375 |

## Target Rows

| class | route | model | domain | object | prompt | candidate | mode | label | top1 | best blocker | gates | ans | blocker red. | orig blocker | margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| `clean_causal_transition` | `clean_causal_transition` | deepseek7b | animal | seal | `format_pressure` | `L27C16651+L24C3875:scale_up` | `scale_up` | `other->strict_canonical` | `other->strict_target` | `other->other` | `['clean_blocker_weakening_gate', 'format_prompt_gate', ['semantic_field_gate']]` | 1.312 | 1.000 | -0.062 | 1.125 |
| `clean_causal_transition` | `clean_causal_transition` | deepseek7b | animal | bat | `nonclean_direct` | `L27C16651+L24C3875:flip` | `flip` | `other->strict_canonical` | `other->strict_target` | `other->other` | `['clean_blocker_weakening_gate', 'direct_prompt_gate', ['semantic_field_gate']]` | 1.438 | 2.000 | -0.062 | 0.562 |
| `nonclean_output_transition` | `format_recovery` | deepseek7b | animal | sheep | `echo_pressure` | `L27C16651+L24C3875:flip` | `flip` | `format_or_empty->answer_alias` | `format_punct->strict_target` | `format_punct->other` | `['format_recovery_gate', 'echo_prompt_gate', ['semantic_field_gate']]` | 3.188 | 2.000 | 0.000 | 2.875 |
| `nonclean_output_transition` | `format_recovery` | deepseek7b | animal | sheep | `echo_pressure` | `L27C16651+L24C3875:half` | `half` | `format_or_empty->answer_alias` | `format_punct->strict_target` | `format_punct->other` | `['format_recovery_gate', 'echo_prompt_gate', ['semantic_field_gate']]` | 0.875 | 2.000 | 0.031 | 0.562 |
| `nonclean_output_transition` | `format_recovery` | deepseek7b | animal | sheep | `echo_pressure` | `L27C16651+L24C3875:zero` | `zero` | `format_or_empty->answer_alias` | `format_punct->strict_target` | `format_punct->format_punct` | `['format_recovery_gate', 'echo_prompt_gate', ['semantic_field_gate']]` | 1.625 | 2.000 | 0.000 | 1.375 |
| `nonclean_output_transition` | `semantic_pressure_transition` | deepseek7b | animal | wolf | `echo_pressure` | `L27C16651+L24C3875:flip` | `flip` | `other->answer_alias` | `other->strict_target` | `other->other` | `['semantic_answer_lift_gate', 'echo_prompt_gate', ['semantic_field_gate']]` | 4.750 | 1.000 | 0.125 | 3.875 |
| `nonclean_output_transition` | `semantic_pressure_transition` | deepseek7b | animal | wolf | `echo_pressure` | `L27C16651+L24C3875:half` | `half` | `other->answer_alias` | `other->strict_target` | `other->other` | `['semantic_answer_lift_gate', 'echo_prompt_gate', ['semantic_field_gate']]` | 1.188 | 1.000 | 0.062 | 0.375 |
| `nonclean_output_transition` | `semantic_pressure_transition` | deepseek7b | animal | wolf | `echo_pressure` | `L27C16651+L24C3875:zero` | `zero` | `other->answer_alias` | `other->strict_target` | `other->other` | `['semantic_answer_lift_gate', 'echo_prompt_gate', ['semantic_field_gate']]` | 2.375 | 1.000 | 0.125 | 1.500 |
| `nonclean_output_transition` | `protocol_pressure_transition` | deepseek7b | animal | wolf | `format_pressure` | `L27C16651+L24C3875:flip` | `flip` | `other->strict_canonical` | `other->strict_target` | `other->other` | `['protocol_pressure_gate', 'format_prompt_gate', ['protocol_field_gate']]` | 0.812 | 1.000 | 0.062 | 0.562 |
| `nonclean_output_transition` | `protocol_pressure_transition` | deepseek7b | animal | wolf | `format_pressure` | `L27C16651+L24C3875:zero` | `zero` | `other->strict_canonical` | `other->strict_target` | `other->other` | `['protocol_pressure_gate', 'format_prompt_gate', ['protocol_field_gate']]` | 0.438 | 1.000 | 0.062 | 0.188 |
| `clean_causal_transition` | `clean_causal_transition` | deepseek7b | color | navy | `nonclean_direct` | `L27C15369+L26C8587:flip` | `flip` | `other->strict_canonical` | `other->strict_target` | `other->other` | `['clean_blocker_weakening_gate', 'direct_prompt_gate', ['semantic_field_gate']]` | 1.438 | 3.000 | -0.042 | 0.812 |
| `clean_causal_transition` | `clean_causal_transition` | deepseek7b | color | navy | `nonclean_direct` | `L27C15369+L26C8587:zero` | `zero` | `other->strict_canonical` | `other->strict_target` | `other->other` | `['clean_blocker_weakening_gate', 'direct_prompt_gate', ['semantic_field_gate']]` | 0.688 | 3.000 | -0.042 | 0.062 |
