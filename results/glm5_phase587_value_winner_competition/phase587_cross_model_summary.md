# Phase 587 Value Winner Competition Summary

Confirm setting: 32 value cases per model. Target case means base is wrong and repair prompt is correct.

| model | target cases | best patch | target switch | correct gain | top-wrong gain | margin gain | correct-up & competitor-up | correct-up but margin<0 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 4 | prompt_last|L27|add_repair | 2/4 (50.0%) | 0.588 | 0.400 | 0.188 | 3/4 | 2/4 |
| glm4 | 3 | query_relation|L38|replace_repair | 0/3 (0.0%) | 0.014 | 0.014 | 0.000 | 2/3 | 2/3 |
| deepseek7b | 12 | prompt_last|L21|wrong_relation | 0/12 (0.0%) | 1.223 | 1.153 | 0.069 | 7/12 | 7/12 |

## DS7B Main Diagnostic Patch

`prompt_last|L21|add_repair`:

- target switch: 0/12 (0.0%)
- mean correct gain: 4.782
- mean old-top-wrong gain: 4.718
- mean margin gain: 0.063
- correct-up and competitor-up: 12/12
- correct-up but final margin negative: 12/12

## Objective Facts

- DS7B confirms the Phase586 suspicion: the repair patch raises correct value and top wrong value together.
- DS7B prompt_last L21 add_repair has correct gain +4.782 but old-top-wrong gain +4.718, so margin gain is only +0.063 and target switch remains 0/12.
- This means value-gate failure is not support-only. The missing component is winner-margin control, likely competitor suppression or relation-bound selection.
- Qwen3 has partial switch (2/4) because margin gain is larger relative to its target cases.
- GLM4 target cases remain too few and do not provide a stable value-gate conclusion.
