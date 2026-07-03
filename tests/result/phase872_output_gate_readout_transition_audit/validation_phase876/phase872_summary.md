# Phase 872 Output Gate / Readout Transition Audit (validation)

- Boundary: offline audit from existing row-level logits and rollouts; no new model run.
- Goal: test whether output/readout dominance explains failures left by FieldAdmissible + GearEffect.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `output_gate_raw_rule` | `intervened_rollout_clear_answer_class` | 288 | 63 | 0 | 12 | 213 | 1.000 | 0.840 | 0.958 |
| `output_gate_raw_rule` | `intervened_rollout_strict_canonical` | 288 | 34 | 29 | 4 | 221 | 0.540 | 0.895 | 0.885 |
| `output_gate_raw_rule` | `target_output_clean_transition` | 288 | 12 | 51 | 0 | 225 | 0.190 | 1.000 | 0.823 |
| `output_gate_raw_rule` | `target_clean_transition` | 288 | 4 | 59 | 0 | 225 | 0.063 | 1.000 | 0.795 |
| `output_gate_field_rule` | `intervened_rollout_clear_answer_class` | 288 | 63 | 0 | 12 | 213 | 1.000 | 0.840 | 0.958 |
| `output_gate_field_rule` | `intervened_rollout_strict_canonical` | 288 | 34 | 29 | 4 | 221 | 0.540 | 0.895 | 0.885 |
| `output_gate_field_rule` | `target_output_clean_transition` | 288 | 12 | 51 | 0 | 225 | 0.190 | 1.000 | 0.823 |
| `output_gate_field_rule` | `target_clean_transition` | 288 | 4 | 59 | 0 | 225 | 0.063 | 1.000 | 0.795 |
| `field_strict_plus_effect_rule` | `intervened_rollout_clear_answer_class` | 288 | 0 | 0 | 75 | 213 | 0.000 | 0.000 | 0.740 |
| `field_strict_plus_effect_rule` | `intervened_rollout_strict_canonical` | 288 | 0 | 0 | 38 | 250 | 0.000 | 0.000 | 0.868 |
| `field_strict_plus_effect_rule` | `target_output_clean_transition` | 288 | 0 | 0 | 12 | 276 | 0.000 | 0.000 | 0.958 |
| `field_strict_plus_effect_rule` | `target_clean_transition` | 288 | 0 | 0 | 4 | 284 | 0.000 | 0.000 | 0.986 |
| `readout_rank_closure_rule` | `intervened_rollout_clear_answer_class` | 288 | 0 | 0 | 75 | 213 | 0.000 | 0.000 | 0.740 |
| `readout_rank_closure_rule` | `intervened_rollout_strict_canonical` | 288 | 0 | 0 | 38 | 250 | 0.000 | 0.000 | 0.868 |
| `readout_rank_closure_rule` | `target_output_clean_transition` | 288 | 0 | 0 | 12 | 276 | 0.000 | 0.000 | 0.958 |
| `readout_rank_closure_rule` | `target_clean_transition` | 288 | 0 | 0 | 4 | 284 | 0.000 | 0.000 | 0.986 |
| `output_gate_top1_rule` | `intervened_rollout_clear_answer_class` | 288 | 0 | 0 | 75 | 213 | 0.000 | 0.000 | 0.740 |
| `output_gate_top1_rule` | `intervened_rollout_strict_canonical` | 288 | 0 | 0 | 38 | 250 | 0.000 | 0.000 | 0.868 |
| `output_gate_top1_rule` | `target_output_clean_transition` | 288 | 0 | 0 | 12 | 276 | 0.000 | 0.000 | 0.958 |
| `output_gate_top1_rule` | `target_clean_transition` | 288 | 0 | 0 | 4 | 284 | 0.000 | 0.000 | 0.986 |
| `output_gate_margin_rule` | `intervened_rollout_clear_answer_class` | 288 | 0 | 0 | 75 | 213 | 0.000 | 0.000 | 0.740 |
| `output_gate_margin_rule` | `intervened_rollout_strict_canonical` | 288 | 0 | 0 | 38 | 250 | 0.000 | 0.000 | 0.868 |
| `output_gate_margin_rule` | `target_output_clean_transition` | 288 | 0 | 0 | 12 | 276 | 0.000 | 0.000 | 0.958 |
| `output_gate_margin_rule` | `target_clean_transition` | 288 | 0 | 0 | 4 | 284 | 0.000 | 0.000 | 0.986 |
| `output_gate_strict_margin_rule` | `intervened_rollout_clear_answer_class` | 288 | 0 | 0 | 75 | 213 | 0.000 | 0.000 | 0.740 |
| `output_gate_strict_margin_rule` | `intervened_rollout_strict_canonical` | 288 | 0 | 0 | 38 | 250 | 0.000 | 0.000 | 0.868 |
| `output_gate_strict_margin_rule` | `target_output_clean_transition` | 288 | 0 | 0 | 12 | 276 | 0.000 | 0.000 | 0.958 |
| `output_gate_strict_margin_rule` | `target_clean_transition` | 288 | 0 | 0 | 4 | 284 | 0.000 | 0.000 | 0.986 |

## Summary

- Transfer status counts: `{'source_clean_failed': 143, 'stable_nonclean': 141, 'emergent_clean': 3, 'stable_clean': 1}`
- Field-strict transfer status counts: `{}`
- Field-strict top1 role counts: `{}`
- Field-strict best non-target role counts: `{}`

## Gate-Candidate Rows

| model | domain | object | prompt | mode | target | field+effect | gate | top1 | margin | non-target | failure | rollout |
|---|---|---|---|---|---|---|---|---|---:|---|---|---|
| deepseek7b | animal | seal | `format_pressure` | `scale_up` | True | False | False | `strict_target:animal` | 1.125 | `other:marine` | `none` | `other -> strict_canonical` |
| deepseek7b | animal | bat | `nonclean_direct` | `flip` | True | False | False | `strict_target: animal` | 0.562 | `other: plural` | `none` | `other -> strict_canonical` |
| deepseek7b | color | navy | `nonclean_direct` | `flip` | True | False | False | `strict_target: color` | 0.812 | `other: one` | `none` | `other -> strict_canonical` |
| deepseek7b | color | navy | `nonclean_direct` | `zero` | True | False | False | `strict_target: color` | 0.062 | `other: one` | `none` | `other -> strict_canonical` |
