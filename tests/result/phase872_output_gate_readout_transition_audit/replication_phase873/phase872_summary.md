# Phase 872 Output Gate / Readout Transition Audit (replication)

- Boundary: offline audit from existing row-level logits and rollouts; no new model run.
- Goal: test whether output/readout dominance explains failures left by FieldAdmissible + GearEffect.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `output_gate_raw_rule` | `intervened_rollout_clear_answer_class` | 216 | 68 | 0 | 0 | 148 | 1.000 | 1.000 | 1.000 |
| `output_gate_raw_rule` | `intervened_rollout_strict_canonical` | 216 | 44 | 24 | 0 | 148 | 0.647 | 1.000 | 0.889 |
| `output_gate_raw_rule` | `target_output_clean_transition` | 216 | 2 | 66 | 0 | 148 | 0.029 | 1.000 | 0.694 |
| `output_gate_raw_rule` | `target_clean_transition` | 216 | 0 | 68 | 0 | 148 | 0.000 | 0.000 | 0.685 |
| `output_gate_field_rule` | `intervened_rollout_clear_answer_class` | 216 | 67 | 0 | 1 | 148 | 1.000 | 0.985 | 0.995 |
| `output_gate_field_rule` | `intervened_rollout_strict_canonical` | 216 | 43 | 24 | 1 | 148 | 0.642 | 0.977 | 0.884 |
| `output_gate_field_rule` | `target_output_clean_transition` | 216 | 1 | 66 | 1 | 148 | 0.015 | 0.500 | 0.690 |
| `output_gate_field_rule` | `target_clean_transition` | 216 | 0 | 67 | 0 | 149 | 0.000 | 0.000 | 0.690 |
| `field_strict_plus_effect_rule` | `intervened_rollout_clear_answer_class` | 216 | 0 | 0 | 68 | 148 | 0.000 | 0.000 | 0.685 |
| `field_strict_plus_effect_rule` | `intervened_rollout_strict_canonical` | 216 | 0 | 0 | 44 | 172 | 0.000 | 0.000 | 0.796 |
| `field_strict_plus_effect_rule` | `target_output_clean_transition` | 216 | 0 | 0 | 2 | 214 | 0.000 | 0.000 | 0.991 |
| `field_strict_plus_effect_rule` | `target_clean_transition` | 216 | 0 | 0 | 0 | 216 | 0.000 | 0.000 | 1.000 |
| `readout_rank_closure_rule` | `intervened_rollout_clear_answer_class` | 216 | 0 | 0 | 68 | 148 | 0.000 | 0.000 | 0.685 |
| `readout_rank_closure_rule` | `intervened_rollout_strict_canonical` | 216 | 0 | 0 | 44 | 172 | 0.000 | 0.000 | 0.796 |
| `readout_rank_closure_rule` | `target_output_clean_transition` | 216 | 0 | 0 | 2 | 214 | 0.000 | 0.000 | 0.991 |
| `readout_rank_closure_rule` | `target_clean_transition` | 216 | 0 | 0 | 0 | 216 | 0.000 | 0.000 | 1.000 |
| `output_gate_top1_rule` | `intervened_rollout_clear_answer_class` | 216 | 0 | 0 | 68 | 148 | 0.000 | 0.000 | 0.685 |
| `output_gate_top1_rule` | `intervened_rollout_strict_canonical` | 216 | 0 | 0 | 44 | 172 | 0.000 | 0.000 | 0.796 |
| `output_gate_top1_rule` | `target_output_clean_transition` | 216 | 0 | 0 | 2 | 214 | 0.000 | 0.000 | 0.991 |
| `output_gate_top1_rule` | `target_clean_transition` | 216 | 0 | 0 | 0 | 216 | 0.000 | 0.000 | 1.000 |
| `output_gate_margin_rule` | `intervened_rollout_clear_answer_class` | 216 | 0 | 0 | 68 | 148 | 0.000 | 0.000 | 0.685 |
| `output_gate_margin_rule` | `intervened_rollout_strict_canonical` | 216 | 0 | 0 | 44 | 172 | 0.000 | 0.000 | 0.796 |
| `output_gate_margin_rule` | `target_output_clean_transition` | 216 | 0 | 0 | 2 | 214 | 0.000 | 0.000 | 0.991 |
| `output_gate_margin_rule` | `target_clean_transition` | 216 | 0 | 0 | 0 | 216 | 0.000 | 0.000 | 1.000 |
| `output_gate_strict_margin_rule` | `intervened_rollout_clear_answer_class` | 216 | 0 | 0 | 68 | 148 | 0.000 | 0.000 | 0.685 |
| `output_gate_strict_margin_rule` | `intervened_rollout_strict_canonical` | 216 | 0 | 0 | 44 | 172 | 0.000 | 0.000 | 0.796 |
| `output_gate_strict_margin_rule` | `target_output_clean_transition` | 216 | 0 | 0 | 2 | 214 | 0.000 | 0.000 | 0.991 |
| `output_gate_strict_margin_rule` | `target_clean_transition` | 216 | 0 | 0 | 0 | 216 | 0.000 | 0.000 | 1.000 |

## Summary

- Transfer status counts: `{'source_clean_failed': 108, 'stable_nonclean': 108}`
- Field-strict transfer status counts: `{}`
- Field-strict top1 role counts: `{}`
- Field-strict best non-target role counts: `{}`

## Gate-Candidate Rows

| model | domain | object | prompt | mode | target | field+effect | gate | top1 | margin | non-target | failure | rollout |
|---|---|---|---|---|---|---|---|---|---:|---|---|---|
