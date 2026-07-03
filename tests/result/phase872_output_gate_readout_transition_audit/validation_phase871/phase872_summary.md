# Phase 872 Output Gate / Readout Transition Audit (validation)

- Boundary: offline audit from existing row-level logits and rollouts; no new model run.
- Goal: test whether output/readout dominance explains failures left by FieldAdmissible + GearEffect.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `output_gate_raw_rule` | `intervened_rollout_clear_answer_class` | 144 | 53 | 0 | 0 | 91 | 1.000 | 1.000 | 1.000 |
| `output_gate_raw_rule` | `intervened_rollout_strict_canonical` | 144 | 45 | 8 | 0 | 91 | 0.849 | 1.000 | 0.944 |
| `output_gate_raw_rule` | `target_output_clean_transition` | 144 | 7 | 46 | 0 | 91 | 0.132 | 1.000 | 0.681 |
| `output_gate_raw_rule` | `target_clean_transition` | 144 | 1 | 52 | 0 | 91 | 0.019 | 1.000 | 0.639 |
| `output_gate_field_rule` | `intervened_rollout_clear_answer_class` | 144 | 49 | 0 | 4 | 91 | 1.000 | 0.925 | 0.972 |
| `output_gate_field_rule` | `intervened_rollout_strict_canonical` | 144 | 41 | 8 | 4 | 91 | 0.837 | 0.911 | 0.917 |
| `output_gate_field_rule` | `target_output_clean_transition` | 144 | 3 | 46 | 4 | 91 | 0.061 | 0.429 | 0.653 |
| `output_gate_field_rule` | `target_clean_transition` | 144 | 1 | 48 | 0 | 95 | 0.020 | 1.000 | 0.667 |
| `field_strict_plus_effect_rule` | `intervened_rollout_clear_answer_class` | 144 | 1 | 1 | 52 | 90 | 0.500 | 0.019 | 0.632 |
| `field_strict_plus_effect_rule` | `intervened_rollout_strict_canonical` | 144 | 1 | 1 | 44 | 98 | 0.500 | 0.022 | 0.688 |
| `field_strict_plus_effect_rule` | `target_output_clean_transition` | 144 | 1 | 1 | 6 | 136 | 0.500 | 0.143 | 0.951 |
| `field_strict_plus_effect_rule` | `target_clean_transition` | 144 | 1 | 1 | 0 | 142 | 0.500 | 1.000 | 0.993 |
| `readout_rank_closure_rule` | `intervened_rollout_clear_answer_class` | 144 | 1 | 1 | 52 | 90 | 0.500 | 0.019 | 0.632 |
| `readout_rank_closure_rule` | `intervened_rollout_strict_canonical` | 144 | 1 | 1 | 44 | 98 | 0.500 | 0.022 | 0.688 |
| `readout_rank_closure_rule` | `target_output_clean_transition` | 144 | 1 | 1 | 6 | 136 | 0.500 | 0.143 | 0.951 |
| `readout_rank_closure_rule` | `target_clean_transition` | 144 | 1 | 1 | 0 | 142 | 0.500 | 1.000 | 0.993 |
| `output_gate_top1_rule` | `intervened_rollout_clear_answer_class` | 144 | 1 | 0 | 52 | 91 | 1.000 | 0.019 | 0.639 |
| `output_gate_top1_rule` | `intervened_rollout_strict_canonical` | 144 | 1 | 0 | 44 | 99 | 1.000 | 0.022 | 0.694 |
| `output_gate_top1_rule` | `target_output_clean_transition` | 144 | 1 | 0 | 6 | 137 | 1.000 | 0.143 | 0.958 |
| `output_gate_top1_rule` | `target_clean_transition` | 144 | 1 | 0 | 0 | 143 | 1.000 | 1.000 | 1.000 |
| `output_gate_margin_rule` | `intervened_rollout_clear_answer_class` | 144 | 1 | 0 | 52 | 91 | 1.000 | 0.019 | 0.639 |
| `output_gate_margin_rule` | `intervened_rollout_strict_canonical` | 144 | 1 | 0 | 44 | 99 | 1.000 | 0.022 | 0.694 |
| `output_gate_margin_rule` | `target_output_clean_transition` | 144 | 1 | 0 | 6 | 137 | 1.000 | 0.143 | 0.958 |
| `output_gate_margin_rule` | `target_clean_transition` | 144 | 1 | 0 | 0 | 143 | 1.000 | 1.000 | 1.000 |
| `output_gate_strict_margin_rule` | `intervened_rollout_clear_answer_class` | 144 | 1 | 0 | 52 | 91 | 1.000 | 0.019 | 0.639 |
| `output_gate_strict_margin_rule` | `intervened_rollout_strict_canonical` | 144 | 1 | 0 | 44 | 99 | 1.000 | 0.022 | 0.694 |
| `output_gate_strict_margin_rule` | `target_output_clean_transition` | 144 | 1 | 0 | 6 | 137 | 1.000 | 0.143 | 0.958 |
| `output_gate_strict_margin_rule` | `target_clean_transition` | 144 | 1 | 0 | 0 | 143 | 1.000 | 1.000 | 1.000 |

## Summary

- Transfer status counts: `{'source_clean_failed': 72, 'stable_nonclean': 71, 'emergent_clean': 1}`
- Field-strict transfer status counts: `{'source_clean_failed': 1, 'emergent_clean': 1}`
- Field-strict top1 role counts: `{'format_space': 1, 'strict_target': 1}`
- Field-strict best non-target role counts: `{'format_space': 1, 'format_punct': 1}`

## Gate-Candidate Rows

| model | domain | object | prompt | mode | target | field+effect | gate | top1 | margin | non-target | failure | rollout |
|---|---|---|---|---|---|---|---|---|---:|---|---|---|
| qwen3 | material | concrete | `validation_direct` | `flip` | False | True | False | `format_space: 

` | 0.000 | `format_space: 

` | `top1_non_target:format_space` | `other -> other` |
| deepseek7b | color | cyan | `validation_direct` | `flip` | True | True | True | `strict_target: color` | 1.188 | `format_punct: {` | `none` | `other -> strict_canonical` |
