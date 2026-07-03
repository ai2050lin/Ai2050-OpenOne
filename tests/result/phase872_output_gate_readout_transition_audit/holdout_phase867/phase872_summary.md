# Phase 872 Output Gate / Readout Transition Audit (holdout)

- Boundary: offline audit from existing row-level logits and rollouts; no new model run.
- Goal: test whether output/readout dominance explains failures left by FieldAdmissible + GearEffect.

## Rule Results

| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `output_gate_raw_rule` | `intervened_rollout_clear_answer_class` | 144 | 49 | 0 | 0 | 95 | 1.000 | 1.000 | 1.000 |
| `output_gate_raw_rule` | `intervened_rollout_strict_canonical` | 144 | 49 | 0 | 0 | 95 | 1.000 | 1.000 | 1.000 |
| `output_gate_raw_rule` | `target_output_clean_transition` | 144 | 12 | 37 | 0 | 95 | 0.245 | 1.000 | 0.743 |
| `output_gate_raw_rule` | `target_clean_transition` | 144 | 4 | 45 | 0 | 95 | 0.082 | 1.000 | 0.688 |
| `output_gate_field_rule` | `intervened_rollout_clear_answer_class` | 144 | 47 | 0 | 2 | 95 | 1.000 | 0.959 | 0.986 |
| `output_gate_field_rule` | `intervened_rollout_strict_canonical` | 144 | 47 | 0 | 2 | 95 | 1.000 | 0.959 | 0.986 |
| `output_gate_field_rule` | `target_output_clean_transition` | 144 | 10 | 37 | 2 | 95 | 0.213 | 0.833 | 0.729 |
| `output_gate_field_rule` | `target_clean_transition` | 144 | 4 | 43 | 0 | 97 | 0.085 | 1.000 | 0.701 |
| `field_strict_plus_effect_rule` | `intervened_rollout_clear_answer_class` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `field_strict_plus_effect_rule` | `intervened_rollout_strict_canonical` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `field_strict_plus_effect_rule` | `target_output_clean_transition` | 144 | 4 | 0 | 8 | 132 | 1.000 | 0.333 | 0.944 |
| `field_strict_plus_effect_rule` | `target_clean_transition` | 144 | 4 | 0 | 0 | 140 | 1.000 | 1.000 | 1.000 |
| `readout_rank_closure_rule` | `intervened_rollout_clear_answer_class` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `readout_rank_closure_rule` | `intervened_rollout_strict_canonical` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `readout_rank_closure_rule` | `target_output_clean_transition` | 144 | 4 | 0 | 8 | 132 | 1.000 | 0.333 | 0.944 |
| `readout_rank_closure_rule` | `target_clean_transition` | 144 | 4 | 0 | 0 | 140 | 1.000 | 1.000 | 1.000 |
| `output_gate_top1_rule` | `intervened_rollout_clear_answer_class` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `output_gate_top1_rule` | `intervened_rollout_strict_canonical` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `output_gate_top1_rule` | `target_output_clean_transition` | 144 | 4 | 0 | 8 | 132 | 1.000 | 0.333 | 0.944 |
| `output_gate_top1_rule` | `target_clean_transition` | 144 | 4 | 0 | 0 | 140 | 1.000 | 1.000 | 1.000 |
| `output_gate_margin_rule` | `intervened_rollout_clear_answer_class` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `output_gate_margin_rule` | `intervened_rollout_strict_canonical` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `output_gate_margin_rule` | `target_output_clean_transition` | 144 | 4 | 0 | 8 | 132 | 1.000 | 0.333 | 0.944 |
| `output_gate_margin_rule` | `target_clean_transition` | 144 | 4 | 0 | 0 | 140 | 1.000 | 1.000 | 1.000 |
| `output_gate_strict_margin_rule` | `intervened_rollout_clear_answer_class` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `output_gate_strict_margin_rule` | `intervened_rollout_strict_canonical` | 144 | 4 | 0 | 45 | 95 | 1.000 | 0.082 | 0.688 |
| `output_gate_strict_margin_rule` | `target_output_clean_transition` | 144 | 4 | 0 | 8 | 132 | 1.000 | 0.333 | 0.944 |
| `output_gate_strict_margin_rule` | `target_clean_transition` | 144 | 4 | 0 | 0 | 140 | 1.000 | 1.000 | 1.000 |

## Summary

- Transfer status counts: `{'source_clean_failed': 71, 'stable_nonclean': 69, 'stable_clean': 1, 'emergent_clean': 3}`
- Field-strict transfer status counts: `{'stable_clean': 1, 'emergent_clean': 3}`
- Field-strict top1 role counts: `{'strict_target': 4}`
- Field-strict best non-target role counts: `{'format_space': 1, 'format_punct': 3}`

## Gate-Candidate Rows

| model | domain | object | prompt | mode | target | field+effect | gate | top1 | margin | non-target | failure | rollout |
|---|---|---|---|---|---|---|---|---|---:|---|---|---|
| qwen3 | material | rubber | `holdout_kind_phrase` | `flip` | True | True | True | `strict_target: material` | 0.375 | `format_space:
` | `none` | `object_echo -> strict_canonical` |
| deepseek7b | color | purple | `holdout_category_short` | `flip` | True | True | True | `strict_target: color` | 6.250 | `format_punct: ?
` | `none` | `format_or_empty -> strict_canonical` |
| deepseek7b | color | purple | `holdout_category_short` | `half` | True | True | True | `strict_target: color` | 0.312 | `format_punct: ?
` | `none` | `format_or_empty -> strict_canonical` |
| deepseek7b | color | purple | `holdout_category_short` | `zero` | True | True | True | `strict_target: color` | 1.812 | `format_punct: ?
` | `none` | `format_or_empty -> strict_canonical` |
