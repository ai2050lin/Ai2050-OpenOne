# Phase 868 Conditional Route Transfer Failure Taxonomy (holdout)

- Source: Phase 867 holdout effects.
- Boundary: offline taxonomy only, no new model run and no closure claim.

## Summary

- Status counts: `{'source_clean_failed': 6, 'stable_nonclean': 4, 'emergent_clean': 2}`
- Failure reason counts: `{'original_blocker_not_negative': 8, 'no_clear_gain': 6, 'answer_not_lifted': 3, 'blocker_not_reduced': 3, 'clear_loss': 2, 'format_or_other_side_effect': 2}`

## Rows

| model | domain | mode | source purity | transfer status | clear gain/loss | ans delta | blocker red. | orig blocker delta | object delta | reasons |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | material | `flip` | `clean_mixed_answer_blocker_route` | `source_clean_failed` | 1/0 | 0.6615 | 11.8333 | 0.0000 | 0.0156 | `['original_blocker_not_negative']` |
| qwen3 | material | `half` | `clean_mixed_answer_blocker_route` | `source_clean_failed` | 0/0 | 0.1823 | 4.1667 | 0.0023 | 0.0000 | `['no_clear_gain', 'original_blocker_not_negative']` |
| qwen3 | material | `scale_up` | `inactive_or_weak` | `stable_nonclean` | 0/0 | -0.4010 | -7.5833 | 0.0023 | -0.0052 | `['no_clear_gain', 'answer_not_lifted', 'blocker_not_reduced', 'original_blocker_not_negative']` |
| qwen3 | material | `zero` | `clean_mixed_answer_blocker_route` | `source_clean_failed` | 0/0 | 0.3646 | 6.5000 | -0.0006 | 0.0000 | `['no_clear_gain']` |
| deepseek7b | animal | `flip` | `clean_mixed_answer_blocker_route` | `source_clean_failed` | 1/0 | 2.3854 | 0.4167 | 0.0604 | 0.0312 | `['original_blocker_not_negative']` |
| deepseek7b | animal | `half` | `clean_mixed_answer_blocker_route` | `source_clean_failed` | 0/0 | 0.6146 | 0.2500 | 0.0640 | 0.0391 | `['no_clear_gain', 'original_blocker_not_negative']` |
| deepseek7b | animal | `scale_up` | `harmful_or_unstable` | `stable_nonclean` | 0/5 | -1.0312 | -0.4167 | 0.0261 | -0.0130 | `['no_clear_gain', 'clear_loss', 'answer_not_lifted', 'blocker_not_reduced', 'original_blocker_not_negative', 'format_or_other_side_effect']` |
| deepseek7b | animal | `zero` | `clean_mixed_answer_blocker_route` | `source_clean_failed` | 1/0 | 1.1875 | 0.2500 | 0.0394 | 0.0339 | `['original_blocker_not_negative']` |
| deepseek7b | color | `flip` | `object_side_effect_risk` | `stable_nonclean` | 5/0 | 2.6589 | 1.5000 | 0.0067 | 0.0938 | `['original_blocker_not_negative']` |
| deepseek7b | color | `half` | `clean_answer_lift_route` | `emergent_clean` | 2/0 | 0.5052 | 0.5000 | -0.0025 | 0.0260 | `[]` |
| deepseek7b | color | `scale_up` | `harmful_or_unstable` | `stable_nonclean` | 0/2 | -0.6849 | -1.9167 | -0.0000 | -0.0208 | `['no_clear_gain', 'clear_loss', 'answer_not_lifted', 'blocker_not_reduced', 'format_or_other_side_effect']` |
| deepseek7b | color | `zero` | `object_side_effect_risk` | `emergent_clean` | 2/0 | 1.0911 | 0.8333 | -0.0079 | 0.0339 | `[]` |
