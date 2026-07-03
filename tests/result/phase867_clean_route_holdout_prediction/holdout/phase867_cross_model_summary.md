# Phase 867 Clean Route Holdout Prediction (holdout)

- Source: Phase 865 full-set route purity rows.
- Fixed rule: Phase 866 CleanMixedRoute, object_delta_threshold=0.25 unless configured.
- Boundary: holdout rule validation, not language closure.

## Cross-Model Summary

| model | status | candidates | domains | source-clean -> holdout-clean stats |
|---|---|---:|---|---|
| qwen3 | complete | 4 | `['material']` | `{'n': 4, 'tp': 0, 'fp': 3, 'fn': 0, 'tn': 1, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.25, 'source_clean_count': 3, 'holdout_clean_count': 0}` |
| glm4 | no_phase865_candidates | 0 | `[]` | `{}` |
| deepseek7b | complete | 8 | `['animal', 'color']` | `{'n': 8, 'tp': 0, 'fp': 3, 'fn': 2, 'tn': 3, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.375, 'source_clean_count': 3, 'holdout_clean_count': 2}` |

## Holdout Effects

| model | domain | mode | source purity | source clean | holdout clean | clear gain/loss | ans delta | blocker red. | orig blocker delta | object delta | side effects |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | material | `flip` | `clean_mixed_answer_blocker_route` | True | False | 1/0 | 0.6615 | 11.8333 | 0.0000 | 0.0156 | echo+0, fmt+0 |
| qwen3 | material | `half` | `clean_mixed_answer_blocker_route` | True | False | 0/0 | 0.1823 | 4.1667 | 0.0023 | 0.0000 | echo+0, fmt+0 |
| qwen3 | material | `scale_up` | `inactive_or_weak` | False | False | 0/0 | -0.4010 | -7.5833 | 0.0023 | -0.0052 | echo+0, fmt+0 |
| qwen3 | material | `zero` | `clean_mixed_answer_blocker_route` | True | False | 0/0 | 0.3646 | 6.5000 | -0.0006 | 0.0000 | echo+0, fmt+0 |
| deepseek7b | animal | `flip` | `clean_mixed_answer_blocker_route` | True | False | 1/0 | 2.3854 | 0.4167 | 0.0604 | 0.0312 | echo+0, fmt+0 |
| deepseek7b | animal | `half` | `clean_mixed_answer_blocker_route` | True | False | 0/0 | 0.6146 | 0.2500 | 0.0640 | 0.0391 | echo+0, fmt+0 |
| deepseek7b | animal | `scale_up` | `harmful_or_unstable` | False | False | 0/5 | -1.0312 | -0.4167 | 0.0261 | -0.0130 | echo+0, fmt+5 |
| deepseek7b | animal | `zero` | `clean_mixed_answer_blocker_route` | True | False | 1/0 | 1.1875 | 0.2500 | 0.0394 | 0.0339 | echo+0, fmt+0 |
| deepseek7b | color | `flip` | `object_side_effect_risk` | False | False | 5/0 | 2.6589 | 1.5000 | 0.0067 | 0.0938 | echo+0, fmt+0 |
| deepseek7b | color | `half` | `clean_answer_lift_route` | False | True | 2/0 | 0.5052 | 0.5000 | -0.0025 | 0.0260 | echo+0, fmt+0 |
| deepseek7b | color | `scale_up` | `harmful_or_unstable` | False | False | 0/2 | -0.6849 | -1.9167 | -0.0000 | -0.0208 | echo+0, fmt+2 |
| deepseek7b | color | `zero` | `object_side_effect_risk` | False | True | 2/0 | 1.0911 | 0.8333 | -0.0079 | 0.0339 | echo+0, fmt+0 |
