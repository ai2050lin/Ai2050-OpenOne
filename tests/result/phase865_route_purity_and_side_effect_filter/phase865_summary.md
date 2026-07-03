# Phase 865 Route Purity and Side-Effect Filter

- Source: Phase 864 route separation.
- Boundary: offline filter, not new model intervention and not closure.

## Summary

- full_set_purity_class_counts: `{'clean_mixed_answer_blocker_route': 6, 'harmful_or_unstable': 2, 'object_side_effect_risk': 2, 'clean_answer_lift_route': 1, 'inactive_or_weak': 1}`
- dominant_channel_purity_class_counts: `{'clean_mixed_answer_blocker_route': 6, 'harmful_or_unstable': 2, 'object_side_effect_risk': 2, 'clean_answer_lift_route': 1, 'inactive_or_weak': 1}`

## Clean Full-Set Routes

| model | domain | mode | route | gain/loss | answer delta | blocker reduction | blocker delta | object delta | purity |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| deepseek7b | animal | `flip` | `mixed_answer_lift_and_blocker_weakening` | 10/0 | 2.9750 | 1.2667 | -0.1429 | -0.1417 | `clean_mixed_answer_blocker_route` |
| deepseek7b | animal | `half` | `mixed_answer_lift_and_blocker_weakening` | 5/0 | 0.6292 | 0.6000 | -0.0371 | -0.0250 | `clean_mixed_answer_blocker_route` |
| deepseek7b | animal | `zero` | `mixed_answer_lift_and_blocker_weakening` | 6/0 | 1.3208 | 0.8667 | -0.0650 | -0.0437 | `clean_mixed_answer_blocker_route` |
| deepseek7b | color | `half` | `answer_lift_dominant` | 2/0 | 0.8833 | 0.6667 | 0.0027 | 0.2188 | `clean_answer_lift_route` |
| qwen3 | material | `flip` | `mixed_answer_lift_and_blocker_weakening` | 5/0 | 0.7583 | 2.4000 | -0.1058 | -0.3000 | `clean_mixed_answer_blocker_route` |
| qwen3 | material | `half` | `mixed_answer_lift_and_blocker_weakening` | 2/0 | 0.1833 | 0.9333 | -0.0431 | -0.0750 | `clean_mixed_answer_blocker_route` |
| qwen3 | material | `zero` | `mixed_answer_lift_and_blocker_weakening` | 2/0 | 0.4083 | 1.7333 | -0.0494 | -0.1417 | `clean_mixed_answer_blocker_route` |

## Domain Full-Set Classes

`{'deepseek7b:animal': {'clean_mixed_answer_blocker_route': 3, 'harmful_or_unstable': 1}, 'deepseek7b:color': {'object_side_effect_risk': 2, 'clean_answer_lift_route': 1, 'harmful_or_unstable': 1}, 'qwen3:material': {'clean_mixed_answer_blocker_route': 3, 'inactive_or_weak': 1}}`
