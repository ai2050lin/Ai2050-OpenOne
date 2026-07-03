# Phase 864 Answer-Lift vs Blocker-Weakening Route Separation

- Source: Phase 862 main rows + Phase 863 channel roles.
- Boundary: offline route separation, not new model intervention and not closure.

## Summary

- route_class_counts: `{'mixed_answer_lift_and_blocker_weakening': 15, 'weak_or_unresolved': 7, 'harmful_or_blocker_amplifying': 6, 'mixed_answer_blocker_with_object_side_effect': 2, 'answer_lift_dominant': 4, 'answer_lift_with_object_echo_side_effect': 2}`

## Full-Set Routes

| model | domain | mode | route | gain/loss | answer delta | blocker reduction | blocker delta | object delta | object echo +/- | format/other +/- |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | material | `flip` | `mixed_answer_lift_and_blocker_weakening` | 5/0 | 0.7583 | 2.4000 | -0.1058 | -0.3000 | 0/0 | 3/0 |
| qwen3 | material | `half` | `mixed_answer_lift_and_blocker_weakening` | 2/0 | 0.1833 | 0.9333 | -0.0431 | -0.0750 | 0/0 | 1/0 |
| qwen3 | material | `scale_up` | `weak_or_unresolved` | 0/0 | -0.4292 | -2.4667 | 0.0153 | 0.1083 | 0/0 | 0/0 |
| qwen3 | material | `zero` | `mixed_answer_lift_and_blocker_weakening` | 2/0 | 0.4083 | 1.7333 | -0.0494 | -0.1417 | 0/0 | 1/0 |
| deepseek7b | animal | `flip` | `mixed_answer_lift_and_blocker_weakening` | 10/0 | 2.9750 | 1.2667 | -0.1429 | -0.1417 | 0/0 | 10/0 |
| deepseek7b | animal | `half` | `mixed_answer_lift_and_blocker_weakening` | 5/0 | 0.6292 | 0.6000 | -0.0371 | -0.0250 | 0/0 | 5/0 |
| deepseek7b | animal | `scale_up` | `harmful_or_blocker_amplifying` | 0/3 | -0.9854 | -2.5333 | 0.0475 | 0.0292 | 0/1 | 0/2 |
| deepseek7b | animal | `zero` | `mixed_answer_lift_and_blocker_weakening` | 6/0 | 1.3208 | 0.8667 | -0.0650 | -0.0437 | 0/0 | 6/0 |
| deepseek7b | color | `flip` | `mixed_answer_blocker_with_object_side_effect` | 5/0 | 4.9250 | 1.1333 | -0.0004 | 1.2458 | 0/0 | 6/0 |
| deepseek7b | color | `half` | `answer_lift_dominant` | 2/0 | 0.8833 | 0.6667 | 0.0027 | 0.2188 | 0/0 | 2/0 |
| deepseek7b | color | `scale_up` | `harmful_or_blocker_amplifying` | 0/2 | -1.2542 | -2.0667 | -0.0144 | -0.2917 | 0/0 | 0/2 |
| deepseek7b | color | `zero` | `answer_lift_with_object_echo_side_effect` | 5/0 | 1.9792 | 1.1333 | 0.0033 | 0.4833 | 0/0 | 6/0 |

## Dominant Channel Routes

| model | domain | gear | mode | channel role | route | gain/loss | answer delta | blocker reduction | blocker delta | object delta |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3 | material | `L31C2257` | `flip` | `['dominant_answer_and_blocker_channel']` | `mixed_answer_lift_and_blocker_weakening` | 4/0 | 0.6417 | 2.2000 | -0.0358 | -0.1083 |
| qwen3 | material | `L31C2257` | `half` | `['dominant_answer_and_blocker_channel']` | `mixed_answer_lift_and_blocker_weakening` | 2/0 | 0.1375 | 0.7333 | -0.0222 | -0.0333 |
| qwen3 | material | `L31C2257` | `scale_up` | `['dominant_answer_and_blocker_channel']` | `weak_or_unresolved` | 0/0 | -0.3583 | -2.0667 | 0.0111 | 0.0583 |
| qwen3 | material | `L31C2257` | `zero` | `['dominant_answer_and_blocker_channel']` | `mixed_answer_lift_and_blocker_weakening` | 2/0 | 0.3125 | 1.4000 | -0.0311 | -0.0833 |
| deepseek7b | animal | `L27C16651` | `flip` | `['dominant_answer_and_blocker_channel']` | `mixed_answer_lift_and_blocker_weakening` | 9/0 | 2.9417 | 1.2000 | -0.0633 | 0.1229 |
| deepseek7b | animal | `L27C16651` | `half` | `['dominant_answer_and_blocker_channel']` | `mixed_answer_lift_and_blocker_weakening` | 5/0 | 0.6333 | 0.6000 | -0.0138 | 0.0375 |
| deepseek7b | animal | `L27C16651` | `scale_up` | `['dominant_answer_and_blocker_channel']` | `harmful_or_blocker_amplifying` | 0/2 | -1.0000 | -2.2667 | 0.0158 | -0.0688 |
| deepseek7b | animal | `L27C16651` | `zero` | `['dominant_answer_and_blocker_channel']` | `mixed_answer_lift_and_blocker_weakening` | 6/0 | 1.3167 | 0.8667 | -0.0262 | 0.0750 |
| deepseek7b | color | `L27C15369` | `flip` | `['dominant_answer_lift_channel']` | `mixed_answer_blocker_with_object_side_effect` | 5/0 | 2.5667 | 1.0000 | -0.0088 | 0.6167 |
| deepseek7b | color | `L27C15369` | `half` | `['dominant_answer_lift_channel']` | `answer_lift_dominant` | 2/0 | 0.6500 | 0.6000 | 0.0071 | 0.1562 |
| deepseek7b | color | `L27C15369` | `scale_up` | `['dominant_answer_lift_channel']` | `harmful_or_blocker_amplifying` | 0/2 | -1.3042 | -2.4667 | 0.0125 | -0.3250 |
| deepseek7b | color | `L27C15369` | `zero` | `['dominant_answer_lift_channel']` | `answer_lift_with_object_echo_side_effect` | 3/0 | 1.2958 | 0.8000 | 0.0033 | 0.3167 |
