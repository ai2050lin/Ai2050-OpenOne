# Phase 857 Prompt-Gated Causal Gear Rollout Validation (main)

- Source: Phase 854 full-combo and necessary-blocker-reducer rows.
- Boundary: prompt-gated causal replay, not a new gear search.

## Cross-Model Summary

| model | sources | rows | full first | full rollout | full clear | full echo | full vs original clear gain/loss | echo reduced/induced | without necessary clear loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 6 | 54 | 9 | 9 | 9 | 0 | 7/0 | 0/0 | 7 |
| glm4 | 6 | 39 | 12 | 12 | 12 | 0 | 0/0 | 0/0 | 0 |
| deepseek7b | 6 | 42 | 11 | 11 | 9 | 0 | 9/0 | 0/0 | 3 |

## Prompt / Condition Summary

| model | prompt::condition | n | first class | rollout class | clear rollout | object echo | class blockers | class-object margin | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `natural_category::full_combo` | 6 | 6 | 6 | 6 | 0 | 0.0000 | 3.9375 | 1.0000 |
| qwen3 | `natural_category::original` | 6 | 2 | 2 | 2 | 0 | 0.6667 | 3.1458 | 1.0000 |
| qwen3 | `natural_category::without_necessary` | 6 | 2 | 2 | 2 | 0 | 0.6667 | 4.3750 | 1.0000 |
| qwen3 | `natural_question::full_combo` | 6 | 3 | 3 | 3 | 0 | 0.6667 | 2.6875 | 1.0000 |
| qwen3 | `natural_question::original` | 6 | 0 | 0 | 0 | 0 | 1.0000 | 2.3750 | 0.0000 |
| qwen3 | `natural_question::without_necessary` | 6 | 1 | 1 | 1 | 0 | 1.0000 | 3.7917 | 1.0000 |
| qwen3 | `object_only::full_combo` | 6 | 0 | 0 | 0 | 0 | 10.8333 | -0.8438 | 0.0000 |
| qwen3 | `object_only::original` | 6 | 0 | 0 | 0 | 0 | 6.6667 | -0.3750 | 0.0000 |
| qwen3 | `object_only::without_necessary` | 6 | 0 | 0 | 0 | 0 | 6.5000 | 0.4479 | 0.0000 |
| glm4 | `natural_category::full_combo` | 6 | 6 | 6 | 6 | 0 | 0.0000 | 1.7917 | 1.0000 |
| glm4 | `natural_category::original` | 6 | 6 | 6 | 6 | 0 | 0.0000 | 1.1042 | 1.0000 |
| glm4 | `natural_category::without_necessary` | 1 | 1 | 1 | 1 | 0 | 0.0000 | 2.0625 | 1.0000 |
| glm4 | `natural_question::full_combo` | 6 | 6 | 6 | 6 | 0 | 0.0000 | 2.7604 | 1.0000 |
| glm4 | `natural_question::original` | 6 | 6 | 6 | 6 | 0 | 0.0000 | 2.0417 | 1.0000 |
| glm4 | `natural_question::without_necessary` | 1 | 1 | 1 | 1 | 0 | 0.0000 | 1.8750 | 1.0000 |
| glm4 | `object_only::full_combo` | 6 | 0 | 0 | 0 | 0 | 19.8333 | -0.9219 | 0.0000 |
| glm4 | `object_only::original` | 6 | 0 | 0 | 0 | 0 | 25.5000 | -1.2969 | 0.0000 |
| glm4 | `object_only::without_necessary` | 1 | 0 | 0 | 0 | 0 | 3.0000 | 0.3750 | 0.0000 |
| deepseek7b | `natural_category::full_combo` | 6 | 4 | 4 | 4 | 0 | 0.3333 | 1.2396 | 1.0000 |
| deepseek7b | `natural_category::original` | 6 | 0 | 0 | 0 | 0 | 1.1667 | 1.3646 | 0.0000 |
| deepseek7b | `natural_category::without_necessary` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 1.5625 | 0.0000 |
| deepseek7b | `natural_question::full_combo` | 6 | 6 | 6 | 5 | 0 | 0.0000 | 0.8229 | 1.0000 |
| deepseek7b | `natural_question::original` | 6 | 1 | 1 | 0 | 0 | 0.8333 | 0.8854 | 1.0000 |
| deepseek7b | `natural_question::without_necessary` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 |
| deepseek7b | `object_only::full_combo` | 6 | 1 | 1 | 0 | 0 | 3.8333 | -0.1250 | 1.0000 |
| deepseek7b | `object_only::original` | 6 | 1 | 1 | 0 | 0 | 3.3333 | 0.0521 | 1.0000 |
| deepseek7b | `object_only::without_necessary` | 2 | 0 | 0 | 0 | 0 | 5.0000 | -0.1250 | 0.0000 |

## Pairwise Effects

| model | comparison | pairs | answer gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | class-object gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `full_vs_original` | 18 | 7/0 | 7/0 | 7/0 | 0/0 | -1.0556 | 0.2118 |
| qwen3 | `without_necessary_vs_full` | 18 | 1/7 | 1/7 | 1/7 | 0/0 | 1.1111 | 0.9444 |
| glm4 | `full_vs_original` | 18 | 0/0 | 0/0 | 0/0 | 0/0 | 1.8889 | 0.5938 |
| glm4 | `without_necessary_vs_full` | 3 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | -0.3125 |
| deepseek7b | `full_vs_original` | 18 | 9/0 | 9/0 | 9/0 | 0/0 | 0.3889 | -0.1215 |
| deepseek7b | `without_necessary_vs_full` | 6 | 0/3 | 0/3 | 0/3 | 0/0 | -0.5000 | 0.0417 |
