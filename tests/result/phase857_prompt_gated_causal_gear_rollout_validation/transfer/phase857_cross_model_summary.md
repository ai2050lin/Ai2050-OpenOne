# Phase 857 Prompt-Gated Causal Gear Rollout Validation (transfer)

- Source: Phase 854 full-combo and necessary-blocker-reducer rows.
- Boundary: prompt-gated causal replay, not a new gear search.

## Cross-Model Summary

| model | sources | rows | full first | full rollout | full clear | full echo | full vs original clear gain/loss | echo reduced/induced | without necessary clear loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 2 | 72 | 8 | 9 | 9 | 2 | 1/0 | 0/0 | 1 |
| glm4 | 2 | 60 | 14 | 14 | 14 | 0 | 0/0 | 0/0 | 0 |
| deepseek7b | 2 | 72 | 5 | 6 | 6 | 4 | 0/0 | 0/0 | 0 |

## Prompt / Condition Summary

| model | prompt::condition | n | first class | rollout class | clear rollout | object echo | class blockers | class-object margin | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `natural_category::full_combo` | 12 | 4 | 5 | 5 | 0 | 22.0000 | 2.4010 | 0.6667 |
| qwen3 | `natural_category::original` | 12 | 4 | 4 | 4 | 0 | 21.8333 | 2.3125 | 1.0000 |
| qwen3 | `natural_category::without_necessary` | 12 | 5 | 5 | 5 | 0 | 20.9167 | 2.5052 | 1.0000 |
| qwen3 | `natural_question::full_combo` | 12 | 4 | 4 | 4 | 2 | 37.5833 | 1.0677 | 1.0000 |
| qwen3 | `natural_question::original` | 12 | 4 | 4 | 4 | 2 | 37.3333 | 1.2500 | 1.0000 |
| qwen3 | `natural_question::without_necessary` | 12 | 5 | 5 | 5 | 2 | 36.8333 | 1.4062 | 1.0000 |
| glm4 | `natural_category::full_combo` | 12 | 8 | 8 | 8 | 0 | 2.1667 | 1.1198 | 1.0000 |
| glm4 | `natural_category::original` | 12 | 8 | 8 | 8 | 0 | 2.1667 | 1.1615 | 1.0000 |
| glm4 | `natural_category::without_necessary` | 6 | 4 | 4 | 4 | 0 | 2.1667 | 1.0781 | 1.0000 |
| glm4 | `natural_question::full_combo` | 12 | 6 | 6 | 6 | 0 | 3.2500 | 1.8073 | 1.0000 |
| glm4 | `natural_question::original` | 12 | 6 | 6 | 6 | 0 | 3.1667 | 1.8438 | 1.0000 |
| glm4 | `natural_question::without_necessary` | 6 | 3 | 3 | 3 | 0 | 3.1667 | 1.7917 | 1.0000 |
| deepseek7b | `natural_category::full_combo` | 12 | 2 | 2 | 2 | 0 | 2.0833 | 3.3464 | 1.0000 |
| deepseek7b | `natural_category::original` | 12 | 2 | 2 | 2 | 0 | 2.3333 | 3.3802 | 1.0000 |
| deepseek7b | `natural_category::without_necessary` | 12 | 2 | 2 | 2 | 0 | 2.3333 | 3.4062 | 1.0000 |
| deepseek7b | `natural_question::full_combo` | 12 | 3 | 4 | 4 | 4 | 11.0000 | 0.6276 | 0.5714 |
| deepseek7b | `natural_question::original` | 12 | 2 | 4 | 4 | 4 | 11.6667 | 0.6979 | 0.6667 |
| deepseek7b | `natural_question::without_necessary` | 12 | 2 | 4 | 4 | 4 | 10.8333 | 0.6927 | 0.6667 |

## Pairwise Effects

| model | comparison | pairs | answer gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | class-object gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `full_vs_original` | 24 | 2/2 | 1/0 | 1/0 | 0/0 | -0.2083 | -0.0469 |
| qwen3 | `without_necessary_vs_full` | 24 | 3/1 | 2/1 | 2/1 | 0/0 | 0.9167 | 0.2214 |
| glm4 | `full_vs_original` | 24 | 0/0 | 0/0 | 0/0 | 0/0 | -0.0417 | -0.0391 |
| glm4 | `without_necessary_vs_full` | 12 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0833 | -0.0052 |
| deepseek7b | `full_vs_original` | 24 | 1/0 | 0/0 | 0/0 | 0/0 | 0.4583 | -0.0521 |
| deepseek7b | `without_necessary_vs_full` | 24 | 0/1 | 0/0 | 0/0 | 0/0 | -0.0417 | 0.0625 |
