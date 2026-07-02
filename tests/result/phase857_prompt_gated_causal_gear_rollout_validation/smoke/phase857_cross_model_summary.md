# Phase 857 Prompt-Gated Causal Gear Rollout Validation (smoke)

- Source: Phase 854 full-combo and necessary-blocker-reducer rows.
- Boundary: prompt-gated causal replay, not a new gear search.

## Cross-Model Summary

| model | sources | rows | full first | full rollout | full clear | full echo | full vs original clear gain/loss | echo reduced/induced | without necessary clear loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 2 | 18 | 2 | 2 | 2 | 0 | 0/0 | 0/0 | 0 |
| glm4 | 2 | 15 | 4 | 4 | 4 | 0 | 0/0 | 0/0 | 0 |
| deepseek7b | 2 | 18 | 3 | 3 | 3 | 0 | 3/0 | 0/0 | 3 |

## Prompt / Condition Summary

| model | prompt::condition | n | first class | rollout class | clear rollout | object echo | class blockers | class-object margin | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `natural_category::full_combo` | 2 | 2 | 2 | 2 | 0 | 0.0000 | 7.7500 | 1.0000 |
| qwen3 | `natural_category::original` | 2 | 2 | 2 | 2 | 0 | 0.0000 | 7.1875 | 1.0000 |
| qwen3 | `natural_category::without_necessary` | 2 | 2 | 2 | 2 | 0 | 0.0000 | 9.3125 | 1.0000 |
| qwen3 | `natural_question::full_combo` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 4.0000 | 0.0000 |
| qwen3 | `natural_question::original` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 4.6250 | 0.0000 |
| qwen3 | `natural_question::without_necessary` | 2 | 1 | 1 | 1 | 0 | 1.0000 | 6.6250 | 1.0000 |
| qwen3 | `object_only::full_combo` | 2 | 0 | 0 | 0 | 0 | 17.5000 | -1.0312 | 0.0000 |
| qwen3 | `object_only::original` | 2 | 0 | 0 | 0 | 0 | 10.0000 | -0.6250 | 0.0000 |
| qwen3 | `object_only::without_necessary` | 2 | 0 | 0 | 0 | 0 | 10.5000 | -0.4062 | 0.0000 |
| glm4 | `natural_category::full_combo` | 2 | 2 | 2 | 2 | 0 | 0.0000 | 1.7500 | 1.0000 |
| glm4 | `natural_category::original` | 2 | 2 | 2 | 2 | 0 | 0.0000 | 1.4375 | 1.0000 |
| glm4 | `natural_category::without_necessary` | 1 | 1 | 1 | 1 | 0 | 0.0000 | 2.0625 | 1.0000 |
| glm4 | `natural_question::full_combo` | 2 | 2 | 2 | 2 | 0 | 0.0000 | 2.1250 | 1.0000 |
| glm4 | `natural_question::original` | 2 | 2 | 2 | 2 | 0 | 0.0000 | 2.1250 | 1.0000 |
| glm4 | `natural_question::without_necessary` | 1 | 1 | 1 | 1 | 0 | 0.0000 | 1.8750 | 1.0000 |
| glm4 | `object_only::full_combo` | 2 | 0 | 0 | 0 | 0 | 16.5000 | -0.7188 | 0.0000 |
| glm4 | `object_only::original` | 2 | 0 | 0 | 0 | 0 | 16.5000 | -0.3281 | 0.0000 |
| glm4 | `object_only::without_necessary` | 1 | 0 | 0 | 0 | 0 | 3.0000 | 0.3750 | 0.0000 |
| deepseek7b | `natural_category::full_combo` | 2 | 1 | 1 | 1 | 0 | 0.5000 | 1.5000 | 1.0000 |
| deepseek7b | `natural_category::original` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 1.5625 | 0.0000 |
| deepseek7b | `natural_category::without_necessary` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 1.5625 | 0.0000 |
| deepseek7b | `natural_question::full_combo` | 2 | 2 | 2 | 2 | 0 | 0.0000 | 1.0312 | 1.0000 |
| deepseek7b | `natural_question::original` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 1.0625 | 0.0000 |
| deepseek7b | `natural_question::without_necessary` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 |
| deepseek7b | `object_only::full_combo` | 2 | 0 | 0 | 0 | 0 | 5.0000 | -0.2188 | 0.0000 |
| deepseek7b | `object_only::original` | 2 | 0 | 0 | 0 | 0 | 4.0000 | 0.0625 | 0.0000 |
| deepseek7b | `object_only::without_necessary` | 2 | 0 | 0 | 0 | 0 | 5.0000 | -0.1250 | 0.0000 |

## Pairwise Effects

| model | comparison | pairs | answer gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | class-object gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `full_vs_original` | 6 | 0/0 | 0/0 | 0/0 | 0/0 | -2.5000 | -0.1562 |
| qwen3 | `without_necessary_vs_full` | 6 | 1/0 | 1/0 | 1/0 | 0/0 | 2.3333 | 1.6042 |
| glm4 | `full_vs_original` | 6 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | -0.0260 |
| glm4 | `without_necessary_vs_full` | 3 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | -0.3125 |
| deepseek7b | `full_vs_original` | 6 | 3/0 | 3/0 | 3/0 | 0/0 | 0.1667 | -0.1250 |
| deepseek7b | `without_necessary_vs_full` | 6 | 0/3 | 0/3 | 0/3 | 0/0 | -0.5000 | 0.0417 |
