# Phase 857 Prompt-Gated Causal Gear Rollout Validation (confirm)

- Source: Phase 854 full-combo and necessary-blocker-reducer rows.
- Boundary: prompt-gated causal replay, not a new gear search.

## Cross-Model Summary

| model | sources | rows | full first | full rollout | full clear | full echo | full vs original clear gain/loss | echo reduced/induced | without necessary clear loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 10 | 90 | 17 | 17 | 17 | 0 | 15/0 | 0/0 | 15 |
| glm4 | 10 | 63 | 20 | 20 | 20 | 0 | 0/0 | 0/0 | 0 |
| deepseek7b | 10 | 66 | 20 | 20 | 9 | 0 | 9/0 | 0/0 | 3 |

## Prompt / Condition Summary

| model | prompt::condition | n | first class | rollout class | clear rollout | object echo | class blockers | class-object margin | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `natural_category::full_combo` | 10 | 10 | 10 | 10 | 0 | 0.0000 | 2.9500 | 1.0000 |
| qwen3 | `natural_category::original` | 10 | 2 | 2 | 2 | 0 | 0.8000 | 2.3375 | 1.0000 |
| qwen3 | `natural_category::without_necessary` | 10 | 2 | 2 | 2 | 0 | 0.8000 | 3.4250 | 1.0000 |
| qwen3 | `natural_question::full_combo` | 10 | 7 | 7 | 7 | 0 | 0.4000 | 2.3375 | 1.0000 |
| qwen3 | `natural_question::original` | 10 | 0 | 0 | 0 | 0 | 1.0000 | 1.9250 | 0.0000 |
| qwen3 | `natural_question::without_necessary` | 10 | 1 | 1 | 1 | 0 | 1.0000 | 3.2500 | 1.0000 |
| qwen3 | `object_only::full_combo` | 10 | 0 | 0 | 0 | 0 | 10.1000 | -1.0188 | 0.0000 |
| qwen3 | `object_only::original` | 10 | 0 | 0 | 0 | 0 | 6.0000 | -0.3250 | 0.0000 |
| qwen3 | `object_only::without_necessary` | 10 | 0 | 0 | 0 | 0 | 5.7000 | 0.6062 | 0.0000 |
| glm4 | `natural_category::full_combo` | 10 | 10 | 10 | 10 | 0 | 0.0000 | 1.6125 | 1.0000 |
| glm4 | `natural_category::original` | 10 | 10 | 10 | 10 | 0 | 0.0000 | 1.0375 | 1.0000 |
| glm4 | `natural_category::without_necessary` | 1 | 1 | 1 | 1 | 0 | 0.0000 | 2.0625 | 1.0000 |
| glm4 | `natural_question::full_combo` | 10 | 10 | 10 | 10 | 0 | 0.0000 | 2.6000 | 1.0000 |
| glm4 | `natural_question::original` | 10 | 10 | 10 | 10 | 0 | 0.0000 | 2.0250 | 1.0000 |
| glm4 | `natural_question::without_necessary` | 1 | 1 | 1 | 1 | 0 | 0.0000 | 1.8750 | 1.0000 |
| glm4 | `object_only::full_combo` | 10 | 0 | 0 | 0 | 0 | 22.1000 | -1.1375 | 0.0000 |
| glm4 | `object_only::original` | 10 | 0 | 0 | 0 | 0 | 27.3000 | -1.4906 | 0.0000 |
| glm4 | `object_only::without_necessary` | 1 | 0 | 0 | 0 | 0 | 3.0000 | 0.3750 | 0.0000 |
| deepseek7b | `natural_category::full_combo` | 10 | 5 | 5 | 4 | 0 | 0.5000 | 0.8125 | 1.0000 |
| deepseek7b | `natural_category::original` | 10 | 0 | 0 | 0 | 0 | 1.5000 | 0.9688 | 0.0000 |
| deepseek7b | `natural_category::without_necessary` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 1.5625 | 0.0000 |
| deepseek7b | `natural_question::full_combo` | 10 | 10 | 10 | 5 | 0 | 0.0000 | 0.4938 | 1.0000 |
| deepseek7b | `natural_question::original` | 10 | 5 | 5 | 0 | 0 | 0.5000 | 0.5312 | 1.0000 |
| deepseek7b | `natural_question::without_necessary` | 2 | 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 |
| deepseek7b | `object_only::full_combo` | 10 | 5 | 5 | 0 | 0 | 2.3000 | -0.0750 | 1.0000 |
| deepseek7b | `object_only::original` | 10 | 5 | 5 | 0 | 0 | 2.0000 | 0.0312 | 1.0000 |
| deepseek7b | `object_only::without_necessary` | 2 | 0 | 0 | 0 | 0 | 5.0000 | -0.1250 | 0.0000 |

## Pairwise Effects

| model | comparison | pairs | answer gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | class-object gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `full_vs_original` | 30 | 15/0 | 15/0 | 15/0 | 0/0 | -0.9000 | 0.1104 |
| qwen3 | `without_necessary_vs_full` | 30 | 1/15 | 1/15 | 1/15 | 0/0 | 1.0000 | 1.0042 |
| glm4 | `full_vs_original` | 30 | 0/0 | 0/0 | 0/0 | 0/0 | 1.7333 | 0.5010 |
| glm4 | `without_necessary_vs_full` | 3 | 0/0 | 0/0 | 0/0 | 0/0 | 0.0000 | -0.3125 |
| deepseek7b | `full_vs_original` | 30 | 10/0 | 10/0 | 9/0 | 0/0 | 0.4000 | -0.1000 |
| deepseek7b | `without_necessary_vs_full` | 6 | 0/3 | 0/3 | 0/3 | 0/0 | -0.5000 | 0.0417 |
