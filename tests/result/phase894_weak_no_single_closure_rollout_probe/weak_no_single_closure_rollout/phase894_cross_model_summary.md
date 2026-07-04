# Phase 894 weak no-single closure replication and rollout boundary probe

## Overall

- models: qwen3, glm4, deepseek7b
- selected_conditions: 381
- output_first_rows: 933
- output_rollout_rows: 123
- output_head_rows: 176
- closure_from_open: 115
- closure_without_single_axis_closure: 22
- phase893_exact_no_single_replicated: 7
- expanded_pair_no_single_conditions: 4
- rollout_class_hit: 93
- rollout_without_single_axis_hit: 22
- head_combo_closure_lost: 40
- head_combo_damage_gt_0_25: 38

## Subset groups

| model | subset | closure | no-single | exact replicated | expanded no-single | mean lift | mean comp | no-single objects |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b | L26C8587+L27C15369 | 36 | 11 | 7 | 4 | 3.568 | 0.767 | black,brown,color,cyan,gray,green,grey,navy,violet |
| deepseek7b | L26C8587+L27C15369+L27C16651 | 36 | 11 | 7 | 4 | 3.547 | 0.790 | black,brown,color,cyan,gray,green,grey,navy,violet |
| deepseek7b | L27C15369 | 24 | 0 | 0 | 0 | 2.417 | 0.000 |  |
| qwen3 | L31C2257 | 11 | 0 | 0 | 0 | 0.739 | 0.000 |  |
| deepseek7b | L26C8587 | 8 | 0 | 0 | 0 | 0.750 | 0.000 |  |
| glm4 | L31C6437 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |  |
| deepseek7b | L27C16651 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |  |

## Rollout groups

| model | subset | rows | class hit | no-single hit | object echo |
| --- | --- | ---: | ---: | ---: | ---: |
| deepseek7b | L26C8587+L27C15369 | 28 | 28 | 11 | 5 |
| deepseek7b | L26C8587+L27C15369+L27C16651 | 28 | 28 | 11 | 5 |
| deepseek7b | L27C15369 | 28 | 16 | 0 | 4 |
| qwen3 | L31C2257 | 11 | 11 | 0 | 3 |
| deepseek7b | L26C8587 | 28 | 10 | 0 | 3 |

## Head groups

| model | head set | rows | closure lost | damage > 0.25 | mean damage | max damage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3 | L31H19+L31H26+L31H30+L31H12+L31H17 | 11 | 10 | 7 | 0.420 | 0.750 |
| qwen3 | L31H19 | 11 | 6 | 0 | 0.159 | 0.250 |
| qwen3 | L31H26 | 11 | 6 | 1 | 0.136 | 0.375 |
| deepseek7b | L26H3+L26H7+L26H11+L26H14 | 11 | 4 | 8 | 0.511 | 1.625 |
| qwen3 | L31H17 | 11 | 4 | 2 | 0.170 | 0.375 |
| qwen3 | L31H30 | 11 | 3 | 0 | 0.023 | 0.125 |
| deepseek7b | L26H7+L26H11 | 11 | 2 | 8 | 0.347 | 0.688 |
| qwen3 | L31H12 | 11 | 2 | 1 | 0.034 | 0.375 |
| deepseek7b | L26H3+L26H7 | 11 | 1 | 4 | 0.233 | 0.625 |
| deepseek7b | L26H3+L26H11 | 11 | 1 | 3 | 0.222 | 0.500 |
| deepseek7b | L26H7 | 11 | 1 | 2 | 0.170 | 0.375 |
| deepseek7b | L26H11 | 11 | 0 | 0 | 0.176 | 0.250 |
| deepseek7b | L26H14 | 11 | 0 | 1 | 0.119 | 0.688 |
| deepseek7b | L26H3 | 11 | 0 | 1 | 0.062 | 0.312 |
| qwen3 | none | 11 | 0 | 0 | 0.000 | 0.000 |
| deepseek7b | none | 11 | 0 | 0 | 0.000 | 0.000 |
