# Phase 893 attention-head attribution and pairwise complementarity holdout probe

## Overall

- models: qwen3, glm4, deepseek7b
- selected_case_prompts: 275
- output_subset_rows: 2805
- output_head_rows: 960
- closure_from_open: 399
- positive_complementarity_rows: 68
- holdout_positive_complementarity_rows: 28
- closure_without_single_axis_closure: 14
- mean_multi_complementarity_over_best: 0.138
- head_zero_closure_lost: 26
- head_zero_damage_gt_0_25: 31
- mean_head_target_lift_damage_vs_none: 0.035

## Subset groups

| model | subset | relation | closure | holdout closure | comp rows | holdout comp | no-single closure | mean lift | mean comp | modes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b | L26C8587+L27C15369+L27C16651 | model_U | 94 | 47 | 34 | 14 | 7 | 2.798 | 0.628 | {"flip": 41, "half": 17, "zero": 36} |
| deepseek7b | L26C8587+L27C15369 | ds7b_color_complementary_pair | 46 | 21 | 34 | 14 | 7 | 3.067 | 1.306 | {"flip": 20, "half": 9, "zero": 17} |
| deepseek7b | L27C15369+L27C16651 | multi_axis | 84 | 42 | 0 | 0 | 0 | 2.262 | -0.010 | {"flip": 38, "half": 17, "zero": 29} |
| deepseek7b | L26C8587+L27C16651 | multi_axis | 60 | 31 | 0 | 0 | 0 | 2.241 | -0.006 | {"flip": 29, "half": 9, "zero": 22} |
| deepseek7b | L27C16651 | ds7b_animal_single_axis | 52 | 29 | 0 | 0 | 0 | 2.439 | 0.000 | {"flip": 24, "half": 9, "zero": 19} |
| deepseek7b | L27C15369 | single_axis | 37 | 17 | 0 | 0 | 0 | 1.833 | 0.000 | {"flip": 17, "half": 9, "zero": 11} |
| qwen3 | L31C2257 | model_U | 14 | 5 | 0 | 0 | 0 | 0.723 | 0.000 | {"flip": 8, "half": 2, "zero": 4} |
| deepseek7b | L26C8587 | single_axis | 12 | 5 | 0 | 0 | 0 | 0.984 | 0.000 | {"flip": 8, "half": 1, "zero": 3} |
| glm4 | L31C6437 | model_U | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | {} |

## Head groups

| model | head | closure lost | damage > 0.25 | mean damage | max damage | subsets |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| qwen3 | L31H19 | 2 | 0 | 0.172 | 0.250 | L31C2257 |
| qwen3 | L31H26 | 2 | 0 | 0.156 | 0.250 | L31C2257 |
| qwen3 | L31H30 | 2 | 0 | 0.078 | 0.125 | L31C2257 |
| qwen3 | L31H12 | 2 | 1 | 0.062 | 0.375 | L31C2257 |
| qwen3 | L31H17 | 1 | 2 | 0.188 | 0.375 | L31C2257 |
| qwen3 | L31H16 | 1 | 0 | 0.172 | 0.250 | L31C2257 |
| qwen3 | L31H18 | 1 | 0 | 0.156 | 0.250 | L31C2257 |
| qwen3 | L31H1 | 1 | 0 | 0.109 | 0.125 | L31C2257 |
| qwen3 | L31H31 | 1 | 0 | 0.094 | 0.250 | L31C2257 |
| qwen3 | L31H13 | 1 | 0 | 0.062 | 0.125 | L31C2257 |
| qwen3 | L31H22 | 1 | 0 | 0.062 | 0.125 | L31C2257 |
| qwen3 | L31H0 | 1 | 0 | 0.047 | 0.125 | L31C2257 |
| qwen3 | L31H2 | 1 | 0 | 0.047 | 0.250 | L31C2257 |
| qwen3 | L31H6 | 1 | 0 | 0.047 | 0.125 | L31C2257 |
| qwen3 | L31H7 | 1 | 0 | 0.031 | 0.125 | L31C2257 |
| qwen3 | L31H10 | 1 | 0 | 0.031 | 0.125 | L31C2257 |
| qwen3 | L31H20 | 1 | 0 | 0.031 | 0.125 | L31C2257 |
| qwen3 | L31H23 | 1 | 0 | 0.031 | 0.125 | L31C2257 |
| qwen3 | L31H25 | 1 | 0 | 0.031 | 0.125 | L31C2257 |
| qwen3 | L31H14 | 1 | 0 | 0.016 | 0.125 | L31C2257 |
| qwen3 | L31H27 | 1 | 0 | 0.016 | 0.125 | L31C2257 |
| qwen3 | L31H21 | 1 | 0 | 0.000 | 0.125 | L31C2257 |
| deepseek7b | L26H3 | 0 | 4 | 0.617 | 1.875 | L26C8587+L27C15369 |
| deepseek7b | L26H7 | 0 | 6 | 0.492 | 0.750 | L26C8587+L27C15369 |
| deepseek7b | L26H11 | 0 | 5 | 0.367 | 0.625 | L26C8587+L27C15369 |
| deepseek7b | L26H14 | 0 | 3 | 0.273 | 0.875 | L26C8587+L27C15369 |
| deepseek7b | L26H13 | 0 | 1 | 0.203 | 0.375 | L26C8587+L27C15369 |
| deepseek7b | L27H15 | 0 | 2 | 0.195 | 0.500 | L26C8587+L27C15369 |
| deepseek7b | L26H23 | 0 | 2 | 0.180 | 0.500 | L26C8587+L27C15369 |
| deepseek7b | L26H12 | 0 | 1 | 0.148 | 0.375 | L26C8587+L27C15369 |
