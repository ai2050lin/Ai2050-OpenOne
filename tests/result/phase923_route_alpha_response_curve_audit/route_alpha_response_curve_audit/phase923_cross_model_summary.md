# Phase 923 route alpha response curve audit

## Overall

- models: qwen3, glm4, deepseek7b
- all_improved_margin_vs_alpha1: 122
- all_lost_margin_closure_vs_alpha1: 21
- all_margin_nonnegative: 53
- all_new_margin_closure_vs_alpha1: 2
- all_new_top1_vs_alpha1: 2
- all_rows: 324
- all_strict_clean_candidate: 39
- all_top1: 53
- alpha1_improved_margin_vs_alpha1: 0
- alpha1_lost_margin_closure_vs_alpha1: 0
- alpha1_margin_nonnegative: 8
- alpha1_new_margin_closure_vs_alpha1: 0
- alpha1_new_top1_vs_alpha1: 0
- alpha1_rows: 36
- alpha1_strict_clean_candidate: 6
- alpha1_top1: 8
- curves_best_alpha_eq_1: 0
- curves_best_alpha_gt_1: 24
- curves_best_alpha_lt_1: 12
- curves_curve_count: 36
- curves_monotonic_non_decreasing: 0
- curves_monotonic_non_increasing: 0
- curves_with_closure_alpha: 10
- non_alpha1_improved_margin_vs_alpha1: 122
- non_alpha1_lost_margin_closure_vs_alpha1: 21
- non_alpha1_margin_nonnegative: 45
- non_alpha1_new_margin_closure_vs_alpha1: 2
- non_alpha1_new_top1_vs_alpha1: 2
- non_alpha1_rows: 288
- non_alpha1_strict_clean_candidate: 33
- non_alpha1_top1: 45
- selected_phase915_l39_candidates: 12
- target_state_count: 12

## Model Summaries

| model | selected | states | curves | best<1 | best=1 | best>1 | mono up | mono down | alpha1 top1 | non-alpha1 new top1 | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |
| glm4 | 12 | 12 | 36 | 12 | 0 | 24 | 0 | 0 | 8 | 2 | route_alpha_response_nonmonotonic_mixed_peak |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |

## Top Alphas

| model | alpha | rows | top1 | margin | strict | improved | new margin | new top1 | lost margin | mean delta | median margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | 0.875 | 36 | 10 | 10 | 6 | 27 | 2 | 2 | 0 | 0.07291666666666667 | -0.5 |
| glm4 | 1.25 | 36 | 8 | 8 | 6 | 32 | 0 | 0 | 0 | 0.1440972222222222 | -0.375 |
| glm4 | 1.375 | 36 | 8 | 8 | 6 | 21 | 0 | 0 | 0 | 0.1032986111111111 | -0.4375 |
| glm4 | 1.125 | 36 | 8 | 8 | 6 | 14 | 0 | 0 | 0 | 0.03298611111111111 | -0.5 |
| glm4 | 1.5 | 36 | 8 | 8 | 6 | 14 | 0 | 0 | 0 | 0.011284722222222222 | -0.4375 |
| glm4 | 0.75 | 36 | 3 | 3 | 3 | 11 | 0 | 0 | 5 | -0.1605902777777778 | -0.75 |
| glm4 | 0.625 | 36 | 0 | 0 | 0 | 3 | 0 | 0 | 8 | -4.694444444444445 | -3.375 |
| glm4 | 0.5 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | -12.883246527777779 | -13.44140625 |
| glm4 | 1.0 | 36 | 8 | 8 | 6 | 0 | 0 | 0 | 0 | 0.0 | -0.5625 |

## Top Curves

| model | state | factor | best alpha | best margin | alpha1 margin | delta | closure alphas | monotonic up | monotonic down |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.375 | 1.25 | 0.3125 | 0.1875 | 0.125 | [0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.375 | 1.25 | 0.3125 | 0.1875 | 0.125 | [0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.375 | 1.25 | 0.3125 | 0.1875 | 0.125 | [0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5] | False | False |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.375 | 0.875 | 0.4375 | 0.3125 | 0.125 | [0.875, 1.0, 1.125, 1.25, 1.375, 1.5] | False | False |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.3 | 1.375 | 0.875 | 0.4375 | 0.3125 | 0.125 | [0.875, 1.0, 1.125, 1.25, 1.375, 1.5] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.4 | 1.375 | 1.25 | 0.25 | 0.125 | 0.125 | [0.875, 1.0, 1.125, 1.25, 1.375, 1.5] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.375 | 1.25 | 0.25 | 0.125 | 0.125 | [0.875, 1.0, 1.125, 1.25, 1.375, 1.5] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.375 | 1.25 | 0.25 | 0.125 | 0.125 | [0.875, 1.0, 1.125, 1.25, 1.375, 1.5] | False | False |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.25 | 0.875 | 0.0 | -0.125 | 0.125 | [0.875] | False | False |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.3 | 1.25 | 0.875 | 0.0 | -0.125 | 0.125 | [0.875] | False | False |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.25 | 1.375 | -0.625 | -1.0625 | 0.4375 | [] | False | False |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.125 | 1.375 | -1.40625 | -1.8125 | 0.40625 | [] | False | False |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.125 | 1.375 | -1.0 | -1.375 | 0.375 | [] | False | False |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.375 | 1.25 | -0.25 | -0.625 | 0.375 | [] | False | False |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.25 | 1.375 | -1.0625 | -1.375 | 0.3125 | [] | False | False |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.375 | 1.375 | -0.75 | -1.0625 | 0.3125 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.125 | 1.25 | -0.375 | -0.625 | 0.25 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.125 | 1.25 | -0.375 | -0.625 | 0.25 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.125 | 1.25 | -0.375 | -0.625 | 0.25 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.4 | 1.125 | 1.375 | -0.5 | -0.75 | 0.25 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.125 | 1.375 | -0.5 | -0.75 | 0.25 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.125 | 1.375 | -0.5 | -0.75 | 0.25 | [] | False | False |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.4 | 1.25 | 0.875 | -1.125 | -1.3125 | 0.1875 | [] | False | False |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.4 | 1.375 | 0.875 | -0.75 | -0.9375 | 0.1875 | [] | False | False |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.4 | 1.25 | 0.875 | -1.125 | -1.3125 | 0.1875 | [] | False | False |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.4 | 1.375 | 0.875 | -0.75 | -0.9375 | 0.1875 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.4 | 1.25 | 1.25 | -0.125 | -0.3125 | 0.1875 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.25 | 1.25 | -0.125 | -0.3125 | 0.1875 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.25 | 1.25 | -0.125 | -0.3125 | 0.1875 | [] | False | False |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.4 | 1.125 | 0.875 | -1.5625 | -1.6875 | 0.125 | [] | False | False |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.4 | 1.125 | 0.875 | -1.5625 | -1.6875 | 0.125 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.25 | 1.25 | -0.0625 | -0.1875 | 0.125 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.25 | 1.25 | -0.0625 | -0.1875 | 0.125 | [] | False | False |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.25 | 1.25 | -0.0625 | -0.1875 | 0.125 | [] | False | False |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.125 | 0.875 | -0.4375 | -0.5 | 0.0625 | [] | False | False |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.3 | 1.125 | 0.875 | -0.4375 | -0.5 | 0.0625 | [] | False | False |
