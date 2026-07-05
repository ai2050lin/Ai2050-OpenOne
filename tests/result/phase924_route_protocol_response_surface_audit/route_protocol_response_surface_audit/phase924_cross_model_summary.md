# Phase 924 route alpha protocol pressure response surface audit

## Overall

- models: qwen3, glm4, deepseek7b
- all_improved_margin_vs_surface_base: 241
- all_lost_margin_closure_vs_surface_base: 20
- all_margin_nonnegative: 174
- all_new_margin_closure_vs_surface_base: 2
- all_new_top1_vs_surface_base: 2
- all_rows: 576
- all_strict_clean_candidate: 132
- all_top1: 174
- non_base_improved_margin_vs_surface_base: 241
- non_base_lost_margin_closure_vs_surface_base: 20
- non_base_margin_nonnegative: 166
- non_base_new_margin_closure_vs_surface_base: 2
- non_base_new_top1_vs_surface_base: 2
- non_base_rows: 552
- non_base_strict_clean_candidate: 126
- non_base_top1: 166
- selected_phase915_l39_candidates: 12
- surface_base_improved_margin_vs_surface_base: 0
- surface_base_lost_margin_closure_vs_surface_base: 0
- surface_base_margin_nonnegative: 8
- surface_base_new_margin_closure_vs_surface_base: 0
- surface_base_new_top1_vs_surface_base: 0
- surface_base_rows: 24
- surface_base_strict_clean_candidate: 6
- surface_base_top1: 8
- surfaces_best_alpha_eq_1: 0
- surfaces_best_alpha_gt_1: 16
- surfaces_best_alpha_lt_1: 8
- surfaces_best_coord_is_base: 0
- surfaces_best_protocol_eq_1: 10
- surfaces_best_protocol_gt_1: 3
- surfaces_best_protocol_lt_1: 11
- surfaces_surface_count: 24
- surfaces_with_closure_coord: 10
- target_state_count: 12

## Model Summaries

| model | selected | states | surfaces | best base | best prot<1 | best prot=1 | best prot>1 | base top1 | non-base new top1 | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |
| glm4 | 12 | 12 | 24 | 0 | 11 | 10 | 3 | 8 | 2 | route_protocol_surface_changes_best_coordinate |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |

## Top Alpha Protocol Coordinates

| model | alpha | protocol | rows | top1 | margin | strict | improved | new margin | new top1 | lost margin | mean delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | 0.875 | 1.0 | 24 | 10 | 10 | 6 | 18 | 2 | 2 | 0 | 0.08333333333333333 |
| glm4 | 1.25 | 1.0 | 24 | 8 | 8 | 6 | 22 | 0 | 0 | 0 | 0.1328125 |
| glm4 | 1.375 | 1.1 | 24 | 8 | 8 | 6 | 18 | 0 | 0 | 0 | 0.10416666666666667 |
| glm4 | 0.75 | 0.9 | 24 | 6 | 6 | 6 | 17 | 0 | 0 | 2 | 0.015625 |
| glm4 | 1.125 | 0.85 | 24 | 8 | 8 | 6 | 16 | 0 | 0 | 0 | 0.09114583333333333 |
| glm4 | 1.125 | 0.9 | 24 | 8 | 8 | 6 | 16 | 0 | 0 | 0 | 0.06510416666666667 |
| glm4 | 0.875 | 0.85 | 24 | 8 | 8 | 6 | 15 | 0 | 0 | 0 | 0.0390625 |
| glm4 | 0.75 | 0.85 | 24 | 6 | 6 | 6 | 14 | 0 | 0 | 2 | 0.0 |
| glm4 | 1.375 | 1.0 | 24 | 8 | 8 | 6 | 13 | 0 | 0 | 0 | 0.0703125 |
| glm4 | 1.0 | 1.1 | 24 | 8 | 8 | 6 | 12 | 0 | 0 | 0 | 0.0390625 |
| glm4 | 1.0 | 0.85 | 24 | 8 | 8 | 6 | 12 | 0 | 0 | 0 | 0.018229166666666668 |
| glm4 | 1.125 | 1.0 | 24 | 8 | 8 | 6 | 10 | 0 | 0 | 0 | 0.033854166666666664 |
| glm4 | 1.25 | 0.85 | 24 | 8 | 8 | 6 | 9 | 0 | 0 | 0 | 0.052083333333333336 |
| glm4 | 1.375 | 0.85 | 24 | 8 | 8 | 6 | 8 | 0 | 0 | 0 | -0.033854166666666664 |
| glm4 | 1.25 | 0.9 | 24 | 8 | 8 | 6 | 7 | 0 | 0 | 0 | 0.052083333333333336 |
| glm4 | 1.25 | 1.1 | 24 | 8 | 8 | 6 | 7 | 0 | 0 | 0 | 0.0026041666666666665 |
| glm4 | 0.875 | 0.9 | 24 | 8 | 8 | 6 | 6 | 0 | 0 | 0 | -0.013020833333333334 |
| glm4 | 1.375 | 0.9 | 24 | 8 | 8 | 6 | 6 | 0 | 0 | 0 | -0.0234375 |
| glm4 | 0.875 | 1.1 | 24 | 5 | 5 | 3 | 5 | 0 | 0 | 3 | -0.09765625 |
| glm4 | 1.0 | 0.9 | 24 | 8 | 8 | 6 | 4 | 0 | 0 | 0 | -0.018229166666666668 |
| glm4 | 0.75 | 1.0 | 24 | 3 | 3 | 3 | 4 | 0 | 0 | 5 | -0.20833333333333334 |
| glm4 | 1.125 | 1.1 | 24 | 8 | 8 | 6 | 2 | 0 | 0 | 0 | -0.057291666666666664 |
| glm4 | 0.75 | 1.1 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | -3.462890625 |
| glm4 | 1.0 | 1.0 | 24 | 8 | 8 | 6 | 0 | 0 | 0 | 0 | 0.0 |

## Top Surfaces

| model | state | factor | best alpha | best protocol | best margin | base margin | delta | closures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.375 | 1.375 | 1.1 | 0.375 | 0.1875 | 0.1875 | [[0.75, 0.85], [0.75, 0.9], [0.75, 1.0], [0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [0.875, 1.1], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.375 | 1.375 | 1.1 | 0.375 | 0.1875 | 0.1875 | [[0.75, 0.85], [0.75, 0.9], [0.75, 1.0], [0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [0.875, 1.1], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.375 | 1.375 | 1.1 | 0.375 | 0.1875 | 0.1875 | [[0.75, 0.85], [0.75, 0.9], [0.75, 1.0], [0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [0.875, 1.1], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.4 | 1.375 | 1.25 | 1.0 | 0.25 | 0.125 | 0.125 | [[0.75, 0.85], [0.75, 0.9], [0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.375 | 1.25 | 1.0 | 0.25 | 0.125 | 0.125 | [[0.75, 0.85], [0.75, 0.9], [0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.375 | 1.25 | 1.0 | 0.25 | 0.125 | 0.125 | [[0.75, 0.85], [0.75, 0.9], [0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.375 | 0.875 | 1.0 | 0.4375 | 0.3125 | 0.125 | [[0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [0.875, 1.1], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.3 | 1.375 | 0.875 | 1.0 | 0.4375 | 0.3125 | 0.125 | [[0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [0.875, 1.1], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.25 | 0.875 | 1.0 | 0.0 | -0.125 | 0.125 | [[0.875, 1.0]] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.3 | 1.25 | 0.875 | 1.0 | 0.0 | -0.125 | 0.125 | [[0.875, 1.0]] |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.25 | 1.25 | 0.9 | -0.5625 | -1.0625 | 0.5 | [] |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.3 | 1.375 | 1.125 | 0.85 | -0.1875 | -0.625 | 0.4375 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.375 | 1.25 | 0.9 | -0.625 | -1.0625 | 0.4375 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.25 | 1.25 | 0.9 | -1.0 | -1.375 | 0.375 | [] |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.4 | 1.25 | 0.75 | 0.9 | -1.0625 | -1.3125 | 0.25 | [] |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.4 | 1.375 | 0.75 | 0.9 | -0.6875 | -0.9375 | 0.25 | [] |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.4 | 1.25 | 0.75 | 0.9 | -1.0625 | -1.3125 | 0.25 | [] |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.4 | 1.375 | 0.75 | 0.9 | -0.6875 | -0.9375 | 0.25 | [] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.4 | 1.25 | 1.25 | 1.0 | -0.125 | -0.3125 | 0.1875 | [] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.25 | 1.25 | 1.0 | -0.125 | -0.3125 | 0.1875 | [] |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.4 | 1.25 | 1.25 | 1.0 | -0.125 | -0.3125 | 0.1875 | [] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.3 | 1.25 | 1.125 | 0.9 | -0.0625 | -0.1875 | 0.125 | [] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.25 | 1.125 | 0.9 | -0.0625 | -0.1875 | 0.125 | [] |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.3 | 1.25 | 1.125 | 0.9 | -0.0625 | -0.1875 | 0.125 | [] |
