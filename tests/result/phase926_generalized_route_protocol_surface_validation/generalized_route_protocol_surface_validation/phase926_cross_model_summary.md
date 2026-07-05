# Phase 926 generalized route-protocol surface validation

## Overall

- models: qwen3, glm4, deepseek7b
- all_improved_margin_vs_surface_base: 542
- all_lost_margin_closure_vs_surface_base: 25
- all_margin_nonnegative: 97
- all_new_margin_closure_vs_surface_base: 2
- all_new_strict_vs_surface_base: 0
- all_new_top1_vs_surface_base: 2
- all_rows: 1440
- all_strict_clean_candidate: 57
- all_top1: 97
- expected_rows_if_all_reconstructed: 1440
- non_base_improved_margin_vs_surface_base: 542
- non_base_lost_margin_closure_vs_surface_base: 25
- non_base_margin_nonnegative: 92
- non_base_new_margin_closure_vs_surface_base: 2
- non_base_new_strict_vs_surface_base: 0
- non_base_new_top1_vs_surface_base: 2
- non_base_rows: 1380
- non_base_strict_clean_candidate: 54
- non_base_top1: 92
- phase925_selected_seed_count: 30
- surface_base_improved_margin_vs_surface_base: 0
- surface_base_lost_margin_closure_vs_surface_base: 0
- surface_base_margin_nonnegative: 5
- surface_base_new_margin_closure_vs_surface_base: 0
- surface_base_new_strict_vs_surface_base: 0
- surface_base_new_top1_vs_surface_base: 0
- surface_base_rows: 60
- surface_base_strict_clean_candidate: 3
- surface_base_top1: 5
- surfaces_best_alpha_eq_1: 14
- surfaces_best_alpha_gt_1: 35
- surfaces_best_alpha_lt_1: 11
- surfaces_best_coord_is_base: 8
- surfaces_best_protocol_eq_1: 11
- surfaces_best_protocol_gt_1: 18
- surfaces_best_protocol_lt_1: 31
- surfaces_surface_count: 60
- surfaces_with_closure_coord: 6
- target_state_count: 30

## Model Summaries

| model | selected | states | rows | surfaces | best base | best alpha<1 | best alpha>1 | best prot<1 | best prot>1 | non-base new top1 | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase925_surface_seeds |
| glm4 | 30 | 30 | 1440 | 60 | 8 | 11 | 35 | 31 | 18 | 2 | generalized_surface_adds_top1_closure |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase925_surface_seeds |

## Top Surfaces

| model | state | domain | blocker class | group | factor | best alpha | best protocol | best margin | base margin | delta | closures |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.3 | object | article_a | top_abs_64 | 1.375 | 1.25 | 1.0 | 0.1875 | 0.0 | 0.1875 | [[0.75, 0.85], [0.75, 0.9], [0.75, 1.0], [0.875, 0.85], [0.875, 1.0], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.3 | object | article_a | top_abs_64 | 1.375 | 1.25 | 1.0 | 0.1875 | 0.0 | 0.1875 | [[0.75, 0.85], [0.75, 0.9], [0.75, 1.0], [0.875, 0.85], [0.875, 1.0], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.3 | object | article_a | top_abs_64 | 1.375 | 1.25 | 1.0 | 0.1875 | 0.0 | 0.1875 | [[0.75, 0.85], [0.75, 0.9], [0.75, 1.0], [0.875, 0.85], [0.875, 1.0], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.3 | animal | article_a | top_abs_64 | 1.375 | 1.0 | 1.1 | 0.25 | 0.1875 | 0.0625 | [[0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.3 | animal | article_a | top_abs_64 | 1.375 | 1.0 | 1.1 | 0.25 | 0.1875 | 0.0625 | [[0.875, 0.85], [0.875, 0.9], [0.875, 1.0], [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], [1.125, 0.85], [1.125, 0.9], [1.125, 1.0], [1.125, 1.1], [1.25, 0.85], [1.25, 0.9], [1.25, 1.0], [1.25, 1.1], [1.375, 0.85], [1.375, 0.9], [1.375, 1.0], [1.375, 1.1]] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|band32_support_64|0.4 | animal | article_a | band32_support_64 | 1.375 | 1.375 | 0.9 | 0.0625 | -0.0625 | 0.125 | [[1.375, 0.85], [1.375, 0.9]] |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.3 | material | article_a | top_abs_64 | 1.25 | 1.25 | 0.9 | -0.625 | -1.125 | 0.5 | [] |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.3 | material | article_a | top_abs_64 | 1.375 | 1.125 | 0.85 | -0.375 | -0.875 | 0.5 | [] |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.4 | material | article_a | top_abs_64 | 1.25 | 1.25 | 0.9 | -0.75 | -1.25 | 0.5 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|zero|same_domain_holdout_case|top_abs_64|0.3 | animal | article_a | top_abs_64 | 1.375 | 1.25 | 0.9 | -0.75 | -1.1875 | 0.4375 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.3 | animal | article_a | top_abs_64 | 1.375 | 1.25 | 0.9 | -0.75 | -1.1875 | 0.4375 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|zero|same_domain_holdout_case|top_abs_64|0.3 | animal | article_a | top_abs_64 | 1.25 | 1.25 | 0.9 | -1.0625 | -1.4375 | 0.375 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.3 | animal | article_a | top_abs_64 | 1.25 | 1.25 | 0.9 | -1.0625 | -1.4375 | 0.375 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|zero|same_domain_holdout_case|top_abs_64|0.4 | animal | article_a | top_abs_64 | 1.375 | 1.25 | 0.9 | -0.875 | -1.25 | 0.375 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.4 | animal | article_a | top_abs_64 | 1.375 | 1.25 | 0.9 | -0.875 | -1.25 | 0.375 | [] |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.4 | material | article_a | top_abs_64 | 1.375 | 1.25 | 0.9 | -0.5 | -0.875 | 0.375 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|zero|same_domain_holdout_case|top_abs_64|0.4 | animal | article_a | top_abs_64 | 1.25 | 1.25 | 0.9 | -1.25 | -1.5625 | 0.3125 | [] |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.4 | animal | article_a | top_abs_64 | 1.25 | 1.25 | 0.9 | -1.25 | -1.5625 | 0.3125 | [] |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | animal | punctuation_period | band16_support_32 | 1.25 | 0.875 | 0.85 | -3.25 | -3.5625 | 0.3125 | [] |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | animal | punctuation_period | band16_support_32 | 1.375 | 0.875 | 0.85 | -3.0625 | -3.375 | 0.3125 | [] |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | animal | punctuation_period | band16_support_32 | 1.25 | 0.875 | 0.85 | -3.25 | -3.5625 | 0.3125 | [] |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | animal | punctuation_period | band16_support_32 | 1.375 | 0.875 | 0.85 | -3.0625 | -3.375 | 0.3125 | [] |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band32_support_64|0.3 | material | article_a | band32_support_64 | 1.25 | 1.25 | 0.9 | -0.6875 | -1.0 | 0.3125 | [] |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.4 | animal | article_a | top_abs_64 | 1.25 | 0.75 | 0.9 | -1.125 | -1.375 | 0.25 | [] |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.4 | animal | article_a | top_abs_64 | 1.375 | 0.75 | 0.9 | -0.8125 | -1.0625 | 0.25 | [] |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | animal | punctuation_period | low_abs_64 | 1.375 | 1.375 | 0.85 | -3.0625 | -3.3125 | 0.25 | [] |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | object | punctuation_period | low_abs_64 | 1.25 | 1.25 | 1.1 | -5.34375 | -5.59375 | 0.25 | [] |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | object | punctuation_period | low_abs_64 | 1.25 | 1.25 | 1.1 | -5.34375 | -5.59375 | 0.25 | [] |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | object | punctuation_period | low_abs_64 | 1.25 | 1.25 | 1.1 | -5.34375 | -5.59375 | 0.25 | [] |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | object | punctuation_period | low_abs_64 | 1.25 | 1.25 | 1.1 | -5.34375 | -5.59375 | 0.25 | [] |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.3 | object | article_a | top_abs_64 | 1.25 | 1.125 | 0.85 | -0.1875 | -0.375 | 0.1875 | [] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.3 | object | article_a | top_abs_64 | 1.25 | 1.125 | 0.85 | -0.1875 | -0.375 | 0.1875 | [] |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.3 | object | article_a | top_abs_64 | 1.25 | 1.125 | 0.85 | -0.1875 | -0.375 | 0.1875 | [] |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | animal | punctuation_period | low_abs_64 | 1.25 | 1.125 | 0.85 | -3.25 | -3.4375 | 0.1875 | [] |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | animal | punctuation_period | low_abs_64 | 1.375 | 0.875 | 0.9 | -3.125 | -3.3125 | 0.1875 | [] |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | animal | punctuation_period | low_abs_64 | 1.25 | 1.125 | 0.85 | -3.25 | -3.4375 | 0.1875 | [] |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band32_support_64|0.3 | material | article_a | band32_support_64 | 1.375 | 1.25 | 0.9 | -0.4375 | -0.625 | 0.1875 | [] |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | material | punctuation_period | low_abs_64 | 1.25 | 1.0 | 1.1 | -3.75 | -3.9375 | 0.1875 | [] |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | material | punctuation_period | low_abs_64 | 1.375 | 0.875 | 1.1 | -3.5625 | -3.75 | 0.1875 | [] |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | material | punctuation_period | low_abs_64 | 1.25 | 1.0 | 1.1 | -3.75 | -3.9375 | 0.1875 | [] |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | material | punctuation_period | low_abs_64 | 1.375 | 0.875 | 1.1 | -3.5625 | -3.75 | 0.1875 | [] |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | object | punctuation_period | low_abs_64 | 1.375 | 1.25 | 1.1 | -5.1875 | -5.375 | 0.1875 | [] |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | object | punctuation_period | low_abs_64 | 1.375 | 1.25 | 1.1 | -5.1875 | -5.375 | 0.1875 | [] |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | object | punctuation_period | low_abs_64 | 1.375 | 1.25 | 1.1 | -5.1875 | -5.375 | 0.1875 | [] |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | object | punctuation_period | low_abs_64 | 1.375 | 1.25 | 1.1 | -5.1875 | -5.375 | 0.1875 | [] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.3 | animal | article_a | top_abs_64 | 1.25 | 1.375 | 0.85 | -0.0625 | -0.1875 | 0.125 | [] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.3 | animal | article_a | top_abs_64 | 1.25 | 1.375 | 0.85 | -0.0625 | -0.1875 | 0.125 | [] |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|band32_support_64|0.4 | animal | article_a | band32_support_64 | 1.25 | 1.375 | 0.9 | -0.3125 | -0.4375 | 0.125 | [] |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | material | punctuation_period | low_abs_64 | 1.25 | 1.0 | 1.1 | -3.8125 | -3.875 | 0.0625 | [] |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | material | punctuation_period | low_abs_64 | 1.375 | 0.875 | 1.1 | -3.625 | -3.6875 | 0.0625 | [] |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | material | punctuation_period | low_abs_64 | 1.25 | 1.0 | 1.1 | -3.8125 | -3.875 | 0.0625 | [] |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | material | punctuation_period | low_abs_64 | 1.375 | 0.875 | 1.1 | -3.625 | -3.6875 | 0.0625 | [] |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|zero|source_case_prompt_variant|band32_support_64|0.8 | object | article_a | band32_support_64 | 1.25 | 1.0 | 1.0 | -0.5 | -0.5 | 0.0 | [] |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|zero|source_case_prompt_variant|band32_support_64|0.8 | object | article_a | band32_support_64 | 1.375 | 1.0 | 1.0 | -0.25 | -0.25 | 0.0 | [] |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|flip|source_case_prompt_variant|band32_support_64|0.8 | object | article_a | band32_support_64 | 1.25 | 1.0 | 1.0 | -0.5 | -0.5 | 0.0 | [] |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|flip|source_case_prompt_variant|band32_support_64|0.8 | object | article_a | band32_support_64 | 1.375 | 1.0 | 1.0 | -0.25 | -0.25 | 0.0 | [] |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|low_abs_64|0.5 | material | article_a | low_abs_64 | 1.25 | 1.0 | 1.0 | -0.625 | -0.625 | 0.0 | [] |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|low_abs_64|0.5 | material | article_a | low_abs_64 | 1.375 | 1.0 | 1.0 | -0.25 | -0.25 | 0.0 | [] |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|low_abs_64|0.4 | material | article_a | low_abs_64 | 1.25 | 1.0 | 1.0 | -0.625 | -0.625 | 0.0 | [] |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|low_abs_64|0.4 | material | article_a | low_abs_64 | 1.375 | 1.0 | 1.0 | -0.25 | -0.25 | 0.0 | [] |

## Top Alpha Protocol Coordinates

| model | alpha | protocol | rows | top1 | margin | strict | improved | new margin | new top1 | mean delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | 1.375 | 0.85 | 60 | 3 | 3 | 0 | 28 | 1 | 1 | -0.003125 |
| glm4 | 1.375 | 0.9 | 60 | 3 | 3 | 0 | 25 | 1 | 1 | -0.005208333333333333 |
| glm4 | 1.0 | 1.1 | 60 | 5 | 5 | 3 | 36 | 0 | 0 | -0.036458333333333336 |
| glm4 | 1.25 | 1.1 | 60 | 5 | 5 | 3 | 29 | 0 | 0 | -0.009375 |
| glm4 | 0.875 | 1.1 | 60 | 0 | 0 | 0 | 29 | 0 | 0 | -2.4945963541666667 |
| glm4 | 1.125 | 0.85 | 60 | 5 | 5 | 3 | 28 | 0 | 0 | 0.014583333333333334 |
| glm4 | 1.375 | 1.1 | 60 | 5 | 5 | 3 | 28 | 0 | 0 | -0.003125 |
| glm4 | 0.75 | 0.9 | 60 | 3 | 3 | 3 | 28 | 0 | 0 | -2.4880859375 |
| glm4 | 1.125 | 0.9 | 60 | 5 | 5 | 3 | 28 | 0 | 0 | 0.0 |
| glm4 | 1.25 | 1.0 | 60 | 5 | 5 | 3 | 26 | 0 | 0 | 0.019791666666666666 |
| glm4 | 1.25 | 0.85 | 60 | 5 | 5 | 3 | 25 | 0 | 0 | 0.03854166666666667 |
| glm4 | 1.25 | 0.9 | 60 | 5 | 5 | 3 | 24 | 0 | 0 | 0.03125 |
| glm4 | 1.375 | 1.0 | 60 | 5 | 5 | 3 | 24 | 0 | 0 | 0.021875 |
| glm4 | 0.875 | 0.85 | 60 | 5 | 5 | 3 | 23 | 0 | 0 | -0.0625 |
| glm4 | 0.75 | 0.85 | 60 | 3 | 3 | 3 | 23 | 0 | 0 | -2.0662109375 |
| glm4 | 0.875 | 1.0 | 60 | 5 | 5 | 3 | 22 | 0 | 0 | -2.039453125 |
| glm4 | 0.75 | 1.0 | 60 | 3 | 3 | 3 | 21 | 0 | 0 | -2.550065104166667 |
| glm4 | 1.0 | 0.85 | 60 | 5 | 5 | 3 | 19 | 0 | 0 | -0.017708333333333333 |
| glm4 | 1.125 | 1.0 | 60 | 5 | 5 | 3 | 16 | 0 | 0 | -0.030208333333333334 |
| glm4 | 0.875 | 0.9 | 60 | 2 | 2 | 0 | 16 | 0 | 0 | -0.13854166666666667 |
| glm4 | 0.75 | 1.1 | 60 | 0 | 0 | 0 | 16 | 0 | 0 | -4.3810546875 |
| glm4 | 1.125 | 1.1 | 60 | 5 | 5 | 3 | 15 | 0 | 0 | -0.03125 |
| glm4 | 1.0 | 0.9 | 60 | 5 | 5 | 3 | 13 | 0 | 0 | -0.042708333333333334 |
| glm4 | 1.0 | 1.0 | 60 | 5 | 5 | 3 | 0 | 0 | 0 | 0.0 |
