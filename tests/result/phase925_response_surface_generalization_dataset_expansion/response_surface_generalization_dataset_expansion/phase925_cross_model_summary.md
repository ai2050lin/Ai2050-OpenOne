# Phase 925 response surface generalization dataset expansion

## Overall

- models: qwen3, glm4, deepseek7b
- candidate_unique_states: 1380
- phase914_rows: 1688
- selected_already_surface_tested_phase924: 12
- selected_blocker_is_target: 66
- selected_new_surface_seed_vs_phase924: 84
- selected_present_in_phase915_boundary_set: 12
- selected_rows: 96
- selected_strict_clean_candidate: 0
- selected_strong_holdout_candidate: 0
- selected_surface_seeds: 96
- selected_top10: 70
- selected_top5: 18
- selected_top50: 96
- selected_unique_cases: 10
- selected_unique_domains: 3
- selected_unique_groups: 5
- selected_unique_prompt_variants: 4
- selected_weak_holdout_candidate: 12

## Model Summaries

| model | phase914 rows | candidate states | selected | new vs P924 | cases | domains | blocker target | top5 | top10 | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_expandable_response_surface_candidates |
| glm4 | 1496 | 1380 | 96 | 84 | 10 | 3 | 66 | 18 | 70 | expanded_surface_seed_set_ready |
| deepseek7b | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_expandable_response_surface_candidates |

## Top Selected Seeds

| model | state | case | domain | prompt | group | factor | blocker | rank | margin | score | new vs P924 |
| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.3 | p856_008_animal_bird | animal | natural_question | top_abs_64 | 0.3 | a | 9 | -2.15625 | 212.90625 | False |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|zero|same_domain_holdout_case|top_abs_64|0.3 | p856_008_animal_bird | animal | natural_question | top_abs_64 | 0.3 | a | 9 | -2.15625 | 212.90625 | True |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|top_abs_64|0.4 | p856_008_animal_bird | animal | natural_question | top_abs_64 | 0.4 | a | 9 | -2.21875 | 212.298828125 | True |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|zero|same_domain_holdout_case|top_abs_64|0.4 | p856_008_animal_bird | animal | natural_question | top_abs_64 | 0.4 | a | 9 | -2.21875 | 212.298828125 | True |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.3 | p856_022_material_iron | material | natural_question | top_abs_64 | 0.3 | a | 11 | -1.84375 | 175.857421875 | False |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.3 | p856_009_animal_fish | animal | question_plain | top_abs_64 | 0.3 | a | 5 | -0.9375 | 175.322265625 | False |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.3 | p856_009_animal_fish | animal | question_plain | top_abs_64 | 0.3 | a | 5 | -0.9375 | 175.322265625 | False |
| glm4 | p856_022_material_iron|natural_question|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.4 | p856_022_material_iron | material | natural_question | top_abs_64 | 0.4 | a | 12 | -1.90625 | 174.912109375 | True |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.3 | p856_038_object_object | object | natural_question | top_abs_64 | 0.3 | a | 5 | -1.125 | 173.279296875 | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.3 | p856_038_object_object | object | natural_question | top_abs_64 | 0.3 | a | 5 | -1.125 | 173.279296875 | False |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.3 | p856_038_object_object | object | natural_question | top_abs_64 | 0.3 | a | 5 | -1.125 | 173.279296875 | False |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.4 | p856_009_animal_fish | animal | natural_question | top_abs_64 | 0.4 | a | 14 | -2.0625 | 173.0625 | False |
| glm4 | p856_009_animal_fish|natural_question|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.4 | p856_009_animal_fish | animal | natural_question | top_abs_64 | 0.4 | a | 14 | -2.0625 | 173.0625 | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|top_abs_64|0.4 | p856_038_object_object | object | natural_question | top_abs_64 | 0.4 | a | 5 | -1.1875 | 172.6015625 | False |
| glm4 | p856_038_object_object|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|top_abs_64|0.4 | p856_038_object_object | object | natural_question | top_abs_64 | 0.4 | a | 5 | -1.1875 | 172.6015625 | False |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|top_abs_64|0.4 | p856_038_object_object | object | natural_question | top_abs_64 | 0.4 | a | 5 | -1.1875 | 172.6015625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|top_abs_64|0.3 | p856_021_material_wood | material | natural_question | top_abs_64 | 0.3 |  . | 7 | -4.25 | 169.966796875 | True |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|top_abs_64|0.3 | p856_021_material_wood | material | natural_question | top_abs_64 | 0.3 |  . | 7 | -4.25 | 169.966796875 | True |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|top_abs_64|0.4 | p856_021_material_wood | material | natural_question | top_abs_64 | 0.4 |  . | 7 | -4.25 | 169.69140625 | True |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|top_abs_64|0.4 | p856_021_material_wood | material | natural_question | top_abs_64 | 0.4 |  . | 7 | -4.25 | 169.69140625 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | animal | natural_category | band16_support_32 | 0.9 |  . | 5 | -3.8125 | 121.173828125 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | animal | natural_category | band16_support_32 | 0.9 |  . | 5 | -3.8125 | 121.173828125 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band32_support_64|0.7 | p885_047_animal_shark | animal | natural_category | band32_support_64 | 0.7 |  . | 5 | -3.8125 | 121.158203125 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | animal | natural_category | low_abs_64 | 0.6 |  . | 5 | -3.8125 | 121.158203125 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | animal | natural_category | low_abs_64 | 0.5 |  . | 5 | -3.8125 | 121.158203125 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band32_support_64|0.7 | p885_047_animal_shark | animal | natural_category | band32_support_64 | 0.7 |  . | 5 | -3.8125 | 121.158203125 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | animal | natural_category | low_abs_64 | 0.6 |  . | 5 | -3.8125 | 121.158203125 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | animal | natural_category | low_abs_64 | 0.5 |  . | 5 | -3.8125 | 121.158203125 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_64|0.6 | p885_047_animal_shark | animal | natural_category | band16_support_64 | 0.6 |  . | 5 | -3.8125 | 121.134765625 | True |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_64|0.6 | p885_047_animal_shark | animal | natural_category | band16_support_64 | 0.6 |  . | 5 | -3.8125 | 121.134765625 | True |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|top_abs_64|0.3 | p856_038_object_object | object | natural_question | top_abs_64 | 0.3 | a | 6 | -0.8125 | 106.4921875 | True |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|top_abs_64|0.4 | p856_009_animal_fish | animal | question_plain | top_abs_64 | 0.4 | a | 6 | -0.9375 | 105.203125 | True |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|top_abs_64|0.4 | p856_009_animal_fish | animal | question_plain | top_abs_64 | 0.4 | a | 6 | -0.9375 | 105.203125 | True |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|top_abs_64|0.5 | p856_038_object_object | object | natural_question | top_abs_64 | 0.5 | a | 6 | -0.9375 | 105.125 | True |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|top_abs_64|0.4 | p856_038_object_object | object | natural_question | top_abs_64 | 0.4 | a | 6 | -0.9375 | 105.0 | True |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.6 | p856_022_material_iron | material | question_plain | top_abs_64 | 0.6 | a | 8 | -1.0 | 104.01171875 | True |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|top_abs_64|0.5 | p856_022_material_iron | material | question_plain | top_abs_64 | 0.5 | a | 8 | -1.0 | 104.0 | True |
| glm4 | p856_038_object_object|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|top_abs_64|0.7 | p856_038_object_object | object | natural_question | top_abs_64 | 0.7 | a | 7 | -1.0625 | 103.525390625 | True |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|band32_support_64|0.4 | p856_009_animal_fish | animal | question_plain | band32_support_64 | 0.4 | a | 9 | -1.09375 | 103.30078125 | True |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|band32_support_64|0.4 | p856_009_animal_fish | animal | question_plain | band32_support_64 | 0.4 | a | 9 | -1.09375 | 103.30078125 | True |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|flip|source_case_prompt_variant|top_abs_64|0.3 | p856_036_object_car | object | question_plain | top_abs_64 | 0.3 | a | 10 | -1.03125 | 103.298828125 | True |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|zero|source_case_prompt_variant|top_abs_64|0.3 | p856_036_object_car | object | question_plain | top_abs_64 | 0.3 | a | 10 | -1.03125 | 103.298828125 | True |
| glm4 | p856_036_object_car|question_plain|L39C3652+L39C11316|flip|source_case_prompt_variant|top_abs_64|0.3 | p856_036_object_car | object | question_plain | top_abs_64 | 0.3 | a | 10 | -1.03125 | 103.298828125 | True |
| glm4 | p856_036_object_car|question_plain|L39C3652+L39C11316|zero|source_case_prompt_variant|top_abs_64|0.3 | p856_036_object_car | object | question_plain | top_abs_64 | 0.3 | a | 10 | -1.03125 | 103.298828125 | True |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|flip|source_case_prompt_variant|top_abs_64|0.5 | p856_036_object_car | object | question_plain | top_abs_64 | 0.5 | a | 9 | -1.0625 | 103.181640625 | True |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|zero|source_case_prompt_variant|top_abs_64|0.5 | p856_036_object_car | object | question_plain | top_abs_64 | 0.5 | a | 9 | -1.0625 | 103.181640625 | True |
| glm4 | p856_036_object_car|question_plain|L39C3652+L39C11316|flip|source_case_prompt_variant|top_abs_64|0.5 | p856_036_object_car | object | question_plain | top_abs_64 | 0.5 | a | 9 | -1.0625 | 103.181640625 | True |
| glm4 | p856_036_object_car|question_plain|L39C3652+L39C11316|zero|source_case_prompt_variant|top_abs_64|0.5 | p856_036_object_car | object | question_plain | top_abs_64 | 0.5 | a | 9 | -1.0625 | 103.181640625 | True |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|flip|source_case_prompt_variant|band16_support_64|0.5 | p856_009_animal_fish | animal | question_plain | band16_support_64 | 0.5 | a | 7 | -1.125 | 103.138671875 | True |
| glm4 | p856_009_animal_fish|question_plain|L35C8824|zero|source_case_prompt_variant|band16_support_64|0.5 | p856_009_animal_fish | animal | question_plain | band16_support_64 | 0.5 | a | 7 | -1.125 | 103.138671875 | True |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|flip|source_case_prompt_variant|band32_support_64|0.8 | p856_036_object_car | object | question_plain | band32_support_64 | 0.8 | a | 10 | -1.21875 | 101.44921875 | True |
| glm4 | p856_036_object_car|question_plain|L39C11316+L39C5585|zero|source_case_prompt_variant|band32_support_64|0.8 | p856_036_object_car | object | question_plain | band32_support_64 | 0.8 | a | 10 | -1.21875 | 101.44921875 | True |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|low_abs_64|0.5 | p856_022_material_iron | material | question_plain | low_abs_64 | 0.5 | a | 10 | -1.21875 | 101.34375 | True |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|low_abs_64|0.4 | p856_022_material_iron | material | question_plain | low_abs_64 | 0.4 | a | 10 | -1.21875 | 101.34375 | True |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|band16_support_32|0.9 | p856_022_material_iron | material | question_plain | band16_support_32 | 0.9 | a | 10 | -1.21875 | 101.287109375 | True |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|low_abs_64|0.3 | p856_022_material_iron | material | question_plain | low_abs_64 | 0.3 | a | 10 | -1.28125 | 100.71484375 | True |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|low_abs_64|0.9 | p856_022_material_iron | material | question_plain | low_abs_64 | 0.9 | a | 10 | -1.3125 | 100.3125 | True |
| glm4 | p856_022_material_iron|question_plain|L39C638+L39C2682|flip|source_case_prompt_variant|low_abs_64|0.8 | p856_022_material_iron | material | question_plain | low_abs_64 | 0.8 | a | 10 | -1.3125 | 100.3125 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band32_support_64|0.3 | p856_023_material_plastic | material | natural_question | band32_support_64 | 0.3 | a | 11 | -1.625 | 57.654296875 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band16_support_32|0.3 | p856_023_material_plastic | material | natural_question | band16_support_32 | 0.3 | a | 11 | -1.65625 | 57.349609375 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band16_support_64|0.3 | p856_023_material_plastic | material | natural_question | band16_support_64 | 0.3 | a | 11 | -1.6875 | 56.974609375 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band16_support_64|0.4 | p856_023_material_plastic | material | natural_question | band16_support_64 | 0.4 | a | 11 | -1.71875 | 56.587890625 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band32_support_64|0.4 | p856_023_material_plastic | material | natural_question | band32_support_64 | 0.4 | a | 11 | -1.71875 | 56.55078125 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band16_support_32|0.4 | p856_023_material_plastic | material | natural_question | band16_support_32 | 0.4 | a | 11 | -1.71875 | 56.533203125 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band16_support_64|0.5 | p856_023_material_plastic | material | natural_question | band16_support_64 | 0.5 | a | 11 | -1.8125 | 55.546875 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band16_support_32|0.5 | p856_023_material_plastic | material | natural_question | band16_support_32 | 0.5 | a | 11 | -1.8125 | 55.470703125 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band32_support_64|0.5 | p856_023_material_plastic | material | natural_question | band32_support_64 | 0.5 | a | 11 | -1.84375 | 55.21484375 | True |
| glm4 | p856_023_material_plastic|natural_question|L39C638+L39C1630|flip|source_case_prompt_variant|band16_support_32|0.7 | p856_023_material_plastic | material | natural_question | band16_support_32 | 0.7 | a | 12 | -1.90625 | 54.185546875 | True |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|band32_support_64|0.3 | p856_008_animal_bird | animal | natural_question | band32_support_64 | 0.3 | a | 11 | -2.25 | 51.439453125 | True |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|zero|same_domain_holdout_case|band32_support_64|0.3 | p856_008_animal_bird | animal | natural_question | band32_support_64 | 0.3 | a | 11 | -2.25 | 51.439453125 | True |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|flip|same_domain_holdout_case|band16_support_64|0.3 | p856_008_animal_bird | animal | natural_question | band16_support_64 | 0.3 | a | 11 | -2.25 | 51.390625 | True |
| glm4 | p856_008_animal_bird|natural_question|L35C8824|zero|same_domain_holdout_case|band16_support_64|0.3 | p856_008_animal_bird | animal | natural_question | band16_support_64 | 0.3 | a | 11 | -2.25 | 51.390625 | True |
| glm4 | p856_008_animal_bird|classification|L35C8824|flip|same_domain_holdout_case|band32_support_64|0.3 | p856_008_animal_bird | animal | classification | band32_support_64 | 0.3 | a | 15 | -2.25 | 50.927734375 | True |
| glm4 | p856_008_animal_bird|classification|L35C8824|zero|same_domain_holdout_case|band32_support_64|0.3 | p856_008_animal_bird | animal | classification | band32_support_64 | 0.3 | a | 15 | -2.25 | 50.927734375 | True |
| glm4 | p856_010_animal_mammal|classification|L35C8824|flip|same_domain_holdout_case|band16_support_64|0.3 | p856_010_animal_mammal | animal | classification | band16_support_64 | 0.3 | a | 12 | -2.40625 | 49.55078125 | True |
| glm4 | p856_010_animal_mammal|classification|L35C8824|zero|same_domain_holdout_case|band16_support_64|0.3 | p856_010_animal_mammal | animal | classification | band16_support_64 | 0.3 | a | 12 | -2.40625 | 49.55078125 | True |
| glm4 | p856_010_animal_mammal|classification|L35C8824|flip|same_domain_holdout_case|band16_support_32|0.3 | p856_010_animal_mammal | animal | classification | band16_support_32 | 0.3 | a | 12 | -2.40625 | 49.421875 | True |
| glm4 | p856_010_animal_mammal|classification|L35C8824|zero|same_domain_holdout_case|band16_support_32|0.3 | p856_010_animal_mammal | animal | classification | band16_support_32 | 0.3 | a | 12 | -2.40625 | 49.421875 | True |
| glm4 | p856_010_animal_mammal|classification|L35C8824|flip|same_domain_holdout_case|band32_support_64|0.3 | p856_010_animal_mammal | animal | classification | band32_support_64 | 0.3 | a | 12 | -2.40625 | 49.4140625 | True |
| glm4 | p856_010_animal_mammal|classification|L35C8824|zero|same_domain_holdout_case|band32_support_64|0.3 | p856_010_animal_mammal | animal | classification | band32_support_64 | 0.3 | a | 12 | -2.40625 | 49.4140625 | True |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | material | natural_question | low_abs_64 | 0.8 |  . | 8 | -4.1875 | 48.8671875 | True |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | material | natural_question | low_abs_64 | 0.8 |  . | 8 | -4.1875 | 48.8671875 | True |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | material | natural_question | low_abs_64 | 0.7 |  . | 8 | -4.25 | 48.705078125 | True |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | material | natural_question | low_abs_64 | 0.7 |  . | 8 | -4.25 | 48.705078125 | True |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.6 | p856_021_material_wood | material | natural_question | low_abs_64 | 0.6 |  . | 8 | -4.25 | 48.69921875 | True |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.6 | p856_021_material_wood | material | natural_question | low_abs_64 | 0.6 |  . | 8 | -4.25 | 48.69921875 | True |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | object | natural_question | low_abs_64 | 0.8 |  . | 9 | -5.90625 | 48.57421875 | True |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | object | natural_question | low_abs_64 | 0.8 |  . | 9 | -5.90625 | 48.57421875 | True |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | object | natural_question | low_abs_64 | 0.8 |  . | 9 | -5.90625 | 48.57421875 | True |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | object | natural_question | low_abs_64 | 0.8 |  . | 9 | -5.90625 | 48.57421875 | True |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|band16_support_32|0.8 | p856_035_object_chair | object | natural_question | band16_support_32 | 0.8 |  . | 9 | -5.96875 | 48.568359375 | True |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_035_object_chair | object | natural_question | low_abs_64 | 0.7 |  . | 9 | -5.96875 | 48.568359375 | True |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | object | natural_question | low_abs_64 | 0.9 |  . | 9 | -6.03125 | 48.552734375 | True |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | object | natural_question | low_abs_64 | 0.9 |  . | 9 | -6.03125 | 48.552734375 | True |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | object | natural_question | low_abs_64 | 0.9 |  . | 9 | -6.03125 | 48.552734375 | True |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | object | natural_question | low_abs_64 | 0.9 |  . | 9 | -6.03125 | 48.552734375 | True |
