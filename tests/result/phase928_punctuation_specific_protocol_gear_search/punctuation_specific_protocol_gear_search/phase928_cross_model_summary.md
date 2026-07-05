# Phase 928 punctuation-specific protocol gear search

## Overall

- models: qwen3, glm4, deepseek7b
- expected_rows_if_all_reconstructed: 2268
- overall_candidate_margin_nonnegative: 28
- overall_candidate_rows: 2184
- overall_candidate_strict_clean_candidate: 0
- overall_candidate_top1: 28
- overall_coordinate_baseline_rows: 84
- overall_improved_margin_vs_coordinate_base: 1411
- overall_margin_nonnegative: 28
- overall_new_margin_closure_vs_coordinate_base: 28
- overall_new_strict_vs_coordinate_base: 0
- overall_new_top1_vs_coordinate_base: 28
- overall_rows: 2268
- overall_strict_clean_candidate: 0
- overall_target_state_coverage_margin: 4
- overall_target_state_coverage_strict: 0
- overall_target_state_coverage_top1: 4
- overall_top1: 28
- overall_unique_cases: 3
- overall_unique_states: 12
- overall_worsened_margin_vs_coordinate_base: 240
- selected_punctuation_seeds: 12

## Evidence

- no_punctuation_period_seeds: 2
- punctuation_specific_closure_candidate_found: 1

## Top Group Factors

| model | group | factor | rows | top1 | margin | strict | improved | new margin | new top1 | mean delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | margin_support_pos_64 | 2.0 | 84 | 28 | 28 | 0 | 84 | 28 | 28 | 4.410714285714286 |
| glm4 | eos_support_64 | 2.0 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 2.9367559523809526 |
| glm4 | a_blocker_support_64 | 0.25 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 2.5922619047619047 |
| glm4 | margin_support_neg_64 | 0.25 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 2.5922619047619047 |
| glm4 | margin_support_pos_64 | 1.5 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 2.0811011904761907 |
| glm4 | a_blocker_support_64 | 0.5 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 1.671875 |
| glm4 | margin_support_neg_64 | 0.5 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 1.671875 |
| glm4 | a_logit_support_64 | 0.25 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 1.5461309523809523 |
| glm4 | eos_support_64 | 1.5 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 1.3139880952380953 |
| glm4 | a_logit_support_64 | 0.5 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 1.0625 |
| glm4 | margin_support_pos_64 | 1.25 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 0.9709821428571429 |
| glm4 | a_blocker_support_64 | 0.75 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 0.7849702380952381 |
| glm4 | margin_support_neg_64 | 0.75 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 0.7849702380952381 |
| glm4 | top_abs_64 | 1.5 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 0.6078869047619048 |
| glm4 | eos_support_64 | 1.25 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 0.5870535714285714 |
| glm4 | a_logit_support_64 | 0.75 | 84 | 0 | 0 | 0 | 84 | 0 | 0 | 0.5338541666666666 |
| glm4 | band_blocker_support_64 | 0.5 | 84 | 0 | 0 | 0 | 28 | 0 | 0 | -0.09598214285714286 |
| glm4 | band_blocker_support_64 | 0.25 | 84 | 0 | 0 | 0 | 28 | 0 | 0 | -0.1130952380952381 |
| glm4 | band_blocker_support_64 | 0.75 | 84 | 0 | 0 | 0 | 11 | 0 | 0 | -0.05952380952380952 |
| glm4 | top_abs_64 | 0.5 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | -0.49516369047619047 |
| glm4 | coordinate_only | 1.0 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | None |
| glm4 | a_blocker_support_64 | 1.0 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| glm4 | a_logit_support_64 | 1.0 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| glm4 | margin_support_neg_64 | 1.0 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| glm4 | band_blocker_support_64 | 1.0 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| glm4 | low_abs_64 | 0.5 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| glm4 | low_abs_64 | 1.5 | 84 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |

## Top Candidate Rows

| model | state | case | group | factor | alpha | protocol | rank | margin | base margin | delta | top1 | strict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 1 | 0.5 | -3.8125 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 1 | 0.3125 | -4.0 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 1 | 0.75 | -3.5625 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 1 | 0.5 | -3.8125 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 1 | 0.3125 | -4.0 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 1 | 0.75 | -3.5625 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 1 | 0.1875 | -4.125 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 1 | 0.3125 | -4.0 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 1 | 0.25 | -4.0625 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 1 | 0.6875 | -3.625 | 4.3125 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 1 | 0.3125 | -3.9375 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 1 | 0.625 | -3.625 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 1 | 0.625 | -3.625 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 1 | 0.5625 | -3.6875 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 1 | 0.3125 | -3.9375 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 1 | 0.625 | -3.625 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 1 | 0.625 | -3.625 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 1 | 0.5625 | -3.6875 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 1 | 0.4375 | -3.8125 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 1 | 0.25 | -4.0 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 1 | 0.625 | -3.625 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 1 | 0.625 | -3.625 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 1 | 0.6875 | -3.5625 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 1 | 0.4375 | -3.8125 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 1 | 0.625 | -3.625 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 1 | 0.5 | -3.75 | 4.25 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 1 | 0.5625 | -3.625 | 4.1875 | True | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 1 | 0.5625 | -3.625 | 4.1875 | True | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 3 | -0.5625 | -6.0625 | 5.5 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 3 | -0.5625 | -6.0625 | 5.5 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 3 | -0.5625 | -6.0625 | 5.5 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 3 | -0.5625 | -6.0625 | 5.5 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 3 | -0.625 | -6.09375 | 5.46875 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 3 | -0.625 | -6.09375 | 5.46875 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 3 | -0.625 | -6.09375 | 5.46875 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 3 | -0.625 | -6.09375 | 5.46875 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 3 | -0.5 | -5.90625 | 5.40625 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 3 | -0.5 | -5.90625 | 5.40625 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 3 | -0.5 | -5.90625 | 5.40625 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 3 | -0.5 | -5.90625 | 5.40625 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 3 | -0.375 | -5.75 | 5.375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 3 | -0.8125 | -6.1875 | 5.375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 3 | -0.375 | -5.75 | 5.375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 3 | -0.8125 | -6.1875 | 5.375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 3 | -0.375 | -5.75 | 5.375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 3 | -0.8125 | -6.1875 | 5.375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 3 | -0.375 | -5.75 | 5.375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 3 | -0.8125 | -6.1875 | 5.375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 3 | -0.6875 | -6.03125 | 5.34375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 3 | -0.6875 | -6.03125 | 5.34375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 3 | -0.6875 | -6.03125 | 5.34375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 3 | -0.6875 | -6.03125 | 5.34375 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 3 | -0.5 | -5.8125 | 5.3125 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 3 | -0.5 | -5.8125 | 5.3125 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 3 | -0.5 | -5.8125 | 5.3125 | False | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 3 | -0.5 | -5.8125 | 5.3125 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 2 | -0.5 | -4.1875 | 3.6875 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 2 | -0.5 | -4.1875 | 3.6875 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 2 | -0.4375 | -4.09375 | 3.65625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 2 | -0.5 | -4.15625 | 3.65625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 2 | -0.4375 | -4.09375 | 3.65625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.25 | 1.1 | 2 | -0.5 | -4.15625 | 3.65625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 2 | -0.5625 | -4.1875 | 3.625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 2 | -0.5625 | -4.1875 | 3.625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 2 | -0.625 | -4.25 | 3.625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 2 | -0.4375 | -4.0625 | 3.625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.0 | 1.0 | 2 | -0.625 | -4.25 | 3.625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 0.875 | 1.1 | 2 | -0.4375 | -4.0625 | 3.625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 2 | -0.8125 | -4.375 | 3.5625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 2 | -0.8125 | -4.375 | 3.5625 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 2 | -0.875 | -4.375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 2 | -0.875 | -4.375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 2 | -0.9375 | -4.4375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 2 | -0.875 | -4.375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 2 | -0.875 | -4.375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 2 | -0.9375 | -4.4375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 2 | -0.75 | -4.25 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 2 | -0.875 | -4.375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 2 | -0.9375 | -4.4375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 2 | -0.875 | -4.375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 0.875 | 0.85 | 2 | -0.75 | -4.25 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.25 | 0.85 | 2 | -0.875 | -4.375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.375 | 0.85 | 2 | -0.9375 | -4.4375 | 3.5 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | margin_support_pos_64 | 2.0 | 1.375 | 0.9 | 2 | -0.875 | -4.375 | 3.5 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 0.875 | 1.1 | 2 | -0.6875 | -4.0 | 3.3125 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.375 | 0.9 | 2 | -0.375 | -3.6875 | 3.3125 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 0.875 | 1.1 | 2 | -0.6875 | -4.0 | 3.3125 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.375 | 0.9 | 2 | -0.375 | -3.6875 | 3.3125 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.25 | 1.1 | 2 | -0.8125 | -4.125 | 3.3125 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.25 | 1.1 | 2 | -0.75 | -4.0625 | 3.3125 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.375 | 0.9 | 2 | -0.4375 | -3.75 | 3.3125 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.0 | 1.0 | 2 | -0.5625 | -3.8125 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.25 | 1.1 | 2 | -0.6875 | -3.9375 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 0.875 | 0.85 | 2 | -0.3125 | -3.5625 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.25 | 0.85 | 2 | -0.375 | -3.625 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.375 | 0.85 | 2 | -0.375 | -3.625 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.0 | 1.0 | 2 | -0.5625 | -3.8125 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.25 | 1.1 | 2 | -0.6875 | -3.9375 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 0.875 | 0.85 | 2 | -0.3125 | -3.5625 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.25 | 0.85 | 2 | -0.375 | -3.625 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.375 | 0.85 | 2 | -0.375 | -3.625 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | eos_support_64 | 2.0 | 0.875 | 1.1 | 2 | -0.75 | -4.0 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.25 | 0.85 | 2 | -0.375 | -3.625 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | eos_support_64 | 2.0 | 0.875 | 1.1 | 2 | -0.75 | -4.0 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | eos_support_64 | 2.0 | 0.875 | 0.85 | 2 | -0.375 | -3.625 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.375 | 0.85 | 2 | -0.375 | -3.625 | 3.25 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.0 | 1.0 | 2 | -0.625 | -3.8125 | 3.1875 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | eos_support_64 | 2.0 | 0.875 | 0.85 | 2 | -0.4375 | -3.625 | 3.1875 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.375 | 0.85 | 2 | -0.375 | -3.5625 | 3.1875 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.375 | 0.9 | 2 | -0.4375 | -3.625 | 3.1875 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.0 | 1.0 | 2 | -0.625 | -3.8125 | 3.1875 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | eos_support_64 | 2.0 | 1.25 | 0.85 | 2 | -0.4375 | -3.625 | 3.1875 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.375 | 0.85 | 4 | -0.5625 | -3.625 | 3.0625 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.375 | 0.85 | 4 | -0.5625 | -3.625 | 3.0625 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.375 | 0.85 | 4 | -0.5625 | -3.625 | 3.0625 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.375 | 0.85 | 4 | -0.5625 | -3.625 | 3.0625 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.0 | 1.0 | 4 | -0.8125 | -3.8125 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.0 | 1.0 | 4 | -0.8125 | -3.8125 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.375 | 0.9 | 4 | -0.6875 | -3.6875 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.375 | 0.9 | 4 | -0.6875 | -3.6875 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.0 | 1.0 | 4 | -0.8125 | -3.8125 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.0 | 1.0 | 4 | -0.8125 | -3.8125 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.375 | 0.9 | 4 | -0.6875 | -3.6875 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.375 | 0.9 | 4 | -0.6875 | -3.6875 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.375 | 0.85 | 4 | -0.625 | -3.625 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.375 | 0.85 | 4 | -0.625 | -3.625 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.375 | 0.9 | 4 | -0.75 | -3.75 | 3.0 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.375 | 0.9 | 4 | -0.75 | -3.75 | 3.0 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 0.875 | 0.85 | 2 | -1.375 | -4.375 | 3.0 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 0.875 | 0.85 | 2 | -1.375 | -4.375 | 3.0 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | eos_support_64 | 2.0 | 0.875 | 1.1 | 2 | -1.0625 | -4.0625 | 3.0 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | eos_support_64 | 2.0 | 1.25 | 1.1 | 2 | -1.1875 | -4.1875 | 3.0 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | eos_support_64 | 2.0 | 1.25 | 0.85 | 2 | -1.375 | -4.375 | 3.0 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | eos_support_64 | 2.0 | 0.875 | 1.1 | 2 | -1.0625 | -4.0625 | 3.0 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | eos_support_64 | 2.0 | 1.25 | 1.1 | 2 | -1.1875 | -4.1875 | 3.0 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | eos_support_64 | 2.0 | 1.25 | 0.85 | 2 | -1.375 | -4.375 | 3.0 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 0.875 | 1.1 | 2 | -1.125 | -4.09375 | 2.96875 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 1.25 | 1.1 | 2 | -1.1875 | -4.15625 | 2.96875 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 0.875 | 1.1 | 2 | -1.125 | -4.09375 | 2.96875 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 1.25 | 1.1 | 2 | -1.1875 | -4.15625 | 2.96875 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.25 | 0.85 | 4 | -0.6875 | -3.625 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.25 | 0.85 | 4 | -0.6875 | -3.625 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.25 | 0.85 | 4 | -0.6875 | -3.625 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|flip|source_case_prompt_variant|band16_support_32|0.9 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.25 | 0.85 | 4 | -0.6875 | -3.625 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.0 | 1.0 | 4 | -0.875 | -3.8125 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.0 | 1.0 | 4 | -0.875 | -3.8125 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.25 | 1.1 | 5 | -1.1875 | -4.125 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.25 | 1.1 | 5 | -1.1875 | -4.125 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.375 | 0.9 | 4 | -0.6875 | -3.625 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.6 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.375 | 0.9 | 4 | -0.6875 | -3.625 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.0 | 1.0 | 4 | -0.875 | -3.8125 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.0 | 1.0 | 4 | -0.875 | -3.8125 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.25 | 1.1 | 4 | -1.125 | -4.0625 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.25 | 1.1 | 4 | -1.125 | -4.0625 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | a_blocker_support_64 | 0.25 | 1.25 | 0.85 | 4 | -0.6875 | -3.625 | 2.9375 | False | False |
| glm4 | p885_047_animal_shark|natural_category|L35C8824|zero|source_case_prompt_variant|low_abs_64|0.5 | p885_047_animal_shark | margin_support_neg_64 | 0.25 | 1.25 | 0.85 | 4 | -0.6875 | -3.625 | 2.9375 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 1.0 | 1.0 | 2 | -1.25 | -4.1875 | 2.9375 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 1.25 | 0.85 | 2 | -1.4375 | -4.375 | 2.9375 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 1.375 | 0.85 | 2 | -1.4375 | -4.375 | 2.9375 | False | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | eos_support_64 | 2.0 | 1.375 | 0.9 | 2 | -1.5 | -4.4375 | 2.9375 | False | False |
