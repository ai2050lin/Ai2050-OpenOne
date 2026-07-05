# Phase 929 punctuation margin gear holdout validation

## Overall

- expected_rows_if_all_reconstructed: 1680
- overall_candidate_margin_nonnegative: 270
- overall_candidate_rows: 1470
- overall_candidate_strict_clean_candidate: 0
- overall_candidate_top1: 270
- overall_coordinate_baseline_rows: 210
- overall_improved_margin_vs_coordinate_base: 1470
- overall_margin_nonnegative: 270
- overall_new_margin_closure_vs_coordinate_base: 270
- overall_new_strict_vs_coordinate_base: 0
- overall_new_top1_vs_coordinate_base: 270
- overall_rows: 1680
- overall_strict_clean_candidate: 0
- overall_target_state_coverage_margin: 30
- overall_target_state_coverage_strict: 0
- overall_target_state_coverage_top1: 30
- overall_top1: 270
- overall_unique_cases: 3
- overall_unique_states: 30
- overall_worsened_margin_vs_coordinate_base: 0
- selected_punctuation_seeds: 30

## Evidence

- no_punctuation_period_seeds: 2
- punctuation_margin_gear_unseen_seed_positive: 1

## Top New Margin Rows

| model | state | case | seen | closure_seen | group | factor | alpha | protocol | rank | margin | base margin | delta | strict |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|band16_support_32|0.8 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 0.85 | 1 | 0.6875 | -5.96875 | 6.65625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 0.85 | 1 | 0.5625 | -6.09375 | 6.65625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 0.85 | 1 | 0.5625 | -6.09375 | 6.65625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 0.85 | 1 | 0.5625 | -6.09375 | 6.65625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 0.85 | 1 | 0.5625 | -6.09375 | 6.65625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.375 | 0.85 | 1 | 0.5625 | -6.09375 | 6.65625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.375 | 0.85 | 1 | 0.5625 | -6.09375 | 6.65625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 0.875 | 0.85 | 1 | 0.4375 | -6.1875 | 6.625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 0.875 | 0.85 | 1 | 0.4375 | -6.1875 | 6.625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 0.875 | 0.85 | 1 | 0.4375 | -6.1875 | 6.625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|band16_support_32|0.8 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.625 | -5.96875 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.625 | -5.96875 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.5625 | -6.03125 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.9 | 1 | 0.5625 | -6.03125 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.5625 | -6.03125 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.9 | 1 | 0.5625 | -6.03125 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.5625 | -6.03125 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.9 | 1 | 0.5625 | -6.03125 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.5625 | -6.03125 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.9 | 1 | 0.5625 | -6.03125 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|band16_support_32|0.8 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.9 | 1 | 0.5625 | -6.03125 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 0.85 | 1 | 0.5 | -6.09375 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.85 | 1 | 0.5 | -6.09375 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 0.85 | 1 | 0.5 | -6.09375 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.85 | 1 | 0.5 | -6.09375 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 0.85 | 1 | 0.5 | -6.09375 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.85 | 1 | 0.5 | -6.09375 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 0.85 | 1 | 0.5 | -6.09375 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.85 | 1 | 0.5 | -6.09375 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.9 | 1 | 0.4375 | -6.15625 | 6.59375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|band16_support_32|0.8 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.8125 | -5.75 | 6.5625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|band16_support_32|0.8 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.85 | 1 | 0.625 | -5.9375 | 6.5625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.375 | 0.9 | 1 | 0.5 | -6.0625 | 6.5625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.375 | 0.9 | 1 | 0.5 | -6.0625 | 6.5625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 0.85 | 1 | 0.4375 | -6.125 | 6.5625 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.75 | -5.78125 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.75 | -5.78125 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.75 | -5.78125 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.75 | -5.78125 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.6875 | -5.84375 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.6875 | -5.84375 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.6875 | -5.84375 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|zero|same_domain_holdout_case|low_abs_64|0.9 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.6875 | -5.84375 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.625 | -5.90625 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.375 | 0.85 | 1 | 0.5 | -6.03125 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.25 | 0.85 | 1 | 0.5 | -6.03125 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.25 | 0.85 | 1 | 0.5 | -6.03125 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 0.85 | 1 | 0.4375 | -6.09375 | 6.53125 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.6875 | -5.8125 | 6.5 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.6875 | -5.8125 | 6.5 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.6875 | -5.8125 | 6.5 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.625 | -5.875 | 6.5 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|band16_support_32|0.8 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 0.875 | 0.85 | 1 | 0.5 | -6.0 | 6.5 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.5625 | -5.90625 | 6.46875 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.5625 | -5.90625 | 6.46875 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.5625 | -5.90625 | 6.46875 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|band16_support_32|0.8 | p856_035_object_chair | False | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.625 | -5.8125 | 6.4375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.625 | -5.75 | 6.375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C11316+L39C5585|zero|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.625 | -5.75 | 6.375 | False |
| glm4 | p856_035_object_chair|natural_question|L39C3652+L39C11316|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_035_object_chair | True | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.625 | -5.75 | 6.375 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.6 | p856_021_material_wood | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.5625 | -4.09375 | 4.65625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.6 | p856_021_material_wood | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.5625 | -4.09375 | 4.65625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.5625 | -4.0625 | 4.625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.5625 | -4.0625 | 4.625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.4375 | -4.1875 | 4.625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.4375 | -4.1875 | 4.625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.5 | -4.09375 | 4.59375 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.5 | -4.09375 | 4.59375 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.4375 | -4.15625 | 4.59375 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.4375 | -4.15625 | 4.59375 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.6 | p856_021_material_wood | False | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.375 | -4.21875 | 4.59375 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.6 | p856_021_material_wood | False | False | margin_support_pos_64 | 2.25 | 1.25 | 1.1 | 1 | 0.375 | -4.21875 | 4.59375 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.375 | -4.1875 | 4.5625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.8 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.375 | -4.1875 | 4.5625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.6 | p856_021_material_wood | False | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.3125 | -4.25 | 4.5625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.6 | p856_021_material_wood | False | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.3125 | -4.25 | 4.5625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.3125 | -4.25 | 4.5625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|low_abs_64|0.7 | p856_021_material_wood | True | False | margin_support_pos_64 | 2.25 | 1.0 | 1.0 | 1 | 0.3125 | -4.25 | 4.5625 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C1630|flip|same_domain_holdout_case|top_abs_64|0.4 | p856_021_material_wood | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.4375 | -4.0625 | 4.5 | False |
| glm4 | p856_021_material_wood|natural_question|L39C638+L39C2682|flip|same_domain_holdout_case|top_abs_64|0.4 | p856_021_material_wood | False | False | margin_support_pos_64 | 2.25 | 0.875 | 1.1 | 1 | 0.4375 | -4.0625 | 4.5 | False |
