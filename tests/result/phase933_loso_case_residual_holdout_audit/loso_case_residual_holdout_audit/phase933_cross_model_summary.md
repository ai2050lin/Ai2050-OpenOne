# Phase 933 LOSO case residual holdout audit

## Overall

- expected_rows_if_all_reconstructed: 2730
- overall_candidate_margin_nonnegative: 1183
- overall_candidate_rows: 2520
- overall_candidate_strict_clean_candidate: 0
- overall_candidate_top1: 1183
- overall_coordinate_baseline_rows: 210
- overall_improved_margin_vs_coordinate_base: 2498
- overall_margin_nonnegative: 1183
- overall_new_margin_closure_vs_coordinate_base: 1183
- overall_new_strict_vs_coordinate_base: 0
- overall_new_top1_vs_coordinate_base: 1183
- overall_rows: 2730
- overall_strict_clean_candidate: 0
- overall_target_state_coverage_margin: 30
- overall_target_state_coverage_strict: 0
- overall_target_state_coverage_top1: 30
- overall_top1: 1183
- overall_unique_cases: 3
- overall_unique_states: 30
- overall_worsened_margin_vs_coordinate_base: 0
- selected_punctuation_seeds: 30

## Evidence

- loso_case_residual_matches_state_specific_without_strict_clean: 1
- no_punctuation_period_seeds: 2

## LOSO Residual Inventory

### deepseek7b


### glm4

- p856_021_material_wood: states=10, holdout_each=9, inter_residual_size=2..2, union_residual_size=9..9
- p856_035_object_chair: states=10, holdout_each=9, inter_residual_size=17..17, union_residual_size=19..19
- p885_047_animal_shark: states=10, holdout_each=9, inter_residual_size=13..13, union_residual_size=13..13

### qwen3


## Top Repair Coverage

| model | group | factor | all states |
| --- | --- | ---: | ---: |
| glm4 | fixed_plus_loso_case_inter_residual | 2.25 | 30 |
| glm4 | fixed_plus_loso_case_union_residual | 2.25 | 30 |
| glm4 | state_specific_margin_support_pos_64 | 2.25 | 30 |
| glm4 | fixed_plus_loso_case_union_residual | 2.1 | 30 |
| glm4 | fixed_plus_loso_case_inter_residual | 2.1 | 22 |
| glm4 | state_specific_margin_support_pos_64 | 2.1 | 22 |
| glm4 | fixed_topfreq_64 | 2.25 | 20 |
| glm4 | fixed_topfreq_64 | 2.1 | 10 |
| glm4 | loso_case_inter_residual_only | 2.25 | 0 |
| glm4 | loso_case_union_residual_only | 2.25 | 0 |
| glm4 | loso_case_inter_residual_only | 2.1 | 0 |
| glm4 | loso_case_union_residual_only | 2.1 | 0 |

## Top Group Factor Rows

| model | group | factor | rows | top1 | margin | strict | new top1 | states | mean delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | fixed_plus_loso_case_union_residual | 2.25 | 210 | 210 | 210 | 0 | 210 | 30 | 5.633035714285715 |
| glm4 | fixed_plus_loso_case_inter_residual | 2.25 | 210 | 202 | 202 | 0 | 202 | 30 | 5.5529761904761905 |
| glm4 | state_specific_margin_support_pos_64 | 2.25 | 210 | 200 | 200 | 0 | 200 | 30 | 5.430654761904762 |
| glm4 | fixed_plus_loso_case_union_residual | 2.1 | 210 | 158 | 158 | 0 | 158 | 30 | 5.021130952380952 |
| glm4 | fixed_plus_loso_case_inter_residual | 2.1 | 210 | 123 | 123 | 0 | 123 | 22 | 4.955357142857143 |
| glm4 | state_specific_margin_support_pos_64 | 2.1 | 210 | 98 | 98 | 0 | 98 | 22 | 4.845238095238095 |
| glm4 | fixed_topfreq_64 | 2.25 | 210 | 122 | 122 | 0 | 122 | 20 | 4.613988095238096 |
| glm4 | fixed_topfreq_64 | 2.1 | 210 | 70 | 70 | 0 | 70 | 10 | 4.045535714285714 |
| glm4 | loso_case_union_residual_only | 2.25 | 210 | 0 | 0 | 0 | 0 | 0 | 1.021279761904762 |
| glm4 | loso_case_inter_residual_only | 2.25 | 210 | 0 | 0 | 0 | 0 | 0 | 0.959375 |
| glm4 | loso_case_union_residual_only | 2.1 | 210 | 0 | 0 | 0 | 0 | 0 | 0.8964285714285715 |
| glm4 | loso_case_inter_residual_only | 2.1 | 210 | 0 | 0 | 0 | 0 | 0 | 0.8482142857142857 |
| glm4 | coordinate_only | 1.0 | 210 | 0 | 0 | 0 | 0 | 0 | None |
