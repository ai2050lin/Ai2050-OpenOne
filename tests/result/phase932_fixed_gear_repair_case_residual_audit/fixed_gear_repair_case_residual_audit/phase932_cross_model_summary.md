# Phase 932 fixed gear repair and case-specific residual audit

## Overall

- expected_rows_if_all_reconstructed: 4410
- overall_candidate_margin_nonnegative: 1698
- overall_candidate_rows: 4200
- overall_candidate_strict_clean_candidate: 0
- overall_candidate_top1: 1698
- overall_coordinate_baseline_rows: 210
- overall_improved_margin_vs_coordinate_base: 3618
- overall_margin_nonnegative: 1698
- overall_new_margin_closure_vs_coordinate_base: 1698
- overall_new_strict_vs_coordinate_base: 0
- overall_new_top1_vs_coordinate_base: 1698
- overall_rows: 4410
- overall_strict_clean_candidate: 0
- overall_target_state_coverage_margin: 30
- overall_target_state_coverage_strict: 0
- overall_target_state_coverage_top1: 30
- overall_top1: 1698
- overall_unique_cases: 3
- overall_unique_states: 30
- overall_worsened_margin_vs_coordinate_base: 560
- selected_punctuation_seeds: 30

## Evidence

- case_residual_repair_matches_state_specific_without_strict_clean: 1
- no_punctuation_period_seeds: 2

## Case Residual Inventory

### deepseek7b


### glm4

- p856_021_material_wood: states=10, intersection=58, union=69, inter_residual=2, union_residual=9
- p856_035_object_chair: states=10, intersection=63, union=65, inter_residual=17, union_residual=19
- p885_047_animal_shark: states=10, intersection=64, union=64, inter_residual=13, union_residual=13

### qwen3


## Top Repair Coverage

| model | group | factor | all states |
| --- | --- | ---: | ---: |
| glm4 | fixed_plus_case_inter_residual | 2.25 | 30 |
| glm4 | fixed_plus_case_union_residual | 2.25 | 30 |
| glm4 | state_specific_margin_support_pos_64 | 2.25 | 30 |
| glm4 | fixed_plus_case_union_residual | 2.1 | 30 |
| glm4 | fixed_plus_case_inter_residual | 2.1 | 22 |
| glm4 | state_specific_margin_support_pos_64 | 2.1 | 22 |
| glm4 | fixed_topfreq_64 | 2.25 | 20 |
| glm4 | fixed_plus_chair_inter_residual | 2.25 | 20 |
| glm4 | fixed_plus_chair_union_residual | 2.25 | 20 |
| glm4 | fixed_plus_chair_inter_residual | 2.1 | 20 |
| glm4 | fixed_plus_chair_union_residual | 2.1 | 20 |
| glm4 | fixed_topfreq_64 | 2.1 | 10 |
| glm4 | chair_inter_residual_only | 2.25 | 0 |
| glm4 | chair_union_residual_only | 2.25 | 0 |
| glm4 | case_inter_residual_only | 2.25 | 0 |
| glm4 | case_union_residual_only | 2.25 | 0 |
| glm4 | chair_inter_residual_only | 2.1 | 0 |
| glm4 | chair_union_residual_only | 2.1 | 0 |
| glm4 | case_inter_residual_only | 2.1 | 0 |
| glm4 | case_union_residual_only | 2.1 | 0 |

## Top Group Factor Rows

| model | group | factor | rows | top1 | margin | strict | new top1 | states | mean delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | fixed_plus_case_union_residual | 2.25 | 210 | 210 | 210 | 0 | 210 | 30 | 5.633035714285715 |
| glm4 | fixed_plus_case_inter_residual | 2.25 | 210 | 202 | 202 | 0 | 202 | 30 | 5.5529761904761905 |
| glm4 | state_specific_margin_support_pos_64 | 2.25 | 210 | 200 | 200 | 0 | 200 | 30 | 5.430654761904762 |
| glm4 | fixed_plus_case_union_residual | 2.1 | 210 | 158 | 158 | 0 | 158 | 30 | 5.021130952380952 |
| glm4 | fixed_plus_case_inter_residual | 2.1 | 210 | 123 | 123 | 0 | 123 | 22 | 4.955357142857143 |
| glm4 | state_specific_margin_support_pos_64 | 2.1 | 210 | 98 | 98 | 0 | 98 | 22 | 4.845238095238095 |
| glm4 | fixed_plus_chair_union_residual | 2.25 | 210 | 140 | 140 | 0 | 140 | 20 | 5.0217261904761905 |
| glm4 | fixed_plus_chair_inter_residual | 2.25 | 210 | 140 | 140 | 0 | 140 | 20 | 4.991666666666666 |
| glm4 | fixed_plus_chair_union_residual | 2.1 | 210 | 126 | 126 | 0 | 126 | 20 | 4.472321428571429 |
| glm4 | fixed_topfreq_64 | 2.25 | 210 | 122 | 122 | 0 | 122 | 20 | 4.613988095238096 |
| glm4 | fixed_plus_chair_inter_residual | 2.1 | 210 | 109 | 109 | 0 | 109 | 20 | 4.444047619047619 |
| glm4 | fixed_topfreq_64 | 2.1 | 210 | 70 | 70 | 0 | 70 | 10 | 4.045535714285714 |
| glm4 | case_union_residual_only | 2.25 | 210 | 0 | 0 | 0 | 0 | 0 | 1.021279761904762 |
| glm4 | case_inter_residual_only | 2.25 | 210 | 0 | 0 | 0 | 0 | 0 | 0.959375 |
| glm4 | case_union_residual_only | 2.1 | 210 | 0 | 0 | 0 | 0 | 0 | 0.8964285714285715 |
| glm4 | case_inter_residual_only | 2.1 | 210 | 0 | 0 | 0 | 0 | 0 | 0.8482142857142857 |
| glm4 | chair_union_residual_only | 2.25 | 210 | 0 | 0 | 0 | 0 | 0 | 0.42336309523809523 |
| glm4 | chair_inter_residual_only | 2.25 | 210 | 0 | 0 | 0 | 0 | 0 | 0.41235119047619045 |
| glm4 | chair_union_residual_only | 2.1 | 210 | 0 | 0 | 0 | 0 | 0 | 0.37410714285714286 |
| glm4 | chair_inter_residual_only | 2.1 | 210 | 0 | 0 | 0 | 0 | 0 | 0.36041666666666666 |
| glm4 | coordinate_only | 1.0 | 210 | 0 | 0 | 0 | 0 | 0 | None |
