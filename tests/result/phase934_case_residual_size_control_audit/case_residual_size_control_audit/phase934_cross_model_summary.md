# Phase 934 case residual size-control audit

## Overall

- expected_rows_if_all_reconstructed: 4410
- overall_candidate_margin_nonnegative: 2317
- overall_candidate_rows: 4200
- overall_candidate_strict_clean_candidate: 0
- overall_candidate_top1: 2317
- overall_coordinate_baseline_rows: 210
- overall_improved_margin_vs_coordinate_base: 4200
- overall_margin_nonnegative: 2317
- overall_new_margin_closure_vs_coordinate_base: 2317
- overall_new_strict_vs_coordinate_base: 0
- overall_new_top1_vs_coordinate_base: 2317
- overall_rows: 4410
- overall_strict_clean_candidate: 0
- overall_target_state_coverage_margin: 30
- overall_target_state_coverage_strict: 0
- overall_target_state_coverage_top1: 30
- overall_top1: 2317
- overall_unique_cases: 3
- overall_unique_states: 30
- overall_worsened_margin_vs_coordinate_base: 0
- selected_punctuation_seeds: 30

## Evidence

- case_residual_beats_size_controls_without_strict_clean: 1
- no_punctuation_period_seeds: 2

## Control Inventory

### deepseek7b

- residual_pool_size: 0

### glm4

- residual_pool_size: 41
- p856_021_material_wood: states=10, loso_inter_sizes=[2], loso_union_sizes=[9], noncase_inter_control_sizes=[2], noncase_union_control_sizes=[9]
- p856_035_object_chair: states=10, loso_inter_sizes=[17], loso_union_sizes=[19], noncase_inter_control_sizes=[17], noncase_union_control_sizes=[19]
- p885_047_animal_shark: states=10, loso_inter_sizes=[13], loso_union_sizes=[13], noncase_inter_control_sizes=[13], noncase_union_control_sizes=[13]

### qwen3

- residual_pool_size: 0

## Top Size-Control Coverage

| model | group | factor | all states |
| --- | --- | ---: | ---: |
| glm4 | fixed_plus_loso_case_inter_residual | 2.25 | 30 |
| glm4 | fixed_plus_loso_case_union_residual | 2.25 | 30 |
| glm4 | state_specific_margin_support_pos_64 | 2.25 | 30 |
| glm4 | fixed_plus_loso_case_union_residual | 2.1 | 30 |
| glm4 | fixed_plus_loso_case_inter_residual | 2.1 | 22 |
| glm4 | state_specific_margin_support_pos_64 | 2.1 | 22 |
| glm4 | fixed_topfreq_64 | 2.25 | 20 |
| glm4 | fixed_plus_noncase_inter_size_control | 2.25 | 20 |
| glm4 | fixed_plus_global_inter_size_control | 2.25 | 20 |
| glm4 | fixed_plus_pseudorandom_inter_size_control | 2.25 | 20 |
| glm4 | fixed_plus_noncase_union_size_control | 2.25 | 20 |
| glm4 | fixed_plus_global_union_size_control | 2.25 | 20 |
| glm4 | fixed_plus_pseudorandom_union_size_control | 2.25 | 19 |
| glm4 | fixed_plus_pseudorandom_union_size_control | 2.1 | 14 |
| glm4 | fixed_plus_pseudorandom_inter_size_control | 2.1 | 11 |
| glm4 | fixed_topfreq_64 | 2.1 | 10 |
| glm4 | fixed_plus_noncase_inter_size_control | 2.1 | 10 |
| glm4 | fixed_plus_global_inter_size_control | 2.1 | 10 |
| glm4 | fixed_plus_noncase_union_size_control | 2.1 | 10 |
| glm4 | fixed_plus_global_union_size_control | 2.1 | 10 |

## Top Group Factor Rows

| model | group | factor | rows | top1 | margin | strict | new top1 | states | mean delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | fixed_plus_loso_case_union_residual | 2.25 | 210 | 210 | 210 | 0 | 210 | 30 | 5.633035714285715 |
| glm4 | fixed_plus_loso_case_inter_residual | 2.25 | 210 | 202 | 202 | 0 | 202 | 30 | 5.5529761904761905 |
| glm4 | state_specific_margin_support_pos_64 | 2.25 | 210 | 200 | 200 | 0 | 200 | 30 | 5.430654761904762 |
| glm4 | fixed_plus_loso_case_union_residual | 2.1 | 210 | 158 | 158 | 0 | 158 | 30 | 5.021130952380952 |
| glm4 | fixed_plus_loso_case_inter_residual | 2.1 | 210 | 123 | 123 | 0 | 123 | 22 | 4.955357142857143 |
| glm4 | state_specific_margin_support_pos_64 | 2.1 | 210 | 98 | 98 | 0 | 98 | 22 | 4.845238095238095 |
| glm4 | fixed_plus_pseudorandom_inter_size_control | 2.25 | 210 | 122 | 122 | 0 | 122 | 20 | 4.650297619047619 |
| glm4 | fixed_topfreq_64 | 2.25 | 210 | 122 | 122 | 0 | 122 | 20 | 4.613988095238096 |
| glm4 | fixed_plus_noncase_inter_size_control | 2.25 | 210 | 120 | 120 | 0 | 120 | 20 | 4.680357142857143 |
| glm4 | fixed_plus_global_inter_size_control | 2.25 | 210 | 120 | 120 | 0 | 120 | 20 | 4.680357142857143 |
| glm4 | fixed_plus_noncase_union_size_control | 2.25 | 210 | 116 | 116 | 0 | 116 | 20 | 4.69077380952381 |
| glm4 | fixed_plus_global_union_size_control | 2.25 | 210 | 116 | 116 | 0 | 116 | 20 | 4.69077380952381 |
| glm4 | fixed_plus_pseudorandom_union_size_control | 2.25 | 210 | 118 | 118 | 0 | 118 | 19 | 4.656845238095238 |
| glm4 | fixed_plus_pseudorandom_union_size_control | 2.1 | 210 | 75 | 75 | 0 | 75 | 14 | 4.087202380952381 |
| glm4 | fixed_plus_pseudorandom_inter_size_control | 2.1 | 210 | 67 | 67 | 0 | 67 | 11 | 4.08125 |
| glm4 | fixed_plus_noncase_union_size_control | 2.1 | 210 | 70 | 70 | 0 | 70 | 10 | 4.120535714285714 |
| glm4 | fixed_plus_global_union_size_control | 2.1 | 210 | 70 | 70 | 0 | 70 | 10 | 4.120535714285714 |
| glm4 | fixed_plus_noncase_inter_size_control | 2.1 | 210 | 70 | 70 | 0 | 70 | 10 | 4.111011904761905 |
| glm4 | fixed_plus_global_inter_size_control | 2.1 | 210 | 70 | 70 | 0 | 70 | 10 | 4.111011904761905 |
| glm4 | fixed_topfreq_64 | 2.1 | 210 | 70 | 70 | 0 | 70 | 10 | 4.045535714285714 |
| glm4 | coordinate_only | 1.0 | 210 | 0 | 0 | 0 | 0 | 0 | None |
