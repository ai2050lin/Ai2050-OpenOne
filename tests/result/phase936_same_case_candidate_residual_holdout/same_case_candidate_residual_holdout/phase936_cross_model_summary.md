# Phase 936 same-case candidate residual holdout

## Overall

- expected_rows_if_all_reconstructed: 10710
- overall_candidate_margin_nonnegative: 5840
- overall_candidate_rows: 10080
- overall_candidate_strict_clean_candidate: 0
- overall_candidate_top1: 5840
- overall_coordinate_baseline_rows: 630
- overall_improved_margin_vs_coordinate_base: 10080
- overall_margin_nonnegative: 5840
- overall_new_margin_closure_vs_coordinate_base: 5840
- overall_new_strict_vs_coordinate_base: 0
- overall_new_top1_vs_coordinate_base: 5840
- overall_rows: 10710
- overall_strict_clean_candidate: 0
- overall_target_state_coverage_margin: 90
- overall_target_state_coverage_strict: 0
- overall_target_state_coverage_top1: 90
- overall_top1: 5840
- overall_unique_cases: 3
- overall_unique_states: 90
- overall_worsened_margin_vs_coordinate_base: 0
- selected_holdout_seeds: 90

## Evidence

- no_punctuation_period_holdout_seeds: 2
- same_case_only_trained_case_residual_transfer_positive: 1

## Availability

### deepseek7b

- candidate_punctuation_cases: {}
- candidate_punctuation_rows: 0
- case_ids: []
- deduped_unseen_punctuation_states: 0
- new_case_available: False
- phase925_selected_punctuation_keys: 0
- selected_holdout_cases: {}
- selected_holdout_states: 0

### glm4

- candidate_punctuation_cases: {'p856_021_material_wood': 70, 'p885_047_animal_shark': 70, 'p856_035_object_chair': 124}
- candidate_punctuation_rows: 264
- case_ids: ['p856_021_material_wood', 'p856_035_object_chair', 'p885_047_animal_shark']
- deduped_unseen_punctuation_states: 234
- new_case_available: False
- phase925_selected_punctuation_keys: 30
- selected_holdout_cases: {'p856_021_material_wood': 30, 'p856_035_object_chair': 30, 'p885_047_animal_shark': 30}
- selected_holdout_states: 90

### qwen3

- candidate_punctuation_cases: {}
- candidate_punctuation_rows: 0
- case_ids: []
- deduped_unseen_punctuation_states: 0
- new_case_available: False
- phase925_selected_punctuation_keys: 0
- selected_holdout_cases: {}
- selected_holdout_states: 0

## Top Holdout Coverage

| model | group | factor | all states |
| --- | --- | ---: | ---: |
| glm4 | fixed_plus_train_case_inter_residual | 2.25 | 90 |
| glm4 | fixed_plus_train_case_union_residual | 2.25 | 90 |
| glm4 | state_specific_margin_support_pos_64 | 2.25 | 90 |
| glm4 | fixed_plus_train_case_union_residual | 2.1 | 88 |
| glm4 | fixed_plus_train_case_inter_residual | 2.1 | 67 |
| glm4 | state_specific_margin_support_pos_64 | 2.1 | 67 |
| glm4 | fixed_topfreq_64 | 2.25 | 60 |
| glm4 | fixed_plus_noncase_inter_size_control | 2.25 | 60 |
| glm4 | fixed_plus_noncase_union_size_control | 2.25 | 60 |
| glm4 | fixed_plus_pseudorandom_inter_size_control | 2.25 | 59 |
| glm4 | fixed_plus_pseudorandom_union_size_control | 2.25 | 55 |
| glm4 | fixed_plus_pseudorandom_union_size_control | 2.1 | 34 |
| glm4 | fixed_plus_pseudorandom_inter_size_control | 2.1 | 31 |
| glm4 | fixed_topfreq_64 | 2.1 | 30 |
| glm4 | fixed_plus_noncase_inter_size_control | 2.1 | 30 |
| glm4 | fixed_plus_noncase_union_size_control | 2.1 | 30 |

## Top Group Factor Rows

| model | group | factor | rows | top1 | margin | strict | new top1 | states | mean delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | fixed_plus_train_case_union_residual | 2.25 | 630 | 630 | 630 | 0 | 630 | 90 | 5.658680555555556 |
| glm4 | fixed_plus_train_case_inter_residual | 2.25 | 630 | 616 | 616 | 0 | 616 | 90 | 5.5772321428571425 |
| glm4 | state_specific_margin_support_pos_64 | 2.25 | 630 | 606 | 606 | 0 | 606 | 90 | 5.464037698412699 |
| glm4 | fixed_plus_train_case_union_residual | 2.1 | 630 | 474 | 474 | 0 | 474 | 88 | 5.046875 |
| glm4 | fixed_plus_train_case_inter_residual | 2.1 | 630 | 403 | 403 | 0 | 403 | 67 | 4.973164682539682 |
| glm4 | state_specific_margin_support_pos_64 | 2.1 | 630 | 324 | 324 | 0 | 324 | 67 | 4.86671626984127 |
| glm4 | fixed_plus_noncase_inter_size_control | 2.25 | 630 | 362 | 362 | 0 | 362 | 60 | 4.707688492063492 |
| glm4 | fixed_plus_noncase_union_size_control | 2.25 | 630 | 356 | 356 | 0 | 356 | 60 | 4.71780753968254 |
| glm4 | fixed_topfreq_64 | 2.25 | 630 | 348 | 348 | 0 | 348 | 60 | 4.633680555555555 |
| glm4 | fixed_plus_pseudorandom_inter_size_control | 2.25 | 630 | 362 | 362 | 0 | 362 | 59 | 4.670089285714286 |
| glm4 | fixed_plus_pseudorandom_union_size_control | 2.25 | 630 | 326 | 326 | 0 | 326 | 55 | 4.632986111111111 |
| glm4 | fixed_plus_pseudorandom_union_size_control | 2.1 | 630 | 202 | 202 | 0 | 202 | 34 | 4.068303571428571 |
| glm4 | fixed_plus_pseudorandom_inter_size_control | 2.1 | 630 | 201 | 201 | 0 | 201 | 31 | 4.099454365079365 |
| glm4 | fixed_plus_noncase_union_size_control | 2.1 | 630 | 210 | 210 | 0 | 210 | 30 | 4.137847222222222 |
| glm4 | fixed_plus_noncase_inter_size_control | 2.1 | 630 | 210 | 210 | 0 | 210 | 30 | 4.134275793650794 |
| glm4 | fixed_topfreq_64 | 2.1 | 630 | 210 | 210 | 0 | 210 | 30 | 4.060664682539683 |
| glm4 | coordinate_only | 1.0 | 630 | 0 | 0 | 0 | 0 | 0 | None |
