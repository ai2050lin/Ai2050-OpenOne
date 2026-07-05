# Phase 931 fixed punctuation margin gear causal validation

## Overall

- expected_rows_if_all_reconstructed: 2310
- overall_candidate_margin_nonnegative: 720
- overall_candidate_rows: 2100
- overall_candidate_strict_clean_candidate: 0
- overall_candidate_top1: 720
- overall_coordinate_baseline_rows: 210
- overall_improved_margin_vs_coordinate_base: 2100
- overall_margin_nonnegative: 720
- overall_new_margin_closure_vs_coordinate_base: 720
- overall_new_strict_vs_coordinate_base: 0
- overall_new_top1_vs_coordinate_base: 720
- overall_rows: 2310
- overall_strict_clean_candidate: 0
- overall_target_state_coverage_margin: 30
- overall_target_state_coverage_strict: 0
- overall_target_state_coverage_top1: 30
- overall_top1: 720
- overall_unique_cases: 3
- overall_unique_states: 30
- overall_worsened_margin_vs_coordinate_base: 0
- selected_punctuation_seeds: 30

## Evidence

- fixed_punctuation_margin_gear_causal_positive: 1
- no_punctuation_period_seeds: 2

## Top Group Factor Rows

| model | group | factor | rows | top1 | margin | strict | new top1 | new margin | states | mean delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | state_specific_margin_support_pos_64 | 2.25 | 210 | 200 | 200 | 0 | 200 | 200 | 30 | 5.430654761904762 |
| glm4 | fixed_topfreq_64 | 2.25 | 210 | 122 | 122 | 0 | 122 | 122 | 20 | 4.613988095238096 |
| glm4 | state_specific_margin_support_pos_64 | 2.1 | 210 | 98 | 98 | 0 | 98 | 98 | 22 | 4.845238095238095 |
| glm4 | fixed_half_or_more | 2.25 | 210 | 96 | 96 | 0 | 96 | 96 | 20 | 4.442261904761905 |
| glm4 | fixed_topfreq_64 | 2.1 | 210 | 70 | 70 | 0 | 70 | 70 | 10 | 4.045535714285714 |
| glm4 | fixed_half_or_more | 2.1 | 210 | 70 | 70 | 0 | 70 | 70 | 10 | 3.892261904761905 |
| glm4 | fixed_intersection_all | 2.25 | 210 | 32 | 32 | 0 | 32 | 32 | 10 | 3.624702380952381 |
| glm4 | fixed_topfreq_31 | 2.25 | 210 | 32 | 32 | 0 | 32 | 32 | 10 | 3.624702380952381 |
| glm4 | fixed_intersection_all | 2.1 | 210 | 0 | 0 | 0 | 0 | 0 | 0 | 3.1598214285714286 |
| glm4 | fixed_topfreq_31 | 2.1 | 210 | 0 | 0 | 0 | 0 | 0 | 0 | 3.1598214285714286 |
| glm4 | coordinate_only | 1.0 | 210 | 0 | 0 | 0 | 0 | 0 | 0 | None |
