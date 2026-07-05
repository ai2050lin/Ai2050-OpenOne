# Phase 930 natural gate and strict-clean transition audit

## Overall

- expected_rows_if_all_reconstructed: 1470
- overall_candidate_margin_nonnegative: 774
- overall_candidate_rows: 1260
- overall_candidate_strict_clean_candidate: 0
- overall_candidate_top1: 774
- overall_coordinate_baseline_rows: 210
- overall_improved_margin_vs_coordinate_base: 1260
- overall_margin_nonnegative: 774
- overall_new_margin_closure_vs_coordinate_base: 774
- overall_new_strict_vs_coordinate_base: 0
- overall_new_top1_vs_coordinate_base: 774
- overall_rows: 1470
- overall_strict_clean_candidate: 0
- overall_target_state_coverage_margin: 30
- overall_target_state_coverage_strict: 0
- overall_target_state_coverage_top1: 30
- overall_top1: 774
- overall_unique_cases: 3
- overall_unique_states: 30
- overall_worsened_margin_vs_coordinate_base: 0
- selected_punctuation_seeds: 30
- threshold_opened: 30
- threshold_opened_at_or_below_2_00: 10
- threshold_opened_at_or_below_2_10: 22
- threshold_opened_at_or_below_2_25: 30
- threshold_states: 30
- threshold_strict_clean_at_opening: 0

## Evidence

- no_punctuation_period_seeds: 2
- threshold_gate_candidate_found_without_strict_clean: 1

## Top Gate Candidate Splits

| model | feature | target | threshold | polarity | accuracy | correct | total | true | false |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| glm4 | target_route_delta_norm | opened_at_or_below_2_00 | 0.03834216110408306 | ge_true | 1.0 | 30 | 30 | 10 | 20 |
| glm4 | target_boundary_eos_margin_vs_blocker | opened_at_or_below_2_00 | -4.0 | ge_true | 1.0 | 30 | 30 | 10 | 20 |
| glm4 | target_boundary_eos_rank | opened_at_or_below_2_00 | 6.0 | le_true | 1.0 | 30 | 30 | 10 | 20 |
| glm4 | boundary_period_gap_vs_eos | opened_at_or_below_2_00 | 4.0 | le_true | 1.0 | 30 | 30 | 10 | 20 |
| glm4 | boundary_punctuation_gap_vs_eos | opened_at_or_below_2_00 | 4.0 | le_true | 1.0 | 30 | 30 | 10 | 20 |
| glm4 | l39_activation_abs_top | opened_at_or_below_2_00 | 32.75 | ge_true | 1.0 | 30 | 30 | 10 | 20 |
| glm4 | l39_margin_pos_mean_score | opened_at_or_below_2_00 | 0.22324757277965546 | ge_true | 1.0 | 30 | 30 | 10 | 20 |
| glm4 | l39_margin_pos_min_score | opened_at_or_below_2_00 | 0.07634490728378296 | ge_true | 1.0 | 30 | 30 | 10 | 20 |
| glm4 | l39_eos_support_mean_score | opened_at_or_below_2_00 | 0.46091899275779724 | ge_true | 1.0 | 30 | 30 | 10 | 20 |
| glm4 | target_boundary_blocker_logit | opened_at_or_below_2_10 | 12.4375 | ge_true | 0.9333333333333333 | 28 | 30 | 22 | 8 |
| glm4 | l39_activation_abs_median | opened_at_or_below_2_10 | 0.074462890625 | ge_true | 0.9333333333333333 | 28 | 30 | 22 | 8 |
| glm4 | l39_margin_pos_mean_score | opened_at_or_below_2_10 | 0.21317481994628906 | ge_true | 0.9333333333333333 | 28 | 30 | 22 | 8 |
| glm4 | l39_neg_margin_mean_score | opened_at_or_below_2_10 | -0.15297437459230423 | ge_true | 0.9333333333333333 | 28 | 30 | 22 | 8 |
| glm4 | l39_activation_abs_top | opened_at_or_below_2_10 | 31.3125 | ge_true | 0.8666666666666667 | 26 | 30 | 22 | 8 |
| glm4 | l39_margin_pos_max_score | opened_at_or_below_2_10 | 1.1999151110649109 | ge_true | 0.8666666666666667 | 26 | 30 | 22 | 8 |
| glm4 | phase925_factor | opened_at_or_below_2_10 | 0.45 | ge_true | 0.8666666666666667 | 26 | 30 | 22 | 8 |
| glm4 | l39_activation_abs_median | opened_at_or_below_2_00 | 0.080322265625 | ge_true | 0.7 | 21 | 30 | 10 | 20 |
| glm4 | l39_margin_pos_min_score | opened_at_or_below_2_10 | 0.05772619694471359 | ge_true | 0.7 | 21 | 30 | 22 | 8 |
| glm4 | l39_eos_support_mean_score | opened_at_or_below_2_10 | 0.391430526971817 | ge_true | 0.7 | 21 | 30 | 22 | 8 |
| glm4 | target_boundary_blocker_logit | opened_at_or_below_2_00 | 12.4375 | ge_true | 0.6666666666666666 | 20 | 30 | 10 | 20 |
| glm4 | l39_margin_pos_max_score | opened_at_or_below_2_00 | 1.2322205305099487 | le_true | 0.6666666666666666 | 20 | 30 | 10 | 20 |
| glm4 | l39_neg_margin_mean_score | opened_at_or_below_2_00 | -0.15248312801122665 | ge_true | 0.6666666666666666 | 20 | 30 | 10 | 20 |
| glm4 | phase925_factor | opened_at_or_below_2_00 | 0.6499999999999999 | le_true | 0.6666666666666666 | 20 | 30 | 10 | 20 |
| glm4 | target_route_delta_norm | opened_at_or_below_2_10 | 0.036896469071507454 | le_true | 0.6 | 18 | 30 | 22 | 8 |
| glm4 | target_boundary_eos_margin_vs_blocker | opened_at_or_below_2_10 | -6.0 | ge_true | 0.6 | 18 | 30 | 22 | 8 |
| glm4 | target_boundary_eos_rank | opened_at_or_below_2_10 | 6.0 | le_true | 0.6 | 18 | 30 | 22 | 8 |
| glm4 | boundary_period_gap_vs_eos | opened_at_or_below_2_10 | 4.0 | le_true | 0.6 | 18 | 30 | 22 | 8 |
| glm4 | boundary_punctuation_gap_vs_eos | opened_at_or_below_2_10 | 4.0 | le_true | 0.6 | 18 | 30 | 22 | 8 |

## Model Thresholds

### qwen3

- states: 0
- opened: 0
- opened_at_or_below_2_00: 0
- opened_at_or_below_2_10: 0
- opened_at_or_below_2_25: 0
- strict_clean_at_opening: 0
- threshold_median: None
- threshold_mean: None
- by_case:
- channel_stability: union=0, intersection=0, half=None, quarter=None

### glm4

- states: 30
- opened: 30
- opened_at_or_below_2_00: 10
- opened_at_or_below_2_10: 22
- opened_at_or_below_2_25: 30
- strict_clean_at_opening: 0
- threshold_median: 2.1
- threshold_mean: 2.08
- by_case:
  - p856_021_material_wood: states=10, opened=10, threshold_median=2.15, <=2.00=0, <=2.10=2, <=2.25=10
  - p856_035_object_chair: states=10, opened=10, threshold_median=2.1, <=2.00=0, <=2.10=10, <=2.25=10
  - p885_047_animal_shark: states=10, opened=10, threshold_median=2.0, <=2.00=10, <=2.10=10, <=2.25=10
- channel_stability: union=105, intersection=31, half=59, quarter=97

### deepseek7b

- states: 0
- opened: 0
- opened_at_or_below_2_00: 0
- opened_at_or_below_2_10: 0
- opened_at_or_below_2_25: 0
- strict_clean_at_opening: 0
- threshold_median: None
- threshold_mean: None
- by_case:
- channel_stability: union=0, intersection=0, half=None, quarter=None

