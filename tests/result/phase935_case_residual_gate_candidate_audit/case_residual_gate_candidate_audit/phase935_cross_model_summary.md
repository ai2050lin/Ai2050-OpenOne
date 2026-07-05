# Phase 935 case residual gate candidate audit

## Overall

- fixed_success_2_25: 20
- residual_needed_2_25: 10
- size_control_success_2_25: 20
- state_rows: 30
- true_beats_controls_2_25: 10
- true_loso_repair_success_2_25: 30

## Evidence

- case_residual_gate_candidate_case_confounded: 1
- no_phase934_gate_data: 2

## Top Feature Splits

| model | target | feature | threshold | polarity | accuracy | true | false | case_confounded |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| glm4 | residual_needed_2_25 | target_route_delta_norm | 0.036896469071507454 | le_true | 1.0 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | target_boundary_eos_margin_vs_blocker | -5.078125 | le_true | 1.0 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | target_boundary_eos_rank | 8.5 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | boundary_period_gap_vs_eos | 5.078125 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | boundary_punctuation_gap_vs_eos | 5.078125 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | l39_margin_pos_max_score | 1.3660572171211243 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | l39_margin_pos_min_score | 0.06339829787611961 | le_true | 1.0 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | l39_eos_support_mean_score | 0.41953757405281067 | le_true | 1.0 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | l39_neg_margin_mean_score | -0.14916343241930008 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | target_route_delta_norm | 0.036896469071507454 | le_true | 1.0 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | target_boundary_eos_margin_vs_blocker | -5.078125 | le_true | 1.0 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | target_boundary_eos_rank | 8.5 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | boundary_period_gap_vs_eos | 5.078125 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | boundary_punctuation_gap_vs_eos | 5.078125 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | l39_margin_pos_max_score | 1.3660572171211243 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | l39_margin_pos_min_score | 0.06339829787611961 | le_true | 1.0 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | l39_eos_support_mean_score | 0.41953757405281067 | le_true | 1.0 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | l39_neg_margin_mean_score | -0.14916343241930008 | ge_true | 1.0 | 10 | 20 | True |
| glm4 | fixed_success_2_25 | target_route_delta_norm | 0.036896469071507454 | ge_true | 1.0 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | target_boundary_eos_margin_vs_blocker | -5.078125 | ge_true | 1.0 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | target_boundary_eos_rank | 8.5 | le_true | 1.0 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | boundary_period_gap_vs_eos | 5.078125 | le_true | 1.0 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | boundary_punctuation_gap_vs_eos | 5.078125 | le_true | 1.0 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | l39_margin_pos_max_score | 1.3660572171211243 | le_true | 1.0 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | l39_margin_pos_min_score | 0.06339829787611961 | ge_true | 1.0 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | l39_eos_support_mean_score | 0.41953757405281067 | ge_true | 1.0 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | l39_neg_margin_mean_score | -0.14916343241930008 | le_true | 1.0 | 20 | 10 | True |
| glm4 | residual_needed_2_25 | l39_activation_abs_median | 0.081787109375 | ge_true | 0.9666666666666667 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | l39_activation_abs_median | 0.081787109375 | ge_true | 0.9666666666666667 | 10 | 20 | True |
| glm4 | fixed_success_2_25 | l39_activation_abs_median | 0.081787109375 | le_true | 0.9666666666666667 | 20 | 10 | True |
| glm4 | residual_needed_2_25 | phase925_factor | 0.75 | ge_true | 0.8333333333333334 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | phase925_factor | 0.75 | ge_true | 0.8333333333333334 | 10 | 20 | True |
| glm4 | fixed_success_2_25 | phase925_factor | 0.75 | le_true | 0.8333333333333334 | 20 | 10 | True |
| glm4 | residual_needed_2_25 | opening_threshold_factor | 2.05 | ge_true | 0.6666666666666666 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | l39_activation_abs_top | 32.75 | le_true | 0.6666666666666666 | 10 | 20 | True |
| glm4 | residual_needed_2_25 | l39_margin_pos_mean_score | 0.21527249366044998 | ge_true | 0.6666666666666666 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | opening_threshold_factor | 2.05 | ge_true | 0.6666666666666666 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | l39_activation_abs_top | 32.75 | le_true | 0.6666666666666666 | 10 | 20 | True |
| glm4 | true_beats_controls_2_25 | l39_margin_pos_mean_score | 0.21527249366044998 | ge_true | 0.6666666666666666 | 10 | 20 | True |
| glm4 | fixed_success_2_25 | opening_threshold_factor | 2.05 | le_true | 0.6666666666666666 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | l39_activation_abs_top | 32.75 | ge_true | 0.6666666666666666 | 20 | 10 | True |
| glm4 | fixed_success_2_25 | l39_margin_pos_mean_score | 0.21527249366044998 | le_true | 0.6666666666666666 | 20 | 10 | True |

## Case Summary

### glm4

- p856_021_material_wood: states=10, fixed=10, residual_needed=0, true_repair=10, size_control=10, true_beats_controls=0
- p856_035_object_chair: states=10, fixed=0, residual_needed=10, true_repair=10, size_control=0, true_beats_controls=10
- p885_047_animal_shark: states=10, fixed=10, residual_needed=0, true_repair=10, size_control=10, true_beats_controls=0

