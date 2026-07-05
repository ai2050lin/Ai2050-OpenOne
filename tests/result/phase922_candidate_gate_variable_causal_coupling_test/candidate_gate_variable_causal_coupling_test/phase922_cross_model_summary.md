# Phase 922 candidate gate variable causal coupling test

## Overall

- models: qwen3, glm4, deepseek7b
- all_improved_margin_vs_l39_only: 188
- all_margin_nonnegative: 114
- all_new_margin_closure_vs_l39_only: 2
- all_new_strict_vs_l39_only: 0
- all_new_top1_vs_l39_only: 2
- all_rows: 504
- all_strict_clean_candidate: 84
- all_top1: 114
- candidate_plus_improved_margin_vs_l39_only: 133
- candidate_plus_margin_nonnegative: 80
- candidate_plus_new_margin_closure_vs_l39_only: 0
- candidate_plus_new_strict_vs_l39_only: 0
- candidate_plus_new_top1_vs_l39_only: 0
- candidate_plus_rows: 360
- candidate_plus_strict_clean_candidate: 60
- candidate_plus_top1: 80
- coupled_nonbaseline_improved_margin_vs_l39_only: 188
- coupled_nonbaseline_margin_nonnegative: 106
- coupled_nonbaseline_new_margin_closure_vs_l39_only: 2
- coupled_nonbaseline_new_strict_vs_l39_only: 0
- coupled_nonbaseline_new_top1_vs_l39_only: 2
- coupled_nonbaseline_rows: 468
- coupled_nonbaseline_strict_clean_candidate: 78
- coupled_nonbaseline_top1: 106
- direction_control_improved_margin_vs_l39_only: 55
- direction_control_margin_nonnegative: 26
- direction_control_new_margin_closure_vs_l39_only: 2
- direction_control_new_strict_vs_l39_only: 0
- direction_control_new_top1_vs_l39_only: 2
- direction_control_rows: 108
- direction_control_strict_clean_candidate: 18
- direction_control_top1: 26
- l39_only_improved_margin_vs_l39_only: 0
- l39_only_margin_nonnegative: 8
- l39_only_new_margin_closure_vs_l39_only: 0
- l39_only_new_strict_vs_l39_only: 0
- l39_only_new_top1_vs_l39_only: 0
- l39_only_rows: 36
- l39_only_strict_clean_candidate: 6
- l39_only_top1: 8
- selected_phase915_l39_candidates: 12
- target_state_count: 12

## Model Summaries

| model | selected | states | l39 rows | l39 top1 | l39 margin | l39 strict | candidate rows | candidate new margin | candidate new top1 | candidate mean delta | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | no_phase915_l39_candidates |
| glm4 | 12 | 12 | 36 | 8 | 8 | 6 | 360 | 0 | 0 | 0.021527777777777778 | candidate_moves_margin_but_direction_control_only_adds_closure |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | no_phase915_l39_candidates |

## Top Controls

| model | control | class | family | rows | top1 | margin | strict | improved | new margin | new top1 | new strict | lost margin | mean delta vs l39 | median patched margin |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | route_alpha_0.875_direction_control | direction_control | route_direction_control | 36 | 10 | 10 | 6 | 27 | 2 | 2 | 0 | 0 | 0.07291666666666667 | -0.5 |
| glm4 | route_alpha_1.25 | candidate_plus | route_candidate_plus | 36 | 8 | 8 | 6 | 32 | 0 | 0 | 0 | 0 | 0.1440972222222222 | -0.375 |
| glm4 | route_1.125_protocol_last8_0.90 | candidate_plus_combo | route_protocol_combo_candidate_plus | 36 | 8 | 8 | 6 | 24 | 0 | 0 | 0 | 0 | 0.07291666666666667 | -0.4375 |
| glm4 | protocol_last8_1.10_direction_control | direction_control | protocol_pressure_direction_control | 36 | 8 | 8 | 6 | 18 | 0 | 0 | 0 | 0 | 0.036458333333333336 | -0.5625 |
| glm4 | route_1.125_l4_1.05 | candidate_plus_combo | route_l4_combo_candidate_plus | 36 | 8 | 8 | 6 | 17 | 0 | 0 | 0 | 0 | 0.017361111111111112 | -0.5625 |
| glm4 | route_1.125_l4_1.05_protocol_last8_0.90 | candidate_plus_combo | route_l4_protocol_combo_candidate_plus | 36 | 8 | 8 | 6 | 15 | 0 | 0 | 0 | 0 | 0.050347222222222224 | -0.5625 |
| glm4 | route_alpha_1.125 | candidate_plus | route_candidate_plus | 36 | 8 | 8 | 6 | 14 | 0 | 0 | 0 | 0 | 0.03298611111111111 | -0.5 |
| glm4 | l4_boundary_0.95_direction_control | direction_control | l4_boundary_direction_control | 36 | 8 | 8 | 6 | 10 | 0 | 0 | 0 | 0 | 0.008680555555555556 | -0.53125 |
| glm4 | l4_boundary_1.05 | candidate_plus | l4_boundary_candidate_plus | 36 | 8 | 8 | 6 | 10 | 0 | 0 | 0 | 0 | 0.001736111111111111 | -0.5625 |
| glm4 | protocol_last8_0.90 | candidate_plus | protocol_pressure_candidate_suppress | 36 | 8 | 8 | 6 | 9 | 0 | 0 | 0 | 0 | -0.019097222222222224 | -0.625 |
| glm4 | l4_1.05_protocol_last8_0.90 | candidate_plus_combo | l4_protocol_combo_candidate_plus | 36 | 8 | 8 | 6 | 5 | 0 | 0 | 0 | 0 | -0.012152777777777778 | -0.5625 |
| glm4 | protocol_answer_last_0.90 | candidate_plus | protocol_pressure_candidate_suppress | 36 | 8 | 8 | 6 | 5 | 0 | 0 | 0 | 0 | -0.029513888888888888 | -0.625 |
| glm4 | l4_boundary_1.10 | candidate_plus | l4_boundary_candidate_plus | 36 | 8 | 8 | 6 | 2 | 0 | 0 | 0 | 0 | -0.043402777777777776 | -0.625 |
| glm4 | l39_only | baseline | l39_low_factor_baseline | 36 | 8 | 8 | 6 | 0 | 0 | 0 | 0 | 0 | 0.0 | -0.5625 |

## Top Control Factors

| model | control | factor | rows | top1 | margin | strict | improved | new margin | new top1 | mean delta vs l39 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | route_alpha_0.875_direction_control | 1.25 | 12 | 2 | 2 | 0 | 6 | 2 | 2 | 0.06770833333333333 |
| glm4 | route_alpha_1.25 | 1.25 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.14583333333333334 |
| glm4 | route_alpha_0.875_direction_control | 1.375 | 12 | 8 | 8 | 6 | 12 | 0 | 0 | 0.09895833333333333 |
| glm4 | route_alpha_1.25 | 1.125 | 12 | 0 | 0 | 0 | 10 | 0 | 0 | 0.16666666666666666 |
| glm4 | route_alpha_1.25 | 1.375 | 12 | 8 | 8 | 6 | 10 | 0 | 0 | 0.11979166666666667 |
| glm4 | protocol_last8_1.10_direction_control | 1.25 | 12 | 0 | 0 | 0 | 9 | 0 | 0 | 0.0625 |
| glm4 | route_alpha_0.875_direction_control | 1.125 | 12 | 0 | 0 | 0 | 9 | 0 | 0 | 0.052083333333333336 |
| glm4 | route_1.125_protocol_last8_0.90 | 1.125 | 12 | 0 | 0 | 0 | 8 | 0 | 0 | 0.08854166666666667 |
| glm4 | route_1.125_protocol_last8_0.90 | 1.25 | 12 | 0 | 0 | 0 | 8 | 0 | 0 | 0.08333333333333333 |
| glm4 | route_1.125_protocol_last8_0.90 | 1.375 | 12 | 8 | 8 | 6 | 8 | 0 | 0 | 0.046875 |
| glm4 | l4_boundary_1.05 | 1.375 | 12 | 8 | 8 | 6 | 7 | 0 | 0 | 0.046875 |
| glm4 | route_1.125_l4_1.05 | 1.375 | 12 | 8 | 8 | 6 | 7 | 0 | 0 | 0.020833333333333332 |
| glm4 | route_1.125_l4_1.05 | 1.25 | 12 | 0 | 0 | 0 | 6 | 0 | 0 | 0.036458333333333336 |
| glm4 | protocol_last8_1.10_direction_control | 1.125 | 12 | 0 | 0 | 0 | 6 | 0 | 0 | 0.03125 |
| glm4 | route_alpha_1.125 | 1.25 | 12 | 0 | 0 | 0 | 6 | 0 | 0 | 0.03125 |
| glm4 | route_1.125_l4_1.05_protocol_last8_0.90 | 1.25 | 12 | 0 | 0 | 0 | 5 | 0 | 0 | 0.0625 |
| glm4 | route_1.125_l4_1.05_protocol_last8_0.90 | 1.125 | 12 | 0 | 0 | 0 | 5 | 0 | 0 | 0.046875 |
| glm4 | route_1.125_l4_1.05_protocol_last8_0.90 | 1.375 | 12 | 8 | 8 | 6 | 5 | 0 | 0 | 0.041666666666666664 |
| glm4 | l4_boundary_0.95_direction_control | 1.375 | 12 | 8 | 8 | 6 | 5 | 0 | 0 | 0.03125 |
| glm4 | protocol_last8_0.90 | 1.125 | 12 | 0 | 0 | 0 | 5 | 0 | 0 | -0.020833333333333332 |
| glm4 | route_alpha_1.125 | 1.375 | 12 | 8 | 8 | 6 | 4 | 0 | 0 | 0.036458333333333336 |
| glm4 | route_alpha_1.125 | 1.125 | 12 | 0 | 0 | 0 | 4 | 0 | 0 | 0.03125 |
| glm4 | l4_boundary_0.95_direction_control | 1.125 | 12 | 0 | 0 | 0 | 4 | 0 | 0 | 0.015625 |
| glm4 | route_1.125_l4_1.05 | 1.125 | 12 | 0 | 0 | 0 | 4 | 0 | 0 | -0.005208333333333333 |
| glm4 | protocol_last8_1.10_direction_control | 1.375 | 12 | 8 | 8 | 6 | 3 | 0 | 0 | 0.015625 |
| glm4 | protocol_last8_0.90 | 1.375 | 12 | 8 | 8 | 6 | 2 | 0 | 0 | -0.005208333333333333 |
| glm4 | l4_1.05_protocol_last8_0.90 | 1.125 | 12 | 0 | 0 | 0 | 2 | 0 | 0 | -0.020833333333333332 |
| glm4 | l4_boundary_1.05 | 1.25 | 12 | 0 | 0 | 0 | 2 | 0 | 0 | -0.020833333333333332 |
| glm4 | protocol_answer_last_0.90 | 1.25 | 12 | 0 | 0 | 0 | 2 | 0 | 0 | -0.026041666666666668 |
| glm4 | protocol_last8_0.90 | 1.25 | 12 | 0 | 0 | 0 | 2 | 0 | 0 | -0.03125 |
| glm4 | protocol_answer_last_0.90 | 1.125 | 12 | 0 | 0 | 0 | 2 | 0 | 0 | -0.052083333333333336 |
| glm4 | l4_1.05_protocol_last8_0.90 | 1.375 | 12 | 8 | 8 | 6 | 2 | 0 | 0 | 0.0 |
| glm4 | protocol_answer_last_0.90 | 1.375 | 12 | 8 | 8 | 6 | 1 | 0 | 0 | -0.010416666666666666 |
| glm4 | l4_1.05_protocol_last8_0.90 | 1.25 | 12 | 0 | 0 | 0 | 1 | 0 | 0 | -0.015625 |
| glm4 | l4_boundary_1.05 | 1.125 | 12 | 0 | 0 | 0 | 1 | 0 | 0 | -0.020833333333333332 |
| glm4 | l4_boundary_0.95_direction_control | 1.25 | 12 | 0 | 0 | 0 | 1 | 0 | 0 | -0.020833333333333332 |
| glm4 | l4_boundary_1.10 | 1.375 | 12 | 8 | 8 | 6 | 1 | 0 | 0 | -0.03125 |
| glm4 | l4_boundary_1.10 | 1.25 | 12 | 0 | 0 | 0 | 1 | 0 | 0 | -0.052083333333333336 |
| glm4 | l4_boundary_1.10 | 1.125 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | -0.046875 |
| glm4 | l39_only | 1.125 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| glm4 | l39_only | 1.25 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| glm4 | l39_only | 1.375 | 12 | 8 | 8 | 6 | 0 | 0 | 0 | 0.0 |
