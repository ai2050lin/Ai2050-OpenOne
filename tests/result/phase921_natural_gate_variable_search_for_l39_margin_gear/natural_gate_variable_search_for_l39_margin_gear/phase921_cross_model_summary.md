# Phase 921 natural gate variable search for consensus L39 signed margin gear

## Overall

- models: qwen3, glm4, deepseek7b
- factor_response_rows: 72
- low_factor_1375_margin: 8
- low_factor_1375_strict: 6
- low_factor_1375_top1: 8
- selected_phase915_l39_candidates: 12
- state_rows: 12

## Model Summaries

| model | selected | states | factor rows | low<=1.375 margin | low<=1.375 top1 | low<=1.375 strict | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |
| glm4 | 12 | 12 | 72 | 8 | 8 | 6 | candidate_gate_variables_separate_low_factor_closure |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |

## Factor Response

| model | factor | rows | top1 | margin>=0 | strict | median margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | 1.125 | 12 | 0 | 0 | 0 | -0.75 |
| glm4 | 1.25 | 12 | 0 | 0 | 0 | -0.3125 |
| glm4 | 1.375 | 12 | 8 | 8 | 6 | 0.125 |
| glm4 | 1.5 | 12 | 8 | 8 | 6 | 0.5 |
| glm4 | 1.75 | 12 | 12 | 12 | 10 | 1.4375 |
| glm4 | 2.0 | 12 | 12 | 12 | 10 | 2.25 |

## Top Gate Variable Candidates

| model | variable | n | pos | neg | pos mean | neg mean | delta | best acc | direction | threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| glm4 | route_eos_rank | 12 | 8 | 4 | 7.0 | 16.5 | -9.5 | 1.0 | low_true | 7.0 |
| glm4 | boundary_eos_rank | 12 | 8 | 4 | 5.0 | 12.0 | -7.0 | 1.0 | low_true | 5.0 |
| glm4 | simple_gate_pressure | 12 | 8 | 4 | -18.268249034881592 | -15.808308124542236 | -2.4599409103393555 | 1.0 | low_true | -17.78944206237793 |
| glm4 | protocol_blocker_pressure | 12 | 8 | 4 | 1.578125 | 3.609375 | -2.03125 | 1.0 | low_true | 1.6875 |
| glm4 | consensus_eos_support_sum | 12 | 8 | 4 | 24.233452796936035 | 22.678536891937256 | 1.5549159049987793 | 1.0 | high_true | 23.359822273254395 |
| glm4 | consensus_margin_support_sum | 12 | 8 | 4 | 19.369811534881592 | 17.839558124542236 | 1.5302534103393555 | 1.0 | high_true | 18.563444137573242 |
| glm4 | consensus_margin_support_pos_sum | 12 | 8 | 4 | 19.369811534881592 | 17.841750144958496 | 1.5280613899230957 | 1.0 | high_true | 18.567347526550293 |
| glm4 | consensus_activation_abs_median | 12 | 8 | 4 | 12.03125 | 10.65625 | 1.375 | 1.0 | high_true | 11.34375 |
| glm4 | consensus_activation_abs_mean | 12 | 8 | 4 | 15.41302490234375 | 14.19869613647461 | 1.2143287658691406 | 1.0 | high_true | 14.841171264648438 |
| glm4 | protocol_vs_eos | 12 | 8 | 4 | 0.4765625 | 1.578125 | -1.1015625 | 1.0 | low_true | 0.5625 |
| glm4 | route_eos_margin_vs_blocker | 12 | 8 | 4 | -1.515625 | -2.6015625 | 1.0859375 | 1.0 | high_true | -2.0 |
| glm4 | boundary_eos_margin_vs_blocker | 12 | 8 | 4 | -1.1015625 | -2.03125 | 0.9296875 | 1.0 | high_true | -1.515625 |
| glm4 | boundary_gap_to_zero | 12 | 8 | 4 | 1.1015625 | 2.03125 | -0.9296875 | 1.0 | low_true | 1.1875 |
| glm4 | stop_vs_top | 12 | 8 | 4 | -1.1015625 | -2.03125 | 0.9296875 | 1.0 | high_true | -1.515625 |
| glm4 | boundary_eos_logit | 12 | 8 | 4 | 8.328125 | 7.59375 | 0.734375 | 1.0 | high_true | 8.046875 |
| glm4 | boundary_stop_logit | 12 | 8 | 4 | 8.328125 | 7.59375 | 0.734375 | 1.0 | high_true | 8.046875 |
| glm4 | l4_activation_abs_top | 12 | 8 | 4 | 3.60546875 | 4.1953125 | -0.58984375 | 1.0 | low_true | 3.8125 |
| glm4 | min_margin_factor | 12 | 8 | 4 | 1.375 | 1.75 | -0.375 | 1.0 | low_true | 1.375 |
| glm4 | min_strict_factor | 10 | 6 | 4 | 1.375 | 1.75 | -0.375 | 1.0 | low_true | 1.375 |
| glm4 | min_top1_factor | 12 | 8 | 4 | 1.375 | 1.75 | -0.375 | 1.0 | low_true | 1.375 |
