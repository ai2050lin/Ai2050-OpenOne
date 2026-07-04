# Phase 915 near-boundary action gate search

## Overall

- models: qwen3, glm4, deepseek7b
- action_margin_nonnegative: 0
- action_promoted_margin: 0
- action_promoted_top1: 0
- action_promoted_top5: 4
- action_rank_improved: 131
- action_rows: 1104
- action_strict_clean_candidate: 0
- action_top1: 0
- action_top10: 795
- action_top5: 617
- boundary_margin_nonnegative: 0
- boundary_rows: 12
- boundary_top1: 0
- boundary_top5: 8
- diagnostic_margin_nonnegative: 9
- diagnostic_promoted_margin: 9
- diagnostic_rows: 36
- diagnostic_top1: 9
- rows: 1152
- selected_phase914_candidates: 12
- weak_action_candidate: 106

## Model Summaries

| model | selected | rows | action rows | action top1 | action margin>=0 | promoted margin | promoted top5 | weak action | strict | diagnostic margin | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase914_near_boundary_candidates |
| glm4 | 12 | 1152 | 1104 | 0 | 0 | 0 | 4 | 106 | 0 | 9 | diagnostic_blocker_mask_can_close_margin |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase914_near_boundary_candidates |

## Top Controls

| model | control | family | site | direction | beta | scale | rows | top1 | margin>=0 | promoted margin | promoted top5 | weak | rank improved | median margin delta | mean eos delta |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | L39_mlp_output_scale_1.5 | component_output_scale | L39_mlp | None | None | 1.5 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0.9375 | 0.4244791666666667 |
| glm4 | l0_output_eos_minus_blocker_top1_beta_0.25 | readout_action_vector | l0_output | eos_minus_blocker_top1 | 0.25 | None | 12 | 0 | 0 | 0 | 1 | 1 | 1 | -1.625 | -2.1822916666666665 |
| glm4 | l0_output_eos_minus_blocker_top1_beta_0.5 | readout_action_vector | l0_output | eos_minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 1 | 0 | 1 | -3.5 | -3.1686197916666665 |
| glm4 | L36_mlp_eos_boost_beta_0.5 | readout_action_vector | L36_mlp | eos_boost | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 6 | 6 | 0.125 | 0.1484375 |
| glm4 | L36_attn_eos_minus_blocker_top1_beta_0.5 | readout_action_vector | L36_attn | eos_minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 6 | 6 | 0.125 | 0.10416666666666667 |
| glm4 | L36_attn_eos_boost_beta_0.5 | readout_action_vector | L36_attn | eos_boost | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 6 | 6 | 0.125 | 0.14583333333333334 |
| glm4 | L39_attn_output_scale_0.5 | component_output_scale | L39_attn | None | None | 0.5 | 12 | 0 | 0 | 0 | 0 | 6 | 6 | 0.109375 | 0.059895833333333336 |
| glm4 | L39_attn_eos_minus_blocker_top1_beta_0.5 | readout_action_vector | L39_attn | eos_minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 6 | 6 | 0.078125 | 0.08333333333333333 |
| glm4 | L39_mlp_eos_minus_blocker_top1_beta_0.5 | readout_action_vector | L39_mlp | eos_minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 5 | 5 | 0.125 | 0.0625 |
| glm4 | L39_attn_eos_boost_beta_0.5 | readout_action_vector | L39_attn | eos_boost | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 5 | 5 | 0.125 | 0.125 |
| glm4 | L36_mlp_eos_minus_blocker_top1_beta_0.5 | readout_action_vector | L36_mlp | eos_minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 5 | 5 | 0.109375 | 0.08854166666666667 |
| glm4 | L39_attn_eos_boost_beta_0.25 | readout_action_vector | L39_attn | eos_boost | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 5 | 5 | 0.0625 | 0.06510416666666667 |
| glm4 | L36_attn_output_scale_0.5 | component_output_scale | L36_attn | None | None | 0.5 | 12 | 0 | 0 | 0 | 0 | 4 | 6 | 0.1875 | 0.6848958333333334 |
| glm4 | L39_mlp_eos_boost_beta_0.5 | readout_action_vector | L39_mlp | eos_boost | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 4 | 6 | 0.109375 | 0.10416666666666667 |
| glm4 | L36_mlp_output_scale_0.5 | component_output_scale | L36_mlp | None | None | 0.5 | 12 | 0 | 0 | 0 | 0 | 4 | 4 | 0.203125 | 0.20833333333333334 |
| glm4 | L36_attn_eos_boost_beta_0.25 | readout_action_vector | L36_attn | eos_boost | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 3 | 5 | 0.0625 | 0.06770833333333333 |
| glm4 | L39_attn_output_scale_1.5 | component_output_scale | L39_attn | None | None | 1.5 | 12 | 0 | 0 | 0 | 0 | 3 | 4 | -0.0625 | 0.033854166666666664 |
| glm4 | L39_attn_eos_minus_blocker_top1_beta_0.25 | readout_action_vector | L39_attn | eos_minus_blocker_top1 | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 3 | 3 | 0.0625 | 0.06510416666666667 |
| glm4 | L39_mlp_eos_boost_beta_0.25 | readout_action_vector | L39_mlp | eos_boost | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 2 | 2 | 0.0625 | 0.049479166666666664 |
| glm4 | L36_attn_minus_blocker_top3_mean_beta_0.5 | readout_action_vector | L36_attn | minus_blocker_top3_mean | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 2 | 2 | 0.0625 | 0.057291666666666664 |
| glm4 | L39_attn_minus_blocker_top3_mean_beta_0.5 | readout_action_vector | L39_attn | minus_blocker_top3_mean | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 2 | 2 | 0.046875 | 0.036458333333333336 |
| glm4 | l0_output_eos_boost_beta_0.1 | readout_action_vector | l0_output | eos_boost | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 1 | 4 | -0.015625 | -0.036458333333333336 |
| glm4 | l0_output_minus_blocker_top3_mean_beta_0.1 | readout_action_vector | l0_output | minus_blocker_top3_mean | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 1 | 4 | -0.0625 | -0.0703125 |
| glm4 | L36_mlp_eos_boost_beta_0.25 | readout_action_vector | L36_mlp | eos_boost | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.078125 | 0.0625 |
| glm4 | L39_attn_eos_boost_beta_0.1 | readout_action_vector | L39_attn | eos_boost | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0625 | 0.044270833333333336 |
| glm4 | L36_mlp_eos_minus_blocker_top1_beta_0.25 | readout_action_vector | L36_mlp | eos_minus_blocker_top1 | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0625 | 0.044270833333333336 |
| glm4 | L36_mlp_minus_blocker_top3_mean_beta_0.25 | readout_action_vector | L36_mlp | minus_blocker_top3_mean | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0625 | 0.041666666666666664 |
| glm4 | L36_mlp_minus_blocker_top3_mean_beta_0.5 | readout_action_vector | L36_mlp | minus_blocker_top3_mean | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0625 | 0.059895833333333336 |
| glm4 | L36_mlp_eos_boost_beta_0.1 | readout_action_vector | L36_mlp | eos_boost | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0625 | 0.049479166666666664 |
| glm4 | L36_attn_eos_minus_blocker_top1_beta_0.1 | readout_action_vector | L36_attn | eos_minus_blocker_top1 | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0625 | 0.059895833333333336 |
| glm4 | L36_attn_eos_minus_blocker_top1_beta_0.25 | readout_action_vector | L36_attn | eos_minus_blocker_top1 | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0625 | 0.052083333333333336 |
| glm4 | L36_attn_minus_blocker_top1_beta_0.5 | readout_action_vector | L36_attn | minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0625 | 0.03125 |
| glm4 | L36_attn_eos_boost_beta_0.1 | readout_action_vector | L36_attn | eos_boost | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0625 | 0.044270833333333336 |
| glm4 | l0_output_minus_blocker_top3_mean_beta_0.05 | readout_action_vector | l0_output | minus_blocker_top3_mean | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0 | -0.020833333333333332 |
| glm4 | l0_output_eos_boost_beta_0.05 | readout_action_vector | l0_output | eos_boost | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0 | -0.013020833333333334 |
| glm4 | L39_attn_minus_blocker_top3_mean_beta_0.25 | readout_action_vector | L39_attn | minus_blocker_top3_mean | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0 | 0.010416666666666666 |
| glm4 | L36_attn_eos_boost_beta_0.05 | readout_action_vector | L36_attn | eos_boost | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 1 | 1 | 0.0 | 0.013020833333333334 |
| glm4 | l0_output_minus_blocker_top3_mean_beta_0.5 | readout_action_vector | l0_output | minus_blocker_top3_mean | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 0 | 3 | -0.71875 | -0.13541666666666666 |
| glm4 | L36_attn_output_scale_1.5 | component_output_scale | L36_attn | None | None | 1.5 | 12 | 0 | 0 | 0 | 0 | 0 | 2 | 0.140625 | -0.5546875 |
| glm4 | L36_mlp_minus_blocker_top3_mean_beta_0.05 | readout_action_vector | L36_mlp | minus_blocker_top3_mean | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 2 | 0.046875 | 0.013020833333333334 |
| glm4 | l0_output_minus_blocker_top1_beta_0.1 | readout_action_vector | l0_output | minus_blocker_top1 | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 2 | -0.0625 | -0.046875 |
| glm4 | l0_output_eos_minus_blocker_top1_beta_0.1 | readout_action_vector | l0_output | eos_minus_blocker_top1 | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 2 | 0.0 | -0.052083333333333336 |
| glm4 | diagnostic_mask_boundary_blocker_top8 | logit_mask_diagnostic | None | None | None | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | None | None |
| glm4 | L39_mlp_minus_blocker_top1_beta_0.5 | readout_action_vector | L39_mlp | minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.125 | 0.0026041666666666665 |
| glm4 | L39_mlp_eos_minus_blocker_top1_beta_0.25 | readout_action_vector | L39_mlp | eos_minus_blocker_top1 | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.041666666666666664 |
| glm4 | L39_mlp_minus_blocker_top1_beta_0.25 | readout_action_vector | L39_mlp | minus_blocker_top1 | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.0 |
| glm4 | L39_mlp_minus_blocker_top3_mean_beta_0.5 | readout_action_vector | L39_mlp | minus_blocker_top3_mean | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | -0.010416666666666666 |
| glm4 | L39_attn_eos_minus_blocker_top1_beta_0.1 | readout_action_vector | L39_attn | eos_minus_blocker_top1 | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.044270833333333336 |
| glm4 | L36_mlp_minus_blocker_top1_beta_0.1 | readout_action_vector | L36_mlp | minus_blocker_top1 | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.041666666666666664 |
| glm4 | L36_mlp_minus_blocker_top1_beta_0.25 | readout_action_vector | L36_mlp | minus_blocker_top1 | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.044270833333333336 |
| glm4 | L36_mlp_minus_blocker_top1_beta_0.5 | readout_action_vector | L36_mlp | minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.0390625 |
| glm4 | L36_mlp_minus_blocker_top3_mean_beta_0.1 | readout_action_vector | L36_mlp | minus_blocker_top3_mean | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.041666666666666664 |
| glm4 | L36_attn_eos_minus_blocker_top1_beta_0.05 | readout_action_vector | L36_attn | eos_minus_blocker_top1 | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.033854166666666664 |
| glm4 | L36_attn_minus_blocker_top3_mean_beta_0.25 | readout_action_vector | L36_attn | minus_blocker_top3_mean | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.010416666666666666 |
| glm4 | L39_mlp_eos_boost_beta_0.1 | readout_action_vector | L39_mlp | eos_boost | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.046875 | 0.041666666666666664 |
| glm4 | L36_mlp_eos_boost_beta_0.05 | readout_action_vector | L36_mlp | eos_boost | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.046875 | 0.0234375 |
| glm4 | l0_output_eos_minus_blocker_top1_beta_0.05 | readout_action_vector | l0_output | eos_minus_blocker_top1 | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03125 | -0.015625 |
| glm4 | L39_mlp_eos_minus_blocker_top1_beta_0.1 | readout_action_vector | L39_mlp | eos_minus_blocker_top1 | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03125 | 0.026041666666666668 |
| glm4 | L39_attn_eos_minus_blocker_top1_beta_0.05 | readout_action_vector | L39_attn | eos_minus_blocker_top1 | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03125 | 0.026041666666666668 |
| glm4 | L36_attn_minus_blocker_top1_beta_0.05 | readout_action_vector | L36_attn | minus_blocker_top1 | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03125 | 0.036458333333333336 |
| glm4 | L39_attn_minus_blocker_top1_beta_0.25 | readout_action_vector | L39_attn | minus_blocker_top1 | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.015625 | 0.0234375 |
| glm4 | L39_attn_minus_blocker_top1_beta_0.5 | readout_action_vector | L39_attn | minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.015625 | 0.0234375 |
| glm4 | L39_attn_minus_blocker_top3_mean_beta_0.05 | readout_action_vector | L39_attn | minus_blocker_top3_mean | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.015625 | 0.020833333333333332 |
| glm4 | L36_mlp_eos_minus_blocker_top1_beta_0.1 | readout_action_vector | L36_mlp | eos_minus_blocker_top1 | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.015625 | 0.028645833333333332 |
| glm4 | l0_output_minus_blocker_top1_beta_0.05 | readout_action_vector | l0_output | minus_blocker_top1 | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | -0.09375 | -0.044270833333333336 |
| glm4 | l0_output_eos_boost_beta_0.25 | readout_action_vector | l0_output | eos_boost | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | -0.125 | -1.75 |
| glm4 | L36_mlp_output_scale_1.5 | component_output_scale | L36_mlp | None | None | 1.5 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | -0.3125 | -0.25 |
| glm4 | L39_mlp_output_scale_0.5 | component_output_scale | L39_mlp | None | None | 0.5 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | -2.0625 | -1.2838541666666667 |
| glm4 | l0_output_minus_blocker_top1_beta_0.5 | readout_action_vector | l0_output | minus_blocker_top1 | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | -3.0625 | -2.234375 |
| glm4 | l0_output_minus_blocker_top1_beta_0.25 | readout_action_vector | l0_output | minus_blocker_top1 | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | -10.6640625 | -5.282552083333333 |
| glm4 | l0_output_minus_blocker_top3_mean_beta_0.25 | readout_action_vector | l0_output | minus_blocker_top3_mean | 0.25 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | -10.95166015625 | -6.773213704427083 |
| glm4 | l0_output_eos_boost_beta_0.5 | readout_action_vector | l0_output | eos_boost | 0.5 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | -12.16015625 | -7.3983205159505205 |
| glm4 | boundary_precondition_only | boundary_precondition | None | None | None | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | None | None |
| glm4 | diagnostic_mask_boundary_blocker_top1 | logit_mask_diagnostic | None | None | None | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | None | None |
| glm4 | diagnostic_mask_boundary_blocker_top3 | logit_mask_diagnostic | None | None | None | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | None | None |
| glm4 | L39_mlp_eos_minus_blocker_top1_beta_0.05 | readout_action_vector | L39_mlp | eos_minus_blocker_top1 | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.018229166666666668 |
| glm4 | L39_mlp_minus_blocker_top1_beta_0.05 | readout_action_vector | L39_mlp | minus_blocker_top1 | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| glm4 | L39_mlp_minus_blocker_top1_beta_0.1 | readout_action_vector | L39_mlp | minus_blocker_top1 | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| glm4 | L39_mlp_minus_blocker_top3_mean_beta_0.05 | readout_action_vector | L39_mlp | minus_blocker_top3_mean | 0.05 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| glm4 | L39_mlp_minus_blocker_top3_mean_beta_0.1 | readout_action_vector | L39_mlp | minus_blocker_top3_mean | 0.1 | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
