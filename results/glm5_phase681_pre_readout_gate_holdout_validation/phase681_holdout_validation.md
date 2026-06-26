# Phase 681 Pre-Readout Gate Holdout Validation

- generated: `2026-06-26 11:47:36`

| model | group | train_fail | test_fail | pre gate | train score | test score | test capture | test false_pos | ref gate | ref test score |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| qwen3 | overall | 14 | 9 | final_norm_input_gap_gt_0 | 0.714 | 0.920 | 1.000 | 0.080 | REF_final_gap_gt_-0.25 | 1.000 |
| qwen3 | factor_isolation | 9 | 9 | final_norm_input_gap_gt_0 | 1.000 | 1.000 | 1.000 | 0.000 | REF_final_gap_gt_-0.6875 | 1.000 |
| qwen3 | same_format_random_value | 5 | 0 | final_norm_input_gap_gt_-46.5 | 0.703 | 0.000 | 0.000 | 0.000 | REF_final_gap_gt_-0.25 | 0.000 |
| glm4 | overall | 23 | 14 | max_layer_gap_lt_-0.5012 | 0.607 | 0.320 | 0.429 | 0.108 | REF_final_gap_gt_0 | 1.000 |
| glm4 | different_value_same_format | 10 | 6 | mid_to_late_shift_lt_-13.3 | 0.857 | 0.444 | 0.833 | 0.389 | REF_final_gap_gt_0 | 1.000 |
| glm4 | factor_isolation | 5 | 6 | final_norm_input_gap_gt_-4.375 | 0.664 | 0.476 | 0.667 | 0.190 | REF_final_gap_gt_-0.03125 | 0.952 |
| glm4 | same_value_different_format | 8 | 2 | final_norm_input_gap_lt_-26.53 | 0.875 | 0.000 | 0.000 | 0.000 | REF_final_gap_gt_0 | 1.000 |
| deepseek7b | overall | 107 | 100 | final_norm_input_rank_gt_8.256e+04 | 0.535 | -0.106 | 0.640 | 0.746 | REF_final_gap_gt_0 | 1.000 |
| deepseek7b | factor_isolation | 12 | 13 | final_norm_input_rank_gt_9.664e+04 | 0.500 | 0.544 | 0.615 | 0.071 | REF_final_gap_gt_0 | 1.000 |
| deepseek7b | same_value_different_format | 26 | 24 | final_norm_input_gap_gt_-118.8 | 0.684 | 0.000 | 1.000 | 1.000 | REF_final_gap_gt_0 | 1.000 |

## Cross-Family Checks

| model | source | target | gate | source_holdout_score | target_score | target_capture | target_false_pos |
|---|---|---|---|---:|---:|---:|---:|
| qwen3 | factor_isolation | same_format_random_value | final_norm_input_gap_gt_0 | 1.000 | 0.200 | 0.200 | 0.000 |
| qwen3 | same_format_random_value | factor_isolation | final_norm_input_gap_gt_-46.5 | 0.000 | 0.000 | 1.000 | 1.000 |
| qwen3 | factor_isolation | different_value_same_format | final_norm_input_gap_gt_0 | 1.000 | -0.042 | 0.000 | 0.042 |
| qwen3 | factor_isolation | same_prefix_different_continuation | final_norm_input_gap_gt_0 | 1.000 | -0.042 | 0.000 | 0.042 |
| qwen3 | factor_isolation | same_value_different_format | final_norm_input_gap_gt_0 | 1.000 | -0.069 | 0.000 | 0.069 |
| qwen3 | same_format_random_value | different_value_same_format | final_norm_input_gap_gt_-46.5 | 0.000 | -0.854 | 0.000 | 0.854 |
| qwen3 | same_format_random_value | same_value_different_format | final_norm_input_gap_gt_-46.5 | 0.000 | -0.958 | 0.000 | 0.958 |
| qwen3 | same_format_random_value | same_prefix_different_continuation | final_norm_input_gap_gt_-46.5 | 0.000 | -1.000 | 0.000 | 1.000 |
| glm4 | same_value_different_format | different_value_same_format | final_norm_input_gap_lt_-26.53 | 0.000 | 0.656 | 0.938 | 0.281 |
| glm4 | different_value_same_format | same_value_different_format | mid_to_late_shift_lt_-13.3 | 0.444 | 0.487 | 0.800 | 0.313 |
| glm4 | same_value_different_format | factor_isolation | final_norm_input_gap_lt_-26.53 | 0.000 | 0.042 | 0.182 | 0.140 |
| glm4 | same_value_different_format | same_prefix_different_continuation | final_norm_input_gap_lt_-26.53 | 0.000 | 0.000 | 0.000 | 0.000 |
| glm4 | factor_isolation | same_format_random_value | final_norm_input_gap_gt_-4.375 | 0.476 | -0.083 | 0.000 | 0.083 |
| glm4 | different_value_same_format | same_prefix_different_continuation | mid_to_late_shift_lt_-13.3 | 0.444 | -0.167 | 0.000 | 0.167 |
| glm4 | factor_isolation | same_prefix_different_continuation | final_norm_input_gap_gt_-4.375 | 0.476 | -0.375 | 0.000 | 0.375 |
| glm4 | different_value_same_format | factor_isolation | mid_to_late_shift_lt_-13.3 | 0.444 | -0.376 | 0.182 | 0.558 |
| glm4 | factor_isolation | same_value_different_format | final_norm_input_gap_gt_-4.375 | 0.476 | -0.412 | 0.200 | 0.612 |
| glm4 | factor_isolation | different_value_same_format | final_norm_input_gap_gt_-4.375 | 0.476 | -0.594 | 0.062 | 0.656 |
| glm4 | same_value_different_format | same_format_random_value | final_norm_input_gap_lt_-26.53 | 0.000 | -0.597 | 0.000 | 0.597 |
| glm4 | different_value_same_format | same_format_random_value | mid_to_late_shift_lt_-13.3 | 0.444 | -0.778 | 0.000 | 0.778 |
| deepseek7b | same_value_different_format | same_prefix_different_continuation | final_norm_input_gap_gt_-118.8 | 0.000 | 0.870 | 0.870 | 0.000 |
| deepseek7b | factor_isolation | different_value_same_format | final_norm_input_rank_gt_9.664e+04 | 0.544 | 0.565 | 0.565 | 0.000 |
| deepseek7b | same_value_different_format | same_format_random_value | final_norm_input_gap_gt_-118.8 | 0.000 | 0.540 | 0.651 | 0.111 |
| deepseek7b | same_value_different_format | different_value_same_format | final_norm_input_gap_gt_-118.8 | 0.000 | 0.435 | 0.935 | 0.500 |
| deepseek7b | same_value_different_format | factor_isolation | final_norm_input_gap_gt_-118.8 | 0.000 | 0.408 | 0.960 | 0.552 |
| deepseek7b | factor_isolation | same_prefix_different_continuation | final_norm_input_rank_gt_9.664e+04 | 0.544 | 0.348 | 0.348 | 0.000 |
| deepseek7b | factor_isolation | same_value_different_format | final_norm_input_rank_gt_9.664e+04 | 0.544 | 0.289 | 0.800 | 0.511 |
| deepseek7b | factor_isolation | same_format_random_value | final_norm_input_rank_gt_9.664e+04 | 0.544 | 0.238 | 0.238 | 0.000 |
