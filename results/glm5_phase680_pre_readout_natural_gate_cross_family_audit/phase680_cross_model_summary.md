# Phase 680 Pre-Readout Natural Gate and Cross-Family Generalization Audit

- generated: `2026-06-26 11:45:53`

| model | cases | top1_rate | failures | best pre-readout gate | pre score | pre capture | pre false_pos | best near-readout ref | ref score |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| deepseek7b | 342 | 0.395 | 207 | final_norm_input_gap_gt_-118.8 | 0.280 | 0.850 | 0.570 | REF_final_gap_gt_0 | 1.000 |
| glm4 | 342 | 0.892 | 37 | final_norm_input_gap_lt_-36.05 | 0.479 | 0.676 | 0.197 | REF_final_gap_gt_0 | 1.000 |
| qwen3 | 342 | 0.933 | 23 | final_norm_input_gap_gt_0 | 0.785 | 0.826 | 0.041 | REF_final_gap_gt_-0.25 | 1.000 |

## Family Baseline

### deepseek7b

| family | n | top1_rate | failure_rate | mean_rank | mean_final_gap |
|---|---:|---:|---:|---:|---:|
| different_value_same_format | 48 | 0.042 | 0.958 | 122.79 | 3.938 |
| factor_isolation | 54 | 0.537 | 0.463 | 8.41 | 0.677 |
| same_format_random_value | 72 | 0.125 | 0.875 | 469.79 | 6.301 |
| same_prefix_different_continuation | 24 | 0.042 | 0.958 | 10.08 | 2.346 |
| same_value_different_format | 144 | 0.653 | 0.347 | 39.71 | -0.117 |

### glm4

| family | n | top1_rate | failure_rate | mean_rank | mean_final_gap |
|---|---:|---:|---:|---:|---:|
| different_value_same_format | 48 | 0.667 | 0.333 | 1.52 | -0.353 |
| factor_isolation | 54 | 0.796 | 0.204 | 1.20 | -0.649 |
| same_format_random_value | 72 | 1.000 | 0.000 | 1.00 | -3.329 |
| same_prefix_different_continuation | 24 | 1.000 | 0.000 | 1.00 | -2.794 |
| same_value_different_format | 144 | 0.931 | 0.069 | 1.08 | -1.919 |

### qwen3

| family | n | top1_rate | failure_rate | mean_rank | mean_final_gap |
|---|---:|---:|---:|---:|---:|
| different_value_same_format | 48 | 1.000 | 0.000 | 1.00 | -5.492 |
| factor_isolation | 54 | 0.667 | 0.333 | 1.33 | -4.044 |
| same_format_random_value | 72 | 0.931 | 0.069 | 1.49 | -5.341 |
| same_prefix_different_continuation | 24 | 1.000 | 0.000 | 1.00 | -3.609 |
| same_value_different_format | 144 | 1.000 | 0.000 | 1.00 | -8.029 |

