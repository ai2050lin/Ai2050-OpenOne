# Phase 672 Graph Atlas Counterfactual Natural Trajectory Audit

- generated: `2026-06-26 10:36:31`

| model | cases | norm_exact | compact_exact | contains_value | first_top1 | token1 | token2 | mean_rank | mean_margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek7b | 630 | 0.175 | 0.241 | 0.540 | 0.483 | 0.357 | 0.179 | 70.44 | -0.938 |
| glm4 | 630 | 0.543 | 0.648 | 0.668 | 0.913 | 0.592 | 0.344 | 1.11 | 1.888 |
| qwen3 | 630 | 0.568 | 0.700 | 0.710 | 0.963 | 0.546 | 0.343 | 1.09 | 6.843 |

## Family Details

### deepseek7b

| family | n | norm_exact | first_top1 | token1 | mean_rank |
|---|---:|---:|---:|---:|---:|
| different_value_same_format | 48 | 0.042 | 0.042 | 0.000 | 120.21 |
| factor_isolation | 54 | 0.204 | 0.481 | 0.333 | 9.50 |
| same_format_random_value | 72 | 0.111 | 0.111 | 0.111 | 459.96 |
| same_prefix_different_continuation | 24 | 0.042 | 0.042 | 0.042 | 10.42 |
| same_value_different_format | 432 | 0.204 | 0.618 | 0.458 | 10.94 |

### glm4

| family | n | norm_exact | first_top1 | token1 | mean_rank |
|---|---:|---:|---:|---:|---:|
| different_value_same_format | 48 | 0.625 | 0.667 | 0.000 | 1.50 |
| factor_isolation | 54 | 0.481 | 0.852 | 0.333 | 1.15 |
| same_format_random_value | 72 | 1.000 | 1.000 | 1.000 | 1.00 |
| same_prefix_different_continuation | 24 | 0.958 | 1.000 | 0.958 | 1.00 |
| same_value_different_format | 432 | 0.442 | 0.928 | 0.602 | 1.09 |

### qwen3

| family | n | norm_exact | first_top1 | token1 | mean_rank |
|---|---:|---:|---:|---:|---:|
| different_value_same_format | 48 | 1.000 | 1.000 | 0.000 | 1.00 |
| factor_isolation | 54 | 0.333 | 0.667 | 0.333 | 1.33 |
| same_format_random_value | 72 | 0.931 | 0.931 | 0.931 | 1.51 |
| same_prefix_different_continuation | 24 | 1.000 | 1.000 | 1.000 | 1.00 |
| same_value_different_format | 432 | 0.465 | 1.000 | 0.544 | 1.00 |
