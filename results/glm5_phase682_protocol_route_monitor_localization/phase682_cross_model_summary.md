# Phase 682 Protocol-Level Failure Monitor Localization

- generated: `2026-06-26 12:00:47`

| model | cases | top1_rate | failures | best protocol gate | score | capture | false_pos | holdout gate | holdout score | holdout capture | holdout false_pos | ref holdout score |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| deepseek7b | 342 | 0.395 | 207 | protocol_final_norm_input_margin_gt_12.5 | 0.394 | 0.720 | 0.326 | protocol_min_margin_gt_-8.531 | 0.020 | 0.640 | 0.620 | 0.930 |
| glm4 | 342 | 0.892 | 37 | protocol_late_shift_lt_16.94 | 0.275 | 1.000 | 0.725 | protocol_min_margin_gt_-0.1787 | 0.201 | 0.500 | 0.299 | 0.643 |
| qwen3 | 342 | 0.933 | 23 | protocol_min_margin_gt_0 | 0.783 | 0.783 | 0.000 | protocol_min_margin_gt_0 | 1.000 | 1.000 | 0.000 | 0.821 |

## Family Baseline

### deepseek7b

| family | n | top1_rate | failure_rate | mean_rank | target_routes | failure_best_other_route |
|---|---:|---:|---:|---:|---|---|
| different_value_same_format | 48 | 0.042 | 0.958 | 122.79 | {'value': 48} | {'prose': 46} |
| factor_isolation | 54 | 0.537 | 0.463 | 8.41 | {'json': 18, 'value': 18, 'yesno': 18} | {'prose': 25} |
| same_format_random_value | 72 | 0.125 | 0.875 | 469.79 | {'value': 72} | {'prose': 63} |
| same_prefix_different_continuation | 24 | 0.042 | 0.958 | 10.08 | {'value': 24} | {'prose': 23} |
| same_value_different_format | 144 | 0.653 | 0.347 | 39.71 | {'json': 24, 'label': 24, 'list': 24, 'prose': 48, 'value': 24} | {'json': 5, 'prose': 45} |

### glm4

| family | n | top1_rate | failure_rate | mean_rank | target_routes | failure_best_other_route |
|---|---:|---:|---:|---:|---|---|
| different_value_same_format | 48 | 0.667 | 0.333 | 1.52 | {'value': 48} | {'continuation': 16} |
| factor_isolation | 54 | 0.796 | 0.204 | 1.20 | {'json': 18, 'value': 18, 'yesno': 18} | {'continuation': 3, 'prose': 8} |
| same_format_random_value | 72 | 1.000 | 0.000 | 1.00 | {'value': 72} | {} |
| same_prefix_different_continuation | 24 | 1.000 | 0.000 | 1.00 | {'value': 24} | {} |
| same_value_different_format | 144 | 0.931 | 0.069 | 1.08 | {'json': 24, 'label': 24, 'list': 24, 'prose': 48, 'value': 24} | {'continuation': 8, 'value': 2} |

### qwen3

| family | n | top1_rate | failure_rate | mean_rank | target_routes | failure_best_other_route |
|---|---:|---:|---:|---:|---|---|
| different_value_same_format | 48 | 1.000 | 0.000 | 1.00 | {'value': 48} | {} |
| factor_isolation | 54 | 0.667 | 0.333 | 1.33 | {'json': 18, 'value': 18, 'yesno': 18} | {'prose': 18} |
| same_format_random_value | 72 | 0.931 | 0.069 | 1.49 | {'value': 72} | {'continuation': 4, 'prose': 1} |
| same_prefix_different_continuation | 24 | 1.000 | 0.000 | 1.00 | {'value': 24} | {} |
| same_value_different_format | 144 | 1.000 | 0.000 | 1.00 | {'json': 24, 'label': 24, 'list': 24, 'prose': 48, 'value': 24} | {} |

