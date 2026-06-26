# Phase 655 Final Token Policy Decomposition

离线分解 Phase 654 中 rank<=15 但 exact=false 的 bridge failures，查看 correct_prefix 被哪些 final-token policy groups 压过。

## qwen3

- bridge_failures: 108

### By Mode

| pair_task | eval_task | direction | site | n | mean_rank | exact | tok0 | support_no_gen | top0_category | winner_vs_prefix | mean prefix-minus-group |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| explanation_required | explanation_required |  |  | 20 | 1.80 | 9/20 | 10/20 | 0/20 | {'explanation': 10, 'correct_prefix': 10} | {} | space:1.96, newline:2.18, explanation:-0.28, word:1.27, punctuation:4.22, symbol:5.04, correct_prefix:0.00 |
| explanation_required | explanation_required | value_to_task | early_peak_layer_out | 20 | 4.35 | 7/20 | 8/20 | 12/20 | {'space': 11, 'correct_prefix': 8, 'word': 1} | {'space': 10, 'correct_prefix': 1, 'word': 1} | space:-1.02, newline:0.39, explanation:0.16, word:0.76, punctuation:1.99, symbol:3.28, correct_prefix:0.00 |
| explanation_required | explanation_required | value_to_task | separator_input_edge | 20 | 3.10 | 4/20 | 4/20 | 16/20 | {'explanation': 12, 'space': 4, 'correct_prefix': 4} | {'explanation': 11, 'space': 4, 'correct_prefix': 1} | space:-0.06, newline:0.51, explanation:-0.89, word:1.14, punctuation:2.73, symbol:4.55, correct_prefix:0.00 |
| explanation_required | short_value_allowed |  |  | 20 | 9.20 | 0/20 | 0/20 | 0/20 | {'space': 20} | {} | space:-4.96, newline:-2.91, explanation:-1.43, word:-0.03, punctuation:0.31, symbol:2.30, correct_prefix:0.00 |
| explanation_required | short_value_allowed | task_to_value | early_peak_layer_out | 20 | 6.85 | 5/20 | 5/20 | 14/20 | {'space': 15, 'correct_prefix': 5} | {'space': 14} | space:-3.13, newline:-1.32, explanation:-1.47, word:0.52, punctuation:2.13, symbol:2.83, correct_prefix:0.00 |
| explanation_required | short_value_allowed | task_to_value | separator_input_edge | 20 | 8.60 | 1/20 | 1/20 | 15/20 | {'space': 19, 'correct_prefix': 1} | {'space': 15} | space:-4.73, newline:-2.33, explanation:-1.33, word:0.23, punctuation:1.09, symbol:2.86, correct_prefix:0.00 |
| yes_no_required | short_value_allowed |  |  | 20 | 9.20 | 0/20 | 0/20 | 0/20 | {'space': 20} | {} | space:-4.96, newline:-2.91, explanation:-1.43, word:-0.03, punctuation:0.31, symbol:2.30, correct_prefix:0.00 |
| yes_no_required | short_value_allowed | task_to_value | early_peak_layer_out | 20 | 4.40 | 9/20 | 9/20 | 11/20 | {'space': 10, 'correct_prefix': 9, 'explanation': 1} | {'space': 10, 'explanation': 1} | space:-0.96, newline:0.37, explanation:-0.30, word:2.04, punctuation:1.77, symbol:4.99, correct_prefix:0.00 |
| yes_no_required | short_value_allowed | task_to_value | separator_input_edge | 20 | 7.65 | 4/20 | 4/20 | 16/20 | {'space': 16, 'correct_prefix': 4} | {'space': 16} | space:-3.74, newline:-1.32, explanation:-1.53, word:0.72, punctuation:0.69, symbol:3.06, correct_prefix:0.00 |
| yes_no_required | yes_no_required |  |  | 20 | 12.95 | 0/20 | 0/20 | 0/20 | {'explanation': 20} | {} | space:1.07, newline:-0.76, explanation:-2.92, word:0.60, punctuation:0.69, symbol:-0.88, correct_prefix:0.00 |
| yes_no_required | yes_no_required | value_to_task | early_peak_layer_out | 20 | 4.15 | 5/20 | 5/20 | 15/20 | {'space': 15, 'correct_prefix': 5} | {'space': 15} | space:-1.81, newline:-0.33, explanation:0.49, word:2.52, punctuation:3.69, symbol:1.79, correct_prefix:0.00 |
| yes_no_required | yes_no_required | value_to_task | separator_input_edge | 20 | 1.85 | 11/20 | 11/20 | 9/20 | {'correct_prefix': 11, 'space': 8, 'explanation': 1} | {'space': 8, 'explanation': 1} | space:0.79, newline:1.90, explanation:1.49, word:3.54, punctuation:4.65, symbol:2.79, correct_prefix:0.00 |

### Failure Categories

| top0_category | n | mean_rank | mean_margin_vs_top | tasks | sites |
|---|---:|---:|---:|---|---|
| space | 92 | 6.74 | -3.645 | {'yes_no_required': 49, 'explanation_required': 43} | {'early_peak_layer_out': 49, 'separator_input_edge': 43} |
| explanation | 13 | 3.62 | -1.135 | {'explanation_required': 11, 'yes_no_required': 2} | {'separator_input_edge': 12, 'early_peak_layer_out': 1} |
| correct_prefix | 2 | 1.00 | 0.000 | {'explanation_required': 2} | {'separator_input_edge': 1, 'early_peak_layer_out': 1} |
| word | 1 | 3.00 | -0.750 | {'explanation_required': 1} | {'early_peak_layer_out': 1} |

## glm4

- bridge_failures: 72

### By Mode

| pair_task | eval_task | direction | site | n | mean_rank | exact | tok0 | support_no_gen | top0_category | winner_vs_prefix | mean prefix-minus-group |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| explanation_required | explanation_required |  |  | 20 | 58.30 | 0/20 | 0/20 | 0/20 | {'explanation': 20} | {} | space:-2.27, explanation:-5.24, word:-4.55, punctuation:0.85, correct_prefix:0.00 |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 20 | 2.15 | 5/20 | 5/20 | 15/20 | {'space': 15, 'correct_prefix': 5} | {'space': 15} | space:-0.68, newline:4.25, explanation:3.96, word:0.42, punctuation:3.58, symbol:4.36, correct_prefix:0.00 |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 20 | 2.15 | 5/20 | 5/20 | 15/20 | {'space': 15, 'correct_prefix': 5} | {'space': 15} | space:-0.68, newline:4.25, explanation:3.96, word:0.42, punctuation:3.58, symbol:4.36, correct_prefix:0.00 |
| explanation_required | short_value_allowed |  |  | 20 | 2.20 | 3/20 | 3/20 | 0/20 | {'space': 17, 'correct_prefix': 3} | {} | space:-0.99, newline:4.55, explanation:4.16, word:0.37, punctuation:3.34, symbol:4.13, correct_prefix:0.00 |
| explanation_required | short_value_allowed | task_to_value | l22_peak_layer_out | 20 | 44.15 | 0/20 | 0/20 | 4/20 | {'explanation': 20} | {'explanation': 4} | space:-3.02, explanation:-5.07, word:-4.18, punctuation:-0.08, correct_prefix:0.00 |
| explanation_required | short_value_allowed | task_to_value | late_peak_layer_out | 20 | 44.15 | 0/20 | 0/20 | 4/20 | {'explanation': 20} | {'explanation': 4} | space:-3.02, explanation:-5.07, word:-4.18, punctuation:-0.08, correct_prefix:0.00 |
| yes_no_required | short_value_allowed |  |  | 20 | 2.20 | 3/20 | 3/20 | 0/20 | {'space': 17, 'correct_prefix': 3} | {} | space:-0.99, newline:4.55, explanation:4.16, word:0.37, punctuation:3.34, symbol:4.13, correct_prefix:0.00 |
| yes_no_required | short_value_allowed | task_to_value | l22_peak_layer_out | 20 | 135.25 | 0/20 | 0/20 | 0/20 | {'explanation': 16, 'word': 4} | {} | space:-4.35, newline:-4.96, explanation:-6.66, word:-6.29, punctuation:-5.21, symbol:-5.40 |
| yes_no_required | short_value_allowed | task_to_value | late_peak_layer_out | 20 | 135.25 | 0/20 | 0/20 | 0/20 | {'explanation': 16, 'word': 4} | {} | space:-4.35, newline:-4.96, explanation:-6.66, word:-6.29, punctuation:-5.21, symbol:-5.40 |
| yes_no_required | yes_no_required |  |  | 20 | 188.10 | 0/20 | 0/20 | 0/20 | {'explanation': 20} | {} | space:-3.08, newline:-3.96, explanation:-9.12, word:-3.65, punctuation:-6.11, symbol:-5.39 |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 20 | 3.45 | 3/20 | 3/20 | 17/20 | {'space': 15, 'correct_prefix': 3, 'explanation': 2} | {'space': 15, 'explanation': 2} | space:-1.01, newline:4.17, explanation:0.29, word:0.22, punctuation:2.85, symbol:4.69, correct_prefix:0.00 |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 20 | 3.45 | 3/20 | 3/20 | 17/20 | {'space': 15, 'correct_prefix': 3, 'explanation': 2} | {'space': 15, 'explanation': 2} | space:-1.01, newline:4.17, explanation:0.29, word:0.22, punctuation:2.85, symbol:4.69, correct_prefix:0.00 |

### Failure Categories

| top0_category | n | mean_rank | mean_margin_vs_top | tasks | sites |
|---|---:|---:|---:|---|---|
| space | 60 | 3.20 | -1.217 | {'explanation_required': 30, 'yes_no_required': 30} | {'l22_peak_layer_out': 30, 'late_peak_layer_out': 30} |
| explanation | 12 | 6.50 | -2.396 | {'explanation_required': 8, 'yes_no_required': 4} | {'l22_peak_layer_out': 6, 'late_peak_layer_out': 6} |

## deepseek7b

- bridge_failures: 68

### By Mode

| pair_task | eval_task | direction | site | n | mean_rank | exact | tok0 | support_no_gen | top0_category | winner_vs_prefix | mean prefix-minus-group |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| explanation_required | explanation_required |  |  | 20 | 77.20 | 0/20 | 0/20 | 0/20 | {'word': 10, 'newline': 9, 'explanation': 1} | {} | space:-4.50, newline:-5.06, explanation:-5.15, word:-5.34, punctuation:-2.53, symbol:-2.80, correct_prefix:0.00 |
| explanation_required | explanation_required | value_to_task | l22_peak_layer_out | 20 | 8.30 | 3/20 | 2/20 | 14/20 | {'space': 10, 'newline': 8, 'correct_prefix': 2} | {'space': 10, 'newline': 4} | space:-1.98, newline:-1.92, explanation:-0.26, word:0.62, punctuation:1.31, symbol:1.40, correct_prefix:0.00 |
| explanation_required | explanation_required | value_to_task | late_peak_layer_out | 20 | 8.30 | 3/20 | 2/20 | 14/20 | {'space': 10, 'newline': 8, 'correct_prefix': 2} | {'space': 10, 'newline': 4} | space:-1.98, newline:-1.92, explanation:-0.26, word:0.62, punctuation:1.31, symbol:1.40, correct_prefix:0.00 |
| explanation_required | short_value_allowed |  |  | 20 | 8.00 | 0/20 | 0/20 | 0/20 | {'newline': 10, 'space': 10} | {} | space:-2.08, newline:-2.12, explanation:-0.49, word:0.51, punctuation:1.19, symbol:1.30, correct_prefix:0.00 |
| explanation_required | short_value_allowed | task_to_value | l22_peak_layer_out | 20 | 56.15 | 0/20 | 0/20 | 1/20 | {'word': 11, 'newline': 8, 'explanation': 1} | {'word': 1} | space:-3.97, newline:-4.65, explanation:-4.77, word:-5.00, punctuation:-2.11, symbol:-2.32, correct_prefix:0.00 |
| explanation_required | short_value_allowed | task_to_value | late_peak_layer_out | 20 | 56.15 | 0/20 | 0/20 | 1/20 | {'word': 11, 'newline': 8, 'explanation': 1} | {'word': 1} | space:-3.97, newline:-4.65, explanation:-4.77, word:-5.00, punctuation:-2.11, symbol:-2.32, correct_prefix:0.00 |
| yes_no_required | short_value_allowed |  |  | 20 | 8.00 | 0/20 | 0/20 | 0/20 | {'newline': 10, 'space': 10} | {} | space:-2.08, newline:-2.12, explanation:-0.49, word:0.51, punctuation:1.19, symbol:1.30, correct_prefix:0.00 |
| yes_no_required | short_value_allowed | task_to_value | l22_peak_layer_out | 20 | 61.75 | 0/20 | 0/20 | 4/20 | {'newline': 17, 'explanation': 3} | {'newline': 3, 'explanation': 1} | space:-3.82, newline:-5.28, explanation:-4.78, word:-1.88, punctuation:-2.59, symbol:-2.75, correct_prefix:0.00 |
| yes_no_required | short_value_allowed | task_to_value | late_peak_layer_out | 20 | 61.75 | 0/20 | 0/20 | 4/20 | {'newline': 17, 'explanation': 3} | {'newline': 3, 'explanation': 1} | space:-3.82, newline:-5.28, explanation:-4.78, word:-1.88, punctuation:-2.59, symbol:-2.75, correct_prefix:0.00 |
| yes_no_required | yes_no_required |  |  | 20 | 295.55 | 0/20 | 0/20 | 0/20 | {'explanation': 20} | {} | space:-5.80, newline:-7.69, explanation:-9.17, word:-4.48, punctuation:-4.62, symbol:-5.32, correct_prefix:0.00 |
| yes_no_required | yes_no_required | value_to_task | l22_peak_layer_out | 20 | 13.85 | 0/20 | 0/20 | 15/20 | {'newline': 11, 'space': 9} | {'space': 9, 'newline': 6} | space:-2.59, newline:-2.60, explanation:-0.88, word:-0.03, punctuation:0.58, symbol:0.68, correct_prefix:0.00 |
| yes_no_required | yes_no_required | value_to_task | late_peak_layer_out | 20 | 13.85 | 0/20 | 0/20 | 15/20 | {'newline': 11, 'space': 9} | {'space': 9, 'newline': 6} | space:-2.59, newline:-2.60, explanation:-0.88, word:-0.03, punctuation:0.58, symbol:0.68, correct_prefix:0.00 |

### Failure Categories

| top0_category | n | mean_rank | mean_margin_vs_top | tasks | sites |
|---|---:|---:|---:|---|---|
| space | 38 | 4.37 | -1.566 | {'explanation_required': 20, 'yes_no_required': 18} | {'l22_peak_layer_out': 19, 'late_peak_layer_out': 19} |
| newline | 26 | 8.54 | -2.625 | {'yes_no_required': 18, 'explanation_required': 8} | {'l22_peak_layer_out': 13, 'late_peak_layer_out': 13} |
| explanation | 2 | 10.00 | -3.062 | {'yes_no_required': 2} | {'l22_peak_layer_out': 1, 'late_peak_layer_out': 1} |
| word | 2 | 13.00 | -2.688 | {'explanation_required': 2} | {'l22_peak_layer_out': 1, 'late_peak_layer_out': 1} |
