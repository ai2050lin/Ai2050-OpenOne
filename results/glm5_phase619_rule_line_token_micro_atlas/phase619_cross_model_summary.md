# Phase 619 Cross Model Summary

Rule-line token micro-atlas for source-localized pattern/content repair.

## qwen3

rows=9, target_seen=9, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 119, 'empty_correct_rule_line': 0}, layers=[27, 28, 29], heads={'27': 32, '28': 32, '29': 32}, top_heads=[11, 23, 6, 14, 5, 2], specs=66, compact=True, time_min=4.14

### best

| rank | name | group | mode | random | ops | slots | switch | margin | correct_delta | wrong_delta |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | False | 3 | 18 | 3/9 | +0.904 | +0.589 | -0.315 |
| 2 | `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | False | 3 | 18 | 3/9 | +0.904 | +0.589 | -0.315 |
| 3 | `L29_H11_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 3/9 | +0.459 | +0.296 | -0.163 |
| 4 | `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | False | 3 | 18 | 3/9 | +0.389 | +0.246 | -0.143 |
| 5 | `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | False | 3 | 18 | 3/9 | +0.389 | +0.246 | -0.143 |
| 6 | `L29_H11_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 2/9 | +0.709 | +0.515 | -0.194 |
| 7 | `L29_H11_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 2/9 | +0.709 | +0.518 | -0.191 |
| 8 | `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | False | 3 | 18 | 2/9 | +0.473 | +0.337 | -0.136 |
| 9 | `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | False | 3 | 18 | 2/9 | +0.473 | +0.337 | -0.136 |
| 10 | `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | False | 3 | 18 | 2/9 | +0.459 | +0.333 | -0.126 |
| 11 | `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | False | 3 | 18 | 2/9 | +0.459 | +0.333 | -0.126 |
| 12 | `L29_H6_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 1/9 | +0.181 | +0.145 | -0.036 |
| 13 | `L29_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 1/9 | +0.139 | +0.112 | -0.027 |
| 14 | `L29_H6_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 1/9 | +0.125 | +0.079 | -0.046 |
| 15 | `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | True | 3 | 18 | 1/9 | +0.099 | +0.061 | -0.038 |
| 16 | `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | True | 3 | 18 | 1/9 | +0.055 | +0.031 | -0.024 |
| 17 | `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | True | 3 | 18 | 1/9 | +0.047 | +0.016 | -0.031 |
| 18 | `L28_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 1/9 | +0.042 | +0.025 | -0.017 |
| 19 | `L28_H6_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 1/9 | +0.041 | +0.027 | -0.014 |
| 20 | `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | True | 3 | 18 | 1/9 | +0.037 | +0.016 | -0.021 |
| 21 | `L28_H11_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | False | 1 | 1 | 1/9 | +0.027 | +0.015 | -0.013 |
| 22 | `L29_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 1/9 | +0.026 | +0.020 | -0.006 |
| 23 | `L29_H23_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | True | 1 | 1 | 1/9 | +0.025 | +0.009 | -0.016 |
| 24 | `L29_H23_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 1/9 | +0.023 | +0.014 | -0.010 |
| 25 | `L27_H23_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | True | 1 | 1 | 1/9 | +0.023 | +0.010 | -0.012 |
| 26 | `L29_H11_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | True | 1 | 1 | 1/9 | +0.016 | +0.006 | -0.010 |
| 27 | `L27_H23_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 1/9 | +0.014 | +0.008 | -0.006 |
| 28 | `L28_H23_correct_category_token_rr` | correct_category_token | rr_pattern_content | False | 1 | 1 | 1/9 | +0.014 | +0.007 | -0.007 |
| 29 | `L27_H11_correct_category_token_rr` | correct_category_token | rr_pattern_content | False | 1 | 1 | 1/9 | +0.014 | +0.006 | -0.008 |
| 30 | `L27_H23_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 1/9 | +0.014 | +0.008 | -0.005 |
| 31 | `L29_H11_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | False | 1 | 1 | 1/9 | +0.014 | +0.005 | -0.009 |
| 32 | `L29_H6_correct_category_token_rr` | correct_category_token | rr_pattern_content | False | 1 | 1 | 1/9 | +0.014 | +0.006 | -0.008 |
| 33 | `L27_H11_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 1/9 | +0.014 | +0.005 | -0.009 |
| 34 | `L29_H23_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 1/9 | +0.014 | +0.003 | -0.010 |
| 35 | `L28_H6_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | True | 1 | 1 | 1/9 | +0.013 | +0.002 | -0.012 |
| 36 | `L27_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 1/9 | +0.012 | +0.004 | -0.009 |
| 37 | `L28_H23_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 1/9 | +0.010 | -0.005 | -0.015 |
| 38 | `L27_H6_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | False | 1 | 1 | 1/9 | -0.000 | -0.003 | -0.003 |
| 39 | `L28_H11_correct_category_token_rr` | correct_category_token | rr_pattern_content | False | 1 | 1 | 1/9 | -0.000 | -0.005 | -0.005 |
| 40 | `L28_H11_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 1/9 | -0.000 | -0.008 | -0.008 |

### micro_real_top_paths

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | 3 | 18 | 3/9 | +0.904 | +0.589 | -0.315 |
| `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | 3 | 18 | 3/9 | +0.904 | +0.589 | -0.315 |
| `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | 3 | 18 | 3/9 | +0.389 | +0.246 | -0.143 |
| `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | 3 | 18 | 3/9 | +0.389 | +0.246 | -0.143 |
| `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | 3 | 18 | 2/9 | +0.473 | +0.337 | -0.136 |
| `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | 3 | 18 | 2/9 | +0.473 | +0.337 | -0.136 |
| `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | 3 | 18 | 2/9 | +0.459 | +0.333 | -0.126 |
| `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | 3 | 18 | 2/9 | +0.459 | +0.333 | -0.126 |
| `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | 3 | 18 | 1/9 | -0.014 | -0.018 | -0.004 |
| `top6_midlate_correct_relation_token_rr_pattern_content` | correct_relation_token | rr_pattern_content | 3 | 18 | 1/9 | -0.014 | -0.018 | -0.004 |
| `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | 3 | 18 | 1/9 | -0.028 | -0.025 | +0.002 |
| `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | 3 | 18 | 1/9 | -0.028 | -0.025 | +0.002 |
| `top6_midlate_all_value_rule_lines_br_content` | all_value_rule_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_category_token_br_content` | correct_category_token | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_br_content` | correct_relation_token | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_rule_line_br_content` | correct_rule_line | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_value_token_br_content` | correct_value_token | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_category_lines_br_content` | wrong_same_category_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_relation_lines_br_content` | wrong_same_relation_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | 3 | 18 | 0/9 | -0.014 | -0.019 | -0.005 |
| `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | 3 | 18 | 0/9 | -0.014 | -0.019 | -0.005 |

### correct_line_vs_parts_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L29_H11_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 2/9 | +0.709 | +0.515 | -0.194 |
| `L29_H11_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 2/9 | +0.709 | +0.518 | -0.191 |
| `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | 3 | 18 | 2/9 | +0.473 | +0.337 | -0.136 |
| `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | 3 | 18 | 2/9 | +0.473 | +0.337 | -0.136 |
| `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | 3 | 18 | 2/9 | +0.459 | +0.333 | -0.126 |
| `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | 3 | 18 | 2/9 | +0.459 | +0.333 | -0.126 |
| `L29_H6_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/9 | +0.181 | +0.145 | -0.036 |
| `L29_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/9 | +0.139 | +0.112 | -0.027 |
| `L28_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/9 | +0.042 | +0.025 | -0.017 |
| `L28_H6_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/9 | +0.041 | +0.027 | -0.014 |
| `L28_H11_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/9 | +0.027 | +0.015 | -0.013 |
| `L27_H23_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.008 | -0.006 |
| `L28_H23_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.007 | -0.007 |
| `L27_H11_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.006 | -0.008 |
| `L29_H11_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.005 | -0.009 |
| `L29_H6_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.006 | -0.008 |
| `L27_H11_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.005 | -0.009 |
| `L29_H23_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.003 | -0.010 |
| `L27_H6_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.003 | -0.003 |
| `L28_H11_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.005 | -0.005 |
| `L28_H23_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.009 | -0.009 |
| `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | 3 | 18 | 1/9 | -0.014 | -0.018 | -0.004 |
| `top6_midlate_correct_relation_token_rr_pattern_content` | correct_relation_token | rr_pattern_content | 3 | 18 | 1/9 | -0.014 | -0.018 | -0.004 |
| `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | 3 | 18 | 1/9 | -0.028 | -0.025 | +0.002 |
| `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | 3 | 18 | 1/9 | -0.028 | -0.025 | +0.002 |
| `L29_H11_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | -0.028 | -0.026 | +0.002 |
| `L27_H11_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | +0.028 | +0.018 | -0.009 |
| `L27_H23_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.013 | -0.001 |
| `L27_H6_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.006 | -0.008 |
| `L28_H11_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | -0.001 | -0.002 |
| `top6_midlate_correct_category_token_br_content` | correct_category_token | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_br_content` | correct_relation_token | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_rule_line_br_content` | correct_rule_line | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_value_token_br_content` | correct_value_token | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L27_H6_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | +0.001 | +0.001 |
| `L27_H23_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | +0.002 | +0.002 |
| `L29_H23_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.005 | -0.005 |
| `L27_H23_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.000 | -0.000 |
| `L28_H6_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.003 | -0.003 |
| `L29_H23_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.006 | -0.006 |
| `L28_H11_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.003 | -0.003 |
| `L28_H6_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.018 | -0.004 |
| `L27_H11_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.014 | +0.000 |
| `L29_H6_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.013 | +0.001 |
| `L29_H23_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/9 | -0.028 | -0.023 | +0.005 |
| `L27_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | -0.042 | -0.038 | +0.003 |
| `L28_H23_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | -0.376 | -0.344 | +0.032 |
| `L28_H23_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/9 | -0.403 | -0.364 | +0.039 |

### wrong_line_controls_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L29_H11_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 3/9 | +0.459 | +0.296 | -0.163 |
| `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | 3 | 18 | 3/9 | +0.389 | +0.246 | -0.143 |
| `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | 3 | 18 | 3/9 | +0.389 | +0.246 | -0.143 |
| `L29_H6_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.125 | +0.079 | -0.046 |
| `L27_H23_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.008 | -0.005 |
| `L28_H11_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.008 | -0.008 |
| `L28_H6_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.002 | -0.001 |
| `L27_H6_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | -0.002 | -0.016 |
| `top6_midlate_wrong_same_category_lines_br_content` | wrong_same_category_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_relation_lines_br_content` | wrong_same_relation_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `L27_H11_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.003 | -0.002 |
| `L29_H23_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.012 | +0.002 |
| `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | 3 | 18 | 0/9 | -0.014 | -0.019 | -0.005 |
| `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | 3 | 18 | 0/9 | -0.014 | -0.019 | -0.005 |
| `L28_H23_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.111 | -0.089 | +0.022 |

### single_head_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L29_H11_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 3/9 | +0.459 | +0.296 | -0.163 |
| `L29_H11_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 2/9 | +0.709 | +0.515 | -0.194 |
| `L29_H11_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 2/9 | +0.709 | +0.518 | -0.191 |
| `L29_H6_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/9 | +0.181 | +0.145 | -0.036 |
| `L29_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/9 | +0.139 | +0.112 | -0.027 |
| `L29_H6_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.125 | +0.079 | -0.046 |
| `L28_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/9 | +0.042 | +0.025 | -0.017 |
| `L28_H6_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/9 | +0.041 | +0.027 | -0.014 |
| `L28_H11_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/9 | +0.027 | +0.015 | -0.013 |
| `L27_H23_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.008 | -0.006 |
| `L28_H23_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.007 | -0.007 |
| `L27_H11_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.006 | -0.008 |
| `L27_H23_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.008 | -0.005 |
| `L29_H11_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.005 | -0.009 |
| `L29_H6_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.006 | -0.008 |
| `L27_H11_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.005 | -0.009 |
| `L29_H23_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/9 | +0.014 | +0.003 | -0.010 |
| `L27_H6_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.003 | -0.003 |
| `L28_H11_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.005 | -0.005 |
| `L28_H11_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.008 | -0.008 |
| `L28_H23_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.009 | -0.009 |
| `L28_H6_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/9 | -0.000 | -0.002 | -0.001 |
| `L29_H11_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/9 | -0.028 | -0.026 | +0.002 |
| `L27_H11_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | +0.028 | +0.018 | -0.009 |
| `L27_H23_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.013 | -0.001 |
| `L27_H6_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | +0.006 | -0.008 |
| `L27_H6_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/9 | +0.014 | -0.002 | -0.016 |
| `L28_H11_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | +0.000 | -0.001 | -0.002 |
| `L27_H6_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | +0.001 | +0.001 |
| `L27_H11_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.003 | -0.002 |
| `L27_H23_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | +0.002 | +0.002 |
| `L29_H23_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.005 | -0.005 |
| `L27_H23_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.000 | -0.000 |
| `L28_H6_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.003 | -0.003 |
| `L29_H23_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.006 | -0.006 |
| `L28_H11_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/9 | -0.000 | -0.003 | -0.003 |
| `L28_H6_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.018 | -0.004 |
| `L29_H23_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.012 | +0.002 |
| `L27_H11_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.014 | +0.000 |
| `L29_H6_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/9 | -0.014 | -0.013 | +0.001 |
| `L29_H23_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/9 | -0.028 | -0.023 | +0.005 |
| `L27_H6_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | -0.042 | -0.038 | +0.003 |
| `L28_H23_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/9 | -0.111 | -0.089 | +0.022 |
| `L28_H23_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/9 | -0.376 | -0.344 | +0.032 |
| `L28_H23_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/9 | -0.403 | -0.364 | +0.039 |

### random_controls

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | 3 | 18 | 1/9 | +0.099 | +0.061 | -0.038 |
| `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | 3 | 18 | 1/9 | +0.055 | +0.031 | -0.024 |
| `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | 3 | 18 | 1/9 | +0.047 | +0.016 | -0.031 |
| `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | 3 | 18 | 1/9 | +0.037 | +0.016 | -0.021 |
| `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | 3 | 18 | 1/9 | -0.040 | -0.055 | -0.015 |
| `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | 3 | 18 | 0/9 | +0.029 | -0.008 | -0.037 |
| `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | 3 | 18 | 0/9 | +0.027 | +0.008 | -0.019 |
| `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | 3 | 18 | 0/9 | +0.019 | +0.011 | -0.008 |
| `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | 3 | 18 | 0/9 | +0.015 | +0.005 | -0.010 |
| `top6_midlate_correct_relation_token_rr_pattern_content` | correct_relation_token | rr_pattern_content | 3 | 18 | 0/9 | +0.014 | +0.014 | -0.000 |
| `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | 3 | 18 | 0/9 | +0.001 | +0.005 | +0.005 |
| `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | 3 | 18 | 0/9 | +0.001 | +0.011 | +0.010 |
| `top6_midlate_all_value_rule_lines_br_content` | all_value_rule_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_category_token_br_content` | correct_category_token | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_br_content` | correct_relation_token | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_rule_line_br_content` | correct_rule_line | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_value_token_br_content` | correct_value_token | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_category_lines_br_content` | wrong_same_category_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_relation_lines_br_content` | wrong_same_relation_lines | br_content | 3 | 18 | 0/9 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | 3 | 18 | 0/9 | -0.001 | -0.003 | -0.002 |
| `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | 3 | 18 | 0/9 | -0.047 | -0.051 | -0.005 |

## glm4

rows=12, target_seen=12, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 116, 'empty_correct_rule_line': 0}, layers=[32, 33, 34], heads={'32': 32, '33': 32, '34': 32}, top_heads=[12, 8, 4, 28, 6, 7], specs=66, compact=True, time_min=6.88

### best

| rank | name | group | mode | random | ops | slots | switch | margin | correct_delta | wrong_delta |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | False | 3 | 18 | 2/12 | +0.026 | +0.010 | -0.016 |
| 2 | `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | False | 3 | 18 | 2/12 | +0.026 | +0.010 | -0.016 |
| 3 | `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | True | 3 | 18 | 1/12 | +0.038 | +0.008 | -0.030 |
| 4 | `L34_H8_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 1/12 | +0.026 | +0.012 | -0.014 |
| 5 | `L32_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 1/12 | +0.026 | +0.015 | -0.011 |
| 6 | `L34_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 1/12 | +0.010 | +0.009 | -0.001 |
| 7 | `L34_H4_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 1/12 | +0.008 | +0.005 | -0.003 |
| 8 | `L34_H4_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | False | 1 | 1 | 1/12 | +0.005 | +0.001 | -0.004 |
| 9 | `L34_H8_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | False | 1 | 1 | 1/12 | +0.000 | +0.001 | +0.001 |
| 10 | `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | True | 3 | 18 | 1/12 | -0.004 | +0.001 | +0.004 |
| 11 | `L33_H4_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | True | 1 | 1 | 1/12 | -0.004 | +0.001 | +0.005 |
| 12 | `L33_H8_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | True | 1 | 1 | 1/12 | -0.005 | -0.006 | -0.001 |
| 13 | `L33_H12_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | True | 1 | 1 | 1/12 | -0.005 | -0.004 | +0.001 |
| 14 | `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | True | 3 | 18 | 1/12 | -0.005 | +0.001 | +0.006 |
| 15 | `L32_H8_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 1/12 | -0.005 | -0.005 | -0.000 |
| 16 | `L33_H4_correct_category_token_rr` | correct_category_token | rr_pattern_content | True | 1 | 1 | 1/12 | -0.006 | -0.002 | +0.004 |
| 17 | `L34_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | True | 1 | 1 | 1/12 | -0.006 | -0.003 | +0.004 |
| 18 | `L33_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 1/12 | -0.009 | -0.008 | +0.001 |
| 19 | `L32_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 1/12 | -0.010 | -0.005 | +0.005 |
| 20 | `L33_H12_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | True | 1 | 1 | 1/12 | -0.011 | -0.008 | +0.003 |
| 21 | `L34_H4_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | True | 1 | 1 | 1/12 | -0.013 | -0.008 | +0.005 |
| 22 | `L34_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 1/12 | -0.013 | -0.013 | +0.000 |
| 23 | `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | False | 3 | 18 | 1/12 | -0.021 | -0.037 | -0.016 |
| 24 | `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | False | 3 | 18 | 1/12 | -0.021 | -0.037 | -0.016 |
| 25 | `L34_H12_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 0/12 | +0.026 | +0.017 | -0.009 |
| 26 | `L34_H12_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.016 | +0.011 | -0.005 |
| 27 | `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | True | 3 | 18 | 0/12 | +0.013 | +0.008 | -0.005 |
| 28 | `L34_H4_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.010 | +0.011 | +0.000 |
| 29 | `L32_H4_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | False | 1 | 1 | 0/12 | +0.010 | +0.005 | -0.005 |
| 30 | `L34_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 0/12 | +0.010 | +0.005 | -0.005 |
| 31 | `L32_H8_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | False | 1 | 1 | 0/12 | +0.005 | +0.005 | -0.000 |
| 32 | `L34_H8_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 0/12 | +0.005 | +0.004 | -0.001 |
| 33 | `L34_H12_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 0/12 | +0.005 | +0.006 | +0.001 |
| 34 | `L34_H4_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 0/12 | +0.005 | +0.006 | +0.001 |
| 35 | `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | True | 3 | 18 | 0/12 | +0.003 | +0.010 | +0.008 |
| 36 | `L34_H8_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | True | 1 | 1 | 0/12 | +0.002 | +0.001 | -0.001 |
| 37 | `L32_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | True | 1 | 1 | 0/12 | +0.001 | -0.014 | -0.016 |
| 38 | `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | True | 3 | 18 | 0/12 | +0.001 | +0.019 | +0.017 |
| 39 | `L34_H4_correct_category_token_rr` | correct_category_token | rr_pattern_content | True | 1 | 1 | 0/12 | +0.001 | +0.000 | -0.001 |
| 40 | `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | True | 3 | 18 | 0/12 | +0.001 | +0.003 | +0.002 |

### micro_real_top_paths

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | 3 | 18 | 2/12 | +0.026 | +0.010 | -0.016 |
| `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | 3 | 18 | 2/12 | +0.026 | +0.010 | -0.016 |
| `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | 3 | 18 | 1/12 | -0.021 | -0.037 | -0.016 |
| `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | 3 | 18 | 1/12 | -0.021 | -0.037 | -0.016 |
| `top6_midlate_all_value_rule_lines_br_content` | all_value_rule_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_category_token_br_content` | correct_category_token | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_br_content` | correct_relation_token | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_rule_line_br_content` | correct_rule_line | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_value_token_br_content` | correct_value_token | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_category_lines_br_content` | wrong_same_category_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_relation_lines_br_content` | wrong_same_relation_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | 3 | 18 | 0/12 | -0.005 | -0.003 | +0.002 |
| `top6_midlate_correct_relation_token_rr_pattern_content` | correct_relation_token | rr_pattern_content | 3 | 18 | 0/12 | -0.005 | -0.003 | +0.002 |
| `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | 3 | 18 | 0/12 | -0.031 | -0.022 | +0.009 |
| `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | 3 | 18 | 0/12 | -0.031 | -0.022 | +0.009 |
| `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | 3 | 18 | 0/12 | -0.036 | -0.028 | +0.008 |
| `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | 3 | 18 | 0/12 | -0.036 | -0.028 | +0.008 |
| `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | 3 | 18 | 0/12 | -0.036 | -0.018 | +0.019 |
| `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | 3 | 18 | 0/12 | -0.036 | -0.018 | +0.019 |
| `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | 3 | 18 | 0/12 | -0.036 | -0.031 | +0.005 |
| `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | 3 | 18 | 0/12 | -0.036 | -0.031 | +0.005 |

### correct_line_vs_parts_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L34_H4_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/12 | +0.005 | +0.001 | -0.004 |
| `L34_H8_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/12 | +0.000 | +0.001 | +0.001 |
| `L32_H8_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/12 | -0.005 | -0.005 | -0.000 |
| `L32_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/12 | -0.010 | -0.005 | +0.005 |
| `L34_H12_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | +0.026 | +0.017 | -0.009 |
| `L34_H12_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | +0.016 | +0.011 | -0.005 |
| `L34_H4_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.011 | +0.000 |
| `L32_H4_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.005 | -0.005 |
| `L34_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.005 | -0.005 |
| `L32_H8_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.005 | -0.000 |
| `L34_H8_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.004 | -0.001 |
| `L34_H4_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.006 | +0.001 |
| `L33_H12_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.001 | +0.001 |
| `L33_H8_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.002 | +0.002 |
| `L33_H8_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L34_H12_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | -0.000 | -0.000 |
| `L34_H4_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | -0.001 | -0.001 |
| `L33_H12_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.001 | +0.001 |
| `top6_midlate_correct_category_token_br_content` | correct_category_token | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_br_content` | correct_relation_token | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_rule_line_br_content` | correct_rule_line | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_value_token_br_content` | correct_value_token | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L32_H4_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.000 | -0.001 | -0.001 |
| `L33_H4_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.003 | +0.003 |
| `L34_H12_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.003 | +0.002 |
| `L34_H8_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.002 | +0.003 |
| `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | 3 | 18 | 0/12 | -0.005 | -0.003 | +0.002 |
| `top6_midlate_correct_relation_token_rr_pattern_content` | correct_relation_token | rr_pattern_content | 3 | 18 | 0/12 | -0.005 | -0.003 | +0.002 |
| `L32_H12_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | +0.001 | +0.006 |
| `L32_H12_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.007 | +0.003 |
| `L33_H12_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.005 | +0.005 |
| `L33_H12_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.005 | +0.005 |
| `L33_H4_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.003 | +0.007 |
| `L33_H4_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.006 | +0.004 |
| `L32_H12_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.008 | +0.007 |
| `L32_H8_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.008 | +0.008 |
| `L32_H12_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.004 | +0.012 |
| `L33_H4_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.021 | -0.006 | +0.015 |
| `L33_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.026 | -0.016 | +0.010 |
| `L33_H8_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.026 | -0.014 | +0.012 |
| `L32_H4_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.036 | -0.033 | +0.003 |
| `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | 3 | 18 | 0/12 | -0.036 | -0.028 | +0.008 |
| `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | 3 | 18 | 0/12 | -0.036 | -0.028 | +0.008 |
| `L32_H4_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.036 | -0.035 | +0.002 |
| `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | 3 | 18 | 0/12 | -0.036 | -0.018 | +0.019 |
| `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | 3 | 18 | 0/12 | -0.036 | -0.018 | +0.019 |
| `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | 3 | 18 | 0/12 | -0.036 | -0.031 | +0.005 |
| `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | 3 | 18 | 0/12 | -0.036 | -0.031 | +0.005 |

### wrong_line_controls_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | 3 | 18 | 2/12 | +0.026 | +0.010 | -0.016 |
| `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | 3 | 18 | 2/12 | +0.026 | +0.010 | -0.016 |
| `L34_H8_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/12 | +0.026 | +0.012 | -0.014 |
| `L32_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/12 | +0.026 | +0.015 | -0.011 |
| `L34_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/12 | +0.010 | +0.009 | -0.001 |
| `L34_H12_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.006 | +0.001 |
| `top6_midlate_wrong_same_category_lines_br_content` | wrong_same_category_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_relation_lines_br_content` | wrong_same_relation_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L32_H12_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.002 | +0.003 |
| `L33_H12_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.001 | +0.009 |
| `L32_H8_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.005 | +0.005 |
| `L33_H8_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.003 | +0.013 |
| `L33_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.010 | +0.006 |
| `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | 3 | 18 | 0/12 | -0.031 | -0.022 | +0.009 |
| `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | 3 | 18 | 0/12 | -0.031 | -0.022 | +0.009 |

### single_head_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L34_H8_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/12 | +0.026 | +0.012 | -0.014 |
| `L32_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/12 | +0.026 | +0.015 | -0.011 |
| `L34_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/12 | +0.010 | +0.009 | -0.001 |
| `L34_H4_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/12 | +0.005 | +0.001 | -0.004 |
| `L34_H8_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/12 | +0.000 | +0.001 | +0.001 |
| `L32_H8_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/12 | -0.005 | -0.005 | -0.000 |
| `L32_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/12 | -0.010 | -0.005 | +0.005 |
| `L34_H12_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | +0.026 | +0.017 | -0.009 |
| `L34_H12_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | +0.016 | +0.011 | -0.005 |
| `L34_H4_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.011 | +0.000 |
| `L32_H4_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.005 | -0.005 |
| `L34_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | +0.010 | +0.005 | -0.005 |
| `L32_H8_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.005 | -0.000 |
| `L34_H8_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.004 | -0.001 |
| `L34_H12_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.006 | +0.001 |
| `L34_H4_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | +0.005 | +0.006 | +0.001 |
| `L33_H12_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.001 | +0.001 |
| `L33_H8_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.002 | +0.002 |
| `L33_H8_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.000 | +0.000 |
| `L34_H12_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | -0.000 | -0.000 |
| `L34_H4_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | -0.001 | -0.001 |
| `L33_H12_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | +0.000 | +0.001 | +0.001 |
| `L32_H4_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.000 | -0.001 | -0.001 |
| `L32_H12_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.002 | +0.003 |
| `L33_H4_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.003 | +0.003 |
| `L34_H12_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.003 | +0.002 |
| `L34_H8_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | -0.002 | +0.003 |
| `L32_H12_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | -0.005 | +0.001 | +0.006 |
| `L32_H12_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.007 | +0.003 |
| `L33_H12_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.005 | +0.005 |
| `L33_H12_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.001 | +0.009 |
| `L32_H8_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.005 | +0.005 |
| `L33_H12_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.005 | +0.005 |
| `L33_H4_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.003 | +0.007 |
| `L33_H4_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/12 | -0.010 | -0.006 | +0.004 |
| `L32_H12_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.008 | +0.007 |
| `L33_H8_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.003 | +0.013 |
| `L32_H8_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.008 | +0.008 |
| `L32_H12_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.004 | +0.012 |
| `L33_H4_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/12 | -0.016 | -0.010 | +0.006 |
| `L33_H4_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.021 | -0.006 | +0.015 |
| `L33_H8_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.026 | -0.016 | +0.010 |
| `L33_H8_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.026 | -0.014 | +0.012 |
| `L32_H4_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/12 | -0.036 | -0.033 | +0.003 |
| `L32_H4_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/12 | -0.036 | -0.035 | +0.002 |

### random_controls

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | 3 | 18 | 1/12 | +0.038 | +0.008 | -0.030 |
| `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | 3 | 18 | 1/12 | -0.004 | +0.001 | +0.004 |
| `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | 3 | 18 | 1/12 | -0.005 | +0.001 | +0.006 |
| `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | 3 | 18 | 0/12 | +0.013 | +0.008 | -0.005 |
| `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | 3 | 18 | 0/12 | +0.003 | +0.010 | +0.008 |
| `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | 3 | 18 | 0/12 | +0.001 | +0.019 | +0.017 |
| `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | 3 | 18 | 0/12 | +0.001 | +0.003 | +0.002 |
| `top6_midlate_all_value_rule_lines_br_content` | all_value_rule_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_category_token_br_content` | correct_category_token | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_br_content` | correct_relation_token | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_rule_line_br_content` | correct_rule_line | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_value_token_br_content` | correct_value_token | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_category_lines_br_content` | wrong_same_category_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_relation_lines_br_content` | wrong_same_relation_lines | br_content | 3 | 18 | 0/12 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | 3 | 18 | 0/12 | -0.001 | -0.002 | -0.001 |
| `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | 3 | 18 | 0/12 | -0.004 | -0.009 | -0.006 |
| `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | 3 | 18 | 0/12 | -0.007 | -0.001 | +0.006 |
| `top6_midlate_correct_relation_token_rr_pattern_content` | correct_relation_token | rr_pattern_content | 3 | 18 | 0/12 | -0.010 | -0.009 | +0.001 |
| `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | 3 | 18 | 0/12 | -0.011 | -0.003 | +0.008 |
| `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | 3 | 18 | 0/12 | -0.014 | -0.004 | +0.010 |
| `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | 3 | 18 | 0/12 | -0.015 | -0.008 | +0.008 |

## deepseek7b

rows=43, target_seen=43, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 85, 'empty_correct_rule_line': 0}, layers=[20, 21, 22], heads={'20': 28, '21': 28, '22': 28}, top_heads=[3, 1, 7, 24, 25, 13], specs=66, compact=True, time_min=20.49

### best

| rank | name | group | mode | random | ops | slots | switch | margin | correct_delta | wrong_delta |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | False | 3 | 18 | 32/43 | +1.735 | +1.092 | -0.643 |
| 2 | `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | False | 3 | 18 | 32/43 | +1.735 | +1.092 | -0.643 |
| 3 | `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | False | 3 | 18 | 24/43 | +1.227 | +0.917 | -0.309 |
| 4 | `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | False | 3 | 18 | 24/43 | +1.227 | +0.917 | -0.309 |
| 5 | `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | False | 3 | 18 | 24/43 | +1.194 | +0.901 | -0.293 |
| 6 | `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | False | 3 | 18 | 24/43 | +1.194 | +0.901 | -0.293 |
| 7 | `L22_H1_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 8/43 | +0.515 | +0.423 | -0.092 |
| 8 | `L22_H1_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 8/43 | +0.508 | +0.412 | -0.096 |
| 9 | `L22_H7_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 7/43 | +0.302 | +0.263 | -0.039 |
| 10 | `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | False | 3 | 18 | 5/43 | +0.302 | +0.154 | -0.149 |
| 11 | `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | False | 3 | 18 | 5/43 | +0.302 | +0.154 | -0.149 |
| 12 | `L22_H7_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 5/43 | +0.290 | +0.254 | -0.036 |
| 13 | `L22_H3_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | False | 1 | 1 | 4/43 | +0.228 | +0.187 | -0.041 |
| 14 | `L22_H3_correct_value_token_rr` | correct_value_token | rr_pattern_content | False | 1 | 1 | 3/43 | +0.232 | +0.191 | -0.041 |
| 15 | `L22_H1_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 3/43 | +0.130 | +0.064 | -0.066 |
| 16 | `L22_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | False | 1 | 1 | 2/43 | +0.066 | +0.030 | -0.035 |
| 17 | `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | False | 3 | 18 | 2/43 | +0.064 | +0.022 | -0.042 |
| 18 | `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | False | 3 | 18 | 2/43 | +0.064 | +0.022 | -0.042 |
| 19 | `L22_H1_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 2/43 | +0.002 | -0.006 | -0.009 |
| 20 | `L20_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | True | 1 | 1 | 2/43 | -0.003 | -0.008 | -0.005 |
| 21 | `L20_H7_correct_category_token_rr` | correct_category_token | rr_pattern_content | True | 1 | 1 | 2/43 | -0.009 | -0.004 | +0.005 |
| 22 | `L21_H3_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | False | 1 | 1 | 2/43 | -0.015 | -0.009 | +0.006 |
| 23 | `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | True | 3 | 18 | 2/43 | -0.016 | -0.028 | -0.011 |
| 24 | `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | True | 3 | 18 | 2/43 | -0.020 | -0.041 | -0.020 |
| 25 | `L22_H7_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 1/43 | +0.027 | +0.015 | -0.012 |
| 26 | `L22_H1_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | True | 1 | 1 | 1/43 | +0.009 | -0.009 | -0.018 |
| 27 | `L21_H7_correct_category_token_rr` | correct_category_token | rr_pattern_content | False | 1 | 1 | 1/43 | +0.005 | +0.004 | -0.001 |
| 28 | `L22_H7_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | True | 1 | 1 | 1/43 | +0.005 | +0.001 | -0.004 |
| 29 | `L20_H1_correct_category_token_rr` | correct_category_token | rr_pattern_content | False | 1 | 1 | 1/43 | +0.004 | +0.003 | -0.001 |
| 30 | `L22_H3_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | True | 1 | 1 | 1/43 | +0.002 | -0.002 | -0.004 |
| 31 | `L21_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | True | 1 | 1 | 1/43 | -0.001 | -0.005 | -0.004 |
| 32 | `L22_H1_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | True | 1 | 1 | 1/43 | -0.001 | +0.000 | +0.001 |
| 33 | `L22_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | True | 1 | 1 | 1/43 | -0.001 | +0.003 | +0.004 |
| 34 | `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | True | 3 | 18 | 1/43 | -0.002 | -0.015 | -0.013 |
| 35 | `L22_H1_correct_category_token_rr` | correct_category_token | rr_pattern_content | True | 1 | 1 | 1/43 | -0.002 | -0.005 | -0.002 |
| 36 | `L21_H7_correct_value_token_rr` | correct_value_token | rr_pattern_content | True | 1 | 1 | 1/43 | -0.003 | -0.004 | -0.001 |
| 37 | `L21_H7_correct_category_token_rr` | correct_category_token | rr_pattern_content | True | 1 | 1 | 1/43 | -0.003 | -0.005 | -0.002 |
| 38 | `L22_H7_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | False | 1 | 1 | 1/43 | -0.004 | -0.006 | -0.002 |
| 39 | `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | True | 3 | 18 | 1/43 | -0.006 | +0.000 | +0.006 |
| 40 | `L20_H7_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | True | 1 | 1 | 1/43 | -0.006 | -0.001 | +0.004 |

### micro_real_top_paths

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | 3 | 18 | 32/43 | +1.735 | +1.092 | -0.643 |
| `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | 3 | 18 | 32/43 | +1.735 | +1.092 | -0.643 |
| `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | 3 | 18 | 24/43 | +1.227 | +0.917 | -0.309 |
| `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | 3 | 18 | 24/43 | +1.227 | +0.917 | -0.309 |
| `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | 3 | 18 | 24/43 | +1.194 | +0.901 | -0.293 |
| `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | 3 | 18 | 24/43 | +1.194 | +0.901 | -0.293 |
| `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | 3 | 18 | 5/43 | +0.302 | +0.154 | -0.149 |
| `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | 3 | 18 | 5/43 | +0.302 | +0.154 | -0.149 |
| `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | 3 | 18 | 2/43 | +0.064 | +0.022 | -0.042 |
| `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | 3 | 18 | 2/43 | +0.064 | +0.022 | -0.042 |
| `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | 3 | 18 | 1/43 | -0.008 | -0.007 | +0.001 |
| `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | 3 | 18 | 1/43 | -0.008 | -0.007 | +0.001 |
| `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | 3 | 18 | 1/43 | -0.010 | -0.011 | -0.000 |
| `top6_midlate_correct_relation_token_rr_pattern_content` | correct_relation_token | rr_pattern_content | 3 | 18 | 1/43 | -0.010 | -0.011 | -0.000 |
| `top6_midlate_all_value_rule_lines_br_content` | all_value_rule_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_category_token_br_content` | correct_category_token | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_br_content` | correct_relation_token | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_rule_line_br_content` | correct_rule_line | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_value_token_br_content` | correct_value_token | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_category_lines_br_content` | wrong_same_category_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_relation_lines_br_content` | wrong_same_relation_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |

### correct_line_vs_parts_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | 3 | 18 | 24/43 | +1.227 | +0.917 | -0.309 |
| `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | 3 | 18 | 24/43 | +1.227 | +0.917 | -0.309 |
| `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | 3 | 18 | 24/43 | +1.194 | +0.901 | -0.293 |
| `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | 3 | 18 | 24/43 | +1.194 | +0.901 | -0.293 |
| `L22_H1_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 8/43 | +0.515 | +0.423 | -0.092 |
| `L22_H1_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 8/43 | +0.508 | +0.412 | -0.096 |
| `L22_H7_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 7/43 | +0.302 | +0.263 | -0.039 |
| `L22_H7_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 5/43 | +0.290 | +0.254 | -0.036 |
| `L22_H3_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 4/43 | +0.228 | +0.187 | -0.041 |
| `L22_H3_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 3/43 | +0.232 | +0.191 | -0.041 |
| `L21_H3_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 2/43 | -0.015 | -0.009 | +0.006 |
| `L21_H7_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/43 | +0.005 | +0.004 | -0.001 |
| `L20_H1_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/43 | +0.004 | +0.003 | -0.001 |
| `L22_H7_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/43 | -0.004 | -0.006 | -0.002 |
| `L22_H1_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/43 | -0.008 | -0.008 | -0.000 |
| `L21_H3_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/43 | -0.008 | -0.004 | +0.004 |
| `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | 3 | 18 | 1/43 | -0.008 | -0.007 | +0.001 |
| `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | 3 | 18 | 1/43 | -0.008 | -0.007 | +0.001 |
| `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | 3 | 18 | 1/43 | -0.010 | -0.011 | -0.000 |
| `top6_midlate_correct_relation_token_rr_pattern_content` | correct_relation_token | rr_pattern_content | 3 | 18 | 1/43 | -0.010 | -0.011 | -0.000 |
| `L21_H1_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/43 | -0.011 | -0.007 | +0.004 |
| `L20_H7_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/43 | -0.013 | -0.014 | -0.001 |
| `L20_H1_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/43 | -0.015 | -0.010 | +0.005 |
| `L22_H7_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/43 | -0.016 | -0.009 | +0.006 |
| `L21_H7_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/43 | -0.017 | -0.008 | +0.009 |
| `L21_H1_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/43 | -0.018 | -0.015 | +0.003 |
| `L20_H1_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/43 | -0.020 | -0.011 | +0.009 |
| `L22_H3_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/43 | -0.023 | -0.017 | +0.006 |
| `L21_H7_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/43 | -0.025 | -0.017 | +0.007 |
| `L22_H3_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | +0.006 | +0.007 | +0.000 |
| `L20_H3_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | +0.001 | -0.007 | -0.008 |
| `top6_midlate_correct_category_token_br_content` | correct_category_token | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_br_content` | correct_relation_token | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_rule_line_br_content` | correct_rule_line | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_value_token_br_content` | correct_value_token | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L20_H3_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/43 | -0.001 | -0.007 | -0.006 |
| `L20_H1_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/43 | -0.001 | -0.000 | +0.001 |
| `L21_H1_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/43 | -0.003 | +0.002 | +0.005 |
| `L21_H3_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | -0.004 | -0.005 | -0.001 |
| `L20_H7_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/43 | -0.004 | -0.003 | +0.001 |
| `L21_H3_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/43 | -0.006 | -0.005 | +0.001 |
| `L20_H3_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/43 | -0.007 | -0.007 | -0.000 |
| `L20_H7_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | -0.009 | -0.002 | +0.006 |
| `L22_H1_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | -0.010 | -0.008 | +0.002 |
| `L20_H3_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/43 | -0.012 | -0.008 | +0.004 |
| `L21_H7_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/43 | -0.015 | -0.011 | +0.004 |
| `L20_H7_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/43 | -0.016 | -0.014 | +0.002 |
| `L21_H1_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/43 | -0.017 | -0.015 | +0.002 |

### wrong_line_controls_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | 3 | 18 | 5/43 | +0.302 | +0.154 | -0.149 |
| `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | 3 | 18 | 5/43 | +0.302 | +0.154 | -0.149 |
| `L22_H1_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 3/43 | +0.130 | +0.064 | -0.066 |
| `L22_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 2/43 | +0.066 | +0.030 | -0.035 |
| `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | 3 | 18 | 2/43 | +0.064 | +0.022 | -0.042 |
| `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | 3 | 18 | 2/43 | +0.064 | +0.022 | -0.042 |
| `L21_H7_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.012 | -0.010 | +0.003 |
| `L20_H7_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.013 | -0.011 | +0.002 |
| `L20_H1_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.015 | -0.010 | +0.004 |
| `L21_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.016 | -0.010 | +0.006 |
| `L22_H7_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/43 | +0.048 | +0.027 | -0.021 |
| `L21_H1_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | -0.001 | -0.001 |
| `top6_midlate_wrong_same_category_lines_br_content` | wrong_same_category_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_relation_lines_br_content` | wrong_same_relation_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `L20_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/43 | -0.019 | -0.016 | +0.003 |

### single_head_real

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `L22_H1_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 8/43 | +0.515 | +0.423 | -0.092 |
| `L22_H1_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 8/43 | +0.508 | +0.412 | -0.096 |
| `L22_H7_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 7/43 | +0.302 | +0.263 | -0.039 |
| `L22_H7_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 5/43 | +0.290 | +0.254 | -0.036 |
| `L22_H3_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 4/43 | +0.228 | +0.187 | -0.041 |
| `L22_H3_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 3/43 | +0.232 | +0.191 | -0.041 |
| `L22_H1_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 3/43 | +0.130 | +0.064 | -0.066 |
| `L22_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 2/43 | +0.066 | +0.030 | -0.035 |
| `L21_H3_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 2/43 | -0.015 | -0.009 | +0.006 |
| `L21_H7_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/43 | +0.005 | +0.004 | -0.001 |
| `L20_H1_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/43 | +0.004 | +0.003 | -0.001 |
| `L22_H7_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/43 | -0.004 | -0.006 | -0.002 |
| `L22_H1_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/43 | -0.008 | -0.008 | -0.000 |
| `L21_H3_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/43 | -0.008 | -0.004 | +0.004 |
| `L21_H1_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/43 | -0.011 | -0.007 | +0.004 |
| `L21_H7_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.012 | -0.010 | +0.003 |
| `L20_H7_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.013 | -0.011 | +0.002 |
| `L20_H7_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/43 | -0.013 | -0.014 | -0.001 |
| `L20_H1_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.015 | -0.010 | +0.004 |
| `L20_H1_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 1/43 | -0.015 | -0.010 | +0.005 |
| `L21_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 1/43 | -0.016 | -0.010 | +0.006 |
| `L22_H7_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/43 | -0.016 | -0.009 | +0.006 |
| `L21_H7_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/43 | -0.017 | -0.008 | +0.009 |
| `L21_H1_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 1/43 | -0.018 | -0.015 | +0.003 |
| `L20_H1_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/43 | -0.020 | -0.011 | +0.009 |
| `L22_H3_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 1/43 | -0.023 | -0.017 | +0.006 |
| `L21_H7_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 1/43 | -0.025 | -0.017 | +0.007 |
| `L22_H7_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/43 | +0.048 | +0.027 | -0.021 |
| `L22_H3_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | +0.006 | +0.007 | +0.000 |
| `L20_H3_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | +0.001 | -0.007 | -0.008 |
| `L21_H1_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/43 | +0.000 | -0.001 | -0.001 |
| `L20_H3_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/43 | -0.001 | -0.007 | -0.006 |
| `L20_H1_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/43 | -0.001 | -0.000 | +0.001 |
| `L21_H1_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/43 | -0.003 | +0.002 | +0.005 |
| `L21_H3_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | -0.004 | -0.005 | -0.001 |
| `L20_H7_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/43 | -0.004 | -0.003 | +0.001 |
| `L21_H3_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/43 | -0.006 | -0.005 | +0.001 |
| `L20_H3_correct_rule_line_rr` | correct_rule_line | rr_pattern_content | 1 | 1 | 0/43 | -0.007 | -0.007 | -0.000 |
| `L20_H7_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | -0.009 | -0.002 | +0.006 |
| `L22_H1_correct_category_token_rr` | correct_category_token | rr_pattern_content | 1 | 1 | 0/43 | -0.010 | -0.008 | +0.002 |
| `L20_H3_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/43 | -0.012 | -0.008 | +0.004 |
| `L21_H7_correct_value_token_rr` | correct_value_token | rr_pattern_content | 1 | 1 | 0/43 | -0.015 | -0.011 | +0.004 |
| `L20_H7_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/43 | -0.016 | -0.014 | +0.002 |
| `L21_H1_correct_relation_token_rr` | correct_relation_token | rr_pattern_content | 1 | 1 | 0/43 | -0.017 | -0.015 | +0.002 |
| `L20_H3_wrong_same_relation_lines_rr` | wrong_same_relation_lines | rr_pattern_content | 1 | 1 | 0/43 | -0.019 | -0.016 | +0.003 |

### random_controls

| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `top6_midlate_correct_value_token_rr_pattern_content` | correct_value_token | rr_pattern_content | 3 | 18 | 2/43 | -0.016 | -0.028 | -0.011 |
| `top6_midlate_correct_rule_line_rr_pattern_content` | correct_rule_line | rr_pattern_content | 3 | 18 | 2/43 | -0.020 | -0.041 | -0.020 |
| `top6_midlate_correct_rule_line_rb_pattern` | correct_rule_line | rb_pattern | 3 | 18 | 1/43 | -0.002 | -0.015 | -0.013 |
| `top6_midlate_wrong_same_relation_lines_rr_pattern_content` | wrong_same_relation_lines | rr_pattern_content | 3 | 18 | 1/43 | -0.006 | +0.000 | +0.006 |
| `top6_midlate_correct_category_token_rr_pattern_content` | correct_category_token | rr_pattern_content | 3 | 18 | 1/43 | -0.006 | -0.008 | -0.002 |
| `top6_midlate_wrong_same_relation_lines_rb_pattern` | wrong_same_relation_lines | rb_pattern | 3 | 18 | 1/43 | -0.016 | -0.023 | -0.007 |
| `top6_midlate_wrong_same_category_lines_rb_pattern` | wrong_same_category_lines | rb_pattern | 3 | 18 | 1/43 | -0.022 | -0.018 | +0.004 |
| `top6_midlate_all_value_rule_lines_rb_pattern` | all_value_rule_lines | rb_pattern | 3 | 18 | 0/43 | +0.029 | -0.004 | -0.033 |
| `top6_midlate_all_value_rule_lines_br_content` | all_value_rule_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_category_token_br_content` | correct_category_token | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_br_content` | correct_relation_token | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_rule_line_br_content` | correct_rule_line | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_value_token_br_content` | correct_value_token | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_category_lines_br_content` | wrong_same_category_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_wrong_same_relation_lines_br_content` | wrong_same_relation_lines | br_content | 3 | 18 | 0/43 | +0.000 | +0.000 | +0.000 |
| `top6_midlate_correct_relation_token_rr_pattern_content` | correct_relation_token | rr_pattern_content | 3 | 18 | 0/43 | -0.010 | -0.004 | +0.006 |
| `top6_midlate_all_value_rule_lines_rr_pattern_content` | all_value_rule_lines | rr_pattern_content | 3 | 18 | 0/43 | -0.010 | -0.018 | -0.007 |
| `top6_midlate_wrong_same_category_lines_rr_pattern_content` | wrong_same_category_lines | rr_pattern_content | 3 | 18 | 0/43 | -0.013 | -0.013 | -0.000 |
| `top6_midlate_correct_relation_token_rb_pattern` | correct_relation_token | rb_pattern | 3 | 18 | 0/43 | -0.018 | -0.017 | +0.001 |
| `top6_midlate_correct_category_token_rb_pattern` | correct_category_token | rb_pattern | 3 | 18 | 0/43 | -0.022 | -0.010 | +0.011 |
| `top6_midlate_correct_value_token_rb_pattern` | correct_value_token | rb_pattern | 3 | 18 | 0/43 | -0.049 | -0.051 | -0.002 |
