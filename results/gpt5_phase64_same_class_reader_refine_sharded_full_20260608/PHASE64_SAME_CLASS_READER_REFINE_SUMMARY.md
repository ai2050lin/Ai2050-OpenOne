# Phase64 Same Class Reader Refine Summary

## qwen3

cases=384, rows=5376

| rank | reader | acc | min_ctx | min_variant | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | different_natural_control | 0.9766 | 0.9583 | 0.9635 | 1.0465 | 1.0511 | yes |
| 2 | same_compare_values | 0.6823 | 0.4792 | 0.6042 | 0.6436 | 0.9385 | no |
| 3 | b_eq_a_binary | 0.6719 | 0.5625 | 0.3438 | 1.3577 | 2.3682 | no |
| 4 | same_natural_control | 0.6641 | 0.5000 | 0.3281 | 1.5417 | 2.0677 | no |
| 5 | different_option_line | 0.5781 | 0.5000 | 0.1875 | 0.1104 | 1.1084 | no |
| 6 | c_eq_a_binary | 0.5729 | 0.5000 | 0.1458 | 0.9409 | 2.3094 | no |
| 7 | different_json_min | 0.5573 | 0.4792 | 0.4844 | 0.0996 | 0.5449 | no |
| 8 | same_json_min | 0.5365 | 0.4688 | 0.3854 | 0.0457 | 0.6294 | no |
| 9 | same_option_line | 0.5156 | 0.5000 | 0.0312 | 0.3877 | 1.6449 | no |
| 10 | different_compare_values | 0.5078 | 0.4375 | 0.1458 | 0.0667 | 0.9893 | no |
| 11 | different_key_letter | 0.5052 | 0.5000 | 0.0104 | 0.6505 | 2.6072 | no |
| 12 | different_key_space | 0.5000 | 0.5000 | 0.0000 | 0.4785 | 2.8213 | no |
| 13 | same_key_space | 0.5000 | 0.5000 | 0.0000 | -0.1305 | 3.3701 | no |
| 14 | same_key_letter | 0.5000 | 0.5000 | 0.0000 | -0.2254 | 3.2858 | no |

## glm4

cases=384, rows=5376

| rank | reader | acc | min_ctx | min_variant | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | same_compare_values | 0.6198 | 0.5000 | 0.2760 | 0.2995 | 0.8506 | no |
| 2 | c_eq_a_binary | 0.5781 | 0.4375 | 0.4271 | 0.0960 | 0.2757 | no |
| 3 | same_option_line | 0.5417 | 0.5000 | 0.0885 | 0.1647 | 0.8018 | no |
| 4 | b_eq_a_binary | 0.5365 | 0.4896 | 0.0990 | 0.1304 | 0.4279 | no |
| 5 | different_option_line | 0.5026 | 0.5000 | 0.0052 | -0.0076 | 0.6852 | no |
| 6 | different_key_letter | 0.5000 | 0.5000 | 0.0000 | 0.0640 | 2.4909 | no |
| 7 | same_natural_control | 0.5000 | 0.5000 | 0.0000 | 0.0417 | 0.7979 | no |
| 8 | same_key_space | 0.5000 | 0.5000 | 0.0000 | 0.0397 | 1.4033 | no |
| 9 | same_key_letter | 0.5000 | 0.5000 | 0.0000 | 0.0203 | 2.0227 | no |
| 10 | different_natural_control | 0.5000 | 0.5000 | 0.0000 | 0.0119 | 0.6294 | no |
| 11 | different_compare_values | 0.4974 | 0.4062 | 0.1198 | 0.0967 | 0.7933 | no |
| 12 | different_key_space | 0.4896 | 0.4583 | 0.0000 | 0.0378 | 1.5644 | no |
| 13 | different_json_min | 0.4870 | 0.4271 | 0.1146 | 0.0486 | 1.0408 | no |
| 14 | same_json_min | 0.4635 | 0.4062 | 0.2396 | -0.0375 | 0.8969 | no |

## deepseek7b

cases=384, rows=5376

| rank | reader | acc | min_ctx | min_variant | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | b_eq_a_binary | 0.5885 | 0.5000 | 0.2188 | 0.2795 | 1.0750 | no |
| 2 | same_compare_values | 0.5339 | 0.5000 | 0.1042 | 0.2573 | 1.1395 | no |
| 3 | different_option_line | 0.5156 | 0.4583 | 0.1094 | -0.0218 | 1.1341 | no |
| 4 | c_eq_a_binary | 0.5130 | 0.5000 | 0.0729 | 0.1055 | 1.2549 | no |
| 5 | different_json_min | 0.5026 | 0.4583 | 0.5000 | 0.0539 | 0.7706 | no |
| 6 | different_natural_control | 0.5000 | 0.5000 | 0.0000 | 0.2461 | 2.1377 | no |
| 7 | same_natural_control | 0.5000 | 0.5000 | 0.0000 | 0.1348 | 3.1647 | no |
| 8 | different_key_letter | 0.5000 | 0.5000 | 0.0000 | 0.0660 | 2.4771 | no |
| 9 | same_key_space | 0.5000 | 0.5000 | 0.0000 | -0.0130 | 3.5905 | no |
| 10 | same_key_letter | 0.5000 | 0.5000 | 0.0000 | -0.0280 | 4.2774 | no |
| 11 | different_key_space | 0.5000 | 0.4896 | 0.0104 | 0.0449 | 1.5758 | no |
| 12 | same_json_min | 0.4922 | 0.4688 | 0.3594 | -0.0297 | 1.2401 | no |
| 13 | different_compare_values | 0.4766 | 0.4167 | 0.0990 | 0.0029 | 1.1162 | no |
| 14 | same_option_line | 0.4714 | 0.4271 | 0.1250 | 0.1182 | 1.0719 | no |

## Cross Model

| rank | reader | mean_acc | min_acc | min_ctx | min_variant | all_pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | b_eq_a_binary | 0.5990 | 0.5365 | 0.4896 | 0.0990 | no |
| 2 | same_compare_values | 0.6120 | 0.5339 | 0.4792 | 0.1042 | no |
| 3 | c_eq_a_binary | 0.5547 | 0.5130 | 0.4375 | 0.0729 | no |
| 4 | different_option_line | 0.5321 | 0.5026 | 0.4583 | 0.0052 | no |
| 5 | different_natural_control | 0.6589 | 0.5000 | 0.5000 | 0.0000 | no |
| 6 | same_natural_control | 0.5547 | 0.5000 | 0.5000 | 0.0000 | no |
| 7 | different_key_letter | 0.5017 | 0.5000 | 0.5000 | 0.0000 | no |
| 8 | same_key_letter | 0.5000 | 0.5000 | 0.5000 | 0.0000 | no |
| 9 | same_key_space | 0.5000 | 0.5000 | 0.5000 | 0.0000 | no |
| 10 | different_key_space | 0.4965 | 0.4896 | 0.4583 | 0.0000 | no |
| 11 | different_json_min | 0.5156 | 0.4870 | 0.4271 | 0.1146 | no |
| 12 | different_compare_values | 0.4939 | 0.4766 | 0.4062 | 0.0990 | no |
| 13 | same_option_line | 0.5095 | 0.4714 | 0.4271 | 0.0312 | no |
| 14 | same_json_min | 0.4974 | 0.4635 | 0.4062 | 0.2396 | no |
