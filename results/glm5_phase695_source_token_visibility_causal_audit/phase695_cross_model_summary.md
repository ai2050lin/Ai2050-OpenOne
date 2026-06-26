# Phase 695 Source-Token Visibility Causal Audit

- generated: `2026-06-26 15:39:41`

| model | pairs | rows | best_degrade | drop | patched_top1 | rank_effect | target_effect | best_repair | repair | patched_top1 | rank_effect | target_effect |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| deepseek7b | 72 | 2016 | degradation|terse_no_explain|mask_record_line | 1.000 | 0.000 | 12246.75 | 16.685 | repair|short_only|mask_instruction_line | 0.375 | 0.375 | 156.00 | -13.011 |
| glm4 | 5 | 140 | degradation|terse_no_explain|keep_only_question_answer | 1.000 | 0.000 | 505.40 | 7.234 | repair|short_only|mask_answer_context | 0.600 | 0.600 | 0.60 | -0.390 |
| qwen3 | 3 | 84 | degradation|terse_no_explain|keep_only_question_answer | 1.000 | 0.000 | 345.33 | 38.120 | repair|short_only|mask_object_name | 0.000 | 0.000 | -0.67 | 4.614 |

## Best Degradation

### deepseek7b

| condition | change | baseline_top1 | patched_top1 | base_rank | patched_rank | rank_effect | pmv_effect | target_effect | mask_frac | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|terse_no_explain|mask_record_line | 1.000 | 1.000 | 0.000 | 1.00 | 12247.75 | 12246.75 | 11.955 | 16.685 | 0.558 | {'continuation': 1, 'prose': 71} |
| degradation|terse_no_explain|keep_only_instruction_answer | 1.000 | 1.000 | 0.000 | 1.00 | 7650.38 | 7649.38 | 13.469 | 15.897 | 0.755 | {'continuation': 31, 'prose': 41} |
| degradation|terse_no_explain|keep_only_question_answer | 1.000 | 1.000 | 0.000 | 1.00 | 11316.24 | 11315.24 | 11.237 | 14.428 | 0.766 | {'prose': 71, 'yesno': 1} |
| degradation|terse_no_explain|mask_record_without_target_value | 1.000 | 1.000 | 0.000 | 1.00 | 3175.43 | 3174.43 | 10.177 | 12.092 | 0.479 | {'continuation': 13, 'prose': 59} |
| degradation|terse_no_explain|mask_record_value_object_relation | 0.958 | 1.000 | 0.042 | 1.00 | 3215.71 | 3214.71 | 9.847 | 11.667 | 0.268 | {'continuation': 20, 'prose': 52} |
| degradation|terse_no_explain|mask_relation | 0.917 | 1.000 | 0.083 | 1.00 | 613.96 | 612.96 | 7.071 | 12.357 | 0.038 | {'prose': 72} |
| degradation|terse_no_explain|mask_target_value | 0.903 | 1.000 | 0.097 | 1.00 | 2309.67 | 2308.67 | 10.229 | 18.495 | 0.079 | {'continuation': 9, 'prose': 63} |
| degradation|terse_no_explain|keep_only_record_answer | 0.903 | 1.000 | 0.097 | 1.00 | 652.86 | 651.86 | 7.544 | 10.489 | 0.404 | {'continuation': 19, 'prose': 53} |
| degradation|terse_no_explain|mask_question_line | 0.875 | 1.000 | 0.125 | 1.00 | 1911.17 | 1910.17 | 7.694 | 6.121 | 0.196 | {'continuation': 54, 'prose': 18} |
| degradation|terse_no_explain|keep_only_record_instruction_answer | 0.875 | 1.000 | 0.125 | 1.00 | 1911.17 | 1910.17 | 7.694 | 6.121 | 0.196 | {'continuation': 54, 'prose': 18} |
| degradation|terse_no_explain|mask_record_without_value_object_relation | 0.875 | 1.000 | 0.125 | 1.00 | 959.49 | 958.49 | 6.211 | 3.655 | 0.290 | {'continuation': 38, 'prose': 34} |
| degradation|terse_no_explain|mask_instruction_line | 0.806 | 1.000 | 0.194 | 1.00 | 101.62 | 100.62 | 5.518 | 22.596 | 0.208 | {'prose': 72} |
| degradation|terse_no_explain|mask_answer_context | 0.722 | 1.000 | 0.278 | 1.00 | 241.44 | 240.44 | 3.496 | 3.352 | 0.019 | {'continuation': 1, 'json': 1, 'prose': 70} |
| degradation|terse_no_explain|mask_object_name | 0.403 | 1.000 | 0.597 | 1.00 | 3.93 | 2.93 | 0.812 | -1.978 | 0.216 | {'prose': 72} |

### glm4

| condition | change | baseline_top1 | patched_top1 | base_rank | patched_rank | rank_effect | pmv_effect | target_effect | mask_frac | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|terse_no_explain|keep_only_question_answer | 1.000 | 1.000 | 0.000 | 1.00 | 506.40 | 505.40 | 12.353 | 7.234 | 0.755 | {'prose': 5} |
| degradation|terse_no_explain|mask_record_line | 1.000 | 1.000 | 0.000 | 1.00 | 3827.00 | 3826.00 | 11.461 | 7.078 | 0.531 | {'prose': 5} |
| degradation|terse_no_explain|mask_record_without_target_value | 1.000 | 1.000 | 0.000 | 1.00 | 3821.00 | 3820.00 | 11.645 | 6.900 | 0.510 | {'prose': 5} |
| degradation|terse_no_explain|keep_only_record_answer | 1.000 | 1.000 | 0.000 | 1.00 | 57.60 | 56.60 | 7.575 | 6.813 | 0.429 | {'prose': 5} |
| degradation|terse_no_explain|mask_record_value_object_relation | 1.000 | 1.000 | 0.000 | 1.00 | 54.80 | 53.80 | 6.069 | 6.777 | 0.204 | {'continuation': 3, 'prose': 2} |
| degradation|terse_no_explain|keep_only_instruction_answer | 1.000 | 1.000 | 0.000 | 1.00 | 5864.20 | 5863.20 | 13.126 | 6.680 | 0.735 | {'continuation': 5} |
| degradation|terse_no_explain|mask_target_value | 1.000 | 1.000 | 0.000 | 1.00 | 22.00 | 21.00 | 4.669 | 5.896 | 0.020 | {'prose': 5} |
| degradation|terse_no_explain|mask_question_line | 1.000 | 1.000 | 0.000 | 1.00 | 31.40 | 30.40 | 6.150 | 5.361 | 0.204 | {'continuation': 5} |
| degradation|terse_no_explain|keep_only_record_instruction_answer | 1.000 | 1.000 | 0.000 | 1.00 | 31.40 | 30.40 | 6.150 | 5.361 | 0.204 | {'continuation': 5} |
| degradation|terse_no_explain|mask_instruction_line | 1.000 | 1.000 | 0.000 | 1.00 | 5.80 | 4.80 | 7.987 | 4.872 | 0.224 | {'prose': 5} |
| degradation|terse_no_explain|mask_relation | 1.000 | 1.000 | 0.000 | 1.00 | 8.60 | 7.60 | 4.362 | 3.008 | 0.041 | {'continuation': 4, 'prose': 1} |
| degradation|terse_no_explain|mask_record_without_value_object_relation | 1.000 | 1.000 | 0.000 | 1.00 | 3.40 | 2.40 | 4.525 | 1.132 | 0.327 | {'continuation': 1, 'prose': 4} |
| degradation|terse_no_explain|mask_object_name | 0.200 | 1.000 | 0.800 | 1.00 | 1.20 | 0.20 | -0.006 | 2.958 | 0.204 | {'continuation': 5} |
| degradation|terse_no_explain|mask_answer_context | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.00 | -0.225 | 1.653 | 0.020 | {'continuation': 1, 'json': 1, 'prose': 3} |

### qwen3

| condition | change | baseline_top1 | patched_top1 | base_rank | patched_rank | rank_effect | pmv_effect | target_effect | mask_frac | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|terse_no_explain|keep_only_question_answer | 1.000 | 1.000 | 0.000 | 1.00 | 346.33 | 345.33 | 14.646 | 38.120 | 0.782 | {'prose': 3} |
| degradation|terse_no_explain|mask_record_value_object_relation | 1.000 | 1.000 | 0.000 | 1.00 | 207.67 | 206.67 | 10.667 | 32.682 | 0.291 | {'continuation': 3} |
| degradation|terse_no_explain|mask_target_value | 1.000 | 1.000 | 0.000 | 1.00 | 55.33 | 54.33 | 12.083 | 27.511 | 0.127 | {'continuation': 3} |
| degradation|terse_no_explain|mask_record_line | 1.000 | 1.000 | 0.000 | 1.00 | 432.67 | 431.67 | 11.833 | 26.715 | 0.582 | {'continuation': 3} |
| degradation|terse_no_explain|keep_only_record_answer | 1.000 | 1.000 | 0.000 | 1.00 | 109.33 | 108.33 | 11.979 | 26.467 | 0.382 | {'prose': 3} |
| degradation|terse_no_explain|mask_record_without_target_value | 1.000 | 1.000 | 0.000 | 1.00 | 314.00 | 313.00 | 11.729 | 25.499 | 0.455 | {'prose': 3} |
| degradation|terse_no_explain|keep_only_instruction_answer | 1.000 | 1.000 | 0.000 | 1.00 | 867.00 | 866.00 | 11.812 | 24.935 | 0.764 | {'continuation': 3} |
| degradation|terse_no_explain|mask_instruction_line | 1.000 | 1.000 | 0.000 | 1.00 | 16.00 | 15.00 | 11.896 | 22.233 | 0.200 | {'prose': 3} |
| degradation|terse_no_explain|mask_answer_context | 1.000 | 1.000 | 0.000 | 1.00 | 33.33 | 32.33 | 8.396 | 20.113 | 0.018 | {'continuation': 3} |
| degradation|terse_no_explain|mask_record_without_value_object_relation | 1.000 | 1.000 | 0.000 | 1.00 | 28.33 | 27.33 | 8.021 | 14.141 | 0.291 | {'continuation': 1, 'prose': 2} |
| degradation|terse_no_explain|mask_question_line | 1.000 | 1.000 | 0.000 | 1.00 | 2.67 | 1.67 | 3.583 | 10.442 | 0.182 | {'continuation': 2, 'prose': 1} |
| degradation|terse_no_explain|keep_only_record_instruction_answer | 1.000 | 1.000 | 0.000 | 1.00 | 2.67 | 1.67 | 3.583 | 10.442 | 0.182 | {'continuation': 2, 'prose': 1} |
| degradation|terse_no_explain|mask_relation | 1.000 | 1.000 | 0.000 | 1.00 | 5.00 | 4.00 | 5.250 | 6.087 | 0.036 | {'prose': 3} |
| degradation|terse_no_explain|mask_object_name | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.00 | -1.208 | -5.126 | 0.182 | {'continuation': 3} |


## Best Repair

### deepseek7b

| condition | change | baseline_top1 | patched_top1 | base_rank | patched_rank | rank_effect | pmv_effect | target_effect | mask_frac | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| repair|short_only|mask_instruction_line | 0.375 | 0.000 | 0.375 | 167.69 | 11.69 | 156.00 | 2.481 | -13.011 | 0.160 | {'prose': 72} |
| repair|short_only|mask_answer_context | 0.083 | 0.000 | 0.083 | 167.69 | 580.82 | -413.12 | -0.793 | -0.741 | 0.020 | {'continuation': 2, 'prose': 70} |
| repair|short_only|keep_only_record_answer | 0.083 | 0.000 | 0.083 | 167.69 | 586.53 | -418.83 | -1.102 | -3.967 | 0.368 | {'continuation': 22, 'prose': 50} |
| repair|short_only|mask_object_name | 0.028 | 0.000 | 0.028 | 167.69 | 395.62 | -227.93 | -1.281 | -0.972 | 0.229 | {'continuation': 10, 'prose': 62} |
| repair|short_only|mask_record_without_value_object_relation | 0.000 | 0.000 | 0.000 | 167.69 | 1183.58 | -1015.89 | -1.756 | 5.602 | 0.308 | {'continuation': 50, 'prose': 22} |
| repair|short_only|keep_only_instruction_answer | 0.000 | 0.000 | 0.000 | 167.69 | 10430.47 | -10262.78 | -7.787 | 1.428 | 0.800 | {'prose': 72} |
| repair|short_only|mask_record_without_target_value | 0.000 | 0.000 | 0.000 | 167.69 | 2901.89 | -2734.19 | -4.254 | 1.303 | 0.508 | {'continuation': 28, 'prose': 44} |
| repair|short_only|mask_question_line | 0.000 | 0.000 | 0.000 | 167.69 | 2188.88 | -2021.18 | -3.060 | -0.021 | 0.208 | {'continuation': 25, 'prose': 47} |
| repair|short_only|keep_only_record_instruction_answer | 0.000 | 0.000 | 0.000 | 167.69 | 2188.88 | -2021.18 | -3.060 | -0.021 | 0.208 | {'continuation': 25, 'prose': 47} |
| repair|short_only|mask_record_value_object_relation | 0.000 | 0.000 | 0.000 | 167.69 | 1896.35 | -1728.65 | -4.058 | -0.915 | 0.284 | {'continuation': 40, 'prose': 32} |
| repair|short_only|mask_relation | 0.000 | 0.000 | 0.000 | 167.69 | 1742.21 | -1574.51 | -3.857 | -1.493 | 0.040 | {'continuation': 60, 'prose': 12} |
| repair|short_only|mask_record_line | 0.000 | 0.000 | 0.000 | 167.69 | 7509.93 | -7342.24 | -5.788 | -2.651 | 0.592 | {'continuation': 17, 'prose': 55} |
| repair|short_only|mask_target_value | 0.000 | 0.000 | 0.000 | 167.69 | 1457.12 | -1289.43 | -4.747 | -6.723 | 0.083 | {'continuation': 14, 'prose': 58} |
| repair|short_only|keep_only_question_answer | 0.000 | 0.000 | 0.000 | 167.69 | 9980.21 | -9812.51 | -4.812 | -6.756 | 0.752 | {'prose': 68, 'yesno': 4} |

### glm4

| condition | change | baseline_top1 | patched_top1 | base_rank | patched_rank | rank_effect | pmv_effect | target_effect | mask_frac | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| repair|short_only|mask_answer_context | 0.600 | 0.000 | 0.600 | 2.00 | 1.40 | 0.60 | -0.025 | -0.390 | 0.022 | {'continuation': 5} |
| repair|short_only|mask_record_without_value_object_relation | 0.000 | 0.000 | 0.000 | 2.00 | 1073.20 | -1071.20 | -4.918 | -0.668 | 0.348 | {'continuation': 5} |
| repair|short_only|mask_object_name | 0.000 | 0.000 | 0.000 | 2.00 | 2.20 | -0.20 | 0.150 | -1.328 | 0.217 | {'continuation': 5} |
| repair|short_only|mask_question_line | 0.000 | 0.000 | 0.000 | 2.00 | 67.60 | -65.60 | -4.306 | -2.518 | 0.217 | {'continuation': 5} |
| repair|short_only|keep_only_record_instruction_answer | 0.000 | 0.000 | 0.000 | 2.00 | 67.60 | -65.60 | -4.306 | -2.518 | 0.217 | {'continuation': 5} |
| repair|short_only|mask_instruction_line | 0.000 | 0.000 | 0.000 | 2.00 | 4.40 | -2.40 | -6.188 | -2.595 | 0.174 | {'prose': 5} |
| repair|short_only|mask_relation | 0.000 | 0.000 | 0.000 | 2.00 | 39.20 | -37.20 | -4.638 | -2.696 | 0.043 | {'continuation': 5} |
| repair|short_only|keep_only_instruction_answer | 0.000 | 0.000 | 0.000 | 2.00 | 11315.20 | -11313.20 | -11.249 | -3.342 | 0.783 | {'continuation': 5} |
| repair|short_only|mask_record_without_target_value | 0.000 | 0.000 | 0.000 | 2.00 | 7343.00 | -7341.00 | -10.191 | -3.560 | 0.543 | {'continuation': 5} |
| repair|short_only|mask_record_line | 0.000 | 0.000 | 0.000 | 2.00 | 7911.00 | -7909.00 | -10.296 | -3.606 | 0.565 | {'continuation': 5} |
| repair|short_only|mask_target_value | 0.000 | 0.000 | 0.000 | 2.00 | 93.00 | -91.00 | -4.612 | -5.147 | 0.022 | {'continuation': 5} |
| repair|short_only|keep_only_question_answer | 0.000 | 0.000 | 0.000 | 2.00 | 452.40 | -450.40 | -11.113 | -5.587 | 0.739 | {'prose': 5} |
| repair|short_only|keep_only_record_answer | 0.000 | 0.000 | 0.000 | 2.00 | 144.60 | -142.60 | -6.831 | -5.833 | 0.391 | {'prose': 5} |
| repair|short_only|mask_record_value_object_relation | 0.000 | 0.000 | 0.000 | 2.00 | 1172.60 | -1170.60 | -8.802 | -5.988 | 0.217 | {'continuation': 5} |

### qwen3

| condition | change | baseline_top1 | patched_top1 | base_rank | patched_rank | rank_effect | pmv_effect | target_effect | mask_frac | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| repair|short_only|mask_object_name | 0.000 | 0.000 | 0.000 | 2.00 | 2.67 | -0.67 | 0.625 | 4.614 | 0.192 | {'continuation': 3} |
| repair|short_only|mask_relation | 0.000 | 0.000 | 0.000 | 2.00 | 6.67 | -4.67 | -1.708 | -0.010 | 0.038 | {'continuation': 3} |
| repair|short_only|mask_answer_context | 0.000 | 0.000 | 0.000 | 2.00 | 13.67 | -11.67 | -2.812 | -7.543 | 0.019 | {'continuation': 3} |
| repair|short_only|mask_record_without_value_object_relation | 0.000 | 0.000 | 0.000 | 2.00 | 67.00 | -65.00 | -5.729 | -9.078 | 0.308 | {'continuation': 1, 'prose': 2} |
| repair|short_only|mask_question_line | 0.000 | 0.000 | 0.000 | 2.00 | 11.67 | -9.67 | -3.208 | -11.294 | 0.192 | {'continuation': 3} |
| repair|short_only|keep_only_record_instruction_answer | 0.000 | 0.000 | 0.000 | 2.00 | 11.67 | -9.67 | -3.208 | -11.294 | 0.192 | {'continuation': 3} |
| repair|short_only|mask_instruction_line | 0.000 | 0.000 | 0.000 | 2.00 | 17.67 | -15.67 | -8.958 | -15.703 | 0.154 | {'prose': 3} |
| repair|short_only|keep_only_instruction_answer | 0.000 | 0.000 | 0.000 | 2.00 | 922.00 | -920.00 | -8.667 | -17.374 | 0.808 | {'continuation': 3} |
| repair|short_only|mask_record_without_target_value | 0.000 | 0.000 | 0.000 | 2.00 | 400.00 | -398.00 | -8.917 | -19.558 | 0.481 | {'continuation': 2, 'prose': 1} |
| repair|short_only|keep_only_record_answer | 0.000 | 0.000 | 0.000 | 2.00 | 127.33 | -125.33 | -8.812 | -20.461 | 0.346 | {'prose': 3} |
| repair|short_only|mask_record_line | 0.000 | 0.000 | 0.000 | 2.00 | 892.33 | -890.33 | -10.188 | -20.860 | 0.615 | {'continuation': 3} |
| repair|short_only|mask_target_value | 0.000 | 0.000 | 0.000 | 2.00 | 69.67 | -67.67 | -9.958 | -24.870 | 0.135 | {'continuation': 2, 'prose': 1} |
| repair|short_only|mask_record_value_object_relation | 0.000 | 0.000 | 0.000 | 2.00 | 498.33 | -496.33 | -9.396 | -26.255 | 0.308 | {'continuation': 2, 'prose': 1} |
| repair|short_only|keep_only_question_answer | 0.000 | 0.000 | 0.000 | 2.00 | 329.33 | -327.33 | -11.271 | -31.149 | 0.769 | {'prose': 3} |

