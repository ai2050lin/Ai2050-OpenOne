# Phase 699 Answer-Last Head Source-Token Causal Patch Audit

- generated: `2026-06-26 17:22:14`

| model | pairs | layers | top_heads | best_restore | repair | rank_effect | final_effect | best_degrade | drop | rank_effect | final_effect | best_erase | drop | final_effect |
|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | 32 | restore|full_top32_head_slot | 0.736 | 166.17 | 30.117 | degradation|full_top32_head_slot | 0.875 | 65.32 | 32.464 | erase|erase_record_line | 0.931 | 43.697 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | 32 | restore|full_top32_head_slot | 0.000 | -0.20 | 0.536 | degradation|full_top32_head_slot | 0.000 | 0.00 | 1.124 | erase|erase_answer_line | 0.000 | 0.443 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | 32 | restore|full_top32_head_slot | 1.000 | 1.00 | 4.099 | degradation|full_top32_head_slot | 0.333 | 0.33 | 4.224 | erase|erase_target_value | 1.000 | 16.458 |

## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|full_top32_head_slot | 0.736 | 0.736 | 1.53 | 166.17 | 5.622 | 30.117 | 0.00 | {'prose': 72} |
| restore|delta_record_line | 0.375 | 0.375 | 3.35 | 164.35 | 4.009 | 22.612 | 29.76 | {'prose': 72} |
| restore|delta_target_value | 0.306 | 0.306 | 3.42 | 164.28 | 3.858 | 21.795 | 4.32 | {'prose': 72} |
| restore|delta_answer_line | 0.097 | 0.097 | 37.50 | 130.19 | 1.678 | 4.900 | 2.00 | {'prose': 72} |
| restore|delta_object_name | 0.069 | 0.069 | 170.06 | -2.36 | 0.208 | 1.120 | 11.47 | {'prose': 72} |
| restore|delta_record_non_value | 0.069 | 0.069 | 197.82 | -30.12 | 0.064 | 0.081 | 25.44 | {'continuation': 1, 'prose': 71} |
| restore|delta_self_last | 0.042 | 0.042 | 45.72 | 121.97 | 1.449 | 4.080 | 1.00 | {'prose': 72} |
| restore|delta_instruction_line | 0.000 | 0.000 | 160.50 | 7.19 | -0.029 | 0.387 | 8.00 | {'continuation': 1, 'prose': 71} |
| restore|delta_relation | 0.000 | 0.000 | 172.86 | -5.17 | -0.065 | -0.434 | 2.00 | {'prose': 72} |
| restore|delta_question_line | 0.000 | 0.000 | 169.19 | -1.50 | -0.056 | -0.453 | 10.42 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|full_top32_head_slot | 0.000 | 0.000 | 2.20 | -0.20 | 0.019 | 0.536 | 0.00 | {'continuation': 5} |
| restore|delta_answer_line | 0.000 | 0.000 | 2.00 | 0.00 | 0.069 | 0.335 | 2.00 | {'continuation': 5} |
| restore|delta_self_last | 0.000 | 0.000 | 2.20 | -0.20 | 0.050 | 0.250 | 1.00 | {'continuation': 5} |
| restore|delta_instruction_line | 0.000 | 0.000 | 2.00 | 0.00 | 0.000 | 0.173 | 8.00 | {'continuation': 5} |
| restore|delta_record_non_value | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | 0.069 | 25.00 | {'continuation': 5} |
| restore|delta_object_name | 0.000 | 0.000 | 2.00 | 0.00 | -0.013 | 0.004 | 10.00 | {'continuation': 5} |
| restore|delta_record_line | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | 0.003 | 26.00 | {'continuation': 5} |
| restore|delta_question_line | 0.000 | 0.000 | 2.00 | 0.00 | -0.037 | 0.000 | 10.00 | {'continuation': 5} |
| restore|delta_relation | 0.000 | 0.000 | 2.00 | 0.00 | -0.013 | -0.004 | 2.00 | {'continuation': 5} |
| restore|delta_target_value | 0.000 | 0.000 | 2.00 | 0.00 | -0.037 | -0.071 | 1.00 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|full_top32_head_slot | 1.000 | 1.000 | 1.00 | 1.00 | 1.500 | 4.099 | 0.00 | {'continuation': 2, 'prose': 1} |
| restore|delta_self_last | 0.667 | 0.667 | 1.33 | 0.67 | 0.375 | 1.299 | 1.00 | {'continuation': 2, 'prose': 1} |
| restore|delta_answer_line | 0.667 | 0.667 | 1.33 | 0.67 | 0.333 | 0.814 | 2.00 | {'continuation': 2, 'prose': 1} |
| restore|delta_target_value | 0.667 | 0.667 | 1.33 | 0.67 | 0.375 | 0.753 | 7.00 | {'continuation': 2, 'prose': 1} |
| restore|delta_instruction_line | 0.667 | 0.667 | 1.33 | 0.67 | 0.333 | -0.659 | 8.00 | {'continuation': 2, 'prose': 1} |
| restore|delta_record_non_value | 0.333 | 0.333 | 1.67 | 0.33 | 0.083 | 1.974 | 25.00 | {'continuation': 1, 'prose': 2} |
| restore|delta_question_line | 0.333 | 0.333 | 1.67 | 0.33 | 0.083 | 0.313 | 10.00 | {'continuation': 2, 'prose': 1} |
| restore|delta_object_name | 0.333 | 0.333 | 1.67 | 0.33 | 0.125 | 0.025 | 10.00 | {'continuation': 2, 'prose': 1} |
| restore|delta_record_line | 0.333 | 0.333 | 1.67 | 0.33 | -0.208 | -0.477 | 32.00 | {'continuation': 1, 'prose': 2} |
| restore|delta_relation | 0.000 | 0.000 | 2.33 | -0.33 | -0.042 | -0.125 | 2.00 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|full_top32_head_slot | 0.875 | 0.125 | 66.32 | 65.32 | 4.626 | 32.464 | 0.00 | {'prose': 72} |
| degradation|delta_target_value | 0.625 | 0.375 | 8.56 | 7.56 | 2.113 | 18.553 | 4.32 | {'prose': 72} |
| degradation|delta_record_line | 0.597 | 0.403 | 5.50 | 4.50 | 2.034 | 18.170 | 29.76 | {'prose': 72} |
| degradation|delta_answer_line | 0.250 | 0.750 | 1.39 | 0.39 | 0.770 | 3.353 | 2.00 | {'prose': 72} |
| degradation|delta_self_last | 0.181 | 0.819 | 1.25 | 0.25 | 0.637 | 2.066 | 1.00 | {'prose': 72} |
| degradation|delta_object_name | 0.056 | 0.944 | 1.14 | 0.14 | 0.100 | 1.110 | 11.47 | {'prose': 72} |
| degradation|delta_record_non_value | 0.028 | 0.972 | 1.10 | 0.10 | -0.015 | 0.339 | 25.44 | {'prose': 72} |
| degradation|delta_instruction_line | 0.014 | 0.986 | 1.01 | 0.01 | -0.027 | 0.781 | 11.00 | {'prose': 72} |
| degradation|delta_question_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.003 | -0.236 | 10.42 | {'prose': 72} |
| degradation|delta_relation | 0.000 | 1.000 | 1.00 | 0.00 | -0.075 | -0.509 | 2.00 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|full_top32_head_slot | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 1.124 | 0.00 | {'prose': 5} |
| degradation|delta_answer_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.013 | 0.525 | 2.00 | {'prose': 5} |
| degradation|delta_instruction_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.025 | 0.426 | 11.00 | {'prose': 5} |
| degradation|delta_self_last | 0.000 | 1.000 | 1.00 | 0.00 | 0.013 | 0.338 | 1.00 | {'prose': 5} |
| degradation|delta_record_non_value | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.132 | 25.00 | {'prose': 5} |
| degradation|delta_record_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.076 | 26.00 | {'prose': 5} |
| degradation|delta_object_name | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | -0.033 | 10.00 | {'prose': 5} |
| degradation|delta_question_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.013 | -0.034 | 10.00 | {'prose': 5} |
| degradation|delta_relation | 0.000 | 1.000 | 1.00 | 0.00 | -0.013 | -0.041 | 2.00 | {'prose': 5} |
| degradation|delta_target_value | 0.000 | 1.000 | 1.00 | 0.00 | -0.037 | -0.098 | 1.00 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|full_top32_head_slot | 0.333 | 0.667 | 1.33 | 0.33 | 1.708 | 4.224 | 0.00 | {'continuation': 2, 'prose': 1} |
| degradation|delta_record_non_value | 0.000 | 1.000 | 1.00 | 0.00 | 0.208 | 2.430 | 25.00 | {'continuation': 2, 'prose': 1} |
| degradation|delta_self_last | 0.000 | 1.000 | 1.00 | 0.00 | 0.625 | 1.860 | 1.00 | {'continuation': 2, 'prose': 1} |
| degradation|delta_target_value | 0.000 | 1.000 | 1.00 | 0.00 | 0.542 | 1.103 | 7.00 | {'continuation': 2, 'prose': 1} |
| degradation|delta_answer_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.375 | 1.011 | 2.00 | {'continuation': 2, 'prose': 1} |
| degradation|delta_question_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.167 | 0.563 | 10.00 | {'continuation': 2, 'prose': 1} |
| degradation|delta_object_name | 0.000 | 1.000 | 1.00 | 0.00 | 0.083 | 0.402 | 10.00 | {'continuation': 2, 'prose': 1} |
| degradation|delta_relation | 0.000 | 1.000 | 1.00 | 0.00 | 0.083 | 0.253 | 2.00 | {'continuation': 2, 'prose': 1} |
| degradation|delta_record_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.083 | -0.058 | 32.00 | {'continuation': 2, 'prose': 1} |
| degradation|delta_instruction_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.667 | -0.391 | 11.00 | {'continuation': 1, 'prose': 2} |


## Best Erase

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| erase|erase_record_line | 0.931 | 0.069 | 94.83 | 93.83 | 6.318 | 43.697 | 29.76 | {'prose': 72} |
| erase|erase_target_value | 0.819 | 0.181 | 161.85 | 160.85 | 6.191 | 44.054 | 4.32 | {'prose': 72} |
| erase|erase_self_last | 0.403 | 0.597 | 1.88 | 0.88 | 1.155 | 6.307 | 1.00 | {'prose': 72} |
| erase|erase_answer_line | 0.333 | 0.667 | 1.46 | 0.46 | 0.772 | 5.433 | 2.00 | {'prose': 72} |
| erase|erase_object_name | 0.167 | 0.833 | 3.79 | 2.79 | 0.754 | 4.158 | 11.47 | {'prose': 72} |
| erase|erase_record_non_value | 0.153 | 0.847 | 1.71 | 0.71 | 0.290 | 1.468 | 25.44 | {'prose': 72} |
| erase|erase_question_line | 0.042 | 0.958 | 1.04 | 0.04 | -0.082 | 0.361 | 10.42 | {'prose': 72} |
| erase|erase_relation | 0.000 | 1.000 | 1.00 | 0.00 | -0.144 | -0.134 | 2.00 | {'prose': 72} |
| erase|erase_instruction_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.081 | -0.316 | 11.00 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| erase|erase_answer_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.125 | 0.443 | 2.00 | {'prose': 5} |
| erase|erase_self_last | 0.000 | 1.000 | 1.00 | 0.00 | 0.100 | 0.269 | 1.00 | {'prose': 5} |
| erase|erase_relation | 0.000 | 1.000 | 1.00 | 0.00 | -0.025 | -0.030 | 2.00 | {'prose': 5} |
| erase|erase_question_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.050 | -0.126 | 10.00 | {'prose': 5} |
| erase|erase_target_value | 0.000 | 1.000 | 1.00 | 0.00 | -0.087 | -0.155 | 1.00 | {'prose': 5} |
| erase|erase_instruction_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.013 | -0.201 | 11.00 | {'prose': 5} |
| erase|erase_object_name | 0.000 | 1.000 | 1.00 | 0.00 | -0.062 | -0.220 | 10.00 | {'prose': 5} |
| erase|erase_record_non_value | 0.000 | 1.000 | 1.00 | 0.00 | -0.100 | -0.933 | 25.00 | {'prose': 5} |
| erase|erase_record_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.163 | -1.061 | 26.00 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| erase|erase_target_value | 1.000 | 0.000 | 8.00 | 7.00 | 6.917 | 16.458 | 7.00 | {'continuation': 2, 'prose': 1} |
| erase|erase_record_line | 1.000 | 0.000 | 9.33 | 8.33 | 4.250 | -12.144 | 32.00 | {'prose': 3} |
| erase|erase_self_last | 0.000 | 1.000 | 1.00 | 0.00 | 0.667 | 3.024 | 1.00 | {'continuation': 2, 'prose': 1} |
| erase|erase_relation | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.420 | 2.00 | {'continuation': 2, 'prose': 1} |
| erase|erase_answer_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.542 | -0.152 | 2.00 | {'prose': 3} |
| erase|erase_object_name | 0.000 | 1.000 | 1.00 | 0.00 | -0.167 | -0.440 | 10.00 | {'continuation': 2, 'prose': 1} |
| erase|erase_question_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.083 | -0.513 | 10.00 | {'continuation': 2, 'prose': 1} |
| erase|erase_instruction_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.292 | -1.974 | 11.00 | {'continuation': 2, 'prose': 1} |
| erase|erase_record_non_value | 0.000 | 1.000 | 1.00 | 0.00 | -0.167 | -5.582 | 25.00 | {'continuation': 1, 'prose': 2} |

