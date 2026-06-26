# Phase 700 Source Contribution Composition and Scaling Audit

- generated: `2026-06-26 17:30:15`

| model | pairs | layers | top_heads | best_restore | repair | rank_effect | final_effect | best_degrade | drop | rank_effect | final_effect | best_erase | drop | final_effect |
|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | 32 | restore|combo_delta_target_value+answer_line+self_last | 0.764 | 166.39 | 27.221 | degradation|full_top32_head_slot | 0.875 | 65.32 | 32.464 | erase|combo_erase_target_value+answer_line+self_last | 1.000 | 61.573 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | 32 | restore|full_top32_head_slot | 0.000 | -0.20 | 0.536 | degradation|full_top32_head_slot | 0.000 | 0.00 | 1.124 | erase|combo_erase_target_value+answer_line+self_last | 0.000 | 0.661 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | 32 | restore|full_top32_head_slot | 1.000 | 1.00 | 4.099 | degradation|full_top32_head_slot | 0.333 | 0.33 | 4.224 | erase|alpha_erase_target_value_2 | 1.000 | 39.890 |

## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|combo_delta_target_value+answer_line+self_last | 0.764 | 0.764 | 1.31 | 166.39 | 5.449 | 27.221 | 7.32 | {'continuation': 4, 'prose': 68} |
| restore|alpha_target_value_2 | 0.750 | 0.750 | 1.69 | 166.00 | 5.748 | 38.273 | 4.32 | {'continuation': 3, 'prose': 69} |
| restore|full_top32_head_slot | 0.736 | 0.736 | 1.53 | 166.17 | 5.622 | 30.117 | 0.00 | {'prose': 72} |
| restore|combo_delta_target_value+answer_line | 0.597 | 0.597 | 1.67 | 166.03 | 4.880 | 24.902 | 6.32 | {'prose': 72} |
| restore|alpha_target_value_1.5 | 0.569 | 0.569 | 2.10 | 165.60 | 4.894 | 29.816 | 4.32 | {'continuation': 1, 'prose': 71} |
| restore|combo_delta_target_value+self_last | 0.528 | 0.528 | 1.94 | 165.75 | 4.714 | 24.201 | 5.32 | {'prose': 72} |
| restore|combo_delta_target_value+record_non_value | 0.375 | 0.375 | 3.35 | 164.35 | 4.009 | 22.612 | 29.76 | {'prose': 72} |
| restore|combo_delta_record_line | 0.375 | 0.375 | 3.35 | 164.35 | 4.009 | 22.612 | 29.76 | {'prose': 72} |
| restore|alpha_target_value_1 | 0.306 | 0.306 | 3.42 | 164.28 | 3.858 | 21.795 | 4.32 | {'prose': 72} |
| restore|alpha_target_value_0.5 | 0.083 | 0.083 | 11.46 | 156.24 | 2.273 | 12.366 | 4.32 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|full_top32_head_slot | 0.000 | 0.000 | 2.20 | -0.20 | 0.019 | 0.536 | 0.00 | {'continuation': 5} |
| restore|combo_delta_target_value+answer_line+self_last | 0.000 | 0.000 | 2.40 | -0.40 | 0.106 | 0.513 | 4.00 | {'continuation': 5} |
| restore|combo_delta_target_value+answer_line | 0.000 | 0.000 | 2.00 | 0.00 | 0.062 | 0.282 | 3.00 | {'continuation': 5} |
| restore|combo_delta_target_value+self_last | 0.000 | 0.000 | 2.00 | 0.00 | 0.037 | 0.157 | 2.00 | {'continuation': 5} |
| restore|combo_delta_target_value+record_non_value | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | 0.003 | 26.00 | {'continuation': 5} |
| restore|combo_delta_record_line | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | 0.003 | 26.00 | {'continuation': 5} |
| restore|alpha_target_value_0.5 | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | -0.016 | 1.00 | {'continuation': 5} |
| restore|alpha_target_value_1 | 0.000 | 0.000 | 2.00 | 0.00 | -0.037 | -0.071 | 1.00 | {'continuation': 5} |
| restore|alpha_target_value_1.5 | 0.000 | 0.000 | 2.00 | 0.00 | -0.050 | -0.085 | 1.00 | {'continuation': 5} |
| restore|alpha_target_value_2 | 0.000 | 0.000 | 2.00 | 0.00 | -0.050 | -0.128 | 1.00 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|full_top32_head_slot | 1.000 | 1.000 | 1.00 | 1.00 | 1.500 | 4.099 | 0.00 | {'continuation': 2, 'prose': 1} |
| restore|combo_delta_target_value+answer_line+self_last | 1.000 | 1.000 | 1.00 | 1.00 | 1.292 | 3.437 | 10.00 | {'continuation': 2, 'prose': 1} |
| restore|combo_delta_target_value+record_non_value | 0.667 | 0.667 | 1.33 | 0.67 | 0.500 | 2.736 | 32.00 | {'continuation': 2, 'prose': 1} |
| restore|combo_delta_target_value+self_last | 0.667 | 0.667 | 1.33 | 0.67 | 0.917 | 2.127 | 8.00 | {'continuation': 2, 'prose': 1} |
| restore|alpha_target_value_2 | 0.667 | 0.667 | 1.33 | 0.67 | 0.958 | 2.044 | 7.00 | {'continuation': 2, 'prose': 1} |
| restore|combo_delta_target_value+answer_line | 0.667 | 0.667 | 1.33 | 0.67 | 0.875 | 1.979 | 9.00 | {'continuation': 2, 'prose': 1} |
| restore|alpha_target_value_1.5 | 0.667 | 0.667 | 1.33 | 0.67 | 0.667 | 1.236 | 7.00 | {'continuation': 2, 'prose': 1} |
| restore|alpha_target_value_1 | 0.667 | 0.667 | 1.33 | 0.67 | 0.375 | 0.753 | 7.00 | {'continuation': 2, 'prose': 1} |
| restore|alpha_target_value_0.5 | 0.667 | 0.667 | 1.33 | 0.67 | 0.208 | 0.420 | 7.00 | {'continuation': 2, 'prose': 1} |
| restore|combo_delta_record_line | 0.333 | 0.333 | 1.67 | 0.33 | -0.208 | -0.477 | 32.00 | {'continuation': 1, 'prose': 2} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|full_top32_head_slot | 0.875 | 0.125 | 66.32 | 65.32 | 4.626 | 32.464 | 0.00 | {'prose': 72} |
| degradation|combo_delta_target_value+answer_line+self_last | 0.833 | 0.167 | 81.15 | 80.15 | 4.489 | 25.947 | 7.32 | {'prose': 72} |
| degradation|alpha_target_value_2 | 0.806 | 0.194 | 427.86 | 426.86 | 5.804 | 44.766 | 4.32 | {'prose': 72} |
| degradation|alpha_target_value_1.5 | 0.750 | 0.250 | 168.21 | 167.21 | 4.009 | 31.407 | 4.32 | {'prose': 72} |
| degradation|combo_delta_target_value+answer_line | 0.681 | 0.319 | 25.43 | 24.43 | 3.256 | 22.527 | 6.32 | {'prose': 72} |
| degradation|combo_delta_target_value+self_last | 0.681 | 0.319 | 21.82 | 20.82 | 3.112 | 21.235 | 5.32 | {'prose': 72} |
| degradation|alpha_target_value_1 | 0.625 | 0.375 | 8.56 | 7.56 | 2.113 | 18.553 | 4.32 | {'prose': 72} |
| degradation|combo_delta_record_line | 0.597 | 0.403 | 5.50 | 4.50 | 2.034 | 18.170 | 29.76 | {'prose': 72} |
| degradation|combo_delta_target_value+record_non_value | 0.597 | 0.403 | 5.51 | 4.51 | 2.034 | 18.155 | 29.76 | {'prose': 72} |
| degradation|alpha_target_value_0.5 | 0.236 | 0.764 | 1.65 | 0.65 | 0.763 | 8.317 | 4.32 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|full_top32_head_slot | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 1.124 | 0.00 | {'prose': 5} |
| degradation|combo_delta_target_value+answer_line+self_last | 0.000 | 1.000 | 1.00 | 0.00 | 0.037 | 0.859 | 4.00 | {'prose': 5} |
| degradation|combo_delta_target_value+answer_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.037 | 0.478 | 3.00 | {'prose': 5} |
| degradation|combo_delta_target_value+self_last | 0.000 | 1.000 | 1.00 | 0.00 | 0.013 | 0.301 | 2.00 | {'prose': 5} |
| degradation|combo_delta_target_value+record_non_value | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.076 | 26.00 | {'prose': 5} |
| degradation|combo_delta_record_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.076 | 26.00 | {'prose': 5} |
| degradation|alpha_target_value_0.5 | 0.000 | 1.000 | 1.00 | 0.00 | -0.037 | -0.077 | 1.00 | {'prose': 5} |
| degradation|alpha_target_value_1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.037 | -0.098 | 1.00 | {'prose': 5} |
| degradation|alpha_target_value_1.5 | 0.000 | 1.000 | 1.00 | 0.00 | -0.037 | -0.120 | 1.00 | {'prose': 5} |
| degradation|alpha_target_value_2 | 0.000 | 1.000 | 1.00 | 0.00 | -0.050 | -0.161 | 1.00 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|full_top32_head_slot | 0.333 | 0.667 | 1.33 | 0.33 | 1.708 | 4.224 | 0.00 | {'continuation': 2, 'prose': 1} |
| degradation|combo_delta_target_value+answer_line+self_last | 0.333 | 0.667 | 1.33 | 0.33 | 1.417 | 3.568 | 10.00 | {'continuation': 2, 'prose': 1} |
| degradation|combo_delta_target_value+record_non_value | 0.000 | 1.000 | 1.00 | 0.00 | 0.625 | 2.966 | 32.00 | {'continuation': 2, 'prose': 1} |
| degradation|combo_delta_target_value+self_last | 0.000 | 1.000 | 1.00 | 0.00 | 1.083 | 2.671 | 8.00 | {'continuation': 2, 'prose': 1} |
| degradation|alpha_target_value_2 | 0.000 | 1.000 | 1.00 | 0.00 | 1.125 | 2.165 | 7.00 | {'continuation': 2, 'prose': 1} |
| degradation|combo_delta_target_value+answer_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.875 | 1.931 | 9.00 | {'continuation': 2, 'prose': 1} |
| degradation|alpha_target_value_1.5 | 0.000 | 1.000 | 1.00 | 0.00 | 0.833 | 1.653 | 7.00 | {'continuation': 2, 'prose': 1} |
| degradation|alpha_target_value_1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.542 | 1.103 | 7.00 | {'continuation': 2, 'prose': 1} |
| degradation|alpha_target_value_0.5 | 0.000 | 1.000 | 1.00 | 0.00 | 0.417 | 0.752 | 7.00 | {'continuation': 2, 'prose': 1} |
| degradation|combo_delta_record_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.083 | -0.058 | 32.00 | {'continuation': 2, 'prose': 1} |


## Best Erase

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| erase|combo_erase_target_value+answer_line+self_last | 1.000 | 0.000 | 1102.65 | 1101.65 | 9.416 | 61.573 | 7.32 | {'prose': 72} |
| erase|alpha_erase_target_value_2 | 0.958 | 0.042 | 1557.46 | 1556.46 | 9.759 | 101.953 | 4.32 | {'continuation': 1, 'prose': 71} |
| erase|alpha_erase_target_value_1.5 | 0.958 | 0.042 | 1490.38 | 1489.38 | 9.282 | 73.745 | 4.32 | {'prose': 72} |
| erase|combo_erase_target_value+record_non_value | 0.931 | 0.069 | 94.83 | 93.83 | 6.318 | 43.697 | 29.76 | {'prose': 72} |
| erase|combo_erase_record_line | 0.931 | 0.069 | 94.83 | 93.83 | 6.318 | 43.697 | 29.76 | {'prose': 72} |
| erase|combo_erase_target_value+self_last | 0.903 | 0.097 | 603.47 | 602.47 | 8.097 | 52.442 | 5.32 | {'prose': 72} |
| erase|combo_erase_target_value+answer_line | 0.847 | 0.153 | 451.54 | 450.54 | 7.509 | 50.044 | 6.32 | {'prose': 72} |
| erase|alpha_erase_target_value_1 | 0.819 | 0.181 | 161.85 | 160.85 | 6.191 | 44.054 | 4.32 | {'prose': 72} |
| erase|alpha_erase_target_value_0.5 | 0.667 | 0.333 | 4.97 | 3.97 | 2.137 | 18.044 | 4.32 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| erase|combo_erase_target_value+answer_line+self_last | 0.000 | 1.000 | 1.00 | 0.00 | 0.200 | 0.661 | 4.00 | {'prose': 5} |
| erase|combo_erase_target_value+answer_line | 0.000 | 1.000 | 1.00 | 0.00 | 0.025 | 0.269 | 3.00 | {'prose': 5} |
| erase|combo_erase_target_value+self_last | 0.000 | 1.000 | 1.00 | 0.00 | 0.025 | 0.148 | 2.00 | {'prose': 5} |
| erase|alpha_erase_target_value_2 | 0.000 | 1.000 | 1.00 | 0.00 | 0.087 | 0.103 | 1.00 | {'prose': 5} |
| erase|alpha_erase_target_value_0.5 | 0.000 | 1.000 | 1.00 | 0.00 | -0.062 | -0.115 | 1.00 | {'prose': 5} |
| erase|alpha_erase_target_value_1.5 | 0.000 | 1.000 | 1.00 | 0.00 | -0.050 | -0.118 | 1.00 | {'prose': 5} |
| erase|alpha_erase_target_value_1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.087 | -0.155 | 1.00 | {'prose': 5} |
| erase|combo_erase_target_value+record_non_value | 0.000 | 1.000 | 1.00 | 0.00 | -0.163 | -1.061 | 26.00 | {'prose': 5} |
| erase|combo_erase_record_line | 0.000 | 1.000 | 1.00 | 0.00 | -0.163 | -1.061 | 26.00 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_effect | src_len | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| erase|alpha_erase_target_value_2 | 1.000 | 0.000 | 33.33 | 32.33 | 9.708 | 39.890 | 7.00 | {'continuation': 3} |
| erase|alpha_erase_target_value_1.5 | 1.000 | 0.000 | 41.33 | 40.33 | 10.771 | 27.433 | 7.00 | {'continuation': 2, 'prose': 1} |
| erase|combo_erase_target_value+self_last | 1.000 | 0.000 | 12.00 | 11.00 | 8.000 | 19.677 | 8.00 | {'continuation': 2, 'prose': 1} |
| erase|combo_erase_target_value+answer_line+self_last | 1.000 | 0.000 | 28.67 | 27.67 | 7.812 | 19.500 | 10.00 | {'continuation': 2, 'prose': 1} |
| erase|alpha_erase_target_value_1 | 1.000 | 0.000 | 8.00 | 7.00 | 6.917 | 16.458 | 7.00 | {'continuation': 2, 'prose': 1} |
| erase|combo_erase_target_value+answer_line | 1.000 | 0.000 | 10.33 | 9.33 | 6.208 | 15.800 | 9.00 | {'prose': 3} |
| erase|combo_erase_target_value+record_non_value | 1.000 | 0.000 | 7.00 | 6.00 | 6.417 | 10.704 | 32.00 | {'continuation': 2, 'prose': 1} |
| erase|combo_erase_record_line | 1.000 | 0.000 | 9.33 | 8.33 | 4.250 | -12.144 | 32.00 | {'prose': 3} |
| erase|alpha_erase_target_value_0.5 | 0.667 | 0.333 | 2.00 | 1.00 | 2.917 | 7.531 | 7.00 | {'continuation': 2, 'prose': 1} |

