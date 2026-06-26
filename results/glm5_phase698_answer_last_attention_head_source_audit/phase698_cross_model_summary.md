# Phase 698 Answer-Last Attention Head and Source-Token Path Audit

- generated: `2026-06-26 17:01:06`

| model | pairs | layers | best_restore | repair | patched_top1 | rank_effect | final_proj_effect | best_degrade | drop | patched_top1 | rank_effect | final_proj_effect |
|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | restore|global_top32 | 0.736 | 0.736 | 166.17 | 30.117 | degradation|global_top32 | 0.875 | 0.125 | 65.32 | 32.464 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | restore|window_all_heads | 0.200 | 0.200 | 0.00 | -0.226 | degradation|global_top32 | 0.000 | 1.000 | 0.00 | 1.124 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | restore|global_top32 | 1.000 | 1.000 | 1.00 | 4.099 | degradation|global_top32 | 0.333 | 0.667 | 0.33 | 4.224 |

## Top Candidate Heads

### deepseek7b

| head | mean_direct_effect | mean_delta_norm |
|---|---:|---:|
| L26H15 | 4.493 | 29.249 |
| L26H19 | 4.218 | 19.068 |
| L23H11 | 2.974 | 23.178 |
| L27H2 | 2.539 | 11.189 |
| L23H19 | 1.436 | 10.125 |
| L26H0 | 1.223 | 7.610 |
| L27H3 | 0.981 | 7.117 |
| L26H14 | 0.873 | 12.313 |
| L26H24 | 0.775 | 9.492 |
| L27H17 | 0.769 | 14.574 |
| L27H19 | 0.763 | 10.153 |
| L23H18 | 0.738 | 5.079 |
| L26H25 | 0.719 | 12.185 |
| L27H24 | 0.715 | 7.849 |
| L26H3 | 0.690 | 5.540 |
| L26H17 | 0.634 | 9.832 |
| L26H26 | 0.631 | 8.243 |
| L26H23 | 0.433 | 7.552 |
| L24H27 | 0.423 | 7.982 |
| L25H9 | 0.389 | 5.910 |
| L26H6 | 0.361 | 5.517 |
| L24H21 | 0.360 | 6.269 |
| L23H5 | 0.347 | 5.941 |
| L25H12 | 0.312 | 5.102 |
| L24H16 | 0.278 | 4.944 |
| L26H1 | 0.235 | 5.390 |
| L25H14 | 0.231 | 8.338 |
| L27H1 | 0.231 | 8.816 |
| L27H5 | 0.216 | 6.022 |
| L23H6 | 0.212 | 3.714 |
| L24H23 | 0.197 | 4.027 |
| L26H11 | 0.182 | 5.571 |

### glm4

| head | mean_direct_effect | mean_delta_norm |
|---|---:|---:|
| L34H4 | 0.072 | 4.920 |
| L34H8 | 0.046 | 9.291 |
| L38H5 | 0.043 | 10.641 |
| L38H7 | 0.040 | 15.189 |
| L38H15 | 0.038 | 11.447 |
| L39H22 | 0.037 | 9.642 |
| L36H2 | 0.034 | 7.472 |
| L35H11 | 0.033 | 9.919 |
| L36H27 | 0.031 | 9.179 |
| L34H20 | 0.031 | 10.022 |
| L39H21 | 0.030 | 12.911 |
| L39H23 | 0.030 | 8.467 |
| L39H24 | 0.030 | 10.524 |
| L35H30 | 0.028 | 4.984 |
| L35H10 | 0.028 | 19.074 |
| L35H28 | 0.027 | 6.743 |
| L37H10 | 0.026 | 8.947 |
| L35H8 | 0.025 | 10.654 |
| L38H11 | 0.024 | 7.391 |
| L37H11 | 0.024 | 9.197 |
| L36H16 | 0.024 | 10.424 |
| L35H3 | 0.023 | 7.257 |
| L36H13 | 0.022 | 4.332 |
| L39H11 | 0.022 | 6.551 |
| L36H25 | 0.022 | 7.288 |
| L38H12 | 0.020 | 7.713 |
| L36H7 | 0.020 | 9.825 |
| L39H2 | 0.019 | 6.784 |
| L34H12 | 0.017 | 4.820 |
| L39H9 | 0.017 | 7.968 |
| L38H8 | 0.016 | 5.268 |
| L34H18 | 0.016 | 5.120 |

### qwen3

| head | mean_direct_effect | mean_delta_norm |
|---|---:|---:|
| L31H14 | 1.134 | 5.878 |
| L32H8 | 1.037 | 5.510 |
| L33H20 | 0.671 | 4.875 |
| L34H1 | 0.474 | 4.126 |
| L35H26 | 0.427 | 6.892 |
| L34H28 | 0.383 | 7.564 |
| L31H6 | 0.378 | 2.628 |
| L32H25 | 0.331 | 10.892 |
| L35H2 | 0.322 | 7.689 |
| L34H19 | 0.299 | 15.512 |
| L35H1 | 0.246 | 6.256 |
| L35H8 | 0.243 | 5.017 |
| L31H19 | 0.234 | 3.224 |
| L30H27 | 0.219 | 2.886 |
| L35H15 | 0.216 | 5.812 |
| L34H0 | 0.185 | 4.533 |
| L32H24 | 0.171 | 4.337 |
| L34H9 | 0.156 | 8.639 |
| L35H29 | 0.142 | 5.873 |
| L32H11 | 0.137 | 7.314 |
| L34H23 | 0.133 | 10.198 |
| L35H25 | 0.123 | 10.200 |
| L34H2 | 0.122 | 2.430 |
| L34H31 | 0.105 | 4.727 |
| L34H20 | 0.104 | 7.930 |
| L34H21 | 0.104 | 2.946 |
| L33H7 | 0.096 | 6.037 |
| L31H28 | 0.092 | 7.090 |
| L31H31 | 0.087 | 4.726 |
| L32H0 | 0.085 | 2.976 |
| L33H22 | 0.084 | 1.237 |
| L33H30 | 0.084 | 4.623 |


## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|global_top32 | 0.736 | 0.736 | 1.53 | 166.17 | 5.622 | 30.117 | {'prose': 72} |
| restore|window_all_heads | 0.653 | 0.653 | 1.58 | 166.11 | 5.135 | 26.700 | {'prose': 72} |
| restore|global_top16 | 0.556 | 0.556 | 2.92 | 164.78 | 4.671 | 25.845 | {'prose': 72} |
| restore|late_top16 | 0.472 | 0.472 | 2.90 | 164.79 | 4.362 | 22.560 | {'prose': 72} |
| restore|global_top8 | 0.389 | 0.389 | 6.68 | 161.01 | 3.845 | 22.009 | {'prose': 72} |
| restore|early_top16 | 0.111 | 0.111 | 21.04 | 146.65 | 2.037 | 11.136 | {'prose': 72} |
| restore|L26_top4 | 0.056 | 0.056 | 25.94 | 141.75 | 1.731 | 10.691 | {'prose': 72} |
| restore|L23_top4 | 0.042 | 0.042 | 55.60 | 112.10 | 1.081 | 5.097 | {'prose': 72} |
| restore|L27_top4 | 0.014 | 0.014 | 49.90 | 117.79 | 0.972 | 5.190 | {'prose': 72} |
| restore|L27_top1 | 0.014 | 0.014 | 58.79 | 108.90 | 0.636 | 3.443 | {'prose': 72} |
| restore|L24_top4 | 0.014 | 0.014 | 92.39 | 75.31 | 0.248 | 2.371 | {'prose': 72} |
| restore|L23_top1 | 0.014 | 0.014 | 108.00 | 59.69 | 0.399 | 0.656 | {'prose': 72} |
| restore|L25_top4 | 0.014 | 0.014 | 176.97 | -9.28 | 0.001 | 0.018 | {'continuation': 1, 'prose': 71} |
| restore|L25_top1 | 0.014 | 0.014 | 178.39 | -10.69 | -0.011 | -0.484 | {'prose': 72} |
| restore|global_random_window | 0.000 | 0.000 | 50.62 | 117.07 | 0.682 | 9.280 | {'prose': 72} |
| restore|L26_top1 | 0.000 | 0.000 | 57.18 | 110.51 | 0.661 | 6.574 | {'prose': 72} |
| restore|L24_top1 | 0.000 | 0.000 | 165.36 | 2.33 | -0.065 | 0.368 | {'continuation': 1, 'prose': 71} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|window_all_heads | 0.200 | 0.200 | 2.00 | 0.00 | -0.062 | -0.226 | {'continuation': 5} |
| restore|global_top32 | 0.000 | 0.000 | 2.20 | -0.20 | 0.019 | 0.536 | {'continuation': 5} |
| restore|early_top16 | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | 0.374 | {'continuation': 5} |
| restore|global_top16 | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | 0.203 | {'continuation': 5} |
| restore|late_top16 | 0.000 | 0.000 | 2.00 | 0.00 | 0.062 | 0.164 | {'continuation': 5} |
| restore|L36_top4 | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | 0.142 | {'continuation': 5} |
| restore|L35_top4 | 0.000 | 0.000 | 2.00 | 0.00 | -0.013 | 0.100 | {'continuation': 5} |
| restore|L37_top4 | 0.000 | 0.000 | 2.00 | 0.00 | 0.013 | 0.081 | {'continuation': 5} |
| restore|L35_top1 | 0.000 | 0.000 | 2.00 | 0.00 | 0.037 | 0.044 | {'continuation': 5} |
| restore|L39_top4 | 0.000 | 0.000 | 2.00 | 0.00 | 0.050 | 0.038 | {'continuation': 5} |
| restore|L37_top1 | 0.000 | 0.000 | 2.00 | 0.00 | 0.000 | 0.023 | {'continuation': 5} |
| restore|global_top8 | 0.000 | 0.000 | 2.00 | 0.00 | 0.000 | 0.022 | {'continuation': 5} |
| restore|L39_top1 | 0.000 | 0.000 | 2.00 | 0.00 | 0.000 | 0.014 | {'continuation': 5} |
| restore|L34_top1 | 0.000 | 0.000 | 2.00 | 0.00 | -0.037 | -0.003 | {'continuation': 5} |
| restore|L38_top1 | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | -0.004 | {'continuation': 5} |
| restore|L36_top1 | 0.000 | 0.000 | 2.00 | 0.00 | -0.025 | -0.004 | {'continuation': 5} |
| restore|global_random_window | 0.000 | 0.000 | 2.00 | 0.00 | 0.025 | -0.012 | {'continuation': 5} |
| restore|L38_top4 | 0.000 | 0.000 | 2.00 | 0.00 | 0.000 | -0.047 | {'continuation': 5} |
| restore|L34_top4 | 0.000 | 0.000 | 2.00 | 0.00 | -0.037 | -0.082 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|global_top32 | 1.000 | 1.000 | 1.00 | 1.00 | 1.500 | 4.099 | {'continuation': 2, 'prose': 1} |
| restore|global_top16 | 1.000 | 1.000 | 1.00 | 1.00 | 1.333 | 3.818 | {'continuation': 2, 'prose': 1} |
| restore|window_all_heads | 0.667 | 0.667 | 1.33 | 0.67 | 0.917 | 3.618 | {'prose': 3} |
| restore|global_top8 | 0.667 | 0.667 | 1.33 | 0.67 | 0.708 | 2.656 | {'continuation': 1, 'prose': 2} |
| restore|early_top16 | 0.667 | 0.667 | 1.33 | 0.67 | 0.625 | 2.261 | {'continuation': 2, 'prose': 1} |
| restore|L31_top4 | 0.667 | 0.667 | 1.33 | 0.67 | 0.625 | 2.036 | {'continuation': 2, 'prose': 1} |
| restore|late_top16 | 0.667 | 0.667 | 1.33 | 0.67 | 0.792 | 1.746 | {'continuation': 2, 'prose': 1} |
| restore|L30_top4 | 0.667 | 0.667 | 1.33 | 0.67 | 0.333 | 1.315 | {'continuation': 2, 'prose': 1} |
| restore|L31_top1 | 0.667 | 0.667 | 1.33 | 0.67 | 0.417 | 1.174 | {'continuation': 2, 'prose': 1} |
| restore|L34_top4 | 0.667 | 0.667 | 1.33 | 0.67 | 0.250 | 0.809 | {'continuation': 1, 'prose': 2} |
| restore|L35_top4 | 0.667 | 0.667 | 1.33 | 0.67 | 0.417 | 0.451 | {'continuation': 2, 'prose': 1} |
| restore|L35_top1 | 0.667 | 0.667 | 1.33 | 0.67 | 0.083 | 0.328 | {'continuation': 2, 'prose': 1} |
| restore|L30_top1 | 0.667 | 0.667 | 1.33 | 0.67 | 0.000 | -0.061 | {'continuation': 2, 'prose': 1} |
| restore|global_random_window | 0.333 | 0.333 | 2.00 | 0.00 | -0.292 | 0.025 | {'continuation': 1, 'prose': 2} |
| restore|L33_top1 | 0.000 | 0.000 | 2.00 | 0.00 | 0.000 | 0.729 | {'continuation': 1, 'prose': 2} |
| restore|L33_top4 | 0.000 | 0.000 | 2.33 | -0.33 | -0.042 | 0.689 | {'continuation': 2, 'prose': 1} |
| restore|L34_top1 | 0.000 | 0.000 | 2.00 | 0.00 | 0.042 | 0.167 | {'continuation': 2, 'prose': 1} |
| restore|L32_top1 | 0.000 | 0.000 | 2.33 | -0.33 | -0.167 | -0.509 | {'continuation': 2, 'prose': 1} |
| restore|L32_top4 | 0.000 | 0.000 | 2.00 | 0.00 | -0.167 | -0.988 | {'continuation': 1, 'prose': 2} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|global_top32 | 0.875 | 0.125 | 66.32 | 65.32 | 4.626 | 32.464 | {'prose': 72} |
| degradation|window_all_heads | 0.833 | 0.167 | 87.14 | 86.14 | 4.507 | 31.503 | {'prose': 72} |
| degradation|late_top16 | 0.694 | 0.306 | 17.90 | 16.90 | 3.336 | 22.322 | {'prose': 72} |
| degradation|global_top16 | 0.681 | 0.319 | 13.57 | 12.57 | 3.099 | 23.131 | {'prose': 72} |
| degradation|global_top8 | 0.611 | 0.389 | 8.42 | 7.42 | 2.418 | 20.028 | {'prose': 72} |
| degradation|L26_top4 | 0.347 | 0.653 | 1.92 | 0.92 | 0.990 | 10.482 | {'prose': 72} |
| degradation|early_top16 | 0.292 | 0.708 | 1.49 | 0.49 | 0.661 | 8.518 | {'prose': 72} |
| degradation|L27_top4 | 0.181 | 0.819 | 1.47 | 0.47 | 0.598 | 4.145 | {'prose': 72} |
| degradation|L27_top1 | 0.153 | 0.847 | 1.36 | 0.36 | 0.446 | 3.187 | {'prose': 72} |
| degradation|L23_top4 | 0.139 | 0.861 | 1.15 | 0.15 | 0.251 | 1.500 | {'prose': 72} |
| degradation|L26_top1 | 0.097 | 0.903 | 1.12 | 0.12 | 0.345 | 7.170 | {'prose': 72} |
| degradation|global_random_window | 0.083 | 0.917 | 1.12 | 0.12 | 0.112 | 8.031 | {'prose': 72} |
| degradation|L23_top1 | 0.083 | 0.917 | 1.11 | 0.11 | 0.149 | -0.221 | {'prose': 72} |
| degradation|L25_top4 | 0.028 | 0.972 | 1.03 | 0.03 | -0.030 | 0.641 | {'prose': 72} |
| degradation|L24_top4 | 0.000 | 1.000 | 1.00 | 0.00 | -0.052 | 2.060 | {'prose': 72} |
| degradation|L24_top1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.127 | 0.185 | {'prose': 72} |
| degradation|L25_top1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.036 | -0.202 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|global_top32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 1.124 | {'prose': 5} |
| degradation|global_top16 | 0.000 | 1.000 | 1.00 | 0.00 | -0.037 | 0.689 | {'prose': 5} |
| degradation|early_top16 | 0.000 | 1.000 | 1.00 | 0.00 | -0.050 | 0.558 | {'prose': 5} |
| degradation|late_top16 | 0.000 | 1.000 | 1.00 | 0.00 | 0.037 | 0.522 | {'prose': 5} |
| degradation|window_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | -0.138 | 0.382 | {'continuation': 1, 'prose': 4} |
| degradation|global_top8 | 0.000 | 1.000 | 1.00 | 0.00 | -0.013 | 0.334 | {'prose': 5} |
| degradation|L38_top4 | 0.000 | 1.000 | 1.00 | 0.00 | -0.025 | 0.265 | {'prose': 5} |
| degradation|L35_top4 | 0.000 | 1.000 | 1.00 | 0.00 | -0.037 | 0.232 | {'prose': 5} |
| degradation|L36_top4 | 0.000 | 1.000 | 1.00 | 0.00 | -0.025 | 0.165 | {'prose': 5} |
| degradation|L39_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.037 | 0.108 | {'prose': 5} |
| degradation|L37_top4 | 0.000 | 1.000 | 1.00 | 0.00 | -0.025 | 0.080 | {'prose': 5} |
| degradation|L35_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.027 | {'prose': 5} |
| degradation|L36_top1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.025 | 0.016 | {'prose': 5} |
| degradation|L39_top1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.013 | 0.002 | {'prose': 5} |
| degradation|L37_top1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.050 | -0.021 | {'prose': 5} |
| degradation|L34_top1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.025 | -0.040 | {'prose': 5} |
| degradation|L38_top1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.025 | -0.057 | {'prose': 5} |
| degradation|L34_top4 | 0.000 | 1.000 | 1.00 | 0.00 | -0.037 | -0.081 | {'prose': 5} |
| degradation|global_random_window | 0.000 | 1.000 | 1.00 | 0.00 | -0.013 | -0.142 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|global_top32 | 0.333 | 0.667 | 1.33 | 0.33 | 1.708 | 4.224 | {'continuation': 2, 'prose': 1} |
| degradation|global_top16 | 0.333 | 0.667 | 1.33 | 0.33 | 1.542 | 4.126 | {'continuation': 2, 'prose': 1} |
| degradation|window_all_heads | 0.333 | 0.667 | 1.33 | 0.33 | 1.125 | 3.845 | {'continuation': 2, 'prose': 1} |
| degradation|global_top8 | 0.000 | 1.000 | 1.00 | 0.00 | 0.750 | 2.632 | {'continuation': 2, 'prose': 1} |
| degradation|early_top16 | 0.000 | 1.000 | 1.00 | 0.00 | 0.500 | 2.208 | {'continuation': 2, 'prose': 1} |
| degradation|L31_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.792 | 2.198 | {'continuation': 2, 'prose': 1} |
| degradation|late_top16 | 0.000 | 1.000 | 1.00 | 0.00 | 0.917 | 1.831 | {'continuation': 2, 'prose': 1} |
| degradation|L30_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.333 | 1.708 | {'continuation': 2, 'prose': 1} |
| degradation|L31_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.708 | 1.547 | {'continuation': 2, 'prose': 1} |
| degradation|L34_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.333 | 1.010 | {'continuation': 2, 'prose': 1} |
| degradation|L33_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.125 | 0.994 | {'continuation': 2, 'prose': 1} |
| degradation|L33_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.042 | 0.614 | {'continuation': 2, 'prose': 1} |
| degradation|L35_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.583 | 0.374 | {'continuation': 2, 'prose': 1} |
| degradation|L35_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.042 | 0.290 | {'continuation': 2, 'prose': 1} |
| degradation|L34_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.167 | 0.233 | {'continuation': 2, 'prose': 1} |
| degradation|global_random_window | 0.000 | 1.000 | 1.00 | 0.00 | -0.208 | 0.169 | {'continuation': 2, 'prose': 1} |
| degradation|L30_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.125 | 0.165 | {'continuation': 2, 'prose': 1} |
| degradation|L32_top1 | 0.000 | 1.000 | 1.00 | 0.00 | -0.167 | -0.640 | {'continuation': 2, 'prose': 1} |
| degradation|L32_top4 | 0.000 | 1.000 | 1.00 | 0.00 | -0.208 | -0.731 | {'continuation': 2, 'prose': 1} |


## Source Attention

### deepseek7b

| variant | rows | value_in_record | record | question | instruction | answer | self | object | relation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short_only | 2304 | 0.066 | 0.463 | 0.190 | 0.156 | 0.190 | 0.118 | 0.059 | 0.042 |
| terse_no_explain | 2304 | 0.152 | 0.561 | 0.180 | 0.089 | 0.170 | 0.101 | 0.081 | 0.058 |

| high-value head | value_in_record | record | instruction | answer | self | top_tokens |
|---|---:|---:|---:|---:|---:|---|
| terse_no_explain|L23H11 | 0.624 | 0.951 | 0.011 | 0.004 | 0.003 | {' brush': 1, ' library': 2, ' market': 2, ' nonce': 38, ' north': 3, ' silver': 1, ' south': 4, ' square': 2, ' west': 3, 'Record': 1} |
| terse_no_explain|L26H19 | 0.433 | 0.812 | 0.048 | 0.009 | 0.008 | {' blue': 6, ' nonce': 38, ' san': 1, 'Record': 27} |
| terse_no_explain|L26H15 | 0.412 | 0.856 | 0.042 | 0.009 | 0.006 | {' knife': 1, ' nonce': 38, ' vor': 1, ' wrench': 1, 'Record': 31} |
| terse_no_explain|L27H2 | 0.404 | 0.745 | 0.040 | 0.031 | 0.024 | {' anchor': 1, ' bucket': 1, ' hammer': 3, ' knife': 1, ' nonce': 38, ' spiral': 2, ' the': 1, ' vor': 3, ' wrench': 1, 'Record': 17} |
| terse_no_explain|L23H19 | 0.347 | 0.738 | 0.123 | 0.036 | 0.034 | {' nonce': 38, ' not': 1, '.\n': 1, 'Record': 32} |
| terse_no_explain|L27H17 | 0.347 | 0.584 | 0.135 | 0.055 | 0.054 | {' nonce': 38, ' the': 15, ':': 5, 'Record': 14} |
| short_only|L27H2 | 0.294 | 0.666 | 0.103 | 0.039 | 0.024 | {' bucket': 1, ' garden': 1, ' knife': 1, ' nonce': 36, ' sar': 1, ' spoon': 1, ' temple': 1, ' vor': 3, ' wrench': 1, 'Record': 25} |
| short_only|L26H19 | 0.227 | 0.686 | 0.088 | 0.012 | 0.010 | {' blue': 5, ' cross': 2, ' nonce': 30, ' pr': 1, ' san': 1, 'Record': 33} |
| terse_no_explain|L24H21 | 0.207 | 0.868 | 0.020 | 0.007 | 0.006 | {' library': 1, ' market': 2, ' nonce': 5, ' orange': 1, ' silver': 1, ' station': 1, ' symbol': 5, ' tool': 9, ' yellow': 2, 'Record': 40} |
| terse_no_explain|L23H18 | 0.199 | 0.880 | 0.015 | 0.041 | 0.040 | {' black': 1, ' far': 2, ' north': 3, ' orange': 1, ' purple': 1, ' silver': 1, ' south': 4, ' west': 3, ';': 2, 'Record': 50} |
| terse_no_explain|L23H6 | 0.197 | 0.631 | 0.031 | 0.108 | 0.102 | {' blue': 1, ' color': 10, ' nonce': 18, ' silver': 1, ' symbol': 3, ' temple': 1, ' west': 2, ' yellow': 2, ':': 10, 'Record': 20} |
| terse_no_explain|L23H5 | 0.190 | 0.344 | 0.131 | 0.260 | 0.157 | {' library': 2, ' market': 3, ' nonce': 19, ' station': 1, ' temple': 1, ':': 20, '?\n': 15, 'Answer': 6, 'Instruction': 2, 'Question': 3} |
| terse_no_explain|L26H14 | 0.179 | 0.757 | 0.063 | 0.017 | 0.011 | {' library': 2, ' market': 2, ' nonce': 13, ' north': 3, ' south': 4, ' square': 4, ' to': 2, ' tool': 2, ' west': 3, 'Record': 23} |
| terse_no_explain|L26H0 | 0.171 | 0.763 | 0.068 | 0.035 | 0.033 | {' arrow': 2, ' brush': 1, ' bucket': 1, ' hammer': 3, ' knife': 1, ' library': 2, ' spiral': 2, ' station': 1, ' wrench': 1, 'Record': 55} |
| short_only|L27H17 | 0.168 | 0.413 | 0.236 | 0.053 | 0.051 | {' nonce': 26, ' the': 22, ' west': 3, '?\n': 1, 'Record': 20} |
| short_only|L24H21 | 0.163 | 0.842 | 0.034 | 0.007 | 0.007 | {' brush': 1, ' circle': 1, ' green': 1, ' nonce': 3, ' silver': 1, ' symbol': 2, 'Record': 63} |

### glm4

| variant | rows | value_in_record | record | question | instruction | answer | self | object | relation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short_only | 160 | 0.019 | 0.265 | 0.059 | 0.242 | 0.434 | 0.334 | 0.033 | 0.004 |
| terse_no_explain | 160 | 0.028 | 0.307 | 0.057 | 0.170 | 0.466 | 0.340 | 0.033 | 0.006 |

| high-value head | value_in_record | record | instruction | answer | self | top_tokens |
|---|---:|---:|---:|---:|---:|---|
| terse_no_explain|L34H8 | 0.253 | 0.809 | 0.046 | 0.073 | 0.073 | {' hammer': 1, ' orange': 1, ' spoon': 1, '.\n': 1, 'Record': 1} |
| terse_no_explain|L34H4 | 0.188 | 0.845 | 0.040 | 0.048 | 0.048 | {' sar': 1, ' silver': 1, ' spoon': 1, ' square': 1, 'Record': 1} |
| short_only|L34H8 | 0.174 | 0.809 | 0.055 | 0.055 | 0.055 | {' hammer': 1, ' ladder': 1, ' spiral': 1, 'Record': 2} |
| terse_no_explain|L34H12 | 0.172 | 0.793 | 0.068 | 0.085 | 0.085 | {' hammer': 1, ' sar': 1, ' square': 1, 'Record': 2} |
| terse_no_explain|L34H20 | 0.160 | 0.720 | 0.103 | 0.076 | 0.073 | {' spoon': 1, ' wave': 1, ' yellow': 1, 'Record': 2} |
| short_only|L34H4 | 0.158 | 0.828 | 0.048 | 0.039 | 0.039 | {' sar': 1, ' silver': 1, ' spoon': 1, ' square': 1, 'Record': 1} |
| short_only|L34H12 | 0.142 | 0.792 | 0.079 | 0.063 | 0.063 | {' hammer': 1, ' sar': 1, ' square': 1, 'Record': 2} |
| short_only|L34H20 | 0.050 | 0.451 | 0.417 | 0.059 | 0.057 | {' only': 5} |
| terse_no_explain|L35H30 | 0.038 | 0.736 | 0.135 | 0.087 | 0.083 | {' library': 1, 'Record': 4} |
| short_only|L35H30 | 0.027 | 0.778 | 0.123 | 0.050 | 0.047 | {':': 2, 'Record': 3} |
| terse_no_explain|L35H28 | 0.025 | 0.736 | 0.131 | 0.054 | 0.053 | {':': 2, 'Record': 3} |
| terse_no_explain|L36H25 | 0.018 | 0.458 | 0.119 | 0.306 | 0.158 | {' is': 1, ' value': 1, ':': 1, 'Answer': 1, 'Record': 1} |
| short_only|L35H28 | 0.016 | 0.652 | 0.234 | 0.039 | 0.038 | {' value': 1, ':': 4} |
| terse_no_explain|L36H13 | 0.013 | 0.581 | 0.216 | 0.036 | 0.032 | {'Record': 5} |
| short_only|L36H13 | 0.010 | 0.428 | 0.360 | 0.031 | 0.024 | {'.\n': 3, 'Record': 2} |
| short_only|L36H25 | 0.008 | 0.295 | 0.485 | 0.175 | 0.120 | {' value': 2, ' with': 1, ':': 2} |

### qwen3

| variant | rows | value_in_record | record | question | instruction | answer | self | object | relation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short_only | 96 | 0.125 | 0.643 | 0.056 | 0.049 | 0.252 | 0.144 | 0.026 | 0.029 |
| terse_no_explain | 96 | 0.139 | 0.609 | 0.053 | 0.058 | 0.280 | 0.153 | 0.026 | 0.026 |

| high-value head | value_in_record | record | instruction | answer | self | top_tokens |
|---|---:|---:|---:|---:|---:|---|
| terse_no_explain|L31H28 | 0.673 | 0.964 | 0.010 | 0.002 | 0.001 | {' nonce': 3} |
| terse_no_explain|L32H8 | 0.651 | 0.982 | 0.006 | 0.001 | 0.001 | {' nonce': 3} |
| terse_no_explain|L31H14 | 0.598 | 0.978 | 0.010 | 0.001 | 0.001 | {' nonce': 3} |
| short_only|L32H8 | 0.576 | 0.978 | 0.002 | 0.001 | 0.001 | {' nonce': 3} |
| short_only|L31H28 | 0.524 | 0.952 | 0.011 | 0.002 | 0.002 | {' nonce': 3} |
| terse_no_explain|L34H1 | 0.503 | 0.928 | 0.017 | 0.019 | 0.018 | {' nonce': 3} |
| short_only|L31H14 | 0.494 | 0.984 | 0.003 | 0.002 | 0.002 | {' nonce': 1, 'Record': 2} |
| short_only|L34H1 | 0.493 | 0.941 | 0.010 | 0.015 | 0.015 | {' nonce': 3} |
| terse_no_explain|L30H27 | 0.384 | 0.928 | 0.043 | 0.006 | 0.005 | {' nonce': 1, 'Record': 2} |
| short_only|L30H27 | 0.333 | 0.962 | 0.012 | 0.003 | 0.003 | {'Record': 3} |
| terse_no_explain|L33H30 | 0.322 | 0.859 | 0.019 | 0.013 | 0.008 | {' nonce': 1, 'Record': 2} |
| terse_no_explain|L34H2 | 0.316 | 0.835 | 0.010 | 0.018 | 0.018 | {'Record': 3} |
| short_only|L33H30 | 0.315 | 0.883 | 0.020 | 0.007 | 0.006 | {' nonce': 3} |
| short_only|L34H2 | 0.301 | 0.853 | 0.008 | 0.017 | 0.017 | {'Record': 3} |
| terse_no_explain|L34H0 | 0.234 | 0.866 | 0.017 | 0.025 | 0.024 | {'Record': 3} |
| short_only|L34H0 | 0.217 | 0.880 | 0.007 | 0.025 | 0.025 | {'Record': 3} |

