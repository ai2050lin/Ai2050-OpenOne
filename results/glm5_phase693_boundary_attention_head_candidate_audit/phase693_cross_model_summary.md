# Phase 693 Boundary Attention Head Candidate Audit

- generated: `2026-06-26 15:20:07`

| model | pairs | target | scan_layers | best_restore | repair | target_gain | rank_effect | best_degrade | drop | target_loss | rank_effect |
|---|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|
| deepseek7b | 72 | L26_layer_input | [13, 14, 15, 16, 17, 18] | restore|all_top16 | 0.431 | 7.889 | 141.43 | degradation|all_top16 | 0.625 | 6.360 | 19.28 |
| glm4 | 5 | L38_layer_input | [23, 24, 25, 26, 27, 28, 29, 30] | restore|all_top16 | 0.800 | 0.618 | 0.80 | degradation|all_top16 | 0.000 | 0.508 | 0.00 |
| qwen3 | 3 | L33_layer_input | [18, 19, 20, 21, 22, 23, 24, 25] | restore|late_top16 | 1.000 | 5.407 | 1.00 | degradation|late_top8 | 1.000 | 5.389 | 1.33 |

## Top Candidate Heads

### deepseek7b

| head | mean_direct_effect | mean_delta_norm |
|---|---:|---:|
| L18H6 | 0.215 | 3.459 |
| L14H17 | 0.214 | 4.268 |
| L13H20 | 0.191 | 6.129 |
| L14H15 | 0.188 | 4.573 |
| L18H19 | 0.180 | 2.759 |
| L13H13 | 0.125 | 3.776 |
| L13H4 | 0.124 | 3.896 |
| L16H27 | 0.124 | 3.004 |
| L13H15 | 0.106 | 2.784 |
| L18H10 | 0.104 | 1.601 |
| L18H1 | 0.094 | 2.995 |
| L13H19 | 0.088 | 5.185 |
| L18H7 | 0.079 | 8.506 |
| L16H15 | 0.074 | 3.971 |
| L15H25 | 0.070 | 1.564 |
| L13H0 | 0.057 | 2.646 |
| L17H11 | 0.054 | 0.899 |
| L16H16 | 0.053 | 1.410 |
| L18H12 | 0.053 | 1.039 |
| L15H24 | 0.053 | 1.475 |
| L16H25 | 0.049 | 2.965 |
| L13H23 | 0.048 | 2.781 |
| L18H5 | 0.045 | 1.351 |
| L15H7 | 0.045 | 1.147 |

### glm4

| head | mean_direct_effect | mean_delta_norm |
|---|---:|---:|
| L29H28 | 0.111 | 18.339 |
| L29H18 | 0.083 | 22.157 |
| L26H15 | 0.075 | 8.488 |
| L26H9 | 0.062 | 6.258 |
| L30H20 | 0.031 | 6.493 |
| L27H24 | 0.028 | 11.170 |
| L26H17 | 0.027 | 9.125 |
| L29H21 | 0.023 | 7.841 |
| L28H2 | 0.022 | 5.392 |
| L28H6 | 0.022 | 4.261 |
| L29H29 | 0.021 | 4.545 |
| L24H9 | 0.019 | 3.272 |
| L27H7 | 0.018 | 4.194 |
| L24H12 | 0.014 | 4.348 |
| L29H26 | 0.013 | 19.743 |
| L29H14 | 0.013 | 6.758 |
| L26H12 | 0.011 | 3.333 |
| L26H31 | 0.011 | 5.150 |
| L28H12 | 0.010 | 3.292 |
| L28H26 | 0.010 | 6.437 |
| L29H15 | 0.010 | 6.896 |
| L27H8 | 0.010 | 1.194 |
| L27H25 | 0.010 | 8.977 |
| L30H13 | 0.010 | 3.791 |

### qwen3

| head | mean_direct_effect | mean_delta_norm |
|---|---:|---:|
| L22H16 | 0.119 | 2.578 |
| L23H18 | 0.088 | 2.460 |
| L20H16 | 0.087 | 2.768 |
| L23H27 | 0.087 | 1.379 |
| L23H2 | 0.085 | 1.999 |
| L24H28 | 0.066 | 0.921 |
| L25H30 | 0.064 | 1.408 |
| L19H20 | 0.045 | 1.875 |
| L24H21 | 0.044 | 0.916 |
| L24H8 | 0.044 | 1.007 |
| L24H17 | 0.042 | 1.492 |
| L24H23 | 0.042 | 0.895 |
| L23H8 | 0.038 | 1.667 |
| L23H29 | 0.038 | 0.936 |
| L18H24 | 0.037 | 1.526 |
| L25H12 | 0.036 | 1.798 |
| L22H14 | 0.035 | 1.738 |
| L19H22 | 0.032 | 0.937 |
| L18H16 | 0.031 | 1.571 |
| L18H22 | 0.031 | 0.658 |
| L24H11 | 0.031 | 0.866 |
| L20H12 | 0.031 | 1.403 |
| L21H18 | 0.030 | 1.657 |
| L25H28 | 0.028 | 0.755 |


## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|all_top16 | 0.431 | 0.431 | 26.26 | 141.43 | 7.889 | 0.723 | 3.971 | {'continuation': 1, 'prose': 71} |
| restore|L13_all_heads | 0.264 | 0.264 | 28.35 | 139.35 | 5.663 | 0.520 | 2.816 | {'continuation': 9, 'prose': 63} |
| restore|all_top8 | 0.264 | 0.264 | 59.74 | 107.96 | 4.157 | 0.393 | 2.434 | {'prose': 72} |
| restore|early_top8 | 0.236 | 0.236 | 62.17 | 105.53 | 5.156 | 0.411 | 2.294 | {'prose': 72} |
| restore|late_top8 | 0.222 | 0.222 | 54.15 | 113.54 | 3.421 | 0.398 | 2.317 | {'continuation': 1, 'prose': 71} |
| restore|L18_all_heads | 0.194 | 0.194 | 78.43 | 89.26 | 1.740 | 0.159 | 1.629 | {'continuation': 2, 'prose': 70} |
| restore|late_random | 0.194 | 0.194 | 60.22 | 107.47 | 1.030 | 0.098 | 1.713 | {'prose': 72} |
| restore|late_top16 | 0.139 | 0.139 | 108.31 | 59.39 | 2.069 | 0.290 | 1.320 | {'continuation': 1, 'prose': 71} |
| restore|early_top16 | 0.125 | 0.125 | 78.29 | 89.40 | 4.874 | 0.517 | 1.523 | {'prose': 72} |
| restore|L13_top4 | 0.097 | 0.097 | 73.83 | 93.86 | 4.506 | 0.267 | 1.670 | {'continuation': 1, 'prose': 71} |
| restore|L18_top4 | 0.097 | 0.097 | 77.76 | 89.93 | 2.194 | 0.166 | 1.249 | {'prose': 72} |
| restore|L16_top2 | 0.069 | 0.069 | 95.47 | 72.22 | 1.461 | 0.124 | 1.074 | {'prose': 72} |
| restore|L13_top2 | 0.056 | 0.056 | 79.86 | 87.83 | 3.391 | 0.197 | 1.443 | {'continuation': 2, 'prose': 70} |
| restore|L13_top1 | 0.056 | 0.056 | 104.18 | 63.51 | 2.927 | 0.234 | 0.911 | {'continuation': 2, 'prose': 70} |
| restore|L16_top4 | 0.056 | 0.056 | 113.29 | 54.40 | 1.145 | 0.146 | 0.740 | {'continuation': 1, 'prose': 71} |
| restore|L15_random4 | 0.056 | 0.056 | 125.22 | 42.47 | 0.294 | -0.118 | 0.506 | {'prose': 72} |
| restore|early_random | 0.042 | 0.042 | 133.81 | 33.89 | 0.267 | 0.037 | 0.262 | {'continuation': 1, 'prose': 71} |
| restore|L18_random4 | 0.042 | 0.042 | 189.42 | -21.72 | 0.039 | 0.134 | -0.010 | {'prose': 72} |
| restore|L14_all_heads | 0.042 | 0.042 | 175.46 | -7.76 | -0.028 | 0.191 | 0.248 | {'prose': 72} |
| restore|L13_random4 | 0.028 | 0.028 | 157.74 | 9.96 | 0.471 | -0.013 | 0.368 | {'prose': 72} |
| restore|L16_random4 | 0.028 | 0.028 | 104.72 | 62.97 | 0.471 | 0.162 | 0.526 | {'prose': 72} |
| restore|L16_top1 | 0.028 | 0.028 | 121.49 | 46.21 | 0.197 | 0.049 | 0.693 | {'prose': 72} |
| restore|L17_top1 | 0.028 | 0.028 | 161.42 | 6.28 | 0.094 | 0.007 | 0.096 | {'prose': 72} |
| restore|L16_all_heads | 0.028 | 0.028 | 139.19 | 28.50 | -0.571 | 0.168 | -0.135 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|all_top16 | 0.800 | 0.800 | 1.20 | 0.80 | 0.618 | 0.395 | 0.388 | {'continuation': 5} |
| restore|late_top16 | 0.800 | 0.800 | 1.20 | 0.80 | 0.553 | 0.359 | 0.438 | {'continuation': 5} |
| restore|all_top8 | 0.800 | 0.800 | 1.20 | 0.80 | 0.443 | 0.276 | 0.325 | {'continuation': 5} |
| restore|late_top8 | 0.800 | 0.800 | 1.20 | 0.80 | 0.416 | 0.259 | 0.325 | {'continuation': 5} |
| restore|L27_all_heads | 0.800 | 0.800 | 1.20 | 0.80 | 0.343 | 0.224 | 0.237 | {'continuation': 5} |
| restore|early_top16 | 0.800 | 0.800 | 1.20 | 0.80 | 0.303 | 0.192 | 0.362 | {'continuation': 5} |
| restore|L27_top4 | 0.800 | 0.800 | 1.20 | 0.80 | 0.200 | 0.120 | 0.212 | {'continuation': 5} |
| restore|L24_all_heads | 0.800 | 0.800 | 1.20 | 0.80 | 0.073 | 0.028 | 0.237 | {'continuation': 5} |
| restore|L23_all_heads | 0.800 | 0.800 | 1.20 | 0.80 | 0.038 | 0.045 | 0.200 | {'continuation': 5} |
| restore|all_random | 0.600 | 0.600 | 1.40 | 0.60 | 0.098 | 0.120 | 0.188 | {'continuation': 5} |
| restore|L27_top2 | 0.600 | 0.600 | 1.40 | 0.60 | 0.092 | 0.049 | 0.125 | {'continuation': 5} |
| restore|L29_top4 | 0.400 | 0.400 | 1.60 | 0.40 | 0.307 | 0.192 | 0.237 | {'continuation': 5} |
| restore|L29_all_heads | 0.400 | 0.400 | 1.60 | 0.40 | 0.254 | 0.168 | 0.138 | {'continuation': 5} |
| restore|L27_top1 | 0.400 | 0.400 | 1.60 | 0.40 | 0.079 | 0.046 | 0.113 | {'continuation': 5} |
| restore|L23_top4 | 0.400 | 0.400 | 1.60 | 0.40 | 0.024 | 0.007 | 0.200 | {'continuation': 5} |
| restore|L23_top2 | 0.400 | 0.400 | 1.60 | 0.40 | 0.002 | -0.003 | 0.175 | {'continuation': 5} |
| restore|L29_top2 | 0.200 | 0.200 | 1.80 | 0.20 | 0.297 | 0.175 | 0.188 | {'continuation': 5} |
| restore|early_top8 | 0.200 | 0.200 | 1.80 | 0.20 | 0.214 | 0.114 | 0.125 | {'continuation': 5} |
| restore|L25_all_heads | 0.200 | 0.200 | 1.80 | 0.20 | 0.134 | 0.121 | 0.150 | {'continuation': 5} |
| restore|L24_top4 | 0.200 | 0.200 | 1.80 | 0.20 | 0.129 | 0.090 | 0.087 | {'continuation': 5} |
| restore|L23_top1 | 0.200 | 0.200 | 1.80 | 0.20 | 0.062 | 0.031 | 0.037 | {'continuation': 5} |
| restore|L30_all_heads | 0.200 | 0.200 | 1.80 | 0.20 | 0.039 | 0.022 | 0.037 | {'continuation': 5} |
| restore|L26_top4 | 0.200 | 0.200 | 1.80 | 0.20 | 0.025 | 0.004 | 0.000 | {'continuation': 5} |
| restore|L30_random4 | 0.200 | 0.200 | 1.80 | 0.20 | -0.016 | -0.017 | -0.013 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|late_top16 | 1.000 | 1.000 | 1.00 | 1.00 | 5.407 | 0.767 | 1.708 | {'continuation': 2, 'prose': 1} |
| restore|all_top16 | 0.667 | 0.667 | 1.33 | 0.67 | 5.242 | 0.730 | 1.833 | {'continuation': 2, 'prose': 1} |
| restore|all_top8 | 0.667 | 0.667 | 1.33 | 0.67 | 3.849 | 0.527 | 1.125 | {'continuation': 2, 'prose': 1} |
| restore|late_top8 | 0.667 | 0.667 | 1.33 | 0.67 | 3.684 | 0.509 | 0.792 | {'continuation': 2, 'prose': 1} |
| restore|L24_top4 | 0.667 | 0.667 | 1.33 | 0.67 | 2.588 | 0.348 | 0.667 | {'continuation': 2, 'prose': 1} |
| restore|L24_top2 | 0.667 | 0.667 | 1.33 | 0.67 | 2.481 | 0.341 | 0.750 | {'continuation': 2, 'prose': 1} |
| restore|L24_all_heads | 0.667 | 0.667 | 1.33 | 0.67 | 2.452 | 0.329 | 1.000 | {'continuation': 2, 'prose': 1} |
| restore|L23_top4 | 0.667 | 0.667 | 1.33 | 0.67 | 2.377 | 0.321 | 0.833 | {'continuation': 2, 'prose': 1} |
| restore|L23_top1 | 0.667 | 0.667 | 1.33 | 0.67 | 2.324 | 0.313 | 0.750 | {'continuation': 2, 'prose': 1} |
| restore|L18_all_heads | 0.667 | 0.667 | 1.33 | 0.67 | 2.314 | 0.306 | 0.792 | {'continuation': 2, 'prose': 1} |
| restore|L23_top2 | 0.667 | 0.667 | 1.33 | 0.67 | 2.191 | 0.294 | 0.833 | {'continuation': 2, 'prose': 1} |
| restore|L24_top1 | 0.667 | 0.667 | 1.33 | 0.67 | 1.732 | 0.237 | 0.458 | {'continuation': 2, 'prose': 1} |
| restore|L20_all_heads | 0.667 | 0.667 | 1.33 | 0.67 | 1.517 | 0.207 | 0.292 | {'continuation': 2, 'prose': 1} |
| restore|all_random | 0.667 | 0.667 | 1.33 | 0.67 | 1.500 | 0.190 | 0.375 | {'continuation': 1, 'prose': 2} |
| restore|L22_random4 | 0.667 | 0.667 | 1.33 | 0.67 | 1.117 | 0.150 | 0.208 | {'continuation': 1, 'prose': 2} |
| restore|L20_top2 | 0.667 | 0.667 | 1.33 | 0.67 | 1.018 | 0.124 | 0.208 | {'continuation': 1, 'prose': 2} |
| restore|L18_random4 | 0.667 | 0.667 | 1.33 | 0.67 | 0.971 | 0.129 | 0.292 | {'continuation': 2, 'prose': 1} |
| restore|L21_top4 | 0.667 | 0.667 | 1.33 | 0.67 | 0.704 | 0.101 | 0.292 | {'continuation': 1, 'prose': 2} |
| restore|L19_top1 | 0.667 | 0.667 | 1.33 | 0.67 | 0.400 | 0.037 | 0.125 | {'continuation': 1, 'prose': 2} |
| restore|L23_all_heads | 0.333 | 0.333 | 1.67 | 0.33 | 2.341 | 0.313 | 0.875 | {'continuation': 2, 'prose': 1} |
| restore|L19_all_heads | 0.333 | 0.333 | 1.67 | 0.33 | 2.022 | 0.272 | 0.625 | {'continuation': 2, 'prose': 1} |
| restore|L20_top1 | 0.333 | 0.333 | 1.67 | 0.33 | 1.458 | 0.187 | 0.542 | {'continuation': 2, 'prose': 1} |
| restore|L25_top2 | 0.333 | 0.333 | 1.67 | 0.33 | 1.401 | 0.200 | 0.458 | {'continuation': 2, 'prose': 1} |
| restore|L21_random4 | 0.333 | 0.333 | 1.67 | 0.33 | 1.259 | 0.156 | 0.083 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|all_top16 | 0.625 | 0.375 | 20.28 | 19.28 | 6.360 | 0.635 | 2.478 | {'prose': 72} |
| degradation|early_top8 | 0.542 | 0.458 | 5.24 | 4.24 | 4.439 | 0.305 | 1.845 | {'prose': 72} |
| degradation|all_top8 | 0.472 | 0.528 | 4.50 | 3.50 | 3.028 | 0.382 | 1.624 | {'prose': 72} |
| degradation|L13_all_heads | 0.458 | 0.542 | 2.78 | 1.78 | 4.996 | 0.315 | 1.556 | {'prose': 72} |
| degradation|early_top16 | 0.444 | 0.556 | 3.07 | 2.07 | 3.104 | 0.429 | 1.277 | {'prose': 72} |
| degradation|L13_top4 | 0.250 | 0.750 | 1.60 | 0.60 | 2.644 | 0.133 | 0.769 | {'prose': 72} |
| degradation|late_random | 0.236 | 0.764 | 1.47 | 0.47 | 0.880 | -0.008 | 0.696 | {'prose': 72} |
| degradation|late_top8 | 0.208 | 0.792 | 1.67 | 0.67 | 2.633 | 0.343 | 0.668 | {'prose': 72} |
| degradation|L13_top2 | 0.181 | 0.819 | 1.46 | 0.46 | 2.293 | 0.080 | 0.596 | {'prose': 72} |
| degradation|L14_top2 | 0.153 | 0.847 | 1.28 | 0.28 | -0.313 | 0.184 | 0.352 | {'prose': 72} |
| degradation|L13_top1 | 0.125 | 0.875 | 1.28 | 0.28 | 1.788 | 0.072 | 0.451 | {'prose': 72} |
| degradation|L18_top4 | 0.125 | 0.875 | 1.19 | 0.19 | 1.451 | 0.152 | 0.363 | {'prose': 72} |
| degradation|L14_top4 | 0.125 | 0.875 | 1.25 | 0.25 | -0.368 | 0.179 | 0.308 | {'prose': 72} |
| degradation|L18_all_heads | 0.111 | 0.889 | 1.24 | 0.24 | 1.219 | 0.208 | 0.199 | {'prose': 72} |
| degradation|L15_random4 | 0.097 | 0.903 | 1.18 | 0.18 | 0.154 | -0.098 | 0.298 | {'prose': 72} |
| degradation|L13_random4 | 0.083 | 0.917 | 1.11 | 0.11 | 0.078 | 0.023 | 0.227 | {'prose': 72} |
| degradation|late_top16 | 0.069 | 0.931 | 1.35 | 0.35 | 2.004 | 0.179 | 0.073 | {'prose': 72} |
| degradation|L15_all_heads | 0.069 | 0.931 | 1.07 | 0.07 | 1.297 | 0.177 | 0.102 | {'prose': 72} |
| degradation|L16_top2 | 0.069 | 0.931 | 1.11 | 0.11 | 0.877 | 0.152 | 0.338 | {'prose': 72} |
| degradation|L18_top2 | 0.069 | 0.931 | 1.10 | 0.10 | -0.068 | -0.019 | 0.032 | {'prose': 72} |
| degradation|L17_top4 | 0.056 | 0.944 | 1.06 | 0.06 | 1.027 | -0.073 | -0.149 | {'prose': 72} |
| degradation|L17_top2 | 0.056 | 0.944 | 1.06 | 0.06 | 0.634 | -0.101 | -0.059 | {'prose': 72} |
| degradation|L16_top4 | 0.056 | 0.944 | 1.08 | 0.08 | 0.586 | 0.162 | 0.234 | {'prose': 72} |
| degradation|L14_top1 | 0.056 | 0.944 | 1.10 | 0.10 | -0.683 | 0.053 | 0.108 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|all_top16 | 0.000 | 1.000 | 1.00 | 0.00 | 0.508 | 0.319 | 0.125 | {'prose': 5} |
| degradation|late_top16 | 0.000 | 1.000 | 1.00 | 0.00 | 0.443 | 0.282 | 0.100 | {'prose': 5} |
| degradation|early_top16 | 0.000 | 1.000 | 1.00 | 0.00 | 0.394 | 0.264 | 0.225 | {'prose': 5} |
| degradation|all_top8 | 0.000 | 1.000 | 1.00 | 0.00 | 0.326 | 0.197 | 0.087 | {'prose': 5} |
| degradation|late_top8 | 0.000 | 1.000 | 1.00 | 0.00 | 0.259 | 0.171 | 0.025 | {'prose': 5} |
| degradation|early_top8 | 0.000 | 1.000 | 1.00 | 0.00 | 0.247 | 0.143 | 0.050 | {'prose': 5} |
| degradation|L27_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | 0.245 | 0.159 | 0.025 | {'prose': 5} |
| degradation|L23_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | 0.219 | 0.163 | 0.163 | {'continuation': 1, 'prose': 4} |
| degradation|L29_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.161 | 0.111 | 0.050 | {'prose': 5} |
| degradation|L27_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.145 | 0.083 | -0.013 | {'prose': 5} |
| degradation|L26_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.125 | 0.084 | 0.013 | {'prose': 5} |
| degradation|L29_top2 | 0.000 | 1.000 | 1.00 | 0.00 | 0.122 | 0.082 | 0.000 | {'prose': 5} |
| degradation|L29_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | 0.120 | 0.079 | -0.050 | {'prose': 5} |
| degradation|L26_top2 | 0.000 | 1.000 | 1.00 | 0.00 | 0.117 | 0.071 | -0.037 | {'prose': 5} |
| degradation|L23_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.099 | 0.066 | 0.087 | {'prose': 5} |
| degradation|L28_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | 0.080 | 0.047 | -0.037 | {'prose': 5} |
| degradation|L28_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.065 | 0.043 | 0.013 | {'prose': 5} |
| degradation|L27_top2 | 0.000 | 1.000 | 1.00 | 0.00 | 0.059 | 0.039 | -0.025 | {'prose': 5} |
| degradation|L23_top2 | 0.000 | 1.000 | 1.00 | 0.00 | 0.056 | 0.051 | 0.087 | {'prose': 5} |
| degradation|L27_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.054 | 0.038 | -0.050 | {'prose': 5} |
| degradation|L26_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.053 | 0.038 | -0.050 | {'prose': 5} |
| degradation|L24_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.043 | 0.045 | 0.000 | {'prose': 5} |
| degradation|L29_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.041 | 0.035 | -0.013 | {'prose': 5} |
| degradation|L23_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.031 | 0.024 | 0.000 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|late_top8 | 1.000 | 0.000 | 2.33 | 1.33 | 5.389 | 0.769 | 2.083 | {'continuation': 1, 'prose': 2} |
| degradation|late_top16 | 0.667 | 0.333 | 2.00 | 1.00 | 6.016 | 0.863 | 2.583 | {'continuation': 1, 'prose': 2} |
| degradation|all_top16 | 0.667 | 0.333 | 1.67 | 0.67 | 4.883 | 0.688 | 2.167 | {'continuation': 1, 'prose': 2} |
| degradation|all_top8 | 0.333 | 0.667 | 1.33 | 0.33 | 3.792 | 0.547 | 1.625 | {'continuation': 1, 'prose': 2} |
| degradation|all_random | 0.333 | 0.667 | 1.33 | 0.33 | 2.780 | 0.380 | 1.750 | {'continuation': 2, 'prose': 1} |
| degradation|L24_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 2.426 | 0.333 | 1.000 | {'continuation': 2, 'prose': 1} |
| degradation|L24_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | 2.167 | 0.294 | 1.292 | {'continuation': 2, 'prose': 1} |
| degradation|L24_top2 | 0.000 | 1.000 | 1.00 | 0.00 | 2.123 | 0.292 | 0.750 | {'continuation': 2, 'prose': 1} |
| degradation|L24_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 1.910 | 0.266 | 0.667 | {'continuation': 2, 'prose': 1} |
| degradation|L20_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | 1.554 | 0.228 | 0.708 | {'continuation': 1, 'prose': 2} |
| degradation|L23_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | 1.481 | 0.194 | 1.250 | {'prose': 3} |
| degradation|L23_top2 | 0.000 | 1.000 | 1.00 | 0.00 | 1.341 | 0.180 | 0.958 | {'continuation': 1, 'prose': 2} |
| degradation|L23_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 1.199 | 0.163 | 0.750 | {'continuation': 1, 'prose': 2} |
| degradation|L23_top4 | 0.000 | 1.000 | 1.00 | 0.00 | 1.163 | 0.161 | 0.958 | {'continuation': 1, 'prose': 2} |
| degradation|L22_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | 0.858 | 0.130 | 0.625 | {'continuation': 1, 'prose': 2} |
| degradation|L20_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.562 | 0.072 | 0.417 | {'continuation': 2, 'prose': 1} |
| degradation|L18_random4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.498 | 0.072 | 0.333 | {'continuation': 2, 'prose': 1} |
| degradation|L22_random4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.495 | 0.067 | 0.458 | {'continuation': 2, 'prose': 1} |
| degradation|L18_top2 | 0.000 | 1.000 | 1.00 | 0.00 | 0.434 | 0.062 | 0.083 | {'continuation': 2, 'prose': 1} |
| degradation|L18_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.280 | 0.043 | 0.167 | {'continuation': 1, 'prose': 2} |
| degradation|L18_all_heads | 0.000 | 1.000 | 1.00 | 0.00 | 0.263 | 0.031 | 0.250 | {'continuation': 2, 'prose': 1} |
| degradation|L25_top1 | 0.000 | 1.000 | 1.00 | 0.00 | 0.236 | 0.036 | 0.250 | {'continuation': 2, 'prose': 1} |
| degradation|L21_random4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.213 | 0.030 | 0.000 | {'continuation': 2, 'prose': 1} |
| degradation|L20_random4 | 0.000 | 1.000 | 1.00 | 0.00 | 0.211 | 0.020 | 0.167 | {'continuation': 2, 'prose': 1} |

