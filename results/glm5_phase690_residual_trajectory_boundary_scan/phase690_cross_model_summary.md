# Phase 690 Residual Trajectory Boundary Scan

- generated: `2026-06-26 14:14:34`

| model | pairs | early_target | final_target | layer_out_scan | best_restore | repair | final_gain | rank_effect | best_degrade | drop | final_loss | rank_effect |
|---|---:|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| deepseek7b | 72 | L18_layer_input | L26_layer_input | [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] | restore|add_delta|L18_layer_out | 0.792 | 6.682 | 166.10 | degradation|replace_short|L13_layer_out | 0.875 | 6.198 | 24.18 |
| glm4 | 5 | L30_layer_input | L38_layer_input | [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] | restore|add_delta|L21_layer_out | 1.000 | 1.673 | 1.00 | degradation|replace_short|L27_layer_out | 0.600 | 1.554 | 0.60 |
| qwen3 | 3 | L25_layer_input | L33_layer_input | [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | restore|add_delta|L23_layer_out | 1.000 | 9.068 | 1.00 | degradation|replace_short|L23_layer_out | 1.000 | 8.964 | 1.67 |

## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | early_effect | early_frac | final_effect | final_frac | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| restore|add_delta|L18_layer_out | 0.792 | 0.792 | 1.60 | 166.10 | 0.000 | 0.000 | 6.682 | 0.864 | 5.627 | {'prose': 72} |
| restore|add_delta|L15_layer_out | 0.597 | 0.597 | 5.90 | 161.79 | -1.785 | 1.034 | 6.263 | 0.621 | 4.923 | {'prose': 72} |
| restore|add_delta|L17_layer_out | 0.569 | 0.569 | 3.69 | 164.00 | -1.625 | 1.001 | 5.118 | 0.678 | 4.651 | {'prose': 72} |
| restore|add_delta|L16_layer_out | 0.486 | 0.486 | 4.92 | 162.78 | -1.637 | 0.950 | 4.871 | 0.592 | 4.371 | {'prose': 72} |
| restore|add_delta|L14_layer_out | 0.431 | 0.431 | 11.01 | 156.68 | -1.534 | 0.775 | 5.901 | 0.624 | 4.226 | {'continuation': 3, 'prose': 69} |
| restore|add_delta|L13_layer_out | 0.389 | 0.389 | 13.99 | 153.71 | -1.243 | 0.568 | 6.540 | 0.590 | 3.923 | {'continuation': 1, 'prose': 71} |
| restore|add_delta|L18_attn_out | 0.181 | 0.181 | 78.96 | 88.74 | 0.000 | 0.000 | 1.716 | 0.180 | 1.629 | {'continuation': 2, 'prose': 70} |
| restore|add_delta|L9_layer_out | 0.111 | 0.111 | 89.44 | 78.25 | -0.407 | 0.173 | 1.329 | 0.118 | 1.447 | {'prose': 72} |
| restore|add_delta|L12_layer_out | 0.097 | 0.097 | 99.94 | 67.75 | -0.582 | 0.048 | 2.071 | 0.265 | 1.765 | {'continuation': 2, 'prose': 70} |
| restore|add_delta|L11_layer_out | 0.097 | 0.097 | 88.04 | 79.65 | -0.382 | 0.042 | 0.900 | 0.108 | 1.713 | {'continuation': 1, 'prose': 71} |
| restore|add_delta|L18_mlp_out | 0.083 | 0.083 | 47.46 | 120.24 | 0.000 | 0.000 | 4.000 | 0.103 | 1.702 | {'prose': 72} |
| restore|add_delta|L8_layer_out | 0.083 | 0.083 | 106.29 | 61.40 | -0.433 | 0.220 | 0.604 | 0.011 | 0.960 | {'prose': 72} |
| restore|add_delta|L10_layer_out | 0.056 | 0.056 | 110.93 | 56.76 | -0.645 | 0.383 | 0.923 | 0.075 | 1.294 | {'continuation': 1, 'prose': 71} |
| restore|add_delta|L17_mlp_out | 0.042 | 0.042 | 221.33 | -53.64 | -0.141 | -0.083 | -0.394 | -0.128 | -0.567 | {'continuation': 9, 'prose': 63} |
| restore|add_delta|L16_attn_out | 0.028 | 0.028 | 138.15 | 29.54 | 0.213 | -0.036 | -0.562 | 0.174 | -0.122 | {'prose': 72} |
| restore|add_delta|L17_attn_out | 0.000 | 0.000 | 269.69 | -102.00 | -0.063 | 0.081 | -0.137 | -0.198 | -0.452 | {'continuation': 3, 'prose': 69} |
| restore|add_delta|L16_mlp_out | 0.000 | 0.000 | 260.50 | -92.81 | -0.110 | 0.084 | -3.544 | -0.212 | -1.698 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | early_effect | early_frac | final_effect | final_frac | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| restore|add_delta|L21_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.674 | 1.213 | 1.673 | 1.091 | 1.325 | {'continuation': 3, 'prose': 2} |
| restore|add_delta|L22_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.580 | 1.051 | 1.557 | 0.988 | 1.337 | {'continuation': 1, 'prose': 4} |
| restore|add_delta|L30_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 0.000 | 1.546 | 0.972 | 1.550 | {'prose': 5} |
| restore|add_delta|L27_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.613 | 1.080 | 1.538 | 0.975 | 1.650 | {'continuation': 1, 'prose': 4} |
| restore|add_delta|L29_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.574 | 1.001 | 1.530 | 0.962 | 1.550 | {'prose': 5} |
| restore|add_delta|L28_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.573 | 0.997 | 1.523 | 0.955 | 1.550 | {'continuation': 1, 'prose': 4} |
| restore|add_delta|L23_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.596 | 1.056 | 1.451 | 0.927 | 1.538 | {'continuation': 1, 'prose': 4} |
| restore|add_delta|L25_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.555 | 0.979 | 1.400 | 0.892 | 1.575 | {'continuation': 1, 'prose': 4} |
| restore|add_delta|L26_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.542 | 0.947 | 1.398 | 0.886 | 1.575 | {'continuation': 1, 'prose': 4} |
| restore|add_delta|L20_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.599 | 1.060 | 1.367 | 0.873 | 1.062 | {'continuation': 4, 'prose': 1} |
| restore|add_delta|L24_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.548 | 0.963 | 1.342 | 0.852 | 1.525 | {'continuation': 1, 'prose': 4} |
| restore|add_delta|L29_mlp_out | 0.600 | 0.600 | 1.40 | 0.60 | 0.124 | 0.247 | 0.296 | 0.174 | 0.100 | {'continuation': 5} |
| restore|add_delta|L28_mlp_out | 0.600 | 0.600 | 1.40 | 0.60 | 0.080 | 0.150 | 0.141 | 0.088 | 0.150 | {'continuation': 5} |
| restore|add_delta|L30_mlp_out | 0.600 | 0.600 | 1.40 | 0.60 | 0.000 | 0.000 | 0.040 | -0.013 | 0.050 | {'continuation': 5} |
| restore|add_delta|L29_attn_out | 0.400 | 0.400 | 1.60 | 0.40 | 0.254 | 0.446 | 0.258 | 0.170 | 0.100 | {'continuation': 5} |
| restore|add_delta|L30_attn_out | 0.200 | 0.200 | 1.80 | 0.20 | 0.000 | 0.000 | 0.037 | 0.025 | 0.062 | {'continuation': 5} |
| restore|add_delta|L28_attn_out | 0.000 | 0.000 | 2.00 | 0.00 | -0.020 | -0.047 | 0.054 | 0.020 | 0.025 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | early_effect | early_frac | final_effect | final_frac | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| restore|add_delta|L23_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.364 | 0.862 | 9.068 | 1.261 | 3.458 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L24_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.416 | 0.999 | 8.631 | 1.200 | 3.583 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L25_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 0.000 | 8.097 | 1.120 | 3.333 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L20_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -0.217 | -0.671 | 7.199 | 0.988 | 2.250 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L18_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -0.224 | -0.620 | 6.977 | 0.955 | 2.375 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L19_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -0.261 | -0.720 | 6.888 | 0.947 | 2.167 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L21_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -0.159 | -0.468 | 6.764 | 0.933 | 2.125 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L22_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | -0.037 | -0.197 | 6.667 | 0.922 | 2.167 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L17_layer_out | 0.667 | 0.667 | 1.33 | 0.67 | -0.317 | -0.898 | 5.385 | 0.736 | 1.583 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L24_mlp_out | 0.667 | 0.667 | 1.33 | 0.67 | 0.323 | 0.835 | 3.143 | 0.437 | 1.042 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L24_attn_out | 0.667 | 0.667 | 1.33 | 0.67 | 0.174 | 0.451 | 2.384 | 0.319 | 1.042 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L25_mlp_out | 0.667 | 0.667 | 1.33 | 0.67 | 0.000 | 0.000 | 2.039 | 0.275 | 0.833 | {'continuation': 1, 'prose': 2} |
| restore|add_delta|L16_layer_out | 0.333 | 0.333 | 1.67 | 0.33 | -0.369 | -1.032 | 3.773 | 0.515 | 0.708 | {'continuation': 1, 'prose': 2} |
| restore|add_delta|L23_attn_out | 0.333 | 0.333 | 1.67 | 0.33 | 0.295 | 0.744 | 1.901 | 0.255 | 0.708 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L15_layer_out | 0.333 | 0.333 | 2.67 | -0.67 | -0.310 | -0.944 | -0.552 | -0.096 | -1.125 | {'continuation': 1, 'prose': 2} |
| restore|add_delta|L25_attn_out | 0.000 | 0.000 | 2.33 | -0.33 | 0.000 | 0.000 | 0.874 | 0.128 | 0.167 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L23_mlp_out | 0.000 | 0.000 | 3.00 | -1.00 | -0.170 | -0.576 | -1.194 | -0.185 | -0.458 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | early_effect | early_frac | final_effect | final_frac | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|replace_short|L13_layer_out | 0.875 | 0.125 | 25.18 | 24.18 | -0.986 | 0.608 | 6.198 | 0.574 | 3.958 | {'prose': 72} |
| degradation|replace_short|L14_layer_out | 0.847 | 0.153 | 16.57 | 15.57 | -1.244 | 0.672 | 5.350 | 0.612 | 3.738 | {'prose': 72} |
| degradation|replace_short|L18_layer_out | 0.833 | 0.167 | 34.14 | 33.14 | 0.000 | 0.000 | 6.536 | 0.871 | 4.037 | {'prose': 72} |
| degradation|replace_short|L15_layer_out | 0.792 | 0.208 | 22.94 | 21.94 | -1.607 | 0.983 | 7.644 | 0.868 | 3.862 | {'prose': 72} |
| degradation|replace_short|L16_layer_out | 0.750 | 0.250 | 18.53 | 17.53 | -1.635 | 1.009 | 6.950 | 1.000 | 3.379 | {'prose': 72} |
| degradation|replace_short|L17_layer_out | 0.681 | 0.319 | 13.99 | 12.99 | -1.625 | 1.000 | 5.206 | 0.824 | 2.957 | {'prose': 72} |
| degradation|replace_short|L11_layer_out | 0.361 | 0.639 | 2.07 | 1.07 | -0.597 | 0.521 | -0.757 | -0.240 | 1.023 | {'prose': 72} |
| degradation|replace_short|L18_mlp_out | 0.347 | 0.653 | 2.19 | 1.19 | 0.000 | 0.000 | 4.295 | 0.139 | 0.934 | {'prose': 72} |
| degradation|replace_short|L12_layer_out | 0.347 | 0.653 | 1.94 | 0.94 | -0.994 | 0.777 | -0.872 | 0.052 | 0.868 | {'prose': 72} |
| degradation|replace_short|L10_layer_out | 0.250 | 0.750 | 1.43 | 0.43 | -0.970 | 0.901 | -2.467 | -0.459 | 0.484 | {'continuation': 1, 'prose': 71} |
| degradation|replace_short|L9_layer_out | 0.194 | 0.806 | 1.26 | 0.26 | -0.692 | 0.741 | -2.491 | -0.372 | 0.257 | {'prose': 72} |
| degradation|replace_short|L18_attn_out | 0.111 | 0.889 | 1.24 | 0.24 | 0.000 | 0.000 | 1.219 | 0.208 | 0.199 | {'prose': 72} |
| degradation|replace_short|L17_mlp_out | 0.056 | 0.944 | 1.06 | 0.06 | -0.139 | -0.086 | -0.041 | 0.137 | -0.127 | {'prose': 72} |
| degradation|replace_short|L8_layer_out | 0.056 | 0.944 | 1.06 | 0.06 | -0.602 | 0.533 | -2.547 | -0.249 | -0.141 | {'prose': 72} |
| degradation|replace_short|L16_attn_out | 0.042 | 0.958 | 1.07 | 0.07 | 0.234 | -0.045 | -1.580 | -0.051 | -0.318 | {'prose': 72} |
| degradation|replace_short|L17_attn_out | 0.028 | 0.972 | 1.03 | 0.03 | -0.037 | -0.003 | 0.838 | -0.137 | -0.047 | {'prose': 72} |
| degradation|replace_short|L16_mlp_out | 0.014 | 0.986 | 1.01 | 0.01 | 0.063 | 0.038 | -4.125 | 0.041 | -0.800 | {'continuation': 2, 'prose': 70} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | early_effect | early_frac | final_effect | final_frac | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|replace_short|L27_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 0.582 | 1.006 | 1.554 | 0.998 | 1.562 | {'continuation': 5} |
| degradation|replace_short|L28_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 0.564 | 0.981 | 1.545 | 0.982 | 1.488 | {'continuation': 5} |
| degradation|replace_short|L29_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 0.574 | 1.000 | 1.543 | 0.979 | 1.488 | {'continuation': 5} |
| degradation|replace_short|L30_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 0.000 | 0.000 | 1.594 | 1.019 | 1.450 | {'continuation': 5} |
| degradation|replace_short|L25_layer_out | 0.200 | 0.800 | 1.20 | 0.20 | 0.497 | 0.850 | 1.283 | 0.813 | 1.312 | {'continuation': 5} |
| degradation|replace_short|L23_layer_out | 0.200 | 0.800 | 1.20 | 0.20 | 0.501 | 0.877 | 1.283 | 0.820 | 1.212 | {'continuation': 5} |
| degradation|replace_short|L26_layer_out | 0.200 | 0.800 | 1.20 | 0.20 | 0.501 | 0.859 | 1.277 | 0.809 | 1.325 | {'continuation': 5} |
| degradation|replace_short|L24_layer_out | 0.200 | 0.800 | 1.20 | 0.20 | 0.443 | 0.743 | 1.206 | 0.752 | 1.238 | {'continuation': 5} |
| degradation|replace_short|L22_layer_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.417 | 0.742 | 1.389 | 0.883 | 0.900 | {'continuation': 5} |
| degradation|replace_short|L21_layer_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.372 | 0.631 | 1.340 | 0.851 | 0.775 | {'continuation': 5} |
| degradation|replace_short|L20_layer_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.380 | 0.655 | 1.268 | 0.805 | 0.631 | {'continuation': 5} |
| degradation|replace_short|L29_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.122 | 0.242 | 0.270 | 0.157 | 0.000 | {'continuation': 1, 'prose': 4} |
| degradation|replace_short|L28_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.100 | 0.179 | 0.138 | 0.075 | 0.013 | {'continuation': 1, 'prose': 4} |
| degradation|replace_short|L29_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.248 | 0.433 | 0.120 | 0.079 | -0.050 | {'prose': 5} |
| degradation|replace_short|L28_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | -0.000 | 0.003 | 0.080 | 0.047 | -0.037 | {'prose': 5} |
| degradation|replace_short|L30_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.000 | 0.048 | -0.015 | -0.025 | {'continuation': 2, 'prose': 3} |
| degradation|replace_short|L30_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.000 | -0.015 | -0.002 | 0.013 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | early_effect | early_frac | final_effect | final_frac | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|replace_short|L23_layer_out | 1.000 | 0.000 | 2.67 | 1.67 | 0.357 | 0.851 | 8.964 | 1.247 | 4.083 | {'prose': 3} |
| degradation|replace_short|L24_layer_out | 1.000 | 0.000 | 2.67 | 1.67 | 0.416 | 1.000 | 8.296 | 1.153 | 4.083 | {'prose': 3} |
| degradation|replace_short|L25_layer_out | 1.000 | 0.000 | 2.33 | 1.33 | 0.000 | 0.000 | 7.764 | 1.073 | 3.708 | {'prose': 3} |
| degradation|replace_short|L22_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | 0.010 | -0.075 | 5.709 | 0.795 | 2.417 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L20_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | -0.175 | -0.487 | 4.880 | 0.681 | 1.958 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L18_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | -0.168 | -0.476 | 4.155 | 0.571 | 1.958 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L17_layer_out | 0.333 | 0.667 | 1.33 | 0.33 | -0.309 | -0.818 | 3.238 | 0.444 | 1.417 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L19_layer_out | 0.000 | 1.000 | 1.00 | 0.00 | -0.188 | -0.531 | 4.291 | 0.595 | 2.083 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L21_layer_out | 0.000 | 1.000 | 1.00 | 0.00 | -0.112 | -0.366 | 3.970 | 0.550 | 1.542 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L16_layer_out | 0.000 | 1.000 | 1.00 | 0.00 | -0.307 | -0.814 | 2.826 | 0.389 | 1.500 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L24_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.194 | 0.502 | 2.167 | 0.294 | 1.292 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L24_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.326 | 0.847 | 1.903 | 0.269 | 1.167 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L25_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.000 | 1.573 | 0.208 | 1.083 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L23_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.373 | 0.971 | 1.481 | 0.194 | 1.250 | {'prose': 3} |
| degradation|replace_short|L23_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | -0.124 | -0.437 | 0.901 | 0.112 | 0.833 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L15_layer_out | 0.000 | 1.000 | 1.00 | 0.00 | -0.233 | -0.670 | -0.045 | -0.011 | 0.333 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L25_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.000 | 0.000 | -0.139 | -0.012 | 0.000 | {'continuation': 2, 'prose': 1} |

