# Phase 689 Pre-L26 Source Path Localization

- generated: `2026-06-26 14:06:23`

| model | pairs | target | source_layers | best_restore | repair | target_gain | rank_effect | best_degrade | drop | target_loss | rank_effect |
|---|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|
| deepseek7b | 72 | L26_layer_input | [18, 19, 20, 21, 22, 23, 24, 25] | restore|add_delta|L24_layer_out | 1.000 | 8.105 | 166.69 | degradation|replace_short|L23_layer_out | 1.000 | 8.202 | 144.42 |
| glm4 | 5 | L38_layer_input | [30, 31, 32, 33, 34, 35, 36, 37] | restore|add_delta|L36_layer_out | 1.000 | 1.636 | 1.00 | degradation|replace_short|L37_layer_out | 1.000 | 1.580 | 1.00 |
| qwen3 | 3 | L33_layer_input | [25, 26, 27, 28, 29, 30, 31, 32] | restore|add_delta|L27_layer_out | 1.000 | 8.208 | 1.00 | degradation|replace_short|L26_layer_out | 1.000 | 7.826 | 1.33 |

## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|add_delta|L24_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | 8.105 | 0.936 | 6.335 | {'prose': 72} |
| restore|add_delta|L25_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | 7.285 | 1.000 | 6.309 | {'prose': 72} |
| restore|add_delta|L23_layer_out | 0.986 | 0.986 | 1.01 | 166.68 | 8.195 | 0.972 | 6.365 | {'prose': 72} |
| restore|add_delta|L22_layer_out | 0.986 | 0.986 | 1.01 | 166.68 | 7.810 | 0.911 | 6.418 | {'prose': 72} |
| restore|add_delta|L21_layer_out | 0.986 | 0.986 | 1.01 | 166.68 | 7.307 | 0.861 | 6.357 | {'prose': 72} |
| restore|add_delta|L20_layer_out | 0.972 | 0.972 | 1.06 | 166.64 | 7.385 | 1.011 | 6.303 | {'prose': 72} |
| restore|add_delta|L19_layer_out | 0.889 | 0.889 | 1.31 | 166.39 | 6.892 | 0.927 | 6.002 | {'prose': 72} |
| restore|add_delta|L18_layer_out | 0.792 | 0.792 | 1.60 | 166.10 | 6.682 | 0.864 | 5.627 | {'prose': 72} |
| restore|add_delta|L22_attn_out | 0.319 | 0.319 | 9.54 | 158.15 | 1.917 | 0.610 | 3.701 | {'continuation': 1, 'prose': 71} |
| restore|add_delta|L18_attn_out | 0.181 | 0.181 | 78.96 | 88.74 | 1.716 | 0.180 | 1.629 | {'continuation': 2, 'prose': 70} |
| restore|add_delta|L19_attn_out | 0.167 | 0.167 | 34.24 | 133.46 | 1.731 | 0.155 | 2.291 | {'continuation': 1, 'prose': 71} |
| restore|add_delta|L19_mlp_out | 0.125 | 0.125 | 48.19 | 119.50 | 5.984 | 0.834 | 2.243 | {'continuation': 15, 'prose': 57} |
| restore|add_delta|L20_attn_out | 0.111 | 0.111 | 58.81 | 108.89 | 2.975 | 0.061 | 1.976 | {'continuation': 8, 'prose': 64} |
| restore|add_delta|L21_mlp_out | 0.111 | 0.111 | 82.83 | 84.86 | 1.678 | -0.061 | 1.486 | {'continuation': 2, 'prose': 70} |
| restore|add_delta|L22_mlp_out | 0.097 | 0.097 | 25.60 | 142.10 | 3.304 | 0.162 | 2.282 | {'continuation': 2, 'prose': 70} |
| restore|add_delta|L18_mlp_out | 0.083 | 0.083 | 47.46 | 120.24 | 4.000 | 0.103 | 1.702 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|add_delta|L36_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.636 | 1.033 | 1.525 | {'prose': 5} |
| restore|add_delta|L37_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.580 | 1.000 | 1.562 | {'prose': 5} |
| restore|add_delta|L30_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.546 | 0.972 | 1.550 | {'prose': 5} |
| restore|add_delta|L33_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.524 | 0.957 | 1.525 | {'prose': 5} |
| restore|add_delta|L32_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.520 | 0.951 | 1.562 | {'prose': 5} |
| restore|add_delta|L31_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.501 | 0.942 | 1.525 | {'prose': 5} |
| restore|add_delta|L35_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.487 | 0.932 | 1.538 | {'prose': 5} |
| restore|add_delta|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.466 | 0.923 | 1.562 | {'continuation': 1, 'prose': 4} |
| restore|add_delta|L31_mlp_out | 0.800 | 0.800 | 1.20 | 0.80 | -0.088 | -0.041 | 0.100 | {'continuation': 5} |
| restore|add_delta|L30_mlp_out | 0.600 | 0.600 | 1.40 | 0.60 | 0.040 | -0.013 | 0.050 | {'continuation': 5} |
| restore|add_delta|L36_mlp_out | 0.600 | 0.600 | 1.40 | 0.60 | -0.272 | -0.126 | -0.013 | {'continuation': 5} |
| restore|add_delta|L32_mlp_out | 0.400 | 0.400 | 1.60 | 0.40 | 0.221 | 0.144 | 0.100 | {'continuation': 5} |
| restore|add_delta|L33_mlp_out | 0.400 | 0.400 | 1.60 | 0.40 | 0.102 | 0.058 | 0.138 | {'continuation': 5} |
| restore|add_delta|L32_attn_out | 0.400 | 0.400 | 1.60 | 0.40 | 0.077 | 0.071 | 0.237 | {'continuation': 5} |
| restore|add_delta|L34_mlp_out | 0.400 | 0.400 | 1.60 | 0.40 | 0.021 | 0.014 | -0.075 | {'continuation': 5} |
| restore|add_delta|L37_attn_out | 0.400 | 0.400 | 1.60 | 0.40 | -0.046 | -0.032 | -0.025 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|add_delta|L27_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 8.208 | 1.131 | 3.292 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L26_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 8.104 | 1.116 | 3.208 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L25_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 8.097 | 1.120 | 3.333 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L29_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 8.035 | 1.109 | 3.208 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L28_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 7.765 | 1.072 | 3.333 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L30_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 7.457 | 1.028 | 3.167 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L31_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 7.275 | 1.003 | 2.958 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L32_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 7.256 | 1.000 | 2.875 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L26_attn_out | 0.667 | 0.667 | 1.33 | 0.67 | 3.405 | 0.478 | 0.667 | {'continuation': 1, 'prose': 2} |
| restore|add_delta|L25_mlp_out | 0.667 | 0.667 | 1.33 | 0.67 | 2.039 | 0.275 | 0.833 | {'continuation': 1, 'prose': 2} |
| restore|add_delta|L27_mlp_out | 0.667 | 0.667 | 1.33 | 0.67 | 1.984 | 0.276 | 0.458 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L29_attn_out | 0.667 | 0.667 | 1.33 | 0.67 | 1.456 | 0.194 | 0.375 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L31_mlp_out | 0.667 | 0.667 | 1.33 | 0.67 | 1.433 | 0.196 | 0.292 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L30_attn_out | 0.667 | 0.667 | 1.33 | 0.67 | 1.378 | 0.189 | 0.417 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L26_mlp_out | 0.667 | 0.667 | 1.33 | 0.67 | 1.248 | 0.162 | 0.500 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L28_mlp_out | 0.333 | 0.333 | 1.67 | 0.33 | 2.081 | 0.275 | 0.458 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|replace_short|L23_layer_out | 1.000 | 0.000 | 145.42 | 144.42 | 8.202 | 0.968 | 6.244 | {'prose': 72} |
| degradation|replace_short|L20_layer_out | 0.986 | 0.014 | 100.03 | 99.03 | 8.388 | 1.083 | 5.782 | {'prose': 72} |
| degradation|replace_short|L24_layer_out | 0.986 | 0.014 | 146.56 | 145.56 | 8.251 | 0.972 | 6.199 | {'prose': 72} |
| degradation|replace_short|L22_layer_out | 0.972 | 0.028 | 114.01 | 113.01 | 7.139 | 0.877 | 5.998 | {'prose': 72} |
| degradation|replace_short|L21_layer_out | 0.972 | 0.028 | 106.07 | 105.07 | 6.872 | 0.870 | 5.789 | {'prose': 72} |
| degradation|replace_short|L25_layer_out | 0.958 | 0.042 | 168.11 | 167.11 | 7.285 | 1.000 | 6.233 | {'prose': 72} |
| degradation|replace_short|L19_layer_out | 0.931 | 0.069 | 54.96 | 53.96 | 8.127 | 1.178 | 4.878 | {'prose': 72} |
| degradation|replace_short|L18_layer_out | 0.833 | 0.167 | 34.14 | 33.14 | 6.536 | 0.871 | 4.037 | {'prose': 72} |
| degradation|replace_short|L22_attn_out | 0.472 | 0.528 | 4.35 | 3.35 | 1.253 | 0.617 | 1.772 | {'prose': 72} |
| degradation|replace_short|L19_mlp_out | 0.403 | 0.597 | 2.44 | 1.44 | 7.049 | 0.664 | 1.505 | {'prose': 72} |
| degradation|replace_short|L18_mlp_out | 0.347 | 0.653 | 2.19 | 1.19 | 4.295 | 0.139 | 0.934 | {'prose': 72} |
| degradation|replace_short|L19_attn_out | 0.333 | 0.667 | 1.99 | 0.99 | -0.111 | 0.312 | 1.086 | {'prose': 72} |
| degradation|replace_short|L20_attn_out | 0.264 | 0.736 | 1.44 | 0.44 | 2.499 | 0.313 | 0.944 | {'prose': 72} |
| degradation|replace_short|L22_mlp_out | 0.208 | 0.792 | 1.54 | 0.54 | 4.029 | 0.120 | 0.805 | {'prose': 72} |
| degradation|replace_short|L23_attn_out | 0.139 | 0.861 | 1.21 | 0.21 | 4.119 | 0.695 | 0.339 | {'prose': 72} |
| degradation|replace_short|L24_mlp_out | 0.125 | 0.875 | 1.18 | 0.18 | 1.601 | 0.127 | 0.378 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|replace_short|L37_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | 1.580 | 1.000 | 1.550 | {'continuation': 5} |
| degradation|replace_short|L33_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 1.571 | 0.996 | 1.475 | {'continuation': 5} |
| degradation|replace_short|L34_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 1.530 | 0.969 | 1.525 | {'continuation': 5} |
| degradation|replace_short|L35_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 1.519 | 0.957 | 1.512 | {'continuation': 5} |
| degradation|replace_short|L36_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 1.628 | 1.031 | 1.525 | {'continuation': 5} |
| degradation|replace_short|L30_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 1.594 | 1.019 | 1.450 | {'continuation': 5} |
| degradation|replace_short|L32_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 1.554 | 0.981 | 1.500 | {'continuation': 5} |
| degradation|replace_short|L31_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 1.548 | 0.983 | 1.475 | {'continuation': 5} |
| degradation|replace_short|L32_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.333 | 0.206 | -0.013 | {'continuation': 1, 'prose': 4} |
| degradation|replace_short|L33_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.190 | 0.119 | 0.075 | {'prose': 5} |
| degradation|replace_short|L32_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.113 | 0.104 | 0.138 | {'prose': 5} |
| degradation|replace_short|L33_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.112 | 0.014 | -0.113 | {'prose': 5} |
| degradation|replace_short|L36_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.090 | 0.056 | -0.025 | {'continuation': 1, 'prose': 4} |
| degradation|replace_short|L34_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.059 | 0.034 | -0.125 | {'continuation': 1, 'prose': 4} |
| degradation|replace_short|L30_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.048 | -0.015 | -0.025 | {'continuation': 2, 'prose': 3} |
| degradation|replace_short|L35_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.041 | 0.017 | -0.138 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|replace_short|L26_layer_out | 1.000 | 0.000 | 2.33 | 1.33 | 7.826 | 1.078 | 3.750 | {'prose': 3} |
| degradation|replace_short|L25_layer_out | 1.000 | 0.000 | 2.33 | 1.33 | 7.764 | 1.073 | 3.708 | {'prose': 3} |
| degradation|replace_short|L27_layer_out | 1.000 | 0.000 | 2.33 | 1.33 | 7.667 | 1.057 | 3.667 | {'prose': 3} |
| degradation|replace_short|L28_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | 7.301 | 1.007 | 3.583 | {'prose': 3} |
| degradation|replace_short|L32_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | 7.256 | 1.000 | 3.167 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L29_layer_out | 0.333 | 0.667 | 1.33 | 0.33 | 7.501 | 1.034 | 3.542 | {'prose': 3} |
| degradation|replace_short|L30_layer_out | 0.333 | 0.667 | 1.33 | 0.33 | 7.099 | 0.978 | 3.417 | {'prose': 3} |
| degradation|replace_short|L31_layer_out | 0.333 | 0.667 | 1.33 | 0.33 | 7.006 | 0.967 | 3.292 | {'prose': 3} |
| degradation|replace_short|L26_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 2.643 | 0.371 | 0.583 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L25_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 1.573 | 0.208 | 1.083 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L28_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 1.421 | 0.192 | 0.417 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L31_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 1.159 | 0.159 | 0.542 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L29_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 1.067 | 0.145 | 0.458 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L27_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 1.050 | 0.152 | 0.583 | {'continuation': 1, 'prose': 2} |
| degradation|replace_short|L30_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.959 | 0.132 | 0.500 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L32_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.879 | 0.120 | -0.458 | {'continuation': 2, 'prose': 1} |


## Largest Target Effects

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|replace_short|L20_layer_out | 0.986 | 0.014 | 100.03 | 99.03 | 8.388 | 1.083 | 5.782 | {'prose': 72} |
| degradation|replace_short|L24_layer_out | 0.986 | 0.014 | 146.56 | 145.56 | 8.251 | 0.972 | 6.199 | {'prose': 72} |
| degradation|replace_short|L23_layer_out | 1.000 | 0.000 | 145.42 | 144.42 | 8.202 | 0.968 | 6.244 | {'prose': 72} |
| restore|add_delta|L23_layer_out | 0.986 | 0.986 | 1.01 | 166.68 | 8.195 | 0.972 | 6.365 | {'prose': 72} |
| degradation|replace_short|L19_layer_out | 0.931 | 0.069 | 54.96 | 53.96 | 8.127 | 1.178 | 4.878 | {'prose': 72} |
| restore|add_delta|L24_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | 8.105 | 0.936 | 6.335 | {'prose': 72} |
| restore|add_delta|L22_layer_out | 0.986 | 0.986 | 1.01 | 166.68 | 7.810 | 0.911 | 6.418 | {'prose': 72} |
| restore|add_delta|L20_layer_out | 0.972 | 0.972 | 1.06 | 166.64 | 7.385 | 1.011 | 6.303 | {'prose': 72} |
| restore|add_delta|L21_layer_out | 0.986 | 0.986 | 1.01 | 166.68 | 7.307 | 0.861 | 6.357 | {'prose': 72} |
| restore|add_delta|L25_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | 7.285 | 1.000 | 6.309 | {'prose': 72} |
| degradation|replace_short|L25_layer_out | 0.958 | 0.042 | 168.11 | 167.11 | 7.285 | 1.000 | 6.233 | {'prose': 72} |
| degradation|replace_short|L22_layer_out | 0.972 | 0.028 | 114.01 | 113.01 | 7.139 | 0.877 | 5.998 | {'prose': 72} |
| degradation|replace_short|L19_mlp_out | 0.403 | 0.597 | 2.44 | 1.44 | 7.049 | 0.664 | 1.505 | {'prose': 72} |
| restore|add_delta|L19_layer_out | 0.889 | 0.889 | 1.31 | 166.39 | 6.892 | 0.927 | 6.002 | {'prose': 72} |
| degradation|replace_short|L21_layer_out | 0.972 | 0.028 | 106.07 | 105.07 | 6.872 | 0.870 | 5.789 | {'prose': 72} |
| restore|add_delta|L18_layer_out | 0.792 | 0.792 | 1.60 | 166.10 | 6.682 | 0.864 | 5.627 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|add_delta|L36_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.636 | 1.033 | 1.525 | {'prose': 5} |
| degradation|replace_short|L36_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 1.628 | 1.031 | 1.525 | {'continuation': 5} |
| degradation|replace_short|L30_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 1.594 | 1.019 | 1.450 | {'continuation': 5} |
| degradation|replace_short|L37_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | 1.580 | 1.000 | 1.550 | {'continuation': 5} |
| restore|add_delta|L37_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.580 | 1.000 | 1.562 | {'prose': 5} |
| degradation|replace_short|L33_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 1.571 | 0.996 | 1.475 | {'continuation': 5} |
| degradation|replace_short|L32_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 1.554 | 0.981 | 1.500 | {'continuation': 5} |
| degradation|replace_short|L31_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 1.548 | 0.983 | 1.475 | {'continuation': 5} |
| restore|add_delta|L30_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.546 | 0.972 | 1.550 | {'prose': 5} |
| degradation|replace_short|L34_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 1.530 | 0.969 | 1.525 | {'continuation': 5} |
| restore|add_delta|L33_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.524 | 0.957 | 1.525 | {'prose': 5} |
| restore|add_delta|L32_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.520 | 0.951 | 1.562 | {'prose': 5} |
| degradation|replace_short|L35_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 1.519 | 0.957 | 1.512 | {'continuation': 5} |
| restore|add_delta|L31_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.501 | 0.942 | 1.525 | {'prose': 5} |
| restore|add_delta|L35_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.487 | 0.932 | 1.538 | {'prose': 5} |
| restore|add_delta|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.466 | 0.923 | 1.562 | {'continuation': 1, 'prose': 4} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|add_delta|L27_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 8.208 | 1.131 | 3.292 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L26_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 8.104 | 1.116 | 3.208 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L25_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 8.097 | 1.120 | 3.333 | {'continuation': 2, 'prose': 1} |
| restore|add_delta|L29_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 8.035 | 1.109 | 3.208 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L26_layer_out | 1.000 | 0.000 | 2.33 | 1.33 | 7.826 | 1.078 | 3.750 | {'prose': 3} |
| restore|add_delta|L28_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 7.765 | 1.072 | 3.333 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L25_layer_out | 1.000 | 0.000 | 2.33 | 1.33 | 7.764 | 1.073 | 3.708 | {'prose': 3} |
| degradation|replace_short|L27_layer_out | 1.000 | 0.000 | 2.33 | 1.33 | 7.667 | 1.057 | 3.667 | {'prose': 3} |
| degradation|replace_short|L29_layer_out | 0.333 | 0.667 | 1.33 | 0.33 | 7.501 | 1.034 | 3.542 | {'prose': 3} |
| restore|add_delta|L30_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 7.457 | 1.028 | 3.167 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L28_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | 7.301 | 1.007 | 3.583 | {'prose': 3} |
| restore|add_delta|L31_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 7.275 | 1.003 | 2.958 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L32_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | 7.256 | 1.000 | 3.167 | {'continuation': 1, 'prose': 2} |
| restore|add_delta|L32_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 7.256 | 1.000 | 2.875 | {'continuation': 2, 'prose': 1} |
| degradation|replace_short|L30_layer_out | 0.333 | 0.667 | 1.33 | 0.33 | 7.099 | 0.978 | 3.417 | {'prose': 3} |
| degradation|replace_short|L31_layer_out | 0.333 | 0.667 | 1.33 | 0.33 | 7.006 | 0.967 | 3.292 | {'prose': 3} |

