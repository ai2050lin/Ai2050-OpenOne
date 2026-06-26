# Phase 691 Boundary Component and Residual-Carry Decomposition

- generated: `2026-06-26 14:37:33`

| model | pairs | target | scan_layers | best_restore | repair | target_gain | rank_effect | best_degrade | drop | target_loss | rank_effect |
|---|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|
| deepseek7b | 72 | L26_layer_input | [13, 14, 15, 16, 17, 18] | restore|full_layer_delta|L18 | 0.792 | 6.682 | 166.10 | degradation|remove_carry_est_layerout|L15 | 0.889 | 3.645 | 19.18 |
| glm4 | 5 | L38_layer_input | [23, 24, 25, 26, 27, 28, 29, 30] | restore|layer_minus_attn_delta|L30 | 1.000 | 1.565 | 1.00 | degradation|full_layer_replace_short|L27 | 0.600 | 1.554 | 0.60 |
| qwen3 | 3 | L33_layer_input | [18, 19, 20, 21, 22, 23, 24, 25] | restore|full_layer_delta|L23 | 1.000 | 9.068 | 1.00 | degradation|remove_carry_est_layerout|L23 | 1.000 | 9.464 | 2.67 |

## Mode Averages

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|attn_mlp_replace_short | 0.315 | 0.685 | 3.93 | 2.93 | 2.330 | 0.117 | 0.935 | {'continuation': 2, 'prose': 430} |
| degradation|attn_replace_short | 0.120 | 0.880 | 1.37 | 0.37 | 0.772 | 0.076 | 0.214 | {'prose': 432} |
| degradation|full_layer_replace_short | 0.796 | 0.204 | 21.89 | 20.89 | 6.314 | 0.791 | 3.655 | {'prose': 432} |
| degradation|mlp_replace_short | 0.199 | 0.801 | 1.56 | 0.56 | 1.293 | 0.024 | 0.291 | {'continuation': 2, 'prose': 430} |
| degradation|random_layer_same_norm | 0.347 | 0.653 | 13.26 | 12.26 | 2.856 | 0.356 | 1.162 | {'continuation': 1, 'prose': 431} |
| degradation|remove_attn_est_layerout | 0.206 | 0.794 | 1.61 | 0.61 | 0.665 | 0.067 | 0.540 | {'prose': 432} |
| degradation|remove_carry_est_layerout | 0.683 | 0.317 | 24.06 | 23.06 | 4.358 | 0.618 | 3.135 | {'continuation': 4, 'prose': 428} |
| degradation|remove_mlp_est_layerout | 0.190 | 0.810 | 1.53 | 0.53 | 1.280 | 0.015 | 0.294 | {'continuation': 3, 'prose': 429} |
| restore|attn_delta | 0.081 | 0.081 | 158.78 | 8.91 | 1.112 | 0.140 | 0.516 | {'continuation': 13, 'prose': 419} |
| restore|attn_mlp_delta | 0.111 | 0.111 | 199.20 | -31.51 | 1.038 | 0.122 | 0.430 | {'continuation': 21, 'prose': 411} |
| restore|carry_est_layerout | 0.417 | 0.417 | 31.35 | 136.34 | 4.893 | 0.666 | 3.738 | {'continuation': 34, 'prose': 398} |
| restore|full_layer_delta | 0.544 | 0.544 | 6.85 | 160.84 | 5.896 | 0.662 | 4.620 | {'continuation': 4, 'prose': 428} |
| restore|layer_minus_attn_delta | 0.421 | 0.421 | 25.75 | 141.94 | 5.147 | 0.542 | 3.810 | {'continuation': 5, 'prose': 427} |
| restore|layer_minus_mlp_delta | 0.521 | 0.521 | 11.87 | 155.83 | 5.734 | 0.719 | 4.513 | {'continuation': 20, 'prose': 412} |
| restore|mlp_delta | 0.037 | 0.037 | 211.79 | -44.10 | 0.123 | 0.036 | -0.307 | {'continuation': 17, 'prose': 415} |
| restore|random_layer_same_norm | 0.037 | 0.037 | 388.74 | -221.04 | -1.946 | -0.001 | -1.068 | {'continuation': 57, 'prose': 375} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|attn_mlp_replace_short | 0.025 | 0.975 | 1.02 | 0.03 | 0.345 | 0.214 | 0.182 | {'continuation': 14, 'prose': 26} |
| degradation|attn_replace_short | 0.000 | 1.000 | 1.00 | 0.00 | 0.061 | 0.048 | -0.003 | {'continuation': 2, 'prose': 38} |
| degradation|full_layer_replace_short | 0.375 | 0.625 | 1.38 | 0.38 | 1.411 | 0.897 | 1.384 | {'continuation': 40} |
| degradation|mlp_replace_short | 0.000 | 1.000 | 1.00 | 0.00 | 0.260 | 0.158 | 0.156 | {'continuation': 10, 'prose': 30} |
| degradation|random_layer_same_norm | 0.025 | 0.975 | 1.02 | 0.03 | 0.048 | 0.038 | 0.048 | {'continuation': 5, 'prose': 35} |
| degradation|remove_attn_est_layerout | 0.000 | 1.000 | 1.00 | 0.00 | 0.071 | 0.055 | -0.009 | {'continuation': 2, 'prose': 38} |
| degradation|remove_carry_est_layerout | 0.100 | 0.900 | 1.12 | 0.12 | 1.032 | 0.646 | 0.970 | {'continuation': 40} |
| degradation|remove_mlp_est_layerout | 0.000 | 1.000 | 1.00 | 0.00 | 0.259 | 0.154 | 0.157 | {'continuation': 10, 'prose': 30} |
| restore|attn_delta | 0.425 | 0.425 | 1.57 | 0.42 | 0.058 | 0.043 | 0.075 | {'continuation': 40} |
| restore|attn_mlp_delta | 0.850 | 0.850 | 1.15 | 0.85 | 0.394 | 0.258 | 0.474 | {'continuation': 40} |
| restore|carry_est_layerout | 0.975 | 0.975 | 1.02 | 0.97 | 1.092 | 0.698 | 1.348 | {'continuation': 16, 'prose': 24} |
| restore|full_layer_delta | 1.000 | 1.000 | 1.00 | 1.00 | 1.466 | 0.928 | 1.564 | {'continuation': 6, 'prose': 34} |
| restore|layer_minus_attn_delta | 1.000 | 1.000 | 1.00 | 1.00 | 1.402 | 0.886 | 1.547 | {'continuation': 7, 'prose': 33} |
| restore|layer_minus_mlp_delta | 0.975 | 0.975 | 1.02 | 0.97 | 1.180 | 0.757 | 1.355 | {'continuation': 11, 'prose': 29} |
| restore|mlp_delta | 0.750 | 0.750 | 1.25 | 0.75 | 0.312 | 0.200 | 0.400 | {'continuation': 40} |
| restore|random_layer_same_norm | 0.225 | 0.225 | 1.85 | 0.15 | 0.149 | 0.056 | -0.095 | {'continuation': 40} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|attn_mlp_replace_short | 0.083 | 0.917 | 1.08 | 0.08 | 0.910 | 0.119 | 0.786 | {'continuation': 11, 'prose': 13} |
| degradation|attn_replace_short | 0.000 | 1.000 | 1.00 | 0.00 | 0.414 | 0.057 | 0.427 | {'continuation': 11, 'prose': 13} |
| degradation|full_layer_replace_short | 0.625 | 0.375 | 1.83 | 0.83 | 6.004 | 0.833 | 2.729 | {'continuation': 6, 'prose': 18} |
| degradation|mlp_replace_short | 0.042 | 0.958 | 1.04 | 0.04 | 0.408 | 0.045 | 0.578 | {'continuation': 12, 'prose': 12} |
| degradation|random_layer_same_norm | 0.083 | 0.917 | 1.08 | 0.08 | 0.212 | 0.026 | 0.151 | {'continuation': 12, 'prose': 12} |
| degradation|remove_attn_est_layerout | 0.000 | 1.000 | 1.00 | 0.00 | 0.410 | 0.054 | 0.401 | {'continuation': 13, 'prose': 11} |
| degradation|remove_carry_est_layerout | 0.583 | 0.417 | 1.88 | 0.88 | 5.724 | 0.798 | 2.495 | {'continuation': 10, 'prose': 14} |
| degradation|remove_mlp_est_layerout | 0.000 | 1.000 | 1.00 | 0.00 | 0.304 | 0.035 | 0.536 | {'continuation': 13, 'prose': 11} |
| restore|attn_delta | 0.292 | 0.292 | 1.83 | 0.17 | 1.276 | 0.171 | 0.396 | {'continuation': 15, 'prose': 9} |
| restore|attn_mlp_delta | 0.208 | 0.208 | 2.38 | -0.38 | 0.802 | 0.102 | 0.245 | {'continuation': 14, 'prose': 10} |
| restore|carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 6.521 | 0.905 | 2.125 | {'continuation': 13, 'prose': 11} |
| restore|full_layer_delta | 1.000 | 1.000 | 1.00 | 1.00 | 7.536 | 1.041 | 2.682 | {'continuation': 16, 'prose': 8} |
| restore|layer_minus_attn_delta | 0.875 | 0.875 | 1.12 | 0.88 | 6.619 | 0.913 | 2.349 | {'continuation': 15, 'prose': 9} |
| restore|layer_minus_mlp_delta | 1.000 | 1.000 | 1.00 | 1.00 | 7.171 | 0.998 | 2.385 | {'continuation': 14, 'prose': 10} |
| restore|mlp_delta | 0.208 | 0.208 | 2.33 | -0.33 | -0.054 | -0.019 | -0.125 | {'continuation': 13, 'prose': 11} |
| restore|random_layer_same_norm | 0.292 | 0.292 | 2.54 | -0.54 | -1.053 | -0.157 | -0.490 | {'continuation': 10, 'prose': 14} |


## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|full_layer_delta|L18 | 0.792 | 0.792 | 1.60 | 166.10 | 6.682 | 0.864 | 5.627 | {'prose': 72} |
| restore|layer_minus_mlp_delta|L16 | 0.778 | 0.778 | 2.99 | 164.71 | 10.315 | 1.029 | 5.961 | {'continuation': 12, 'prose': 60} |
| restore|layer_minus_mlp_delta|L15 | 0.750 | 0.750 | 2.79 | 164.90 | 8.333 | 1.036 | 5.975 | {'continuation': 4, 'prose': 68} |
| restore|layer_minus_attn_delta|L17 | 0.722 | 0.722 | 2.88 | 164.82 | 5.443 | 0.571 | 5.188 | {'prose': 72} |
| restore|carry_est_layerout|L15 | 0.708 | 0.708 | 3.21 | 164.49 | 7.619 | 0.976 | 5.933 | {'continuation': 11, 'prose': 61} |
| restore|carry_est_layerout|L17 | 0.639 | 0.639 | 3.43 | 164.26 | 5.437 | 0.731 | 5.115 | {'prose': 72} |
| restore|layer_minus_mlp_delta|L17 | 0.611 | 0.611 | 3.57 | 164.12 | 5.686 | 0.749 | 4.835 | {'prose': 72} |
| restore|full_layer_delta|L15 | 0.597 | 0.597 | 5.90 | 161.79 | 6.263 | 0.621 | 4.923 | {'prose': 72} |
| restore|full_layer_delta|L17 | 0.569 | 0.569 | 3.69 | 164.00 | 5.118 | 0.678 | 4.651 | {'prose': 72} |
| restore|layer_minus_attn_delta|L15 | 0.556 | 0.556 | 5.19 | 162.50 | 5.214 | 0.434 | 5.000 | {'continuation': 1, 'prose': 71} |
| restore|layer_minus_attn_delta|L18 | 0.486 | 0.486 | 6.03 | 161.67 | 5.219 | 0.621 | 4.543 | {'prose': 72} |
| restore|full_layer_delta|L16 | 0.486 | 0.486 | 4.92 | 162.78 | 4.871 | 0.592 | 4.371 | {'prose': 72} |
| restore|carry_est_layerout|L16 | 0.472 | 0.472 | 10.60 | 157.10 | 10.800 | 1.198 | 4.932 | {'continuation': 19, 'prose': 53} |
| restore|full_layer_delta|L14 | 0.431 | 0.431 | 11.01 | 156.68 | 5.901 | 0.624 | 4.226 | {'continuation': 3, 'prose': 69} |
| restore|layer_minus_attn_delta|L14 | 0.417 | 0.417 | 17.14 | 150.56 | 6.333 | 0.501 | 3.816 | {'continuation': 1, 'prose': 71} |
| restore|layer_minus_mlp_delta|L18 | 0.403 | 0.403 | 9.17 | 158.53 | 2.117 | 0.668 | 3.790 | {'prose': 72} |
| restore|full_layer_delta|L13 | 0.389 | 0.389 | 13.99 | 153.71 | 6.540 | 0.590 | 3.923 | {'continuation': 1, 'prose': 71} |
| restore|layer_minus_mlp_delta|L14 | 0.361 | 0.361 | 13.96 | 153.74 | 4.249 | 0.705 | 3.931 | {'continuation': 3, 'prose': 69} |
| restore|attn_mlp_delta|L13 | 0.347 | 0.347 | 24.60 | 143.10 | 6.281 | 0.681 | 3.356 | {'continuation': 12, 'prose': 60} |
| restore|carry_est_layerout|L14 | 0.333 | 0.333 | 20.79 | 146.90 | 4.259 | 0.687 | 3.279 | {'continuation': 3, 'prose': 69} |
| restore|attn_delta|L13 | 0.250 | 0.250 | 27.94 | 139.75 | 5.649 | 0.507 | 2.837 | {'continuation': 8, 'prose': 64} |
| restore|attn_mlp_delta|L18 | 0.250 | 0.250 | 16.86 | 150.83 | 5.576 | 0.232 | 3.009 | {'prose': 72} |
| restore|layer_minus_attn_delta|L16 | 0.236 | 0.236 | 26.72 | 140.97 | 6.189 | 0.713 | 2.957 | {'continuation': 1, 'prose': 71} |
| restore|layer_minus_mlp_delta|L13 | 0.222 | 0.222 | 38.74 | 128.96 | 3.703 | 0.123 | 2.588 | {'continuation': 1, 'prose': 71} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|layer_minus_attn_delta|L30 | 1.000 | 1.000 | 1.00 | 1.00 | 1.565 | 0.981 | 1.500 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L30 | 1.000 | 1.000 | 1.00 | 1.00 | 1.546 | 0.972 | 1.550 | {'prose': 5} |
| restore|full_layer_delta|L27 | 1.000 | 1.000 | 1.00 | 1.00 | 1.538 | 0.975 | 1.650 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L29 | 1.000 | 1.000 | 1.00 | 1.00 | 1.530 | 0.962 | 1.550 | {'prose': 5} |
| restore|full_layer_delta|L28 | 1.000 | 1.000 | 1.00 | 1.00 | 1.523 | 0.955 | 1.550 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_mlp_delta|L30 | 1.000 | 1.000 | 1.00 | 1.00 | 1.505 | 0.993 | 1.550 | {'continuation': 2, 'prose': 3} |
| restore|carry_est_layerout|L30 | 1.000 | 1.000 | 1.00 | 1.00 | 1.494 | 0.988 | 1.500 | {'continuation': 2, 'prose': 3} |
| restore|layer_minus_attn_delta|L28 | 1.000 | 1.000 | 1.00 | 1.00 | 1.458 | 0.923 | 1.550 | {'prose': 5} |
| restore|layer_minus_attn_delta|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 1.456 | 0.893 | 1.462 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 1.451 | 0.927 | 1.538 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_attn_delta|L26 | 1.000 | 1.000 | 1.00 | 1.00 | 1.445 | 0.929 | 1.575 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 1.400 | 0.892 | 1.575 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L26 | 1.000 | 1.000 | 1.00 | 1.00 | 1.398 | 0.886 | 1.575 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_mlp_delta|L28 | 1.000 | 1.000 | 1.00 | 1.00 | 1.395 | 0.893 | 1.562 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_attn_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 1.378 | 0.888 | 1.625 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_attn_delta|L29 | 1.000 | 1.000 | 1.00 | 1.00 | 1.348 | 0.863 | 1.512 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_mlp_delta|L27 | 1.000 | 1.000 | 1.00 | 1.00 | 1.344 | 0.876 | 1.613 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 1.342 | 0.852 | 1.525 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_attn_delta|L27 | 1.000 | 1.000 | 1.00 | 1.00 | 1.332 | 0.844 | 1.650 | {'continuation': 1, 'prose': 4} |
| restore|carry_est_layerout|L28 | 1.000 | 1.000 | 1.00 | 1.00 | 1.270 | 0.817 | 1.550 | {'continuation': 1, 'prose': 4} |
| restore|carry_est_layerout|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 1.263 | 0.781 | 1.150 | {'continuation': 2, 'prose': 3} |
| restore|layer_minus_attn_delta|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 1.233 | 0.764 | 1.500 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_mlp_delta|L29 | 1.000 | 1.000 | 1.00 | 1.00 | 1.215 | 0.783 | 1.462 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_mlp_delta|L26 | 1.000 | 1.000 | 1.00 | 1.00 | 1.182 | 0.752 | 1.462 | {'continuation': 2, 'prose': 3} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|full_layer_delta|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 9.068 | 1.261 | 3.458 | {'continuation': 2, 'prose': 1} |
| restore|full_layer_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 8.631 | 1.200 | 3.583 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L21 | 1.000 | 1.000 | 1.00 | 1.00 | 8.498 | 1.176 | 3.125 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_attn_delta|L20 | 1.000 | 1.000 | 1.00 | 1.00 | 8.350 | 1.146 | 2.750 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_attn_delta|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 8.252 | 1.156 | 2.667 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 8.198 | 1.146 | 3.000 | {'continuation': 2, 'prose': 1} |
| restore|carry_est_layerout|L21 | 1.000 | 1.000 | 1.00 | 1.00 | 8.121 | 1.128 | 2.750 | {'continuation': 1, 'prose': 2} |
| restore|full_layer_delta|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 8.097 | 1.120 | 3.333 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_attn_delta|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 8.067 | 1.107 | 3.292 | {'continuation': 2, 'prose': 1} |
| restore|carry_est_layerout|L20 | 1.000 | 1.000 | 1.00 | 1.00 | 8.044 | 1.117 | 2.333 | {'continuation': 1, 'prose': 2} |
| restore|carry_est_layerout|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 7.271 | 1.019 | 2.333 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L18 | 1.000 | 1.000 | 1.00 | 1.00 | 7.263 | 1.019 | 2.333 | {'continuation': 2, 'prose': 1} |
| restore|full_layer_delta|L20 | 1.000 | 1.000 | 1.00 | 1.00 | 7.199 | 0.988 | 2.250 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_attn_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 7.068 | 0.985 | 3.000 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 7.036 | 0.973 | 2.667 | {'continuation': 2, 'prose': 1} |
| restore|full_layer_delta|L18 | 1.000 | 1.000 | 1.00 | 1.00 | 6.977 | 0.955 | 2.375 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L22 | 1.000 | 1.000 | 1.00 | 1.00 | 6.966 | 0.972 | 2.458 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L20 | 1.000 | 1.000 | 1.00 | 1.00 | 6.931 | 0.968 | 1.458 | {'continuation': 1, 'prose': 2} |
| restore|full_layer_delta|L19 | 1.000 | 1.000 | 1.00 | 1.00 | 6.888 | 0.947 | 2.167 | {'continuation': 2, 'prose': 1} |
| restore|carry_est_layerout|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 6.829 | 0.935 | 2.375 | {'continuation': 2, 'prose': 1} |
| restore|full_layer_delta|L21 | 1.000 | 1.000 | 1.00 | 1.00 | 6.764 | 0.933 | 2.125 | {'continuation': 2, 'prose': 1} |
| restore|full_layer_delta|L22 | 1.000 | 1.000 | 1.00 | 1.00 | 6.667 | 0.922 | 2.167 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L19 | 1.000 | 1.000 | 1.00 | 1.00 | 6.251 | 0.862 | 1.708 | {'continuation': 1, 'prose': 2} |
| restore|layer_minus_mlp_delta|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 6.227 | 0.866 | 2.333 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|remove_carry_est_layerout|L15 | 0.889 | 0.111 | 20.18 | 19.18 | 3.645 | 0.797 | 4.033 | {'prose': 72} |
| degradation|remove_carry_est_layerout|L17 | 0.875 | 0.125 | 86.24 | 85.24 | 6.875 | 1.037 | 4.905 | {'continuation': 4, 'prose': 68} |
| degradation|full_layer_replace_short|L13 | 0.875 | 0.125 | 25.18 | 24.18 | 6.198 | 0.574 | 3.958 | {'prose': 72} |
| degradation|remove_carry_est_layerout|L16 | 0.861 | 0.139 | 19.67 | 18.67 | 11.059 | 0.931 | 4.240 | {'prose': 72} |
| degradation|full_layer_replace_short|L14 | 0.847 | 0.153 | 16.57 | 15.57 | 5.350 | 0.612 | 3.738 | {'prose': 72} |
| degradation|full_layer_replace_short|L18 | 0.833 | 0.167 | 34.14 | 33.14 | 6.536 | 0.871 | 4.037 | {'prose': 72} |
| degradation|full_layer_replace_short|L15 | 0.792 | 0.208 | 22.94 | 21.94 | 7.644 | 0.868 | 3.862 | {'prose': 72} |
| degradation|full_layer_replace_short|L16 | 0.750 | 0.250 | 18.53 | 17.53 | 6.950 | 1.000 | 3.379 | {'prose': 72} |
| degradation|remove_carry_est_layerout|L14 | 0.722 | 0.278 | 12.99 | 11.99 | 3.743 | 0.415 | 3.356 | {'prose': 72} |
| degradation|full_layer_replace_short|L17 | 0.681 | 0.319 | 13.99 | 12.99 | 5.206 | 0.824 | 2.957 | {'prose': 72} |
| degradation|attn_mlp_replace_short|L18 | 0.639 | 0.361 | 13.81 | 12.81 | 6.507 | 0.301 | 2.647 | {'prose': 72} |
| degradation|remove_carry_est_layerout|L18 | 0.597 | 0.403 | 4.15 | 3.15 | 2.874 | 0.872 | 2.127 | {'prose': 72} |
| degradation|attn_mlp_replace_short|L13 | 0.569 | 0.431 | 4.26 | 3.26 | 5.591 | 0.524 | 2.210 | {'prose': 72} |
| degradation|attn_replace_short|L13 | 0.458 | 0.542 | 2.78 | 1.78 | 4.996 | 0.315 | 1.556 | {'prose': 72} |
| degradation|remove_attn_est_layerout|L13 | 0.458 | 0.542 | 2.99 | 1.99 | 3.256 | 0.129 | 1.907 | {'prose': 72} |
| degradation|random_layer_same_norm|L16 | 0.403 | 0.597 | 29.96 | 28.96 | 4.446 | 0.473 | 1.487 | {'prose': 72} |
| degradation|mlp_replace_short|L13 | 0.361 | 0.639 | 2.24 | 1.24 | 3.642 | 0.231 | 1.044 | {'prose': 72} |
| degradation|remove_mlp_est_layerout|L13 | 0.361 | 0.639 | 2.12 | 1.12 | 3.641 | 0.213 | 1.016 | {'prose': 72} |
| degradation|random_layer_same_norm|L17 | 0.361 | 0.639 | 6.14 | 5.14 | 2.789 | 0.464 | 1.378 | {'prose': 72} |
| degradation|random_layer_same_norm|L13 | 0.361 | 0.639 | 3.36 | 2.36 | 1.107 | -0.553 | 0.793 | {'prose': 72} |
| degradation|remove_mlp_est_layerout|L18 | 0.347 | 0.653 | 2.14 | 1.14 | 4.326 | 0.153 | 0.950 | {'prose': 72} |
| degradation|mlp_replace_short|L18 | 0.347 | 0.653 | 2.19 | 1.19 | 4.295 | 0.139 | 0.934 | {'prose': 72} |
| degradation|random_layer_same_norm|L18 | 0.333 | 0.667 | 11.81 | 10.81 | 2.792 | 0.664 | 1.237 | {'prose': 72} |
| degradation|remove_attn_est_layerout|L18 | 0.333 | 0.667 | 1.79 | 0.79 | 0.926 | 0.061 | 0.684 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|full_layer_replace_short|L27 | 0.600 | 0.400 | 1.60 | 0.60 | 1.554 | 0.998 | 1.562 | {'continuation': 5} |
| degradation|full_layer_replace_short|L28 | 0.600 | 0.400 | 1.60 | 0.60 | 1.545 | 0.982 | 1.488 | {'continuation': 5} |
| degradation|full_layer_replace_short|L29 | 0.600 | 0.400 | 1.60 | 0.60 | 1.543 | 0.979 | 1.488 | {'continuation': 5} |
| degradation|full_layer_replace_short|L30 | 0.400 | 0.600 | 1.40 | 0.40 | 1.594 | 1.019 | 1.450 | {'continuation': 5} |
| degradation|remove_carry_est_layerout|L30 | 0.200 | 0.800 | 1.20 | 0.20 | 1.506 | 0.994 | 1.425 | {'continuation': 5} |
| degradation|remove_carry_est_layerout|L28 | 0.200 | 0.800 | 1.20 | 0.20 | 1.324 | 0.847 | 1.312 | {'continuation': 5} |
| degradation|full_layer_replace_short|L25 | 0.200 | 0.800 | 1.20 | 0.20 | 1.283 | 0.813 | 1.312 | {'continuation': 5} |
| degradation|full_layer_replace_short|L23 | 0.200 | 0.800 | 1.20 | 0.20 | 1.283 | 0.820 | 1.212 | {'continuation': 5} |
| degradation|full_layer_replace_short|L26 | 0.200 | 0.800 | 1.20 | 0.20 | 1.277 | 0.809 | 1.325 | {'continuation': 5} |
| degradation|full_layer_replace_short|L24 | 0.200 | 0.800 | 1.20 | 0.20 | 1.206 | 0.752 | 1.238 | {'continuation': 5} |
| degradation|remove_carry_est_layerout|L27 | 0.200 | 0.800 | 1.40 | 0.40 | 1.181 | 0.758 | 1.300 | {'continuation': 5} |
| degradation|attn_mlp_replace_short|L23 | 0.200 | 0.800 | 1.20 | 0.20 | 1.022 | 0.712 | 0.800 | {'continuation': 3, 'prose': 2} |
| degradation|remove_carry_est_layerout|L29 | 0.200 | 0.800 | 1.20 | 0.20 | 1.011 | 0.643 | 1.212 | {'continuation': 5} |
| degradation|random_layer_same_norm|L24 | 0.200 | 0.800 | 1.20 | 0.20 | 0.064 | 0.106 | 0.400 | {'prose': 5} |
| degradation|remove_carry_est_layerout|L26 | 0.000 | 1.000 | 1.00 | 0.00 | 1.158 | 0.707 | 1.100 | {'continuation': 5} |
| degradation|remove_carry_est_layerout|L25 | 0.000 | 1.000 | 1.00 | 0.00 | 0.905 | 0.537 | 0.594 | {'continuation': 5} |
| degradation|remove_carry_est_layerout|L24 | 0.000 | 1.000 | 1.00 | 0.00 | 0.832 | 0.524 | 0.750 | {'continuation': 5} |
| degradation|remove_mlp_est_layerout|L23 | 0.000 | 1.000 | 1.00 | 0.00 | 0.803 | 0.553 | 0.725 | {'prose': 5} |
| degradation|mlp_replace_short|L23 | 0.000 | 1.000 | 1.00 | 0.00 | 0.768 | 0.533 | 0.700 | {'prose': 5} |
| degradation|attn_mlp_replace_short|L29 | 0.000 | 1.000 | 1.00 | 0.00 | 0.443 | 0.264 | 0.037 | {'continuation': 1, 'prose': 4} |
| degradation|attn_mlp_replace_short|L27 | 0.000 | 1.000 | 1.00 | 0.00 | 0.413 | 0.242 | 0.000 | {'continuation': 2, 'prose': 3} |
| degradation|remove_carry_est_layerout|L23 | 0.000 | 1.000 | 1.00 | 0.00 | 0.341 | 0.159 | 0.069 | {'continuation': 5} |
| degradation|attn_mlp_replace_short|L24 | 0.000 | 1.000 | 1.00 | 0.00 | 0.338 | 0.198 | 0.156 | {'continuation': 2, 'prose': 3} |
| degradation|random_layer_same_norm|L27 | 0.000 | 1.000 | 1.00 | 0.00 | 0.304 | 0.143 | 0.113 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|remove_carry_est_layerout|L23 | 1.000 | 0.000 | 3.67 | 2.67 | 9.464 | 1.329 | 3.875 | {'continuation': 1, 'prose': 2} |
| degradation|full_layer_replace_short|L23 | 1.000 | 0.000 | 2.67 | 1.67 | 8.964 | 1.247 | 4.083 | {'prose': 3} |
| degradation|full_layer_replace_short|L24 | 1.000 | 0.000 | 2.67 | 1.67 | 8.296 | 1.153 | 4.083 | {'prose': 3} |
| degradation|full_layer_replace_short|L25 | 1.000 | 0.000 | 2.33 | 1.33 | 7.764 | 1.073 | 3.708 | {'prose': 3} |
| degradation|remove_carry_est_layerout|L19 | 1.000 | 0.000 | 2.33 | 1.33 | 5.680 | 0.793 | 2.042 | {'continuation': 2, 'prose': 1} |
| degradation|remove_carry_est_layerout|L20 | 0.667 | 0.333 | 1.67 | 0.67 | 6.562 | 0.908 | 2.667 | {'continuation': 2, 'prose': 1} |
| degradation|remove_carry_est_layerout|L21 | 0.667 | 0.333 | 2.00 | 1.00 | 6.351 | 0.882 | 3.167 | {'continuation': 2, 'prose': 1} |
| degradation|full_layer_replace_short|L22 | 0.667 | 0.333 | 1.67 | 0.67 | 5.709 | 0.795 | 2.417 | {'continuation': 1, 'prose': 2} |
| degradation|remove_carry_est_layerout|L22 | 0.667 | 0.333 | 1.67 | 0.67 | 5.346 | 0.749 | 2.500 | {'continuation': 1, 'prose': 2} |
| degradation|full_layer_replace_short|L20 | 0.667 | 0.333 | 1.67 | 0.67 | 4.880 | 0.681 | 1.958 | {'continuation': 1, 'prose': 2} |
| degradation|full_layer_replace_short|L18 | 0.667 | 0.333 | 1.67 | 0.67 | 4.155 | 0.571 | 1.958 | {'continuation': 2, 'prose': 1} |
| degradation|remove_carry_est_layerout|L25 | 0.333 | 0.667 | 1.33 | 0.33 | 5.581 | 0.764 | 2.500 | {'continuation': 1, 'prose': 2} |
| degradation|remove_carry_est_layerout|L24 | 0.333 | 0.667 | 1.33 | 0.33 | 3.828 | 0.533 | 1.875 | {'prose': 3} |
| degradation|attn_mlp_replace_short|L24 | 0.333 | 0.667 | 1.33 | 0.33 | 3.672 | 0.513 | 2.000 | {'continuation': 1, 'prose': 2} |
| degradation|random_layer_same_norm|L18 | 0.333 | 0.667 | 1.33 | 0.33 | 3.536 | 0.468 | 1.833 | {'continuation': 2, 'prose': 1} |
| degradation|attn_mlp_replace_short|L18 | 0.333 | 0.667 | 1.33 | 0.33 | 1.614 | 0.198 | 0.833 | {'continuation': 2, 'prose': 1} |
| degradation|mlp_replace_short|L19 | 0.333 | 0.667 | 1.33 | 0.33 | 0.161 | 0.014 | 0.833 | {'continuation': 1, 'prose': 2} |
| degradation|random_layer_same_norm|L20 | 0.333 | 0.667 | 1.33 | 0.33 | -0.917 | -0.150 | 0.125 | {'continuation': 1, 'prose': 2} |
| degradation|full_layer_replace_short|L19 | 0.000 | 1.000 | 1.00 | 0.00 | 4.291 | 0.595 | 2.083 | {'continuation': 1, 'prose': 2} |
| degradation|full_layer_replace_short|L21 | 0.000 | 1.000 | 1.00 | 0.00 | 3.970 | 0.550 | 1.542 | {'continuation': 1, 'prose': 2} |
| degradation|remove_carry_est_layerout|L18 | 0.000 | 1.000 | 1.00 | 0.00 | 2.977 | 0.425 | 1.333 | {'continuation': 1, 'prose': 2} |
| degradation|attn_replace_short|L24 | 0.000 | 1.000 | 1.00 | 0.00 | 2.167 | 0.294 | 1.292 | {'continuation': 2, 'prose': 1} |
| degradation|remove_attn_est_layerout|L24 | 0.000 | 1.000 | 1.00 | 0.00 | 1.986 | 0.266 | 0.958 | {'continuation': 2, 'prose': 1} |
| degradation|mlp_replace_short|L24 | 0.000 | 1.000 | 1.00 | 0.00 | 1.903 | 0.269 | 1.167 | {'continuation': 1, 'prose': 2} |


## Largest Target Effects

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|remove_carry_est_layerout|L16 | 0.861 | 0.139 | 19.67 | 18.67 | 11.059 | 0.931 | 4.240 | {'prose': 72} |
| restore|carry_est_layerout|L16 | 0.472 | 0.472 | 10.60 | 157.10 | 10.800 | 1.198 | 4.932 | {'continuation': 19, 'prose': 53} |
| restore|layer_minus_mlp_delta|L16 | 0.778 | 0.778 | 2.99 | 164.71 | 10.315 | 1.029 | 5.961 | {'continuation': 12, 'prose': 60} |
| restore|layer_minus_mlp_delta|L15 | 0.750 | 0.750 | 2.79 | 164.90 | 8.333 | 1.036 | 5.975 | {'continuation': 4, 'prose': 68} |
| degradation|full_layer_replace_short|L15 | 0.792 | 0.208 | 22.94 | 21.94 | 7.644 | 0.868 | 3.862 | {'prose': 72} |
| restore|carry_est_layerout|L15 | 0.708 | 0.708 | 3.21 | 164.49 | 7.619 | 0.976 | 5.933 | {'continuation': 11, 'prose': 61} |
| degradation|full_layer_replace_short|L16 | 0.750 | 0.250 | 18.53 | 17.53 | 6.950 | 1.000 | 3.379 | {'prose': 72} |
| degradation|remove_carry_est_layerout|L17 | 0.875 | 0.125 | 86.24 | 85.24 | 6.875 | 1.037 | 4.905 | {'continuation': 4, 'prose': 68} |
| restore|full_layer_delta|L18 | 0.792 | 0.792 | 1.60 | 166.10 | 6.682 | 0.864 | 5.627 | {'prose': 72} |
| restore|full_layer_delta|L13 | 0.389 | 0.389 | 13.99 | 153.71 | 6.540 | 0.590 | 3.923 | {'continuation': 1, 'prose': 71} |
| degradation|full_layer_replace_short|L18 | 0.833 | 0.167 | 34.14 | 33.14 | 6.536 | 0.871 | 4.037 | {'prose': 72} |
| degradation|attn_mlp_replace_short|L18 | 0.639 | 0.361 | 13.81 | 12.81 | 6.507 | 0.301 | 2.647 | {'prose': 72} |
| restore|layer_minus_attn_delta|L14 | 0.417 | 0.417 | 17.14 | 150.56 | 6.333 | 0.501 | 3.816 | {'continuation': 1, 'prose': 71} |
| restore|attn_mlp_delta|L13 | 0.347 | 0.347 | 24.60 | 143.10 | 6.281 | 0.681 | 3.356 | {'continuation': 12, 'prose': 60} |
| restore|full_layer_delta|L15 | 0.597 | 0.597 | 5.90 | 161.79 | 6.263 | 0.621 | 4.923 | {'prose': 72} |
| degradation|full_layer_replace_short|L13 | 0.875 | 0.125 | 25.18 | 24.18 | 6.198 | 0.574 | 3.958 | {'prose': 72} |
| restore|layer_minus_attn_delta|L16 | 0.236 | 0.236 | 26.72 | 140.97 | 6.189 | 0.713 | 2.957 | {'continuation': 1, 'prose': 71} |
| restore|full_layer_delta|L14 | 0.431 | 0.431 | 11.01 | 156.68 | 5.901 | 0.624 | 4.226 | {'continuation': 3, 'prose': 69} |
| restore|layer_minus_mlp_delta|L17 | 0.611 | 0.611 | 3.57 | 164.12 | 5.686 | 0.749 | 4.835 | {'prose': 72} |
| restore|attn_delta|L13 | 0.250 | 0.250 | 27.94 | 139.75 | 5.649 | 0.507 | 2.837 | {'continuation': 8, 'prose': 64} |
| degradation|attn_mlp_replace_short|L13 | 0.569 | 0.431 | 4.26 | 3.26 | 5.591 | 0.524 | 2.210 | {'prose': 72} |
| restore|attn_mlp_delta|L18 | 0.250 | 0.250 | 16.86 | 150.83 | 5.576 | 0.232 | 3.009 | {'prose': 72} |
| restore|layer_minus_attn_delta|L17 | 0.722 | 0.722 | 2.88 | 164.82 | 5.443 | 0.571 | 5.188 | {'prose': 72} |
| restore|carry_est_layerout|L17 | 0.639 | 0.639 | 3.43 | 164.26 | 5.437 | 0.731 | 5.115 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|full_layer_replace_short|L30 | 0.400 | 0.600 | 1.40 | 0.40 | 1.594 | 1.019 | 1.450 | {'continuation': 5} |
| restore|layer_minus_attn_delta|L30 | 1.000 | 1.000 | 1.00 | 1.00 | 1.565 | 0.981 | 1.500 | {'continuation': 1, 'prose': 4} |
| degradation|full_layer_replace_short|L27 | 0.600 | 0.400 | 1.60 | 0.60 | 1.554 | 0.998 | 1.562 | {'continuation': 5} |
| restore|full_layer_delta|L30 | 1.000 | 1.000 | 1.00 | 1.00 | 1.546 | 0.972 | 1.550 | {'prose': 5} |
| degradation|full_layer_replace_short|L28 | 0.600 | 0.400 | 1.60 | 0.60 | 1.545 | 0.982 | 1.488 | {'continuation': 5} |
| degradation|full_layer_replace_short|L29 | 0.600 | 0.400 | 1.60 | 0.60 | 1.543 | 0.979 | 1.488 | {'continuation': 5} |
| restore|full_layer_delta|L27 | 1.000 | 1.000 | 1.00 | 1.00 | 1.538 | 0.975 | 1.650 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L29 | 1.000 | 1.000 | 1.00 | 1.00 | 1.530 | 0.962 | 1.550 | {'prose': 5} |
| restore|full_layer_delta|L28 | 1.000 | 1.000 | 1.00 | 1.00 | 1.523 | 0.955 | 1.550 | {'continuation': 1, 'prose': 4} |
| degradation|remove_carry_est_layerout|L30 | 0.200 | 0.800 | 1.20 | 0.20 | 1.506 | 0.994 | 1.425 | {'continuation': 5} |
| restore|layer_minus_mlp_delta|L30 | 1.000 | 1.000 | 1.00 | 1.00 | 1.505 | 0.993 | 1.550 | {'continuation': 2, 'prose': 3} |
| restore|carry_est_layerout|L30 | 1.000 | 1.000 | 1.00 | 1.00 | 1.494 | 0.988 | 1.500 | {'continuation': 2, 'prose': 3} |
| restore|layer_minus_attn_delta|L28 | 1.000 | 1.000 | 1.00 | 1.00 | 1.458 | 0.923 | 1.550 | {'prose': 5} |
| restore|layer_minus_attn_delta|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 1.456 | 0.893 | 1.462 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 1.451 | 0.927 | 1.538 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_attn_delta|L26 | 1.000 | 1.000 | 1.00 | 1.00 | 1.445 | 0.929 | 1.575 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 1.400 | 0.892 | 1.575 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L26 | 1.000 | 1.000 | 1.00 | 1.00 | 1.398 | 0.886 | 1.575 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_mlp_delta|L28 | 1.000 | 1.000 | 1.00 | 1.00 | 1.395 | 0.893 | 1.562 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_attn_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 1.378 | 0.888 | 1.625 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_attn_delta|L29 | 1.000 | 1.000 | 1.00 | 1.00 | 1.348 | 0.863 | 1.512 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_mlp_delta|L27 | 1.000 | 1.000 | 1.00 | 1.00 | 1.344 | 0.876 | 1.613 | {'continuation': 1, 'prose': 4} |
| restore|full_layer_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 1.342 | 0.852 | 1.525 | {'continuation': 1, 'prose': 4} |
| restore|layer_minus_attn_delta|L27 | 1.000 | 1.000 | 1.00 | 1.00 | 1.332 | 0.844 | 1.650 | {'continuation': 1, 'prose': 4} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|remove_carry_est_layerout|L23 | 1.000 | 0.000 | 3.67 | 2.67 | 9.464 | 1.329 | 3.875 | {'continuation': 1, 'prose': 2} |
| restore|full_layer_delta|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 9.068 | 1.261 | 3.458 | {'continuation': 2, 'prose': 1} |
| degradation|full_layer_replace_short|L23 | 1.000 | 0.000 | 2.67 | 1.67 | 8.964 | 1.247 | 4.083 | {'prose': 3} |
| restore|full_layer_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 8.631 | 1.200 | 3.583 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L21 | 1.000 | 1.000 | 1.00 | 1.00 | 8.498 | 1.176 | 3.125 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_attn_delta|L20 | 1.000 | 1.000 | 1.00 | 1.00 | 8.350 | 1.146 | 2.750 | {'continuation': 2, 'prose': 1} |
| degradation|full_layer_replace_short|L24 | 1.000 | 0.000 | 2.67 | 1.67 | 8.296 | 1.153 | 4.083 | {'prose': 3} |
| restore|layer_minus_attn_delta|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 8.252 | 1.156 | 2.667 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 8.198 | 1.146 | 3.000 | {'continuation': 2, 'prose': 1} |
| restore|carry_est_layerout|L21 | 1.000 | 1.000 | 1.00 | 1.00 | 8.121 | 1.128 | 2.750 | {'continuation': 1, 'prose': 2} |
| restore|full_layer_delta|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 8.097 | 1.120 | 3.333 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_attn_delta|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 8.067 | 1.107 | 3.292 | {'continuation': 2, 'prose': 1} |
| restore|carry_est_layerout|L20 | 1.000 | 1.000 | 1.00 | 1.00 | 8.044 | 1.117 | 2.333 | {'continuation': 1, 'prose': 2} |
| degradation|full_layer_replace_short|L25 | 1.000 | 0.000 | 2.33 | 1.33 | 7.764 | 1.073 | 3.708 | {'prose': 3} |
| restore|carry_est_layerout|L23 | 1.000 | 1.000 | 1.00 | 1.00 | 7.271 | 1.019 | 2.333 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L18 | 1.000 | 1.000 | 1.00 | 1.00 | 7.263 | 1.019 | 2.333 | {'continuation': 2, 'prose': 1} |
| restore|full_layer_delta|L20 | 1.000 | 1.000 | 1.00 | 1.00 | 7.199 | 0.988 | 2.250 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_attn_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 7.068 | 0.985 | 3.000 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L24 | 1.000 | 1.000 | 1.00 | 1.00 | 7.036 | 0.973 | 2.667 | {'continuation': 2, 'prose': 1} |
| restore|full_layer_delta|L18 | 1.000 | 1.000 | 1.00 | 1.00 | 6.977 | 0.955 | 2.375 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L22 | 1.000 | 1.000 | 1.00 | 1.00 | 6.966 | 0.972 | 2.458 | {'continuation': 2, 'prose': 1} |
| restore|layer_minus_mlp_delta|L20 | 1.000 | 1.000 | 1.00 | 1.00 | 6.931 | 0.968 | 1.458 | {'continuation': 1, 'prose': 2} |
| restore|full_layer_delta|L19 | 1.000 | 1.000 | 1.00 | 1.00 | 6.888 | 0.947 | 2.167 | {'continuation': 2, 'prose': 1} |
| restore|carry_est_layerout|L25 | 1.000 | 1.000 | 1.00 | 1.00 | 6.829 | 0.935 | 2.375 | {'continuation': 2, 'prose': 1} |

