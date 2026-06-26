# Phase 697 Answer-Last Route Transfer Path Decomposition

- generated: `2026-06-26 15:59:10`

| model | pairs | layers | best_restore | repair | patched_top1 | rank_effect | final_proj_effect | best_degrade | drop | patched_top1 | rank_effect | final_proj_effect |
|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | restore|L25_layer_out | 1.000 | 1.000 | 166.69 | 35.773 | degradation|L23_layer_out | 1.000 | 0.000 | 144.42 | 35.924 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | restore|L39_carry_est_layerout | 1.000 | 1.000 | 1.00 | 3.425 | degradation|L39_carry_est_layerout | 1.000 | 0.000 | 1.20 | 3.421 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | restore|L35_carry_est_layerout | 1.000 | 1.000 | 1.00 | 8.925 | degradation|L33_carry_est_layerout | 1.000 | 0.000 | 1.33 | 8.291 |

## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|L25_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | 6.294 | 35.773 | {'prose': 72} |
| restore|L26_layer_input | 1.000 | 1.000 | 1.00 | 166.69 | 6.294 | 35.773 | {'prose': 72} |
| restore|L26_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | 6.251 | 35.757 | {'prose': 72} |
| restore|L27_layer_input | 1.000 | 1.000 | 1.00 | 166.69 | 6.251 | 35.757 | {'prose': 72} |
| restore|input_window | 1.000 | 1.000 | 1.00 | 166.69 | 6.251 | 35.757 | {'prose': 72} |
| restore|L27_layer_out | 1.000 | 1.000 | 1.00 | 166.69 | 6.165 | 34.718 | {'prose': 72} |
| restore|layer_window | 1.000 | 1.000 | 1.00 | 166.69 | 6.165 | 34.718 | {'prose': 72} |
| restore|L24_layer_out | 0.986 | 0.986 | 1.01 | 166.68 | 6.332 | 35.636 | {'prose': 72} |
| restore|L25_layer_input | 0.986 | 0.986 | 1.01 | 166.68 | 6.332 | 35.636 | {'prose': 72} |
| restore|L23_layer_input | 0.986 | 0.986 | 1.01 | 166.68 | 6.416 | 35.549 | {'prose': 72} |
| restore|L23_layer_out | 0.986 | 0.986 | 1.01 | 166.68 | 6.374 | 35.531 | {'prose': 72} |
| restore|L24_layer_input | 0.986 | 0.986 | 1.01 | 166.68 | 6.374 | 35.531 | {'prose': 72} |
| restore|L25_carry_est_layerout | 0.931 | 0.931 | 1.08 | 166.61 | 6.657 | 38.721 | {'prose': 72} |
| restore|L23_carry_est_layerout | 0.903 | 0.903 | 1.11 | 166.58 | 5.952 | 28.670 | {'prose': 72} |
| restore|L24_carry_est_layerout | 0.889 | 0.889 | 1.28 | 166.42 | 5.891 | 34.608 | {'prose': 72} |
| restore|attn_mlp_window | 0.819 | 0.819 | 1.24 | 166.46 | 5.778 | 34.684 | {'continuation': 3, 'prose': 69} |
| restore|attn_window | 0.653 | 0.653 | 1.58 | 166.11 | 5.135 | 26.700 | {'prose': 72} |
| restore|L27_carry_est_layerout | 0.597 | 0.597 | 2.90 | 164.79 | 5.459 | 24.374 | {'continuation': 2, 'prose': 70} |
| restore|L26_carry_est_layerout | 0.514 | 0.514 | 3.28 | 164.42 | 4.572 | 19.055 | {'prose': 72} |
| restore|L26_attn_out | 0.069 | 0.069 | 10.74 | 156.96 | 2.431 | 12.620 | {'prose': 72} |
| restore|L23_attn_out | 0.056 | 0.056 | 44.47 | 123.22 | 1.216 | 5.543 | {'prose': 72} |
| restore|L24_mlp_out | 0.056 | 0.056 | 64.74 | 102.96 | 0.778 | 1.039 | {'continuation': 1, 'prose': 71} |
| restore|L27_attn_out | 0.042 | 0.042 | 51.43 | 116.26 | 1.001 | 5.973 | {'prose': 72} |
| restore|L27_random_layer_same_norm | 0.042 | 0.042 | 231.51 | -63.82 | 0.841 | -1.206 | {'continuation': 18, 'prose': 54} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|L39_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 1.762 | 3.425 | {'prose': 5} |
| restore|L38_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.512 | 3.296 | {'prose': 5} |
| restore|L39_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 1.512 | 3.296 | {'prose': 5} |
| restore|input_window | 1.000 | 1.000 | 1.00 | 1.00 | 1.512 | 3.296 | {'prose': 5} |
| restore|L35_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 1.663 | 3.224 | {'continuation': 1, 'prose': 4} |
| restore|L39_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.525 | 3.222 | {'prose': 5} |
| restore|layer_window | 1.000 | 1.000 | 1.00 | 1.00 | 1.525 | 3.222 | {'prose': 5} |
| restore|L37_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 1.650 | 3.206 | {'prose': 5} |
| restore|L36_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.550 | 3.202 | {'prose': 5} |
| restore|L37_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 1.550 | 3.202 | {'prose': 5} |
| restore|L37_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.562 | 3.190 | {'prose': 5} |
| restore|L38_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 1.562 | 3.190 | {'prose': 5} |
| restore|L35_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.525 | 3.063 | {'prose': 5} |
| restore|L36_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 1.525 | 3.063 | {'prose': 5} |
| restore|L34_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 1.550 | 3.037 | {'prose': 5} |
| restore|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 1.550 | 2.964 | {'continuation': 1, 'prose': 4} |
| restore|L35_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 1.550 | 2.964 | {'continuation': 1, 'prose': 4} |
| restore|L34_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 1.663 | 2.943 | {'continuation': 1, 'prose': 4} |
| restore|L36_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 1.500 | 2.821 | {'continuation': 1, 'prose': 4} |
| restore|L38_mlp_out | 1.000 | 1.000 | 1.00 | 1.00 | 0.450 | 1.488 | {'continuation': 5} |
| restore|L38_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 1.312 | 1.370 | {'continuation': 4, 'prose': 1} |
| restore|attn_mlp_window | 1.000 | 1.000 | 1.00 | 1.00 | 0.600 | 1.204 | {'continuation': 2, 'prose': 3} |
| restore|mlp_window | 1.000 | 1.000 | 1.00 | 1.00 | 0.494 | 0.965 | {'continuation': 3, 'prose': 2} |
| restore|L37_attn_out | 0.600 | 0.600 | 1.40 | 0.60 | -0.037 | -0.052 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|L35_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 3.042 | 8.925 | {'prose': 3} |
| restore|L32_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 3.000 | 8.593 | {'continuation': 2, 'prose': 1} |
| restore|L34_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 2.375 | 8.480 | {'continuation': 2, 'prose': 1} |
| restore|L34_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 2.833 | 8.370 | {'continuation': 2, 'prose': 1} |
| restore|L35_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 2.833 | 8.370 | {'continuation': 2, 'prose': 1} |
| restore|input_window | 1.000 | 1.000 | 1.00 | 1.00 | 2.833 | 8.370 | {'continuation': 2, 'prose': 1} |
| restore|L33_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 2.875 | 8.312 | {'continuation': 2, 'prose': 1} |
| restore|L34_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 2.875 | 8.312 | {'continuation': 2, 'prose': 1} |
| restore|L30_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 3.083 | 8.246 | {'continuation': 2, 'prose': 1} |
| restore|L31_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 3.083 | 8.246 | {'continuation': 2, 'prose': 1} |
| restore|L32_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 2.958 | 8.077 | {'continuation': 2, 'prose': 1} |
| restore|L33_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 2.958 | 8.077 | {'continuation': 2, 'prose': 1} |
| restore|L31_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 3.000 | 8.066 | {'continuation': 2, 'prose': 1} |
| restore|L32_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 3.000 | 8.066 | {'continuation': 2, 'prose': 1} |
| restore|L30_layer_input | 1.000 | 1.000 | 1.00 | 1.00 | 3.167 | 7.944 | {'continuation': 2, 'prose': 1} |
| restore|L33_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 2.917 | 7.821 | {'continuation': 2, 'prose': 1} |
| restore|L35_layer_out | 1.000 | 1.000 | 1.00 | 1.00 | 3.208 | 7.373 | {'continuation': 2, 'prose': 1} |
| restore|layer_window | 1.000 | 1.000 | 1.00 | 1.00 | 3.208 | 7.373 | {'continuation': 2, 'prose': 1} |
| restore|L30_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 2.625 | 6.092 | {'continuation': 2, 'prose': 1} |
| restore|L31_carry_est_layerout | 1.000 | 1.000 | 1.00 | 1.00 | 2.208 | 4.912 | {'continuation': 2, 'prose': 1} |
| restore|attn_mlp_window | 0.667 | 0.667 | 1.33 | 0.67 | 1.792 | 4.699 | {'continuation': 2, 'prose': 1} |
| restore|attn_window | 0.667 | 0.667 | 1.33 | 0.67 | 0.917 | 3.618 | {'prose': 3} |
| restore|L30_attn_out | 0.667 | 0.667 | 1.33 | 0.67 | 0.375 | 1.644 | {'continuation': 2, 'prose': 1} |
| restore|L31_mlp_out | 0.667 | 0.667 | 1.33 | 0.67 | 0.333 | 1.206 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|L23_layer_out | 1.000 | 0.000 | 145.42 | 144.42 | 6.244 | 35.924 | {'prose': 72} |
| degradation|L24_layer_input | 1.000 | 0.000 | 145.42 | 144.42 | 6.244 | 35.924 | {'prose': 72} |
| degradation|L26_layer_out | 1.000 | 0.000 | 168.12 | 167.12 | 6.207 | 35.313 | {'prose': 72} |
| degradation|L27_layer_input | 1.000 | 0.000 | 168.12 | 167.12 | 6.207 | 35.313 | {'prose': 72} |
| degradation|input_window | 1.000 | 0.000 | 168.12 | 167.12 | 6.207 | 35.313 | {'prose': 72} |
| degradation|L27_layer_out | 1.000 | 0.000 | 167.69 | 166.69 | 6.165 | 34.718 | {'prose': 72} |
| degradation|layer_window | 1.000 | 0.000 | 167.69 | 166.69 | 6.165 | 34.718 | {'prose': 72} |
| degradation|L25_carry_est_layerout | 0.986 | 0.014 | 219.57 | 218.57 | 6.668 | 39.005 | {'prose': 72} |
| degradation|L24_layer_out | 0.986 | 0.014 | 146.56 | 145.56 | 6.199 | 36.611 | {'prose': 72} |
| degradation|L25_layer_input | 0.986 | 0.014 | 146.56 | 145.56 | 6.199 | 36.611 | {'prose': 72} |
| degradation|L23_layer_input | 0.972 | 0.028 | 114.01 | 113.01 | 5.998 | 35.038 | {'prose': 72} |
| degradation|L24_carry_est_layerout | 0.972 | 0.028 | 33.22 | 32.22 | 5.088 | 33.911 | {'prose': 72} |
| degradation|L25_layer_out | 0.958 | 0.042 | 168.11 | 167.11 | 6.233 | 35.679 | {'prose': 72} |
| degradation|L26_layer_input | 0.958 | 0.042 | 168.11 | 167.11 | 6.233 | 35.679 | {'prose': 72} |
| degradation|attn_mlp_window | 0.958 | 0.042 | 99.79 | 98.79 | 5.678 | 34.695 | {'prose': 72} |
| degradation|L23_carry_est_layerout | 0.958 | 0.042 | 70.15 | 69.15 | 5.157 | 27.052 | {'continuation': 1, 'prose': 71} |
| degradation|L27_carry_est_layerout | 0.958 | 0.042 | 37.99 | 36.99 | 4.825 | 24.366 | {'continuation': 1, 'prose': 71} |
| degradation|L26_carry_est_layerout | 0.875 | 0.125 | 9.78 | 8.78 | 3.703 | 18.755 | {'continuation': 1, 'prose': 71} |
| degradation|attn_window | 0.833 | 0.167 | 87.14 | 86.14 | 4.507 | 31.503 | {'prose': 72} |
| degradation|L26_attn_out | 0.431 | 0.569 | 2.85 | 1.85 | 1.467 | 11.707 | {'prose': 72} |
| degradation|L25_random_layer_same_norm | 0.347 | 0.653 | 1.86 | 0.86 | 0.570 | 2.104 | {'continuation': 4, 'prose': 68} |
| degradation|L27_random_layer_same_norm | 0.347 | 0.653 | 1.71 | 0.71 | 0.127 | 1.206 | {'continuation': 6, 'json': 4, 'prose': 62} |
| degradation|mlp_window | 0.333 | 0.667 | 2.57 | 1.57 | 0.740 | 10.414 | {'prose': 72} |
| degradation|L23_random_layer_same_norm | 0.319 | 0.681 | 1.61 | 0.61 | 0.386 | 5.559 | {'continuation': 1, 'prose': 71} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|L39_carry_est_layerout | 1.000 | 0.000 | 2.20 | 1.20 | 1.538 | 3.421 | {'continuation': 5} |
| degradation|L38_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | 1.538 | 3.343 | {'continuation': 5} |
| degradation|L39_layer_input | 1.000 | 0.000 | 2.00 | 1.00 | 1.538 | 3.343 | {'continuation': 5} |
| degradation|input_window | 1.000 | 0.000 | 2.00 | 1.00 | 1.538 | 3.343 | {'continuation': 5} |
| degradation|L37_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | 1.550 | 3.317 | {'continuation': 5} |
| degradation|L38_layer_input | 1.000 | 0.000 | 2.00 | 1.00 | 1.550 | 3.317 | {'continuation': 5} |
| degradation|L39_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | 1.525 | 3.222 | {'continuation': 5} |
| degradation|layer_window | 1.000 | 0.000 | 2.00 | 1.00 | 1.525 | 3.222 | {'continuation': 5} |
| degradation|L35_carry_est_layerout | 0.600 | 0.400 | 1.60 | 0.60 | 1.587 | 3.494 | {'continuation': 5} |
| degradation|L34_layer_input | 0.600 | 0.400 | 1.60 | 0.60 | 1.475 | 3.273 | {'continuation': 5} |
| degradation|L35_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 1.512 | 3.226 | {'continuation': 5} |
| degradation|L36_layer_input | 0.600 | 0.400 | 1.60 | 0.60 | 1.512 | 3.226 | {'continuation': 5} |
| degradation|L34_layer_out | 0.600 | 0.400 | 1.60 | 0.60 | 1.525 | 3.211 | {'continuation': 5} |
| degradation|L35_layer_input | 0.600 | 0.400 | 1.60 | 0.60 | 1.525 | 3.211 | {'continuation': 5} |
| degradation|L37_carry_est_layerout | 0.400 | 0.600 | 1.40 | 0.40 | 1.625 | 3.445 | {'continuation': 5} |
| degradation|L36_layer_out | 0.400 | 0.600 | 1.40 | 0.40 | 1.525 | 3.336 | {'continuation': 5} |
| degradation|L37_layer_input | 0.400 | 0.600 | 1.40 | 0.40 | 1.525 | 3.336 | {'continuation': 5} |
| degradation|L34_carry_est_layerout | 0.400 | 0.600 | 1.40 | 0.40 | 1.562 | 3.099 | {'continuation': 5} |
| degradation|L36_carry_est_layerout | 0.400 | 0.600 | 1.40 | 0.40 | 1.438 | 2.990 | {'continuation': 5} |
| degradation|L38_carry_est_layerout | 0.200 | 0.800 | 1.20 | 0.20 | 1.062 | 1.649 | {'continuation': 5} |
| degradation|L39_random_layer_same_norm | 0.200 | 0.800 | 1.20 | 0.20 | 0.588 | 1.368 | {'prose': 5} |
| degradation|L38_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.237 | 1.785 | {'continuation': 4, 'prose': 1} |
| degradation|attn_mlp_window | 0.000 | 1.000 | 1.00 | 0.00 | 0.425 | 1.217 | {'continuation': 5} |
| degradation|mlp_window | 0.000 | 1.000 | 1.00 | 0.00 | 0.275 | 0.978 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | pmv_effect | final_proj_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|L33_carry_est_layerout | 1.000 | 0.000 | 2.33 | 1.33 | 3.333 | 8.291 | {'continuation': 1, 'prose': 2} |
| degradation|L35_layer_out | 1.000 | 0.000 | 2.00 | 1.00 | 3.208 | 7.373 | {'continuation': 2, 'prose': 1} |
| degradation|layer_window | 1.000 | 0.000 | 2.00 | 1.00 | 3.208 | 7.373 | {'continuation': 2, 'prose': 1} |
| degradation|L33_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | 3.333 | 8.425 | {'continuation': 1, 'prose': 2} |
| degradation|L34_layer_input | 0.667 | 0.333 | 1.67 | 0.67 | 3.333 | 8.425 | {'continuation': 1, 'prose': 2} |
| degradation|L34_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | 3.250 | 8.347 | {'continuation': 2, 'prose': 1} |
| degradation|L35_layer_input | 0.667 | 0.333 | 1.67 | 0.67 | 3.250 | 8.347 | {'continuation': 2, 'prose': 1} |
| degradation|input_window | 0.667 | 0.333 | 1.67 | 0.67 | 3.250 | 8.347 | {'continuation': 2, 'prose': 1} |
| degradation|L32_layer_out | 0.667 | 0.333 | 1.67 | 0.67 | 3.167 | 7.947 | {'continuation': 1, 'prose': 2} |
| degradation|L33_layer_input | 0.667 | 0.333 | 1.67 | 0.67 | 3.167 | 7.947 | {'continuation': 1, 'prose': 2} |
| degradation|L35_carry_est_layerout | 0.333 | 0.667 | 1.33 | 0.33 | 3.250 | 8.917 | {'continuation': 2, 'prose': 1} |
| degradation|L32_carry_est_layerout | 0.333 | 0.667 | 1.33 | 0.33 | 3.458 | 8.832 | {'continuation': 1, 'prose': 2} |
| degradation|L31_layer_out | 0.333 | 0.667 | 1.33 | 0.33 | 3.292 | 8.500 | {'prose': 3} |
| degradation|L32_layer_input | 0.333 | 0.667 | 1.33 | 0.33 | 3.292 | 8.500 | {'prose': 3} |
| degradation|L34_carry_est_layerout | 0.333 | 0.667 | 1.33 | 0.33 | 2.625 | 8.450 | {'continuation': 1, 'prose': 2} |
| degradation|L30_layer_input | 0.333 | 0.667 | 1.33 | 0.33 | 3.542 | 8.431 | {'prose': 3} |
| degradation|L30_layer_out | 0.333 | 0.667 | 1.33 | 0.33 | 3.417 | 8.418 | {'prose': 3} |
| degradation|L31_layer_input | 0.333 | 0.667 | 1.33 | 0.33 | 3.417 | 8.418 | {'prose': 3} |
| degradation|L30_carry_est_layerout | 0.333 | 0.667 | 1.33 | 0.33 | 2.750 | 6.268 | {'prose': 3} |
| degradation|L31_carry_est_layerout | 0.333 | 0.667 | 1.33 | 0.33 | 2.500 | 5.548 | {'continuation': 1, 'prose': 2} |
| degradation|attn_window | 0.333 | 0.667 | 1.33 | 0.33 | 1.125 | 3.845 | {'continuation': 2, 'prose': 1} |
| degradation|attn_mlp_window | 0.000 | 1.000 | 1.00 | 0.00 | 1.875 | 4.644 | {'continuation': 2, 'prose': 1} |
| degradation|L31_attn_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.542 | 2.239 | {'continuation': 1, 'prose': 2} |
| degradation|L31_mlp_out | 0.000 | 1.000 | 1.00 | 0.00 | 0.542 | 2.102 | {'continuation': 2, 'prose': 1} |

