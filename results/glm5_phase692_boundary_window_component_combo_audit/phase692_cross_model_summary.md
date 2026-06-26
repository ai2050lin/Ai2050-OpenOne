# Phase 692 Boundary Window Component Combo Audit

- generated: `2026-06-26 14:43:09`

| model | pairs | target | windows | best_restore | repair | target_gain | rank_effect | best_degrade | drop | target_loss | rank_effect |
|---|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|
| deepseek7b | 72 | L26_layer_input | {'all': [13, 14, 15, 16, 17, 18], 'early': [13, 14, 15], 'late': [16, 17, 18]} | restore|layer_window|late | 0.806 | 6.643 | 166.08 | degradation|layer_window|late | 0.833 | 6.536 | 33.14 |
| glm4 | 5 | L38_layer_input | {'all': [23, 24, 25, 26, 27, 28, 29, 30], 'early': [23, 24, 25, 26], 'late': [27, 28, 29, 30]} | restore|attn_mlp_window|all | 1.000 | 1.560 | 1.00 | degradation|layer_window|late | 0.400 | 1.594 | 0.40 |
| qwen3 | 3 | L33_layer_input | {'all': [18, 19, 20, 21, 22, 23, 24, 25], 'early': [18, 19, 20, 21], 'late': [22, 23, 24, 25]} | restore|attn_mlp_window|all | 1.000 | 8.213 | 1.00 | degradation|layer_window|late | 1.000 | 7.764 | 1.33 |

## Mode Averages

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|attn_mlp_window | 0.634 | 0.366 | 23.07 | 22.07 | 6.232 | 0.683 | 2.802 | {'continuation': 1, 'prose': 215} |
| degradation|attn_window | 0.315 | 0.685 | 3.60 | 2.60 | 1.867 | 0.227 | 0.653 | {'continuation': 1, 'prose': 215} |
| degradation|layer_window | 0.819 | 0.181 | 30.41 | 29.41 | 6.905 | 0.870 | 3.979 | {'prose': 216} |
| degradation|mlp_window | 0.431 | 0.569 | 3.90 | 2.90 | 5.097 | 0.297 | 1.278 | {'continuation': 1, 'prose': 215} |
| degradation|random_layer_window | 0.347 | 0.653 | 22.44 | 21.44 | 3.823 | 0.523 | 1.080 | {'continuation': 7, 'json': 2, 'prose': 207} |
| restore|attn_mlp_window | 0.435 | 0.435 | 10.88 | 156.81 | 5.906 | 0.528 | 3.973 | {'continuation': 1, 'prose': 215} |
| restore|attn_window | 0.162 | 0.162 | 57.96 | 109.73 | 3.521 | 0.480 | 2.123 | {'continuation': 5, 'prose': 211} |
| restore|layer_window | 0.736 | 0.736 | 3.18 | 164.52 | 6.512 | 0.774 | 5.376 | {'prose': 216} |
| restore|mlp_window | 0.079 | 0.079 | 145.47 | 22.22 | 1.335 | 0.165 | 0.302 | {'continuation': 8, 'prose': 208} |
| restore|random_layer_window | 0.060 | 0.060 | 434.61 | -266.92 | -1.685 | -0.081 | -1.018 | {'continuation': 26, 'prose': 190} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|attn_mlp_window | 0.067 | 0.933 | 1.07 | 0.07 | 1.308 | 0.826 | 0.979 | {'continuation': 15} |
| degradation|attn_window | 0.000 | 1.000 | 1.00 | 0.00 | 0.600 | 0.375 | 0.156 | {'continuation': 8, 'prose': 7} |
| degradation|layer_window | 0.333 | 0.667 | 1.33 | 0.33 | 1.488 | 0.949 | 1.408 | {'continuation': 15} |
| degradation|mlp_window | 0.000 | 1.000 | 1.00 | 0.00 | 0.993 | 0.615 | 0.750 | {'continuation': 15} |
| degradation|random_layer_window | 0.000 | 1.000 | 1.00 | 0.00 | -0.321 | -0.157 | -0.021 | {'continuation': 1, 'prose': 14} |
| restore|attn_mlp_window | 1.000 | 1.000 | 1.00 | 1.00 | 1.279 | 0.827 | 1.279 | {'continuation': 11, 'prose': 4} |
| restore|attn_window | 0.933 | 0.933 | 1.07 | 0.93 | 0.683 | 0.456 | 0.608 | {'continuation': 15} |
| restore|layer_window | 1.000 | 1.000 | 1.00 | 1.00 | 1.498 | 0.945 | 1.529 | {'continuation': 1, 'prose': 14} |
| restore|mlp_window | 1.000 | 1.000 | 1.00 | 1.00 | 1.058 | 0.687 | 1.217 | {'continuation': 14, 'prose': 1} |
| restore|random_layer_window | 0.000 | 0.000 | 2.07 | -0.07 | 0.072 | 0.020 | -0.338 | {'continuation': 15} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|attn_mlp_window | 0.778 | 0.222 | 2.00 | 1.00 | 5.941 | 0.824 | 3.014 | {'continuation': 3, 'prose': 6} |
| degradation|attn_window | 0.222 | 0.778 | 1.22 | 0.22 | 2.758 | 0.390 | 1.764 | {'continuation': 1, 'prose': 8} |
| degradation|layer_window | 0.667 | 0.333 | 1.89 | 0.89 | 6.499 | 0.899 | 2.986 | {'continuation': 1, 'prose': 8} |
| degradation|mlp_window | 0.111 | 0.889 | 1.11 | 0.11 | 2.446 | 0.321 | 1.236 | {'continuation': 4, 'prose': 5} |
| degradation|random_layer_window | 0.000 | 1.000 | 1.00 | 0.00 | 3.095 | 0.423 | 1.222 | {'continuation': 2, 'prose': 7} |
| restore|attn_mlp_window | 0.889 | 0.889 | 1.11 | 0.89 | 6.133 | 0.841 | 2.722 | {'continuation': 5, 'prose': 4} |
| restore|attn_window | 0.889 | 0.889 | 1.11 | 0.89 | 3.738 | 0.520 | 1.667 | {'continuation': 6, 'prose': 3} |
| restore|layer_window | 1.000 | 1.000 | 1.00 | 1.00 | 7.510 | 1.036 | 2.958 | {'continuation': 6, 'prose': 3} |
| restore|mlp_window | 0.667 | 0.667 | 1.33 | 0.67 | 3.945 | 0.535 | 1.583 | {'continuation': 5, 'prose': 4} |
| restore|random_layer_window | 0.333 | 0.333 | 2.11 | -0.11 | -0.898 | -0.116 | -0.333 | {'continuation': 5, 'prose': 4} |


## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|layer_window|late | 0.806 | 0.806 | 1.61 | 166.08 | 6.643 | 0.853 | 5.602 | {'prose': 72} |
| restore|layer_window|all | 0.806 | 0.806 | 1.61 | 166.08 | 6.643 | 0.853 | 5.602 | {'prose': 72} |
| restore|attn_mlp_window|all | 0.625 | 0.625 | 3.31 | 164.39 | 6.844 | 0.956 | 4.738 | {'prose': 72} |
| restore|layer_window|early | 0.597 | 0.597 | 6.31 | 161.39 | 6.250 | 0.617 | 4.925 | {'prose': 72} |
| restore|attn_mlp_window|early | 0.444 | 0.444 | 10.21 | 157.49 | 8.201 | 0.598 | 4.538 | {'continuation': 1, 'prose': 71} |
| restore|attn_window|early | 0.278 | 0.278 | 21.74 | 145.96 | 5.855 | 0.683 | 3.106 | {'prose': 72} |
| restore|attn_mlp_window|late | 0.236 | 0.236 | 19.12 | 148.57 | 2.672 | 0.029 | 2.642 | {'prose': 72} |
| restore|attn_window|all | 0.167 | 0.167 | 19.61 | 148.08 | 4.026 | 0.652 | 2.665 | {'prose': 72} |
| restore|mlp_window|all | 0.153 | 0.153 | 93.92 | 73.78 | 3.503 | 0.208 | 1.280 | {'prose': 72} |
| restore|random_layer_window|late | 0.069 | 0.069 | 437.14 | -269.44 | -1.476 | -0.337 | -0.985 | {'continuation': 8, 'prose': 64} |
| restore|random_layer_window|all | 0.069 | 0.069 | 437.14 | -269.44 | -1.476 | -0.337 | -0.985 | {'continuation': 8, 'prose': 64} |
| restore|mlp_window|early | 0.056 | 0.056 | 248.40 | -80.71 | -1.259 | 0.397 | -0.913 | {'continuation': 8, 'prose': 64} |
| restore|attn_window|late | 0.042 | 0.042 | 132.54 | 35.15 | 0.683 | 0.106 | 0.597 | {'continuation': 5, 'prose': 67} |
| restore|random_layer_window|early | 0.042 | 0.042 | 429.56 | -261.86 | -2.105 | 0.430 | -1.082 | {'continuation': 10, 'prose': 62} |
| restore|mlp_window|late | 0.028 | 0.028 | 94.10 | 73.60 | 1.762 | -0.111 | 0.539 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|attn_mlp_window|all | 1.000 | 1.000 | 1.00 | 1.00 | 1.560 | 1.020 | 1.450 | {'continuation': 2, 'prose': 3} |
| restore|layer_window|late | 1.000 | 1.000 | 1.00 | 1.00 | 1.552 | 0.975 | 1.512 | {'prose': 5} |
| restore|layer_window|all | 1.000 | 1.000 | 1.00 | 1.00 | 1.552 | 0.975 | 1.512 | {'prose': 5} |
| restore|attn_mlp_window|early | 1.000 | 1.000 | 1.00 | 1.00 | 1.533 | 1.043 | 1.788 | {'continuation': 4, 'prose': 1} |
| restore|layer_window|early | 1.000 | 1.000 | 1.00 | 1.00 | 1.389 | 0.884 | 1.562 | {'continuation': 1, 'prose': 4} |
| restore|mlp_window|all | 1.000 | 1.000 | 1.00 | 1.00 | 1.368 | 0.894 | 1.475 | {'continuation': 4, 'prose': 1} |
| restore|mlp_window|early | 1.000 | 1.000 | 1.00 | 1.00 | 1.310 | 0.901 | 1.700 | {'continuation': 5} |
| restore|attn_window|all | 1.000 | 1.000 | 1.00 | 1.00 | 1.043 | 0.705 | 0.863 | {'continuation': 5} |
| restore|attn_mlp_window|late | 1.000 | 1.000 | 1.00 | 1.00 | 0.745 | 0.419 | 0.600 | {'continuation': 5} |
| restore|mlp_window|late | 1.000 | 1.000 | 1.00 | 1.00 | 0.497 | 0.266 | 0.475 | {'continuation': 5} |
| restore|attn_window|early | 1.000 | 1.000 | 1.00 | 1.00 | 0.411 | 0.278 | 0.700 | {'continuation': 5} |
| restore|attn_window|late | 0.800 | 0.800 | 1.20 | 0.80 | 0.596 | 0.383 | 0.263 | {'continuation': 5} |
| restore|random_layer_window|late | 0.000 | 0.000 | 2.00 | 0.00 | 0.226 | 0.125 | -0.175 | {'continuation': 5} |
| restore|random_layer_window|all | 0.000 | 0.000 | 2.00 | 0.00 | 0.226 | 0.125 | -0.175 | {'continuation': 5} |
| restore|random_layer_window|early | 0.000 | 0.000 | 2.20 | -0.20 | -0.236 | -0.188 | -0.662 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| restore|attn_mlp_window|all | 1.000 | 1.000 | 1.00 | 1.00 | 8.213 | 1.139 | 3.708 | {'continuation': 2, 'prose': 1} |
| restore|layer_window|late | 1.000 | 1.000 | 1.00 | 1.00 | 7.970 | 1.102 | 3.333 | {'continuation': 2, 'prose': 1} |
| restore|layer_window|all | 1.000 | 1.000 | 1.00 | 1.00 | 7.970 | 1.102 | 3.333 | {'continuation': 2, 'prose': 1} |
| restore|layer_window|early | 1.000 | 1.000 | 1.00 | 1.00 | 6.589 | 0.904 | 2.208 | {'continuation': 2, 'prose': 1} |
| restore|attn_mlp_window|late | 1.000 | 1.000 | 1.00 | 1.00 | 6.458 | 0.884 | 3.208 | {'continuation': 2, 'prose': 1} |
| restore|attn_window|late | 1.000 | 1.000 | 1.00 | 1.00 | 5.553 | 0.779 | 2.583 | {'continuation': 2, 'prose': 1} |
| restore|attn_window|all | 1.000 | 1.000 | 1.00 | 1.00 | 4.124 | 0.575 | 2.042 | {'continuation': 2, 'prose': 1} |
| restore|mlp_window|all | 0.667 | 0.667 | 1.33 | 0.67 | 6.291 | 0.856 | 2.583 | {'continuation': 2, 'prose': 1} |
| restore|mlp_window|late | 0.667 | 0.667 | 1.33 | 0.67 | 3.801 | 0.516 | 1.875 | {'continuation': 2, 'prose': 1} |
| restore|attn_mlp_window|early | 0.667 | 0.667 | 1.33 | 0.67 | 3.729 | 0.502 | 1.250 | {'continuation': 1, 'prose': 2} |
| restore|mlp_window|early | 0.667 | 0.667 | 1.33 | 0.67 | 1.744 | 0.233 | 0.292 | {'continuation': 1, 'prose': 2} |
| restore|attn_window|early | 0.667 | 0.667 | 1.33 | 0.67 | 1.537 | 0.207 | 0.375 | {'continuation': 2, 'prose': 1} |
| restore|random_layer_window|late | 0.333 | 0.333 | 2.33 | -0.33 | -0.758 | -0.089 | -0.375 | {'continuation': 2, 'prose': 1} |
| restore|random_layer_window|all | 0.333 | 0.333 | 2.33 | -0.33 | -0.758 | -0.089 | -0.375 | {'continuation': 2, 'prose': 1} |
| restore|random_layer_window|early | 0.333 | 0.333 | 1.67 | 0.33 | -1.177 | -0.171 | -0.250 | {'continuation': 1, 'prose': 2} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|layer_window|late | 0.833 | 0.167 | 34.14 | 33.14 | 6.536 | 0.871 | 4.037 | {'prose': 72} |
| degradation|layer_window|all | 0.833 | 0.167 | 34.14 | 33.14 | 6.536 | 0.871 | 4.037 | {'prose': 72} |
| degradation|layer_window|early | 0.792 | 0.208 | 22.94 | 21.94 | 7.644 | 0.868 | 3.862 | {'prose': 72} |
| degradation|attn_mlp_window|early | 0.750 | 0.250 | 24.38 | 23.38 | 8.676 | 0.710 | 3.552 | {'prose': 72} |
| degradation|attn_mlp_window|all | 0.681 | 0.319 | 35.58 | 34.58 | 6.606 | 1.088 | 3.081 | {'continuation': 1, 'prose': 71} |
| degradation|attn_window|early | 0.528 | 0.472 | 5.08 | 4.08 | 4.330 | 0.359 | 1.714 | {'prose': 72} |
| degradation|mlp_window|all | 0.500 | 0.500 | 5.54 | 4.54 | 6.827 | 0.696 | 1.518 | {'continuation': 1, 'prose': 71} |
| degradation|attn_mlp_window|late | 0.472 | 0.528 | 9.26 | 8.26 | 3.413 | 0.252 | 1.773 | {'prose': 72} |
| degradation|mlp_window|early | 0.444 | 0.556 | 3.03 | 2.03 | 5.094 | -0.106 | 1.457 | {'prose': 72} |
| degradation|random_layer_window|late | 0.375 | 0.625 | 7.61 | 6.61 | 4.487 | 0.888 | 1.225 | {'continuation': 3, 'json': 1, 'prose': 68} |
| degradation|random_layer_window|all | 0.375 | 0.625 | 7.61 | 6.61 | 4.487 | 0.888 | 1.225 | {'continuation': 3, 'json': 1, 'prose': 68} |
| degradation|mlp_window|late | 0.347 | 0.653 | 3.14 | 2.14 | 3.371 | 0.301 | 0.859 | {'prose': 72} |
| degradation|attn_window|all | 0.306 | 0.694 | 4.49 | 3.49 | 1.520 | 0.230 | 0.594 | {'continuation': 1, 'prose': 71} |
| degradation|random_layer_window|early | 0.292 | 0.708 | 52.11 | 51.11 | 2.495 | -0.209 | 0.790 | {'continuation': 1, 'prose': 71} |
| degradation|attn_window|late | 0.111 | 0.889 | 1.24 | 0.24 | -0.248 | 0.092 | -0.347 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|layer_window|late | 0.400 | 0.600 | 1.40 | 0.40 | 1.594 | 1.019 | 1.450 | {'continuation': 5} |
| degradation|layer_window|all | 0.400 | 0.600 | 1.40 | 0.40 | 1.594 | 1.019 | 1.450 | {'continuation': 5} |
| degradation|attn_mlp_window|all | 0.200 | 0.800 | 1.20 | 0.20 | 1.647 | 1.070 | 1.363 | {'continuation': 5} |
| degradation|layer_window|early | 0.200 | 0.800 | 1.20 | 0.20 | 1.277 | 0.809 | 1.325 | {'continuation': 5} |
| degradation|attn_mlp_window|early | 0.000 | 1.000 | 1.00 | 0.00 | 1.320 | 0.856 | 1.275 | {'continuation': 5} |
| degradation|mlp_window|all | 0.000 | 1.000 | 1.00 | 0.00 | 1.298 | 0.831 | 1.025 | {'continuation': 5} |
| degradation|attn_window|all | 0.000 | 1.000 | 1.00 | 0.00 | 1.054 | 0.662 | 0.350 | {'continuation': 4, 'prose': 1} |
| degradation|mlp_window|early | 0.000 | 1.000 | 1.00 | 0.00 | 1.047 | 0.697 | 1.012 | {'continuation': 5} |
| degradation|attn_mlp_window|late | 0.000 | 1.000 | 1.00 | 0.00 | 0.956 | 0.552 | 0.300 | {'continuation': 5} |
| degradation|mlp_window|late | 0.000 | 1.000 | 1.00 | 0.00 | 0.633 | 0.316 | 0.212 | {'continuation': 5} |
| degradation|attn_window|late | 0.000 | 1.000 | 1.00 | 0.00 | 0.475 | 0.309 | -0.025 | {'continuation': 1, 'prose': 4} |
| degradation|attn_window|early | 0.000 | 1.000 | 1.00 | 0.00 | 0.273 | 0.156 | 0.144 | {'continuation': 3, 'prose': 2} |
| degradation|random_layer_window|early | 0.000 | 1.000 | 1.00 | 0.00 | -0.108 | -0.155 | -0.037 | {'continuation': 1, 'prose': 4} |
| degradation|random_layer_window|late | 0.000 | 1.000 | 1.00 | 0.00 | -0.427 | -0.159 | -0.013 | {'prose': 5} |
| degradation|random_layer_window|all | 0.000 | 1.000 | 1.00 | 0.00 | -0.427 | -0.159 | -0.013 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|layer_window|late | 1.000 | 0.000 | 2.33 | 1.33 | 7.764 | 1.073 | 3.708 | {'prose': 3} |
| degradation|layer_window|all | 1.000 | 0.000 | 2.33 | 1.33 | 7.764 | 1.073 | 3.708 | {'prose': 3} |
| degradation|attn_mlp_window|all | 1.000 | 0.000 | 2.67 | 1.67 | 7.397 | 1.023 | 4.083 | {'prose': 3} |
| degradation|attn_mlp_window|late | 0.667 | 0.333 | 1.67 | 0.67 | 6.372 | 0.876 | 3.250 | {'continuation': 1, 'prose': 2} |
| degradation|attn_mlp_window|early | 0.667 | 0.333 | 1.67 | 0.67 | 4.053 | 0.573 | 1.708 | {'continuation': 2, 'prose': 1} |
| degradation|attn_window|late | 0.333 | 0.667 | 1.33 | 0.33 | 4.891 | 0.680 | 2.958 | {'prose': 3} |
| degradation|mlp_window|all | 0.333 | 0.667 | 1.33 | 0.33 | 4.327 | 0.575 | 2.250 | {'continuation': 1, 'prose': 2} |
| degradation|attn_window|all | 0.333 | 0.667 | 1.33 | 0.33 | 3.394 | 0.485 | 2.250 | {'prose': 3} |
| degradation|random_layer_window|late | 0.000 | 1.000 | 1.00 | 0.00 | 4.478 | 0.615 | 1.375 | {'continuation': 1, 'prose': 2} |
| degradation|random_layer_window|all | 0.000 | 1.000 | 1.00 | 0.00 | 4.478 | 0.615 | 1.375 | {'continuation': 1, 'prose': 2} |
| degradation|layer_window|early | 0.000 | 1.000 | 1.00 | 0.00 | 3.970 | 0.550 | 1.542 | {'continuation': 1, 'prose': 2} |
| degradation|mlp_window|late | 0.000 | 1.000 | 1.00 | 0.00 | 2.449 | 0.324 | 1.500 | {'continuation': 1, 'prose': 2} |
| degradation|mlp_window|early | 0.000 | 1.000 | 1.00 | 0.00 | 0.561 | 0.063 | -0.042 | {'continuation': 2, 'prose': 1} |
| degradation|random_layer_window|early | 0.000 | 1.000 | 1.00 | 0.00 | 0.329 | 0.039 | 0.917 | {'prose': 3} |
| degradation|attn_window|early | 0.000 | 1.000 | 1.00 | 0.00 | -0.011 | 0.004 | 0.083 | {'continuation': 1, 'prose': 2} |

