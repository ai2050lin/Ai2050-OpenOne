# Phase 701 Head-Channel Causal Bridge Audit

- generated: `2026-06-26 17:53:32`

| model | pairs | layers | top_heads | best_restore | change | rank_effect | final_effect | best_degrade | drop | rank_effect | final_effect |
|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | 32 | restore|all_positive_channels | 0.750 | 165.96 | 43.798 | degradation|all_positive_channels | 0.833 | 70.78 | 42.742 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | 32 | restore|all_positive_channels | 0.200 | 0.20 | 1.530 | degradation|all_positive_channels | 0.000 | 0.00 | 2.136 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | 32 | restore|all_positive_channels | 1.000 | 1.00 | 15.995 | degradation|all_positive_channels | 1.000 | 1.00 | 15.476 |

## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_channels | 0.750 | 0.750 | 1.74 | 165.96 | 43.798 | 2745.0 | {'continuation': 8, 'prose': 64} |
| restore|top_channel_512 | 0.361 | 0.361 | 10.40 | 157.29 | 29.406 | 512.0 | {'continuation': 2, 'prose': 70} |
| restore|top_channel_256 | 0.181 | 0.181 | 20.67 | 147.03 | 22.693 | 256.0 | {'prose': 72} |
| restore|top_channel_128 | 0.083 | 0.083 | 37.12 | 130.57 | 16.569 | 128.0 | {'prose': 72} |
| restore|top_channel_64 | 0.056 | 0.056 | 42.33 | 125.36 | 9.423 | 64.0 | {'prose': 72} |
| restore|top_channel_32 | 0.042 | 0.042 | 53.43 | 114.26 | 8.207 | 32.0 | {'continuation': 1, 'prose': 71} |
| restore|random_channel_64 | 0.028 | 0.028 | 143.38 | 24.32 | 0.557 | 64.0 | {'prose': 72} |
| restore|random_channel_512 | 0.014 | 0.014 | 57.92 | 109.78 | 2.988 | 512.0 | {'prose': 72} |
| restore|random_channel_32 | 0.014 | 0.014 | 175.93 | -8.24 | -0.554 | 32.0 | {'continuation': 1, 'prose': 71} |
| restore|random_channel_256 | 0.000 | 0.000 | 115.29 | 52.40 | 1.286 | 256.0 | {'prose': 72} |
| restore|random_channel_128 | 0.000 | 0.000 | 162.60 | 5.10 | 0.029 | 128.0 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_channels | 0.200 | 0.200 | 1.80 | 0.20 | 1.530 | 2345.0 | {'continuation': 5} |
| restore|top_channel_512 | 0.000 | 0.000 | 2.00 | 0.00 | 0.819 | 512.0 | {'continuation': 5} |
| restore|top_channel_256 | 0.000 | 0.000 | 2.00 | 0.00 | 0.648 | 256.0 | {'continuation': 5} |
| restore|top_channel_128 | 0.000 | 0.000 | 2.00 | 0.00 | 0.378 | 128.0 | {'continuation': 5} |
| restore|top_channel_64 | 0.000 | 0.000 | 2.00 | 0.00 | 0.217 | 64.0 | {'continuation': 5} |
| restore|top_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | 0.127 | 32.0 | {'continuation': 5} |
| restore|random_channel_512 | 0.000 | 0.000 | 2.00 | 0.00 | 0.056 | 512.0 | {'continuation': 5} |
| restore|random_channel_128 | 0.000 | 0.000 | 2.00 | 0.00 | 0.032 | 128.0 | {'continuation': 5} |
| restore|random_channel_256 | 0.000 | 0.000 | 2.00 | 0.00 | 0.010 | 256.0 | {'continuation': 5} |
| restore|random_channel_64 | 0.000 | 0.000 | 2.00 | 0.00 | -0.010 | 64.0 | {'continuation': 5} |
| restore|random_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | -0.049 | 32.0 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_channels | 1.000 | 1.000 | 1.00 | 1.00 | 15.995 | 2313.0 | {'continuation': 2, 'prose': 1} |
| restore|top_channel_512 | 1.000 | 1.000 | 1.00 | 1.00 | 10.533 | 512.0 | {'continuation': 2, 'prose': 1} |
| restore|top_channel_256 | 0.667 | 0.667 | 1.33 | 0.67 | 7.349 | 256.0 | {'continuation': 2, 'prose': 1} |
| restore|top_channel_128 | 0.667 | 0.667 | 1.33 | 0.67 | 4.681 | 128.0 | {'continuation': 2, 'prose': 1} |
| restore|top_channel_32 | 0.667 | 0.667 | 1.33 | 0.67 | 2.038 | 32.0 | {'continuation': 2, 'prose': 1} |
| restore|random_channel_512 | 0.667 | 0.667 | 1.33 | 0.67 | 0.343 | 512.0 | {'continuation': 1, 'prose': 2} |
| restore|top_channel_64 | 0.333 | 0.333 | 1.67 | 0.33 | 2.807 | 64.0 | {'continuation': 2, 'prose': 1} |
| restore|random_channel_128 | 0.333 | 0.333 | 1.67 | 0.33 | -0.016 | 128.0 | {'continuation': 2, 'prose': 1} |
| restore|random_channel_256 | 0.333 | 0.333 | 1.67 | 0.33 | -0.075 | 256.0 | {'continuation': 2, 'prose': 1} |
| restore|random_channel_64 | 0.000 | 0.000 | 2.00 | 0.00 | 0.199 | 64.0 | {'continuation': 2, 'prose': 1} |
| restore|random_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | -0.324 | 32.0 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_channels | 0.833 | 0.167 | 71.78 | 70.78 | 42.742 | 2745.0 | {'prose': 72} |
| degradation|top_channel_512 | 0.556 | 0.444 | 10.29 | 9.29 | 27.851 | 512.0 | {'prose': 72} |
| degradation|top_channel_256 | 0.486 | 0.514 | 4.53 | 3.53 | 21.736 | 256.0 | {'prose': 72} |
| degradation|top_channel_128 | 0.431 | 0.569 | 2.78 | 1.78 | 16.118 | 128.0 | {'prose': 72} |
| degradation|top_channel_64 | 0.319 | 0.681 | 1.92 | 0.92 | 10.722 | 64.0 | {'prose': 72} |
| degradation|top_channel_32 | 0.153 | 0.847 | 1.26 | 0.26 | 7.355 | 32.0 | {'prose': 72} |
| degradation|random_channel_512 | 0.111 | 0.889 | 1.11 | 0.11 | 2.142 | 512.0 | {'prose': 72} |
| degradation|random_channel_256 | 0.056 | 0.944 | 1.07 | 0.07 | 0.956 | 256.0 | {'prose': 72} |
| degradation|random_channel_32 | 0.028 | 0.972 | 1.03 | 0.03 | -0.505 | 32.0 | {'prose': 72} |
| degradation|random_channel_64 | 0.014 | 0.986 | 1.01 | 0.01 | 0.319 | 64.0 | {'prose': 72} |
| degradation|random_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 1.448 | 128.0 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_channels | 0.000 | 1.000 | 1.00 | 0.00 | 2.136 | 2345.0 | {'prose': 5} |
| degradation|top_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 1.221 | 512.0 | {'prose': 5} |
| degradation|top_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 0.930 | 256.0 | {'prose': 5} |
| degradation|top_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 0.571 | 128.0 | {'prose': 5} |
| degradation|top_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 0.381 | 64.0 | {'prose': 5} |
| degradation|top_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.209 | 32.0 | {'prose': 5} |
| degradation|random_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 0.157 | 512.0 | {'prose': 5} |
| degradation|random_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 0.009 | 64.0 | {'prose': 5} |
| degradation|random_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | -0.025 | 128.0 | {'prose': 5} |
| degradation|random_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | -0.040 | 256.0 | {'prose': 5} |
| degradation|random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | -0.100 | 32.0 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_channels | 1.000 | 0.000 | 2.00 | 1.00 | 15.476 | 2313.0 | {'prose': 3} |
| degradation|top_channel_512 | 0.333 | 0.667 | 1.33 | 0.33 | 10.578 | 512.0 | {'prose': 3} |
| degradation|top_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 7.618 | 256.0 | {'continuation': 1, 'prose': 2} |
| degradation|top_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 4.987 | 128.0 | {'continuation': 1, 'prose': 2} |
| degradation|top_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 3.415 | 64.0 | {'continuation': 1, 'prose': 2} |
| degradation|top_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 2.475 | 32.0 | {'continuation': 2, 'prose': 1} |
| degradation|random_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 0.884 | 512.0 | {'continuation': 2, 'prose': 1} |
| degradation|random_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 0.330 | 128.0 | {'continuation': 2, 'prose': 1} |
| degradation|random_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 0.267 | 64.0 | {'continuation': 2, 'prose': 1} |
| degradation|random_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 0.244 | 256.0 | {'continuation': 2, 'prose': 1} |
| degradation|random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.005 | 32.0 | {'continuation': 2, 'prose': 1} |

