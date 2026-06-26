# Phase 703 Holdout Source-Restricted Channel Validation

- generated: `2026-06-26 18:14:59`

| model | pairs | layers | top_heads | best_restore | change | rank_effect | final_effect | best_degrade | drop | rank_effect | final_effect |
|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | 32 | restore|all_positive_source_channels | 0.722 | 166.01 | 40.179 | degradation|all_positive_source_channels | 0.778 | 55.28 | 36.700 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | 32 | restore|all_positive_source_channels | 0.000 | 0.00 | 1.312 | degradation|all_positive_source_channels | 0.000 | 0.00 | 1.710 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | 32 | restore|all_positive_source_channels | 1.000 | 1.00 | 14.870 | degradation|all_positive_source_channels | 0.333 | 0.33 | 15.195 |

## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 0.722 | 0.722 | 1.68 | 166.01 | 40.179 | 2733.5 | {'continuation': 8, 'prose': 64} |
| restore|source_top_channel_512 | 0.361 | 0.361 | 8.68 | 159.01 | 29.477 | 512.0 | {'continuation': 4, 'prose': 68} |
| restore|source_top_channel_256 | 0.194 | 0.194 | 18.35 | 149.35 | 21.665 | 256.0 | {'continuation': 3, 'prose': 69} |
| restore|source_top_channel_128 | 0.069 | 0.069 | 28.38 | 139.32 | 15.211 | 128.0 | {'continuation': 2, 'prose': 70} |
| restore|source_top_channel_64 | 0.042 | 0.042 | 35.65 | 132.04 | 10.832 | 64.0 | {'continuation': 2, 'prose': 70} |
| restore|source_random_channel_512 | 0.028 | 0.028 | 58.90 | 108.79 | 3.172 | 512.0 | {'prose': 72} |
| restore|source_top_channel_32 | 0.014 | 0.014 | 52.74 | 114.96 | 8.305 | 32.0 | {'continuation': 2, 'prose': 70} |
| restore|source_random_channel_256 | 0.014 | 0.014 | 88.21 | 79.49 | 2.747 | 256.0 | {'prose': 72} |
| restore|source_random_channel_64 | 0.014 | 0.014 | 154.62 | 13.07 | 0.209 | 64.0 | {'prose': 72} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 127.96 | 39.74 | 0.757 | 128.0 | {'prose': 72} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 151.49 | 16.21 | 0.230 | 32.0 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 0.000 | 0.000 | 2.00 | 0.00 | 1.312 | 2223.8 | {'continuation': 5} |
| restore|source_top_channel_512 | 0.000 | 0.000 | 2.00 | 0.00 | 0.937 | 512.0 | {'continuation': 5} |
| restore|source_top_channel_256 | 0.000 | 0.000 | 2.00 | 0.00 | 0.650 | 256.0 | {'continuation': 5} |
| restore|source_top_channel_128 | 0.000 | 0.000 | 2.00 | 0.00 | 0.502 | 128.0 | {'continuation': 5} |
| restore|source_top_channel_64 | 0.000 | 0.000 | 2.00 | 0.00 | 0.299 | 64.0 | {'continuation': 5} |
| restore|source_top_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | 0.199 | 32.0 | {'continuation': 5} |
| restore|source_random_channel_512 | 0.000 | 0.000 | 2.00 | 0.00 | 0.135 | 512.0 | {'continuation': 5} |
| restore|source_random_channel_64 | 0.000 | 0.000 | 2.00 | 0.00 | 0.000 | 64.0 | {'continuation': 5} |
| restore|source_random_channel_256 | 0.000 | 0.000 | 2.00 | 0.00 | -0.002 | 256.0 | {'continuation': 5} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 2.00 | 0.00 | -0.049 | 128.0 | {'continuation': 5} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | -0.050 | 32.0 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 1.000 | 1.000 | 1.00 | 1.00 | 14.870 | 2281.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_512 | 1.000 | 1.000 | 1.00 | 1.00 | 10.768 | 512.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_256 | 0.667 | 0.667 | 1.33 | 0.67 | 7.519 | 256.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_128 | 0.667 | 0.667 | 1.33 | 0.67 | 5.155 | 128.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_64 | 0.667 | 0.667 | 1.33 | 0.67 | 3.516 | 64.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_32 | 0.667 | 0.667 | 1.33 | 0.67 | 2.378 | 32.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_256 | 0.667 | 0.667 | 1.33 | 0.67 | 0.201 | 256.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_512 | 0.667 | 0.667 | 1.33 | 0.67 | -0.020 | 512.0 | {'continuation': 1, 'prose': 2} |
| restore|source_random_channel_64 | 0.333 | 0.333 | 1.67 | 0.33 | 0.116 | 64.0 | {'continuation': 1, 'prose': 2} |
| restore|source_random_channel_128 | 0.333 | 0.333 | 2.00 | 0.00 | -0.448 | 128.0 | {'continuation': 1, 'prose': 2} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | -0.189 | 32.0 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.778 | 0.222 | 56.28 | 55.28 | 36.700 | 2733.5 | {'prose': 72} |
| degradation|source_top_channel_512 | 0.569 | 0.431 | 12.89 | 11.89 | 28.061 | 512.0 | {'prose': 72} |
| degradation|source_top_channel_256 | 0.500 | 0.500 | 5.36 | 4.36 | 21.161 | 256.0 | {'prose': 72} |
| degradation|source_top_channel_128 | 0.431 | 0.569 | 2.85 | 1.85 | 15.571 | 128.0 | {'prose': 72} |
| degradation|source_top_channel_64 | 0.306 | 0.694 | 1.79 | 0.79 | 10.176 | 64.0 | {'prose': 72} |
| degradation|source_top_channel_32 | 0.153 | 0.847 | 1.25 | 0.25 | 7.542 | 32.0 | {'prose': 72} |
| degradation|source_random_channel_512 | 0.111 | 0.889 | 1.12 | 0.12 | 1.858 | 512.0 | {'prose': 72} |
| degradation|source_random_channel_256 | 0.056 | 0.944 | 1.07 | 0.07 | 1.990 | 256.0 | {'prose': 72} |
| degradation|source_random_channel_64 | 0.028 | 0.972 | 1.03 | 0.03 | 0.354 | 64.0 | {'prose': 72} |
| degradation|source_random_channel_128 | 0.014 | 0.986 | 1.01 | 0.01 | 0.666 | 128.0 | {'prose': 72} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.212 | 32.0 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.000 | 1.000 | 1.00 | 0.00 | 1.710 | 2223.8 | {'prose': 5} |
| degradation|source_top_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 1.247 | 512.0 | {'prose': 5} |
| degradation|source_top_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 0.903 | 256.0 | {'prose': 5} |
| degradation|source_top_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 0.710 | 128.0 | {'prose': 5} |
| degradation|source_top_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 0.460 | 64.0 | {'prose': 5} |
| degradation|source_top_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.284 | 32.0 | {'prose': 5} |
| degradation|source_random_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 0.172 | 512.0 | {'prose': 5} |
| degradation|source_random_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | -0.010 | 256.0 | {'prose': 5} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | -0.019 | 64.0 | {'prose': 5} |
| degradation|source_random_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | -0.087 | 128.0 | {'prose': 5} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | -0.104 | 32.0 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.333 | 0.667 | 1.33 | 0.33 | 15.195 | 2281.0 | {'prose': 3} |
| degradation|source_top_channel_512 | 0.333 | 0.667 | 1.33 | 0.33 | 11.381 | 512.0 | {'prose': 3} |
| degradation|source_top_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 7.783 | 256.0 | {'prose': 3} |
| degradation|source_top_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 5.577 | 128.0 | {'prose': 3} |
| degradation|source_top_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 3.870 | 64.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_top_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 2.491 | 32.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 0.746 | 256.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.105 | 32.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | -0.043 | 64.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | -0.059 | 128.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | -0.117 | 512.0 | {'continuation': 2, 'prose': 1} |

