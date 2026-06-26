# Phase 702 Source-Restricted Channel Ensemble Audit

- generated: `2026-06-26 18:00:48`

| model | pairs | layers | top_heads | best_restore | change | rank_effect | final_effect | best_degrade | drop | rank_effect | final_effect |
|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | 32 | restore|all_positive_source_channels | 0.764 | 166.19 | 40.229 | degradation|all_positive_source_channels | 0.806 | 66.03 | 37.597 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | 32 | restore|all_positive_source_channels | 0.000 | 0.00 | 1.865 | degradation|all_positive_source_channels | 0.000 | 0.00 | 2.312 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | 32 | restore|all_positive_source_channels | 1.000 | 1.00 | 16.595 | degradation|all_positive_source_channels | 0.667 | 0.67 | 16.538 |

## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 0.764 | 0.764 | 1.50 | 166.19 | 40.229 | 2838.0 | {'continuation': 8, 'prose': 64} |
| restore|source_top_channel_512 | 0.403 | 0.403 | 6.14 | 161.56 | 30.415 | 512.0 | {'continuation': 7, 'prose': 65} |
| restore|source_top_channel_256 | 0.208 | 0.208 | 18.29 | 149.40 | 21.535 | 256.0 | {'continuation': 3, 'prose': 69} |
| restore|source_top_channel_128 | 0.069 | 0.069 | 28.61 | 139.08 | 15.626 | 128.0 | {'continuation': 3, 'prose': 69} |
| restore|source_top_channel_64 | 0.069 | 0.069 | 32.62 | 135.07 | 11.634 | 64.0 | {'continuation': 2, 'prose': 70} |
| restore|source_random_channel_512 | 0.028 | 0.028 | 49.94 | 117.75 | 3.300 | 512.0 | {'prose': 72} |
| restore|source_top_channel_32 | 0.014 | 0.014 | 52.54 | 115.15 | 8.208 | 32.0 | {'continuation': 1, 'prose': 71} |
| restore|source_random_channel_64 | 0.014 | 0.014 | 153.65 | 14.04 | 0.244 | 64.0 | {'prose': 72} |
| restore|source_random_channel_256 | 0.000 | 0.000 | 90.49 | 77.21 | 1.711 | 256.0 | {'prose': 72} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 113.74 | 53.96 | 0.514 | 128.0 | {'prose': 72} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 159.54 | 8.15 | 0.388 | 32.0 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 0.000 | 0.000 | 2.00 | 0.00 | 1.865 | 2280.0 | {'continuation': 5} |
| restore|source_top_channel_512 | 0.000 | 0.000 | 2.00 | 0.00 | 1.343 | 512.0 | {'continuation': 5} |
| restore|source_top_channel_256 | 0.000 | 0.000 | 2.00 | 0.00 | 0.898 | 256.0 | {'continuation': 5} |
| restore|source_top_channel_128 | 0.000 | 0.000 | 2.00 | 0.00 | 0.662 | 128.0 | {'continuation': 5} |
| restore|source_top_channel_64 | 0.000 | 0.000 | 2.00 | 0.00 | 0.390 | 64.0 | {'continuation': 5} |
| restore|source_top_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | 0.238 | 32.0 | {'continuation': 5} |
| restore|source_random_channel_512 | 0.000 | 0.000 | 2.00 | 0.00 | 0.102 | 512.0 | {'continuation': 5} |
| restore|source_random_channel_256 | 0.000 | 0.000 | 2.00 | 0.00 | 0.035 | 256.0 | {'continuation': 5} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | -0.001 | 32.0 | {'continuation': 5} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 2.00 | 0.00 | -0.002 | 128.0 | {'continuation': 5} |
| restore|source_random_channel_64 | 0.000 | 0.000 | 2.00 | 0.00 | -0.063 | 64.0 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 1.000 | 1.000 | 1.00 | 1.00 | 16.595 | 2275.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_512 | 1.000 | 1.000 | 1.00 | 1.00 | 12.108 | 512.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_256 | 0.667 | 0.667 | 1.33 | 0.67 | 9.183 | 256.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_128 | 0.667 | 0.667 | 1.33 | 0.67 | 5.635 | 128.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_64 | 0.667 | 0.667 | 1.33 | 0.67 | 4.053 | 64.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_32 | 0.667 | 0.667 | 1.33 | 0.67 | 2.416 | 32.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_128 | 0.667 | 0.667 | 1.33 | 0.67 | -0.658 | 128.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_512 | 0.333 | 0.333 | 1.67 | 0.33 | 0.524 | 512.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_256 | 0.333 | 0.333 | 1.67 | 0.33 | 0.213 | 256.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_64 | 0.333 | 0.333 | 2.00 | 0.00 | -0.344 | 64.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | -0.085 | 32.0 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.806 | 0.194 | 67.03 | 66.03 | 37.597 | 2838.0 | {'prose': 72} |
| degradation|source_top_channel_512 | 0.611 | 0.389 | 13.15 | 12.15 | 28.326 | 512.0 | {'prose': 72} |
| degradation|source_top_channel_256 | 0.542 | 0.458 | 5.42 | 4.42 | 20.793 | 256.0 | {'prose': 72} |
| degradation|source_top_channel_128 | 0.431 | 0.569 | 2.81 | 1.81 | 15.930 | 128.0 | {'prose': 72} |
| degradation|source_top_channel_64 | 0.306 | 0.694 | 2.00 | 1.00 | 10.588 | 64.0 | {'prose': 72} |
| degradation|source_random_channel_512 | 0.167 | 0.833 | 1.22 | 0.22 | 2.073 | 512.0 | {'prose': 72} |
| degradation|source_top_channel_32 | 0.153 | 0.847 | 1.25 | 0.25 | 7.325 | 32.0 | {'prose': 72} |
| degradation|source_random_channel_256 | 0.069 | 0.931 | 1.08 | 0.08 | 1.124 | 256.0 | {'prose': 72} |
| degradation|source_random_channel_128 | 0.056 | 0.944 | 1.06 | 0.06 | 0.894 | 128.0 | {'prose': 72} |
| degradation|source_random_channel_64 | 0.014 | 0.986 | 1.01 | 0.01 | -0.174 | 64.0 | {'prose': 72} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.026 | 32.0 | {'prose': 72} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.000 | 1.000 | 1.00 | 0.00 | 2.312 | 2280.0 | {'prose': 5} |
| degradation|source_top_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 1.655 | 512.0 | {'prose': 5} |
| degradation|source_top_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 1.139 | 256.0 | {'prose': 5} |
| degradation|source_top_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 0.881 | 128.0 | {'prose': 5} |
| degradation|source_top_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 0.521 | 64.0 | {'prose': 5} |
| degradation|source_top_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.346 | 32.0 | {'prose': 5} |
| degradation|source_random_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 0.178 | 512.0 | {'prose': 5} |
| degradation|source_random_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 0.020 | 128.0 | {'prose': 5} |
| degradation|source_random_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | -0.002 | 256.0 | {'prose': 5} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | -0.075 | 32.0 | {'prose': 5} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | -0.105 | 64.0 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.667 | 0.333 | 1.67 | 0.67 | 16.538 | 2275.0 | {'prose': 3} |
| degradation|source_top_channel_512 | 0.333 | 0.667 | 1.33 | 0.33 | 12.675 | 512.0 | {'prose': 3} |
| degradation|source_top_channel_256 | 0.333 | 0.667 | 1.33 | 0.33 | 9.357 | 256.0 | {'continuation': 1, 'prose': 2} |
| degradation|source_top_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 6.027 | 128.0 | {'prose': 3} |
| degradation|source_top_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 4.573 | 64.0 | {'continuation': 1, 'prose': 2} |
| degradation|source_top_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 2.576 | 32.0 | {'continuation': 1, 'prose': 2} |
| degradation|source_random_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 0.790 | 512.0 | {'continuation': 1, 'prose': 2} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.330 | 32.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 0.307 | 256.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 0.070 | 128.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | -0.020 | 64.0 | {'continuation': 2, 'prose': 1} |

