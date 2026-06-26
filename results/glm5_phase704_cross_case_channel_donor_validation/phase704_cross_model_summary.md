# Phase 704 Cross-Case Source-Restricted Channel Donor Validation

- generated: `2026-06-26 18:20:51`

| model | pairs | layers | top_heads | best_restore | change | rank_effect | final_effect | best_degrade | drop | rank_effect | final_effect |
|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | 32 | restore|all_positive_source_channels | 0.425 | -375.50 | 13.101 | degradation|all_positive_source_channels | 0.988 | 279.18 | 46.337 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | 32 | restore|all_positive_source_channels | 0.200 | 0.20 | 1.540 | degradation|all_positive_source_channels | 0.000 | 0.00 | 2.363 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | 32 | restore|all_positive_source_channels | 1.000 | 1.00 | 17.380 | degradation|all_positive_source_channels | 0.667 | 1.00 | 16.159 |

## Best Restore

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 0.425 | 0.425 | 528.89 | -375.50 | 13.101 | 2838.0 | {'continuation': 7, 'prose': 73} |
| restore|source_top_channel_512 | 0.287 | 0.287 | 175.68 | -22.29 | 18.562 | 512.0 | {'continuation': 3, 'prose': 77} |
| restore|source_top_channel_256 | 0.087 | 0.087 | 142.43 | 10.96 | 13.452 | 256.0 | {'continuation': 1, 'prose': 79} |
| restore|source_top_channel_128 | 0.062 | 0.062 | 107.64 | 45.75 | 10.066 | 128.0 | {'prose': 80} |
| restore|source_top_channel_64 | 0.062 | 0.062 | 88.05 | 65.34 | 7.873 | 64.0 | {'continuation': 2, 'prose': 78} |
| restore|source_top_channel_32 | 0.050 | 0.050 | 86.83 | 66.56 | 5.704 | 32.0 | {'continuation': 1, 'prose': 79} |
| restore|source_random_channel_512 | 0.037 | 0.037 | 96.35 | 57.04 | 2.637 | 512.0 | {'continuation': 1, 'prose': 79} |
| restore|source_random_channel_64 | 0.013 | 0.013 | 147.36 | 6.03 | 0.097 | 64.0 | {'prose': 80} |
| restore|source_random_channel_256 | 0.000 | 0.000 | 109.08 | 44.31 | 0.842 | 256.0 | {'prose': 80} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 152.91 | 0.47 | 0.109 | 32.0 | {'prose': 80} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 146.49 | 6.90 | -1.033 | 128.0 | {'prose': 80} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 0.200 | 0.200 | 1.80 | 0.20 | 1.540 | 2280.0 | {'continuation': 5} |
| restore|source_random_channel_512 | 0.200 | 0.200 | 1.80 | 0.20 | 0.058 | 512.0 | {'continuation': 5} |
| restore|source_random_channel_128 | 0.200 | 0.200 | 1.80 | 0.20 | 0.001 | 128.0 | {'continuation': 5} |
| restore|source_top_channel_512 | 0.000 | 0.000 | 2.00 | 0.00 | 1.229 | 512.0 | {'continuation': 5} |
| restore|source_top_channel_256 | 0.000 | 0.000 | 2.00 | 0.00 | 0.784 | 256.0 | {'continuation': 5} |
| restore|source_top_channel_128 | 0.000 | 0.000 | 2.00 | 0.00 | 0.531 | 128.0 | {'continuation': 5} |
| restore|source_top_channel_64 | 0.000 | 0.000 | 2.00 | 0.00 | 0.336 | 64.0 | {'continuation': 5} |
| restore|source_top_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | 0.215 | 32.0 | {'continuation': 5} |
| restore|source_random_channel_64 | 0.000 | 0.000 | 2.00 | 0.00 | 0.006 | 64.0 | {'continuation': 5} |
| restore|source_random_channel_256 | 0.000 | 0.000 | 2.00 | 0.00 | -0.015 | 256.0 | {'continuation': 5} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 2.00 | 0.00 | -0.020 | 32.0 | {'continuation': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 1.000 | 1.000 | 1.00 | 1.00 | 17.380 | 2275.0 | {'continuation': 3} |
| restore|source_top_channel_512 | 1.000 | 1.000 | 1.00 | 1.00 | 12.331 | 512.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_256 | 1.000 | 1.000 | 1.00 | 1.00 | 8.857 | 256.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_128 | 0.667 | 0.667 | 1.33 | 0.67 | 5.189 | 128.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_64 | 0.667 | 0.667 | 1.33 | 0.67 | 4.291 | 64.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_32 | 0.667 | 0.667 | 1.33 | 0.67 | 2.399 | 32.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_32 | 0.333 | 0.333 | 1.67 | 0.33 | 0.163 | 32.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_512 | 0.333 | 0.333 | 1.67 | 0.33 | -0.091 | 512.0 | {'continuation': 1, 'prose': 2} |
| restore|source_random_channel_256 | 0.333 | 0.333 | 1.67 | 0.33 | -0.204 | 256.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_64 | 0.333 | 0.333 | 2.00 | 0.00 | -0.347 | 64.0 | {'continuation': 1, 'prose': 2} |
| restore|source_random_channel_128 | 0.333 | 0.333 | 1.67 | 0.33 | -1.003 | 128.0 | {'continuation': 2, 'prose': 1} |


## Best Degradation

### deepseek7b

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.988 | 0.013 | 280.18 | 279.18 | 46.337 | 2838.0 | {'prose': 80} |
| degradation|source_top_channel_512 | 0.637 | 0.362 | 56.75 | 55.75 | 30.039 | 512.0 | {'prose': 80} |
| degradation|source_top_channel_256 | 0.512 | 0.487 | 12.06 | 11.06 | 21.781 | 256.0 | {'prose': 80} |
| degradation|source_top_channel_128 | 0.412 | 0.588 | 3.98 | 2.98 | 16.059 | 128.0 | {'prose': 80} |
| degradation|source_top_channel_64 | 0.312 | 0.688 | 2.24 | 1.24 | 10.706 | 64.0 | {'prose': 80} |
| degradation|source_top_channel_32 | 0.188 | 0.812 | 1.31 | 0.31 | 7.515 | 32.0 | {'prose': 80} |
| degradation|source_random_channel_512 | 0.163 | 0.838 | 1.26 | 0.26 | 4.953 | 512.0 | {'prose': 80} |
| degradation|source_random_channel_256 | 0.037 | 0.963 | 1.04 | 0.04 | 2.650 | 256.0 | {'prose': 80} |
| degradation|source_random_channel_128 | 0.025 | 0.975 | 1.02 | 0.03 | -0.107 | 128.0 | {'prose': 80} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.465 | 32.0 | {'prose': 80} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 0.268 | 64.0 | {'prose': 80} |

### glm4

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.000 | 1.000 | 1.00 | 0.00 | 2.363 | 2280.0 | {'prose': 5} |
| degradation|source_top_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 1.658 | 512.0 | {'prose': 5} |
| degradation|source_top_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 1.176 | 256.0 | {'prose': 5} |
| degradation|source_top_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 0.896 | 128.0 | {'prose': 5} |
| degradation|source_top_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 0.546 | 64.0 | {'prose': 5} |
| degradation|source_top_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.357 | 32.0 | {'prose': 5} |
| degradation|source_random_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 0.049 | 512.0 | {'prose': 5} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | -0.045 | 32.0 | {'prose': 5} |
| degradation|source_random_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | -0.056 | 256.0 | {'prose': 5} |
| degradation|source_random_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | -0.057 | 128.0 | {'prose': 5} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | -0.077 | 64.0 | {'prose': 5} |

### qwen3

| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.667 | 0.333 | 2.00 | 1.00 | 16.159 | 2275.0 | {'prose': 3} |
| degradation|source_top_channel_512 | 0.333 | 0.667 | 1.33 | 0.33 | 12.236 | 512.0 | {'prose': 3} |
| degradation|source_top_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | 8.941 | 256.0 | {'prose': 3} |
| degradation|source_top_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | 6.053 | 128.0 | {'prose': 3} |
| degradation|source_top_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | 4.452 | 64.0 | {'continuation': 1, 'prose': 2} |
| degradation|source_top_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 2.690 | 32.0 | {'continuation': 1, 'prose': 2} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.00 | 0.00 | 0.191 | 32.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_512 | 0.000 | 1.000 | 1.00 | 0.00 | 0.032 | 512.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 1.00 | 0.00 | -0.003 | 64.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_256 | 0.000 | 1.000 | 1.00 | 0.00 | -0.058 | 256.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_128 | 0.000 | 1.000 | 1.00 | 0.00 | -0.503 | 128.0 | {'continuation': 2, 'prose': 1} |

