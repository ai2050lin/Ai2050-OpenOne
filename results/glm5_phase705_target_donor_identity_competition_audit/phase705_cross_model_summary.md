# Phase 705 Target-vs-Donor Value Identity Competition Audit

- generated: `2026-06-26 18:38:32`

| model | pairs | layers | top_heads | best_restore | change | target_top1 | donor_top1 | target-donor | best_degrade | drop | target_top1 | donor_top1 | target-donor |
|---|---:|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | 32 | restore|all_positive_source_channels | 0.438 | 0.438 | 0.562 | -2.394 | degradation|all_positive_source_channels | 0.975 | 0.025 | 0.025 | 1.121 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | 32 | restore|all_positive_source_channels | 0.200 | 0.200 | 0.000 | 6.439 | degradation|all_positive_source_channels | 0.000 | 1.000 | 0.000 | 8.819 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | 32 | restore|all_positive_source_channels | 1.000 | 1.000 | 1.000 | 0.000 | degradation|all_positive_source_channels | 0.667 | 0.333 | 0.333 | 0.000 |

## Best Restore

### deepseek7b

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 0.438 | 0.438 | 0.562 | 478.54 | 30.07 | -2.394 | -2.267 | 2838.0 | {'continuation': 9, 'prose': 71} |
| restore|source_top_channel_512 | 0.300 | 0.300 | 0.325 | 143.16 | 516.34 | 2.046 | -1.895 | 512.0 | {'continuation': 8, 'prose': 72} |
| restore|source_top_channel_256 | 0.100 | 0.100 | 0.100 | 121.00 | 640.33 | 2.556 | -2.625 | 256.0 | {'continuation': 5, 'prose': 75} |
| restore|source_top_channel_128 | 0.062 | 0.062 | 0.062 | 84.54 | 702.71 | 2.953 | -2.925 | 128.0 | {'continuation': 4, 'prose': 76} |
| restore|source_top_channel_64 | 0.050 | 0.050 | 0.037 | 68.99 | 750.74 | 3.216 | -2.964 | 64.0 | {'continuation': 3, 'prose': 77} |
| restore|source_top_channel_32 | 0.025 | 0.025 | 0.013 | 77.25 | 836.00 | 3.377 | -3.407 | 32.0 | {'continuation': 1, 'prose': 79} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 0.000 | 114.34 | 852.54 | 3.324 | -3.917 | 128.0 | {'continuation': 2, 'prose': 78} |
| restore|source_random_channel_256 | 0.000 | 0.000 | 0.000 | 122.17 | 852.45 | 3.250 | -4.015 | 256.0 | {'prose': 80} |
| restore|source_random_channel_512 | 0.000 | 0.000 | 0.000 | 103.97 | 707.51 | 2.863 | -4.088 | 512.0 | {'prose': 80} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 0.000 | 140.04 | 972.23 | 3.477 | -4.099 | 32.0 | {'prose': 80} |
| restore|source_random_channel_64 | 0.000 | 0.000 | 0.000 | 147.38 | 960.56 | 3.453 | -4.247 | 64.0 | {'prose': 80} |

### glm4

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 0.200 | 0.200 | 0.000 | 1.80 | 739.40 | 6.439 | 2.719 | 2280.0 | {'continuation': 5} |
| restore|source_top_channel_512 | 0.000 | 0.000 | 0.000 | 2.00 | 976.20 | 7.486 | 2.669 | 512.0 | {'continuation': 5} |
| restore|source_top_channel_256 | 0.000 | 0.000 | 0.000 | 2.00 | 1220.80 | 7.798 | 2.544 | 256.0 | {'continuation': 5} |
| restore|source_top_channel_128 | 0.000 | 0.000 | 0.000 | 2.00 | 1302.00 | 7.955 | 2.450 | 128.0 | {'continuation': 5} |
| restore|source_top_channel_64 | 0.000 | 0.000 | 0.000 | 2.00 | 1340.40 | 7.971 | 2.381 | 64.0 | {'continuation': 5} |
| restore|source_top_channel_32 | 0.000 | 0.000 | 0.000 | 2.00 | 1374.20 | 8.025 | 2.325 | 32.0 | {'continuation': 5} |
| restore|source_random_channel_512 | 0.000 | 0.000 | 0.000 | 2.00 | 1212.80 | 7.795 | 2.237 | 512.0 | {'continuation': 5} |
| restore|source_random_channel_64 | 0.000 | 0.000 | 0.000 | 2.00 | 1252.20 | 7.904 | 2.188 | 64.0 | {'continuation': 5} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 0.000 | 2.00 | 1165.60 | 7.898 | 2.237 | 128.0 | {'continuation': 5} |
| restore|source_random_channel_256 | 0.000 | 0.000 | 0.000 | 2.00 | 1179.40 | 7.871 | 2.237 | 256.0 | {'continuation': 5} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 0.000 | 2.00 | 1263.60 | 7.944 | 2.200 | 32.0 | {'continuation': 5} |

### qwen3

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 4.958 | 2275.0 | {'continuation': 3} |
| restore|source_top_channel_512 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 3.708 | 512.0 | {'continuation': 3} |
| restore|source_top_channel_256 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 2.833 | 256.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_128 | 0.667 | 0.667 | 0.667 | 1.33 | 1.33 | 0.000 | 2.208 | 128.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_64 | 0.667 | 0.667 | 0.667 | 1.33 | 1.33 | 0.000 | 1.833 | 64.0 | {'continuation': 2, 'prose': 1} |
| restore|source_top_channel_32 | 0.667 | 0.667 | 0.667 | 1.33 | 1.33 | 0.000 | 1.542 | 32.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_512 | 0.667 | 0.667 | 0.667 | 1.33 | 1.33 | 0.000 | 1.292 | 512.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_256 | 0.667 | 0.667 | 0.667 | 1.33 | 1.33 | 0.000 | 1.083 | 256.0 | {'continuation': 1, 'prose': 2} |
| restore|source_random_channel_64 | 0.333 | 0.333 | 0.333 | 1.67 | 1.67 | 0.000 | 1.000 | 64.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 0.000 | 2.00 | 2.00 | 0.000 | 1.042 | 128.0 | {'continuation': 2, 'prose': 1} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 0.000 | 2.00 | 2.00 | 0.000 | 1.042 | 32.0 | {'continuation': 1, 'prose': 2} |


## Best Degradation

### deepseek7b

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.975 | 0.025 | 0.025 | 225.80 | 353.77 | 1.121 | -4.312 | 2838.0 | {'prose': 80} |
| degradation|source_top_channel_512 | 0.613 | 0.388 | 0.100 | 33.10 | 1389.41 | 4.657 | -1.479 | 512.0 | {'prose': 80} |
| degradation|source_top_channel_256 | 0.525 | 0.475 | 0.125 | 8.60 | 1456.56 | 5.102 | -0.495 | 256.0 | {'prose': 80} |
| degradation|source_top_channel_128 | 0.438 | 0.562 | 0.138 | 3.41 | 1454.55 | 5.337 | 0.159 | 128.0 | {'prose': 80} |
| degradation|source_top_channel_64 | 0.312 | 0.688 | 0.263 | 1.98 | 1364.79 | 5.455 | 0.690 | 64.0 | {'prose': 80} |
| degradation|source_top_channel_32 | 0.188 | 0.812 | 0.388 | 1.27 | 1347.15 | 5.537 | 1.194 | 32.0 | {'prose': 80} |
| degradation|source_random_channel_512 | 0.062 | 0.938 | 0.487 | 1.06 | 1161.76 | 5.332 | 1.583 | 512.0 | {'prose': 80} |
| degradation|source_random_channel_256 | 0.037 | 0.963 | 0.487 | 1.05 | 1254.62 | 5.514 | 1.576 | 256.0 | {'prose': 80} |
| degradation|source_random_channel_128 | 0.025 | 0.975 | 0.487 | 1.02 | 1324.58 | 5.581 | 1.635 | 128.0 | {'prose': 80} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 0.512 | 1.00 | 1266.31 | 5.610 | 1.782 | 32.0 | {'prose': 80} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 0.512 | 1.00 | 1239.66 | 5.616 | 1.853 | 64.0 | {'prose': 80} |

### glm4

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.000 | 1.000 | 0.000 | 1.00 | 730.60 | 8.819 | 3.362 | 2280.0 | {'prose': 5} |
| degradation|source_top_channel_512 | 0.000 | 1.000 | 0.000 | 1.00 | 1028.80 | 9.460 | 3.525 | 512.0 | {'prose': 5} |
| degradation|source_top_channel_256 | 0.000 | 1.000 | 0.000 | 1.00 | 1084.60 | 9.575 | 3.550 | 256.0 | {'prose': 5} |
| degradation|source_top_channel_128 | 0.000 | 1.000 | 0.000 | 1.00 | 1039.60 | 9.566 | 3.625 | 128.0 | {'prose': 5} |
| degradation|source_top_channel_64 | 0.000 | 1.000 | 0.000 | 1.00 | 1018.60 | 9.518 | 3.638 | 64.0 | {'prose': 5} |
| degradation|source_top_channel_32 | 0.000 | 1.000 | 0.000 | 1.00 | 1022.60 | 9.533 | 3.675 | 32.0 | {'prose': 5} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 0.000 | 1.00 | 1018.40 | 9.528 | 3.737 | 32.0 | {'prose': 5} |
| degradation|source_random_channel_256 | 0.000 | 1.000 | 0.000 | 1.00 | 927.00 | 9.509 | 3.775 | 256.0 | {'prose': 5} |
| degradation|source_random_channel_128 | 0.000 | 1.000 | 0.000 | 1.00 | 960.20 | 9.537 | 3.788 | 128.0 | {'prose': 5} |
| degradation|source_random_channel_512 | 0.000 | 1.000 | 0.000 | 1.00 | 864.80 | 9.456 | 3.788 | 512.0 | {'prose': 5} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 0.000 | 1.00 | 967.40 | 9.523 | 3.800 | 64.0 | {'prose': 5} |

### qwen3

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.667 | 0.333 | 0.333 | 2.33 | 2.33 | 0.000 | 0.208 | 2275.0 | {'prose': 3} |
| degradation|source_top_channel_512 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 1.125 | 512.0 | {'prose': 3} |
| degradation|source_top_channel_256 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 1.875 | 256.0 | {'prose': 3} |
| degradation|source_top_channel_128 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 2.750 | 128.0 | {'prose': 3} |
| degradation|source_top_channel_64 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 3.125 | 64.0 | {'continuation': 1, 'prose': 2} |
| degradation|source_top_channel_32 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 3.750 | 32.0 | {'continuation': 1, 'prose': 2} |
| degradation|source_random_channel_512 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 3.917 | 512.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_256 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 4.208 | 256.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_128 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 4.292 | 128.0 | {'continuation': 1, 'prose': 2} |
| degradation|source_random_channel_32 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 4.208 | 32.0 | {'continuation': 2, 'prose': 1} |
| degradation|source_random_channel_64 | 0.000 | 1.000 | 1.000 | 1.00 | 1.00 | 0.000 | 4.208 | 64.0 | {'continuation': 2, 'prose': 1} |

