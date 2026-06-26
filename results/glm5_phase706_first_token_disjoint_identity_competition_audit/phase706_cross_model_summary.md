# Phase 706 First-Token-Disjoint Identity Competition Audit

- generated: `2026-06-26 18:44:00`

| model | pairs | layers | top_heads | best_restore | change | target_top1 | donor_top1 | target-donor | best_degrade | drop | target_top1 | donor_top1 | target-donor |
|---|---:|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | 32 | restore|source_random_channel_64 | 0.000 | 0.000 | 0.000 | 6.907 | degradation|all_positive_source_channels | 1.000 | 0.000 | 0.000 | 1.554 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | 32 | restore|all_positive_source_channels | 0.200 | 0.000 | 0.000 | 4.518 | degradation|all_positive_source_channels | 0.000 | 0.000 | 0.000 | 7.098 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | 32 |  | 0.000 | 0.000 | 0.000 | 0.000 |  | 0.000 | 0.000 | 0.000 | 0.000 |

## Best Restore

### deepseek7b

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| restore|source_random_channel_64 | 0.000 | 0.000 | 0.000 | 4243.66 | 43174.97 | 6.907 | -7.732 | 64.0 | {'prose': 38} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 0.000 | 4267.84 | 43175.34 | 6.857 | -7.766 | 32.0 | {'prose': 38} |
| restore|source_top_channel_32 | 0.000 | 0.000 | 0.000 | 4219.03 | 43724.66 | 6.824 | -7.817 | 32.0 | {'prose': 38} |
| restore|source_top_channel_64 | 0.000 | 0.000 | 0.000 | 4209.24 | 42115.37 | 6.537 | -7.915 | 64.0 | {'prose': 38} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 0.000 | 4282.68 | 39843.13 | 6.557 | -7.877 | 128.0 | {'prose': 38} |
| restore|source_top_channel_128 | 0.000 | 0.000 | 0.000 | 4296.50 | 37966.39 | 6.013 | -8.127 | 128.0 | {'prose': 38} |
| restore|source_random_channel_256 | 0.000 | 0.000 | 0.000 | 4226.50 | 35965.24 | 6.308 | -7.924 | 256.0 | {'prose': 38} |
| restore|source_top_channel_256 | 0.000 | 0.000 | 0.000 | 4690.92 | 31723.97 | 5.198 | -8.435 | 256.0 | {'prose': 38} |
| restore|source_top_channel_512 | 0.000 | 0.000 | 0.000 | 4856.16 | 19870.08 | 3.948 | -8.565 | 512.0 | {'prose': 38} |
| restore|source_random_channel_512 | 0.000 | 0.000 | 0.000 | 4527.03 | 28095.58 | 5.305 | -8.215 | 512.0 | {'prose': 38} |
| restore|all_positive_source_channels | 0.000 | 0.000 | 0.000 | 25203.84 | 1881.16 | -4.499 | -11.656 | 2838.0 | {'continuation': 2, 'prose': 36} |

### glm4

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| restore|all_positive_source_channels | 0.200 | 0.000 | 0.000 | 55.40 | 2500.20 | 4.518 | -2.831 | 2280.0 | {'continuation': 5} |
| restore|source_random_channel_256 | 0.200 | 0.000 | 0.000 | 53.80 | 16646.20 | 6.121 | -3.269 | 256.0 | {'continuation': 5} |
| restore|source_top_channel_512 | 0.000 | 0.000 | 0.000 | 51.80 | 9394.20 | 5.744 | -2.856 | 512.0 | {'continuation': 5} |
| restore|source_top_channel_256 | 0.000 | 0.000 | 0.000 | 51.60 | 13342.00 | 6.014 | -2.969 | 256.0 | {'continuation': 5} |
| restore|source_top_channel_128 | 0.000 | 0.000 | 0.000 | 54.60 | 16613.00 | 6.147 | -3.119 | 128.0 | {'continuation': 5} |
| restore|source_top_channel_64 | 0.000 | 0.000 | 0.000 | 53.80 | 16317.40 | 6.188 | -3.156 | 64.0 | {'continuation': 5} |
| restore|source_top_channel_32 | 0.000 | 0.000 | 0.000 | 52.40 | 17601.20 | 6.252 | -3.194 | 32.0 | {'continuation': 5} |
| restore|source_random_channel_512 | 0.000 | 0.000 | 0.000 | 51.20 | 15217.20 | 6.104 | -3.306 | 512.0 | {'continuation': 5} |
| restore|source_random_channel_64 | 0.000 | 0.000 | 0.000 | 52.20 | 18144.60 | 6.229 | -3.269 | 64.0 | {'continuation': 5} |
| restore|source_random_channel_128 | 0.000 | 0.000 | 0.000 | 53.20 | 16509.80 | 6.180 | -3.306 | 128.0 | {'continuation': 5} |
| restore|source_random_channel_32 | 0.000 | 0.000 | 0.000 | 52.20 | 17668.20 | 6.197 | -3.312 | 32.0 | {'continuation': 5} |

### qwen3

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|


## Best Degradation

### deepseek7b

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 1.000 | 0.000 | 0.000 | 6979.84 | 11900.61 | 1.554 | -9.136 | 2838.0 | {'prose': 38} |
| degradation|source_top_channel_512 | 0.395 | 0.000 | 0.000 | 1368.11 | 31239.26 | 7.814 | -5.884 | 512.0 | {'prose': 38} |
| degradation|source_top_channel_256 | 0.263 | 0.000 | 0.000 | 346.11 | 34414.97 | 8.649 | -5.220 | 256.0 | {'prose': 38} |
| degradation|source_top_channel_128 | 0.158 | 0.000 | 0.000 | 202.71 | 38046.87 | 9.244 | -4.794 | 128.0 | {'prose': 38} |
| degradation|source_random_channel_512 | 0.132 | 0.000 | 0.000 | 223.55 | 33046.82 | 8.944 | -4.599 | 512.0 | {'prose': 38} |
| degradation|source_top_channel_64 | 0.105 | 0.000 | 0.000 | 175.79 | 39397.26 | 9.502 | -4.562 | 64.0 | {'prose': 38} |
| degradation|source_top_channel_32 | 0.105 | 0.000 | 0.000 | 187.32 | 39879.37 | 9.646 | -4.329 | 32.0 | {'prose': 38} |
| degradation|source_random_channel_256 | 0.026 | 0.000 | 0.000 | 183.82 | 36455.05 | 9.369 | -4.341 | 256.0 | {'prose': 38} |
| degradation|source_random_channel_64 | 0.026 | 0.000 | 0.000 | 233.16 | 40289.71 | 9.690 | -4.260 | 64.0 | {'prose': 38} |
| degradation|source_random_channel_128 | 0.026 | 0.000 | 0.000 | 262.03 | 39093.79 | 9.576 | -4.294 | 128.0 | {'prose': 38} |
| degradation|source_random_channel_32 | 0.000 | 0.000 | 0.000 | 208.26 | 40262.58 | 9.702 | -4.234 | 32.0 | {'prose': 38} |

### glm4

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| degradation|all_positive_source_channels | 0.000 | 0.000 | 0.000 | 10.80 | 5612.20 | 7.098 | -1.344 | 2280.0 | {'prose': 5} |
| degradation|source_top_channel_512 | 0.000 | 0.000 | 0.000 | 9.60 | 11904.80 | 7.819 | -1.169 | 512.0 | {'prose': 5} |
| degradation|source_top_channel_256 | 0.000 | 0.000 | 0.000 | 9.80 | 14646.00 | 7.940 | -1.119 | 256.0 | {'prose': 5} |
| degradation|source_top_channel_128 | 0.000 | 0.000 | 0.000 | 9.80 | 16189.20 | 7.950 | -1.075 | 128.0 | {'prose': 5} |
| degradation|source_top_channel_64 | 0.000 | 0.000 | 0.000 | 9.60 | 15851.80 | 7.918 | -1.006 | 64.0 | {'prose': 5} |
| degradation|source_top_channel_32 | 0.000 | 0.000 | 0.000 | 9.60 | 17151.80 | 7.917 | -1.019 | 32.0 | {'prose': 5} |
| degradation|source_random_channel_512 | 0.000 | 0.000 | 0.000 | 9.60 | 15680.40 | 7.892 | -0.919 | 512.0 | {'prose': 5} |
| degradation|source_random_channel_64 | 0.000 | 0.000 | 0.000 | 9.80 | 16574.60 | 7.939 | -0.925 | 64.0 | {'prose': 5} |
| degradation|source_random_channel_32 | 0.000 | 0.000 | 0.000 | 9.20 | 15869.40 | 7.915 | -0.894 | 32.0 | {'prose': 5} |
| degradation|source_random_channel_256 | 0.000 | 0.000 | 0.000 | 9.60 | 16235.20 | 7.953 | -0.925 | 256.0 | {'prose': 5} |
| degradation|source_random_channel_128 | 0.000 | 0.000 | 0.000 | 9.20 | 15275.20 | 7.895 | -0.900 | 128.0 | {'prose': 5} |

### qwen3

| condition | change | target_top1 | donor_top1 | target_rank | donor_rank | target-donor | target-prose | n_channels | best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

