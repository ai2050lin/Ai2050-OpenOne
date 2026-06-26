# Phase 707 Full-Value Phrase Likelihood Identity Audit

- generated: `2026-06-26 19:26:41`

| model | pairs | layers | top_heads | best_restore | change | target_phrase_win | donor_phrase_win | phrase target-donor | best_degrade | drop | target_phrase_win | donor_phrase_win | phrase target-donor |
|---|---:|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| deepseek7b | 72 | [23, 24, 25, 26, 27] | 32 | restore|all_positive_source_channels | 0.475 | 0.438 | 0.037 | -0.385 | degradation|all_positive_source_channels | 0.975 | 0.075 | 0.000 | 2.601 |
| glm4 | 5 | [34, 35, 36, 37, 38, 39] | 32 | restore|all_positive_source_channels | 0.200 | 0.000 | 0.000 | 4.778 | degradation|all_positive_source_channels | 0.000 | 0.400 | 0.000 | 7.209 |
| qwen3 | 3 | [30, 31, 32, 33, 34, 35] | 32 | restore|all_positive_source_channels | 1.000 | 1.000 | 0.000 | 2.566 | degradation|all_positive_source_channels | 0.667 | 0.333 | 0.000 | 2.796 |

## Best Restore

### deepseek7b

| condition | change | target_phrase_win | donor_phrase_win | prose_phrase_win | phrase target-donor | phrase target-prose | target_logp | donor_logp | prose_logp | n_channels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| restore|all_positive_source_channels | 0.475 | 0.438 | 0.037 | 0.525 | -0.385 | -3.191 | -3.800 | -3.414 | -0.609 | 2838.0 |
| restore|source_top_channel_512 | 0.338 | 0.412 | 0.000 | 0.588 | 3.761 | -1.942 | -2.500 | -6.262 | -0.559 | 512.0 |
| restore|source_random_channel_512 | 0.025 | 0.087 | 0.000 | 0.912 | 4.144 | -2.079 | -2.598 | -6.741 | -0.518 | 512.0 |

### glm4

| condition | change | target_phrase_win | donor_phrase_win | prose_phrase_win | phrase target-donor | phrase target-prose | target_logp | donor_logp | prose_logp | n_channels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| restore|all_positive_source_channels | 0.200 | 0.000 | 0.000 | 1.000 | 4.778 | -1.076 | -1.715 | -6.493 | -0.639 | 2280.0 |
| restore|source_top_channel_512 | 0.200 | 0.000 | 0.000 | 1.000 | 5.582 | -0.993 | -1.619 | -7.201 | -0.626 | 512.0 |
| restore|source_random_channel_512 | 0.200 | 0.000 | 0.000 | 1.000 | 5.751 | -1.002 | -1.579 | -7.330 | -0.577 | 512.0 |

### qwen3

| condition | change | target_phrase_win | donor_phrase_win | prose_phrase_win | phrase target-donor | phrase target-prose | target_logp | donor_logp | prose_logp | n_channels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| restore|all_positive_source_channels | 1.000 | 1.000 | 0.000 | 0.000 | 2.566 | 0.384 | -0.021 | -2.587 | -0.405 | 2275.0 |
| restore|source_top_channel_512 | 1.000 | 1.000 | 0.000 | 0.000 | 2.566 | 0.279 | -0.053 | -2.619 | -0.332 | 512.0 |
| restore|source_random_channel_512 | 0.333 | 0.667 | 0.000 | 0.333 | 2.566 | 0.009 | -0.224 | -2.790 | -0.233 | 512.0 |


## Best Degradation

### deepseek7b

| condition | change | target_phrase_win | donor_phrase_win | prose_phrase_win | phrase target-donor | phrase target-prose | target_logp | donor_logp | prose_logp | n_channels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| degradation|all_positive_source_channels | 0.975 | 0.075 | 0.000 | 0.925 | 2.601 | -2.079 | -2.697 | -5.298 | -0.618 | 2838.0 |
| degradation|source_top_channel_512 | 0.625 | 0.287 | 0.000 | 0.713 | 5.743 | -0.487 | -1.126 | -6.869 | -0.639 | 512.0 |
| degradation|source_random_channel_512 | 0.237 | 0.650 | 0.000 | 0.350 | 5.846 | -0.073 | -0.757 | -6.603 | -0.684 | 512.0 |

### glm4

| condition | change | target_phrase_win | donor_phrase_win | prose_phrase_win | phrase target-donor | phrase target-prose | target_logp | donor_logp | prose_logp | n_channels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| degradation|all_positive_source_channels | 0.000 | 0.400 | 0.000 | 0.600 | 7.209 | -0.087 | -0.670 | -7.878 | -0.583 | 2280.0 |
| degradation|source_top_channel_512 | 0.000 | 0.600 | 0.000 | 0.400 | 7.974 | -0.035 | -0.636 | -8.610 | -0.600 | 512.0 |
| degradation|source_random_channel_512 | 0.000 | 0.400 | 0.000 | 0.600 | 7.974 | -0.039 | -0.677 | -8.651 | -0.638 | 512.0 |

### qwen3

| condition | change | target_phrase_win | donor_phrase_win | prose_phrase_win | phrase target-donor | phrase target-prose | target_logp | donor_logp | prose_logp | n_channels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| degradation|all_positive_source_channels | 0.667 | 0.333 | 0.000 | 0.667 | 2.796 | -0.034 | -0.203 | -2.999 | -0.170 | 2275.0 |
| degradation|source_top_channel_512 | 0.000 | 0.667 | 0.000 | 0.333 | 2.796 | 0.069 | -0.129 | -2.924 | -0.198 | 512.0 |
| degradation|source_random_channel_512 | 0.000 | 1.000 | 0.000 | 0.000 | 2.796 | 0.335 | -0.020 | -2.816 | -0.355 | 512.0 |

