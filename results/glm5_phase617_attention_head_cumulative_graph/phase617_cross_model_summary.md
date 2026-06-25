# Phase 617 Cross Model Summary

Layer/head-slot decomposition of the multi-layer attention cumulative path.

## qwen3

rows=9, target_seen=9, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 119}, layers=[25, 26, 27, 28, 29], heads={'25': 32, '26': 32, '27': 32, '28': 32, '29': 32}, specs=84, time_min=2.90

### best

| rank | name | kind | random | ops | slots | switch | margin | correct_delta | wrong_delta |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `all_heads_midlate_L27_L29` | all_heads_midlate | False | 3 | 96 | 6/9 | +1.821 | +1.040 | -0.781 |
| 2 | `all_heads_L29` | all_heads_layer | False | 1 | 32 | 6/9 | +1.640 | +0.980 | -0.660 |
| 3 | `known_top8_midlate_L27_L29` | known_top_midlate | False | 3 | 24 | 4/9 | +1.238 | +0.758 | -0.479 |
| 4 | `L29_H11` | single_known_head | False | 1 | 1 | 4/9 | +1.223 | +0.777 | -0.446 |
| 5 | `known_top1_midlate_L27_L29` | known_top_midlate | False | 3 | 3 | 4/9 | +1.209 | +0.770 | -0.440 |
| 6 | `known_top1_all_layers` | known_top_all_layers | False | 5 | 5 | 4/9 | +1.182 | +0.763 | -0.419 |
| 7 | `known_top12_midlate_L27_L29` | known_top_midlate | False | 3 | 36 | 4/9 | +1.098 | +0.688 | -0.411 |
| 8 | `known_top4_midlate_L27_L29` | known_top_midlate | False | 3 | 12 | 3/9 | +0.932 | +0.605 | -0.327 |
| 9 | `known_top6_midlate_L27_L29` | known_top_midlate | False | 3 | 18 | 3/9 | +0.862 | +0.559 | -0.303 |
| 10 | `known_top2_midlate_L27_L29` | known_top_midlate | False | 3 | 6 | 3/9 | +0.570 | +0.367 | -0.203 |
| 11 | `all_heads_all_layers` | all_heads | False | 5 | 160 | 3/9 | +0.529 | +0.206 | -0.323 |
| 12 | `known_top8_all_layers` | known_top_all_layers | False | 5 | 40 | 2/9 | +0.626 | +0.356 | -0.270 |
| 13 | `known_top2_all_layers` | known_top_all_layers | False | 5 | 10 | 2/9 | +0.529 | +0.341 | -0.187 |
| 14 | `L28_H0` | single_known_head | False | 1 | 1 | 2/9 | +0.473 | +0.313 | -0.160 |
| 15 | `L28_coverage_H0` | single_coverage_head | False | 1 | 1 | 2/9 | +0.473 | +0.313 | -0.160 |
| 16 | `known_top12_all_layers` | known_top_all_layers | False | 5 | 60 | 2/9 | +0.432 | +0.197 | -0.234 |
| 17 | `all_heads_L26` | all_heads_layer | False | 1 | 32 | 2/9 | +0.390 | +0.286 | -0.104 |
| 18 | `L29_H6` | single_known_head | False | 1 | 1 | 2/9 | +0.320 | +0.239 | -0.081 |
| 19 | `known_top4_all_layers` | known_top_all_layers | False | 5 | 20 | 2/9 | +0.292 | +0.137 | -0.155 |
| 20 | `all_heads_L28` | all_heads_layer | False | 1 | 32 | 2/9 | +0.181 | +0.094 | -0.087 |
| 21 | `all_heads_midlate_L27_L29` | all_heads_midlate | True | 3 | 96 | 2/9 | +0.175 | +0.146 | -0.029 |
| 22 | `known_top6_midlate_L27_L29` | known_top_midlate | True | 3 | 18 | 2/9 | +0.051 | +0.014 | -0.037 |
| 23 | `known_top1_midlate_L27_L29` | known_top_midlate | True | 3 | 3 | 2/9 | +0.039 | -0.003 | -0.042 |
| 24 | `L27_H0` | single_known_head | True | 1 | 1 | 2/9 | +0.030 | +0.020 | -0.010 |
| 25 | `known_top4_midlate_L27_L29` | known_top_midlate | True | 3 | 12 | 2/9 | +0.022 | -0.024 | -0.046 |
| 26 | `known_top2_midlate_L27_L29` | known_top_midlate | True | 3 | 6 | 2/9 | +0.003 | -0.029 | -0.032 |
| 27 | `known_top6_all_layers` | known_top_all_layers | False | 5 | 30 | 1/9 | +0.278 | +0.117 | -0.162 |
| 28 | `known_top6_all_layers` | known_top_all_layers | True | 5 | 30 | 1/9 | +0.126 | +0.081 | -0.045 |
| 29 | `L28_coverage_H16` | single_coverage_head | False | 1 | 1 | 1/9 | +0.125 | +0.092 | -0.033 |
| 30 | `known_top8_all_layers` | known_top_all_layers | True | 5 | 40 | 1/9 | +0.108 | +0.061 | -0.047 |
| 31 | `all_heads_L29` | all_heads_layer | True | 1 | 32 | 1/9 | +0.107 | +0.070 | -0.036 |
| 32 | `known_top4_all_layers` | known_top_all_layers | True | 5 | 20 | 1/9 | +0.083 | +0.054 | -0.029 |

### all_heads_refs

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `all_heads_midlate_L27_L29` | all_heads_midlate | 3 | 96 | 6/9 | +1.821 | +1.040 | -0.781 |
| `all_heads_L29` | all_heads_layer | 1 | 32 | 6/9 | +1.640 | +0.980 | -0.660 |
| `all_heads_all_layers` | all_heads | 5 | 160 | 3/9 | +0.529 | +0.206 | -0.323 |
| `all_heads_L26` | all_heads_layer | 1 | 32 | 2/9 | +0.390 | +0.286 | -0.104 |
| `all_heads_L28` | all_heads_layer | 1 | 32 | 2/9 | +0.181 | +0.094 | -0.087 |
| `all_heads_L27` | all_heads_layer | 1 | 32 | 0/9 | -0.070 | -0.062 | +0.008 |
| `all_heads_L25` | all_heads_layer | 1 | 32 | 0/9 | -1.391 | -1.251 | +0.140 |

### known_top_cumulative

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `known_top8_midlate_L27_L29` | known_top_midlate | 3 | 24 | 4/9 | +1.238 | +0.758 | -0.479 |
| `known_top1_midlate_L27_L29` | known_top_midlate | 3 | 3 | 4/9 | +1.209 | +0.770 | -0.440 |
| `known_top1_all_layers` | known_top_all_layers | 5 | 5 | 4/9 | +1.182 | +0.763 | -0.419 |
| `known_top12_midlate_L27_L29` | known_top_midlate | 3 | 36 | 4/9 | +1.098 | +0.688 | -0.411 |
| `known_top4_midlate_L27_L29` | known_top_midlate | 3 | 12 | 3/9 | +0.932 | +0.605 | -0.327 |
| `known_top6_midlate_L27_L29` | known_top_midlate | 3 | 18 | 3/9 | +0.862 | +0.559 | -0.303 |
| `known_top2_midlate_L27_L29` | known_top_midlate | 3 | 6 | 3/9 | +0.570 | +0.367 | -0.203 |
| `known_top8_all_layers` | known_top_all_layers | 5 | 40 | 2/9 | +0.626 | +0.356 | -0.270 |
| `known_top2_all_layers` | known_top_all_layers | 5 | 10 | 2/9 | +0.529 | +0.341 | -0.187 |
| `known_top12_all_layers` | known_top_all_layers | 5 | 60 | 2/9 | +0.432 | +0.197 | -0.234 |
| `known_top4_all_layers` | known_top_all_layers | 5 | 20 | 2/9 | +0.292 | +0.137 | -0.155 |
| `known_top6_all_layers` | known_top_all_layers | 5 | 30 | 1/9 | +0.278 | +0.117 | -0.162 |

### single_heads

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `L29_H11` | single_known_head | 1 | 1 | 4/9 | +1.223 | +0.777 | -0.446 |
| `L28_H0` | single_known_head | 1 | 1 | 2/9 | +0.473 | +0.313 | -0.160 |
| `L28_coverage_H0` | single_coverage_head | 1 | 1 | 2/9 | +0.473 | +0.313 | -0.160 |
| `L29_H6` | single_known_head | 1 | 1 | 2/9 | +0.320 | +0.239 | -0.081 |
| `L28_coverage_H16` | single_coverage_head | 1 | 1 | 1/9 | +0.125 | +0.092 | -0.033 |
| `L25_H6` | single_known_head | 1 | 1 | 1/9 | +0.069 | +0.046 | -0.024 |
| `L26_H6` | single_known_head | 1 | 1 | 1/9 | +0.042 | +0.021 | -0.021 |
| `L25_coverage_H8` | single_coverage_head | 1 | 1 | 1/9 | +0.028 | +0.006 | -0.021 |
| `L29_coverage_H24` | single_coverage_head | 1 | 1 | 1/9 | +0.028 | +0.017 | -0.011 |
| `L26_H11` | single_known_head | 1 | 1 | 1/9 | +0.028 | +0.016 | -0.012 |
| `L27_H5` | single_known_head | 1 | 1 | 1/9 | +0.028 | +0.016 | -0.012 |
| `L29_coverage_H16` | single_coverage_head | 1 | 1 | 1/9 | +0.028 | +0.017 | -0.011 |
| `L26_coverage_H24` | single_coverage_head | 1 | 1 | 1/9 | +0.028 | +0.014 | -0.014 |
| `L27_coverage_H31` | single_coverage_head | 1 | 1 | 1/9 | +0.014 | -0.001 | -0.015 |
| `L27_coverage_H8` | single_coverage_head | 1 | 1 | 1/9 | +0.014 | -0.000 | -0.014 |
| `L26_H14` | single_known_head | 1 | 1 | 1/9 | +0.014 | +0.006 | -0.008 |
| `L29_H14` | single_known_head | 1 | 1 | 1/9 | +0.014 | +0.001 | -0.013 |
| `L27_H0` | single_known_head | 1 | 1 | 1/9 | +0.014 | +0.000 | -0.013 |
| `L27_coverage_H0` | single_coverage_head | 1 | 1 | 1/9 | +0.014 | +0.000 | -0.013 |
| `L26_H2` | single_known_head | 1 | 1 | 1/9 | -0.000 | -0.001 | -0.001 |
| `L27_coverage_H24` | single_coverage_head | 1 | 1 | 1/9 | -0.000 | -0.010 | -0.010 |
| `L27_H14` | single_known_head | 1 | 1 | 1/9 | -0.000 | -0.004 | -0.004 |
| `L26_coverage_H16` | single_coverage_head | 1 | 1 | 1/9 | -0.000 | -0.014 | -0.014 |
| `L25_coverage_H16` | single_coverage_head | 1 | 1 | 1/9 | -0.014 | -0.020 | -0.006 |
| `L28_coverage_H31` | single_coverage_head | 1 | 1 | 1/9 | -0.028 | -0.033 | -0.005 |
| `L27_H11` | single_known_head | 1 | 1 | 1/9 | -0.042 | -0.035 | +0.006 |
| `L29_H5` | single_known_head | 1 | 1 | 0/9 | +0.056 | +0.049 | -0.007 |
| `L26_H5` | single_known_head | 1 | 1 | 0/9 | +0.028 | +0.011 | -0.017 |
| `L28_coverage_H24` | single_coverage_head | 1 | 1 | 0/9 | +0.028 | +0.021 | -0.007 |
| `L26_coverage_H31` | single_coverage_head | 1 | 1 | 0/9 | +0.014 | +0.003 | -0.011 |
| `L29_H23` | single_known_head | 1 | 1 | 0/9 | +0.014 | +0.005 | -0.008 |
| `L29_H2` | single_known_head | 1 | 1 | 0/9 | +0.014 | +0.006 | -0.008 |
| `L28_H6` | single_known_head | 1 | 1 | 0/9 | +0.014 | +0.002 | -0.012 |
| `L27_H23` | single_known_head | 1 | 1 | 0/9 | +0.014 | +0.006 | -0.007 |
| `L25_H2` | single_known_head | 1 | 1 | 0/9 | -0.000 | -0.003 | -0.003 |
| `L29_coverage_H31` | single_coverage_head | 1 | 1 | 0/9 | -0.000 | +0.002 | +0.002 |

### random_controls

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `all_heads_midlate_L27_L29` | all_heads_midlate | 3 | 96 | 2/9 | +0.175 | +0.146 | -0.029 |
| `known_top6_midlate_L27_L29` | known_top_midlate | 3 | 18 | 2/9 | +0.051 | +0.014 | -0.037 |
| `known_top1_midlate_L27_L29` | known_top_midlate | 3 | 3 | 2/9 | +0.039 | -0.003 | -0.042 |
| `known_top4_midlate_L27_L29` | known_top_midlate | 3 | 12 | 2/9 | +0.022 | -0.024 | -0.046 |
| `known_top2_midlate_L27_L29` | known_top_midlate | 3 | 6 | 2/9 | +0.003 | -0.029 | -0.032 |
| `known_top6_all_layers` | known_top_all_layers | 5 | 30 | 1/9 | +0.126 | +0.081 | -0.045 |
| `known_top8_all_layers` | known_top_all_layers | 5 | 40 | 1/9 | +0.108 | +0.061 | -0.047 |
| `all_heads_L29` | all_heads_layer | 1 | 32 | 1/9 | +0.107 | +0.070 | -0.036 |
| `known_top4_all_layers` | known_top_all_layers | 5 | 20 | 1/9 | +0.083 | +0.054 | -0.029 |
| `known_top8_midlate_L27_L29` | known_top_midlate | 3 | 24 | 1/9 | +0.008 | -0.003 | -0.011 |
| `all_heads_L25` | all_heads_layer | 1 | 32 | 1/9 | -0.022 | -0.030 | -0.008 |
| `all_heads_L26` | all_heads_layer | 1 | 32 | 1/9 | -0.026 | -0.010 | +0.016 |
| `all_heads_L28` | all_heads_layer | 1 | 32 | 1/9 | -0.036 | -0.032 | +0.005 |
| `all_heads_all_layers` | all_heads | 5 | 160 | 1/9 | -0.052 | -0.107 | -0.054 |
| `all_heads_L27` | all_heads_layer | 1 | 32 | 0/9 | +0.032 | +0.040 | +0.008 |
| `known_top2_all_layers` | known_top_all_layers | 5 | 10 | 0/9 | -0.057 | -0.053 | +0.004 |
| `known_top1_all_layers` | known_top_all_layers | 5 | 5 | 0/9 | -0.067 | -0.064 | +0.003 |
| `known_top12_midlate_L27_L29` | known_top_midlate | 3 | 36 | 0/9 | -0.093 | -0.090 | +0.003 |
| `known_top12_all_layers` | known_top_all_layers | 5 | 60 | 0/9 | -0.166 | -0.107 | +0.059 |

## glm4

rows=12, target_seen=12, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 116}, layers=[30, 31, 32, 33, 34], heads={'30': 32, '31': 32, '32': 32, '33': 32, '34': 32}, specs=84, time_min=5.73

### best

| rank | name | kind | random | ops | slots | switch | margin | correct_delta | wrong_delta |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `L32_coverage_H16` | single_coverage_head | False | 1 | 1 | 2/12 | +0.036 | +0.017 | -0.020 |
| 2 | `known_top4_midlate_L32_L34` | known_top_midlate | True | 3 | 12 | 2/12 | +0.021 | +0.026 | +0.004 |
| 3 | `all_heads_L32` | all_heads_layer | False | 1 | 32 | 1/12 | +0.094 | +0.055 | -0.038 |
| 4 | `all_heads_midlate_L32_L34` | all_heads_midlate | False | 3 | 96 | 1/12 | +0.089 | +0.035 | -0.054 |
| 5 | `all_heads_L33` | all_heads_layer | False | 1 | 32 | 1/12 | +0.042 | +0.023 | -0.019 |
| 6 | `L33_coverage_H24` | single_coverage_head | False | 1 | 1 | 1/12 | +0.036 | +0.016 | -0.020 |
| 7 | `L34_H8` | single_known_head | False | 1 | 1 | 1/12 | +0.031 | +0.018 | -0.013 |
| 8 | `L34_coverage_H8` | single_coverage_head | False | 1 | 1 | 1/12 | +0.031 | +0.018 | -0.013 |
| 9 | `all_heads_L32` | all_heads_layer | True | 1 | 32 | 1/12 | +0.031 | +0.033 | +0.002 |
| 10 | `known_top2_all_layers` | known_top_all_layers | True | 5 | 10 | 1/12 | +0.022 | +0.013 | -0.009 |
| 11 | `known_top6_midlate_L32_L34` | known_top_midlate | True | 3 | 18 | 1/12 | +0.017 | +0.020 | +0.003 |
| 12 | `L33_H2` | single_known_head | False | 1 | 1 | 1/12 | +0.016 | +0.011 | -0.004 |
| 13 | `L34_H4` | single_known_head | False | 1 | 1 | 1/12 | +0.016 | +0.007 | -0.008 |
| 14 | `L33_H7` | single_known_head | False | 1 | 1 | 1/12 | +0.010 | +0.005 | -0.006 |
| 15 | `L31_coverage_H24` | single_coverage_head | False | 1 | 1 | 1/12 | +0.010 | +0.006 | -0.005 |
| 16 | `L32_coverage_H0` | single_coverage_head | True | 1 | 1 | 1/12 | +0.006 | -0.000 | -0.006 |
| 17 | `L32_H26` | single_known_head | False | 1 | 1 | 1/12 | +0.005 | +0.002 | -0.003 |
| 18 | `all_heads_L30` | all_heads_layer | False | 1 | 32 | 1/12 | +0.005 | +0.003 | -0.002 |
| 19 | `L30_H6` | single_known_head | False | 1 | 1 | 1/12 | +0.005 | +0.008 | +0.002 |
| 20 | `L30_H2` | single_known_head | True | 1 | 1 | 1/12 | +0.003 | -0.001 | -0.004 |
| 21 | `L34_coverage_H24` | single_coverage_head | True | 1 | 1 | 1/12 | +0.002 | +0.002 | +0.000 |
| 22 | `all_heads_L33` | all_heads_layer | True | 1 | 32 | 1/12 | +0.001 | -0.004 | -0.006 |
| 23 | `known_top1_midlate_L32_L34` | known_top_midlate | False | 3 | 3 | 1/12 | +0.000 | -0.003 | -0.003 |
| 24 | `L30_coverage_H16` | single_coverage_head | False | 1 | 1 | 1/12 | -0.000 | -0.001 | -0.001 |
| 25 | `known_top2_all_layers` | known_top_all_layers | False | 5 | 10 | 1/12 | -0.000 | -0.004 | -0.004 |
| 26 | `L34_H28` | single_known_head | True | 1 | 1 | 1/12 | -0.001 | +0.000 | +0.002 |
| 27 | `L34_H4` | single_known_head | True | 1 | 1 | 1/12 | -0.002 | -0.002 | -0.000 |
| 28 | `L32_H6` | single_known_head | True | 1 | 1 | 1/12 | -0.003 | -0.003 | +0.000 |
| 29 | `L30_H2` | single_known_head | False | 1 | 1 | 1/12 | -0.005 | -0.000 | +0.005 |
| 30 | `L30_coverage_H31` | single_coverage_head | False | 1 | 1 | 1/12 | -0.005 | -0.002 | +0.003 |
| 31 | `known_top1_all_layers` | known_top_all_layers | False | 5 | 5 | 1/12 | -0.010 | -0.009 | +0.002 |
| 32 | `L31_H28` | single_known_head | False | 1 | 1 | 1/12 | -0.021 | -0.014 | +0.007 |

### all_heads_refs

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `all_heads_L32` | all_heads_layer | 1 | 32 | 1/12 | +0.094 | +0.055 | -0.038 |
| `all_heads_midlate_L32_L34` | all_heads_midlate | 3 | 96 | 1/12 | +0.089 | +0.035 | -0.054 |
| `all_heads_L33` | all_heads_layer | 1 | 32 | 1/12 | +0.042 | +0.023 | -0.019 |
| `all_heads_L30` | all_heads_layer | 1 | 32 | 1/12 | +0.005 | +0.003 | -0.002 |
| `all_heads_all_layers` | all_heads | 5 | 160 | 1/12 | -0.021 | -0.038 | -0.017 |
| `all_heads_L34` | all_heads_layer | 1 | 32 | 0/12 | -0.042 | -0.021 | +0.021 |
| `all_heads_L31` | all_heads_layer | 1 | 32 | 0/12 | -0.083 | -0.064 | +0.020 |

### known_top_cumulative

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `known_top1_midlate_L32_L34` | known_top_midlate | 3 | 3 | 1/12 | +0.000 | -0.003 | -0.003 |
| `known_top2_all_layers` | known_top_all_layers | 5 | 10 | 1/12 | -0.000 | -0.004 | -0.004 |
| `known_top1_all_layers` | known_top_all_layers | 5 | 5 | 1/12 | -0.010 | -0.009 | +0.002 |
| `known_top6_all_layers` | known_top_all_layers | 5 | 30 | 1/12 | -0.036 | -0.047 | -0.010 |
| `known_top12_all_layers` | known_top_all_layers | 5 | 60 | 1/12 | -0.094 | -0.077 | +0.017 |
| `known_top8_midlate_L32_L34` | known_top_midlate | 3 | 24 | 1/12 | -0.104 | -0.080 | +0.024 |
| `known_top12_midlate_L32_L34` | known_top_midlate | 3 | 36 | 1/12 | -0.115 | -0.081 | +0.034 |
| `known_top2_midlate_L32_L34` | known_top_midlate | 3 | 6 | 0/12 | -0.005 | -0.009 | -0.004 |
| `known_top6_midlate_L32_L34` | known_top_midlate | 3 | 18 | 0/12 | -0.016 | -0.032 | -0.016 |
| `known_top4_all_layers` | known_top_all_layers | 5 | 20 | 0/12 | -0.042 | -0.048 | -0.006 |
| `known_top4_midlate_L32_L34` | known_top_midlate | 3 | 12 | 0/12 | -0.047 | -0.044 | +0.003 |
| `known_top8_all_layers` | known_top_all_layers | 5 | 40 | 0/12 | -0.130 | -0.098 | +0.033 |

### single_heads

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `L32_coverage_H16` | single_coverage_head | 1 | 1 | 2/12 | +0.036 | +0.017 | -0.020 |
| `L33_coverage_H24` | single_coverage_head | 1 | 1 | 1/12 | +0.036 | +0.016 | -0.020 |
| `L34_H8` | single_known_head | 1 | 1 | 1/12 | +0.031 | +0.018 | -0.013 |
| `L34_coverage_H8` | single_coverage_head | 1 | 1 | 1/12 | +0.031 | +0.018 | -0.013 |
| `L33_H2` | single_known_head | 1 | 1 | 1/12 | +0.016 | +0.011 | -0.004 |
| `L34_H4` | single_known_head | 1 | 1 | 1/12 | +0.016 | +0.007 | -0.008 |
| `L33_H7` | single_known_head | 1 | 1 | 1/12 | +0.010 | +0.005 | -0.006 |
| `L31_coverage_H24` | single_coverage_head | 1 | 1 | 1/12 | +0.010 | +0.006 | -0.005 |
| `L32_H26` | single_known_head | 1 | 1 | 1/12 | +0.005 | +0.002 | -0.003 |
| `L30_H6` | single_known_head | 1 | 1 | 1/12 | +0.005 | +0.008 | +0.002 |
| `L30_coverage_H16` | single_coverage_head | 1 | 1 | 1/12 | -0.000 | -0.001 | -0.001 |
| `L30_H2` | single_known_head | 1 | 1 | 1/12 | -0.005 | -0.000 | +0.005 |
| `L30_coverage_H31` | single_coverage_head | 1 | 1 | 1/12 | -0.005 | -0.002 | +0.003 |
| `L31_H28` | single_known_head | 1 | 1 | 1/12 | -0.021 | -0.014 | +0.007 |
| `L31_coverage_H16` | single_coverage_head | 1 | 1 | 1/12 | -0.021 | -0.016 | +0.004 |
| `L32_coverage_H24` | single_coverage_head | 1 | 1 | 1/12 | -0.021 | -0.009 | +0.012 |
| `L31_H26` | single_known_head | 1 | 1 | 0/12 | +0.016 | +0.007 | -0.008 |
| `L34_H12` | single_known_head | 1 | 1 | 0/12 | +0.016 | +0.010 | -0.005 |
| `L30_coverage_H0` | single_coverage_head | 1 | 1 | 0/12 | +0.010 | +0.008 | -0.003 |
| `L33_coverage_H16` | single_coverage_head | 1 | 1 | 0/12 | +0.005 | +0.010 | +0.005 |
| `L34_H6` | single_known_head | 1 | 1 | 0/12 | +0.005 | +0.004 | -0.001 |
| `L33_H4` | single_known_head | 1 | 1 | 0/12 | -0.000 | +0.002 | +0.002 |
| `L34_coverage_H31` | single_coverage_head | 1 | 1 | 0/12 | -0.000 | +0.001 | +0.001 |
| `L31_coverage_H0` | single_coverage_head | 1 | 1 | 0/12 | -0.000 | -0.001 | -0.001 |
| `L33_H8` | single_known_head | 1 | 1 | 0/12 | -0.005 | -0.001 | +0.004 |
| `L33_coverage_H8` | single_coverage_head | 1 | 1 | 0/12 | -0.005 | -0.001 | +0.004 |
| `L31_coverage_H31` | single_coverage_head | 1 | 1 | 0/12 | -0.005 | -0.002 | +0.003 |
| `L34_coverage_H24` | single_coverage_head | 1 | 1 | 0/12 | -0.005 | -0.000 | +0.005 |
| `L30_H7` | single_known_head | 1 | 1 | 0/12 | -0.005 | -0.006 | -0.001 |
| `L30_H26` | single_known_head | 1 | 1 | 0/12 | -0.005 | -0.000 | +0.005 |
| `L33_H6` | single_known_head | 1 | 1 | 0/12 | -0.005 | +0.004 | +0.009 |
| `L34_coverage_H16` | single_coverage_head | 1 | 1 | 0/12 | -0.005 | -0.000 | +0.005 |
| `L30_H4` | single_known_head | 1 | 1 | 0/12 | -0.005 | -0.003 | +0.002 |
| `L32_H7` | single_known_head | 1 | 1 | 0/12 | -0.005 | -0.003 | +0.002 |
| `L34_H7` | single_known_head | 1 | 1 | 0/12 | -0.005 | -0.000 | +0.005 |
| `L30_H8` | single_known_head | 1 | 1 | 0/12 | -0.010 | -0.003 | +0.007 |

### random_controls

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `known_top4_midlate_L32_L34` | known_top_midlate | 3 | 12 | 2/12 | +0.021 | +0.026 | +0.004 |
| `all_heads_L32` | all_heads_layer | 1 | 32 | 1/12 | +0.031 | +0.033 | +0.002 |
| `known_top2_all_layers` | known_top_all_layers | 5 | 10 | 1/12 | +0.022 | +0.013 | -0.009 |
| `known_top6_midlate_L32_L34` | known_top_midlate | 3 | 18 | 1/12 | +0.017 | +0.020 | +0.003 |
| `all_heads_L33` | all_heads_layer | 1 | 32 | 1/12 | +0.001 | -0.004 | -0.006 |
| `all_heads_all_layers` | all_heads | 5 | 160 | 0/12 | +0.016 | -0.029 | -0.045 |
| `known_top2_midlate_L32_L34` | known_top_midlate | 3 | 6 | 0/12 | +0.006 | +0.008 | +0.002 |
| `known_top8_midlate_L32_L34` | known_top_midlate | 3 | 24 | 0/12 | +0.002 | +0.013 | +0.011 |
| `all_heads_L31` | all_heads_layer | 1 | 32 | 0/12 | -0.000 | -0.008 | -0.008 |
| `known_top4_all_layers` | known_top_all_layers | 5 | 20 | 0/12 | -0.001 | +0.010 | +0.010 |
| `known_top6_all_layers` | known_top_all_layers | 5 | 30 | 0/12 | -0.001 | +0.006 | +0.007 |
| `all_heads_L30` | all_heads_layer | 1 | 32 | 0/12 | -0.001 | +0.006 | +0.007 |
| `known_top1_midlate_L32_L34` | known_top_midlate | 3 | 3 | 0/12 | -0.004 | +0.009 | +0.012 |
| `all_heads_midlate_L32_L34` | all_heads_midlate | 3 | 96 | 0/12 | -0.012 | -0.022 | -0.010 |
| `known_top12_midlate_L32_L34` | known_top_midlate | 3 | 36 | 0/12 | -0.015 | -0.014 | +0.002 |
| `known_top1_all_layers` | known_top_all_layers | 5 | 5 | 0/12 | -0.016 | -0.013 | +0.003 |
| `known_top8_all_layers` | known_top_all_layers | 5 | 40 | 0/12 | -0.029 | +0.001 | +0.030 |
| `all_heads_L34` | all_heads_layer | 1 | 32 | 0/12 | -0.031 | -0.015 | +0.015 |
| `known_top12_all_layers` | known_top_all_layers | 5 | 60 | 0/12 | -0.037 | -0.011 | +0.026 |

## deepseek7b

rows=43, target_seen=43, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 85}, layers=[18, 19, 20, 21, 22], heads={'18': 28, '19': 28, '20': 28, '21': 28, '22': 28}, specs=44, time_min=8.78

### best

| rank | name | kind | random | ops | slots | switch | margin | correct_delta | wrong_delta |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `all_heads_all_layers` | all_heads | False | 5 | 140 | 43/43 | +3.901 | +1.565 | -2.337 |
| 2 | `known_top6_all_layers` | known_top_all_layers | False | 5 | 30 | 42/43 | +2.920 | +1.535 | -1.386 |
| 3 | `known_top6_midlate_L20_L22` | known_top_midlate | False | 3 | 18 | 41/43 | +2.621 | +1.475 | -1.146 |
| 4 | `all_heads_midlate_L20_L22` | all_heads_midlate | False | 3 | 84 | 39/43 | +2.692 | +1.354 | -1.338 |
| 5 | `known_top4_all_layers` | known_top_all_layers | False | 5 | 20 | 36/43 | +2.249 | +1.297 | -0.952 |
| 6 | `known_top4_midlate_L20_L22` | known_top_midlate | False | 3 | 12 | 33/43 | +1.948 | +1.214 | -0.734 |
| 7 | `all_heads_L22` | all_heads_layer | False | 1 | 28 | 32/43 | +1.727 | +1.068 | -0.659 |
| 8 | `known_top2_midlate_L20_L22` | known_top_midlate | False | 3 | 6 | 30/43 | +1.441 | +0.947 | -0.493 |
| 9 | `known_top2_all_layers` | known_top_all_layers | False | 5 | 10 | 30/43 | +1.435 | +0.932 | -0.503 |
| 10 | `all_heads_L18` | all_heads_layer | False | 1 | 28 | 21/43 | +1.056 | +0.601 | -0.455 |
| 11 | `all_heads_L19` | all_heads_layer | False | 1 | 28 | 21/43 | +0.880 | +0.593 | -0.287 |
| 12 | `all_heads_L20` | all_heads_layer | False | 1 | 28 | 18/43 | +0.935 | +0.555 | -0.380 |
| 13 | `L20_H25` | single_known_head | False | 1 | 1 | 12/43 | +0.724 | +0.533 | -0.191 |
| 14 | `L22_H1` | single_known_head | False | 1 | 1 | 12/43 | +0.714 | +0.506 | -0.208 |
| 15 | `known_top1_midlate_L20_L22` | known_top_midlate | False | 3 | 3 | 10/43 | +0.495 | +0.361 | -0.134 |
| 16 | `L18_H24` | single_known_head | False | 1 | 1 | 8/43 | +0.512 | +0.406 | -0.106 |
| 17 | `known_top1_all_layers` | known_top_all_layers | False | 5 | 5 | 8/43 | +0.475 | +0.346 | -0.129 |
| 18 | `L22_H3` | single_known_head | False | 1 | 1 | 7/43 | +0.381 | +0.262 | -0.119 |
| 19 | `L22_H24` | single_known_head | False | 1 | 1 | 7/43 | +0.355 | +0.282 | -0.073 |
| 20 | `all_heads_all_layers` | all_heads | True | 5 | 140 | 6/43 | -0.100 | -0.218 | -0.118 |
| 21 | `L22_H7` | single_known_head | False | 1 | 1 | 5/43 | +0.370 | +0.289 | -0.081 |
| 22 | `L22_coverage_H7` | single_coverage_head | False | 1 | 1 | 5/43 | +0.370 | +0.289 | -0.081 |
| 23 | `L20_H1` | single_known_head | False | 1 | 1 | 5/43 | +0.223 | +0.210 | -0.013 |
| 24 | `L22_H13` | single_known_head | False | 1 | 1 | 4/43 | +0.314 | +0.241 | -0.073 |
| 25 | `known_top6_all_layers` | known_top_all_layers | True | 5 | 30 | 4/43 | -0.159 | -0.149 | +0.010 |
| 26 | `all_heads_L21` | all_heads_layer | False | 1 | 28 | 3/43 | +0.121 | +0.101 | -0.020 |
| 27 | `L18_H7` | single_known_head | True | 1 | 1 | 3/43 | +0.008 | +0.011 | +0.003 |
| 28 | `L18_H13` | single_known_head | True | 1 | 1 | 3/43 | -0.017 | -0.027 | -0.010 |
| 29 | `all_heads_midlate_L20_L22` | all_heads_midlate | True | 3 | 84 | 3/43 | -0.018 | -0.124 | -0.106 |
| 30 | `known_top2_all_layers` | known_top_all_layers | True | 5 | 10 | 2/43 | +0.009 | +0.000 | -0.009 |
| 31 | `all_heads_L19` | all_heads_layer | True | 1 | 28 | 2/43 | -0.005 | -0.020 | -0.015 |
| 32 | `known_top1_all_layers` | known_top_all_layers | True | 5 | 5 | 2/43 | -0.009 | -0.013 | -0.004 |

### all_heads_refs

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `all_heads_all_layers` | all_heads | 5 | 140 | 43/43 | +3.901 | +1.565 | -2.337 |
| `all_heads_midlate_L20_L22` | all_heads_midlate | 3 | 84 | 39/43 | +2.692 | +1.354 | -1.338 |
| `all_heads_L22` | all_heads_layer | 1 | 28 | 32/43 | +1.727 | +1.068 | -0.659 |
| `all_heads_L18` | all_heads_layer | 1 | 28 | 21/43 | +1.056 | +0.601 | -0.455 |
| `all_heads_L19` | all_heads_layer | 1 | 28 | 21/43 | +0.880 | +0.593 | -0.287 |
| `all_heads_L20` | all_heads_layer | 1 | 28 | 18/43 | +0.935 | +0.555 | -0.380 |
| `all_heads_L21` | all_heads_layer | 1 | 28 | 3/43 | +0.121 | +0.101 | -0.020 |

### known_top_cumulative

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `known_top6_all_layers` | known_top_all_layers | 5 | 30 | 42/43 | +2.920 | +1.535 | -1.386 |
| `known_top6_midlate_L20_L22` | known_top_midlate | 3 | 18 | 41/43 | +2.621 | +1.475 | -1.146 |
| `known_top4_all_layers` | known_top_all_layers | 5 | 20 | 36/43 | +2.249 | +1.297 | -0.952 |
| `known_top4_midlate_L20_L22` | known_top_midlate | 3 | 12 | 33/43 | +1.948 | +1.214 | -0.734 |
| `known_top2_midlate_L20_L22` | known_top_midlate | 3 | 6 | 30/43 | +1.441 | +0.947 | -0.493 |
| `known_top2_all_layers` | known_top_all_layers | 5 | 10 | 30/43 | +1.435 | +0.932 | -0.503 |
| `known_top1_midlate_L20_L22` | known_top_midlate | 3 | 3 | 10/43 | +0.495 | +0.361 | -0.134 |
| `known_top1_all_layers` | known_top_all_layers | 5 | 5 | 8/43 | +0.475 | +0.346 | -0.129 |

### single_heads

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `L20_H25` | single_known_head | 1 | 1 | 12/43 | +0.724 | +0.533 | -0.191 |
| `L22_H1` | single_known_head | 1 | 1 | 12/43 | +0.714 | +0.506 | -0.208 |
| `L18_H24` | single_known_head | 1 | 1 | 8/43 | +0.512 | +0.406 | -0.106 |
| `L22_H3` | single_known_head | 1 | 1 | 7/43 | +0.381 | +0.262 | -0.119 |
| `L22_H24` | single_known_head | 1 | 1 | 7/43 | +0.355 | +0.282 | -0.073 |
| `L22_H7` | single_known_head | 1 | 1 | 5/43 | +0.370 | +0.289 | -0.081 |
| `L22_coverage_H7` | single_coverage_head | 1 | 1 | 5/43 | +0.370 | +0.289 | -0.081 |
| `L20_H1` | single_known_head | 1 | 1 | 5/43 | +0.223 | +0.210 | -0.013 |
| `L22_H13` | single_known_head | 1 | 1 | 4/43 | +0.314 | +0.241 | -0.073 |
| `L18_H13` | single_known_head | 1 | 1 | 1/43 | +0.085 | +0.071 | -0.014 |
| `L20_H3` | single_known_head | 1 | 1 | 1/43 | +0.066 | +0.073 | +0.007 |
| `L22_coverage_H27` | single_coverage_head | 1 | 1 | 1/43 | +0.036 | +0.033 | -0.003 |
| `L20_H7` | single_known_head | 1 | 1 | 1/43 | +0.007 | +0.007 | -0.000 |
| `L22_H18` | single_known_head | 1 | 1 | 1/43 | -0.001 | -0.000 | +0.000 |
| `L22_H19` | single_known_head | 1 | 1 | 1/43 | -0.007 | -0.003 | +0.004 |
| `L22_coverage_H14` | single_coverage_head | 1 | 1 | 1/43 | -0.013 | -0.009 | +0.004 |
| `L18_H18` | single_known_head | 1 | 1 | 1/43 | -0.042 | -0.044 | -0.002 |
| `L18_H1` | single_known_head | 1 | 1 | 0/43 | +0.004 | -0.020 | -0.024 |
| `L22_coverage_H21` | single_coverage_head | 1 | 1 | 0/43 | +0.001 | -0.002 | -0.003 |
| `L20_H19` | single_known_head | 1 | 1 | 0/43 | +0.001 | -0.002 | -0.003 |
| `L20_H18` | single_known_head | 1 | 1 | 0/43 | +0.000 | -0.003 | -0.003 |
| `L18_H25` | single_known_head | 1 | 1 | 0/43 | -0.008 | -0.011 | -0.003 |
| `L20_H13` | single_known_head | 1 | 1 | 0/43 | -0.022 | -0.015 | +0.007 |
| `L22_coverage_H0` | single_coverage_head | 1 | 1 | 0/43 | -0.029 | -0.021 | +0.009 |
| `L18_H3` | single_known_head | 1 | 1 | 0/43 | -0.031 | -0.026 | +0.005 |
| `L18_H19` | single_known_head | 1 | 1 | 0/43 | -0.047 | -0.061 | -0.014 |
| `L18_H7` | single_known_head | 1 | 1 | 0/43 | -0.089 | -0.141 | -0.051 |
| `L20_H24` | single_known_head | 1 | 1 | 0/43 | -0.138 | -0.119 | +0.020 |
| `L22_H25` | single_known_head | 1 | 1 | 0/43 | -0.309 | -0.258 | +0.051 |

### random_controls

| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `all_heads_all_layers` | all_heads | 5 | 140 | 6/43 | -0.100 | -0.218 | -0.118 |
| `known_top6_all_layers` | known_top_all_layers | 5 | 30 | 4/43 | -0.159 | -0.149 | +0.010 |
| `all_heads_midlate_L20_L22` | all_heads_midlate | 3 | 84 | 3/43 | -0.018 | -0.124 | -0.106 |
| `known_top2_all_layers` | known_top_all_layers | 5 | 10 | 2/43 | +0.009 | +0.000 | -0.009 |
| `all_heads_L19` | all_heads_layer | 1 | 28 | 2/43 | -0.005 | -0.020 | -0.015 |
| `known_top1_all_layers` | known_top_all_layers | 5 | 5 | 2/43 | -0.009 | -0.013 | -0.004 |
| `known_top2_midlate_L20_L22` | known_top_midlate | 3 | 6 | 2/43 | -0.017 | -0.018 | -0.001 |
| `all_heads_L22` | all_heads_layer | 1 | 28 | 2/43 | -0.039 | -0.063 | -0.025 |
| `all_heads_L18` | all_heads_layer | 1 | 28 | 2/43 | -0.119 | -0.054 | +0.065 |
| `known_top1_midlate_L20_L22` | known_top_midlate | 3 | 3 | 1/43 | +0.011 | +0.007 | -0.004 |
| `known_top4_midlate_L20_L22` | known_top_midlate | 3 | 12 | 1/43 | -0.032 | -0.028 | +0.003 |
| `known_top6_midlate_L20_L22` | known_top_midlate | 3 | 18 | 1/43 | -0.045 | -0.103 | -0.058 |
| `all_heads_L21` | all_heads_layer | 1 | 28 | 0/43 | +0.017 | -0.022 | -0.039 |
| `known_top4_all_layers` | known_top_all_layers | 5 | 20 | 0/43 | -0.058 | -0.052 | +0.006 |
| `all_heads_L20` | all_heads_layer | 1 | 28 | 0/43 | -0.107 | -0.136 | -0.030 |
