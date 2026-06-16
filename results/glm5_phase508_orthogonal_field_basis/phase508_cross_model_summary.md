# Phase508 Orthogonal Field Basis Decomposition Summary

## qwen3

L=36, d=2560, train=20, test=10, templates=3, rank=4, scale=1.0

| category | layer | ratio | best component | strongest positive | best random | support_top4 | suppressor_top2 | format_top2 |
|---|---:|---:|---|---|---|---|---|---|
| fruit | L18 | 21.99 | 3 -0.004 weak | 2 +0.870 competitor_suppressor | 3 -0.015 weak | [3, 1, 0, 2] +1.440 support_top4 | [2, 0] +1.350 suppressor_top2 | [0, 1] +0.467 format_aligned_top2 |
| fruit | L27 | 14.67 | 3 -0.595 support | 0 +0.484 target_release | 1 -0.011 weak | [3, 1, 2, 0] -0.036 support_top4 | [0, 2] +0.763 suppressor_top2 | [0, 1] +0.325 format_aligned_top2 |
| fruit | L33 | 18.02 | 0 -0.150 weak | 3 +0.057 weak | 0 -0.020 weak | [0, 2, 1, 3] +0.021 support_top4 | [3, 1] +0.105 suppressor_top2 | [0, 1] -0.119 format_aligned_top2 |
| animal | L18 | 25.06 | 3 -0.033 weak | 0 +0.185 weak | 2 +0.001 weak | [3, 2, 1, 0] +0.248 support_top4 | [0, 1] +0.261 suppressor_top2 | [0, 1] +0.261 format_aligned_top2 |
| animal | L27 | 32.56 | 2 -0.465 support | 3 -0.040 weak | 1 -0.017 weak | [2, 0, 1, 3] -0.899 support_top4 | [3, 1] -0.230 suppressor_top2 | [0, 1] -0.430 format_aligned_top2 |
| animal | L33 | 34.16 | 2 -0.304 support | 1 +0.002 weak | 0 -0.021 weak | [2, 0, 3, 1] -0.334 support_top4 | [1, 3] -0.010 suppressor_top2 | [0, 1] -0.069 format_aligned_top2 |
| action | L18 | 47.68 | 1 -0.064 weak | 0 +0.321 target_release | 1 -0.016 weak | [1, 3, 2, 0] +0.400 support_top4 | [0, 2] +0.370 suppressor_top2 | [0, 1] +0.314 format_aligned_top2 |
| action | L27 | 33.52 | 0 -0.294 support | 2 +0.746 target_release | 3 -0.024 weak | [0, 3, 1, 2] +0.991 support_top4 | [2, 1] +1.364 suppressor_top2 | [0, 1] +0.276 format_aligned_top2 |
| action | L33 | 73.60 | 0 -0.230 weak | 2 +0.074 weak | 2 +0.001 weak | [0, 1, 3, 2] -0.152 support_top4 | [2, 3] +0.086 suppressor_top2 | [0, 1] -0.292 format_aligned_top2 |
| emotion | L18 | 75.36 | 0 +0.009 weak | 3 +0.109 weak | 2 -0.007 weak | [0, 2, 1, 3] +0.194 support_top4 | [3, 1] +0.183 suppressor_top2 | [0, 1] +0.049 format_aligned_top2 |
| emotion | L27 | 23.03 | 0 -0.326 support | 2 +0.034 weak | 1 +0.000 weak | [0, 3, 1, 2] -0.366 support_top4 | [2, 1] +0.022 suppressor_top2 | [0, 2] -0.259 format_aligned_top2 |
| emotion | L33 | 47.18 | 0 -0.527 support | 2 +0.017 weak | 1 -0.029 weak | [0, 1, 3, 2] -0.585 support_top4 | [2, 3] +0.007 suppressor_top2 | [0, 1] -0.621 format_aligned_top2 |
| clothing | L18 | 62.77 | 1 -0.133 weak | 0 +0.346 target_release | 1 -0.007 weak | [1, 3, 2, 0] +0.007 support_top4 | [0, 2] +0.283 suppressor_top2 | [0, 1] +0.156 format_aligned_top2 |
| clothing | L27 | 37.80 | 1 -0.153 weak | 0 +0.351 target_release | 2 -0.024 weak | [1, 2, 3, 0] +0.211 support_top4 | [0, 3] +0.354 suppressor_top2 | [0, 1] +0.209 format_aligned_top2 |
| clothing | L33 | 60.62 | 0 -0.138 weak | 2 +0.056 weak | 2 -0.030 weak | [0, 1, 3, 2] -0.082 support_top4 | [2, 3] +0.071 suppressor_top2 | [0, 1] -0.121 format_aligned_top2 |
| color | L18 | 19.38 | 0 -0.037 weak | 2 +0.052 weak | 2 -0.010 weak | [0, 1, 3, 2] -0.078 support_top4 | [2, 3] +0.062 suppressor_top2 | [0, 1] -0.026 format_aligned_top2 |
| color | L27 | 14.82 | 2 -0.143 weak | 0 +0.150 weak | 2 -0.042 weak | [2, 3, 1, 0] +0.143 support_top4 | [0, 1] +0.315 suppressor_top2 | [0, 1] +0.315 format_aligned_top2 |
| color | L33 | 47.40 | 3 -0.124 weak | 2 +0.213 weak | 3 -0.008 weak | [3, 0, 1, 2] +0.318 support_top4 | [2, 1] +0.260 suppressor_top2 | [0, 1] +0.100 format_aligned_top2 |
| vehicle | L18 | 24.78 | 3 +0.013 weak | 0 +0.263 target_release | 1 -0.013 weak | [3, 2, 1, 0] +0.273 support_top4 | [0, 1] +0.275 suppressor_top2 | [0, 1] +0.275 format_aligned_top2 |
| vehicle | L27 | 19.61 | 1 -0.087 weak | 2 +0.164 weak | 2 -0.017 weak | [1, 3, 0, 2] +0.195 support_top4 | [2, 0] +0.306 suppressor_top2 | [0, 1] +0.077 format_aligned_top2 |
| vehicle | L33 | 23.80 | 0 -0.082 weak | 1 +0.171 weak | 3 -0.008 weak | [0, 3, 2, 1] +0.233 support_top4 | [1, 2] +0.262 suppressor_top2 | [0, 1] +0.095 format_aligned_top2 |

| metric | value |
|---|---:|
| mean_ratio | 36.0854 |
| mean_best_delta | -0.1842 |
| mean_pos_delta | 0.2201 |
| mean_random_best | -0.0150 |
| support_label_rate | 0.0833 |
| positive_label_rate | 0.1071 |

## glm4

L=40, d=4096, train=20, test=10, templates=3, rank=4, scale=1.0

| category | layer | ratio | best component | strongest positive | best random | support_top4 | suppressor_top2 | format_top2 |
|---|---:|---:|---|---|---|---|---|---|
| fruit | L20 | 68.91 | 2 -0.524 support | 0 -0.012 weak | 0 -0.005 weak | [2, 1, 3, 0] -1.714 support_top4 | [0, 3] -0.154 suppressor_top2 | [0, 1] -0.245 format_aligned_top2 |
| fruit | L30 | 28.53 | 1 -0.310 support | 2 +0.906 target_release | 3 -0.007 weak | [1, 0, 3, 2] -0.176 support_top4 | [2, 3] +0.819 suppressor_top2 | [3, 0] +0.336 format_aligned_top2 |
| fruit | L37 | 53.35 | 0 -0.306 support | 2 -0.014 weak | 1 -0.021 weak | [0, 3, 1, 2] -0.608 support_top4 | [2, 1] -0.205 suppressor_top2 | [3, 0] -0.391 format_aligned_top2 |
| animal | L20 | 101.20 | 2 -0.182 weak | 0 +0.265 target_release | 3 -0.000 weak | [2, 1, 3, 0] -0.277 support_top4 | [0, 3] +0.283 suppressor_top2 | [1, 0] +0.355 format_aligned_top2 |
| animal | L30 | 41.29 | 2 -0.492 support | 1 +0.107 weak | 0 -0.000 weak | [2, 3, 0, 1] -0.519 support_top4 | [1, 0] +0.010 suppressor_top2 | [1, 0] +0.010 format_aligned_top2 |
| animal | L37 | 50.01 | 2 -0.289 support | 0 +0.058 weak | 2 -0.006 weak | [2, 3, 1, 0] -0.769 support_top4 | [0, 1] -0.167 suppressor_top2 | [0, 1] -0.167 format_aligned_top2 |
| action | L20 | 58.33 | 1 -0.250 weak | 0 +1.145 target_release | 3 +0.001 weak | [1, 2, 3, 0] +0.401 support_top4 | [0, 3] +1.120 suppressor_top2 | [0, 1] +0.391 format_aligned_top2 |
| action | L30 | 133.05 | 1 -0.210 weak | 0 +0.698 target_release | 1 -0.007 weak | [1, 3, 2, 0] +0.216 support_top4 | [0, 2] +0.650 suppressor_top2 | [1, 0] +0.312 format_aligned_top2 |
| action | L37 | 115.51 | 1 -0.336 support | 0 +0.315 target_release | 2 -0.002 weak | [1, 2, 3, 0] -0.180 support_top4 | [0, 3] +0.332 suppressor_top2 | [0, 1] -0.148 format_aligned_top2 |
| emotion | L20 | 69.20 | 1 -1.329 support | 0 +0.121 weak | 2 -0.010 weak | [1, 2, 3, 0] -1.835 support_top4 | [0, 3] -0.072 suppressor_top2 | [0, 1] -1.417 format_aligned_top2 |
| emotion | L30 | 31.16 | 1 -0.627 support | 3 -0.019 weak | 0 -0.007 weak | [1, 2, 0, 3] -0.491 support_top4 | [3, 0] +0.022 suppressor_top2 | [1, 0] -0.355 format_aligned_top2 |
| emotion | L37 | 37.00 | 2 -0.568 support | 3 +0.015 weak | 3 -0.019 weak | [2, 1, 0, 3] -0.596 support_top4 | [3, 0] -0.045 suppressor_top2 | [1, 0] -0.217 format_aligned_top2 |
| clothing | L20 | 69.18 | 1 -0.135 weak | 0 +1.467 target_release | 3 -0.009 weak | [1, 3, 2, 0] +1.102 support_top4 | [0, 2] +1.701 suppressor_top2 | [1, 0] +1.024 format_aligned_top2 |
| clothing | L30 | 55.24 | 2 -0.208 weak | 1 +0.472 target_release | 0 -0.008 weak | [2, 3, 0, 1] -0.109 support_top4 | [1, 0] +0.453 suppressor_top2 | [0, 1] +0.453 format_aligned_top2 |
| clothing | L37 | 55.81 | 3 -0.218 weak | 1 +0.101 weak | 2 -0.009 weak | [3, 0, 2, 1] +0.011 support_top4 | [1, 2] +0.233 suppressor_top2 | [1, 0] +0.106 format_aligned_top2 |
| color | L20 | 77.24 | 2 -0.397 support | 1 +0.031 weak | 2 -0.018 weak | [2, 0, 3, 1] -0.664 support_top4 | [1, 3] +0.029 suppressor_top2 | [1, 0] -0.266 format_aligned_top2 |
| color | L30 | 64.13 | 2 -0.543 support | 1 +0.398 target_release | 3 -0.011 weak | [2, 3, 0, 1] -0.542 support_top4 | [1, 0] +0.298 suppressor_top2 | [0, 1] +0.298 format_aligned_top2 |
| color | L37 | 61.40 | 2 -0.650 support | 1 +0.215 weak | 3 -0.017 weak | [2, 3, 0, 1] -0.628 support_top4 | [1, 0] +0.205 suppressor_top2 | [0, 1] +0.205 format_aligned_top2 |
| vehicle | L20 | 54.11 | 1 -0.312 support | 0 +0.228 weak | 0 -0.018 weak | [1, 2, 3, 0] -1.266 support_top4 | [0, 3] +0.184 suppressor_top2 | [1, 0] -0.480 format_aligned_top2 |
| vehicle | L30 | 33.24 | 2 -0.428 support | 1 +0.142 weak | 0 -0.011 weak | [2, 3, 0, 1] -0.743 support_top4 | [1, 0] +0.064 suppressor_top2 | [1, 0] +0.064 format_aligned_top2 |
| vehicle | L37 | 32.29 | 3 -0.339 support | 0 -0.015 weak | 0 +0.001 weak | [3, 2, 1, 0] -0.757 support_top4 | [0, 1] -0.072 suppressor_top2 | [0, 1] -0.072 format_aligned_top2 |

| metric | value |
|---|---:|
| mean_ratio | 61.4361 |
| mean_best_delta | -0.4119 |
| mean_pos_delta | 0.3154 |
| mean_random_best | -0.0088 |
| support_label_rate | 0.2143 |
| positive_label_rate | 0.1310 |

## deepseek7b

L=28, d=3584, train=20, test=10, templates=3, rank=4, scale=1.0

| category | layer | ratio | best component | strongest positive | best random | support_top4 | suppressor_top2 | format_top2 |
|---|---:|---:|---|---|---|---|---|---|
| fruit | L14 | 160.92 | 3 -0.006 weak | 0 +0.806 target_release | 0 -0.023 weak | [3, 1, 2, 0] +2.029 support_top4 | [0, 2] +1.829 suppressor_top2 | [0, 1] +1.039 format_aligned_top2 |
| fruit | L21 | 121.08 | 3 +0.078 weak | 2 +0.657 competitor_suppressor | 3 -0.062 weak | [3, 1, 0, 2] +1.919 support_top4 | [2, 0] +1.563 suppressor_top2 | [0, 1] +0.942 format_aligned_top2 |
| fruit | L25 | 49.17 | 2 -0.201 weak | 1 +0.178 weak | 0 -0.008 weak | [2, 0, 3, 1] -0.215 support_top4 | [1, 3] +0.071 suppressor_top2 | [0, 1] +0.200 format_aligned_top2 |
| animal | L14 | 159.20 | 1 -0.000 weak | 0 +0.397 suppressor_or_interface | 3 -0.012 weak | [1, 3, 2, 0] +0.622 support_top4 | [0, 2] +0.539 suppressor_top2 | [0, 1] +0.498 format_aligned_top2 |
| animal | L21 | 94.01 | 2 -0.249 weak | 0 +0.397 competitor_suppressor | 3 -0.038 weak | [2, 3, 1, 0] +0.296 support_top4 | [0, 1] +0.509 suppressor_top2 | [0, 1] +0.509 format_aligned_top2 |
| animal | L25 | 39.27 | 0 -0.204 weak | 2 +0.169 weak | 2 -0.027 weak | [0, 1, 3, 2] -0.011 support_top4 | [2, 3] +0.330 suppressor_top2 | [0, 1] -0.253 format_aligned_top2 |
| action | L14 | 45.90 | 2 -0.206 weak | 0 +0.925 target_release | 2 -0.016 weak | [2, 3, 1, 0] +0.761 support_top4 | [0, 1] +0.973 suppressor_top2 | [0, 1] +0.973 format_aligned_top2 |
| action | L21 | 31.11 | 2 -0.032 weak | 0 +1.109 target_release | 1 -0.018 weak | [2, 3, 1, 0] +1.185 support_top4 | [0, 1] +1.202 suppressor_top2 | [0, 1] +1.202 format_aligned_top2 |
| action | L25 | 35.97 | 1 +0.076 weak | 2 +1.483 target_release | 1 -0.016 weak | [1, 3, 0, 2] +1.915 support_top4 | [2, 0] +1.671 suppressor_top2 | [0, 1] +0.349 format_aligned_top2 |
| emotion | L14 | 93.42 | 3 -0.019 weak | 0 +0.186 weak | 0 +0.012 weak | [3, 2, 1, 0] +0.254 support_top4 | [0, 1] +0.359 suppressor_top2 | [0, 1] +0.359 format_aligned_top2 |
| emotion | L21 | 129.66 | 2 -0.009 weak | 1 +0.114 weak | 1 -0.039 weak | [2, 3, 0, 1] +0.013 support_top4 | [1, 0] +0.050 suppressor_top2 | [0, 1] +0.050 format_aligned_top2 |
| emotion | L25 | 84.05 | 2 +0.073 weak | 1 +0.187 weak | 3 -0.001 weak | [2, 3, 0, 1] +0.258 support_top4 | [1, 0] +0.165 suppressor_top2 | [0, 1] +0.165 format_aligned_top2 |
| clothing | L14 | 58.52 | 2 +0.025 weak | 0 +0.146 weak | 2 -0.041 weak | [2, 1, 3, 0] +0.136 support_top4 | [0, 3] +0.188 suppressor_top2 | [0, 1] +0.170 format_aligned_top2 |
| clothing | L21 | 63.00 | 0 -0.319 support | 2 +0.009 weak | 3 -0.017 weak | [0, 1, 3, 2] -0.485 support_top4 | [2, 3] -0.026 suppressor_top2 | [0, 1] -0.361 format_aligned_top2 |
| clothing | L25 | 69.49 | 0 -0.239 weak | 3 +0.005 weak | 1 -0.001 weak | [0, 1, 2, 3] -0.255 support_top4 | [3, 2] +0.018 suppressor_top2 | [0, 1] -0.157 format_aligned_top2 |
| color | L14 | 35.93 | 3 -0.177 weak | 0 +0.438 target_release | 2 -0.011 weak | [3, 1, 2, 0] +0.559 support_top4 | [0, 2] +0.800 suppressor_top2 | [0, 1] +0.451 format_aligned_top2 |
| color | L21 | 39.97 | 3 -0.035 weak | 0 +0.444 competitor_suppressor | 0 -0.002 weak | [3, 2, 1, 0] +0.555 support_top4 | [0, 1] +0.451 suppressor_top2 | [0, 1] +0.451 format_aligned_top2 |
| color | L25 | 46.84 | 1 -0.057 weak | 2 +0.727 target_release | 3 +0.000 weak | [1, 3, 0, 2] +0.853 support_top4 | [2, 0] +0.922 suppressor_top2 | [0, 1] +0.193 format_aligned_top2 |
| vehicle | L14 | 74.86 | 1 +0.029 weak | 0 +0.454 suppressor_or_interface | 0 +0.002 weak | [1, 3, 2, 0] +0.625 support_top4 | [0, 2] +0.569 suppressor_top2 | [0, 1] +0.511 format_aligned_top2 |
| vehicle | L21 | 164.78 | 1 -0.150 weak | 0 +0.105 weak | 0 -0.013 weak | [1, 2, 3, 0] -0.106 support_top4 | [0, 3] +0.065 suppressor_top2 | [0, 1] -0.022 format_aligned_top2 |
| vehicle | L25 | 100.98 | 1 -0.108 weak | 0 +0.039 weak | 0 -0.021 weak | [1, 2, 3, 0] -0.024 support_top4 | [0, 3] +0.118 suppressor_top2 | [0, 1] -0.095 format_aligned_top2 |

| metric | value |
|---|---:|
| mean_ratio | 80.8626 |
| mean_best_delta | -0.0824 |
| mean_pos_delta | 0.4274 |
| mean_random_best | -0.0167 |
| support_label_rate | 0.0119 |
| positive_label_rate | 0.1667 |

## Cross-model Compact

| model | mean ratio | mean best ΔD | mean strongest positive ΔD | mean random best ΔD | support label rate | positive label rate |
|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 36.0854 | -0.1842 | 0.2201 | -0.0150 | 0.0833 | 0.1071 |
| glm4 | 61.4361 | -0.4119 | 0.3154 | -0.0088 | 0.2143 | 0.1310 |
| deepseek7b | 80.8626 | -0.0824 | 0.4274 | -0.0167 | 0.0119 | 0.1667 |

