# Phase509 Rotation-stable Orthogonal Field Factor Audit Summary

## qwen3

L=36, d=2560, categories=fruit,action,emotion, train=20, test=10, templates=3, rank=4, candidate_random_axes=4

| category | layer | ratio | svd best | rotated best | causal best | causal positive | outside best | causal format | causal surface cat | causal surface punct |
|---|---:|---:|---|---|---|---|---|---:|---:|---:|
| fruit | L18 | 21.99 | svd3 -0.004 weak | rot1 +0.000 weak | svd3 -0.004 weak | svd2 +0.870 competitor_suppressor | outside0 -0.032 weak | 0.057 | -0.160 | +0.000 |
| fruit | L27 | 14.67 | svd3 -0.595 support | rot3 -0.487 support | svd3 -0.595 support | combo3 +0.758 target_release | outside1 -0.032 weak | 0.022 | -0.721 | -0.567 |
| fruit | L33 | 18.02 | svd0 -0.150 weak | rot3 -0.108 weak | combo2 -0.192 weak | svd3 +0.057 weak | outside1 -0.008 weak | 0.646 | -0.365 | -0.217 |
| action | L18 | 47.68 | svd1 -0.064 weak | rot1 -0.081 weak | combo3 -0.094 weak | svd0 +0.321 target_release | outside2 -0.007 weak | 0.671 | -0.096 | +0.142 |
| action | L27 | 33.52 | svd0 -0.294 support | rot0 -0.053 weak | svd0 -0.294 support | combo0 +1.526 competitor_suppressor | outside1 -0.024 weak | 0.899 | -0.360 | +0.833 |
| action | L33 | 73.60 | svd0 -0.230 weak | rot3 -0.306 support | svd0 -0.230 weak | svd2 +0.074 weak | outside2 -0.013 weak | 0.881 | -0.110 | +0.679 |
| emotion | L18 | 75.36 | svd0 +0.009 weak | rot0 +0.004 weak | svd0 +0.009 weak | combo0 +0.152 weak | outside3 -0.005 weak | 0.770 | +0.229 | -0.312 |
| emotion | L27 | 23.03 | svd0 -0.326 support | rot0 -0.362 support | combo3 -0.518 support | combo1 +0.092 weak | outside0 -0.015 weak | 0.645 | -0.977 | -0.842 |
| emotion | L33 | 47.18 | svd0 -0.527 support | rot3 -0.300 support | svd0 -0.527 support | svd2 +0.017 weak | outside0 -0.077 weak | 0.733 | -0.556 | -0.392 |

| metric | value |
|---|---:|
| mean_svd_best | -0.2425 |
| mean_rotated_best | -0.1881 |
| mean_causal_best | -0.2718 |
| mean_causal_positive | 0.4296 |
| mean_outside_best | -0.0236 |
| support_rotation_match_rate | 0.3333 |
| mean_causal_surface_category_delta | -0.3463 |
| mean_causal_surface_punctuation_delta | -0.0750 |

## glm4

L=40, d=4096, categories=emotion,color,fruit, train=20, test=10, templates=3, rank=4, candidate_random_axes=4

| category | layer | ratio | svd best | rotated best | causal best | causal positive | outside best | causal format | causal surface cat | causal surface punct |
|---|---:|---:|---|---|---|---|---|---:|---:|---:|
| emotion | L20 | 69.20 | svd1 -1.329 support | rot1 -1.195 support | svd1 -1.329 support | svd0 +0.121 weak | outside0 -0.009 weak | 0.810 | -1.134 | +1.037 |
| emotion | L30 | 31.16 | svd1 -0.627 support | rot2 -0.407 support | svd1 -0.627 support | combo0 +0.069 weak | outside1 -0.010 weak | 0.969 | -0.497 | +0.449 |
| emotion | L37 | 37.00 | svd2 -0.568 support | rot0 -0.330 support | svd2 -0.568 support | svd3 +0.015 weak | outside2 -0.032 weak | 0.053 | -0.982 | -0.308 |
| color | L20 | 77.24 | svd2 -0.397 support | rot3 -0.340 support | svd2 -0.397 support | combo3 +0.172 weak | outside2 -0.006 weak | 0.175 | -0.665 | +0.568 |
| color | L30 | 64.13 | svd2 -0.543 support | rot3 -0.945 support | svd2 -0.543 support | svd1 +0.398 target_release | outside0 -0.004 weak | 0.050 | -1.803 | +1.123 |
| color | L37 | 61.40 | svd2 -0.650 support | rot1 -0.439 support | svd2 -0.650 support | combo0 +0.436 competitor_suppressor | outside3 -0.011 weak | 0.058 | -1.998 | +1.116 |
| fruit | L20 | 68.91 | svd2 -0.524 support | rot2 -0.306 support | svd2 -0.524 support | combo0 +0.230 weak | outside3 -0.012 weak | 0.340 | -1.188 | +0.673 |
| fruit | L30 | 28.53 | svd1 -0.310 support | rot2 -0.428 support | svd1 -0.310 support | svd2 +0.906 target_release | outside1 -0.011 weak | 0.365 | -0.368 | +0.247 |
| fruit | L37 | 53.35 | svd0 -0.306 support | rot3 -0.390 support | combo2 -0.337 support | combo3 +0.282 target_release | outside1 -0.001 weak | 0.215 | -0.720 | -0.059 |

| metric | value |
|---|---:|
| mean_svd_best | -0.5836 |
| mean_rotated_best | -0.5311 |
| mean_causal_best | -0.5872 |
| mean_causal_positive | 0.2921 |
| mean_outside_best | -0.0105 |
| support_rotation_match_rate | 1.0000 |
| mean_causal_surface_category_delta | -1.0393 |
| mean_causal_surface_punctuation_delta | 0.5383 |

## deepseek7b

L=28, d=3584, categories=action,fruit,color, train=20, test=10, templates=3, rank=4, candidate_random_axes=4

| category | layer | ratio | svd best | rotated best | causal best | causal positive | outside best | causal format | causal surface cat | causal surface punct |
|---|---:|---:|---|---|---|---|---|---:|---:|---:|
| action | L14 | 45.90 | svd2 -0.206 weak | rot0 -0.316 support | combo2 -0.401 support | svd0 +0.925 target_release | outside3 -0.027 weak | 0.722 | -0.313 | -0.058 |
| action | L21 | 31.11 | svd2 -0.032 weak | rot3 -0.201 weak | combo2 -0.275 support | svd0 +1.109 target_release | outside3 -0.009 weak | 0.471 | +0.047 | -0.267 |
| action | L25 | 35.97 | svd1 +0.076 weak | rot2 +0.008 weak | combo1 +0.023 weak | svd2 +1.483 target_release | outside1 -0.025 weak | 0.763 | -0.019 | +0.008 |
| fruit | L14 | 160.92 | svd3 -0.006 weak | rot0 +0.037 weak | svd3 -0.006 weak | combo3 +0.922 target_release | outside3 -0.062 weak | 0.029 | -0.042 | -0.004 |
| fruit | L21 | 121.08 | svd3 +0.078 weak | rot1 -0.014 weak | combo0 +0.013 weak | combo1 +0.907 target_release | outside2 -0.047 weak | 0.094 | -0.109 | +0.029 |
| fruit | L25 | 49.17 | svd2 -0.201 weak | rot2 -0.248 weak | combo2 -0.431 support | svd1 +0.178 weak | outside0 -0.091 weak | 0.880 | -0.609 | +0.071 |
| color | L14 | 35.93 | svd3 -0.177 weak | rot2 -0.084 weak | svd3 -0.177 weak | svd0 +0.438 target_release | outside2 +0.003 weak | 0.023 | -0.244 | +0.052 |
| color | L21 | 39.97 | svd3 -0.035 weak | rot1 -0.080 weak | combo3 -0.151 weak | svd0 +0.444 competitor_suppressor | outside3 -0.044 weak | 0.748 | -0.017 | +0.208 |
| color | L25 | 46.84 | svd1 -0.057 weak | rot0 +0.070 weak | svd1 -0.057 weak | svd2 +0.727 target_release | outside1 +0.009 weak | 0.756 | +0.163 | +0.156 |

| metric | value |
|---|---:|
| mean_svd_best | -0.0623 |
| mean_rotated_best | -0.0921 |
| mean_causal_best | -0.1625 |
| mean_causal_positive | 0.7925 |
| mean_outside_best | -0.0325 |
| support_rotation_match_rate | 0.0000 |
| mean_causal_surface_category_delta | -0.1271 |
| mean_causal_surface_punctuation_delta | 0.0218 |

## Cross-model Compact

| model | svd best | rotated best | causal best | causal positive | outside best | rotation support match | surface category | surface punctuation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | -0.2425 | -0.1881 | -0.2718 | 0.4296 | -0.0236 | 0.3333 | -0.3463 | -0.0750 |
| glm4 | -0.5836 | -0.5311 | -0.5872 | 0.2921 | -0.0105 | 1.0000 | -1.0393 | 0.5383 |
| deepseek7b | -0.0623 | -0.0921 | -0.1625 | 0.7925 | -0.0325 | 0.0000 | -0.1271 | 0.0218 |

