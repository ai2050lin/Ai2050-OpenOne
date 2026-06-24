# Phase598 Cross-Model Summary

Downstream trajectory acceptance audit after MLP input state interpolation.

## qwen3

cases=64, rows=5, target_cases_seen=5, alpha=2.0, window=3, time_min=0.46

### Final patch effects

| key | kind | n | switch | generated_down_projection | margin_gain | specific_margin_gain | common_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `query_category|L32|repair_alpha2` | repair | 5 | 2/5 | 1.211 | 0.050 | 0.050 | -0.012 |
| `prompt_last|L34|wrong_alpha2` | wrong | 5 | 1/5 | 0.484 | 0.100 | 0.100 | -3.004 |
| `prompt_last|L32|repair_alpha2` | repair | 5 | 1/5 | 0.371 | 0.075 | 0.075 | 0.299 |
| `query_category|L32|wrong_alpha2` | wrong | 5 | 1/5 | 1.485 | 0.050 | 0.050 | -0.018 |
| `prompt_last|L32|wrong_alpha2` | wrong | 5 | 1/5 | 0.544 | 0.050 | 0.050 | -0.010 |
| `prompt_last|L34|random_alpha2` | random | 5 | 1/5 | -1.100 | 0.025 | 0.025 | -0.276 |
| `prompt_last|L32|random_alpha2` | random | 5 | 0/5 | 0.105 | 0.075 | 0.075 | -0.060 |
| `prompt_last|L34|repair_alpha2` | repair | 5 | 0/5 | 0.558 | 0.025 | 0.025 | -3.096 |
| `query_category|L32|random_alpha2` | random | 5 | 0/5 | 1.217 | -0.000 | -0.000 | -0.033 |

### Downstream hidden trajectory

| key | kind | hidden | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `query_category|L32|wrong_alpha2|H33` | wrong | H33 | 5 | 1.487 | 0.995 | -0.492 | 1.000 |
| `query_category|L32|random_alpha2|H33` | random | H33 | 5 | 1.220 | 0.997 | -0.223 | 0.800 |
| `query_category|L32|repair_alpha2|H33` | repair | H33 | 5 | 1.212 | 0.935 | -0.277 | 1.000 |
| `query_category|L32|wrong_alpha2|H35` | wrong | H35 | 5 | 1.147 | 0.759 | -0.389 | 0.800 |
| `query_category|L32|random_alpha2|H34` | random | H34 | 5 | 1.104 | 0.744 | -0.360 | 0.800 |
| `query_category|L32|random_alpha2|H35` | random | H35 | 5 | 0.992 | 0.761 | -0.230 | 0.600 |
| `query_category|L32|repair_alpha2|H35` | repair | H35 | 5 | 0.892 | 0.730 | -0.162 | 1.000 |
| `prompt_last|L32|wrong_alpha2|H34` | wrong | H34 | 5 | 0.670 | 0.393 | -0.276 | 0.800 |
| `prompt_last|L32|repair_alpha2|H35` | repair | H35 | 5 | 0.668 | 0.353 | -0.315 | 1.000 |
| `prompt_last|L32|wrong_alpha2|H35` | wrong | H35 | 5 | 0.582 | 0.343 | -0.239 | 0.800 |
| `prompt_last|L34|repair_alpha2|H35` | repair | H35 | 5 | 0.558 | 0.468 | -0.090 | 0.800 |
| `prompt_last|L32|wrong_alpha2|H33` | wrong | H33 | 5 | 0.544 | 0.294 | -0.250 | 0.800 |
| `prompt_last|L34|wrong_alpha2|H35` | wrong | H35 | 5 | 0.481 | 0.244 | -0.237 | 0.800 |
| `prompt_last|L32|repair_alpha2|H34` | repair | H34 | 5 | 0.479 | 0.321 | -0.158 | 1.000 |
| `query_category|L32|wrong_alpha2|H34` | wrong | H34 | 5 | 0.468 | 0.437 | -0.031 | 1.000 |
| `prompt_last|L32|repair_alpha2|H33` | repair | H33 | 5 | 0.372 | 0.290 | -0.082 | 1.000 |
| `prompt_last|L34|repair_alpha2|H36` | repair | H36 | 5 | 0.294 | 0.152 | -0.141 | 0.800 |
| `query_category|L32|repair_alpha2|H34` | repair | H34 | 5 | 0.269 | 0.410 | 0.141 | 0.400 |
| `query_category|L32|random_alpha2|H36` | random | H36 | 5 | 0.255 | 0.153 | -0.101 | 1.000 |
| `prompt_last|L32|wrong_alpha2|H36` | wrong | H36 | 5 | 0.250 | 0.127 | -0.124 | 0.800 |

## glm4

cases=64, rows=4, target_cases_seen=4, alpha=2.0, window=3, time_min=0.68

### Final patch effects

| key | kind | n | switch | generated_down_projection | margin_gain | specific_margin_gain | common_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `prompt_last|L38|random_alpha2` | random | 4 | 0/4 | -0.432 | 0.016 | 0.016 | -0.480 |
| `prompt_last|L39|wrong_alpha2` | wrong | 4 | 0/4 | 0.096 | 0.000 | 0.000 | -0.214 |
| `prompt_last|L39|repair_alpha2` | repair | 4 | 0/4 | 0.037 | 0.000 | 0.000 | -0.174 |
| `prompt_last|L39|random_alpha2` | random | 4 | 0/4 | 0.029 | 0.000 | 0.000 | -0.114 |
| `prompt_last|L37|wrong_alpha2` | wrong | 4 | 0/4 | -0.044 | -0.016 | -0.016 | -0.303 |
| `prompt_last|L38|wrong_alpha2` | wrong | 4 | 0/4 | -0.564 | -0.016 | -0.016 | -2.077 |
| `prompt_last|L38|repair_alpha2` | repair | 4 | 0/4 | 0.245 | -0.016 | -0.016 | 0.100 |
| `prompt_last|L37|repair_alpha2` | repair | 4 | 0/4 | 0.062 | -0.031 | -0.031 | -0.259 |
| `prompt_last|L37|random_alpha2` | random | 4 | 0/4 | 0.178 | -0.031 | -0.031 | 0.059 |

### Downstream hidden trajectory

| key | kind | hidden | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `prompt_last|L38|repair_alpha2|H40` | repair | H40 | 4 | 0.364 | 0.130 | -0.234 | 0.750 |
| `prompt_last|L37|random_alpha2|H39` | random | H39 | 4 | 0.350 | 0.191 | -0.159 | 1.000 |
| `prompt_last|L37|random_alpha2|H40` | random | H40 | 4 | 0.283 | 0.164 | -0.120 | 1.000 |
| `prompt_last|L38|repair_alpha2|H39` | repair | H39 | 4 | 0.246 | 0.065 | -0.181 | 0.250 |
| `prompt_last|L37|random_alpha2|H38` | random | H38 | 4 | 0.178 | 0.077 | -0.101 | 1.000 |
| `prompt_last|L37|repair_alpha2|H40` | repair | H40 | 4 | 0.124 | 0.064 | -0.060 | 0.750 |
| `prompt_last|L37|repair_alpha2|H39` | repair | H39 | 4 | 0.110 | 0.074 | -0.036 | 0.500 |
| `prompt_last|L37|repair_alpha2|H38` | repair | H38 | 4 | 0.061 | 0.050 | -0.011 | 0.500 |
| `prompt_last|L39|repair_alpha2|H40` | repair | H40 | 4 | 0.040 | 0.023 | -0.016 | 0.500 |
| `prompt_last|L39|wrong_alpha2|H40` | wrong | H40 | 4 | 0.031 | 0.053 | 0.022 | 0.500 |
| `prompt_last|L39|random_alpha2|H40` | random | H40 | 4 | 0.013 | 0.095 | 0.082 | 0.500 |
| `prompt_last|L37|wrong_alpha2|H38` | wrong | H38 | 4 | -0.045 | 0.008 | 0.054 | 0.250 |
| `prompt_last|L37|wrong_alpha2|H40` | wrong | H40 | 4 | -0.053 | -0.048 | 0.005 | 0.250 |
| `prompt_last|L37|wrong_alpha2|H39` | wrong | H39 | 4 | -0.062 | -0.031 | 0.031 | 0.250 |
| `prompt_last|L38|random_alpha2|H40` | random | H40 | 4 | -0.166 | -0.193 | -0.028 | 0.250 |
| `prompt_last|L38|wrong_alpha2|H40` | wrong | H40 | 4 | -0.323 | -0.299 | 0.024 | 0.250 |
| `prompt_last|L38|random_alpha2|H39` | random | H39 | 4 | -0.434 | -0.300 | 0.134 | 0.250 |
| `prompt_last|L38|wrong_alpha2|H39` | wrong | H39 | 4 | -0.567 | -0.506 | 0.062 | 0.250 |

## deepseek7b

cases=64, rows=21, target_cases_seen=21, alpha=2.0, window=3, time_min=1.97

### Final patch effects

| key | kind | n | switch | generated_down_projection | margin_gain | specific_margin_gain | common_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|repair_alpha2` | repair | 21 | 0/21 | 3.618 | -0.006 | -0.006 | 0.004 |
| `rule_value|L26|random_alpha2` | random | 21 | 0/21 | 4.251 | -0.012 | -0.012 | 0.008 |
| `prompt_last|L26|random_alpha2` | random | 21 | 0/21 | 6.424 | -0.012 | -0.012 | -0.297 |
| `query_relation|L19|wrong_alpha2` | wrong | 21 | 0/21 | -0.454 | -0.015 | -0.015 | -0.396 |
| `rule_value|L26|wrong_alpha2` | wrong | 8 | 0/8 | 0.973 | -0.016 | -0.016 | 0.013 |
| `prompt_last|L26|repair_alpha2` | repair | 21 | 0/21 | 4.414 | -0.022 | -0.022 | -2.500 |
| `query_relation|L19|repair_alpha2` | repair | 21 | 0/21 | 0.353 | -0.026 | -0.026 | 0.140 |
| `prompt_last|L26|wrong_alpha2` | wrong | 21 | 0/21 | 1.557 | -0.036 | -0.036 | -0.021 |
| `query_relation|L19|random_alpha2` | random | 21 | 0/21 | -0.521 | -0.083 | -0.083 | -0.409 |

### Downstream hidden trajectory

| key | kind | hidden | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `prompt_last|L26|random_alpha2|H27` | random | H27 | 21 | 6.382 | 1.838 | -4.544 | 0.905 |
| `prompt_last|L26|repair_alpha2|H27` | repair | H27 | 21 | 4.402 | 2.172 | -2.231 | 0.810 |
| `rule_value|L26|random_alpha2|H27` | random | H27 | 21 | 4.270 | 1.283 | -2.987 | 0.810 |
| `rule_value|L26|repair_alpha2|H27` | repair | H27 | 21 | 3.643 | 0.877 | -2.766 | 0.857 |
| `query_relation|L19|random_alpha2|H23` | random | H23 | 21 | 1.889 | 0.410 | -1.479 | 0.857 |
| `query_relation|L19|random_alpha2|H22` | random | H22 | 21 | 1.569 | 0.426 | -1.143 | 0.857 |
| `prompt_last|L26|wrong_alpha2|H27` | wrong | H27 | 21 | 1.519 | 0.329 | -1.190 | 0.571 |
| `rule_value|L26|wrong_alpha2|H27` | wrong | H27 | 8 | 0.980 | -0.587 | -1.567 | 0.625 |
| `query_relation|L19|repair_alpha2|H21` | repair | H21 | 21 | 0.923 | 0.176 | -0.747 | 0.762 |
| `query_relation|L19|repair_alpha2|H22` | repair | H22 | 21 | 0.650 | -0.004 | -0.654 | 0.762 |
| `query_relation|L19|repair_alpha2|H23` | repair | H23 | 21 | 0.599 | -0.016 | -0.615 | 0.714 |
| `query_relation|L19|random_alpha2|H21` | random | H21 | 21 | 0.558 | 0.092 | -0.466 | 0.524 |
| `query_relation|L19|repair_alpha2|H20` | repair | H20 | 21 | 0.354 | 0.137 | -0.217 | 0.619 |
| `query_relation|L19|wrong_alpha2|H22` | wrong | H22 | 21 | 0.297 | 0.067 | -0.230 | 0.476 |
| `rule_value|L26|wrong_alpha2|H28` | wrong | H28 | 8 | 0.270 | 0.089 | -0.182 | 0.625 |
| `query_relation|L19|wrong_alpha2|H23` | wrong | H23 | 21 | 0.259 | 0.095 | -0.165 | 0.429 |
| `query_relation|L19|random_alpha2|H28` | random | H28 | 21 | 0.190 | 0.045 | -0.145 | 0.762 |
| `prompt_last|L26|repair_alpha2|H28` | repair | H28 | 21 | 0.151 | 0.169 | 0.018 | 0.619 |
| `rule_value|L26|repair_alpha2|H28` | repair | H28 | 21 | 0.151 | 0.052 | -0.099 | 0.619 |
| `prompt_last|L26|random_alpha2|H28` | random | H28 | 21 | 0.091 | 0.049 | -0.042 | 0.571 |

### DS7B watched final effects

| key | kind | n | switch | generated_down_projection | margin_gain | specific_margin_gain | common_delta |
|---|---|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|repair_alpha2` | repair | 21 | 0/21 | 3.618 | -0.006 | -0.006 | 0.004 |
| `rule_value|L26|random_alpha2` | random | 21 | 0/21 | 4.251 | -0.012 | -0.012 | 0.008 |
| `rule_value|L26|wrong_alpha2` | wrong | 8 | 0/8 | 0.973 | -0.016 | -0.016 | 0.013 |
| `prompt_last|L26|repair_alpha2` | repair | 21 | 0/21 | 4.414 | -0.022 | -0.022 | -2.500 |
| `prompt_last|L26|random_alpha2` | random | 21 | 0/21 | 6.424 | -0.012 | -0.012 | -0.297 |
| `prompt_last|L26|wrong_alpha2` | wrong | 21 | 0/21 | 1.557 | -0.036 | -0.036 | -0.021 |
| `query_relation|L19|repair_alpha2` | repair | 21 | 0/21 | 0.353 | -0.026 | -0.026 | 0.140 |
| `query_relation|L19|random_alpha2` | random | 21 | 0/21 | -0.521 | -0.083 | -0.083 | -0.409 |
| `query_relation|L19|wrong_alpha2` | wrong | 21 | 0/21 | -0.454 | -0.015 | -0.015 | -0.396 |

### DS7B watched trajectories

| key | kind | hidden | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|repair_alpha2|H27` | repair | H27 | 21 | 3.643 | 0.877 | -2.766 | 0.857 |
| `rule_value|L26|repair_alpha2|H28` | repair | H28 | 21 | 0.151 | 0.052 | -0.099 | 0.619 |
| `rule_value|L26|random_alpha2|H27` | random | H27 | 21 | 4.270 | 1.283 | -2.987 | 0.810 |
| `rule_value|L26|random_alpha2|H28` | random | H28 | 21 | 0.090 | 0.076 | -0.014 | 0.619 |
| `rule_value|L26|wrong_alpha2|H27` | wrong | H27 | 8 | 0.980 | -0.587 | -1.567 | 0.625 |
| `rule_value|L26|wrong_alpha2|H28` | wrong | H28 | 8 | 0.270 | 0.089 | -0.182 | 0.625 |
| `prompt_last|L26|repair_alpha2|H27` | repair | H27 | 21 | 4.402 | 2.172 | -2.231 | 0.810 |
| `prompt_last|L26|repair_alpha2|H28` | repair | H28 | 21 | 0.151 | 0.169 | 0.018 | 0.619 |
| `prompt_last|L26|random_alpha2|H27` | random | H27 | 21 | 6.382 | 1.838 | -4.544 | 0.905 |
| `prompt_last|L26|random_alpha2|H28` | random | H28 | 21 | 0.091 | 0.049 | -0.042 | 0.571 |
| `prompt_last|L26|wrong_alpha2|H27` | wrong | H27 | 21 | 1.519 | 0.329 | -1.190 | 0.571 |
| `prompt_last|L26|wrong_alpha2|H28` | wrong | H28 | 21 | -0.098 | -0.036 | 0.062 | 0.238 |
| `query_relation|L19|repair_alpha2|H20` | repair | H20 | 21 | 0.354 | 0.137 | -0.217 | 0.619 |
| `query_relation|L19|repair_alpha2|H21` | repair | H21 | 21 | 0.923 | 0.176 | -0.747 | 0.762 |
| `query_relation|L19|repair_alpha2|H22` | repair | H22 | 21 | 0.650 | -0.004 | -0.654 | 0.762 |
| `query_relation|L19|repair_alpha2|H23` | repair | H23 | 21 | 0.599 | -0.016 | -0.615 | 0.714 |
| `query_relation|L19|repair_alpha2|H28` | repair | H28 | 21 | -0.065 | -0.032 | 0.033 | 0.524 |
| `query_relation|L19|random_alpha2|H20` | random | H20 | 21 | -0.533 | -0.213 | 0.320 | 0.476 |
| `query_relation|L19|random_alpha2|H21` | random | H21 | 21 | 0.558 | 0.092 | -0.466 | 0.524 |
| `query_relation|L19|random_alpha2|H22` | random | H22 | 21 | 1.569 | 0.426 | -1.143 | 0.857 |
| `query_relation|L19|random_alpha2|H23` | random | H23 | 21 | 1.889 | 0.410 | -1.479 | 0.857 |
| `query_relation|L19|random_alpha2|H28` | random | H28 | 21 | 0.190 | 0.045 | -0.145 | 0.762 |
| `query_relation|L19|wrong_alpha2|H20` | wrong | H20 | 21 | -0.457 | -0.091 | 0.365 | 0.333 |
| `query_relation|L19|wrong_alpha2|H21` | wrong | H21 | 21 | 0.055 | 0.065 | 0.010 | 0.429 |
| `query_relation|L19|wrong_alpha2|H22` | wrong | H22 | 21 | 0.297 | 0.067 | -0.230 | 0.476 |
| `query_relation|L19|wrong_alpha2|H23` | wrong | H23 | 21 | 0.259 | 0.095 | -0.165 | 0.429 |
| `query_relation|L19|wrong_alpha2|H28` | wrong | H28 | 21 | -0.136 | -0.042 | 0.094 | 0.238 |

