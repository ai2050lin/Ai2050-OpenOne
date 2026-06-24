# Phase597 Cross-Model Summary

State-conditioned MLP input interpolation and recomputation audit.

## qwen3

cases=64, rows=5, target_cases_seen=5, alphas=[0.25, 0.5, 1.0, 1.5, 2.0], time_min=1.07

### Best causal state patches

| key | kind | alpha | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `query_category|L32|repair_alpha2` | repair | 2.000 | 5 | 2/5 | 0.050 | 0.050 | -0.012 | 0.019 | -0.031 |
| `prompt_last|L34|wrong_alpha2` | wrong | 2.000 | 5 | 1/5 | 0.100 | 0.100 | -3.004 | 0.056 | -0.044 |
| `query_category|L32|random_alpha1.5` | random | 1.500 | 5 | 1/5 | 0.075 | 0.075 | -0.008 | 0.051 | -0.024 |
| `prompt_last|L34|wrong_alpha1.5` | wrong | 1.500 | 5 | 1/5 | 0.075 | 0.075 | -1.675 | 0.062 | -0.013 |
| `query_category|L32|repair_alpha0.25` | repair | 0.250 | 5 | 1/5 | 0.075 | 0.075 | -0.042 | 0.049 | -0.026 |
| `prompt_last|L32|repair_alpha2` | repair | 2.000 | 5 | 1/5 | 0.075 | 0.075 | 0.299 | 0.055 | -0.019 |
| `query_category|L32|wrong_alpha2` | wrong | 2.000 | 5 | 1/5 | 0.050 | 0.050 | -0.018 | 0.044 | -0.006 |
| `prompt_last|L32|wrong_alpha2` | wrong | 2.000 | 5 | 1/5 | 0.050 | 0.050 | -0.010 | 0.019 | -0.031 |
| `prompt_last|L34|wrong_alpha0.25` | wrong | 0.250 | 5 | 1/5 | 0.050 | 0.050 | -0.177 | 0.032 | -0.018 |
| `prompt_last|L34|wrong_alpha1` | wrong | 1.000 | 5 | 1/5 | 0.050 | 0.050 | -0.816 | 0.032 | -0.018 |
| `prompt_last|L32|random_alpha0.5` | random | 0.500 | 5 | 1/5 | 0.050 | 0.050 | 0.014 | 0.013 | -0.037 |
| `prompt_last|L34|random_alpha0.5` | random | 0.500 | 5 | 1/5 | 0.050 | 0.050 | -0.063 | 0.032 | -0.018 |
| `query_category|L32|wrong_alpha1.5` | wrong | 1.500 | 5 | 1/5 | 0.050 | 0.050 | -0.050 | 0.031 | -0.019 |
| `prompt_last|L32|repair_alpha1` | repair | 1.000 | 5 | 1/5 | 0.050 | 0.050 | 0.193 | 0.030 | -0.020 |

### Best generated projections

| key | kind | alpha | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `query_category|L32|wrong_alpha2|generated_down` | wrong | 2.000 | 5 | 1.485 | 0.995 | -0.490 | 1.000 |
| `query_category|L32|random_alpha2|generated_down` | random | 2.000 | 5 | 1.217 | 0.996 | -0.221 | 0.800 |
| `query_category|L32|repair_alpha2|generated_down` | repair | 2.000 | 5 | 1.211 | 0.935 | -0.276 | 1.000 |
| `prompt_last|L34|wrong_alpha1|generated_down` | wrong | 1.000 | 5 | 0.943 | 0.084 | -0.859 | 0.800 |
| `query_category|L32|wrong_alpha1.5|generated_down` | wrong | 1.500 | 5 | 0.938 | 0.513 | -0.425 | 0.800 |
| `query_category|L32|random_alpha1.5|generated_down` | random | 1.500 | 5 | 0.801 | 0.614 | -0.187 | 1.000 |
| `query_category|L32|repair_alpha1.5|generated_down` | repair | 1.500 | 5 | 0.790 | 0.503 | -0.287 | 0.800 |
| `prompt_last|L34|wrong_alpha1.5|generated_down` | wrong | 1.500 | 5 | 0.768 | 0.115 | -0.653 | 0.800 |
| `prompt_last|L34|wrong_alpha0.5|generated_down` | wrong | 0.500 | 5 | 0.736 | 0.084 | -0.652 | 0.800 |
| `prompt_last|L34|repair_alpha1|generated_down` | repair | 1.000 | 5 | 0.668 | 0.146 | -0.522 | 0.800 |
| `prompt_last|L34|repair_alpha1.5|generated_down` | repair | 1.500 | 5 | 0.616 | 0.260 | -0.356 | 0.800 |
| `prompt_last|L34|repair_alpha2|generated_down` | repair | 2.000 | 5 | 0.558 | 0.468 | -0.090 | 0.800 |
| `prompt_last|L32|wrong_alpha2|generated_down` | wrong | 2.000 | 5 | 0.544 | 0.293 | -0.250 | 0.800 |
| `prompt_last|L34|repair_alpha0.5|generated_down` | repair | 0.500 | 5 | 0.534 | 0.097 | -0.437 | 0.800 |

## glm4

cases=64, rows=4, target_cases_seen=4, alphas=[0.25, 0.5, 1.0, 1.5, 2.0], time_min=1.55

### Best causal state patches

| key | kind | alpha | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L38|random_alpha2` | random | 2.000 | 4 | 0/4 | 0.016 | 0.016 | -0.480 | 0.004 | -0.012 |
| `prompt_last|L37|wrong_alpha1` | wrong | 1.000 | 4 | 0/4 | 0.000 | 0.000 | -0.110 | 0.014 | 0.014 |
| `prompt_last|L38|wrong_alpha0.25` | wrong | 0.250 | 4 | 0/4 | 0.000 | 0.000 | -0.072 | 0.000 | -0.000 |
| `prompt_last|L39|random_alpha0.25` | random | 0.250 | 4 | 0/4 | 0.000 | 0.000 | 0.015 | 0.000 | 0.000 |
| `prompt_last|L39|random_alpha0.5` | random | 0.500 | 4 | 0/4 | 0.000 | 0.000 | 0.024 | 0.000 | 0.000 |
| `prompt_last|L39|random_alpha1` | random | 1.000 | 4 | 0/4 | 0.000 | 0.000 | 0.013 | 0.000 | 0.000 |
| `prompt_last|L39|random_alpha1.5` | random | 1.500 | 4 | 0/4 | 0.000 | 0.000 | -0.029 | 0.000 | 0.000 |
| `prompt_last|L39|random_alpha2` | random | 2.000 | 4 | 0/4 | 0.000 | 0.000 | -0.114 | 0.000 | 0.000 |
| `prompt_last|L39|repair_alpha0.25` | repair | 0.250 | 4 | 0/4 | 0.000 | 0.000 | -0.036 | 0.000 | 0.000 |
| `prompt_last|L39|repair_alpha0.5` | repair | 0.500 | 4 | 0/4 | 0.000 | 0.000 | -0.097 | 0.000 | 0.000 |
| `prompt_last|L39|repair_alpha1` | repair | 1.000 | 4 | 0/4 | 0.000 | 0.000 | -0.146 | 0.000 | 0.000 |
| `prompt_last|L39|repair_alpha1.5` | repair | 1.500 | 4 | 0/4 | 0.000 | 0.000 | -0.148 | 0.000 | 0.000 |
| `prompt_last|L39|repair_alpha2` | repair | 2.000 | 4 | 0/4 | 0.000 | 0.000 | -0.174 | 0.000 | 0.000 |
| `prompt_last|L39|wrong_alpha0.25` | wrong | 0.250 | 4 | 0/4 | 0.000 | 0.000 | 0.136 | 0.000 | 0.000 |

### Best generated projections

| key | kind | alpha | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `prompt_last|L38|repair_alpha1.5|generated_down` | repair | 1.500 | 4 | 0.336 | 0.113 | -0.223 | 0.500 |
| `prompt_last|L38|repair_alpha1|generated_down` | repair | 1.000 | 4 | 0.329 | 0.122 | -0.207 | 0.750 |
| `prompt_last|L38|repair_alpha2|generated_down` | repair | 2.000 | 4 | 0.245 | 0.064 | -0.181 | 0.250 |
| `prompt_last|L38|repair_alpha0.5|generated_down` | repair | 0.500 | 4 | 0.197 | 0.079 | -0.118 | 0.750 |
| `prompt_last|L37|random_alpha2|generated_down` | random | 2.000 | 4 | 0.178 | 0.077 | -0.101 | 1.000 |
| `prompt_last|L37|random_alpha1.5|generated_down` | random | 1.500 | 4 | 0.108 | 0.046 | -0.062 | 1.000 |
| `prompt_last|L38|repair_alpha0.25|generated_down` | repair | 0.250 | 4 | 0.102 | 0.042 | -0.059 | 0.750 |
| `prompt_last|L37|repair_alpha1.5|generated_down` | repair | 1.500 | 4 | 0.097 | 0.066 | -0.031 | 0.750 |
| `prompt_last|L39|wrong_alpha2|generated_down` | wrong | 2.000 | 4 | 0.096 | 0.028 | -0.067 | 0.750 |
| `prompt_last|L37|repair_alpha1|generated_down` | repair | 1.000 | 4 | 0.094 | 0.059 | -0.035 | 0.750 |
| `prompt_last|L39|wrong_alpha1.5|generated_down` | wrong | 1.500 | 4 | 0.063 | 0.013 | -0.050 | 0.750 |
| `prompt_last|L37|repair_alpha2|generated_down` | repair | 2.000 | 4 | 0.062 | 0.050 | -0.013 | 0.500 |
| `prompt_last|L37|random_alpha1|generated_down` | random | 1.000 | 4 | 0.057 | 0.023 | -0.034 | 1.000 |
| `prompt_last|L37|repair_alpha0.5|generated_down` | repair | 0.500 | 4 | 0.055 | 0.032 | -0.022 | 1.000 |

## deepseek7b

cases=64, rows=21, target_cases_seen=21, alphas=[0.25, 0.5, 1.0, 1.5, 2.0], time_min=6.68

### Best causal state patches

| key | kind | alpha | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `query_relation|L19|random_alpha1.5` | random | 1.500 | 21 | 1/21 | -0.050 | -0.050 | -0.209 | -0.037 | 0.013 |
| `rule_value|L26|wrong_alpha1` | wrong | 1.000 | 8 | 0/8 | 0.016 | 0.016 | 0.003 | 0.009 | -0.007 |
| `query_relation|L19|wrong_alpha1.5` | wrong | 1.500 | 21 | 0/21 | 0.005 | 0.005 | -0.186 | 0.030 | 0.026 |
| `rule_value|L26|repair_alpha0.25` | repair | 0.250 | 21 | 0/21 | 0.000 | 0.000 | -0.007 | 0.001 | 0.001 |
| `rule_value|L26|wrong_alpha1.5` | wrong | 1.500 | 8 | 0/8 | -0.000 | -0.000 | 0.014 | -0.007 | -0.007 |
| `rule_value|L26|wrong_alpha0.5` | wrong | 0.500 | 8 | 0/8 | -0.001 | -0.001 | 0.015 | -0.002 | -0.001 |
| `rule_value|L26|repair_alpha0.5` | repair | 0.500 | 21 | 0/21 | -0.002 | -0.002 | 0.007 | -0.002 | -0.000 |
| `query_relation|L19|repair_alpha1` | repair | 1.000 | 21 | 0/21 | -0.002 | -0.002 | 0.221 | 0.008 | 0.011 |
| `rule_value|L26|random_alpha0.25` | random | 0.250 | 21 | 0/21 | -0.004 | -0.004 | 0.001 | -0.003 | 0.001 |
| `rule_value|L26|random_alpha1.5` | random | 1.500 | 21 | 0/21 | -0.006 | -0.006 | 0.015 | 0.003 | 0.008 |
| `rule_value|L26|repair_alpha1.5` | repair | 1.500 | 21 | 0/21 | -0.006 | -0.006 | -0.001 | 0.002 | 0.008 |
| `rule_value|L26|repair_alpha2` | repair | 2.000 | 21 | 0/21 | -0.006 | -0.006 | 0.004 | -0.004 | 0.002 |
| `query_relation|L19|repair_alpha0.5` | repair | 0.500 | 21 | 0/21 | -0.008 | -0.008 | 0.076 | 0.004 | 0.012 |
| `query_relation|L19|repair_alpha0.25` | repair | 0.250 | 21 | 0/21 | -0.008 | -0.008 | 0.024 | -0.003 | 0.005 |

### Best generated projections

| key | kind | alpha | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `prompt_last|L26|random_alpha2|generated_down` | random | 2.000 | 21 | 6.424 | 1.849 | -4.575 | 0.905 |
| `prompt_last|L26|repair_alpha2|generated_down` | repair | 2.000 | 21 | 4.414 | 2.177 | -2.237 | 0.810 |
| `rule_value|L26|random_alpha2|generated_down` | random | 2.000 | 21 | 4.251 | 1.272 | -2.979 | 0.762 |
| `prompt_last|L26|random_alpha1.5|generated_down` | random | 1.500 | 21 | 4.059 | 1.213 | -2.846 | 0.905 |
| `rule_value|L26|repair_alpha2|generated_down` | repair | 2.000 | 21 | 3.618 | 0.867 | -2.752 | 0.857 |
| `rule_value|L26|random_alpha1.5|generated_down` | random | 1.500 | 21 | 2.690 | 0.895 | -1.794 | 0.762 |
| `rule_value|L26|repair_alpha1.5|generated_down` | repair | 1.500 | 21 | 2.207 | 0.529 | -1.677 | 0.857 |
| `prompt_last|L26|random_alpha1|generated_down` | random | 1.000 | 21 | 2.169 | 0.693 | -1.476 | 0.905 |
| `prompt_last|L26|repair_alpha1.5|generated_down` | repair | 1.500 | 21 | 1.948 | 1.212 | -0.736 | 0.762 |
| `prompt_last|L26|wrong_alpha2|generated_down` | wrong | 2.000 | 21 | 1.557 | 0.345 | -1.212 | 0.571 |
| `rule_value|L26|random_alpha1|generated_down` | random | 1.000 | 21 | 1.453 | 0.560 | -0.893 | 0.762 |
| `rule_value|L26|wrong_alpha1.5|generated_down` | wrong | 1.500 | 8 | 1.277 | -0.059 | -1.336 | 0.625 |
| `rule_value|L26|wrong_alpha1|generated_down` | wrong | 1.000 | 8 | 1.223 | 0.210 | -1.013 | 0.625 |
| `rule_value|L26|repair_alpha1|generated_down` | repair | 1.000 | 21 | 1.212 | 0.305 | -0.908 | 0.762 |

### DS7B watched causal patches

| key | kind | alpha | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|repair_alpha0.25` | repair | 0.250 | 21 | 0/21 | 0.000 | 0.000 | -0.007 | 0.001 | 0.001 |
| `query_relation|L19|repair_alpha0.25` | repair | 0.250 | 21 | 0/21 | -0.008 | -0.008 | 0.024 | -0.003 | 0.005 |
| `rule_value|L26|wrong_alpha0.25` | wrong | 0.250 | 8 | 0/8 | -0.015 | -0.015 | 0.011 | -0.007 | 0.008 |
| `query_relation|L19|wrong_alpha0.25` | wrong | 0.250 | 21 | 0/21 | -0.019 | -0.019 | 0.045 | 0.001 | 0.021 |
| `rule_value|L26|random_alpha0.25` | random | 0.250 | 21 | 0/21 | -0.004 | -0.004 | 0.001 | -0.003 | 0.001 |
| `query_relation|L19|random_alpha0.25` | random | 0.250 | 21 | 0/21 | -0.020 | -0.020 | 0.002 | -0.001 | 0.019 |
| `rule_value|L26|repair_alpha0.5` | repair | 0.500 | 21 | 0/21 | -0.002 | -0.002 | 0.007 | -0.002 | -0.000 |
| `query_relation|L19|repair_alpha0.5` | repair | 0.500 | 21 | 0/21 | -0.008 | -0.008 | 0.076 | 0.004 | 0.012 |
| `rule_value|L26|wrong_alpha0.5` | wrong | 0.500 | 8 | 0/8 | -0.001 | -0.001 | 0.015 | -0.002 | -0.001 |
| `query_relation|L19|wrong_alpha0.5` | wrong | 0.500 | 21 | 0/21 | -0.023 | -0.023 | 0.062 | 0.003 | 0.026 |
| `rule_value|L26|random_alpha0.5` | random | 0.500 | 21 | 0/21 | -0.009 | -0.009 | -0.002 | -0.008 | 0.001 |
| `query_relation|L19|random_alpha0.5` | random | 0.500 | 21 | 0/21 | -0.015 | -0.015 | -0.019 | 0.003 | 0.018 |
| `rule_value|L26|repair_alpha1` | repair | 1.000 | 21 | 0/21 | -0.015 | -0.015 | 0.003 | -0.012 | 0.003 |
| `query_relation|L19|repair_alpha1` | repair | 1.000 | 21 | 0/21 | -0.002 | -0.002 | 0.221 | 0.008 | 0.011 |
| `rule_value|L26|wrong_alpha1` | wrong | 1.000 | 8 | 0/8 | 0.016 | 0.016 | 0.003 | 0.009 | -0.007 |
| `query_relation|L19|wrong_alpha1` | wrong | 1.000 | 21 | 0/21 | -0.010 | -0.010 | -0.010 | 0.017 | 0.027 |
| `rule_value|L26|random_alpha1` | random | 1.000 | 21 | 0/21 | -0.009 | -0.009 | 0.002 | -0.003 | 0.006 |
| `query_relation|L19|random_alpha1` | random | 1.000 | 21 | 0/21 | -0.042 | -0.042 | -0.073 | -0.005 | 0.037 |
| `rule_value|L26|repair_alpha1.5` | repair | 1.500 | 21 | 0/21 | -0.006 | -0.006 | -0.001 | 0.002 | 0.008 |
| `query_relation|L19|repair_alpha1.5` | repair | 1.500 | 21 | 0/21 | -0.030 | -0.030 | 0.285 | 0.002 | 0.032 |
| `rule_value|L26|wrong_alpha1.5` | wrong | 1.500 | 8 | 0/8 | -0.000 | -0.000 | 0.014 | -0.007 | -0.007 |
| `query_relation|L19|wrong_alpha1.5` | wrong | 1.500 | 21 | 0/21 | 0.005 | 0.005 | -0.186 | 0.030 | 0.026 |
| `rule_value|L26|random_alpha1.5` | random | 1.500 | 21 | 0/21 | -0.006 | -0.006 | 0.015 | 0.003 | 0.008 |
| `query_relation|L19|random_alpha1.5` | random | 1.500 | 21 | 1/21 | -0.050 | -0.050 | -0.209 | -0.037 | 0.013 |
| `rule_value|L26|repair_alpha2` | repair | 2.000 | 21 | 0/21 | -0.006 | -0.006 | 0.004 | -0.004 | 0.002 |
| `query_relation|L19|repair_alpha2` | repair | 2.000 | 21 | 0/21 | -0.026 | -0.026 | 0.140 | -0.006 | 0.021 |
| `rule_value|L26|wrong_alpha2` | wrong | 2.000 | 8 | 0/8 | -0.016 | -0.016 | 0.013 | -0.009 | 0.006 |
| `query_relation|L19|wrong_alpha2` | wrong | 2.000 | 21 | 0/21 | -0.015 | -0.015 | -0.396 | -0.001 | 0.014 |
| `rule_value|L26|random_alpha2` | random | 2.000 | 21 | 0/21 | -0.012 | -0.012 | 0.008 | -0.005 | 0.007 |
| `query_relation|L19|random_alpha2` | random | 2.000 | 21 | 0/21 | -0.083 | -0.083 | -0.409 | -0.056 | 0.027 |

### DS7B watched generated projections

| key | kind | alpha | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|repair_alpha0.25|generated_down` | repair | 0.250 | 21 | 0.207 | 0.075 | -0.132 | 0.667 |
| `query_relation|L19|repair_alpha0.25|generated_down` | repair | 0.250 | 21 | 0.196 | 0.077 | -0.119 | 0.857 |
| `rule_value|L26|wrong_alpha0.25|generated_down` | wrong | 0.250 | 8 | 0.418 | 0.125 | -0.293 | 0.750 |
| `query_relation|L19|wrong_alpha0.25|generated_down` | wrong | 0.250 | 21 | 0.059 | 0.041 | -0.018 | 0.619 |
| `rule_value|L26|random_alpha0.25|generated_down` | random | 0.250 | 21 | 0.245 | 0.131 | -0.113 | 0.667 |
| `query_relation|L19|random_alpha0.25|generated_down` | random | 0.250 | 21 | -0.012 | -0.019 | -0.007 | 0.524 |
| `rule_value|L26|repair_alpha0.5|generated_down` | repair | 0.500 | 21 | 0.491 | 0.150 | -0.340 | 0.667 |
| `query_relation|L19|repair_alpha0.5|generated_down` | repair | 0.500 | 21 | 0.344 | 0.133 | -0.211 | 0.857 |
| `rule_value|L26|wrong_alpha0.5|generated_down` | wrong | 0.500 | 8 | 0.773 | 0.211 | -0.562 | 0.750 |
| `query_relation|L19|wrong_alpha0.5|generated_down` | wrong | 0.500 | 21 | 0.101 | 0.067 | -0.034 | 0.476 |
| `rule_value|L26|random_alpha0.5|generated_down` | random | 0.500 | 21 | 0.565 | 0.266 | -0.298 | 0.667 |
| `query_relation|L19|random_alpha0.5|generated_down` | random | 0.500 | 21 | -0.035 | -0.042 | -0.007 | 0.476 |
| `rule_value|L26|repair_alpha1|generated_down` | repair | 1.000 | 21 | 1.212 | 0.305 | -0.908 | 0.762 |
| `query_relation|L19|repair_alpha1|generated_down` | repair | 1.000 | 21 | 0.518 | 0.194 | -0.325 | 0.857 |
| `rule_value|L26|wrong_alpha1|generated_down` | wrong | 1.000 | 8 | 1.223 | 0.210 | -1.013 | 0.625 |
| `query_relation|L19|wrong_alpha1|generated_down` | wrong | 1.000 | 21 | 0.095 | 0.071 | -0.024 | 0.524 |
| `rule_value|L26|random_alpha1|generated_down` | random | 1.000 | 21 | 1.453 | 0.560 | -0.893 | 0.762 |
| `query_relation|L19|random_alpha1|generated_down` | random | 1.000 | 21 | -0.121 | -0.094 | 0.027 | 0.429 |
| `rule_value|L26|repair_alpha1.5|generated_down` | repair | 1.500 | 21 | 2.207 | 0.529 | -1.677 | 0.857 |
| `query_relation|L19|repair_alpha1.5|generated_down` | repair | 1.500 | 21 | 0.527 | 0.194 | -0.333 | 0.857 |
| `rule_value|L26|wrong_alpha1.5|generated_down` | wrong | 1.500 | 8 | 1.277 | -0.059 | -1.336 | 0.625 |
| `query_relation|L19|wrong_alpha1.5|generated_down` | wrong | 1.500 | 21 | -0.086 | 0.012 | 0.098 | 0.476 |
| `rule_value|L26|random_alpha1.5|generated_down` | random | 1.500 | 21 | 2.690 | 0.895 | -1.794 | 0.762 |
| `query_relation|L19|random_alpha1.5|generated_down` | random | 1.500 | 21 | -0.277 | -0.149 | 0.128 | 0.476 |
| `rule_value|L26|repair_alpha2|generated_down` | repair | 2.000 | 21 | 3.618 | 0.867 | -2.752 | 0.857 |
| `query_relation|L19|repair_alpha2|generated_down` | repair | 2.000 | 21 | 0.353 | 0.136 | -0.217 | 0.571 |
| `rule_value|L26|wrong_alpha2|generated_down` | wrong | 2.000 | 8 | 0.973 | -0.593 | -1.566 | 0.625 |
| `query_relation|L19|wrong_alpha2|generated_down` | wrong | 2.000 | 21 | -0.454 | -0.091 | 0.363 | 0.333 |
| `rule_value|L26|random_alpha2|generated_down` | random | 2.000 | 21 | 4.251 | 1.272 | -2.979 | 0.762 |
| `query_relation|L19|random_alpha2|generated_down` | random | 2.000 | 21 | -0.521 | -0.207 | 0.314 | 0.476 |

