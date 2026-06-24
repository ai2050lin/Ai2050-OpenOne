# Phase596 Cross-Model Summary

MLP internal gate/up/z/down path audit.

## qwen3

cases=64, rows=5, target_cases_seen=5, alpha=1.0, topks=[32, 128], time_min=0.57

### Best causal patches

| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L32|wrong_z_pair_raw` | 5 | 1/5 | 0.075 | 0.075 | 0.030 | 0.032 | -0.043 |
| `query_category|L32|z_pair_top32` | 5 | 1/5 | 0.050 | 0.050 | -0.006 | 0.037 | -0.013 |
| `prompt_last|L32|z_pair_top128` | 5 | 1/5 | 0.050 | 0.050 | 0.024 | 0.021 | -0.029 |
| `prompt_last|L34|wrong_z_pair_raw` | 5 | 1/5 | 0.050 | 0.050 | -0.794 | 0.018 | -0.032 |
| `query_category|L32|wrong_gate_up_pair_raw` | 5 | 1/5 | 0.050 | 0.050 | -0.035 | 0.037 | -0.013 |
| `query_category|L32|down_out_random` | 5 | 1/5 | 0.050 | 0.050 | -0.021 | 0.019 | -0.031 |
| `prompt_last|L32|z_pair_top32` | 5 | 1/5 | 0.050 | 0.050 | -0.050 | 0.037 | -0.013 |
| `query_category|L32|gate_up_pair_raw` | 5 | 1/5 | 0.050 | 0.050 | -0.012 | 0.012 | -0.038 |
| `prompt_last|L32|down_out_raw` | 5 | 1/5 | 0.050 | 0.050 | 0.196 | 0.019 | -0.031 |
| `prompt_last|L32|up_raw` | 5 | 1/5 | 0.050 | 0.050 | 0.092 | 0.036 | -0.014 |
| `query_category|L32|up_raw` | 5 | 1/5 | 0.025 | 0.025 | -0.014 | 0.019 | -0.006 |
| `prompt_last|L34|wrong_gate_up_pair_raw` | 5 | 1/5 | 0.025 | 0.025 | -0.839 | 0.018 | -0.007 |
| `query_category|L32|gate_raw` | 5 | 1/5 | 0.025 | 0.025 | -0.005 | 0.011 | -0.014 |
| `prompt_last|L32|gate_up_pair_random` | 5 | 1/5 | 0.025 | 0.025 | 0.068 | 0.012 | -0.013 |

### Best internal projections

| key | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---:|---:|---:|---:|---:|
| `prompt_last|L34|gate_only` | 5 | 0.980 | 0.246 | -0.734 | 0.800 |
| `prompt_last|L34|down_out` | 5 | 0.668 | 0.146 | -0.522 | 0.800 |
| `prompt_last|L34|z_pair` | 5 | 0.666 | 0.145 | -0.521 | 0.800 |
| `prompt_last|L34|gate_up_pair` | 5 | 0.666 | 0.145 | -0.520 | 0.800 |
| `query_category|L32|z_pair` | 5 | 0.445 | 0.225 | -0.220 | 0.800 |
| `query_category|L32|gate_up_pair` | 5 | 0.445 | 0.225 | -0.220 | 0.800 |
| `query_category|L32|down_out` | 5 | 0.445 | 0.225 | -0.220 | 0.800 |
| `query_category|L32|up_only` | 5 | 0.252 | 0.159 | -0.093 | 0.800 |
| `prompt_last|L32|gate_only` | 5 | 0.202 | 0.172 | -0.030 | 1.000 |
| `prompt_last|L32|down_out` | 5 | 0.150 | 0.092 | -0.058 | 1.000 |
| `prompt_last|L32|z_pair` | 5 | 0.150 | 0.092 | -0.058 | 1.000 |
| `prompt_last|L32|gate_up_pair` | 5 | 0.150 | 0.092 | -0.058 | 1.000 |

## glm4

cases=64, rows=4, target_cases_seen=4, alpha=1.0, topks=[32, 128], time_min=0.75

### Best causal patches

| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L37|gate_up_pair_random` | 4 | 0/4 | 0.000 | 0.000 | -0.012 | 0.010 | 0.010 |
| `prompt_last|L38|z_pair_top128` | 4 | 0/4 | 0.000 | 0.000 | 0.074 | 0.000 | -0.000 |
| `prompt_last|L38|z_pair_top32` | 4 | 0/4 | 0.000 | 0.000 | -0.014 | -0.000 | -0.000 |
| `prompt_last|L39|down_out_random` | 4 | 0/4 | 0.000 | 0.000 | -0.256 | 0.000 | 0.000 |
| `prompt_last|L39|down_out_raw` | 4 | 0/4 | 0.000 | 0.000 | -0.152 | 0.000 | 0.000 |
| `prompt_last|L39|gate_up_pair_random` | 4 | 0/4 | 0.000 | 0.000 | -0.165 | 0.000 | 0.000 |
| `prompt_last|L39|gate_up_pair_raw` | 4 | 0/4 | 0.000 | 0.000 | -0.133 | 0.000 | 0.000 |
| `prompt_last|L39|wrong_gate_up_pair_raw` | 4 | 0/4 | 0.000 | 0.000 | 0.164 | 0.000 | 0.000 |
| `prompt_last|L39|wrong_z_pair_raw` | 4 | 0/4 | 0.000 | 0.000 | 0.164 | 0.000 | 0.000 |
| `prompt_last|L39|z_pair_random` | 4 | 0/4 | 0.000 | 0.000 | -0.267 | 0.000 | 0.000 |
| `prompt_last|L39|z_pair_raw` | 4 | 0/4 | 0.000 | 0.000 | -0.157 | 0.000 | 0.000 |
| `prompt_last|L39|z_pair_top128` | 4 | 0/4 | 0.000 | 0.000 | -0.074 | 0.000 | 0.000 |
| `prompt_last|L39|z_pair_top32` | 4 | 0/4 | 0.000 | 0.000 | -0.089 | 0.000 | 0.000 |
| `prompt_last|L38|gate_up_pair_raw` | 4 | 0/4 | -0.016 | -0.016 | 0.131 | -0.012 | 0.004 |

### Best internal projections

| key | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---:|---:|---:|---:|---:|
| `prompt_last|L38|z_pair` | 4 | 0.330 | 0.123 | -0.207 | 0.750 |
| `prompt_last|L38|down_out` | 4 | 0.330 | 0.122 | -0.207 | 0.750 |
| `prompt_last|L38|gate_up_pair` | 4 | 0.329 | 0.122 | -0.207 | 0.750 |
| `prompt_last|L38|up_only` | 4 | 0.282 | 0.156 | -0.126 | 0.750 |
| `prompt_last|L38|gate_only` | 4 | 0.160 | 0.024 | -0.136 | 0.750 |
| `prompt_last|L37|gate_only` | 4 | 0.127 | 0.073 | -0.054 | 0.750 |
| `prompt_last|L37|down_out` | 4 | 0.094 | 0.059 | -0.036 | 0.750 |
| `prompt_last|L37|z_pair` | 4 | 0.094 | 0.058 | -0.035 | 0.750 |
| `prompt_last|L37|gate_up_pair` | 4 | 0.093 | 0.058 | -0.035 | 0.750 |
| `prompt_last|L39|gate_only` | 4 | 0.054 | 0.038 | -0.016 | 0.750 |
| `prompt_last|L39|down_out` | 4 | 0.047 | -0.018 | -0.064 | 0.500 |
| `prompt_last|L37|up_only` | 4 | 0.045 | 0.039 | -0.006 | 1.000 |

## deepseek7b

cases=64, rows=21, target_cases_seen=21, alpha=1.0, topks=[32, 128], time_min=2.38

### Best causal patches

| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|wrong_gate_up_pair_raw` | 8 | 0/8 | 0.007 | 0.007 | 0.011 | -0.006 | -0.014 |
| `query_relation|L19|z_pair_top128` | 21 | 0/21 | 0.007 | 0.007 | 0.085 | 0.013 | 0.006 |
| `query_relation|L19|wrong_z_pair_raw` | 21 | 0/21 | 0.006 | 0.006 | -0.019 | 0.032 | 0.025 |
| `query_relation|L19|gate_up_pair_raw` | 21 | 0/21 | 0.001 | 0.001 | 0.205 | 0.016 | 0.015 |
| `rule_value|L26|z_pair_top32` | 21 | 0/21 | -0.003 | -0.003 | -0.001 | -0.004 | -0.001 |
| `rule_value|L26|z_pair_top128` | 21 | 0/21 | -0.003 | -0.003 | 0.001 | -0.004 | -0.001 |
| `prompt_last|L26|gate_raw` | 21 | 0/21 | -0.003 | -0.003 | 1.605 | 0.000 | 0.003 |
| `rule_value|L26|gate_up_pair_random` | 21 | 0/21 | -0.006 | -0.006 | -0.001 | -0.004 | 0.002 |
| `rule_value|L26|wrong_z_pair_raw` | 8 | 0/8 | -0.008 | -0.008 | 0.000 | -0.013 | -0.005 |
| `rule_value|L26|up_raw` | 21 | 0/21 | -0.009 | -0.009 | 0.001 | 0.003 | 0.012 |
| `rule_value|L26|down_out_raw` | 21 | 0/21 | -0.009 | -0.009 | 0.000 | -0.009 | -0.000 |
| `rule_value|L26|z_pair_random` | 21 | 0/21 | -0.009 | -0.009 | 0.003 | -0.003 | 0.006 |
| `rule_value|L26|down_out_random` | 21 | 0/21 | -0.009 | -0.009 | -0.003 | -0.003 | 0.007 |
| `prompt_last|L26|z_pair_raw` | 21 | 0/21 | -0.012 | -0.012 | -0.022 | -0.004 | 0.008 |

### Best internal projections

| key | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---:|---:|---:|---:|---:|
| `rule_value|L26|z_pair` | 21 | 1.214 | 0.306 | -0.907 | 0.762 |
| `rule_value|L26|gate_up_pair` | 21 | 1.213 | 0.305 | -0.908 | 0.762 |
| `rule_value|L26|down_out` | 21 | 1.206 | 0.309 | -0.897 | 0.762 |
| `rule_value|L26|up_only` | 21 | 1.072 | 0.367 | -0.705 | 0.810 |
| `query_relation|L19|down_out` | 21 | 0.519 | 0.195 | -0.324 | 0.857 |
| `query_relation|L19|z_pair` | 21 | 0.519 | 0.194 | -0.324 | 0.857 |
| `query_relation|L19|gate_up_pair` | 21 | 0.519 | 0.194 | -0.325 | 0.857 |
| `prompt_last|L26|up_only` | 21 | 0.490 | 0.289 | -0.201 | 0.619 |
| `query_relation|L19|gate_only` | 21 | 0.425 | 0.163 | -0.262 | 0.857 |
| `query_relation|L19|up_only` | 21 | 0.401 | 0.158 | -0.243 | 0.857 |
| `prompt_last|L26|gate_only` | 21 | 0.238 | 0.258 | 0.020 | 0.429 |
| `prompt_last|L26|gate_up_pair` | 21 | 0.208 | 0.409 | 0.202 | 0.429 |

### DS7B watched keys

| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|gate_raw` | 21 | 0/21 | -0.018 | -0.018 | -0.007 | -0.003 | 0.015 |
| `rule_value|L26|up_raw` | 21 | 0/21 | -0.009 | -0.009 | 0.001 | 0.003 | 0.012 |
| `rule_value|L26|gate_up_pair_raw` | 21 | 0/21 | -0.021 | -0.021 | -0.005 | -0.009 | 0.012 |
| `rule_value|L26|z_pair_raw` | 21 | 0/21 | -0.021 | -0.021 | -0.003 | -0.007 | 0.013 |
| `rule_value|L26|z_pair_top32` | 21 | 0/21 | -0.003 | -0.003 | -0.001 | -0.004 | -0.001 |
| `rule_value|L26|z_pair_top128` | 21 | 0/21 | -0.003 | -0.003 | 0.001 | -0.004 | -0.001 |
| `rule_value|L26|wrong_z_pair_raw` | 8 | 0/8 | -0.008 | -0.008 | 0.000 | -0.013 | -0.005 |
| `query_relation|L19|gate_up_pair_raw` | 21 | 0/21 | 0.001 | 0.001 | 0.205 | 0.016 | 0.015 |
| `query_relation|L19|z_pair_raw` | 21 | 0/21 | -0.015 | -0.015 | 0.206 | 0.015 | 0.030 |

