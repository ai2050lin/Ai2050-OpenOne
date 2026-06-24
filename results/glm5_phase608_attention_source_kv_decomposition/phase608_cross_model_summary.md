# Phase608 Cross-Model Summary

Attention source-token K/V decomposition.

## qwen3

cases=96, rows=7, target_cases_seen=7, layers=[29], time_min=0.66

### Best Patches

| key | layer | group | mode | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L29|random_position|k_delta` | L29 | random_position | k_delta | 2/7 | -0.018 | -0.024 | -0.006 |
| `L29|answer_prefix|k_delta` | L29 | answer_prefix | k_delta | 1/7 | 0.054 | 0.033 | -0.021 |
| `L29|query_object|k_delta` | L29 | query_object | k_delta | 1/7 | 0.054 | -0.027 | -0.080 |
| `L29|query_relation|v_delta` | L29 | query_relation | v_delta | 1/7 | 0.036 | -0.003 | -0.039 |
| `L29|query_object|v_delta` | L29 | query_object | v_delta | 1/7 | 0.036 | -0.009 | -0.044 |
| `L29|rule_value|k_delta` | L29 | rule_value | k_delta | 1/7 | 0.018 | -0.046 | -0.064 |
| `L29|query_category|k_delta` | L29 | query_category | k_delta | 1/7 | 0.018 | -0.015 | -0.033 |
| `L29|answer_prefix|v_delta` | L29 | answer_prefix | v_delta | 1/7 | 0.018 | 0.005 | -0.013 |
| `L29|rule_relation|k_delta` | L29 | rule_relation | k_delta | 1/7 | 0.018 | -0.029 | -0.047 |
| `L29|query_category|v_delta` | L29 | query_category | v_delta | 1/7 | -0.000 | -0.046 | -0.046 |
| `L29|random_position|kv_delta` | L29 | random_position | kv_delta | 1/7 | -0.035 | -0.027 | 0.009 |
| `L29|random_position|v_delta` | L29 | random_position | v_delta | 0/7 | 0.058 | -0.167 | -0.225 |
| `L29|prompt_last|kv_random` | L29 | prompt_last | kv_random | 0/7 | 0.020 | -0.031 | -0.051 |
| `L29|query_relation|kv_random` | L29 | query_relation | kv_random | 0/7 | 0.019 | -0.031 | -0.050 |
| `L29|prompt_last|kv_delta` | L29 | prompt_last | kv_delta | 0/7 | 0.018 | -0.046 | -0.064 |
| `L29|rule_value|v_delta` | L29 | rule_value | v_delta | 0/7 | 0.018 | -0.013 | -0.030 |
| `L29|query_object|kv_delta` | L29 | query_object | kv_delta | 0/7 | 0.018 | -0.027 | -0.045 |
| `L29|answer_prefix|kv_random` | L29 | answer_prefix | kv_random | 0/7 | 0.013 | 0.013 | -0.001 |
| `L29|query_object|kv_random` | L29 | query_object | kv_random | 0/7 | 0.012 | -0.022 | -0.034 |
| `L29|rule_value|kv_delta` | L29 | rule_value | kv_delta | 0/7 | 0.000 | -0.067 | -0.067 |
| `L29|query_relation|k_delta` | L29 | query_relation | k_delta | 0/7 | 0.000 | -0.033 | -0.033 |
| `L29|rule_relation|kv_delta` | L29 | rule_relation | kv_delta | 0/7 | 0.000 | -0.030 | -0.030 |
| `L29|query_relation|kv_delta` | L29 | query_relation | kv_delta | 0/7 | -0.000 | -0.036 | -0.036 |
| `L29|prompt_last|k_delta` | L29 | prompt_last | k_delta | 0/7 | -0.000 | -0.042 | -0.042 |
| `L29|query_category|kv_delta` | L29 | query_category | kv_delta | 0/7 | -0.000 | -0.050 | -0.050 |
| `L29|rule_relation|v_delta` | L29 | rule_relation | v_delta | 0/7 | -0.000 | -0.022 | -0.022 |
| `L29|rule_value|kv_random` | L29 | rule_value | kv_random | 0/7 | -0.011 | -0.046 | -0.035 |
| `L29|query_category|kv_random` | L29 | query_category | kv_random | 0/7 | -0.013 | -0.037 | -0.024 |
| `L29|answer_prefix|kv_delta` | L29 | answer_prefix | kv_delta | 0/7 | -0.018 | -0.016 | 0.002 |
| `L29|prompt_last|v_delta` | L29 | prompt_last | v_delta | 0/7 | -0.018 | -0.079 | -0.061 |
| `L29|rule_relation|kv_random` | L29 | rule_relation | kv_random | 0/7 | -0.031 | -0.046 | -0.015 |
| `L29|random_position|kv_random` | L29 | random_position | kv_random | 0/7 | -0.061 | -0.022 | 0.039 |

### Group Mode Grid

#### L29

| group | v_delta | k_delta | kv_delta | kv_random |
|---|---:|---:|---:|---:|
| rule_value | 0/7 (0.018) | 1/7 (0.018) | 0/7 (0.000) | 0/7 (-0.011) |
| rule_relation | 0/7 (-0.000) | 1/7 (0.018) | 0/7 (0.000) | 0/7 (-0.031) |
| query_relation | 1/7 (0.036) | 0/7 (0.000) | 0/7 (-0.000) | 0/7 (0.019) |
| query_category | 1/7 (-0.000) | 1/7 (0.018) | 0/7 (-0.000) | 0/7 (-0.013) |
| query_object | 1/7 (0.036) | 1/7 (0.054) | 0/7 (0.018) | 0/7 (0.012) |
| prompt_last | 0/7 (-0.018) | 0/7 (-0.000) | 0/7 (0.018) | 0/7 (0.020) |
| answer_prefix | 1/7 (0.018) | 1/7 (0.054) | 0/7 (-0.018) | 0/7 (0.013) |
| random_position | 0/7 (0.058) | 2/7 (-0.018) | 1/7 (-0.035) | 0/7 (-0.061) |

## glm4

cases=96, rows=13, target_cases_seen=13, layers=[34], time_min=1.51

### Best Patches

| key | layer | group | mode | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L34|rule_value|kv_delta` | L34 | rule_value | kv_delta | 1/13 | 0.014 | -0.004 | -0.018 |
| `L34|rule_value|v_delta` | L34 | rule_value | v_delta | 1/13 | 0.005 | -0.013 | -0.017 |
| `L34|rule_value|k_delta` | L34 | rule_value | k_delta | 1/13 | 0.000 | -0.005 | -0.005 |
| `L34|rule_value|kv_random` | L34 | rule_value | kv_random | 1/13 | -0.030 | -0.029 | 0.001 |
| `L34|query_object|kv_random` | L34 | query_object | kv_random | 0/13 | 0.020 | 0.011 | -0.008 |
| `L34|query_object|v_delta` | L34 | query_object | v_delta | 0/13 | 0.014 | 0.007 | -0.007 |
| `L34|prompt_last|k_delta` | L34 | prompt_last | k_delta | 0/13 | 0.010 | -0.000 | -0.010 |
| `L34|query_relation|k_delta` | L34 | query_relation | k_delta | 0/13 | 0.010 | -0.003 | -0.013 |
| `L34|rule_relation|v_delta` | L34 | rule_relation | v_delta | 0/13 | 0.005 | -0.011 | -0.015 |
| `L34|answer_prefix|kv_delta` | L34 | answer_prefix | kv_delta | 0/13 | 0.005 | 0.006 | 0.001 |
| `L34|prompt_last|kv_random` | L34 | prompt_last | kv_random | 0/13 | 0.000 | -0.005 | -0.005 |
| `L34|random_position|kv_delta` | L34 | random_position | kv_delta | 0/13 | 0.000 | -0.000 | -0.000 |
| `L34|rule_relation|k_delta` | L34 | rule_relation | k_delta | 0/13 | 0.000 | -0.004 | -0.004 |
| `L34|rule_relation|kv_delta` | L34 | rule_relation | kv_delta | 0/13 | 0.000 | -0.011 | -0.011 |
| `L34|query_category|kv_random` | L34 | query_category | kv_random | 0/13 | -0.001 | -0.006 | -0.004 |
| `L34|query_relation|kv_random` | L34 | query_relation | kv_random | 0/13 | -0.003 | -0.011 | -0.009 |
| `L34|prompt_last|v_delta` | L34 | prompt_last | v_delta | 0/13 | -0.005 | -0.007 | -0.002 |
| `L34|query_category|v_delta` | L34 | query_category | v_delta | 0/13 | -0.005 | -0.015 | -0.010 |
| `L34|prompt_last|kv_delta` | L34 | prompt_last | kv_delta | 0/13 | -0.005 | -0.004 | 0.000 |
| `L34|random_position|v_delta` | L34 | random_position | v_delta | 0/13 | -0.005 | 0.003 | 0.007 |
| `L34|query_object|k_delta` | L34 | query_object | k_delta | 0/13 | -0.005 | -0.011 | -0.006 |
| `L34|query_relation|v_delta` | L34 | query_relation | v_delta | 0/13 | -0.005 | -0.005 | -0.001 |
| `L34|answer_prefix|kv_random` | L34 | answer_prefix | kv_random | 0/13 | -0.007 | -0.008 | -0.001 |
| `L34|rule_relation|kv_random` | L34 | rule_relation | kv_random | 0/13 | -0.007 | -0.000 | 0.007 |
| `L34|answer_prefix|v_delta` | L34 | answer_prefix | v_delta | 0/13 | -0.014 | -0.009 | 0.006 |
| `L34|random_position|k_delta` | L34 | random_position | k_delta | 0/13 | -0.014 | -0.021 | -0.006 |
| `L34|answer_prefix|k_delta` | L34 | answer_prefix | k_delta | 0/13 | -0.014 | -0.006 | 0.009 |
| `L34|query_category|k_delta` | L34 | query_category | k_delta | 0/13 | -0.014 | -0.012 | 0.002 |
| `L34|query_category|kv_delta` | L34 | query_category | kv_delta | 0/13 | -0.019 | -0.010 | 0.010 |
| `L34|query_object|kv_delta` | L34 | query_object | kv_delta | 0/13 | -0.019 | -0.017 | 0.002 |
| `L34|query_relation|kv_delta` | L34 | query_relation | kv_delta | 0/13 | -0.019 | -0.013 | 0.006 |
| `L34|random_position|kv_random` | L34 | random_position | kv_random | 0/13 | -0.022 | -0.045 | -0.024 |

### Group Mode Grid

#### L34

| group | v_delta | k_delta | kv_delta | kv_random |
|---|---:|---:|---:|---:|
| rule_value | 1/13 (0.005) | 1/13 (0.000) | 1/13 (0.014) | 1/13 (-0.030) |
| rule_relation | 0/13 (0.005) | 0/13 (0.000) | 0/13 (0.000) | 0/13 (-0.007) |
| query_relation | 0/13 (-0.005) | 0/13 (0.010) | 0/13 (-0.019) | 0/13 (-0.003) |
| query_category | 0/13 (-0.005) | 0/13 (-0.014) | 0/13 (-0.019) | 0/13 (-0.001) |
| query_object | 0/13 (0.014) | 0/13 (-0.005) | 0/13 (-0.019) | 0/13 (0.020) |
| prompt_last | 0/13 (-0.005) | 0/13 (0.010) | 0/13 (-0.005) | 0/13 (0.000) |
| answer_prefix | 0/13 (-0.014) | 0/13 (-0.014) | 0/13 (0.005) | 0/13 (-0.007) |
| random_position | 0/13 (-0.005) | 0/13 (-0.014) | 0/13 (0.000) | 0/13 (-0.022) |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, layers=[22], time_min=2.81

### Best Patches

| key | layer | group | mode | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L22|rule_value|kv_random` | L22 | rule_value | kv_random | 1/37 | 0.115 | 0.037 | -0.078 |
| `L22|answer_prefix|v_delta` | L22 | answer_prefix | v_delta | 1/37 | 0.039 | 0.030 | -0.009 |
| `L22|query_object|kv_random` | L22 | query_object | kv_random | 1/37 | 0.036 | 0.019 | -0.017 |
| `L22|query_relation|kv_random` | L22 | query_relation | kv_random | 1/37 | 0.004 | -0.009 | -0.013 |
| `L22|rule_relation|v_delta` | L22 | rule_relation | v_delta | 1/37 | -0.005 | -0.027 | -0.021 |
| `L22|random_position|kv_random` | L22 | random_position | kv_random | 1/37 | -0.014 | -0.013 | 0.001 |
| `L22|query_category|k_delta` | L22 | query_category | k_delta | 1/37 | -0.019 | -0.016 | 0.002 |
| `L22|random_position|kv_delta` | L22 | random_position | kv_delta | 1/37 | -0.021 | -0.023 | -0.002 |
| `L22|prompt_last|kv_random` | L22 | prompt_last | kv_random | 0/37 | 0.019 | 0.012 | -0.007 |
| `L22|rule_relation|kv_random` | L22 | rule_relation | kv_random | 0/37 | 0.016 | 0.005 | -0.011 |
| `L22|answer_prefix|kv_delta` | L22 | answer_prefix | kv_delta | 0/37 | 0.005 | -0.001 | -0.006 |
| `L22|query_category|kv_random` | L22 | query_category | kv_random | 0/37 | 0.004 | 0.014 | 0.010 |
| `L22|answer_prefix|kv_random` | L22 | answer_prefix | kv_random | 0/37 | 0.002 | -0.011 | -0.013 |
| `L22|prompt_last|v_delta` | L22 | prompt_last | v_delta | 0/37 | -0.001 | -0.074 | -0.073 |
| `L22|rule_relation|k_delta` | L22 | rule_relation | k_delta | 0/37 | -0.003 | -0.000 | 0.002 |
| `L22|query_relation|kv_delta` | L22 | query_relation | kv_delta | 0/37 | -0.005 | 0.014 | 0.018 |
| `L22|random_position|k_delta` | L22 | random_position | k_delta | 0/37 | -0.005 | 0.013 | 0.018 |
| `L22|query_relation|v_delta` | L22 | query_relation | v_delta | 0/37 | -0.008 | 0.002 | 0.010 |
| `L22|rule_value|k_delta` | L22 | rule_value | k_delta | 0/37 | -0.012 | -0.007 | 0.004 |
| `L22|query_object|v_delta` | L22 | query_object | v_delta | 0/37 | -0.013 | 0.003 | 0.015 |
| `L22|random_position|v_delta` | L22 | random_position | v_delta | 0/37 | -0.015 | 0.014 | 0.028 |
| `L22|query_relation|k_delta` | L22 | query_relation | k_delta | 0/37 | -0.016 | 0.013 | 0.029 |
| `L22|rule_relation|kv_delta` | L22 | rule_relation | kv_delta | 0/37 | -0.016 | -0.027 | -0.011 |
| `L22|query_category|v_delta` | L22 | query_category | v_delta | 0/37 | -0.016 | -0.011 | 0.005 |
| `L22|rule_value|v_delta` | L22 | rule_value | v_delta | 0/37 | -0.017 | -0.011 | 0.006 |
| `L22|answer_prefix|k_delta` | L22 | answer_prefix | k_delta | 0/37 | -0.017 | -0.010 | 0.007 |
| `L22|query_object|k_delta` | L22 | query_object | k_delta | 0/37 | -0.017 | -0.005 | 0.012 |
| `L22|query_object|kv_delta` | L22 | query_object | kv_delta | 0/37 | -0.018 | 0.004 | 0.022 |
| `L22|prompt_last|k_delta` | L22 | prompt_last | k_delta | 0/37 | -0.018 | 0.065 | 0.083 |
| `L22|query_category|kv_delta` | L22 | query_category | kv_delta | 0/37 | -0.021 | -0.014 | 0.008 |
| `L22|prompt_last|kv_delta` | L22 | prompt_last | kv_delta | 0/37 | -0.025 | 0.010 | 0.034 |
| `L22|rule_value|kv_delta` | L22 | rule_value | kv_delta | 0/37 | -0.028 | -0.017 | 0.011 |

### Group Mode Grid

#### L22

| group | v_delta | k_delta | kv_delta | kv_random |
|---|---:|---:|---:|---:|
| rule_value | 0/37 (-0.017) | 0/37 (-0.012) | 0/37 (-0.028) | 1/37 (0.115) |
| rule_relation | 1/37 (-0.005) | 0/37 (-0.003) | 0/37 (-0.016) | 0/37 (0.016) |
| query_relation | 0/37 (-0.008) | 0/37 (-0.016) | 0/37 (-0.005) | 1/37 (0.004) |
| query_category | 0/37 (-0.016) | 1/37 (-0.019) | 0/37 (-0.021) | 0/37 (0.004) |
| query_object | 0/37 (-0.013) | 0/37 (-0.017) | 0/37 (-0.018) | 1/37 (0.036) |
| prompt_last | 0/37 (-0.001) | 0/37 (-0.018) | 0/37 (-0.025) | 0/37 (0.019) |
| answer_prefix | 1/37 (0.039) | 0/37 (-0.017) | 0/37 (0.005) | 0/37 (0.002) |
| random_position | 0/37 (-0.015) | 0/37 (-0.005) | 1/37 (-0.021) | 1/37 (-0.014) |

