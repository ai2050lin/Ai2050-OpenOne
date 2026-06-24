# Phase595 Cross-Model Summary

MLP update component causal patch validation.

## qwen3

cases=64, rows=5, target_cases_seen=5, alpha=1.0, time_min=0.50

| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L33|mlp|specific_norm_raw` | 5 | 2/5 | 0.125 | 0.125 | -0.014 | 0.057 | -0.068 |
| `prompt_last|L30|mlp|common_norm_raw` | 5 | 2/5 | 0.075 | 0.075 | -0.222 | 0.044 | -0.031 |
| `prompt_last|L32|mlp|specific_norm_raw` | 5 | 1/5 | 0.125 | 0.125 | 0.074 | 0.057 | -0.068 |
| `prompt_last|L34|mlp|specific_only` | 5 | 1/5 | 0.075 | 0.075 | -0.014 | 0.044 | -0.031 |
| `prompt_last|L33|mlp|raw` | 5 | 1/5 | 0.050 | 0.050 | 0.102 | 0.025 | -0.025 |
| `prompt_last|L33|mlp|common_norm_raw` | 5 | 1/5 | 0.050 | 0.050 | -1.005 | 0.007 | -0.043 |
| `query_category|L32|mlp|common_norm_raw` | 5 | 1/5 | 0.050 | 0.050 | -0.006 | 0.019 | -0.031 |
| `prompt_last|L33|mlp|common_only` | 5 | 1/5 | 0.050 | 0.050 | -0.022 | 0.025 | -0.025 |
| `prompt_last|L32|mlp|raw` | 5 | 1/5 | 0.050 | 0.050 | 0.196 | 0.019 | -0.031 |
| `prompt_last|L33|mlp|random_same_norm` | 5 | 1/5 | 0.050 | 0.050 | -0.026 | 0.013 | -0.037 |
| `prompt_last|L34|mlp|specific_norm_raw` | 5 | 1/5 | 0.050 | 0.050 | 0.171 | -0.006 | -0.056 |
| `query_category|L32|mlp|random_same_norm` | 5 | 1/5 | 0.050 | 0.050 | -0.052 | 0.044 | -0.006 |

## glm4

cases=64, rows=4, target_cases_seen=4, alpha=1.0, time_min=0.69

| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L36|mlp|raw` | 4 | 0/4 | 0.000 | 0.000 | -0.016 | -0.004 | -0.004 |
| `prompt_last|L38|mlp|common_only` | 4 | 0/4 | 0.000 | 0.000 | -0.103 | 0.004 | 0.004 |
| `prompt_last|L37|mlp|specific_only` | 4 | 0/4 | 0.000 | 0.000 | 0.007 | 0.010 | 0.010 |
| `prompt_last|L39|mlp|common_norm_raw` | 4 | 0/4 | 0.000 | 0.000 | -4.814 | 0.000 | 0.000 |
| `prompt_last|L39|mlp|common_only` | 4 | 0/4 | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 |
| `prompt_last|L39|mlp|random_same_norm` | 4 | 0/4 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 |
| `prompt_last|L39|mlp|raw` | 4 | 0/4 | 0.000 | 0.000 | -0.152 | 0.000 | 0.000 |
| `prompt_last|L39|mlp|specific_norm_raw` | 4 | 0/4 | 0.000 | 0.000 | -1.634 | 0.000 | 0.000 |
| `prompt_last|L39|mlp|specific_only` | 4 | 0/4 | 0.000 | 0.000 | 0.017 | 0.000 | 0.000 |
| `prompt_last|L36|mlp|specific_only` | 4 | 0/4 | -0.016 | -0.016 | 0.006 | -0.008 | 0.008 |
| `prompt_last|L35|mlp|common_norm_raw` | 4 | 0/4 | -0.016 | -0.016 | -3.447 | -0.002 | 0.014 |
| `prompt_last|L34|mlp|raw` | 4 | 0/4 | -0.016 | -0.016 | 0.032 | 0.006 | 0.021 |

## deepseek7b

cases=64, rows=21, target_cases_seen=21, alpha=1.0, time_min=1.77

| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rule_relation|L18|mlp|raw` | 21 | 1/21 | 0.024 | 0.024 | 0.033 | 0.014 | -0.010 |
| `prompt_last|L26|mlp|specific_norm_raw` | 21 | 1/21 | 0.006 | 0.006 | -0.345 | 0.000 | -0.006 |
| `query_relation|L16|mlp|random_same_norm` | 21 | 1/21 | -0.055 | -0.055 | -0.065 | 0.012 | 0.067 |
| `rule_relation|L18|mlp|specific_norm_raw` | 21 | 0/21 | 0.022 | 0.022 | 0.022 | 0.015 | -0.007 |
| `rule_relation|L20|mlp|specific_only` | 21 | 0/21 | 0.008 | 0.008 | -0.008 | 0.017 | 0.009 |
| `prompt_last|L26|mlp|common_only` | 21 | 0/21 | 0.006 | 0.006 | 0.011 | 0.002 | -0.004 |
| `prompt_last|L26|mlp|common_norm_raw` | 21 | 0/21 | 0.001 | 0.001 | -1.551 | -0.005 | -0.005 |
| `rule_value|L26|mlp|common_only` | 21 | 0/21 | -0.000 | -0.000 | -0.003 | 0.004 | 0.004 |
| `rule_relation|L18|mlp|specific_only` | 21 | 0/21 | -0.003 | -0.003 | 0.021 | -0.004 | -0.001 |
| `rule_relation|L20|mlp|raw` | 21 | 0/21 | -0.006 | -0.006 | -0.011 | 0.002 | 0.008 |
| `rule_value|L26|mlp|random_same_norm` | 21 | 0/21 | -0.006 | -0.006 | 0.005 | -0.004 | 0.002 |
| `prompt_last|L26|mlp|specific_only` | 21 | 0/21 | -0.006 | -0.006 | -0.002 | -0.000 | 0.006 |

### DS7B watched nodes

| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|mlp|raw` | 21 | 0/21 | -0.009 | -0.009 | 0.000 | -0.009 | -0.000 |
| `rule_value|L26|mlp|specific_only` | 21 | 0/21 | -0.012 | -0.012 | -0.003 | -0.004 | 0.009 |
| `rule_value|L26|mlp|common_only` | 21 | 0/21 | -0.000 | -0.000 | -0.003 | 0.004 | 0.004 |
| `rule_value|L26|mlp|specific_norm_raw` | 21 | 0/21 | -0.007 | -0.007 | -0.002 | -0.003 | 0.004 |
| `rule_value|L26|mlp|common_norm_raw` | 21 | 0/21 | -0.013 | -0.013 | -0.000 | -0.002 | 0.010 |
| `rule_value|L26|mlp|random_same_norm` | 21 | 0/21 | -0.006 | -0.006 | 0.005 | -0.004 | 0.002 |
| `query_relation|L19|mlp|raw` | 21 | 0/21 | -0.016 | -0.016 | 0.229 | 0.008 | 0.024 |
| `query_relation|L19|mlp|specific_only` | 21 | 0/21 | -0.036 | -0.036 | -0.018 | -0.010 | 0.026 |
| `query_relation|L19|mlp|common_only` | 21 | 0/21 | -0.046 | -0.046 | 0.006 | -0.015 | 0.031 |
| `prompt_last|L26|mlp|raw` | 21 | 0/21 | -0.032 | -0.032 | -0.027 | -0.009 | 0.023 |
| `prompt_last|L26|mlp|specific_only` | 21 | 0/21 | -0.006 | -0.006 | -0.002 | -0.000 | 0.006 |

