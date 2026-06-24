# Phase602 Cross-Model Summary

Attention-source factor causal patch.

## qwen3

cases=96, rows=7, target_cases_seen=7, probe_layer=35, alpha=2.0, attn_scale=1.0, time_min=0.73

### Best Effects

| key | mode | n | switch | full_margin_gain | mlp_down_projection | attn_delta_projection | final_norm_projection | final_norm_cos_to_natural | positive_full_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `query_category|L32|mlp_plus_attn_effect` | mlp_plus_attn_effect | 7 | 2/7 | 0.036 | 1.214 | -0.206 | 0.173 | 0.511 | 0.714 |
| `query_category|L32|mlp_repair_only` | mlp_repair_only | 7 | 2/7 | 0.036 | 1.214 | 0.000 | 0.182 | 0.403 | 0.714 |
| `query_category|L32|mlp_plus_attn_random` | mlp_plus_attn_random | 7 | 2/7 | 0.036 | 1.214 | -0.008 | 0.185 | 0.392 | 0.714 |
| `prompt_last|L32|mlp_plus_attn_effect` | mlp_plus_attn_effect | 7 | 1/7 | 0.071 | 0.974 | -0.678 | 0.265 | 0.594 | 0.857 |
| `prompt_last|L32|mlp_repair_only` | mlp_repair_only | 7 | 1/7 | 0.071 | 0.974 | 0.000 | 0.441 | 0.437 | 0.857 |
| `prompt_last|L32|mlp_plus_attn_random` | mlp_plus_attn_random | 7 | 1/7 | 0.071 | 0.974 | -0.002 | 0.444 | 0.367 | 0.857 |
| `prompt_last|L34|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 7 | 1/7 | -0.018 | 0.000 | -0.678 | -0.327 | 0.230 | 0.429 |
| `prompt_last|L32|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 7 | 0/7 | 0.071 | 0.000 | -0.678 | -0.110 | 0.367 | 0.714 |
| `prompt_last|L32|attn_effect_only` | attn_effect_only | 7 | 0/7 | 0.000 | 0.000 | -0.678 | -0.168 | 0.521 | 0.000 |
| `prompt_last|L34|attn_effect_only` | attn_effect_only | 7 | 0/7 | 0.000 | 0.000 | -0.678 | -0.168 | 0.521 | 0.000 |
| `query_category|L32|attn_effect_only` | attn_effect_only | 7 | 0/7 | 0.000 | 0.000 | -0.206 | -0.009 | 0.477 | 0.000 |
| `query_category|L32|attn_random` | attn_random | 7 | 0/7 | 0.000 | 0.000 | -0.008 | 0.021 | 0.051 | 0.000 |
| `prompt_last|L32|attn_random` | attn_random | 7 | 0/7 | 0.000 | 0.000 | -0.002 | 0.004 | -0.003 | 0.000 |
| `prompt_last|L34|attn_random` | attn_random | 7 | 0/7 | 0.000 | 0.000 | 0.132 | 0.041 | -0.018 | 0.000 |
| `query_category|L32|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 7 | 0/7 | -0.000 | 0.000 | -0.206 | 0.153 | 0.244 | 0.429 |
| `prompt_last|L34|mlp_plus_attn_effect` | mlp_plus_attn_effect | 7 | 0/7 | -0.000 | 1.274 | -0.678 | 0.272 | 0.572 | 0.429 |
| `prompt_last|L34|mlp_repair_only` | mlp_repair_only | 7 | 0/7 | -0.000 | 1.274 | 0.000 | 0.448 | 0.397 | 0.429 |
| `prompt_last|L34|mlp_plus_attn_random` | mlp_plus_attn_random | 7 | 0/7 | -0.000 | 1.274 | 0.132 | 0.501 | 0.338 | 0.429 |

## glm4

cases=96, rows=13, target_cases_seen=13, probe_layer=39, alpha=2.0, attn_scale=1.0, time_min=1.63

### Best Effects

| key | mode | n | switch | full_margin_gain | mlp_down_projection | attn_delta_projection | final_norm_projection | final_norm_cos_to_natural | positive_full_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L38|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 13 | 0/13 | 0.005 | 0.000 | 0.009 | -0.080 | 0.044 | 0.462 |
| `prompt_last|L38|mlp_plus_attn_effect` | mlp_plus_attn_effect | 13 | 0/13 | 0.005 | 0.149 | 0.009 | 0.325 | 0.871 | 0.385 |
| `prompt_last|L38|mlp_repair_only` | mlp_repair_only | 13 | 0/13 | 0.005 | 0.149 | 0.000 | 0.299 | 0.864 | 0.385 |
| `prompt_last|L38|mlp_plus_attn_random` | mlp_plus_attn_random | 13 | 0/13 | 0.005 | 0.149 | 0.011 | 0.306 | 0.862 | 0.385 |
| `prompt_last|L39|mlp_plus_attn_effect` | mlp_plus_attn_effect | 13 | 0/13 | 0.000 | 0.184 | 0.009 | 0.100 | 0.613 | 0.000 |
| `prompt_last|L39|mlp_repair_only` | mlp_repair_only | 13 | 0/13 | 0.000 | 0.184 | 0.000 | 0.091 | 0.604 | 0.000 |
| `prompt_last|L39|mlp_plus_attn_random` | mlp_plus_attn_random | 13 | 0/13 | 0.000 | 0.184 | 0.006 | 0.096 | 0.603 | 0.000 |
| `prompt_last|L37|attn_effect_only` | attn_effect_only | 13 | 0/13 | 0.000 | 0.000 | 0.009 | 0.016 | 0.198 | 0.000 |
| `prompt_last|L38|attn_effect_only` | attn_effect_only | 13 | 0/13 | 0.000 | 0.000 | 0.009 | 0.016 | 0.198 | 0.000 |
| `prompt_last|L39|attn_effect_only` | attn_effect_only | 13 | 0/13 | 0.000 | 0.000 | 0.009 | 0.016 | 0.198 | 0.000 |
| `prompt_last|L39|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 13 | 0/13 | 0.000 | 0.000 | 0.009 | 0.281 | 0.044 | 0.000 |
| `prompt_last|L39|attn_random` | attn_random | 13 | 0/13 | 0.000 | 0.000 | 0.006 | 0.002 | 0.005 | 0.000 |
| `prompt_last|L37|attn_random` | attn_random | 13 | 0/13 | 0.000 | 0.000 | 0.024 | 0.015 | -0.001 | 0.000 |
| `prompt_last|L38|attn_random` | attn_random | 13 | 0/13 | 0.000 | 0.000 | 0.011 | 0.013 | -0.005 | 0.000 |
| `prompt_last|L37|mlp_plus_attn_effect` | mlp_plus_attn_effect | 13 | 0/13 | -0.010 | 0.034 | 0.009 | 0.080 | 0.351 | 0.385 |
| `prompt_last|L37|mlp_repair_only` | mlp_repair_only | 13 | 0/13 | -0.010 | 0.034 | 0.000 | 0.065 | 0.326 | 0.385 |
| `prompt_last|L37|mlp_plus_attn_random` | mlp_plus_attn_random | 13 | 0/13 | -0.010 | 0.034 | 0.024 | 0.083 | 0.322 | 0.385 |
| `prompt_last|L37|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 13 | 0/13 | -0.014 | 0.000 | 0.009 | 0.089 | 0.075 | 0.231 |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, probe_layer=27, alpha=2.0, attn_scale=1.0, time_min=3.59

### Best Effects

| key | mode | n | switch | full_margin_gain | mlp_down_projection | attn_delta_projection | final_norm_projection | final_norm_cos_to_natural | positive_full_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|attn_effect_only` | attn_effect_only | 37 | 0/37 | 0.000 | 0.000 | -3.019 | -0.019 | 0.340 | 0.000 |
| `prompt_last|L26|attn_effect_only` | attn_effect_only | 37 | 0/37 | 0.000 | 0.000 | 0.022 | -0.001 | 0.180 | 0.000 |
| `query_relation|L19|attn_effect_only` | attn_effect_only | 37 | 0/37 | 0.000 | 0.000 | 0.923 | 0.007 | 0.167 | 0.000 |
| `rule_value|L26|attn_random` | attn_random | 37 | 0/37 | 0.000 | 0.000 | 0.542 | 0.004 | 0.057 | 0.000 |
| `prompt_last|L26|attn_random` | attn_random | 37 | 0/37 | 0.000 | 0.000 | 0.077 | 0.000 | 0.026 | 0.000 |
| `query_relation|L19|attn_random` | attn_random | 37 | 0/37 | 0.000 | 0.000 | 0.139 | 0.015 | 0.013 | 0.000 |
| `rule_value|L26|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.001 | 1.150 | -3.019 | 0.263 | 0.554 | 0.514 |
| `rule_value|L26|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.001 | 1.150 | 0.000 | 0.222 | 0.420 | 0.514 |
| `rule_value|L26|mlp_plus_attn_random` | mlp_plus_attn_random | 37 | 0/37 | -0.001 | 1.150 | 0.542 | 0.202 | 0.291 | 0.514 |
| `rule_value|L26|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 37 | 0/37 | -0.005 | 0.000 | -3.019 | -0.041 | 0.269 | 0.405 |
| `query_relation|L19|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.013 | -0.091 | 0.923 | 0.018 | 0.278 | 0.541 |
| `query_relation|L19|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.013 | -0.091 | 0.000 | 0.008 | 0.225 | 0.541 |
| `query_relation|L19|mlp_plus_attn_random` | mlp_plus_attn_random | 37 | 0/37 | -0.013 | -0.091 | 0.139 | 0.025 | 0.215 | 0.541 |
| `prompt_last|L26|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.018 | 3.084 | 0.022 | 0.181 | 0.541 | 0.324 |
| `prompt_last|L26|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.018 | 3.084 | 0.000 | 0.194 | 0.530 | 0.324 |
| `prompt_last|L26|mlp_plus_attn_random` | mlp_plus_attn_random | 37 | 0/37 | -0.018 | 3.084 | 0.077 | 0.197 | 0.524 | 0.324 |
| `prompt_last|L26|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 37 | 0/37 | -0.020 | 0.000 | 0.022 | 0.036 | 0.058 | 0.378 |
| `query_relation|L19|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 37 | 0/37 | -0.078 | 0.000 | 0.923 | 0.139 | 0.170 | 0.351 |

### DS7B watched combinations

| key | mode | n | switch | full_margin_gain | mlp_down_projection | attn_delta_projection | final_norm_projection | final_norm_cos_to_natural | positive_full_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.001 | 1.150 | 0.000 | 0.222 | 0.420 | 0.514 |
| `rule_value|L26|attn_effect_only` | attn_effect_only | 37 | 0/37 | 0.000 | 0.000 | -3.019 | -0.019 | 0.340 | 0.000 |
| `rule_value|L26|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.001 | 1.150 | -3.019 | 0.263 | 0.554 | 0.514 |
| `rule_value|L26|attn_random` | attn_random | 37 | 0/37 | 0.000 | 0.000 | 0.542 | 0.004 | 0.057 | 0.000 |
| `rule_value|L26|mlp_plus_attn_random` | mlp_plus_attn_random | 37 | 0/37 | -0.001 | 1.150 | 0.542 | 0.202 | 0.291 | 0.514 |
| `rule_value|L26|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 37 | 0/37 | -0.005 | 0.000 | -3.019 | -0.041 | 0.269 | 0.405 |
| `prompt_last|L26|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.018 | 3.084 | 0.000 | 0.194 | 0.530 | 0.324 |
| `prompt_last|L26|attn_effect_only` | attn_effect_only | 37 | 0/37 | 0.000 | 0.000 | 0.022 | -0.001 | 0.180 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.018 | 3.084 | 0.022 | 0.181 | 0.541 | 0.324 |
| `prompt_last|L26|attn_random` | attn_random | 37 | 0/37 | 0.000 | 0.000 | 0.077 | 0.000 | 0.026 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_random` | mlp_plus_attn_random | 37 | 0/37 | -0.018 | 3.084 | 0.077 | 0.197 | 0.524 | 0.324 |
| `prompt_last|L26|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 37 | 0/37 | -0.020 | 0.000 | 0.022 | 0.036 | 0.058 | 0.378 |
| `query_relation|L19|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.013 | -0.091 | 0.000 | 0.008 | 0.225 | 0.541 |
| `query_relation|L19|attn_effect_only` | attn_effect_only | 37 | 0/37 | 0.000 | 0.000 | 0.923 | 0.007 | 0.167 | 0.000 |
| `query_relation|L19|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.013 | -0.091 | 0.923 | 0.018 | 0.278 | 0.541 |
| `query_relation|L19|attn_random` | attn_random | 37 | 0/37 | 0.000 | 0.000 | 0.139 | 0.015 | 0.013 | 0.000 |
| `query_relation|L19|mlp_plus_attn_random` | mlp_plus_attn_random | 37 | 0/37 | -0.013 | -0.091 | 0.139 | 0.025 | 0.215 | 0.541 |
| `query_relation|L19|mlp_random_plus_attn_effect` | mlp_random_plus_attn_effect | 37 | 0/37 | -0.078 | 0.000 | 0.923 | 0.139 | 0.170 | 0.351 |

