# Phase599 Cross-Model Summary

Final layer washout decomposition.

## qwen3

cases=64, rows=5, target_cases_seen=5, probe_layer=35, alpha=2.0, time_min=0.47

### Final Effects

| key | kind | n | switch | generated_down_projection | full_margin_gain | first_token_logit_margin_gain |
|---|---|---:|---:|---:|---:|---:|
| `query_category|L32|repair_alpha2` | repair | 5 | 2/5 | 1.211 | 0.050 | 0.000 |
| `prompt_last|L34|wrong_alpha2` | wrong | 5 | 1/5 | 0.484 | 0.100 | 0.000 |
| `prompt_last|L32|repair_alpha2` | repair | 5 | 1/5 | 0.371 | 0.075 | 0.000 |
| `query_category|L32|wrong_alpha2` | wrong | 5 | 1/5 | 1.485 | 0.050 | 0.000 |
| `prompt_last|L32|wrong_alpha2` | wrong | 5 | 1/5 | 0.544 | 0.050 | 0.000 |
| `prompt_last|L34|random_alpha2` | random | 5 | 1/5 | -1.100 | 0.025 | 0.000 |
| `prompt_last|L32|random_alpha2` | random | 5 | 0/5 | 0.105 | 0.075 | 0.000 |
| `prompt_last|L34|repair_alpha2` | repair | 5 | 0/5 | 0.558 | 0.025 | 0.000 |
| `query_category|L32|random_alpha2` | random | 5 | 0/5 | 1.217 | -0.000 | 0.000 |

### Component Projections

| key | kind | component | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---|---:|---:|---:|---:|---:|
| `query_category|L32|wrong_alpha2|final_norm_input` | wrong | `final_norm_input` | 5 | 1.374 | 0.816 | -0.558 | 0.800 |
| `query_category|L32|wrong_alpha2|layer_out` | wrong | `layer_out` | 5 | 1.374 | 0.816 | -0.558 | 0.800 |
| `query_category|L32|wrong_alpha2|layer_input` | wrong | `layer_input` | 5 | 1.147 | 0.759 | -0.389 | 0.800 |
| `prompt_last|L34|repair_alpha2|final_norm_input` | repair | `final_norm_input` | 5 | 1.083 | 0.562 | -0.521 | 0.800 |
| `prompt_last|L34|repair_alpha2|layer_out` | repair | `layer_out` | 5 | 1.083 | 0.562 | -0.521 | 0.800 |
| `query_category|L32|repair_alpha2|final_norm_input` | repair | `final_norm_input` | 5 | 1.071 | 0.700 | -0.371 | 0.800 |
| `query_category|L32|repair_alpha2|layer_out` | repair | `layer_out` | 5 | 1.071 | 0.700 | -0.371 | 0.800 |
| `query_category|L32|random_alpha2|final_norm_input` | random | `final_norm_input` | 5 | 1.019 | 0.679 | -0.340 | 0.800 |
| `query_category|L32|random_alpha2|layer_out` | random | `layer_out` | 5 | 1.019 | 0.679 | -0.340 | 0.800 |
| `query_category|L32|random_alpha2|layer_input` | random | `layer_input` | 5 | 0.992 | 0.761 | -0.230 | 0.600 |
| `prompt_last|L32|repair_alpha2|final_norm_input` | repair | `final_norm_input` | 5 | 0.903 | 0.414 | -0.489 | 0.800 |
| `prompt_last|L32|repair_alpha2|layer_out` | repair | `layer_out` | 5 | 0.903 | 0.414 | -0.489 | 0.800 |
| `query_category|L32|repair_alpha2|layer_input` | repair | `layer_input` | 5 | 0.892 | 0.730 | -0.162 | 1.000 |
| `prompt_last|L32|wrong_alpha2|final_norm_input` | wrong | `final_norm_input` | 5 | 0.871 | 0.518 | -0.354 | 0.800 |
| `prompt_last|L32|wrong_alpha2|layer_out` | wrong | `layer_out` | 5 | 0.871 | 0.518 | -0.354 | 0.800 |
| `prompt_last|L34|wrong_alpha2|final_norm_input` | wrong | `final_norm_input` | 5 | 0.787 | 0.286 | -0.501 | 0.800 |
| `prompt_last|L34|wrong_alpha2|layer_out` | wrong | `layer_out` | 5 | 0.787 | 0.286 | -0.501 | 0.800 |
| `prompt_last|L32|repair_alpha2|layer_input` | repair | `layer_input` | 5 | 0.668 | 0.353 | -0.315 | 1.000 |
| `prompt_last|L32|wrong_alpha2|layer_input` | wrong | `layer_input` | 5 | 0.582 | 0.343 | -0.239 | 0.800 |
| `prompt_last|L34|repair_alpha2|layer_input` | repair | `layer_input` | 5 | 0.558 | 0.468 | -0.090 | 0.800 |
| `query_category|L32|random_alpha2|mlp_out` | random | `mlp_out` | 5 | 0.500 | 0.212 | -0.289 | 1.000 |
| `prompt_last|L34|wrong_alpha2|layer_input` | wrong | `layer_input` | 5 | 0.481 | 0.244 | -0.237 | 0.800 |
| `prompt_last|L34|repair_alpha2|attn_out` | repair | `attn_out` | 5 | 0.404 | 0.033 | -0.371 | 0.800 |
| `prompt_last|L34|wrong_alpha2|attn_out` | wrong | `attn_out` | 5 | 0.367 | 0.062 | -0.305 | 0.800 |

## glm4

cases=64, rows=4, target_cases_seen=4, probe_layer=39, alpha=2.0, time_min=0.68

### Final Effects

| key | kind | n | switch | generated_down_projection | full_margin_gain | first_token_logit_margin_gain |
|---|---|---:|---:|---:|---:|---:|
| `prompt_last|L38|random_alpha2` | random | 4 | 0/4 | -0.432 | 0.016 | 0.000 |
| `prompt_last|L39|wrong_alpha2` | wrong | 4 | 0/4 | 0.096 | 0.000 | 0.000 |
| `prompt_last|L39|repair_alpha2` | repair | 4 | 0/4 | 0.037 | 0.000 | 0.000 |
| `prompt_last|L39|random_alpha2` | random | 4 | 0/4 | 0.029 | 0.000 | 0.000 |
| `prompt_last|L37|wrong_alpha2` | wrong | 4 | 0/4 | -0.044 | -0.016 | 0.000 |
| `prompt_last|L38|wrong_alpha2` | wrong | 4 | 0/4 | -0.564 | -0.016 | 0.000 |
| `prompt_last|L38|repair_alpha2` | repair | 4 | 0/4 | 0.245 | -0.016 | 0.000 |
| `prompt_last|L37|repair_alpha2` | repair | 4 | 0/4 | 0.062 | -0.031 | 0.000 |
| `prompt_last|L37|random_alpha2` | random | 4 | 0/4 | 0.178 | -0.031 | 0.000 |

### Component Projections

| key | kind | component | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---|---:|---:|---:|---:|---:|
| `prompt_last|L39|repair_alpha2|mlp_input` | repair | `mlp_input` | 4 | 2.024 | 1.065 | -0.959 | 0.500 |
| `prompt_last|L38|repair_alpha2|final_norm_input` | repair | `final_norm_input` | 4 | 0.444 | 0.142 | -0.302 | 0.750 |
| `prompt_last|L38|repair_alpha2|layer_out` | repair | `layer_out` | 4 | 0.444 | 0.142 | -0.302 | 0.750 |
| `prompt_last|L37|random_alpha2|mlp_input` | random | `mlp_input` | 4 | 0.398 | 0.204 | -0.194 | 1.000 |
| `prompt_last|L38|repair_alpha2|final_norm_output` | repair | `final_norm_output` | 4 | 0.364 | 0.130 | -0.234 | 0.750 |
| `prompt_last|L38|repair_alpha2|mlp_input` | repair | `mlp_input` | 4 | 0.363 | 0.135 | -0.228 | 0.750 |
| `prompt_last|L37|random_alpha2|layer_input` | random | `layer_input` | 4 | 0.350 | 0.191 | -0.159 | 1.000 |
| `prompt_last|L37|random_alpha2|final_norm_input` | random | `final_norm_input` | 4 | 0.309 | 0.146 | -0.163 | 0.500 |
| `prompt_last|L37|random_alpha2|layer_out` | random | `layer_out` | 4 | 0.309 | 0.146 | -0.163 | 0.500 |
| `prompt_last|L37|random_alpha2|final_norm_output` | random | `final_norm_output` | 4 | 0.283 | 0.164 | -0.120 | 1.000 |
| `prompt_last|L39|wrong_alpha2|mlp_input` | wrong | `mlp_input` | 4 | 0.260 | -0.408 | -0.668 | 0.250 |
| `prompt_last|L38|repair_alpha2|layer_input` | repair | `layer_input` | 4 | 0.246 | 0.065 | -0.181 | 0.250 |
| `prompt_last|L38|random_alpha2|mlp_out` | random | `mlp_out` | 4 | 0.220 | 0.147 | -0.073 | 0.750 |
| `prompt_last|L37|repair_alpha2|final_norm_input` | repair | `final_norm_input` | 4 | 0.180 | 0.102 | -0.078 | 0.750 |
| `prompt_last|L37|repair_alpha2|layer_out` | repair | `layer_out` | 4 | 0.180 | 0.102 | -0.078 | 0.750 |
| `prompt_last|L38|repair_alpha2|mlp_out` | repair | `mlp_out` | 4 | 0.169 | 0.064 | -0.105 | 0.750 |
| `prompt_last|L38|wrong_alpha2|mlp_out` | wrong | `mlp_out` | 4 | 0.140 | 0.115 | -0.025 | 1.000 |
| `prompt_last|L37|repair_alpha2|final_norm_output` | repair | `final_norm_output` | 4 | 0.124 | 0.064 | -0.060 | 0.750 |
| `prompt_last|L37|repair_alpha2|mlp_input` | repair | `mlp_input` | 4 | 0.113 | 0.053 | -0.061 | 0.500 |
| `prompt_last|L37|repair_alpha2|layer_input` | repair | `layer_input` | 4 | 0.110 | 0.074 | -0.036 | 0.500 |
| `prompt_last|L37|wrong_alpha2|mlp_out` | wrong | `mlp_out` | 4 | 0.105 | 0.025 | -0.080 | 0.750 |
| `prompt_last|L39|wrong_alpha2|final_norm_input` | wrong | `final_norm_input` | 4 | 0.102 | 0.031 | -0.072 | 0.750 |
| `prompt_last|L39|wrong_alpha2|layer_out` | wrong | `layer_out` | 4 | 0.102 | 0.031 | -0.072 | 0.750 |
| `prompt_last|L39|wrong_alpha2|mlp_out` | wrong | `mlp_out` | 4 | 0.102 | 0.030 | -0.071 | 0.750 |

## deepseek7b

cases=64, rows=21, target_cases_seen=21, probe_layer=27, alpha=2.0, time_min=2.05

### Final Effects

| key | kind | n | switch | generated_down_projection | full_margin_gain | first_token_logit_margin_gain |
|---|---|---:|---:|---:|---:|---:|
| `rule_value|L26|repair_alpha2` | repair | 21 | 0/21 | 3.618 | -0.006 | 0.000 |
| `rule_value|L26|random_alpha2` | random | 21 | 0/21 | 4.251 | -0.012 | 0.000 |
| `prompt_last|L26|random_alpha2` | random | 21 | 0/21 | 6.424 | -0.012 | 0.000 |
| `query_relation|L19|wrong_alpha2` | wrong | 21 | 0/21 | -0.454 | -0.015 | 0.000 |
| `rule_value|L26|wrong_alpha2` | wrong | 8 | 0/8 | 0.973 | -0.016 | 0.000 |
| `prompt_last|L26|repair_alpha2` | repair | 21 | 0/21 | 4.414 | -0.022 | 0.000 |
| `query_relation|L19|repair_alpha2` | repair | 21 | 0/21 | 0.353 | -0.026 | 0.000 |
| `prompt_last|L26|wrong_alpha2` | wrong | 21 | 0/21 | 1.557 | -0.036 | 0.000 |
| `query_relation|L19|random_alpha2` | random | 21 | 0/21 | -0.521 | -0.083 | 0.000 |

### Component Projections

| key | kind | component | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---|---:|---:|---:|---:|---:|
| `query_relation|L19|random_alpha2|final_norm_input` | random | `final_norm_input` | 21 | 7.380 | 2.464 | -4.916 | 0.857 |
| `query_relation|L19|random_alpha2|layer_out` | random | `layer_out` | 21 | 7.380 | 2.464 | -4.916 | 0.857 |
| `prompt_last|L26|random_alpha2|layer_input` | random | `layer_input` | 21 | 6.382 | 1.838 | -4.544 | 0.905 |
| `prompt_last|L26|repair_alpha2|layer_input` | repair | `layer_input` | 21 | 4.402 | 2.172 | -2.231 | 0.810 |
| `rule_value|L26|random_alpha2|layer_input` | random | `layer_input` | 21 | 4.270 | 1.283 | -2.987 | 0.810 |
| `query_relation|L19|random_alpha2|mlp_out` | random | `mlp_out` | 21 | 3.862 | 1.413 | -2.449 | 0.714 |
| `rule_value|L26|repair_alpha2|layer_input` | repair | `layer_input` | 21 | 3.643 | 0.877 | -2.766 | 0.857 |
| `query_relation|L19|random_alpha2|layer_input` | random | `layer_input` | 21 | 2.305 | 0.793 | -1.512 | 0.762 |
| `query_relation|L19|repair_alpha2|final_norm_input` | repair | `final_norm_input` | 21 | 2.304 | 0.729 | -1.574 | 0.619 |
| `query_relation|L19|repair_alpha2|layer_out` | repair | `layer_out` | 21 | 2.304 | 0.729 | -1.574 | 0.619 |
| `query_relation|L19|repair_alpha2|mlp_out` | repair | `mlp_out` | 21 | 1.870 | 0.655 | -1.214 | 0.667 |
| `prompt_last|L26|wrong_alpha2|layer_input` | wrong | `layer_input` | 21 | 1.519 | 0.329 | -1.190 | 0.571 |
| `query_relation|L19|random_alpha2|attn_out` | random | `attn_out` | 21 | 1.187 | 0.246 | -0.941 | 0.714 |
| `query_relation|L19|repair_alpha2|attn_out` | repair | `attn_out` | 21 | 1.163 | 0.306 | -0.857 | 0.714 |
| `rule_value|L26|random_alpha2|final_norm_input` | random | `final_norm_input` | 21 | 1.071 | 1.272 | 0.202 | 0.524 |
| `rule_value|L26|random_alpha2|layer_out` | random | `layer_out` | 21 | 1.071 | 1.272 | 0.202 | 0.524 |
| `rule_value|L26|wrong_alpha2|layer_input` | wrong | `layer_input` | 8 | 0.980 | -0.587 | -1.567 | 0.625 |
| `query_relation|L19|wrong_alpha2|attn_out` | wrong | `attn_out` | 21 | 0.864 | 0.136 | -0.728 | 0.810 |
| `rule_value|L26|repair_alpha2|final_norm_input` | repair | `final_norm_input` | 21 | 0.325 | -0.384 | -0.710 | 0.524 |
| `rule_value|L26|repair_alpha2|layer_out` | repair | `layer_out` | 21 | 0.325 | -0.384 | -0.710 | 0.524 |
| `query_relation|L19|wrong_alpha2|mlp_out` | wrong | `mlp_out` | 21 | 0.319 | 0.015 | -0.304 | 0.571 |
| `rule_value|L26|wrong_alpha2|final_norm_output` | wrong | `final_norm_output` | 8 | 0.270 | 0.089 | -0.182 | 0.625 |
| `query_relation|L19|wrong_alpha2|final_norm_input` | wrong | `final_norm_input` | 21 | 0.239 | -0.122 | -0.361 | 0.476 |
| `query_relation|L19|wrong_alpha2|layer_out` | wrong | `layer_out` | 21 | 0.239 | -0.122 | -0.361 | 0.476 |

### DS7B watched final effects

| key | kind | n | switch | generated_down_projection | full_margin_gain | first_token_logit_margin_gain |
|---|---|---:|---:|---:|---:|---:|
| `rule_value|L26|repair_alpha2` | repair | 21 | 0/21 | 3.618 | -0.006 | 0.000 |
| `rule_value|L26|random_alpha2` | random | 21 | 0/21 | 4.251 | -0.012 | 0.000 |
| `rule_value|L26|wrong_alpha2` | wrong | 8 | 0/8 | 0.973 | -0.016 | 0.000 |
| `prompt_last|L26|repair_alpha2` | repair | 21 | 0/21 | 4.414 | -0.022 | 0.000 |
| `prompt_last|L26|random_alpha2` | random | 21 | 0/21 | 6.424 | -0.012 | 0.000 |
| `prompt_last|L26|wrong_alpha2` | wrong | 21 | 0/21 | 1.557 | -0.036 | 0.000 |

### DS7B watched component path

| key | kind | component | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |
|---|---|---|---:|---:|---:|---:|---:|
| `rule_value|L26|repair_alpha2|layer_input` | repair | `layer_input` | 21 | 3.643 | 0.877 | -2.766 | 0.857 |
| `rule_value|L26|repair_alpha2|attn_out` | repair | `attn_out` | 21 | -2.037 | -0.940 | 1.097 | 0.333 |
| `rule_value|L26|repair_alpha2|mlp_input` | repair | `mlp_input` | 21 | 0.098 | 0.028 | -0.070 | 0.667 |
| `rule_value|L26|repair_alpha2|mlp_out` | repair | `mlp_out` | 21 | -1.272 | -0.310 | 0.962 | 0.286 |
| `rule_value|L26|repair_alpha2|layer_out` | repair | `layer_out` | 21 | 0.325 | -0.384 | -0.710 | 0.524 |
| `rule_value|L26|repair_alpha2|final_norm_input` | repair | `final_norm_input` | 21 | 0.325 | -0.384 | -0.710 | 0.524 |
| `rule_value|L26|repair_alpha2|final_norm_output` | repair | `final_norm_output` | 21 | 0.151 | 0.052 | -0.099 | 0.619 |
| `rule_value|L26|random_alpha2|layer_input` | random | `layer_input` | 21 | 4.270 | 1.283 | -2.987 | 0.810 |
| `rule_value|L26|random_alpha2|attn_out` | random | `attn_out` | 21 | -2.005 | -0.497 | 1.509 | 0.190 |
| `rule_value|L26|random_alpha2|mlp_input` | random | `mlp_input` | 21 | 0.089 | 0.048 | -0.041 | 0.571 |
| `rule_value|L26|random_alpha2|mlp_out` | random | `mlp_out` | 21 | -1.207 | 0.493 | 1.700 | 0.429 |
| `rule_value|L26|random_alpha2|layer_out` | random | `layer_out` | 21 | 1.071 | 1.272 | 0.202 | 0.524 |
| `rule_value|L26|random_alpha2|final_norm_input` | random | `final_norm_input` | 21 | 1.071 | 1.272 | 0.202 | 0.524 |
| `rule_value|L26|random_alpha2|final_norm_output` | random | `final_norm_output` | 21 | 0.090 | 0.076 | -0.014 | 0.619 |
| `rule_value|L26|wrong_alpha2|layer_input` | wrong | `layer_input` | 8 | 0.980 | -0.587 | -1.567 | 0.625 |
| `rule_value|L26|wrong_alpha2|attn_out` | wrong | `attn_out` | 8 | -1.942 | -1.001 | 0.942 | 0.375 |
| `rule_value|L26|wrong_alpha2|mlp_input` | wrong | `mlp_input` | 8 | 0.112 | 0.015 | -0.097 | 0.750 |
| `rule_value|L26|wrong_alpha2|mlp_out` | wrong | `mlp_out` | 8 | -3.200 | -1.711 | 1.489 | 0.250 |
| `rule_value|L26|wrong_alpha2|layer_out` | wrong | `layer_out` | 8 | -4.219 | -3.331 | 0.888 | 0.250 |
| `rule_value|L26|wrong_alpha2|final_norm_input` | wrong | `final_norm_input` | 8 | -4.219 | -3.331 | 0.888 | 0.250 |
| `rule_value|L26|wrong_alpha2|final_norm_output` | wrong | `final_norm_output` | 8 | 0.270 | 0.089 | -0.182 | 0.625 |
| `prompt_last|L26|repair_alpha2|layer_input` | repair | `layer_input` | 21 | 4.402 | 2.172 | -2.231 | 0.810 |
| `prompt_last|L26|repair_alpha2|attn_out` | repair | `attn_out` | 21 | -1.550 | -0.277 | 1.273 | 0.143 |
| `prompt_last|L26|repair_alpha2|mlp_input` | repair | `mlp_input` | 21 | 0.090 | 0.102 | 0.012 | 0.619 |
| `prompt_last|L26|repair_alpha2|mlp_out` | repair | `mlp_out` | 21 | -4.914 | -1.078 | 3.835 | 0.143 |
| `prompt_last|L26|repair_alpha2|layer_out` | repair | `layer_out` | 21 | -2.000 | 0.838 | 2.838 | 0.333 |
| `prompt_last|L26|repair_alpha2|final_norm_input` | repair | `final_norm_input` | 21 | -2.000 | 0.838 | 2.838 | 0.333 |
| `prompt_last|L26|repair_alpha2|final_norm_output` | repair | `final_norm_output` | 21 | 0.151 | 0.169 | 0.018 | 0.619 |
| `prompt_last|L26|random_alpha2|layer_input` | random | `layer_input` | 21 | 6.382 | 1.838 | -4.544 | 0.905 |
| `prompt_last|L26|random_alpha2|attn_out` | random | `attn_out` | 21 | -3.992 | -1.110 | 2.881 | 0.143 |
| `prompt_last|L26|random_alpha2|mlp_input` | random | `mlp_input` | 21 | 0.095 | 0.040 | -0.055 | 0.667 |
| `prompt_last|L26|random_alpha2|mlp_out` | random | `mlp_out` | 21 | -2.497 | -0.242 | 2.255 | 0.476 |
| `prompt_last|L26|random_alpha2|layer_out` | random | `layer_out` | 21 | -0.102 | 0.479 | 0.581 | 0.476 |
| `prompt_last|L26|random_alpha2|final_norm_input` | random | `final_norm_input` | 21 | -0.102 | 0.479 | 0.581 | 0.476 |
| `prompt_last|L26|random_alpha2|final_norm_output` | random | `final_norm_output` | 21 | 0.091 | 0.049 | -0.042 | 0.571 |
| `prompt_last|L26|wrong_alpha2|layer_input` | wrong | `layer_input` | 21 | 1.519 | 0.329 | -1.190 | 0.571 |
| `prompt_last|L26|wrong_alpha2|attn_out` | wrong | `attn_out` | 21 | -0.948 | -0.233 | 0.715 | 0.238 |
| `prompt_last|L26|wrong_alpha2|mlp_input` | wrong | `mlp_input` | 21 | -0.063 | -0.028 | 0.035 | 0.286 |
| `prompt_last|L26|wrong_alpha2|mlp_out` | wrong | `mlp_out` | 21 | -2.126 | -0.867 | 1.259 | 0.190 |
| `prompt_last|L26|wrong_alpha2|layer_out` | wrong | `layer_out` | 21 | -1.542 | -0.775 | 0.767 | 0.286 |
| `prompt_last|L26|wrong_alpha2|final_norm_input` | wrong | `final_norm_input` | 21 | -1.542 | -0.775 | 0.767 | 0.286 |
| `prompt_last|L26|wrong_alpha2|final_norm_output` | wrong | `final_norm_output` | 21 | -0.098 | -0.036 | 0.062 | 0.238 |

