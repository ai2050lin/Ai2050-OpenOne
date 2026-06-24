# Phase606 Cross-Model Summary

Digit1 upstream source decomposition.

## qwen3

cases=96, rows=7, target_cases_seen=7, probe_layer=35, time_min=0.45

### Best Component Patches

| key | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta | positive_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_input` | layer_input | False | 7 | 7/7 | 8.072 | 1.764 | -6.308 | 1.000 |
| `final_norm_input` | final_norm_input | False | 7 | 7/7 | 7.821 | 1.761 | -6.060 | 1.000 |
| `final_norm_output` | final_norm_output | False | 7 | 7/7 | 7.821 | 1.761 | -6.060 | 1.000 |
| `layer_input_random` | layer_input | True | 7 | 1/7 | 0.173 | 0.116 | -0.056 | 0.571 |
| `final_norm_input_random` | final_norm_input | True | 7 | 1/7 | 0.115 | 0.141 | 0.027 | 0.571 |
| `mlp_out_random` | mlp_out | True | 7 | 1/7 | -0.064 | -0.033 | 0.030 | 0.429 |
| `mlp_out` | mlp_out | False | 7 | 1/7 | -0.786 | -0.728 | 0.058 | 0.429 |
| `attn_out_random` | attn_out | True | 7 | 0/7 | -0.069 | -0.065 | 0.004 | 0.286 |
| `attn_out` | attn_out | False | 7 | 0/7 | -1.214 | -1.035 | 0.179 | 0.000 |

### Watched Component Patches

| key | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta | positive_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_input` | layer_input | False | 7 | 7/7 | 8.072 | 1.764 | -6.308 | 1.000 |
| `attn_out` | attn_out | False | 7 | 0/7 | -1.214 | -1.035 | 0.179 | 0.000 |
| `mlp_out` | mlp_out | False | 7 | 1/7 | -0.786 | -0.728 | 0.058 | 0.429 |
| `final_norm_input` | final_norm_input | False | 7 | 7/7 | 7.821 | 1.761 | -6.060 | 1.000 |
| `final_norm_output` | final_norm_output | False | 7 | 7/7 | 7.821 | 1.761 | -6.060 | 1.000 |
| `layer_input_random` | layer_input | True | 7 | 1/7 | 0.173 | 0.116 | -0.056 | 0.571 |
| `attn_out_random` | attn_out | True | 7 | 0/7 | -0.069 | -0.065 | 0.004 | 0.286 |
| `mlp_out_random` | mlp_out | True | 7 | 1/7 | -0.064 | -0.033 | 0.030 | 0.429 |
| `final_norm_input_random` | final_norm_input | True | 7 | 1/7 | 0.115 | 0.141 | 0.027 | 0.571 |

### Attention Source Mass Delta

| source | n | repair_minus_base_mass |
|---|---:|---:|
| `prompt_last` | 7 | 0.014 |
| `other` | 7 | -0.008 |
| `answer_prefix` | 7 | -0.005 |
| `query_relation` | 7 | -0.003 |
| `rule_relation` | 7 | 0.001 |
| `rule_value` | 7 | 0.001 |
| `digit1_position` | 7 | 0.000 |
| `object` | 7 | 0.000 |

## glm4

cases=96, rows=13, target_cases_seen=13, probe_layer=39, time_min=0.84

### Best Component Patches

| key | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta | positive_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_input` | layer_input | False | 13 | 13/13 | 2.923 | 1.004 | -1.919 | 1.000 |
| `final_norm_input` | final_norm_input | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | 1.000 |
| `final_norm_output` | final_norm_output | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | 1.000 |
| `mlp_out` | mlp_out | False | 13 | 3/13 | 0.120 | 0.008 | -0.112 | 0.846 |
| `final_norm_input_random` | final_norm_input | True | 13 | 1/13 | -0.134 | -0.087 | 0.046 | 0.308 |
| `layer_input_random` | layer_input | True | 13 | 0/13 | 0.035 | 0.035 | 0.000 | 0.692 |
| `attn_out_random` | attn_out | True | 13 | 0/13 | -0.023 | -0.024 | -0.002 | 0.462 |
| `attn_out` | attn_out | False | 13 | 0/13 | -0.038 | -0.018 | 0.021 | 0.308 |
| `mlp_out_random` | mlp_out | True | 13 | 0/13 | -0.151 | -0.086 | 0.064 | 0.231 |

### Watched Component Patches

| key | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta | positive_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_input` | layer_input | False | 13 | 13/13 | 2.923 | 1.004 | -1.919 | 1.000 |
| `attn_out` | attn_out | False | 13 | 0/13 | -0.038 | -0.018 | 0.021 | 0.308 |
| `mlp_out` | mlp_out | False | 13 | 3/13 | 0.120 | 0.008 | -0.112 | 0.846 |
| `final_norm_input` | final_norm_input | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | 1.000 |
| `final_norm_output` | final_norm_output | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | 1.000 |
| `layer_input_random` | layer_input | True | 13 | 0/13 | 0.035 | 0.035 | 0.000 | 0.692 |
| `attn_out_random` | attn_out | True | 13 | 0/13 | -0.023 | -0.024 | -0.002 | 0.462 |
| `mlp_out_random` | mlp_out | True | 13 | 0/13 | -0.151 | -0.086 | 0.064 | 0.231 |
| `final_norm_input_random` | final_norm_input | True | 13 | 1/13 | -0.134 | -0.087 | 0.046 | 0.308 |

### Attention Source Mass Delta

| source | n | repair_minus_base_mass |
|---|---:|---:|
| `prompt_last` | 13 | 0.030 |
| `answer_prefix` | 13 | -0.018 |
| `other` | 13 | -0.017 |
| `rule_value` | 13 | 0.005 |
| `query_relation` | 13 | -0.003 |
| `rule_relation` | 13 | 0.002 |
| `digit1_position` | 13 | 0.000 |
| `object` | 13 | 0.000 |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, probe_layer=27, time_min=1.28

### Best Component Patches

| key | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta | positive_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `final_norm_input` | final_norm_input | False | 37 | 37/37 | 7.049 | 2.048 | -5.001 | 1.000 |
| `final_norm_output` | final_norm_output | False | 37 | 37/37 | 7.049 | 2.048 | -5.001 | 1.000 |
| `layer_input` | layer_input | False | 37 | 37/37 | 6.993 | 2.051 | -4.942 | 1.000 |
| `final_norm_input_random` | final_norm_input | True | 37 | 3/37 | -0.058 | -0.379 | -0.321 | 0.568 |
| `mlp_out_random` | mlp_out | True | 37 | 2/37 | 0.063 | -0.079 | -0.142 | 0.541 |
| `attn_out_random` | attn_out | True | 37 | 1/37 | -0.042 | -0.044 | -0.002 | 0.378 |
| `attn_out` | attn_out | False | 37 | 0/37 | -0.061 | -0.072 | -0.011 | 0.324 |
| `layer_input_random` | layer_input | True | 37 | 0/37 | -0.125 | -0.216 | -0.091 | 0.324 |
| `mlp_out` | mlp_out | False | 37 | 0/37 | -1.370 | -1.524 | -0.154 | 0.027 |

### Watched Component Patches

| key | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta | positive_margin_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer_input` | layer_input | False | 37 | 37/37 | 6.993 | 2.051 | -4.942 | 1.000 |
| `attn_out` | attn_out | False | 37 | 0/37 | -0.061 | -0.072 | -0.011 | 0.324 |
| `mlp_out` | mlp_out | False | 37 | 0/37 | -1.370 | -1.524 | -0.154 | 0.027 |
| `final_norm_input` | final_norm_input | False | 37 | 37/37 | 7.049 | 2.048 | -5.001 | 1.000 |
| `final_norm_output` | final_norm_output | False | 37 | 37/37 | 7.049 | 2.048 | -5.001 | 1.000 |
| `layer_input_random` | layer_input | True | 37 | 0/37 | -0.125 | -0.216 | -0.091 | 0.324 |
| `attn_out_random` | attn_out | True | 37 | 1/37 | -0.042 | -0.044 | -0.002 | 0.378 |
| `mlp_out_random` | mlp_out | True | 37 | 2/37 | 0.063 | -0.079 | -0.142 | 0.541 |
| `final_norm_input_random` | final_norm_input | True | 37 | 3/37 | -0.058 | -0.379 | -0.321 | 0.568 |

### Attention Source Mass Delta

| source | n | repair_minus_base_mass |
|---|---:|---:|
| `other` | 37 | -0.062 |
| `prompt_last` | 37 | 0.040 |
| `answer_prefix` | 37 | 0.011 |
| `rule_value` | 37 | 0.008 |
| `rule_relation` | 37 | 0.003 |
| `query_relation` | 37 | -0.000 |
| `digit1_position` | 37 | 0.000 |
| `object` | 37 | 0.000 |

