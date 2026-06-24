# Phase604 Cross-Model Summary

Final norm and readout acceptance audit.

## qwen3

cases=96, rows=7, target_cases_seen=7, probe_layer=35, betas=[0.25, 0.5, 1.0, 1.5, 2.0], time_min=0.83

### Best Interpolations

| key | kind | beta | n | first_switch | first_margin_gain | full_switch | full_margin_gain | correct_full_delta | old_wrong_full_delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `seq_output_interp|beta2` | seq_output_interp | 2.000 | 7 | 6/7 | 0.000 | 7/7 | 18.246 | 2.625 | -15.621 |
| `seq_input_interp|beta2` | seq_input_interp | 2.000 | 7 | 6/7 | 0.000 | 7/7 | 15.945 | 2.603 | -13.342 |
| `seq_output_interp|beta1.5` | seq_output_interp | 1.500 | 7 | 6/7 | 0.000 | 7/7 | 12.686 | 2.643 | -10.043 |
| `seq_input_interp|beta1.5` | seq_input_interp | 1.500 | 7 | 6/7 | 0.000 | 7/7 | 11.971 | 2.628 | -9.343 |
| `seq_input_interp|beta1` | seq_input_interp | 1.000 | 7 | 6/7 | 0.000 | 7/7 | 7.960 | 2.581 | -5.379 |
| `seq_output_interp|beta1` | seq_output_interp | 1.000 | 7 | 6/7 | 0.000 | 7/7 | 7.960 | 2.581 | -5.379 |
| `seq_input_interp|beta0.5` | seq_input_interp | 0.500 | 7 | 6/7 | 0.000 | 7/7 | 4.042 | 2.263 | -1.779 |
| `seq_output_interp|beta0.5` | seq_output_interp | 0.500 | 7 | 6/7 | 0.000 | 7/7 | 3.934 | 2.220 | -1.714 |
| `seq_input_interp|beta0.25` | seq_input_interp | 0.250 | 7 | 6/7 | 0.000 | 5/7 | 2.038 | 1.581 | -0.456 |
| `seq_output_interp|beta0.25` | seq_output_interp | 0.250 | 7 | 6/7 | 0.000 | 4/7 | 1.966 | 1.518 | -0.448 |
| `seq_output_random|beta2` | seq_output_random | 2.000 | 7 | 6/7 | 0.000 | 3/7 | 0.481 | -0.860 | -1.341 |
| `seq_output_random|beta1.5` | seq_output_random | 1.500 | 7 | 6/7 | 0.000 | 2/7 | 0.392 | -0.490 | -0.883 |
| `seq_output_random|beta1` | seq_output_random | 1.000 | 7 | 6/7 | 0.000 | 2/7 | 0.268 | -0.197 | -0.464 |
| `seq_output_random|beta0.5` | seq_output_random | 0.500 | 7 | 6/7 | 0.000 | 2/7 | 0.107 | -0.064 | -0.171 |
| `seq_input_random|beta2` | seq_input_random | 2.000 | 7 | 6/7 | 0.000 | 1/7 | 0.272 | -0.968 | -1.240 |
| `seq_input_random|beta1.5` | seq_input_random | 1.500 | 7 | 6/7 | 0.000 | 1/7 | 0.180 | -0.571 | -0.751 |
| `seq_input_random|beta1` | seq_input_random | 1.000 | 7 | 6/7 | 0.000 | 1/7 | 0.161 | -0.249 | -0.410 |
| `seq_output_random|beta0.25` | seq_output_random | 0.250 | 7 | 6/7 | 0.000 | 1/7 | 0.053 | -0.035 | -0.089 |
| `seq_input_random|beta0.5` | seq_input_random | 0.500 | 7 | 6/7 | 0.000 | 1/7 | 0.036 | -0.095 | -0.131 |
| `seq_input_random|beta0.25` | seq_input_random | 0.250 | 7 | 6/7 | 0.000 | 1/7 | 0.018 | -0.027 | -0.045 |
| `input_interp|beta0.25` | input_interp | 0.250 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | 0.423 | 0.423 |
| `input_interp|beta0.5` | input_interp | 0.500 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | 0.647 | 0.647 |
| `input_interp|beta1` | input_interp | 1.000 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | 0.808 | 0.808 |
| `input_interp|beta1.5` | input_interp | 1.500 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | 0.840 | 0.840 |

### Watched Interpolations

| key | kind | beta | n | first_switch | first_margin_gain | full_switch | full_margin_gain | correct_full_delta | old_wrong_full_delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `input_interp|beta1` | input_interp | 1.000 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | 0.808 | 0.808 |
| `output_interp|beta1` | output_interp | 1.000 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | 0.808 | 0.808 |
| `input_interp|beta2` | input_interp | 2.000 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | 0.825 | 0.825 |
| `output_interp|beta2` | output_interp | 2.000 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | 0.837 | 0.837 |
| `seq_input_interp|beta1` | seq_input_interp | 1.000 | 7 | 6/7 | 0.000 | 7/7 | 7.960 | 2.581 | -5.379 |
| `seq_output_interp|beta1` | seq_output_interp | 1.000 | 7 | 6/7 | 0.000 | 7/7 | 7.960 | 2.581 | -5.379 |
| `seq_input_interp|beta2` | seq_input_interp | 2.000 | 7 | 6/7 | 0.000 | 7/7 | 15.945 | 2.603 | -13.342 |
| `seq_output_interp|beta2` | seq_output_interp | 2.000 | 7 | 6/7 | 0.000 | 7/7 | 18.246 | 2.625 | -15.621 |
| `input_random|beta1` | input_random | 1.000 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | -0.123 | -0.123 |
| `output_random|beta1` | output_random | 1.000 | 7 | 6/7 | 0.000 | 0/7 | 0.000 | -0.382 | -0.382 |
| `seq_input_random|beta1` | seq_input_random | 1.000 | 7 | 6/7 | 0.000 | 1/7 | 0.161 | -0.249 | -0.410 |
| `seq_output_random|beta1` | seq_output_random | 1.000 | 7 | 6/7 | 0.000 | 2/7 | 0.268 | -0.197 | -0.464 |

### Best Local Readout Deltas

| key | position | component | n | projection_margin | effect_norm | base_norm | repair_norm | cos_base_repair |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `prompt_last|final_norm_input` | prompt_last | final_norm_input | 7 | 4.905 | 146.357 | 508.640 | 551.664 | 0.966 |
| `prompt_last|final_norm_output` | prompt_last | final_norm_output | 7 | 1.256 | 38.853 | 157.809 | 153.891 | 0.969 |
| `query_category|final_norm_input` | query_category | final_norm_input | 7 | 1.034 | 323.314 | 625.628 | 769.937 | 0.913 |
| `query_category|final_norm_output` | query_category | final_norm_output | 7 | 0.134 | 70.654 | 130.196 | 117.350 | 0.842 |

## glm4

cases=96, rows=13, target_cases_seen=13, probe_layer=39, betas=[0.25, 0.5, 1.0, 1.5, 2.0], time_min=1.87

### Best Interpolations

| key | kind | beta | n | first_switch | first_margin_gain | full_switch | full_margin_gain | correct_full_delta | old_wrong_full_delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `seq_output_interp|beta2` | seq_output_interp | 2.000 | 13 | 8/13 | 0.000 | 13/13 | 5.829 | -1.338 | -7.167 |
| `seq_input_interp|beta2` | seq_input_interp | 2.000 | 13 | 8/13 | 0.000 | 13/13 | 5.541 | -1.408 | -6.949 |
| `seq_output_interp|beta1.5` | seq_output_interp | 1.500 | 13 | 8/13 | 0.000 | 13/13 | 4.380 | -0.360 | -4.739 |
| `seq_input_interp|beta1.5` | seq_input_interp | 1.500 | 13 | 8/13 | 0.000 | 13/13 | 4.284 | -0.414 | -4.698 |
| `seq_input_interp|beta1` | seq_input_interp | 1.000 | 13 | 8/13 | 0.000 | 13/13 | 2.913 | 0.330 | -2.584 |
| `seq_output_interp|beta1` | seq_output_interp | 1.000 | 13 | 8/13 | 0.000 | 13/13 | 2.913 | 0.330 | -2.584 |
| `seq_input_interp|beta0.5` | seq_input_interp | 0.500 | 13 | 8/13 | 0.000 | 11/13 | 1.471 | 0.620 | -0.851 |
| `seq_output_interp|beta0.5` | seq_output_interp | 0.500 | 13 | 8/13 | 0.000 | 11/13 | 1.452 | 0.562 | -0.889 |
| `seq_output_random|beta2` | seq_output_random | 2.000 | 13 | 8/13 | 0.000 | 7/13 | 0.659 | -0.916 | -1.575 |
| `seq_input_interp|beta0.25` | seq_input_interp | 0.250 | 13 | 8/13 | 0.000 | 6/13 | 0.736 | 0.434 | -0.302 |
| `seq_output_interp|beta0.25` | seq_output_interp | 0.250 | 13 | 8/13 | 0.000 | 6/13 | 0.726 | 0.399 | -0.327 |
| `seq_output_random|beta1.5` | seq_output_random | 1.500 | 13 | 8/13 | 0.000 | 5/13 | 0.481 | -0.441 | -0.922 |
| `seq_input_random|beta2` | seq_input_random | 2.000 | 13 | 8/13 | 0.000 | 4/13 | 0.654 | -1.242 | -1.895 |
| `seq_input_random|beta1.5` | seq_input_random | 1.500 | 13 | 8/13 | 0.000 | 4/13 | 0.500 | -0.613 | -1.113 |
| `seq_input_random|beta1` | seq_input_random | 1.000 | 13 | 8/13 | 0.000 | 3/13 | 0.327 | -0.200 | -0.527 |
| `seq_output_random|beta1` | seq_output_random | 1.000 | 13 | 8/13 | 0.000 | 3/13 | 0.322 | -0.132 | -0.454 |
| `seq_input_random|beta0.5` | seq_input_random | 0.500 | 13 | 8/13 | 0.000 | 3/13 | 0.168 | 0.004 | -0.165 |
| `seq_output_random|beta0.5` | seq_output_random | 0.500 | 13 | 8/13 | 0.000 | 2/13 | 0.149 | 0.011 | -0.138 |
| `seq_output_random|beta0.25` | seq_output_random | 0.250 | 13 | 8/13 | 0.000 | 2/13 | 0.082 | 0.027 | -0.055 |
| `seq_input_random|beta0.25` | seq_input_random | 0.250 | 13 | 8/13 | 0.000 | 2/13 | 0.072 | 0.029 | -0.043 |
| `input_interp|beta0.25` | input_interp | 0.250 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | 0.082 | 0.082 |
| `input_interp|beta0.5` | input_interp | 0.500 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | -0.045 | -0.045 |
| `input_interp|beta1` | input_interp | 1.000 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | -0.635 | -0.635 |
| `input_interp|beta1.5` | input_interp | 1.500 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | -1.464 | -1.464 |

### Watched Interpolations

| key | kind | beta | n | first_switch | first_margin_gain | full_switch | full_margin_gain | correct_full_delta | old_wrong_full_delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `input_interp|beta1` | input_interp | 1.000 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | -0.635 | -0.635 |
| `output_interp|beta1` | output_interp | 1.000 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | -0.635 | -0.635 |
| `input_interp|beta2` | input_interp | 2.000 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | -2.395 | -2.395 |
| `output_interp|beta2` | output_interp | 2.000 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | -2.400 | -2.400 |
| `seq_input_interp|beta1` | seq_input_interp | 1.000 | 13 | 8/13 | 0.000 | 13/13 | 2.913 | 0.330 | -2.584 |
| `seq_output_interp|beta1` | seq_output_interp | 1.000 | 13 | 8/13 | 0.000 | 13/13 | 2.913 | 0.330 | -2.584 |
| `seq_input_interp|beta2` | seq_input_interp | 2.000 | 13 | 8/13 | 0.000 | 13/13 | 5.541 | -1.408 | -6.949 |
| `seq_output_interp|beta2` | seq_output_interp | 2.000 | 13 | 8/13 | 0.000 | 13/13 | 5.829 | -1.338 | -7.167 |
| `input_random|beta1` | input_random | 1.000 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | -0.106 | -0.106 |
| `output_random|beta1` | output_random | 1.000 | 13 | 8/13 | 0.000 | 0/13 | 0.000 | -0.248 | -0.248 |
| `seq_input_random|beta1` | seq_input_random | 1.000 | 13 | 8/13 | 0.000 | 3/13 | 0.327 | -0.200 | -0.527 |
| `seq_output_random|beta1` | seq_output_random | 1.000 | 13 | 8/13 | 0.000 | 3/13 | 0.322 | -0.132 | -0.454 |

### Best Local Readout Deltas

| key | position | component | n | projection_margin | effect_norm | base_norm | repair_norm | cos_base_repair |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `prompt_last|final_norm_input` | prompt_last | final_norm_input | 13 | 0.509 | 102.693 | 301.060 | 297.069 | 0.940 |
| `prompt_last|final_norm_output` | prompt_last | final_norm_output | 13 | 0.366 | 76.760 | 185.786 | 191.721 | 0.917 |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, probe_layer=27, betas=[1.0, 2.0], time_min=1.82

### Best Interpolations

| key | kind | beta | n | first_switch | first_margin_gain | full_switch | full_margin_gain | correct_full_delta | old_wrong_full_delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `seq_output_interp|beta2` | seq_output_interp | 2.000 | 37 | 8/37 | 0.000 | 37/37 | 17.438 | 8.519 | -8.919 |
| `seq_input_interp|beta2` | seq_input_interp | 2.000 | 37 | 8/37 | 0.000 | 37/37 | 15.285 | 8.382 | -6.903 |
| `seq_input_interp|beta1` | seq_input_interp | 1.000 | 37 | 8/37 | 0.000 | 37/37 | 7.716 | 7.443 | -0.273 |
| `seq_output_interp|beta1` | seq_output_interp | 1.000 | 37 | 8/37 | 0.000 | 37/37 | 7.716 | 7.443 | -0.273 |
| `seq_output_random|beta2` | seq_output_random | 2.000 | 37 | 8/37 | 0.000 | 7/37 | -0.334 | -3.863 | -3.528 |
| `seq_input_random|beta2` | seq_input_random | 2.000 | 37 | 8/37 | 0.000 | 5/37 | 0.122 | -2.504 | -2.626 |
| `seq_output_random|beta1` | seq_output_random | 1.000 | 37 | 8/37 | 0.000 | 4/37 | -0.088 | -0.788 | -0.700 |
| `seq_input_random|beta1` | seq_input_random | 1.000 | 37 | 8/37 | 0.000 | 2/37 | -0.105 | -0.717 | -0.612 |
| `input_interp|beta1` | input_interp | 1.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | 5.549 | 5.549 |
| `input_interp|beta2` | input_interp | 2.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | 6.431 | 6.431 |
| `input_random|beta1` | input_random | 1.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | -0.038 | -0.038 |
| `input_random|beta2` | input_random | 2.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | -0.372 | -0.372 |
| `output_interp|beta1` | output_interp | 1.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | 5.549 | 5.549 |
| `output_interp|beta2` | output_interp | 2.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | 6.496 | 6.496 |
| `output_random|beta1` | output_random | 1.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | -0.603 | -0.603 |
| `output_random|beta2` | output_random | 2.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | -2.014 | -2.014 |

### Watched Interpolations

| key | kind | beta | n | first_switch | first_margin_gain | full_switch | full_margin_gain | correct_full_delta | old_wrong_full_delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `input_interp|beta1` | input_interp | 1.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | 5.549 | 5.549 |
| `output_interp|beta1` | output_interp | 1.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | 5.549 | 5.549 |
| `input_interp|beta2` | input_interp | 2.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | 6.431 | 6.431 |
| `output_interp|beta2` | output_interp | 2.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | 6.496 | 6.496 |
| `seq_input_interp|beta1` | seq_input_interp | 1.000 | 37 | 8/37 | 0.000 | 37/37 | 7.716 | 7.443 | -0.273 |
| `seq_output_interp|beta1` | seq_output_interp | 1.000 | 37 | 8/37 | 0.000 | 37/37 | 7.716 | 7.443 | -0.273 |
| `seq_input_interp|beta2` | seq_input_interp | 2.000 | 37 | 8/37 | 0.000 | 37/37 | 15.285 | 8.382 | -6.903 |
| `seq_output_interp|beta2` | seq_output_interp | 2.000 | 37 | 8/37 | 0.000 | 37/37 | 17.438 | 8.519 | -8.919 |
| `input_random|beta1` | input_random | 1.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | -0.038 | -0.038 |
| `output_random|beta1` | output_random | 1.000 | 37 | 8/37 | 0.000 | 0/37 | 0.000 | -0.603 | -0.603 |
| `seq_input_random|beta1` | seq_input_random | 1.000 | 37 | 8/37 | 0.000 | 2/37 | -0.105 | -0.717 | -0.612 |
| `seq_output_random|beta1` | seq_output_random | 1.000 | 37 | 8/37 | 0.000 | 4/37 | -0.088 | -0.788 | -0.700 |

### Best Local Readout Deltas

| key | position | component | n | projection_margin | effect_norm | base_norm | repair_norm | cos_base_repair |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `rule_value|final_norm_input` | rule_value | final_norm_input | 37 | 4.220 | 999.080 | 1286.103 | 1907.114 | 0.893 |
| `prompt_last|final_norm_input` | prompt_last | final_norm_input | 37 | 1.595 | 547.217 | 1490.005 | 1557.716 | 0.936 |
| `prompt_last|final_norm_output` | prompt_last | final_norm_output | 37 | 0.326 | 70.717 | 194.220 | 194.005 | 0.933 |
| `query_relation|final_norm_input` | query_relation | final_norm_input | 37 | 0.043 | 670.280 | 1713.519 | 1617.243 | 0.921 |
| `rule_value|final_norm_output` | rule_value | final_norm_output | 37 | 0.020 | 87.589 | 181.941 | 190.598 | 0.883 |
| `query_relation|final_norm_output` | query_relation | final_norm_output | 37 | -0.013 | 78.238 | 210.717 | 212.989 | 0.932 |

### DS7B prompt_last local readout

| key | position | component | n | projection_margin | effect_norm | base_norm | repair_norm | cos_base_repair |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `prompt_last|final_norm_input` | prompt_last | final_norm_input | 37 | 1.595 | 547.217 | 1490.005 | 1557.716 | 0.936 |
| `prompt_last|final_norm_output` | prompt_last | final_norm_output | 37 | 0.326 | 70.717 | 194.220 | 194.005 | 0.933 |

