# Phase603 Cross-Model Summary

Post-attention MLP compensation audit.

## qwen3

cases=96, rows=7, target_cases_seen=7, probe_layer=35, alpha=2.0, attn_scale=1.0, mlpout_scale=1.0, time_min=0.89

### Best Diagnostics

| key | mode | part | n | cos_to_natural | norm_ratio | projection_margin |
|---|---|---|---:|---:|---:|---:|
| `prompt_last|L32|mlp_plus_attn_effect|mlp_out` | mlp_plus_attn_effect | `mlp_out` | 7 | 0.691 | 0.795 | 0.082 |
| `prompt_last|L32|mlp_plus_attn_effect|down` | mlp_plus_attn_effect | `down` | 7 | 0.690 | 0.794 | 0.072 |
| `query_category|L32|mlp_plus_attn_effect|down` | mlp_plus_attn_effect | `down` | 7 | 0.667 | 0.641 | 0.195 |
| `query_category|L32|mlp_plus_attn_effect|mlp_out` | mlp_plus_attn_effect | `mlp_out` | 7 | 0.666 | 0.641 | 0.195 |
| `prompt_last|L32|mlp_plus_attn_effect|z` | mlp_plus_attn_effect | `z` | 7 | 0.664 | 0.822 | 0.000 |
| `prompt_last|L32|mlp_random_plus_attn_effect|mlp_out` | mlp_random_plus_attn_effect | `mlp_out` | 7 | 0.647 | 0.770 | 0.005 |
| `query_category|L32|mlp_plus_attn_random|down` | mlp_plus_attn_random | `down` | 7 | 0.647 | 0.574 | 0.200 |
| `prompt_last|L32|mlp_random_plus_attn_effect|down` | mlp_random_plus_attn_effect | `down` | 7 | 0.646 | 0.776 | 0.002 |
| `query_category|L32|mlp_plus_attn_random|mlp_out` | mlp_plus_attn_random | `mlp_out` | 7 | 0.646 | 0.572 | 0.200 |
| `query_category|L32|mlp_repair_only|mlp_out` | mlp_repair_only | `mlp_out` | 7 | 0.627 | 0.545 | 0.104 |
| `query_category|L32|mlp_repair_only|down` | mlp_repair_only | `down` | 7 | 0.627 | 0.544 | 0.106 |
| `prompt_last|L32|mlp_plus_attn_effect|final_norm_output` | mlp_plus_attn_effect | `final_norm_output` | 7 | 0.594 | 0.693 | 0.265 |
| `prompt_last|L32|mlp_plus_attn_effect|layer_out` | mlp_plus_attn_effect | `layer_out` | 7 | 0.593 | 0.687 | 0.941 |
| `prompt_last|L34|mlp_plus_attn_effect|down` | mlp_plus_attn_effect | `down` | 7 | 0.585 | 0.830 | 0.150 |
| `prompt_last|L34|mlp_plus_attn_effect|mlp_out` | mlp_plus_attn_effect | `mlp_out` | 7 | 0.583 | 0.828 | 0.155 |
| `prompt_last|L34|mlp_plus_attn_effect|final_norm_output` | mlp_plus_attn_effect | `final_norm_output` | 7 | 0.572 | 0.769 | 0.272 |
| `prompt_last|L34|mlp_plus_attn_effect|layer_out` | mlp_plus_attn_effect | `layer_out` | 7 | 0.566 | 0.773 | 0.945 |
| `prompt_last|L32|mlp_random_plus_attn_effect|z` | mlp_random_plus_attn_effect | `z` | 7 | 0.526 | 0.712 | 0.000 |
| `query_category|L32|mlp_plus_attn_effect|final_norm_output` | mlp_plus_attn_effect | `final_norm_output` | 7 | 0.511 | 0.670 | 0.173 |
| `prompt_last|L32|mlp_plus_attn_effect|mlp_input` | mlp_plus_attn_effect | `mlp_input` | 7 | 0.505 | 0.645 | 0.126 |
| `prompt_last|L32|mlp_repair_only|z` | mlp_repair_only | `z` | 7 | 0.486 | 0.448 | 0.000 |
| `prompt_last|L32|mlp_plus_attn_effect|gate` | mlp_plus_attn_effect | `gate` | 7 | 0.471 | 0.627 | 0.000 |
| `query_category|L32|mlp_random_plus_attn_effect|mlp_out` | mlp_random_plus_attn_effect | `mlp_out` | 7 | 0.469 | 0.699 | 0.158 |
| `query_category|L32|mlp_random_plus_attn_effect|down` | mlp_random_plus_attn_effect | `down` | 7 | 0.469 | 0.699 | 0.150 |
| `prompt_last|L32|mlp_plus_attn_effect|up` | mlp_plus_attn_effect | `up` | 7 | 0.457 | 0.638 | 0.000 |
| `prompt_last|L34|mlp_plus_attn_effect|z` | mlp_plus_attn_effect | `z` | 7 | 0.456 | 0.858 | 0.000 |
| `prompt_last|L34|mlp_plus_attn_effect|gate` | mlp_plus_attn_effect | `gate` | 7 | 0.453 | 0.807 | 0.000 |
| `query_category|L32|mlp_plus_attn_effect|layer_out` | mlp_plus_attn_effect | `layer_out` | 7 | 0.448 | 0.689 | 0.962 |
| `prompt_last|L32|mlp_repair_only|final_norm_output` | mlp_repair_only | `final_norm_output` | 7 | 0.437 | 0.497 | 0.441 |
| `prompt_last|L32|mlp_repair_only|layer_out` | mlp_repair_only | `layer_out` | 7 | 0.431 | 0.464 | 1.655 |

### Patch Effects

| key | mode | n | switch | full_margin_gain | positive_margin_rate |
|---|---|---:|---:|---:|---:|
| `query_category|L32|mlp_plus_attn_effect` | mlp_plus_attn_effect | 7 | 2/7 | 0.036 | 0.714 |
| `query_category|L32|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 7 | 2/7 | 0.036 | 0.714 |
| `query_category|L32|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 7 | 2/7 | 0.036 | 0.714 |
| `query_category|L32|mlp_repair_only` | mlp_repair_only | 7 | 2/7 | 0.036 | 0.714 |
| `prompt_last|L32|mlp_plus_attn_effect` | mlp_plus_attn_effect | 7 | 1/7 | 0.071 | 0.857 |
| `prompt_last|L32|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 7 | 1/7 | 0.071 | 0.857 |
| `prompt_last|L32|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 7 | 1/7 | 0.071 | 0.857 |
| `prompt_last|L32|mlp_repair_only` | mlp_repair_only | 7 | 1/7 | 0.071 | 0.857 |
| `prompt_last|L32|mlpout_effect_only` | mlpout_effect_only | 7 | 0/7 | 0.000 | 0.000 |
| `prompt_last|L34|mlpout_effect_only` | mlpout_effect_only | 7 | 0/7 | 0.000 | 0.000 |
| `query_category|L32|mlpout_effect_only` | mlpout_effect_only | 7 | 0/7 | 0.000 | 0.000 |
| `prompt_last|L34|mlp_plus_attn_effect` | mlp_plus_attn_effect | 7 | 0/7 | -0.000 | 0.429 |
| `prompt_last|L34|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 7 | 0/7 | -0.000 | 0.429 |
| `prompt_last|L34|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 7 | 0/7 | -0.000 | 0.429 |
| `prompt_last|L34|mlp_repair_only` | mlp_repair_only | 7 | 0/7 | -0.000 | 0.429 |

## glm4

cases=96, rows=13, target_cases_seen=13, probe_layer=39, alpha=2.0, attn_scale=1.0, mlpout_scale=1.0, time_min=2.21

### Best Diagnostics

| key | mode | part | n | cos_to_natural | norm_ratio | projection_margin |
|---|---|---|---:|---:|---:|---:|
| `prompt_last|L39|mlp_plus_attn_effect|mlp_input` | mlp_plus_attn_effect | `mlp_input` | 13 | 1.000 | 2.000 | 0.956 |
| `prompt_last|L39|mlp_plus_attn_random|mlp_input` | mlp_plus_attn_random | `mlp_input` | 13 | 1.000 | 2.000 | 0.956 |
| `prompt_last|L39|mlp_repair_only|mlp_input` | mlp_repair_only | `mlp_input` | 13 | 1.000 | 2.000 | 0.956 |
| `prompt_last|L39|mlp_plus_attn_effect|up` | mlp_plus_attn_effect | `up` | 13 | 1.000 | 2.000 | 0.000 |
| `prompt_last|L39|mlp_plus_attn_random|up` | mlp_plus_attn_random | `up` | 13 | 1.000 | 2.000 | 0.000 |
| `prompt_last|L39|mlp_repair_only|up` | mlp_repair_only | `up` | 13 | 1.000 | 2.000 | 0.000 |
| `prompt_last|L39|mlp_plus_attn_effect|gate` | mlp_plus_attn_effect | `gate` | 13 | 1.000 | 2.000 | 0.000 |
| `prompt_last|L39|mlp_plus_attn_random|gate` | mlp_plus_attn_random | `gate` | 13 | 1.000 | 2.000 | 0.000 |
| `prompt_last|L39|mlp_repair_only|gate` | mlp_repair_only | `gate` | 13 | 1.000 | 2.000 | 0.000 |
| `prompt_last|L39|mlp_plus_attn_effect|mlp_out` | mlp_plus_attn_effect | `mlp_out` | 13 | 0.962 | 2.125 | 0.186 |
| `prompt_last|L39|mlp_plus_attn_random|mlp_out` | mlp_plus_attn_random | `mlp_out` | 13 | 0.962 | 2.125 | 0.186 |
| `prompt_last|L39|mlp_repair_only|mlp_out` | mlp_repair_only | `mlp_out` | 13 | 0.962 | 2.125 | 0.186 |
| `prompt_last|L39|mlp_plus_attn_effect|down` | mlp_plus_attn_effect | `down` | 13 | 0.962 | 2.125 | 0.184 |
| `prompt_last|L39|mlp_plus_attn_random|down` | mlp_plus_attn_random | `down` | 13 | 0.962 | 2.125 | 0.184 |
| `prompt_last|L39|mlp_repair_only|down` | mlp_repair_only | `down` | 13 | 0.962 | 2.125 | 0.184 |
| `prompt_last|L39|mlp_plus_attn_effect|z` | mlp_plus_attn_effect | `z` | 13 | 0.953 | 2.049 | 0.000 |
| `prompt_last|L39|mlp_plus_attn_random|z` | mlp_plus_attn_random | `z` | 13 | 0.953 | 2.049 | 0.000 |
| `prompt_last|L39|mlp_repair_only|z` | mlp_repair_only | `z` | 13 | 0.953 | 2.049 | 0.000 |
| `prompt_last|L38|mlp_plus_attn_effect|final_norm_output` | mlp_plus_attn_effect | `final_norm_output` | 13 | 0.871 | 1.194 | 0.325 |
| `prompt_last|L38|mlp_plus_attn_effect|layer_out` | mlp_plus_attn_effect | `layer_out` | 13 | 0.865 | 1.213 | 0.446 |
| `prompt_last|L38|mlp_repair_only|final_norm_output` | mlp_repair_only | `final_norm_output` | 13 | 0.864 | 1.170 | 0.299 |
| `prompt_last|L38|mlp_plus_attn_random|final_norm_output` | mlp_plus_attn_random | `final_norm_output` | 13 | 0.862 | 1.172 | 0.306 |
| `prompt_last|L38|mlp_repair_only|layer_out` | mlp_repair_only | `layer_out` | 13 | 0.855 | 1.191 | 0.398 |
| `prompt_last|L38|mlp_plus_attn_random|layer_out` | mlp_plus_attn_random | `layer_out` | 13 | 0.853 | 1.194 | 0.412 |
| `prompt_last|L38|mlp_plus_attn_effect|mlp_input` | mlp_plus_attn_effect | `mlp_input` | 13 | 0.847 | 1.205 | 0.228 |
| `prompt_last|L38|mlp_repair_only|mlp_input` | mlp_repair_only | `mlp_input` | 13 | 0.845 | 1.188 | 0.216 |
| `prompt_last|L38|mlp_plus_attn_random|mlp_input` | mlp_plus_attn_random | `mlp_input` | 13 | 0.842 | 1.191 | 0.228 |
| `prompt_last|L38|mlp_plus_attn_effect|up` | mlp_plus_attn_effect | `up` | 13 | 0.809 | 1.232 | 0.000 |
| `prompt_last|L38|mlp_repair_only|up` | mlp_repair_only | `up` | 13 | 0.805 | 1.213 | 0.000 |
| `prompt_last|L38|mlp_plus_attn_random|up` | mlp_plus_attn_random | `up` | 13 | 0.801 | 1.217 | 0.000 |

### Patch Effects

| key | mode | n | switch | full_margin_gain | positive_margin_rate |
|---|---|---:|---:|---:|---:|
| `prompt_last|L38|mlp_plus_attn_effect` | mlp_plus_attn_effect | 13 | 0/13 | 0.005 | 0.385 |
| `prompt_last|L38|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 13 | 0/13 | 0.005 | 0.385 |
| `prompt_last|L38|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 13 | 0/13 | 0.005 | 0.385 |
| `prompt_last|L38|mlp_repair_only` | mlp_repair_only | 13 | 0/13 | 0.005 | 0.385 |
| `prompt_last|L37|mlpout_effect_only` | mlpout_effect_only | 13 | 0/13 | 0.000 | 0.000 |
| `prompt_last|L38|mlpout_effect_only` | mlpout_effect_only | 13 | 0/13 | 0.000 | 0.000 |
| `prompt_last|L39|mlp_plus_attn_effect` | mlp_plus_attn_effect | 13 | 0/13 | 0.000 | 0.000 |
| `prompt_last|L39|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 13 | 0/13 | 0.000 | 0.000 |
| `prompt_last|L39|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 13 | 0/13 | 0.000 | 0.000 |
| `prompt_last|L39|mlp_repair_only` | mlp_repair_only | 13 | 0/13 | 0.000 | 0.000 |
| `prompt_last|L39|mlpout_effect_only` | mlpout_effect_only | 13 | 0/13 | 0.000 | 0.000 |
| `prompt_last|L37|mlp_plus_attn_effect` | mlp_plus_attn_effect | 13 | 0/13 | -0.010 | 0.385 |
| `prompt_last|L37|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 13 | 0/13 | -0.010 | 0.385 |
| `prompt_last|L37|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 13 | 0/13 | -0.010 | 0.385 |
| `prompt_last|L37|mlp_repair_only` | mlp_repair_only | 13 | 0/13 | -0.010 | 0.385 |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, probe_layer=27, alpha=2.0, attn_scale=1.0, mlpout_scale=1.0, time_min=5.72

### Best Diagnostics

| key | mode | part | n | cos_to_natural | norm_ratio | projection_margin |
|---|---|---|---:|---:|---:|---:|
| `rule_value|L26|mlp_plus_attn_effect|up` | mlp_plus_attn_effect | `up` | 37 | 0.676 | 1.059 | 0.000 |
| `rule_value|L26|mlp_plus_attn_effect|gate` | mlp_plus_attn_effect | `gate` | 37 | 0.644 | 1.066 | 0.000 |
| `rule_value|L26|mlp_plus_attn_effect|z` | mlp_plus_attn_effect | `z` | 37 | 0.590 | 0.969 | 0.000 |
| `rule_value|L26|mlp_plus_attn_effect|mlp_out` | mlp_plus_attn_effect | `mlp_out` | 37 | 0.586 | 0.872 | 4.081 |
| `rule_value|L26|mlp_plus_attn_effect|down` | mlp_plus_attn_effect | `down` | 37 | 0.586 | 0.872 | 4.073 |
| `rule_value|L26|mlp_plus_attn_effect|final_norm_output` | mlp_plus_attn_effect | `final_norm_output` | 37 | 0.554 | 1.128 | 0.263 |
| `prompt_last|L26|mlp_plus_attn_effect|final_norm_output` | mlp_plus_attn_effect | `final_norm_output` | 37 | 0.541 | 1.048 | 0.181 |
| `prompt_last|L26|mlp_repair_only|final_norm_output` | mlp_repair_only | `final_norm_output` | 37 | 0.530 | 0.982 | 0.194 |
| `prompt_last|L26|mlp_plus_attn_random|final_norm_output` | mlp_plus_attn_random | `final_norm_output` | 37 | 0.524 | 0.997 | 0.197 |
| `rule_value|L26|mlp_plus_attn_effect|layer_out` | mlp_plus_attn_effect | `layer_out` | 37 | 0.512 | 0.865 | 1.956 |
| `rule_value|L26|mlp_random_plus_attn_effect|up` | mlp_random_plus_attn_effect | `up` | 37 | 0.505 | 1.099 | 0.000 |
| `prompt_last|L26|mlp_repair_only|mlp_out` | mlp_repair_only | `mlp_out` | 37 | 0.496 | 1.119 | -2.589 |
| `prompt_last|L26|mlp_repair_only|down` | mlp_repair_only | `down` | 37 | 0.495 | 1.119 | -2.583 |
| `prompt_last|L26|mlp_plus_attn_random|mlp_out` | mlp_plus_attn_random | `mlp_out` | 37 | 0.494 | 1.129 | -2.750 |
| `prompt_last|L26|mlp_plus_attn_random|down` | mlp_plus_attn_random | `down` | 37 | 0.493 | 1.129 | -2.744 |
| `prompt_last|L26|mlp_plus_attn_effect|mlp_out` | mlp_plus_attn_effect | `mlp_out` | 37 | 0.492 | 1.232 | -2.776 |
| `prompt_last|L26|mlp_plus_attn_effect|down` | mlp_plus_attn_effect | `down` | 37 | 0.491 | 1.232 | -2.777 |
| `rule_value|L26|mlp_plus_attn_effect|mlp_input` | mlp_plus_attn_effect | `mlp_input` | 37 | 0.490 | 1.028 | 0.093 |
| `prompt_last|L26|mlp_plus_attn_effect|z` | mlp_plus_attn_effect | `z` | 37 | 0.488 | 1.256 | 0.000 |
| `rule_value|L26|mlp_random_plus_attn_effect|gate` | mlp_random_plus_attn_effect | `gate` | 37 | 0.482 | 1.104 | 0.000 |
| `prompt_last|L26|mlp_repair_only|z` | mlp_repair_only | `z` | 37 | 0.478 | 1.168 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_random|z` | mlp_plus_attn_random | `z` | 37 | 0.474 | 1.175 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_effect|layer_out` | mlp_plus_attn_effect | `layer_out` | 37 | 0.454 | 1.008 | -0.226 |
| `prompt_last|L26|mlp_repair_only|layer_out` | mlp_repair_only | `layer_out` | 37 | 0.439 | 0.946 | -0.017 |
| `prompt_last|L26|mlp_plus_attn_random|layer_out` | mlp_plus_attn_random | `layer_out` | 37 | 0.431 | 0.969 | -0.136 |
| `rule_value|L26|mlp_repair_only|final_norm_output` | mlp_repair_only | `final_norm_output` | 37 | 0.420 | 0.905 | 0.222 |
| `rule_value|L26|mlp_repair_only|up` | mlp_repair_only | `up` | 37 | 0.413 | 0.601 | 0.000 |
| `rule_value|L26|mlp_random_plus_attn_effect|mlp_out` | mlp_random_plus_attn_effect | `mlp_out` | 37 | 0.404 | 0.926 | 3.877 |
| `rule_value|L26|mlp_repair_only|gate` | mlp_repair_only | `gate` | 37 | 0.404 | 0.605 | 0.000 |
| `rule_value|L26|mlp_random_plus_attn_effect|down` | mlp_random_plus_attn_effect | `down` | 37 | 0.404 | 0.926 | 3.873 |

### Patch Effects

| key | mode | n | switch | full_margin_gain | positive_margin_rate |
|---|---|---:|---:|---:|---:|
| `prompt_last|L26|mlpout_effect_only` | mlpout_effect_only | 37 | 0/37 | 0.000 | 0.000 |
| `query_relation|L19|mlpout_effect_only` | mlpout_effect_only | 37 | 0/37 | 0.000 | 0.000 |
| `rule_value|L26|mlpout_effect_only` | mlpout_effect_only | 37 | 0/37 | 0.000 | 0.000 |
| `rule_value|L26|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.001 | 0.514 |
| `rule_value|L26|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 37 | 0/37 | -0.001 | 0.514 |
| `rule_value|L26|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 37 | 0/37 | -0.001 | 0.514 |
| `rule_value|L26|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.001 | 0.514 |
| `query_relation|L19|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.013 | 0.541 |
| `query_relation|L19|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 37 | 0/37 | -0.013 | 0.541 |
| `query_relation|L19|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 37 | 0/37 | -0.013 | 0.541 |
| `query_relation|L19|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.013 | 0.541 |
| `prompt_last|L26|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.018 | 0.324 |
| `prompt_last|L26|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 37 | 0/37 | -0.018 | 0.324 |
| `prompt_last|L26|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 37 | 0/37 | -0.018 | 0.324 |
| `prompt_last|L26|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.018 | 0.324 |

### DS7B watched diagnostics

| key | mode | part | n | cos_to_natural | norm_ratio | projection_margin |
|---|---|---|---:|---:|---:|---:|
| `rule_value|L26|mlp_repair_only|mlp_input` | mlp_repair_only | `mlp_input` | 37 | 0.324 | 0.704 | 0.131 |
| `rule_value|L26|mlp_repair_only|gate` | mlp_repair_only | `gate` | 37 | 0.404 | 0.605 | 0.000 |
| `rule_value|L26|mlp_repair_only|up` | mlp_repair_only | `up` | 37 | 0.413 | 0.601 | 0.000 |
| `rule_value|L26|mlp_repair_only|z` | mlp_repair_only | `z` | 37 | 0.300 | 0.659 | 0.000 |
| `rule_value|L26|mlp_repair_only|down` | mlp_repair_only | `down` | 37 | 0.271 | 0.624 | -1.761 |
| `rule_value|L26|mlp_repair_only|mlp_out` | mlp_repair_only | `mlp_out` | 37 | 0.272 | 0.625 | -1.768 |
| `rule_value|L26|mlp_repair_only|layer_out` | mlp_repair_only | `layer_out` | 37 | 0.281 | 0.650 | -0.869 |
| `rule_value|L26|mlp_repair_only|final_norm_output` | mlp_repair_only | `final_norm_output` | 37 | 0.420 | 0.905 | 0.222 |
| `rule_value|L26|mlp_plus_attn_effect|mlp_input` | mlp_plus_attn_effect | `mlp_input` | 37 | 0.490 | 1.028 | 0.093 |
| `rule_value|L26|mlp_plus_attn_effect|gate` | mlp_plus_attn_effect | `gate` | 37 | 0.644 | 1.066 | 0.000 |
| `rule_value|L26|mlp_plus_attn_effect|up` | mlp_plus_attn_effect | `up` | 37 | 0.676 | 1.059 | 0.000 |
| `rule_value|L26|mlp_plus_attn_effect|z` | mlp_plus_attn_effect | `z` | 37 | 0.590 | 0.969 | 0.000 |
| `rule_value|L26|mlp_plus_attn_effect|down` | mlp_plus_attn_effect | `down` | 37 | 0.586 | 0.872 | 4.073 |
| `rule_value|L26|mlp_plus_attn_effect|mlp_out` | mlp_plus_attn_effect | `mlp_out` | 37 | 0.586 | 0.872 | 4.081 |
| `rule_value|L26|mlp_plus_attn_effect|layer_out` | mlp_plus_attn_effect | `layer_out` | 37 | 0.512 | 0.865 | 1.956 |
| `rule_value|L26|mlp_plus_attn_effect|final_norm_output` | mlp_plus_attn_effect | `final_norm_output` | 37 | 0.554 | 1.128 | 0.263 |
| `rule_value|L26|mlp_plus_attn_random|mlp_input` | mlp_plus_attn_random | `mlp_input` | 37 | 0.210 | 0.970 | 0.169 |
| `rule_value|L26|mlp_plus_attn_random|gate` | mlp_plus_attn_random | `gate` | 37 | 0.166 | 0.760 | 0.000 |
| `rule_value|L26|mlp_plus_attn_random|up` | mlp_plus_attn_random | `up` | 37 | 0.166 | 0.757 | 0.000 |
| `rule_value|L26|mlp_plus_attn_random|z` | mlp_plus_attn_random | `z` | 37 | 0.141 | 0.736 | 0.000 |
| `rule_value|L26|mlp_plus_attn_random|down` | mlp_plus_attn_random | `down` | 37 | 0.212 | 0.720 | -1.179 |
| `rule_value|L26|mlp_plus_attn_random|mlp_out` | mlp_plus_attn_random | `mlp_out` | 37 | 0.212 | 0.720 | -1.181 |
| `rule_value|L26|mlp_plus_attn_random|layer_out` | mlp_plus_attn_random | `layer_out` | 37 | 0.176 | 0.888 | 0.215 |
| `rule_value|L26|mlp_plus_attn_random|final_norm_output` | mlp_plus_attn_random | `final_norm_output` | 37 | 0.291 | 1.226 | 0.202 |
| `rule_value|L26|mlp_random_plus_attn_effect|mlp_input` | mlp_random_plus_attn_effect | `mlp_input` | 37 | 0.272 | 1.060 | -0.025 |
| `rule_value|L26|mlp_random_plus_attn_effect|gate` | mlp_random_plus_attn_effect | `gate` | 37 | 0.482 | 1.104 | 0.000 |
| `rule_value|L26|mlp_random_plus_attn_effect|up` | mlp_random_plus_attn_effect | `up` | 37 | 0.505 | 1.099 | 0.000 |
| `rule_value|L26|mlp_random_plus_attn_effect|z` | mlp_random_plus_attn_effect | `z` | 37 | 0.375 | 0.920 | 0.000 |
| `rule_value|L26|mlp_random_plus_attn_effect|down` | mlp_random_plus_attn_effect | `down` | 37 | 0.404 | 0.926 | 3.873 |
| `rule_value|L26|mlp_random_plus_attn_effect|mlp_out` | mlp_random_plus_attn_effect | `mlp_out` | 37 | 0.404 | 0.926 | 3.877 |
| `rule_value|L26|mlp_random_plus_attn_effect|layer_out` | mlp_random_plus_attn_effect | `layer_out` | 37 | 0.269 | 0.861 | 1.761 |
| `rule_value|L26|mlp_random_plus_attn_effect|final_norm_output` | mlp_random_plus_attn_effect | `final_norm_output` | 37 | 0.269 | 1.001 | -0.041 |
| `prompt_last|L26|mlp_repair_only|mlp_input` | mlp_repair_only | `mlp_input` | 37 | 0.294 | 1.018 | 0.126 |
| `prompt_last|L26|mlp_repair_only|gate` | mlp_repair_only | `gate` | 37 | 0.201 | 1.093 | 0.000 |
| `prompt_last|L26|mlp_repair_only|up` | mlp_repair_only | `up` | 37 | 0.234 | 1.132 | 0.000 |
| `prompt_last|L26|mlp_repair_only|z` | mlp_repair_only | `z` | 37 | 0.478 | 1.168 | 0.000 |
| `prompt_last|L26|mlp_repair_only|down` | mlp_repair_only | `down` | 37 | 0.495 | 1.119 | -2.583 |
| `prompt_last|L26|mlp_repair_only|mlp_out` | mlp_repair_only | `mlp_out` | 37 | 0.496 | 1.119 | -2.589 |
| `prompt_last|L26|mlp_repair_only|layer_out` | mlp_repair_only | `layer_out` | 37 | 0.439 | 0.946 | -0.017 |
| `prompt_last|L26|mlp_repair_only|final_norm_output` | mlp_repair_only | `final_norm_output` | 37 | 0.530 | 0.982 | 0.194 |
| `prompt_last|L26|mlp_plus_attn_effect|mlp_input` | mlp_plus_attn_effect | `mlp_input` | 37 | 0.313 | 1.065 | 0.132 |
| `prompt_last|L26|mlp_plus_attn_effect|gate` | mlp_plus_attn_effect | `gate` | 37 | 0.231 | 1.132 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_effect|up` | mlp_plus_attn_effect | `up` | 37 | 0.246 | 1.196 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_effect|z` | mlp_plus_attn_effect | `z` | 37 | 0.488 | 1.256 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_effect|down` | mlp_plus_attn_effect | `down` | 37 | 0.491 | 1.232 | -2.777 |
| `prompt_last|L26|mlp_plus_attn_effect|mlp_out` | mlp_plus_attn_effect | `mlp_out` | 37 | 0.492 | 1.232 | -2.776 |
| `prompt_last|L26|mlp_plus_attn_effect|layer_out` | mlp_plus_attn_effect | `layer_out` | 37 | 0.454 | 1.008 | -0.226 |
| `prompt_last|L26|mlp_plus_attn_effect|final_norm_output` | mlp_plus_attn_effect | `final_norm_output` | 37 | 0.541 | 1.048 | 0.181 |
| `prompt_last|L26|mlp_plus_attn_random|mlp_input` | mlp_plus_attn_random | `mlp_input` | 37 | 0.287 | 1.049 | 0.132 |
| `prompt_last|L26|mlp_plus_attn_random|gate` | mlp_plus_attn_random | `gate` | 37 | 0.201 | 1.115 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_random|up` | mlp_plus_attn_random | `up` | 37 | 0.231 | 1.151 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_random|z` | mlp_plus_attn_random | `z` | 37 | 0.474 | 1.175 | 0.000 |
| `prompt_last|L26|mlp_plus_attn_random|down` | mlp_plus_attn_random | `down` | 37 | 0.493 | 1.129 | -2.744 |
| `prompt_last|L26|mlp_plus_attn_random|mlp_out` | mlp_plus_attn_random | `mlp_out` | 37 | 0.494 | 1.129 | -2.750 |
| `prompt_last|L26|mlp_plus_attn_random|layer_out` | mlp_plus_attn_random | `layer_out` | 37 | 0.431 | 0.969 | -0.136 |
| `prompt_last|L26|mlp_plus_attn_random|final_norm_output` | mlp_plus_attn_random | `final_norm_output` | 37 | 0.524 | 0.997 | 0.197 |
| `prompt_last|L26|mlp_random_plus_attn_effect|mlp_input` | mlp_random_plus_attn_effect | `mlp_input` | 37 | 0.055 | 1.127 | 0.057 |
| `prompt_last|L26|mlp_random_plus_attn_effect|gate` | mlp_random_plus_attn_effect | `gate` | 37 | 0.003 | 1.137 | 0.000 |
| `prompt_last|L26|mlp_random_plus_attn_effect|up` | mlp_random_plus_attn_effect | `up` | 37 | 0.015 | 1.140 | 0.000 |
| `prompt_last|L26|mlp_random_plus_attn_effect|z` | mlp_random_plus_attn_effect | `z` | 37 | 0.040 | 1.091 | 0.000 |
| `prompt_last|L26|mlp_random_plus_attn_effect|down` | mlp_random_plus_attn_effect | `down` | 37 | 0.042 | 1.053 | -1.015 |
| `prompt_last|L26|mlp_random_plus_attn_effect|mlp_out` | mlp_random_plus_attn_effect | `mlp_out` | 37 | 0.043 | 1.054 | -1.021 |
| `prompt_last|L26|mlp_random_plus_attn_effect|layer_out` | mlp_random_plus_attn_effect | `layer_out` | 37 | 0.057 | 1.000 | 0.093 |
| `prompt_last|L26|mlp_random_plus_attn_effect|final_norm_output` | mlp_random_plus_attn_effect | `final_norm_output` | 37 | 0.058 | 0.954 | 0.036 |
| `query_relation|L19|mlp_repair_only|mlp_input` | mlp_repair_only | `mlp_input` | 37 | 0.292 | 0.771 | -0.003 |
| `query_relation|L19|mlp_repair_only|gate` | mlp_repair_only | `gate` | 37 | 0.336 | 0.778 | 0.000 |
| `query_relation|L19|mlp_repair_only|up` | mlp_repair_only | `up` | 37 | 0.326 | 0.761 | 0.000 |
| `query_relation|L19|mlp_repair_only|z` | mlp_repair_only | `z` | 37 | 0.353 | 0.737 | 0.000 |
| `query_relation|L19|mlp_repair_only|down` | mlp_repair_only | `down` | 37 | 0.332 | 0.807 | -0.010 |
| `query_relation|L19|mlp_repair_only|mlp_out` | mlp_repair_only | `mlp_out` | 37 | 0.332 | 0.808 | -0.009 |
| `query_relation|L19|mlp_repair_only|layer_out` | mlp_repair_only | `layer_out` | 37 | 0.329 | 0.791 | 0.537 |
| `query_relation|L19|mlp_repair_only|final_norm_output` | mlp_repair_only | `final_norm_output` | 37 | 0.225 | 0.737 | 0.008 |
| `query_relation|L19|mlp_plus_attn_effect|mlp_input` | mlp_plus_attn_effect | `mlp_input` | 37 | 0.289 | 0.833 | 0.004 |
| `query_relation|L19|mlp_plus_attn_effect|gate` | mlp_plus_attn_effect | `gate` | 37 | 0.296 | 0.813 | 0.000 |
| `query_relation|L19|mlp_plus_attn_effect|up` | mlp_plus_attn_effect | `up` | 37 | 0.302 | 0.809 | 0.000 |
| `query_relation|L19|mlp_plus_attn_effect|z` | mlp_plus_attn_effect | `z` | 37 | 0.380 | 0.790 | 0.000 |
| `query_relation|L19|mlp_plus_attn_effect|down` | mlp_plus_attn_effect | `down` | 37 | 0.364 | 0.893 | 0.972 |
| `query_relation|L19|mlp_plus_attn_effect|mlp_out` | mlp_plus_attn_effect | `mlp_out` | 37 | 0.364 | 0.892 | 0.975 |
| `query_relation|L19|mlp_plus_attn_effect|layer_out` | mlp_plus_attn_effect | `layer_out` | 37 | 0.364 | 0.870 | 2.466 |
| `query_relation|L19|mlp_plus_attn_effect|final_norm_output` | mlp_plus_attn_effect | `final_norm_output` | 37 | 0.278 | 0.803 | 0.018 |
| `query_relation|L19|mlp_plus_attn_random|mlp_input` | mlp_plus_attn_random | `mlp_input` | 37 | 0.276 | 0.818 | 0.006 |
| `query_relation|L19|mlp_plus_attn_random|gate` | mlp_plus_attn_random | `gate` | 37 | 0.316 | 0.821 | 0.000 |
| `query_relation|L19|mlp_plus_attn_random|up` | mlp_plus_attn_random | `up` | 37 | 0.308 | 0.805 | 0.000 |
| `query_relation|L19|mlp_plus_attn_random|z` | mlp_plus_attn_random | `z` | 37 | 0.342 | 0.760 | 0.000 |
| `query_relation|L19|mlp_plus_attn_random|down` | mlp_plus_attn_random | `down` | 37 | 0.325 | 0.828 | 0.015 |
| `query_relation|L19|mlp_plus_attn_random|mlp_out` | mlp_plus_attn_random | `mlp_out` | 37 | 0.325 | 0.827 | 0.036 |
| `query_relation|L19|mlp_plus_attn_random|layer_out` | mlp_plus_attn_random | `layer_out` | 37 | 0.315 | 0.824 | 0.745 |
| `query_relation|L19|mlp_plus_attn_random|final_norm_output` | mlp_plus_attn_random | `final_norm_output` | 37 | 0.215 | 0.772 | 0.025 |
| `query_relation|L19|mlp_random_plus_attn_effect|mlp_input` | mlp_random_plus_attn_effect | `mlp_input` | 37 | 0.162 | 1.151 | 0.065 |
| `query_relation|L19|mlp_random_plus_attn_effect|gate` | mlp_random_plus_attn_effect | `gate` | 37 | 0.135 | 1.137 | 0.000 |
| `query_relation|L19|mlp_random_plus_attn_effect|up` | mlp_random_plus_attn_effect | `up` | 37 | 0.148 | 1.123 | 0.000 |
| `query_relation|L19|mlp_random_plus_attn_effect|z` | mlp_random_plus_attn_effect | `z` | 37 | 0.265 | 1.096 | 0.000 |
| `query_relation|L19|mlp_random_plus_attn_effect|down` | mlp_random_plus_attn_effect | `down` | 37 | 0.280 | 1.214 | 2.740 |
| `query_relation|L19|mlp_random_plus_attn_effect|mlp_out` | mlp_random_plus_attn_effect | `mlp_out` | 37 | 0.281 | 1.215 | 2.732 |
| `query_relation|L19|mlp_random_plus_attn_effect|layer_out` | mlp_random_plus_attn_effect | `layer_out` | 37 | 0.270 | 1.170 | 5.258 |
| `query_relation|L19|mlp_random_plus_attn_effect|final_norm_output` | mlp_random_plus_attn_effect | `final_norm_output` | 37 | 0.170 | 1.122 | 0.139 |

### DS7B watched patch effects

| key | mode | n | switch | full_margin_gain | positive_margin_rate |
|---|---|---:|---:|---:|---:|
| `rule_value|L26|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.001 | 0.514 |
| `rule_value|L26|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.001 | 0.514 |
| `rule_value|L26|mlpout_effect_only` | mlpout_effect_only | 37 | 0/37 | 0.000 | 0.000 |
| `rule_value|L26|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 37 | 0/37 | -0.001 | 0.514 |
| `rule_value|L26|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 37 | 0/37 | -0.001 | 0.514 |
| `prompt_last|L26|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.018 | 0.324 |
| `prompt_last|L26|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.018 | 0.324 |
| `prompt_last|L26|mlpout_effect_only` | mlpout_effect_only | 37 | 0/37 | 0.000 | 0.000 |
| `prompt_last|L26|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 37 | 0/37 | -0.018 | 0.324 |
| `prompt_last|L26|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 37 | 0/37 | -0.018 | 0.324 |
| `query_relation|L19|mlp_repair_only` | mlp_repair_only | 37 | 0/37 | -0.013 | 0.541 |
| `query_relation|L19|mlp_plus_attn_effect` | mlp_plus_attn_effect | 37 | 0/37 | -0.013 | 0.541 |
| `query_relation|L19|mlpout_effect_only` | mlpout_effect_only | 37 | 0/37 | 0.000 | 0.000 |
| `query_relation|L19|mlp_plus_mlpout_effect` | mlp_plus_mlpout_effect | 37 | 0/37 | -0.013 | 0.541 |
| `query_relation|L19|mlp_plus_attn_plus_mlpout_effect` | mlp_plus_attn_plus_mlpout_effect | 37 | 0/37 | -0.013 | 0.541 |

