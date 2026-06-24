# Phase600 Cross-Model Summary

Final-layer acceptance rule audit: natural correct/wrong trajectories vs artificial repair/random/wrong trajectories.

## qwen3

cases=96, rows=7, target_cases_seen=7, probe_layer=35, alpha=2.0, capture_attn=True, time_min=0.80

### Artificial Final Effects

| key | trajectory | n | switch | generated_down_projection | full_margin_gain |
|---|---|---:|---:|---:|---:|
| `query_category|L32|artificial_repair` | artificial_repair | 7 | 2/7 | 1.214 | 0.036 |
| `prompt_last|L32|artificial_repair` | artificial_repair | 7 | 1/7 | 0.974 | 0.071 |
| `prompt_last|L32|artificial_wrong` | artificial_wrong | 7 | 1/7 | 0.516 | 0.054 |
| `prompt_last|L34|artificial_wrong` | artificial_wrong | 7 | 1/7 | 0.453 | 0.054 |
| `query_category|L32|artificial_wrong` | artificial_wrong | 7 | 1/7 | 1.448 | 0.018 |
| `prompt_last|L34|artificial_random` | artificial_random | 7 | 1/7 | -0.951 | -0.018 |
| `prompt_last|L32|artificial_random` | artificial_random | 7 | 0/7 | 0.195 | 0.071 |
| `query_category|L32|artificial_random` | artificial_random | 7 | 0/7 | 1.199 | -0.000 |
| `prompt_last|L34|artificial_repair` | artificial_repair | 7 | 0/7 | 1.274 | -0.000 |

### Best Projection Components

| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L32|natural_correct|layer_input` | natural_correct | `layer_input` | 7 | 5.321 | 114.330 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L34|natural_correct|layer_input` | natural_correct | `layer_input` | 7 | 5.321 | 114.330 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L32|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 7 | 4.905 | 146.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L32|natural_correct|layer_out` | natural_correct | `layer_out` | 7 | 4.905 | 146.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L34|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 7 | 4.905 | 146.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L34|natural_correct|layer_out` | natural_correct | `layer_out` | 7 | 4.905 | 146.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L32|natural_wrong|layer_input` | natural_wrong | `layer_input` | 7 | 3.234 | 119.515 | 0.655 | 1.048 | 0.857 | 0.000 |
| `prompt_last|L34|natural_wrong|layer_input` | natural_wrong | `layer_input` | 7 | 3.234 | 119.515 | 0.655 | 1.048 | 0.857 | 0.000 |
| `prompt_last|L32|natural_wrong|final_norm_input` | natural_wrong | `final_norm_input` | 7 | 3.141 | 182.268 | 0.751 | 1.246 | 0.857 | 0.000 |
| `prompt_last|L32|natural_wrong|layer_out` | natural_wrong | `layer_out` | 7 | 3.141 | 182.268 | 0.751 | 1.246 | 0.857 | 0.000 |
| `prompt_last|L34|natural_wrong|final_norm_input` | natural_wrong | `final_norm_input` | 7 | 3.141 | 182.268 | 0.751 | 1.246 | 0.857 | 0.000 |
| `prompt_last|L34|natural_wrong|layer_out` | natural_wrong | `layer_out` | 7 | 3.141 | 182.268 | 0.751 | 1.246 | 0.857 | 0.000 |
| `prompt_last|L34|artificial_repair|final_norm_input` | artificial_repair | `final_norm_input` | 7 | 1.672 | 88.155 | 0.375 | 0.602 | 0.857 | 0.000 |
| `prompt_last|L34|artificial_repair|layer_out` | artificial_repair | `layer_out` | 7 | 1.672 | 88.155 | 0.375 | 0.602 | 0.857 | 0.000 |
| `prompt_last|L32|artificial_repair|final_norm_input` | artificial_repair | `final_norm_input` | 7 | 1.655 | 67.561 | 0.431 | 0.464 | 0.857 | 0.000 |
| `prompt_last|L32|artificial_repair|layer_out` | artificial_repair | `layer_out` | 7 | 1.655 | 67.561 | 0.431 | 0.464 | 0.857 | 0.000 |
| `query_category|L32|natural_wrong|final_norm_input` | natural_wrong | `final_norm_input` | 7 | 1.622 | 324.216 | 0.965 | 1.003 | 0.571 | 0.000 |
| `query_category|L32|natural_wrong|layer_out` | natural_wrong | `layer_out` | 7 | 1.622 | 324.216 | 0.965 | 1.003 | 0.571 | 0.000 |
| `prompt_last|L32|artificial_repair|layer_input` | artificial_repair | `layer_input` | 7 | 1.510 | 60.776 | 0.390 | 0.533 | 1.000 | 0.000 |
| `query_category|L32|natural_wrong|layer_input` | natural_wrong | `layer_input` | 7 | 1.437 | 264.260 | 0.959 | 0.996 | 0.571 | 0.000 |
| `query_category|L32|natural_correct|layer_input` | natural_correct | `layer_input` | 7 | 1.273 | 265.411 | 1.000 | 1.000 | 0.714 | 0.000 |
| `prompt_last|L34|artificial_repair|layer_input` | artificial_repair | `layer_input` | 7 | 1.271 | 78.460 | 0.261 | 0.686 | 0.857 | 0.000 |
| `query_category|L32|artificial_wrong|final_norm_input` | artificial_wrong | `final_norm_input` | 7 | 1.258 | 200.176 | 0.348 | 0.621 | 0.857 | 0.000 |
| `query_category|L32|artificial_wrong|layer_out` | artificial_wrong | `layer_out` | 7 | 1.258 | 200.176 | 0.348 | 0.621 | 0.857 | 0.000 |
| `prompt_last|L32|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 7 | 1.256 | 38.853 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L34|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 7 | 1.256 | 38.853 | 1.000 | 1.000 | 0.857 | 0.000 |
| `query_category|L32|artificial_wrong|layer_input` | artificial_wrong | `layer_input` | 7 | 1.084 | 174.954 | 0.217 | 0.660 | 0.857 | 0.000 |
| `query_category|L32|artificial_repair|final_norm_input` | artificial_repair | `final_norm_input` | 7 | 1.075 | 198.607 | 0.356 | 0.616 | 0.857 | 0.000 |

### Best Natural Alignment Components

| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `query_category|L32|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 7 | 1.034 | 323.314 | 1.000 | 1.000 | 0.571 | 0.000 |
| `query_category|L32|natural_correct|layer_out` | natural_correct | `layer_out` | 7 | 1.034 | 323.314 | 1.000 | 1.000 | 0.571 | 0.000 |
| `query_category|L32|natural_correct|layer_input` | natural_correct | `layer_input` | 7 | 1.273 | 265.411 | 1.000 | 1.000 | 0.714 | 0.000 |
| `query_category|L32|natural_correct|mlp_out` | natural_correct | `mlp_out` | 7 | -0.034 | 141.084 | 1.000 | 1.000 | 0.571 | 0.000 |
| `prompt_last|L32|natural_correct|layer_input` | natural_correct | `layer_input` | 7 | 5.321 | 114.330 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L34|natural_correct|layer_input` | natural_correct | `layer_input` | 7 | 5.321 | 114.330 | 1.000 | 1.000 | 0.857 | 0.000 |
| `query_category|L32|natural_correct|mlp_input` | natural_correct | `mlp_input` | 7 | 0.106 | 44.487 | 1.000 | 1.000 | 0.857 | 0.000 |
| `query_category|L32|natural_correct|attn_out` | natural_correct | `attn_out` | 7 | -0.206 | 72.940 | 1.000 | 1.000 | 0.429 | 0.891 |
| `query_category|L32|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 7 | 0.134 | 70.654 | 1.000 | 1.000 | 0.714 | 0.000 |
| `prompt_last|L32|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 7 | 4.905 | 146.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L32|natural_correct|layer_out` | natural_correct | `layer_out` | 7 | 4.905 | 146.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L34|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 7 | 4.905 | 146.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L34|natural_correct|layer_out` | natural_correct | `layer_out` | 7 | 4.905 | 146.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L32|natural_correct|mlp_out` | natural_correct | `mlp_out` | 7 | 0.265 | 64.955 | 1.000 | 1.000 | 1.000 | 0.000 |
| `prompt_last|L34|natural_correct|mlp_out` | natural_correct | `mlp_out` | 7 | 0.265 | 64.955 | 1.000 | 1.000 | 1.000 | 0.000 |
| `prompt_last|L32|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 7 | 1.256 | 38.853 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L34|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 7 | 1.256 | 38.853 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L32|natural_correct|mlp_input` | natural_correct | `mlp_input` | 7 | 0.637 | 19.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L34|natural_correct|mlp_input` | natural_correct | `mlp_input` | 7 | 0.637 | 19.357 | 1.000 | 1.000 | 0.857 | 0.000 |
| `prompt_last|L32|natural_correct|attn_out` | natural_correct | `attn_out` | 7 | -0.678 | 42.301 | 1.000 | 1.000 | 0.143 | 0.811 |
| `prompt_last|L34|natural_correct|attn_out` | natural_correct | `attn_out` | 7 | -0.678 | 42.301 | 1.000 | 1.000 | 0.143 | 0.811 |
| `query_category|L32|natural_wrong|mlp_out` | natural_wrong | `mlp_out` | 7 | 0.206 | 143.187 | 0.982 | 1.019 | 0.571 | 0.000 |
| `query_category|L32|natural_wrong|final_norm_output` | natural_wrong | `final_norm_output` | 7 | 0.249 | 71.522 | 0.971 | 1.012 | 0.571 | 0.000 |
| `query_category|L32|natural_wrong|mlp_input` | natural_wrong | `mlp_input` | 7 | 0.154 | 45.020 | 0.970 | 1.012 | 0.714 | 0.000 |
| `query_category|L32|natural_wrong|attn_out` | natural_wrong | `attn_out` | 7 | -0.037 | 72.857 | 0.969 | 0.999 | 0.286 | 0.878 |
| `query_category|L32|natural_wrong|final_norm_input` | natural_wrong | `final_norm_input` | 7 | 1.622 | 324.216 | 0.965 | 1.003 | 0.571 | 0.000 |
| `query_category|L32|natural_wrong|layer_out` | natural_wrong | `layer_out` | 7 | 1.622 | 324.216 | 0.965 | 1.003 | 0.571 | 0.000 |
| `query_category|L32|natural_wrong|layer_input` | natural_wrong | `layer_input` | 7 | 1.437 | 264.260 | 0.959 | 0.996 | 0.571 | 0.000 |

## glm4

cases=96, rows=13, target_cases_seen=13, probe_layer=39, alpha=2.0, capture_attn=True, time_min=1.81

### Artificial Final Effects

| key | trajectory | n | switch | generated_down_projection | full_margin_gain |
|---|---|---:|---:|---:|---:|
| `prompt_last|L38|artificial_wrong` | artificial_wrong | 13 | 1/13 | -0.148 | 0.005 |
| `prompt_last|L38|artificial_random` | artificial_random | 13 | 0/13 | -0.168 | 0.005 |
| `prompt_last|L38|artificial_repair` | artificial_repair | 13 | 0/13 | 0.149 | 0.005 |
| `prompt_last|L39|artificial_random` | artificial_random | 13 | 0/13 | 0.495 | 0.000 |
| `prompt_last|L39|artificial_repair` | artificial_repair | 13 | 0/13 | 0.184 | 0.000 |
| `prompt_last|L39|artificial_wrong` | artificial_wrong | 13 | 0/13 | 0.169 | 0.000 |
| `prompt_last|L37|artificial_repair` | artificial_repair | 13 | 0/13 | 0.034 | -0.010 |
| `prompt_last|L37|artificial_wrong` | artificial_wrong | 13 | 0/13 | 0.068 | -0.010 |
| `prompt_last|L37|artificial_random` | artificial_random | 13 | 0/13 | 0.079 | -0.014 |

### Best Projection Components

| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L39|artificial_repair|mlp_input` | artificial_repair | `mlp_input` | 13 | 0.956 | 178.090 | 1.000 | 2.000 | 0.462 | 0.000 |
| `prompt_last|L37|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L37|natural_correct|layer_out` | natural_correct | `layer_out` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L38|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L38|natural_correct|layer_out` | natural_correct | `layer_out` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L39|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L39|natural_correct|layer_out` | natural_correct | `layer_out` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L39|artificial_random|mlp_out` | artificial_random | `mlp_out` | 13 | 0.494 | 116.286 | 0.036 | 2.399 | 0.846 | 0.000 |
| `prompt_last|L39|artificial_random|final_norm_input` | artificial_random | `final_norm_input` | 13 | 0.494 | 116.251 | 0.023 | 1.120 | 0.846 | 0.000 |
| `prompt_last|L39|artificial_random|layer_out` | artificial_random | `layer_out` | 13 | 0.494 | 116.251 | 0.023 | 1.120 | 0.846 | 0.000 |
| `prompt_last|L37|natural_correct|mlp_input` | natural_correct | `mlp_input` | 13 | 0.478 | 89.047 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L38|natural_correct|mlp_input` | natural_correct | `mlp_input` | 13 | 0.478 | 89.047 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L39|natural_correct|mlp_input` | natural_correct | `mlp_input` | 13 | 0.478 | 89.047 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L38|artificial_repair|final_norm_input` | artificial_repair | `final_norm_input` | 13 | 0.398 | 121.790 | 0.855 | 1.191 | 0.769 | 0.000 |
| `prompt_last|L38|artificial_repair|layer_out` | artificial_repair | `layer_out` | 13 | 0.398 | 121.790 | 0.855 | 1.191 | 0.769 | 0.000 |
| `prompt_last|L37|natural_correct|layer_input` | natural_correct | `layer_input` | 13 | 0.393 | 80.443 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L38|natural_correct|layer_input` | natural_correct | `layer_input` | 13 | 0.393 | 80.443 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L39|natural_correct|layer_input` | natural_correct | `layer_input` | 13 | 0.393 | 80.443 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L37|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 13 | 0.366 | 76.760 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L38|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 13 | 0.366 | 76.760 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L39|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 13 | 0.366 | 76.760 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L38|artificial_repair|final_norm_output` | artificial_repair | `final_norm_output` | 13 | 0.299 | 89.411 | 0.864 | 1.170 | 0.769 | 0.000 |
| `prompt_last|L39|artificial_random|final_norm_output` | artificial_random | `final_norm_output` | 13 | 0.271 | 76.640 | 0.029 | 1.002 | 0.846 | 0.000 |
| `prompt_last|L38|artificial_repair|mlp_input` | artificial_repair | `mlp_input` | 13 | 0.216 | 105.404 | 0.845 | 1.188 | 0.692 | 0.000 |
| `prompt_last|L38|artificial_repair|mlp_out` | artificial_repair | `mlp_out` | 13 | 0.215 | 61.143 | 0.755 | 1.270 | 0.846 | 0.000 |
| `prompt_last|L39|artificial_repair|mlp_out` | artificial_repair | `mlp_out` | 13 | 0.186 | 103.062 | 0.962 | 2.125 | 0.769 | 0.000 |
| `prompt_last|L39|artificial_repair|final_norm_input` | artificial_repair | `final_norm_input` | 13 | 0.186 | 103.055 | 0.603 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L39|artificial_repair|layer_out` | artificial_repair | `layer_out` | 13 | 0.186 | 103.055 | 0.603 | 1.000 | 0.692 | 0.000 |

### Best Natural Alignment Components

| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt_last|L37|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L37|natural_correct|layer_out` | natural_correct | `layer_out` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L38|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L38|natural_correct|layer_out` | natural_correct | `layer_out` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L39|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L39|natural_correct|layer_out` | natural_correct | `layer_out` | 13 | 0.509 | 102.693 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L37|natural_correct|layer_input` | natural_correct | `layer_input` | 13 | 0.393 | 80.443 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L38|natural_correct|layer_input` | natural_correct | `layer_input` | 13 | 0.393 | 80.443 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L39|natural_correct|layer_input` | natural_correct | `layer_input` | 13 | 0.393 | 80.443 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L37|natural_correct|mlp_out` | natural_correct | `mlp_out` | 13 | 0.108 | 48.317 | 1.000 | 1.000 | 0.769 | 0.000 |
| `prompt_last|L38|natural_correct|mlp_out` | natural_correct | `mlp_out` | 13 | 0.108 | 48.317 | 1.000 | 1.000 | 0.769 | 0.000 |
| `prompt_last|L39|natural_correct|mlp_out` | natural_correct | `mlp_out` | 13 | 0.108 | 48.317 | 1.000 | 1.000 | 0.769 | 0.000 |
| `prompt_last|L37|natural_correct|mlp_input` | natural_correct | `mlp_input` | 13 | 0.478 | 89.047 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L38|natural_correct|mlp_input` | natural_correct | `mlp_input` | 13 | 0.478 | 89.047 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L39|natural_correct|mlp_input` | natural_correct | `mlp_input` | 13 | 0.478 | 89.047 | 1.000 | 1.000 | 0.462 | 0.000 |
| `prompt_last|L37|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 13 | 0.366 | 76.760 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L38|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 13 | 0.366 | 76.760 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L39|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 13 | 0.366 | 76.760 | 1.000 | 1.000 | 0.692 | 0.000 |
| `prompt_last|L37|natural_correct|attn_out` | natural_correct | `attn_out` | 13 | 0.009 | 7.633 | 1.000 | 1.000 | 0.615 | 1.565 |
| `prompt_last|L38|natural_correct|attn_out` | natural_correct | `attn_out` | 13 | 0.009 | 7.633 | 1.000 | 1.000 | 0.615 | 1.565 |
| `prompt_last|L39|natural_correct|attn_out` | natural_correct | `attn_out` | 13 | 0.009 | 7.633 | 1.000 | 1.000 | 0.615 | 1.565 |
| `prompt_last|L39|artificial_repair|mlp_input` | artificial_repair | `mlp_input` | 13 | 0.956 | 178.090 | 1.000 | 2.000 | 0.462 | 0.000 |
| `prompt_last|L39|artificial_repair|mlp_out` | artificial_repair | `mlp_out` | 13 | 0.186 | 103.062 | 0.962 | 2.125 | 0.769 | 0.000 |
| `prompt_last|L38|artificial_repair|final_norm_output` | artificial_repair | `final_norm_output` | 13 | 0.299 | 89.411 | 0.864 | 1.170 | 0.769 | 0.000 |
| `prompt_last|L38|artificial_repair|final_norm_input` | artificial_repair | `final_norm_input` | 13 | 0.398 | 121.790 | 0.855 | 1.191 | 0.769 | 0.000 |
| `prompt_last|L38|artificial_repair|layer_out` | artificial_repair | `layer_out` | 13 | 0.398 | 121.790 | 0.855 | 1.191 | 0.769 | 0.000 |
| `prompt_last|L38|artificial_repair|mlp_input` | artificial_repair | `mlp_input` | 13 | 0.216 | 105.404 | 0.845 | 1.188 | 0.692 | 0.000 |
| `prompt_last|L38|artificial_repair|layer_input` | artificial_repair | `layer_input` | 13 | 0.149 | 97.354 | 0.840 | 1.213 | 0.538 | 0.000 |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, probe_layer=27, alpha=2.0, capture_attn=True, time_min=3.38

### Artificial Final Effects

| key | trajectory | n | switch | generated_down_projection | full_margin_gain |
|---|---|---:|---:|---:|---:|
| `rule_value|L26|artificial_random` | artificial_random | 17 | 0/17 | -0.285 | 0.016 |
| `rule_value|L26|artificial_repair` | artificial_repair | 17 | 0/17 | -0.888 | 0.009 |
| `rule_value|L26|artificial_wrong` | artificial_wrong | 17 | 0/17 | -1.797 | -0.007 |
| `query_relation|L19|artificial_repair` | artificial_repair | 37 | 0/37 | -0.091 | -0.013 |
| `query_relation|L19|artificial_wrong` | artificial_wrong | 37 | 0/37 | -0.102 | -0.014 |
| `prompt_last|L26|artificial_repair` | artificial_repair | 37 | 0/37 | 3.084 | -0.018 |
| `prompt_last|L26|artificial_random` | artificial_random | 37 | 0/37 | 2.998 | -0.020 |
| `prompt_last|L26|artificial_wrong` | artificial_wrong | 37 | 0/37 | 1.149 | -0.023 |
| `query_relation|L19|artificial_random` | artificial_random | 37 | 0/37 | -0.394 | -0.078 |

### Best Projection Components

| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|natural_wrong|mlp_out` | natural_wrong | `mlp_out` | 17 | 25.853 | 1163.051 | 0.652 | 3.465 | 0.706 | 0.000 |
| `rule_value|L26|natural_correct|mlp_out` | natural_correct | `mlp_out` | 17 | 19.472 | 1105.300 | 1.000 | 1.000 | 0.706 | 0.000 |
| `rule_value|L26|natural_wrong|final_norm_input` | natural_wrong | `final_norm_input` | 17 | 9.454 | 1012.045 | 0.567 | 1.814 | 0.941 | 0.000 |
| `rule_value|L26|natural_wrong|layer_out` | natural_wrong | `layer_out` | 17 | 9.454 | 1012.045 | 0.567 | 1.814 | 0.941 | 0.000 |
| `rule_value|L26|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 17 | 7.778 | 1019.505 | 1.000 | 1.000 | 1.000 | 0.000 |
| `rule_value|L26|natural_correct|layer_out` | natural_correct | `layer_out` | 17 | 7.778 | 1019.505 | 1.000 | 1.000 | 1.000 | 0.000 |
| `query_relation|L19|artificial_random|final_norm_input` | artificial_random | `final_norm_input` | 37 | 3.542 | 725.645 | 0.238 | 1.090 | 0.676 | 0.000 |
| `query_relation|L19|artificial_random|layer_out` | artificial_random | `layer_out` | 37 | 3.542 | 725.645 | 0.238 | 1.090 | 0.676 | 0.000 |
| `prompt_last|L26|artificial_repair|layer_input` | artificial_repair | `layer_input` | 37 | 3.088 | 422.619 | 0.262 | 1.060 | 0.730 | 0.000 |
| `prompt_last|L26|artificial_random|layer_input` | artificial_random | `layer_input` | 37 | 2.989 | 474.051 | -0.024 | 1.190 | 0.649 | 0.000 |
| `rule_value|L26|artificial_random|final_norm_input` | artificial_random | `final_norm_input` | 17 | 2.857 | 481.778 | 0.025 | 0.663 | 0.529 | 0.000 |
| `rule_value|L26|artificial_random|layer_out` | artificial_random | `layer_out` | 17 | 2.857 | 481.778 | 0.025 | 0.663 | 0.529 | 0.000 |
| `rule_value|L26|artificial_random|mlp_out` | artificial_random | `mlp_out` | 17 | 2.426 | 266.926 | 0.122 | 0.585 | 0.647 | 0.000 |
| `prompt_last|L26|natural_correct|layer_input` | natural_correct | `layer_input` | 37 | 2.378 | 397.770 | 1.000 | 1.000 | 0.568 | 0.000 |
| `query_relation|L19|artificial_random|mlp_out` | artificial_random | `mlp_out` | 37 | 1.937 | 409.251 | 0.258 | 1.126 | 0.595 | 0.000 |
| `prompt_last|L26|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 37 | 1.595 | 547.217 | 1.000 | 1.000 | 0.541 | 0.000 |
| `prompt_last|L26|natural_correct|layer_out` | natural_correct | `layer_out` | 37 | 1.595 | 547.217 | 1.000 | 1.000 | 0.541 | 0.000 |
| `query_relation|L19|artificial_random|layer_input` | artificial_random | `layer_input` | 37 | 1.280 | 540.986 | 0.179 | 1.073 | 0.676 | 0.000 |
| `prompt_last|L26|artificial_wrong|layer_input` | artificial_wrong | `layer_input` | 37 | 1.138 | 280.984 | 0.241 | 0.707 | 0.595 | 0.000 |
| `query_relation|L19|natural_correct|attn_out` | natural_correct | `attn_out` | 37 | 0.923 | 132.137 | 1.000 | 1.000 | 0.595 | 0.841 |
| `rule_value|L26|artificial_wrong|attn_out` | artificial_wrong | `attn_out` | 17 | 0.817 | 160.402 | -0.342 | 0.821 | 0.706 | 0.351 |
| `rule_value|L26|artificial_repair|attn_out` | artificial_repair | `attn_out` | 17 | 0.775 | 166.081 | -0.250 | 0.749 | 0.706 | 0.344 |
| `query_relation|L19|natural_wrong|attn_out` | natural_wrong | `attn_out` | 37 | 0.655 | 131.968 | 0.701 | 1.032 | 0.568 | 0.794 |
| `rule_value|L26|natural_wrong|layer_input` | natural_wrong | `layer_input` | 17 | 0.649 | 366.845 | 0.590 | 1.054 | 0.529 | 0.000 |
| `rule_value|L26|artificial_random|attn_out` | artificial_random | `attn_out` | 17 | 0.634 | 119.160 | -0.169 | 0.563 | 0.706 | 0.305 |
| `query_relation|L19|artificial_repair|final_norm_input` | artificial_repair | `final_norm_input` | 37 | 0.537 | 527.998 | 0.329 | 0.791 | 0.541 | 0.000 |
| `query_relation|L19|artificial_repair|layer_out` | artificial_repair | `layer_out` | 37 | 0.537 | 527.998 | 0.329 | 0.791 | 0.541 | 0.000 |
| `query_relation|L19|artificial_random|attn_out` | artificial_random | `attn_out` | 37 | 0.329 | 113.382 | 0.316 | 0.894 | 0.514 | 0.381 |

### Best Natural Alignment Components

| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `query_relation|L19|natural_correct|layer_input` | natural_correct | `layer_input` | 37 | -0.641 | 504.599 | 1.000 | 1.000 | 0.405 | 0.000 |
| `rule_value|L26|natural_correct|layer_input` | natural_correct | `layer_input` | 17 | 0.039 | 368.981 | 1.000 | 1.000 | 0.353 | 0.000 |
| `prompt_last|L26|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 37 | 1.595 | 547.217 | 1.000 | 1.000 | 0.541 | 0.000 |
| `prompt_last|L26|natural_correct|layer_out` | natural_correct | `layer_out` | 37 | 1.595 | 547.217 | 1.000 | 1.000 | 0.541 | 0.000 |
| `prompt_last|L26|natural_correct|layer_input` | natural_correct | `layer_input` | 37 | 2.378 | 397.770 | 1.000 | 1.000 | 0.568 | 0.000 |
| `query_relation|L19|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 37 | 0.043 | 670.280 | 1.000 | 1.000 | 0.514 | 0.000 |
| `query_relation|L19|natural_correct|layer_out` | natural_correct | `layer_out` | 37 | 0.043 | 670.280 | 1.000 | 1.000 | 0.514 | 0.000 |
| `prompt_last|L26|natural_correct|attn_out` | natural_correct | `attn_out` | 37 | 0.022 | 100.075 | 1.000 | 1.000 | 0.459 | 1.103 |
| `prompt_last|L26|natural_correct|mlp_input` | natural_correct | `mlp_input` | 37 | 0.204 | 30.148 | 1.000 | 1.000 | 0.649 | 0.000 |
| `rule_value|L26|natural_correct|final_norm_input` | natural_correct | `final_norm_input` | 17 | 7.778 | 1019.505 | 1.000 | 1.000 | 1.000 | 0.000 |
| `rule_value|L26|natural_correct|layer_out` | natural_correct | `layer_out` | 17 | 7.778 | 1019.505 | 1.000 | 1.000 | 1.000 | 0.000 |
| `query_relation|L19|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 37 | -0.013 | 78.238 | 1.000 | 1.000 | 0.432 | 0.000 |
| `query_relation|L19|natural_correct|attn_out` | natural_correct | `attn_out` | 37 | 0.923 | 132.137 | 1.000 | 1.000 | 0.595 | 0.841 |
| `prompt_last|L26|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 37 | 0.326 | 70.717 | 1.000 | 1.000 | 0.676 | 0.000 |
| `query_relation|L19|natural_correct|mlp_input` | natural_correct | `mlp_input` | 37 | -0.012 | 37.964 | 1.000 | 1.000 | 0.432 | 0.000 |
| `query_relation|L19|natural_correct|mlp_out` | natural_correct | `mlp_out` | 37 | -0.237 | 372.247 | 1.000 | 1.000 | 0.568 | 0.000 |
| `prompt_last|L26|natural_correct|mlp_out` | natural_correct | `mlp_out` | 37 | -0.813 | 322.450 | 1.000 | 1.000 | 0.351 | 0.000 |
| `rule_value|L26|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 17 | 0.084 | 83.142 | 1.000 | 1.000 | 0.412 | 0.000 |
| `rule_value|L26|natural_correct|mlp_input` | natural_correct | `mlp_input` | 17 | -0.276 | 50.357 | 1.000 | 1.000 | 0.353 | 0.000 |
| `rule_value|L26|natural_correct|mlp_out` | natural_correct | `mlp_out` | 17 | 19.472 | 1105.300 | 1.000 | 1.000 | 0.706 | 0.000 |
| `rule_value|L26|natural_correct|attn_out` | natural_correct | `attn_out` | 17 | -11.752 | 523.688 | 1.000 | 1.000 | 0.471 | 0.991 |
| `prompt_last|L26|natural_wrong|attn_out` | natural_wrong | `attn_out` | 37 | -0.069 | 80.993 | 0.736 | 0.810 | 0.405 | 1.007 |
| `query_relation|L19|natural_wrong|attn_out` | natural_wrong | `attn_out` | 37 | 0.655 | 131.968 | 0.701 | 1.032 | 0.568 | 0.794 |
| `rule_value|L26|natural_wrong|mlp_out` | natural_wrong | `mlp_out` | 17 | 25.853 | 1163.051 | 0.652 | 3.465 | 0.706 | 0.000 |
| `prompt_last|L26|natural_wrong|mlp_out` | natural_wrong | `mlp_out` | 37 | -0.699 | 227.793 | 0.639 | 0.710 | 0.351 | 0.000 |
| `query_relation|L19|natural_wrong|final_norm_input` | natural_wrong | `final_norm_input` | 37 | 0.025 | 609.280 | 0.637 | 0.910 | 0.568 | 0.000 |
| `query_relation|L19|natural_wrong|layer_out` | natural_wrong | `layer_out` | 37 | 0.025 | 609.280 | 0.637 | 0.910 | 0.568 | 0.000 |
| `prompt_last|L26|natural_wrong|final_norm_output` | natural_wrong | `final_norm_output` | 37 | -0.085 | 51.398 | 0.634 | 0.730 | 0.405 | 0.000 |

### DS7B watched acceptance path

| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rule_value|L26|natural_correct|layer_input` | natural_correct | `layer_input` | 17 | 0.039 | 368.981 | 1.000 | 1.000 | 0.353 | 0.000 |
| `rule_value|L26|natural_correct|attn_out` | natural_correct | `attn_out` | 17 | -11.752 | 523.688 | 1.000 | 1.000 | 0.471 | 0.991 |
| `rule_value|L26|natural_correct|mlp_input` | natural_correct | `mlp_input` | 17 | -0.276 | 50.357 | 1.000 | 1.000 | 0.353 | 0.000 |
| `rule_value|L26|natural_correct|mlp_out` | natural_correct | `mlp_out` | 17 | 19.472 | 1105.300 | 1.000 | 1.000 | 0.706 | 0.000 |
| `rule_value|L26|natural_correct|layer_out` | natural_correct | `layer_out` | 17 | 7.778 | 1019.505 | 1.000 | 1.000 | 1.000 | 0.000 |
| `rule_value|L26|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 17 | 0.084 | 83.142 | 1.000 | 1.000 | 0.412 | 0.000 |
| `rule_value|L26|natural_wrong|layer_input` | natural_wrong | `layer_input` | 17 | 0.649 | 366.845 | 0.590 | 1.054 | 0.529 | 0.000 |
| `rule_value|L26|natural_wrong|attn_out` | natural_wrong | `attn_out` | 17 | -17.096 | 597.491 | 0.372 | 3.184 | 0.294 | 0.994 |
| `rule_value|L26|natural_wrong|mlp_input` | natural_wrong | `mlp_input` | 17 | -0.216 | 52.036 | 0.448 | 1.576 | 0.353 | 0.000 |
| `rule_value|L26|natural_wrong|mlp_out` | natural_wrong | `mlp_out` | 17 | 25.853 | 1163.051 | 0.652 | 3.465 | 0.706 | 0.000 |
| `rule_value|L26|natural_wrong|layer_out` | natural_wrong | `layer_out` | 17 | 9.454 | 1012.045 | 0.567 | 1.814 | 0.941 | 0.000 |
| `rule_value|L26|natural_wrong|final_norm_output` | natural_wrong | `final_norm_output` | 17 | 0.164 | 81.458 | 0.577 | 1.168 | 0.529 | 0.000 |
| `rule_value|L26|artificial_repair|layer_input` | artificial_repair | `layer_input` | 17 | -0.841 | 406.428 | 0.298 | 1.070 | 0.294 | 0.000 |
| `rule_value|L26|artificial_repair|attn_out` | artificial_repair | `attn_out` | 17 | 0.775 | 166.081 | -0.250 | 0.749 | 0.706 | 0.344 |
| `rule_value|L26|artificial_repair|mlp_input` | artificial_repair | `mlp_input` | 17 | 0.158 | 29.744 | 0.333 | 0.715 | 0.588 | 0.000 |
| `rule_value|L26|artificial_repair|mlp_out` | artificial_repair | `mlp_out` | 17 | -2.371 | 278.825 | 0.252 | 0.628 | 0.294 | 0.000 |
| `rule_value|L26|artificial_repair|layer_out` | artificial_repair | `layer_out` | 17 | -2.407 | 458.115 | 0.288 | 0.643 | 0.294 | 0.000 |
| `rule_value|L26|artificial_repair|final_norm_output` | artificial_repair | `final_norm_output` | 17 | 0.289 | 72.858 | 0.457 | 0.922 | 0.647 | 0.000 |
| `rule_value|L26|artificial_random|layer_input` | artificial_random | `layer_input` | 17 | -0.238 | 410.770 | 0.034 | 1.093 | 0.412 | 0.000 |
| `rule_value|L26|artificial_random|attn_out` | artificial_random | `attn_out` | 17 | 0.634 | 119.160 | -0.169 | 0.563 | 0.706 | 0.305 |
| `rule_value|L26|artificial_random|mlp_input` | artificial_random | `mlp_input` | 17 | 0.014 | 31.540 | 0.055 | 0.768 | 0.529 | 0.000 |
| `rule_value|L26|artificial_random|mlp_out` | artificial_random | `mlp_out` | 17 | 2.426 | 266.926 | 0.122 | 0.585 | 0.647 | 0.000 |
| `rule_value|L26|artificial_random|layer_out` | artificial_random | `layer_out` | 17 | 2.857 | 481.778 | 0.025 | 0.663 | 0.529 | 0.000 |
| `rule_value|L26|artificial_random|final_norm_output` | artificial_random | `final_norm_output` | 17 | -0.028 | 71.875 | 0.069 | 0.901 | 0.529 | 0.000 |
| `rule_value|L26|artificial_wrong|layer_input` | artificial_wrong | `layer_input` | 17 | -1.766 | 400.586 | 0.179 | 1.155 | 0.294 | 0.000 |
| `rule_value|L26|artificial_wrong|attn_out` | artificial_wrong | `attn_out` | 17 | 0.817 | 160.402 | -0.342 | 0.821 | 0.706 | 0.351 |
| `rule_value|L26|artificial_wrong|mlp_input` | artificial_wrong | `mlp_input` | 17 | 0.130 | 29.852 | 0.205 | 0.896 | 0.706 | 0.000 |
| `rule_value|L26|artificial_wrong|mlp_out` | artificial_wrong | `mlp_out` | 17 | -3.292 | 278.400 | 0.097 | 0.854 | 0.176 | 0.000 |
| `rule_value|L26|artificial_wrong|layer_out` | artificial_wrong | `layer_out` | 17 | -4.206 | 462.007 | 0.096 | 0.832 | 0.353 | 0.000 |
| `rule_value|L26|artificial_wrong|final_norm_output` | artificial_wrong | `final_norm_output` | 17 | 0.238 | 73.289 | 0.233 | 1.059 | 0.647 | 0.000 |
| `prompt_last|L26|natural_correct|layer_input` | natural_correct | `layer_input` | 37 | 2.378 | 397.770 | 1.000 | 1.000 | 0.568 | 0.000 |
| `prompt_last|L26|natural_correct|attn_out` | natural_correct | `attn_out` | 37 | 0.022 | 100.075 | 1.000 | 1.000 | 0.459 | 1.103 |
| `prompt_last|L26|natural_correct|mlp_input` | natural_correct | `mlp_input` | 37 | 0.204 | 30.148 | 1.000 | 1.000 | 0.649 | 0.000 |
| `prompt_last|L26|natural_correct|mlp_out` | natural_correct | `mlp_out` | 37 | -0.813 | 322.450 | 1.000 | 1.000 | 0.351 | 0.000 |
| `prompt_last|L26|natural_correct|layer_out` | natural_correct | `layer_out` | 37 | 1.595 | 547.217 | 1.000 | 1.000 | 0.541 | 0.000 |
| `prompt_last|L26|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 37 | 0.326 | 70.717 | 1.000 | 1.000 | 0.676 | 0.000 |
| `prompt_last|L26|natural_wrong|layer_input` | natural_wrong | `layer_input` | 37 | -0.958 | 306.005 | 0.626 | 0.771 | 0.351 | 0.000 |
| `prompt_last|L26|natural_wrong|attn_out` | natural_wrong | `attn_out` | 37 | -0.069 | 80.993 | 0.736 | 0.810 | 0.405 | 1.007 |
| `prompt_last|L26|natural_wrong|mlp_input` | natural_wrong | `mlp_input` | 37 | -0.047 | 23.881 | 0.619 | 0.794 | 0.405 | 0.000 |
| `prompt_last|L26|natural_wrong|mlp_out` | natural_wrong | `mlp_out` | 37 | -0.699 | 227.793 | 0.639 | 0.710 | 0.351 | 0.000 |
| `prompt_last|L26|natural_wrong|layer_out` | natural_wrong | `layer_out` | 37 | -1.698 | 414.834 | 0.631 | 0.760 | 0.324 | 0.000 |
| `prompt_last|L26|natural_wrong|final_norm_output` | natural_wrong | `final_norm_output` | 37 | -0.085 | 51.398 | 0.634 | 0.730 | 0.405 | 0.000 |
| `prompt_last|L26|artificial_repair|layer_input` | artificial_repair | `layer_input` | 37 | 3.088 | 422.619 | 0.262 | 1.060 | 0.730 | 0.000 |
| `prompt_last|L26|artificial_repair|attn_out` | artificial_repair | `attn_out` | 37 | -0.577 | 130.561 | 0.153 | 1.306 | 0.378 | 0.336 |
| `prompt_last|L26|artificial_repair|mlp_input` | artificial_repair | `mlp_input` | 37 | 0.126 | 30.737 | 0.294 | 1.018 | 0.676 | 0.000 |
| `prompt_last|L26|artificial_repair|mlp_out` | artificial_repair | `mlp_out` | 37 | -2.589 | 359.748 | 0.496 | 1.119 | 0.405 | 0.000 |
| `prompt_last|L26|artificial_repair|layer_out` | artificial_repair | `layer_out` | 37 | -0.017 | 518.057 | 0.439 | 0.946 | 0.486 | 0.000 |
| `prompt_last|L26|artificial_repair|final_norm_output` | artificial_repair | `final_norm_output` | 37 | 0.194 | 69.363 | 0.530 | 0.982 | 0.703 | 0.000 |
| `prompt_last|L26|artificial_random|layer_input` | artificial_random | `layer_input` | 37 | 2.989 | 474.051 | -0.024 | 1.190 | 0.649 | 0.000 |
| `prompt_last|L26|artificial_random|attn_out` | artificial_random | `attn_out` | 37 | -1.924 | 178.014 | 0.078 | 1.784 | 0.405 | 0.366 |
| `prompt_last|L26|artificial_random|mlp_input` | artificial_random | `mlp_input` | 37 | 0.050 | 32.772 | 0.027 | 1.085 | 0.541 | 0.000 |
| `prompt_last|L26|artificial_random|mlp_out` | artificial_random | `mlp_out` | 37 | -0.930 | 328.774 | 0.008 | 1.023 | 0.595 | 0.000 |
| `prompt_last|L26|artificial_random|layer_out` | artificial_random | `layer_out` | 37 | 0.156 | 534.256 | 0.019 | 0.974 | 0.514 | 0.000 |
| `prompt_last|L26|artificial_random|final_norm_output` | artificial_random | `final_norm_output` | 37 | 0.047 | 65.829 | 0.017 | 0.931 | 0.541 | 0.000 |
| `prompt_last|L26|artificial_wrong|layer_input` | artificial_wrong | `layer_input` | 37 | 1.138 | 280.984 | 0.241 | 0.707 | 0.595 | 0.000 |
| `prompt_last|L26|artificial_wrong|attn_out` | artificial_wrong | `attn_out` | 37 | -0.374 | 72.400 | 0.138 | 0.725 | 0.432 | 0.209 |
| `prompt_last|L26|artificial_wrong|mlp_input` | artificial_wrong | `mlp_input` | 37 | 0.004 | 20.341 | 0.256 | 0.675 | 0.486 | 0.000 |
| `prompt_last|L26|artificial_wrong|mlp_out` | artificial_wrong | `mlp_out` | 37 | -0.866 | 188.600 | 0.355 | 0.587 | 0.405 | 0.000 |
| `prompt_last|L26|artificial_wrong|layer_out` | artificial_wrong | `layer_out` | 37 | -0.060 | 341.131 | 0.339 | 0.624 | 0.514 | 0.000 |
| `prompt_last|L26|artificial_wrong|final_norm_output` | artificial_wrong | `final_norm_output` | 37 | 0.012 | 45.588 | 0.360 | 0.646 | 0.486 | 0.000 |
| `query_relation|L19|natural_correct|layer_input` | natural_correct | `layer_input` | 37 | -0.641 | 504.599 | 1.000 | 1.000 | 0.405 | 0.000 |
| `query_relation|L19|natural_correct|attn_out` | natural_correct | `attn_out` | 37 | 0.923 | 132.137 | 1.000 | 1.000 | 0.595 | 0.841 |
| `query_relation|L19|natural_correct|mlp_input` | natural_correct | `mlp_input` | 37 | -0.012 | 37.964 | 1.000 | 1.000 | 0.432 | 0.000 |
| `query_relation|L19|natural_correct|mlp_out` | natural_correct | `mlp_out` | 37 | -0.237 | 372.247 | 1.000 | 1.000 | 0.568 | 0.000 |
| `query_relation|L19|natural_correct|layer_out` | natural_correct | `layer_out` | 37 | 0.043 | 670.280 | 1.000 | 1.000 | 0.514 | 0.000 |
| `query_relation|L19|natural_correct|final_norm_output` | natural_correct | `final_norm_output` | 37 | -0.013 | 78.238 | 1.000 | 1.000 | 0.432 | 0.000 |
| `query_relation|L19|natural_wrong|layer_input` | natural_wrong | `layer_input` | 37 | -0.627 | 462.089 | 0.622 | 0.917 | 0.405 | 0.000 |
| `query_relation|L19|natural_wrong|attn_out` | natural_wrong | `attn_out` | 37 | 0.655 | 131.968 | 0.701 | 1.032 | 0.568 | 0.794 |
| `query_relation|L19|natural_wrong|mlp_input` | natural_wrong | `mlp_input` | 37 | -0.031 | 34.892 | 0.617 | 0.920 | 0.459 | 0.000 |
| `query_relation|L19|natural_wrong|mlp_out` | natural_wrong | `mlp_out` | 37 | 0.009 | 338.298 | 0.629 | 0.918 | 0.405 | 0.000 |
| `query_relation|L19|natural_wrong|layer_out` | natural_wrong | `layer_out` | 37 | 0.025 | 609.280 | 0.637 | 0.910 | 0.568 | 0.000 |
| `query_relation|L19|natural_wrong|final_norm_output` | natural_wrong | `final_norm_output` | 37 | -0.050 | 69.626 | 0.611 | 0.892 | 0.351 | 0.000 |
| `query_relation|L19|artificial_repair|layer_input` | artificial_repair | `layer_input` | 37 | 0.288 | 392.396 | 0.311 | 0.778 | 0.541 | 0.000 |
| `query_relation|L19|artificial_repair|attn_out` | artificial_repair | `attn_out` | 37 | 0.274 | 81.420 | 0.384 | 0.631 | 0.541 | 0.278 |
| `query_relation|L19|artificial_repair|mlp_input` | artificial_repair | `mlp_input` | 37 | -0.003 | 29.271 | 0.292 | 0.771 | 0.459 | 0.000 |
| `query_relation|L19|artificial_repair|mlp_out` | artificial_repair | `mlp_out` | 37 | -0.009 | 294.701 | 0.332 | 0.808 | 0.432 | 0.000 |
| `query_relation|L19|artificial_repair|layer_out` | artificial_repair | `layer_out` | 37 | 0.537 | 527.998 | 0.329 | 0.791 | 0.541 | 0.000 |
| `query_relation|L19|artificial_repair|final_norm_output` | artificial_repair | `final_norm_output` | 37 | 0.008 | 57.173 | 0.225 | 0.737 | 0.541 | 0.000 |
| `query_relation|L19|artificial_random|layer_input` | artificial_random | `layer_input` | 37 | 1.280 | 540.986 | 0.179 | 1.073 | 0.676 | 0.000 |
| `query_relation|L19|artificial_random|attn_out` | artificial_random | `attn_out` | 37 | 0.329 | 113.382 | 0.316 | 0.894 | 0.514 | 0.381 |
| `query_relation|L19|artificial_random|mlp_input` | artificial_random | `mlp_input` | 37 | 0.060 | 41.306 | 0.158 | 1.090 | 0.649 | 0.000 |
| `query_relation|L19|artificial_random|mlp_out` | artificial_random | `mlp_out` | 37 | 1.937 | 409.251 | 0.258 | 1.126 | 0.595 | 0.000 |
| `query_relation|L19|artificial_random|layer_out` | artificial_random | `layer_out` | 37 | 3.542 | 725.645 | 0.238 | 1.090 | 0.676 | 0.000 |
| `query_relation|L19|artificial_random|final_norm_output` | artificial_random | `final_norm_output` | 37 | 0.140 | 82.171 | 0.129 | 1.058 | 0.703 | 0.000 |
| `query_relation|L19|artificial_wrong|layer_input` | artificial_wrong | `layer_input` | 37 | -0.334 | 373.359 | 0.152 | 0.741 | 0.595 | 0.000 |
| `query_relation|L19|artificial_wrong|attn_out` | artificial_wrong | `attn_out` | 37 | 0.204 | 82.823 | 0.395 | 0.635 | 0.595 | 0.313 |
| `query_relation|L19|artificial_wrong|mlp_input` | artificial_wrong | `mlp_input` | 37 | -0.069 | 28.381 | 0.134 | 0.748 | 0.486 | 0.000 |
| `query_relation|L19|artificial_wrong|mlp_out` | artificial_wrong | `mlp_out` | 37 | 0.036 | 314.202 | 0.284 | 0.844 | 0.486 | 0.000 |
| `query_relation|L19|artificial_wrong|layer_out` | artificial_wrong | `layer_out` | 37 | -0.076 | 521.685 | 0.225 | 0.776 | 0.459 | 0.000 |
| `query_relation|L19|artificial_wrong|final_norm_output` | artificial_wrong | `final_norm_output` | 37 | -0.099 | 60.694 | 0.128 | 0.772 | 0.351 | 0.000 |

