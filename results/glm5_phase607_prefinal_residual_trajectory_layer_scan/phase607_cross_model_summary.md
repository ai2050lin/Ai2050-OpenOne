# Phase607 Cross-Model Summary

Pre-final residual trajectory layer scan.

## qwen3

cases=96, rows=7, target_cases_seen=7, layers=[28, 29, 30, 31, 32, 33, 34, 35], time_min=1.00

### First Effective

| component | key | layer | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---|---:|---:|---:|---:|---:|
| layer_input | `L28|layer_input` | L28 | 7/7 | 8.663 | 1.764 | -6.899 |
| layer_out | `L28|layer_out` | L28 | 7/7 | 8.717 | 1.768 | -6.949 |
| attn_out | `L28|attn_out` | L28 | 2/7 | 0.143 | 0.092 | -0.051 |
| mlp_out | `L28|mlp_out` | L28 | 3/7 | 1.054 | 0.735 | -0.319 |

### Best Patches

| key | layer | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `L28|layer_out` | L28 | layer_out | False | 7 | 7/7 | 8.717 | 1.768 | -6.949 |
| `L29|layer_input` | L29 | layer_input | False | 7 | 7/7 | 8.717 | 1.768 | -6.949 |
| `L28|layer_input` | L28 | layer_input | False | 7 | 7/7 | 8.663 | 1.764 | -6.899 |
| `L29|layer_out` | L29 | layer_out | False | 7 | 7/7 | 8.663 | 1.768 | -6.895 |
| `L30|layer_input` | L30 | layer_input | False | 7 | 7/7 | 8.663 | 1.768 | -6.895 |
| `L30|layer_out` | L30 | layer_out | False | 7 | 7/7 | 8.538 | 1.767 | -6.770 |
| `L31|layer_input` | L31 | layer_input | False | 7 | 7/7 | 8.538 | 1.767 | -6.770 |
| `L31|layer_out` | L31 | layer_out | False | 7 | 7/7 | 8.448 | 1.767 | -6.682 |
| `L32|layer_input` | L32 | layer_input | False | 7 | 7/7 | 8.448 | 1.767 | -6.682 |
| `L32|layer_out` | L32 | layer_out | False | 7 | 7/7 | 8.216 | 1.765 | -6.451 |
| `L33|layer_input` | L33 | layer_input | False | 7 | 7/7 | 8.216 | 1.765 | -6.451 |
| `L33|layer_out` | L33 | layer_out | False | 7 | 7/7 | 8.073 | 1.764 | -6.309 |
| `L34|layer_input` | L34 | layer_input | False | 7 | 7/7 | 8.073 | 1.764 | -6.309 |
| `L34|layer_out` | L34 | layer_out | False | 7 | 7/7 | 8.072 | 1.764 | -6.308 |
| `L35|layer_input` | L35 | layer_input | False | 7 | 7/7 | 8.072 | 1.764 | -6.308 |
| `L35|layer_out` | L35 | layer_out | False | 7 | 7/7 | 7.821 | 1.761 | -6.060 |
| `L34|mlp_out` | L34 | mlp_out | False | 7 | 7/7 | 2.536 | 1.360 | -1.176 |
| `L29|attn_out` | L29 | attn_out | False | 7 | 6/7 | 2.037 | 1.166 | -0.871 |
| `L28|mlp_out` | L28 | mlp_out | False | 7 | 3/7 | 1.054 | 0.735 | -0.319 |
| `L32|mlp_out` | L32 | mlp_out | False | 7 | 3/7 | 1.018 | 0.685 | -0.334 |
| `L31|mlp_out` | L31 | mlp_out | False | 7 | 2/7 | 0.857 | 0.645 | -0.212 |
| `L30|attn_out` | L30 | attn_out | False | 7 | 2/7 | 0.518 | 0.353 | -0.165 |
| `L28|attn_out` | L28 | attn_out | False | 7 | 2/7 | 0.143 | 0.092 | -0.051 |
| `L29|mlp_out` | L29 | mlp_out | False | 7 | 2/7 | 0.125 | 0.084 | -0.041 |
| `L35|layer_input|random` | L35 | layer_input | True | 7 | 1/7 | 0.316 | 0.334 | 0.017 |
| `L32|attn_out` | L32 | attn_out | False | 7 | 1/7 | 0.196 | 0.158 | -0.039 |
| `L29|layer_out|random` | L29 | layer_out | True | 7 | 1/7 | 0.188 | 0.180 | -0.008 |
| `L30|layer_out|random` | L30 | layer_out | True | 7 | 1/7 | 0.133 | 0.068 | -0.064 |
| `L31|mlp_out|random` | L31 | mlp_out | True | 7 | 1/7 | 0.108 | 0.114 | 0.007 |
| `L30|mlp_out` | L30 | mlp_out | False | 7 | 1/7 | 0.072 | 0.043 | -0.028 |
| `L32|layer_input|random` | L32 | layer_input | True | 7 | 1/7 | 0.066 | 0.030 | -0.036 |
| `L30|attn_out|random` | L30 | attn_out | True | 7 | 1/7 | 0.065 | 0.060 | -0.005 |

### Layer Component Grid

| layer | layer_input | layer_out | attn_out | mlp_out |
|---:|---:|---:|---:|---:|
| L28 | 7/7 (8.663) | 7/7 (8.717) | 2/7 (0.143) | 3/7 (1.054) |
| L29 | 7/7 (8.717) | 7/7 (8.663) | 6/7 (2.037) | 2/7 (0.125) |
| L30 | 7/7 (8.663) | 7/7 (8.538) | 2/7 (0.518) | 1/7 (0.072) |
| L31 | 7/7 (8.538) | 7/7 (8.448) | 0/7 (-0.232) | 2/7 (0.857) |
| L32 | 7/7 (8.448) | 7/7 (8.216) | 1/7 (0.196) | 3/7 (1.018) |
| L33 | 7/7 (8.216) | 7/7 (8.073) | 0/7 (-0.072) | 1/7 (-0.018) |
| L34 | 7/7 (8.073) | 7/7 (8.072) | 0/7 (-0.089) | 7/7 (2.536) |
| L35 | 7/7 (8.072) | 7/7 (7.821) | 0/7 (-1.214) | 1/7 (-0.786) |

## glm4

cases=96, rows=13, target_cases_seen=13, layers=[32, 33, 34, 35, 36, 37, 38, 39], time_min=2.50

### First Effective

| component | key | layer | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---|---:|---:|---:|---:|---:|
| layer_input | `L32|layer_input` | L32 | 12/13 | 2.356 | 0.791 | -1.565 |
| layer_out | `L32|layer_out` | L32 | 13/13 | 2.413 | 0.865 | -1.549 |
| attn_out | `L33|attn_out` | L33 | 2/13 | 0.082 | -0.011 | -0.093 |
| mlp_out | `L32|mlp_out` | L32 | 1/13 | 0.125 | 0.081 | -0.044 |

### Best Patches

| key | layer | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `L36|layer_out` | L36 | layer_out | False | 13 | 13/13 | 2.942 | 1.005 | -1.938 |
| `L37|layer_input` | L37 | layer_input | False | 13 | 13/13 | 2.942 | 1.005 | -1.938 |
| `L37|layer_out` | L37 | layer_out | False | 13 | 13/13 | 2.928 | 1.006 | -1.922 |
| `L38|layer_input` | L38 | layer_input | False | 13 | 13/13 | 2.928 | 1.006 | -1.922 |
| `L38|layer_out` | L38 | layer_out | False | 13 | 13/13 | 2.923 | 1.004 | -1.919 |
| `L39|layer_input` | L39 | layer_input | False | 13 | 13/13 | 2.923 | 1.004 | -1.919 |
| `L39|layer_out` | L39 | layer_out | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 |
| `L35|layer_out` | L35 | layer_out | False | 13 | 13/13 | 2.822 | 0.987 | -1.835 |
| `L36|layer_input` | L36 | layer_input | False | 13 | 13/13 | 2.822 | 0.987 | -1.835 |
| `L32|layer_out` | L32 | layer_out | False | 13 | 13/13 | 2.413 | 0.865 | -1.549 |
| `L33|layer_input` | L33 | layer_input | False | 13 | 13/13 | 2.413 | 0.865 | -1.549 |
| `L34|layer_out` | L34 | layer_out | False | 13 | 12/13 | 2.543 | 0.919 | -1.625 |
| `L35|layer_input` | L35 | layer_input | False | 13 | 12/13 | 2.543 | 0.919 | -1.625 |
| `L33|layer_out` | L33 | layer_out | False | 13 | 12/13 | 2.399 | 0.864 | -1.535 |
| `L34|layer_input` | L34 | layer_input | False | 13 | 12/13 | 2.399 | 0.864 | -1.535 |
| `L32|layer_input` | L32 | layer_input | False | 13 | 12/13 | 2.356 | 0.791 | -1.565 |
| `L34|attn_out` | L34 | attn_out | False | 13 | 4/13 | 0.168 | 0.085 | -0.083 |
| `L39|mlp_out` | L39 | mlp_out | False | 13 | 3/13 | 0.120 | 0.008 | -0.112 |
| `L39|layer_out|random` | L39 | layer_out | True | 13 | 3/13 | -0.017 | -0.106 | -0.089 |
| `L37|layer_out|random` | L37 | layer_out | True | 13 | 2/13 | 0.258 | 0.095 | -0.162 |
| `L35|layer_input|random` | L35 | layer_input | True | 13 | 2/13 | 0.149 | 0.016 | -0.133 |
| `L32|layer_out|random` | L32 | layer_out | True | 13 | 2/13 | 0.123 | 0.006 | -0.116 |
| `L37|layer_input|random` | L37 | layer_input | True | 13 | 2/13 | 0.102 | 0.026 | -0.077 |
| `L34|layer_input|random` | L34 | layer_input | True | 13 | 2/13 | 0.095 | 0.005 | -0.090 |
| `L33|attn_out` | L33 | attn_out | False | 13 | 2/13 | 0.082 | -0.011 | -0.093 |
| `L38|mlp_out|random` | L38 | mlp_out | True | 13 | 2/13 | 0.077 | 0.078 | 0.001 |
| `L36|layer_out|random` | L36 | layer_out | True | 13 | 2/13 | 0.066 | 0.023 | -0.043 |
| `L39|mlp_out|random` | L39 | mlp_out | True | 13 | 2/13 | 0.029 | -0.041 | -0.070 |
| `L32|layer_input|random` | L32 | layer_input | True | 13 | 2/13 | 0.005 | -0.009 | -0.015 |
| `L39|layer_input|random` | L39 | layer_input | True | 13 | 2/13 | -0.012 | -0.042 | -0.030 |
| `L32|mlp_out` | L32 | mlp_out | False | 13 | 1/13 | 0.125 | 0.081 | -0.044 |
| `L33|layer_out|random` | L33 | layer_out | True | 13 | 1/13 | 0.097 | 0.116 | 0.019 |

### Layer Component Grid

| layer | layer_input | layer_out | attn_out | mlp_out |
|---:|---:|---:|---:|---:|
| L32 | 12/13 (2.356) | 13/13 (2.413) | 0/13 (0.024) | 1/13 (0.125) |
| L33 | 13/13 (2.413) | 12/13 (2.399) | 2/13 (0.082) | 0/13 (0.067) |
| L34 | 12/13 (2.399) | 12/13 (2.543) | 4/13 (0.168) | 1/13 (0.091) |
| L35 | 12/13 (2.543) | 13/13 (2.822) | 0/13 (-0.111) | 0/13 (0.139) |
| L36 | 13/13 (2.822) | 13/13 (2.942) | 1/13 (0.014) | 0/13 (-0.048) |
| L37 | 13/13 (2.942) | 13/13 (2.928) | 0/13 (-0.005) | 0/13 (0.029) |
| L38 | 13/13 (2.928) | 13/13 (2.923) | 0/13 (0.005) | 0/13 (-0.394) |
| L39 | 13/13 (2.923) | 13/13 (2.913) | 0/13 (-0.038) | 3/13 (0.120) |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, layers=[20, 21, 22, 23, 24, 25, 26, 27], time_min=5.11

### First Effective

| component | key | layer | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---|---:|---:|---:|---:|---:|
| layer_input | `L20|layer_input` | L20 | 3/37 | -0.389 | -0.767 | -0.378 |
| layer_out | `L20|layer_out` | L20 | 5/37 | -0.486 | -1.315 | -0.829 |
| attn_out | `L20|attn_out` | L20 | 5/37 | 0.023 | -0.037 | -0.060 |
| mlp_out | `L20|mlp_out` | L20 | 3/37 | 0.134 | 0.075 | -0.059 |

### Best Patches

| key | layer | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `L27|layer_out` | L27 | layer_out | False | 37 | 37/37 | 7.049 | 2.048 | -5.001 |
| `L26|layer_out` | L26 | layer_out | False | 37 | 37/37 | 6.993 | 2.051 | -4.942 |
| `L27|layer_input` | L27 | layer_input | False | 37 | 37/37 | 6.993 | 2.051 | -4.942 |
| `L25|layer_out` | L25 | layer_out | False | 37 | 37/37 | 6.701 | 2.046 | -4.654 |
| `L26|layer_input` | L26 | layer_input | False | 37 | 37/37 | 6.701 | 2.046 | -4.654 |
| `L24|layer_out` | L24 | layer_out | False | 37 | 37/37 | 6.603 | 2.054 | -4.549 |
| `L25|layer_input` | L25 | layer_input | False | 37 | 37/37 | 6.603 | 2.054 | -4.549 |
| `L23|layer_out` | L23 | layer_out | False | 37 | 37/37 | 6.287 | 2.020 | -4.267 |
| `L24|layer_input` | L24 | layer_input | False | 37 | 37/37 | 6.287 | 2.020 | -4.267 |
| `L22|layer_out` | L22 | layer_out | False | 37 | 33/37 | 4.726 | 1.755 | -2.970 |
| `L23|layer_input` | L23 | layer_input | False | 37 | 33/37 | 4.726 | 1.755 | -2.970 |
| `L22|attn_out` | L22 | attn_out | False | 37 | 33/37 | 3.423 | 1.582 | -1.841 |
| `L23|attn_out` | L23 | attn_out | False | 37 | 24/37 | 1.665 | 1.096 | -0.569 |
| `L26|mlp_out` | L26 | mlp_out | False | 37 | 21/37 | 1.274 | 0.767 | -0.508 |
| `L25|mlp_out` | L25 | mlp_out | False | 37 | 14/37 | 0.881 | 0.612 | -0.269 |
| `L24|mlp_out` | L24 | mlp_out | False | 37 | 12/37 | 0.844 | 0.587 | -0.257 |
| `L26|attn_out` | L26 | attn_out | False | 37 | 8/37 | 0.627 | 0.456 | -0.171 |
| `L23|mlp_out` | L23 | mlp_out | False | 37 | 6/37 | 0.434 | 0.297 | -0.137 |
| `L21|layer_out` | L21 | layer_out | False | 37 | 6/37 | -0.183 | -1.326 | -1.143 |
| `L22|layer_input` | L22 | layer_input | False | 37 | 6/37 | -0.183 | -1.326 | -1.143 |
| `L20|attn_out` | L20 | attn_out | False | 37 | 5/37 | 0.023 | -0.037 | -0.060 |
| `L20|layer_out` | L20 | layer_out | False | 37 | 5/37 | -0.486 | -1.315 | -0.829 |
| `L21|layer_input` | L21 | layer_input | False | 37 | 5/37 | -0.486 | -1.315 | -0.829 |
| `L24|attn_out` | L24 | attn_out | False | 37 | 4/37 | 0.445 | 0.350 | -0.096 |
| `L21|layer_out|random` | L21 | layer_out | True | 37 | 4/37 | 0.022 | -0.243 | -0.265 |
| `L26|layer_out|random` | L26 | layer_out | True | 37 | 4/37 | 0.003 | -0.110 | -0.113 |
| `L22|layer_out|random` | L22 | layer_out | True | 37 | 4/37 | -0.036 | -0.154 | -0.118 |
| `L20|layer_out|random` | L20 | layer_out | True | 37 | 4/37 | -0.074 | -0.356 | -0.282 |
| `L22|layer_input|random` | L22 | layer_input | True | 37 | 4/37 | -0.088 | -0.335 | -0.247 |
| `L21|layer_input|random` | L21 | layer_input | True | 37 | 4/37 | -0.171 | -0.409 | -0.238 |
| `L20|mlp_out` | L20 | mlp_out | False | 37 | 3/37 | 0.134 | 0.075 | -0.059 |
| `L21|mlp_out` | L21 | mlp_out | False | 37 | 3/37 | 0.034 | -0.114 | -0.148 |

### Layer Component Grid

| layer | layer_input | layer_out | attn_out | mlp_out |
|---:|---:|---:|---:|---:|
| L20 | 3/37 (-0.389) | 5/37 (-0.486) | 5/37 (0.023) | 3/37 (0.134) |
| L21 | 5/37 (-0.486) | 6/37 (-0.183) | 1/37 (-0.025) | 3/37 (0.034) |
| L22 | 6/37 (-0.183) | 33/37 (4.726) | 33/37 (3.423) | 1/37 (-0.076) |
| L23 | 33/37 (4.726) | 37/37 (6.287) | 24/37 (1.665) | 6/37 (0.434) |
| L24 | 37/37 (6.287) | 37/37 (6.603) | 4/37 (0.445) | 12/37 (0.844) |
| L25 | 37/37 (6.603) | 37/37 (6.701) | 0/37 (-0.298) | 14/37 (0.881) |
| L26 | 37/37 (6.701) | 37/37 (6.993) | 8/37 (0.627) | 21/37 (1.274) |
| L27 | 37/37 (6.993) | 37/37 (7.049) | 0/37 (-0.061) | 0/37 (-1.370) |

