# Phase 625 Cross-Model Summary

Final readout bridge and MLP causal split.

## deepseek7b

- rows: 82 / raw 256
- target cases seen: 82
- patch layers: [20, 21, 22]
- MLP split layer: L23

### Score Modes

| mode | switch | margin | correct_delta | wrong_delta |
|---|---:|---:|---:|---:|
| result_only | 75/82 | 2.890 | 1.537 | -1.352 |
| selection_both_plus_result | 75/82 | 2.892 | 1.540 | -1.352 |
| result_random_norm | 2/82 | -0.092 | -0.094 | -0.002 |
| mlp_full_delta | 14/82 | 0.326 | 0.314 | -0.012 |
| mlp_correct_up | 4/82 | 0.094 | 0.065 | -0.029 |
| mlp_wrong_down | 1/82 | 0.037 | 0.015 | -0.022 |
| mlp_correct_plus_wrong | 4/82 | 0.132 | 0.083 | -0.049 |
| mlp_margin_span | 4/82 | 0.101 | 0.066 | -0.034 |
| mlp_orthogonal | 10/82 | 0.229 | 0.257 | 0.027 |
| mlp_random_same_norm | 3/82 | -0.016 | -0.007 | 0.009 |

### Final Bridge

| mode | input_proj | input_cos | output_proj | output_cos | output_margin_proxy | correct_proxy | wrong_proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| result_only | 0.340 | 0.668 | 0.372 | 0.667 | 0.579 | 0.433 | -0.145 |

## glm4

- rows: 31 / raw 256
- target cases seen: 31
- patch layers: [31, 32, 34]
- MLP split layer: L39

### Score Modes

| mode | switch | margin | correct_delta | wrong_delta |
|---|---:|---:|---:|---:|
| result_only | 29/31 | 2.131 | 0.974 | -1.157 |
| selection_both_plus_result | 29/31 | 2.131 | 0.974 | -1.157 |
| result_random_norm | 3/31 | -0.069 | -0.066 | 0.003 |
| mlp_full_delta | 4/31 | -0.212 | -0.188 | 0.024 |
| mlp_correct_up | 3/31 | -0.083 | -0.083 | -0.001 |
| mlp_wrong_down | 5/31 | -0.117 | -0.104 | 0.013 |
| mlp_correct_plus_wrong | 6/31 | -0.200 | -0.191 | 0.009 |
| mlp_margin_span | 4/31 | -0.145 | -0.145 | -0.000 |
| mlp_orthogonal | 0/31 | -0.081 | -0.040 | 0.041 |
| mlp_random_same_norm | 2/31 | -0.010 | -0.019 | -0.010 |

### Final Bridge

| mode | input_proj | input_cos | output_proj | output_cos | output_margin_proxy | correct_proxy | wrong_proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| result_only | 0.320 | 0.582 | 0.311 | 0.565 | 0.533 | 0.334 | -0.199 |

## qwen3

- rows: 17 / raw 256
- target cases seen: 17
- patch layers: [26, 27, 29]
- MLP split layer: L31

### Score Modes

| mode | switch | margin | correct_delta | wrong_delta |
|---|---:|---:|---:|---:|
| result_only | 15/17 | 4.407 | 1.814 | -2.593 |
| selection_both_plus_result | 15/17 | 4.407 | 1.814 | -2.593 |
| result_random_norm | 2/17 | 0.088 | 0.060 | -0.029 |
| mlp_full_delta | 4/17 | 0.478 | 0.399 | -0.079 |
| mlp_correct_up | 2/17 | 0.250 | 0.198 | -0.052 |
| mlp_wrong_down | 2/17 | 0.169 | 0.096 | -0.073 |
| mlp_correct_plus_wrong | 2/17 | 0.390 | 0.278 | -0.112 |
| mlp_margin_span | 2/17 | 0.280 | 0.211 | -0.068 |
| mlp_orthogonal | 3/17 | 0.162 | 0.162 | 0.001 |
| mlp_random_same_norm | 1/17 | -0.006 | -0.004 | 0.001 |

### Final Bridge

| mode | input_proj | input_cos | output_proj | output_cos | output_margin_proxy | correct_proxy | wrong_proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| result_only | 0.363 | 0.654 | 0.355 | 0.667 | 0.739 | 0.453 | -0.285 |
