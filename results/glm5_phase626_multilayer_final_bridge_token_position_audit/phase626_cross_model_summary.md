# Phase 626 Cross-Model Summary

Multi-layer final bridge and token-position readout audit.

## deepseek7b

- rows: 82 / raw 256
- target cases seen: 82
- result patch layers: [22]
- downstream layers: [22, 23, 24, 25, 26, 27]
- tokenization: `{'v05': {'ids': [348, 15, 20], 'tokens': [' v', '0', '5']}, 'v91': {'ids': [348, 24, 16], 'tokens': [' v', '9', '1']}, 'v22': {'ids': [348, 17, 17], 'tokens': [' v', '2', '2']}, 'v48': {'ids': [348, 19, 23], 'tokens': [' v', '4', '8']}}`

### Score Modes

| mode | switch | margin | correct_delta | wrong_delta |
|---|---:|---:|---:|---:|
| result_only | 75/82 | 2.890 | 1.537 | -1.352 |
| final_input_all | 82/82 | 3.602 | 5.825 | 2.222 |
| final_output_all | 82/82 | 3.602 | 5.825 | 2.222 |
| final_output_token0 | 0/82 | 0.000 | 4.105 | 4.105 |
| final_output_last | 2/82 | 0.053 | 0.024 | -0.030 |
| final_output_random_all | 17/82 | 0.271 | -0.178 | -0.449 |
| cumulative_mlp_out | 44/82 | 1.414 | 0.973 | -0.442 |
| cumulative_attn_out | 79/82 | 3.114 | 1.565 | -1.550 |
| cumulative_layer_out | 81/82 | 3.561 | 1.711 | -1.850 |
| cumulative_layer_out_random | 9/82 | 0.077 | -0.106 | -0.182 |

### Token Position Deltas Under Result-Only

| token_pos | correct_delta | wrong_delta | margin_delta |
|---:|---:|---:|---:|
| tok0 | 0.000 | 0.000 | 0.000 |
| tok1 | 1.525 | -1.354 | 2.879 |
| tok2 | 0.012 | 0.002 | 0.011 |

## glm4

- rows: 31 / raw 256
- target cases seen: 31
- result patch layers: [34]
- downstream layers: [34, 35, 36, 37, 38, 39]
- tokenization: `{'v05': {'ids': [348, 100002], 'tokens': [' v', '05']}, 'v91': {'ids': [348, 104327], 'tokens': [' v', '91']}, 'v22': {'ids': [348, 99241], 'tokens': [' v', '22']}, 'v48': {'ids': [348, 100933], 'tokens': [' v', '48']}}`

### Score Modes

| mode | switch | margin | correct_delta | wrong_delta |
|---|---:|---:|---:|---:|
| result_only | 29/31 | 2.131 | 0.974 | -1.157 |
| final_input_all | 31/31 | 2.300 | 1.968 | -0.332 |
| final_output_all | 31/31 | 2.300 | 1.968 | -0.332 |
| final_output_token0 | 0/31 | 0.000 | 0.964 | 0.964 |
| final_output_last | 31/31 | 2.300 | 1.004 | -1.296 |
| final_output_random_all | 6/31 | -0.015 | -0.045 | -0.030 |
| cumulative_mlp_out | 11/31 | 0.518 | 0.347 | -0.171 |
| cumulative_attn_out | 3/31 | -0.121 | -0.066 | 0.055 |
| cumulative_layer_out | 31/31 | 2.300 | 1.004 | -1.296 |
| cumulative_layer_out_random | 4/31 | 0.090 | 0.025 | -0.066 |

### Token Position Deltas Under Result-Only

| token_pos | correct_delta | wrong_delta | margin_delta |
|---:|---:|---:|---:|
| tok0 | 0.000 | 0.000 | 0.000 |
| tok1 | 0.974 | -1.157 | 2.131 |

## qwen3

- rows: 17 / raw 256
- target cases seen: 17
- result patch layers: [29]
- downstream layers: [29, 30, 31, 32, 33, 34, 35]
- tokenization: `{'v05': {'ids': [348, 15, 20], 'tokens': [' v', '0', '5']}, 'v91': {'ids': [348, 24, 16], 'tokens': [' v', '9', '1']}, 'v22': {'ids': [348, 17, 17], 'tokens': [' v', '2', '2']}, 'v48': {'ids': [348, 19, 23], 'tokens': [' v', '4', '8']}}`

### Score Modes

| mode | switch | margin | correct_delta | wrong_delta |
|---|---:|---:|---:|---:|
| result_only | 15/17 | 4.407 | 1.814 | -2.593 |
| final_input_all | 17/17 | 5.377 | 2.348 | -3.028 |
| final_output_all | 17/17 | 5.377 | 2.348 | -3.028 |
| final_output_token0 | 0/17 | 0.000 | 0.470 | 0.470 |
| final_output_last | 2/17 | 0.075 | 0.003 | -0.073 |
| final_output_random_all | 1/17 | -0.126 | -0.257 | -0.131 |
| cumulative_mlp_out | 12/17 | 2.545 | 1.335 | -1.210 |
| cumulative_attn_out | 12/17 | 2.326 | 1.361 | -0.964 |
| cumulative_layer_out | 17/17 | 5.305 | 1.878 | -3.426 |
| cumulative_layer_out_random | 2/17 | 0.020 | 0.009 | -0.011 |

### Token Position Deltas Under Result-Only

| token_pos | correct_delta | wrong_delta | margin_delta |
|---:|---:|---:|---:|
| tok0 | 0.000 | 0.000 | 0.000 |
| tok1 | 1.812 | -2.593 | 4.404 |
| tok2 | 0.002 | -0.001 | 0.003 |
