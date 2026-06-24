# Phase 616 Cross Model Summary

Residual-state cumulative additive patch with single-replace references.

## qwen3

rows=9, target_seen=9, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 119}, layers=[25, 26, 27, 28, 29], specs=26, time_min=1.17

### best

| rank | name | mode | random | ops | switch | margin | correct_delta | wrong_delta | pos_margin |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `add_layer_out_L28_L29_bridge` | add | False | 2 | 9/9 | +7.282 | +1.704 | -5.578 | 9/9 |
| 2 | `replace_L28_layer_out_ref` | replace | False | 1 | 9/9 | +4.392 | +1.591 | -2.801 | 9/9 |
| 3 | `replace_L29_layer_input_ref` | replace | False | 1 | 9/9 | +4.392 | +1.591 | -2.801 | 9/9 |
| 4 | `add_L26_layer_out` | add | False | 1 | 9/9 | +4.364 | +1.592 | -2.772 | 9/9 |
| 5 | `add_L25_layer_out` | add | False | 1 | 9/9 | +4.364 | +1.592 | -2.772 | 9/9 |
| 6 | `add_L29_layer_out` | add | False | 1 | 9/9 | +4.350 | +1.586 | -2.764 | 9/9 |
| 7 | `add_L27_layer_out` | add | False | 1 | 9/9 | +4.336 | +1.579 | -2.758 | 9/9 |
| 8 | `add_layer_out_all` | add | False | 5 | 8/9 | +11.337 | +1.699 | -9.638 | 9/9 |
| 9 | `add_L28_layer_out` | add | False | 1 | 8/9 | +4.378 | +1.582 | -2.796 | 9/9 |
| 10 | `add_attn_mlp_midlate_L27_L29` | add | False | 6 | 6/9 | +2.947 | +1.358 | -1.589 | 9/9 |
| 11 | `add_mlp_all` | add | False | 5 | 6/9 | +1.988 | +1.111 | -0.877 | 9/9 |
| 12 | `add_attn_late_L28_L29` | add | False | 2 | 6/9 | +1.849 | +1.039 | -0.810 | 9/9 |
| 13 | `add_L29_attn_out` | add | False | 1 | 6/9 | +1.599 | +0.972 | -0.627 | 9/9 |
| 14 | `add_attn_mlp_all` | add | False | 10 | 5/9 | +2.238 | +1.086 | -1.153 | 9/9 |
| 15 | `add_mlp_midlate_L27_L29` | add | False | 3 | 5/9 | +1.307 | +0.834 | -0.473 | 9/9 |
| 16 | `add_L28_mlp_out` | add | False | 1 | 3/9 | +0.598 | +0.450 | -0.148 | 9/9 |
| 17 | `add_attn_all` | add | False | 5 | 3/9 | +0.543 | +0.215 | -0.328 | 7/9 |
| 18 | `add_L27_mlp_out` | add | False | 1 | 2/9 | +0.681 | +0.484 | -0.197 | 9/9 |
| 19 | `add_L28_attn_out` | add | False | 1 | 2/9 | +0.167 | +0.091 | -0.076 | 7/9 |
| 20 | `add_mlp_all` | add | True | 5 | 2/9 | +0.127 | +0.120 | -0.007 | 6/9 |
| 21 | `add_L29_mlp_out` | add | False | 1 | 2/9 | +0.083 | +0.056 | -0.027 | 7/9 |
| 22 | `add_L25_mlp_out` | add | False | 1 | 1/9 | +0.528 | +0.392 | -0.136 | 9/9 |
| 23 | `add_L26_attn_out` | add | False | 1 | 1/9 | +0.389 | +0.288 | -0.102 | 7/9 |
| 24 | `add_L26_mlp_out` | add | False | 1 | 1/9 | +0.111 | +0.087 | -0.024 | 8/9 |
| 25 | `add_attn_late_L28_L29` | add | True | 2 | 1/9 | +0.042 | +0.020 | -0.021 | 5/9 |
| 26 | `replace_L28_layer_out_ref` | replace | True | 1 | 1/9 | +0.037 | -0.026 | -0.063 | 5/9 |
| 27 | `add_layer_out_L28_L29_bridge` | add | True | 2 | 1/9 | +0.030 | +0.027 | -0.003 | 6/9 |
| 28 | `add_L27_mlp_out` | add | True | 1 | 1/9 | -0.005 | -0.012 | -0.007 | 6/9 |

### key_real

| name | mode | ops | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|
| `add_layer_out_L28_L29_bridge` | add | 2 | 9/9 | +7.282 | +1.704 | -5.578 |
| `replace_L28_layer_out_ref` | replace | 1 | 9/9 | +4.392 | +1.591 | -2.801 |
| `replace_L29_layer_input_ref` | replace | 1 | 9/9 | +4.392 | +1.591 | -2.801 |
| `add_layer_out_all` | add | 5 | 8/9 | +11.337 | +1.699 | -9.638 |
| `add_attn_mlp_midlate_L27_L29` | add | 6 | 6/9 | +2.947 | +1.358 | -1.589 |
| `add_mlp_all` | add | 5 | 6/9 | +1.988 | +1.111 | -0.877 |
| `add_attn_late_L28_L29` | add | 2 | 6/9 | +1.849 | +1.039 | -0.810 |
| `add_attn_mlp_all` | add | 10 | 5/9 | +2.238 | +1.086 | -1.153 |
| `add_mlp_midlate_L27_L29` | add | 3 | 5/9 | +1.307 | +0.834 | -0.473 |
| `add_attn_all` | add | 5 | 3/9 | +0.543 | +0.215 | -0.328 |
| `add_attn_early_L25_L27` | add | 3 | 0/9 | -1.196 | -1.081 | +0.115 |

### random_controls

| name | mode | ops | switch | margin |
|---|---|---:|---:|---:|
| `add_mlp_all` | add | 5 | 2/9 | +0.127 |
| `add_attn_late_L28_L29` | add | 2 | 1/9 | +0.042 |
| `replace_L28_layer_out_ref` | replace | 1 | 1/9 | +0.037 |
| `add_layer_out_L28_L29_bridge` | add | 2 | 1/9 | +0.030 |
| `add_attn_mlp_midlate_L27_L29` | add | 6 | 1/9 | -0.006 |
| `add_attn_early_L25_L27` | add | 3 | 0/9 | -0.011 |
| `add_layer_out_all` | add | 5 | 0/9 | -0.033 |
| `add_attn_mlp_all` | add | 10 | 0/9 | -0.081 |
| `add_mlp_midlate_L27_L29` | add | 3 | 0/9 | -0.085 |
| `add_attn_all` | add | 5 | 0/9 | -0.113 |
| `replace_L29_layer_input_ref` | replace | 1 | 0/9 | -0.130 |

## glm4

rows=12, target_seen=12, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 116}, layers=[30, 31, 32, 33, 34], specs=26, time_min=2.21

### best

| rank | name | mode | random | ops | switch | margin | correct_delta | wrong_delta | pos_margin |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `add_layer_out_L33_L34_bridge` | add | False | 2 | 11/12 | +4.310 | +0.924 | -3.386 | 12/12 |
| 2 | `add_L33_layer_out` | add | False | 1 | 11/12 | +1.943 | +0.739 | -1.204 | 12/12 |
| 3 | `replace_L33_layer_out_ref` | replace | False | 1 | 11/12 | +1.932 | +0.735 | -1.198 | 12/12 |
| 4 | `replace_L34_layer_input_ref` | replace | False | 1 | 11/12 | +1.932 | +0.735 | -1.198 | 12/12 |
| 5 | `add_L32_layer_out` | add | False | 1 | 11/12 | +1.917 | +0.729 | -1.187 | 12/12 |
| 6 | `add_L34_layer_out` | add | False | 1 | 11/12 | +1.906 | +0.730 | -1.176 | 12/12 |
| 7 | `add_layer_out_all` | add | False | 5 | 10/12 | +10.632 | +0.816 | -9.817 | 12/12 |
| 8 | `add_L31_layer_out` | add | False | 1 | 10/12 | +1.932 | +0.729 | -1.204 | 12/12 |
| 9 | `add_L30_layer_out` | add | False | 1 | 10/12 | +1.906 | +0.723 | -1.183 | 12/12 |
| 10 | `add_attn_mlp_midlate_L32_L34` | add | False | 6 | 5/12 | +0.438 | +0.221 | -0.216 | 10/12 |
| 11 | `add_mlp_all` | add | False | 5 | 4/12 | +0.417 | +0.228 | -0.189 | 11/12 |
| 12 | `add_mlp_midlate_L32_L34` | add | False | 3 | 3/12 | +0.328 | +0.189 | -0.139 | 12/12 |
| 13 | `add_attn_mlp_all` | add | False | 10 | 2/12 | +0.479 | +0.217 | -0.263 | 10/12 |
| 14 | `add_L34_mlp_out` | add | False | 1 | 2/12 | +0.146 | +0.089 | -0.057 | 10/12 |
| 15 | `add_L31_mlp_out` | add | False | 1 | 2/12 | +0.068 | +0.041 | -0.027 | 6/12 |
| 16 | `add_L32_attn_out` | add | False | 1 | 1/12 | +0.104 | +0.058 | -0.047 | 10/12 |
| 17 | `add_L32_mlp_out` | add | False | 1 | 1/12 | +0.099 | +0.060 | -0.039 | 10/12 |
| 18 | `replace_L34_layer_input_ref` | replace | True | 1 | 1/12 | +0.091 | +0.057 | -0.033 | 7/12 |
| 19 | `add_L30_layer_out` | add | True | 1 | 1/12 | +0.080 | +0.052 | -0.028 | 7/12 |
| 20 | `add_L34_layer_out` | add | True | 1 | 1/12 | +0.050 | +0.036 | -0.013 | 7/12 |
| 21 | `add_L33_attn_out` | add | False | 1 | 1/12 | +0.031 | +0.015 | -0.016 | 6/12 |
| 22 | `add_attn_early_L30_L32` | add | False | 3 | 1/12 | +0.016 | -0.008 | -0.024 | 5/12 |
| 23 | `add_layer_out_all` | add | True | 5 | 1/12 | +0.014 | -0.037 | -0.051 | 6/12 |
| 24 | `add_L33_attn_out` | add | True | 1 | 1/12 | -0.003 | -0.014 | -0.012 | 4/12 |
| 25 | `add_attn_late_L33_L34` | add | False | 2 | 1/12 | -0.010 | -0.006 | +0.005 | 4/12 |
| 26 | `add_attn_all` | add | False | 5 | 1/12 | -0.016 | -0.037 | -0.021 | 5/12 |
| 27 | `add_L30_attn_out` | add | False | 1 | 1/12 | -0.016 | -0.011 | +0.005 | 1/12 |
| 28 | `add_L33_layer_out` | add | True | 1 | 1/12 | -0.018 | -0.017 | +0.001 | 5/12 |

### key_real

| name | mode | ops | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|
| `add_layer_out_L33_L34_bridge` | add | 2 | 11/12 | +4.310 | +0.924 | -3.386 |
| `replace_L33_layer_out_ref` | replace | 1 | 11/12 | +1.932 | +0.735 | -1.198 |
| `replace_L34_layer_input_ref` | replace | 1 | 11/12 | +1.932 | +0.735 | -1.198 |
| `add_layer_out_all` | add | 5 | 10/12 | +10.632 | +0.816 | -9.817 |
| `add_attn_mlp_midlate_L32_L34` | add | 6 | 5/12 | +0.438 | +0.221 | -0.216 |
| `add_mlp_all` | add | 5 | 4/12 | +0.417 | +0.228 | -0.189 |
| `add_mlp_midlate_L32_L34` | add | 3 | 3/12 | +0.328 | +0.189 | -0.139 |
| `add_attn_mlp_all` | add | 10 | 2/12 | +0.479 | +0.217 | -0.263 |
| `add_attn_early_L30_L32` | add | 3 | 1/12 | +0.016 | -0.008 | -0.024 |
| `add_attn_late_L33_L34` | add | 2 | 1/12 | -0.010 | -0.006 | +0.005 |
| `add_attn_all` | add | 5 | 1/12 | -0.016 | -0.037 | -0.021 |

### random_controls

| name | mode | ops | switch | margin |
|---|---|---:|---:|---:|
| `replace_L34_layer_input_ref` | replace | 1 | 1/12 | +0.091 |
| `add_layer_out_all` | add | 5 | 1/12 | +0.014 |
| `add_mlp_midlate_L32_L34` | add | 3 | 1/12 | -0.019 |
| `add_attn_mlp_midlate_L32_L34` | add | 6 | 1/12 | -0.022 |
| `replace_L33_layer_out_ref` | replace | 1 | 1/12 | -0.027 |
| `add_mlp_all` | add | 5 | 0/12 | +0.026 |
| `add_attn_early_L30_L32` | add | 3 | 0/12 | +0.021 |
| `add_attn_mlp_all` | add | 10 | 0/12 | +0.017 |
| `add_layer_out_L33_L34_bridge` | add | 2 | 0/12 | +0.009 |
| `add_attn_late_L33_L34` | add | 2 | 0/12 | -0.041 |
| `add_attn_all` | add | 5 | 0/12 | -0.073 |

## deepseek7b

rows=43, target_seen=43, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 85}, layers=[18, 19, 20, 21, 22], specs=26, time_min=5.10

### best

| rank | name | mode | random | ops | switch | margin | correct_delta | wrong_delta | pos_margin |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `add_layer_out_L21_L22_bridge` | add | False | 2 | 43/43 | +6.142 | +1.924 | -4.218 | 43/43 |
| 2 | `add_attn_mlp_all` | add | False | 10 | 43/43 | +5.580 | +1.748 | -3.832 | 43/43 |
| 3 | `add_attn_all` | add | False | 5 | 43/43 | +3.900 | +1.567 | -2.333 | 43/43 |
| 4 | `replace_L21_layer_out_ref` | replace | False | 1 | 43/43 | +3.370 | +1.682 | -1.689 | 43/43 |
| 5 | `replace_L22_layer_input_ref` | replace | False | 1 | 43/43 | +3.370 | +1.682 | -1.689 | 43/43 |
| 6 | `add_L22_layer_out` | add | False | 1 | 43/43 | +3.333 | +1.675 | -1.657 | 43/43 |
| 7 | `add_layer_out_all` | add | False | 5 | 42/43 | +9.748 | +1.716 | -8.032 | 43/43 |
| 8 | `add_L21_layer_out` | add | False | 1 | 42/43 | +3.350 | +1.681 | -1.669 | 43/43 |
| 9 | `add_L20_layer_out` | add | False | 1 | 42/43 | +3.113 | +1.636 | -1.477 | 43/43 |
| 10 | `add_attn_mlp_midlate_L20_L22` | add | False | 6 | 41/43 | +3.925 | +1.590 | -2.335 | 43/43 |
| 11 | `add_attn_early_L18_L20` | add | False | 3 | 37/43 | +2.401 | +1.161 | -1.241 | 43/43 |
| 12 | `add_attn_late_L21_L22` | add | False | 2 | 34/43 | +1.902 | +1.155 | -0.746 | 43/43 |
| 13 | `add_L22_attn_out` | add | False | 1 | 32/43 | +1.736 | +1.073 | -0.663 | 43/43 |
| 14 | `add_L19_layer_out` | add | False | 1 | 31/43 | +1.973 | +1.293 | -0.680 | 43/43 |
| 15 | `add_L18_layer_out` | add | False | 1 | 30/43 | +1.957 | +1.290 | -0.667 | 43/43 |
| 16 | `add_L18_attn_out` | add | False | 1 | 20/43 | +1.073 | +0.604 | -0.468 | 42/43 |
| 17 | `add_L20_attn_out` | add | False | 1 | 19/43 | +0.925 | +0.547 | -0.377 | 39/43 |
| 18 | `add_L19_attn_out` | add | False | 1 | 18/43 | +0.883 | +0.595 | -0.288 | 40/43 |
| 19 | `add_mlp_midlate_L20_L22` | add | False | 3 | 15/43 | +0.914 | +0.599 | -0.315 | 38/43 |
| 20 | `add_mlp_all` | add | False | 5 | 14/43 | +0.853 | +0.592 | -0.261 | 31/43 |
| 21 | `add_L20_mlp_out` | add | False | 1 | 10/43 | +0.418 | +0.236 | -0.182 | 35/43 |
| 22 | `add_L21_mlp_out` | add | False | 1 | 6/43 | +0.345 | +0.227 | -0.118 | 35/43 |
| 23 | `replace_L22_layer_input_ref` | replace | True | 1 | 6/43 | +0.021 | -0.063 | -0.085 | 21/43 |
| 24 | `add_layer_out_L21_L22_bridge` | add | True | 2 | 4/43 | +0.008 | -0.074 | -0.082 | 25/43 |
| 25 | `add_attn_mlp_all` | add | True | 10 | 4/43 | -0.068 | -0.129 | -0.062 | 21/43 |
| 26 | `add_L18_mlp_out` | add | False | 1 | 3/43 | +0.066 | +0.116 | +0.050 | 25/43 |
| 27 | `add_attn_all` | add | True | 5 | 3/43 | +0.065 | -0.011 | -0.076 | 24/43 |
| 28 | `add_L18_layer_out` | add | True | 1 | 3/43 | +0.013 | -0.040 | -0.053 | 22/43 |

### key_real

| name | mode | ops | switch | margin | correct_delta | wrong_delta |
|---|---|---:|---:|---:|---:|---:|
| `add_layer_out_L21_L22_bridge` | add | 2 | 43/43 | +6.142 | +1.924 | -4.218 |
| `add_attn_mlp_all` | add | 10 | 43/43 | +5.580 | +1.748 | -3.832 |
| `add_attn_all` | add | 5 | 43/43 | +3.900 | +1.567 | -2.333 |
| `replace_L21_layer_out_ref` | replace | 1 | 43/43 | +3.370 | +1.682 | -1.689 |
| `replace_L22_layer_input_ref` | replace | 1 | 43/43 | +3.370 | +1.682 | -1.689 |
| `add_layer_out_all` | add | 5 | 42/43 | +9.748 | +1.716 | -8.032 |
| `add_attn_mlp_midlate_L20_L22` | add | 6 | 41/43 | +3.925 | +1.590 | -2.335 |
| `add_attn_early_L18_L20` | add | 3 | 37/43 | +2.401 | +1.161 | -1.241 |
| `add_attn_late_L21_L22` | add | 2 | 34/43 | +1.902 | +1.155 | -0.746 |
| `add_mlp_midlate_L20_L22` | add | 3 | 15/43 | +0.914 | +0.599 | -0.315 |
| `add_mlp_all` | add | 5 | 14/43 | +0.853 | +0.592 | -0.261 |

### random_controls

| name | mode | ops | switch | margin |
|---|---|---:|---:|---:|
| `replace_L22_layer_input_ref` | replace | 1 | 6/43 | +0.021 |
| `add_layer_out_L21_L22_bridge` | add | 2 | 4/43 | +0.008 |
| `add_attn_mlp_all` | add | 10 | 4/43 | -0.068 |
| `add_attn_all` | add | 5 | 3/43 | +0.065 |
| `add_layer_out_all` | add | 5 | 3/43 | -0.384 |
| `add_mlp_all` | add | 5 | 2/43 | -0.092 |
| `add_attn_early_L18_L20` | add | 3 | 1/43 | +0.082 |
| `add_mlp_midlate_L20_L22` | add | 3 | 1/43 | +0.058 |
| `add_attn_mlp_midlate_L20_L22` | add | 6 | 1/43 | -0.105 |
| `add_attn_late_L21_L22` | add | 2 | 0/43 | -0.016 |
| `replace_L21_layer_out_ref` | replace | 1 | 0/43 | -0.047 |
