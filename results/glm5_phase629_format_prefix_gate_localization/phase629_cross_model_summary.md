# Phase 629 Cross-Model Summary

Prompt-last format/prefix gate localization with semantic cumulative combination.

## deepseek7b

- rows: 82 / raw 256
- target cases seen: 82
- format layers: [20, 21, 22, 23, 24, 25]
- downstream layers: [22, 23, 24, 25, 26, 27]
- components: ['layer_input', 'attn_out', 'mlp_out', 'layer_out']
- tokenization: `{'v05': {'ids': [348, 15, 20], 'tokens': [' v', '0', '5']}, 'v91': {'ids': [348, 24, 16], 'tokens': [' v', '9', '1']}, 'v22': {'ids': [348, 17, 17], 'tokens': [' v', '2', '2']}, 'v48': {'ids': [348, 19, 23], 'tokens': [' v', '4', '8']}}`

### Baselines

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| base | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.000, tok2:0.000 |
| repair_prompt | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.256, tok2:0.256 |
| semantic_cumulative_only | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.988, tok2:0.024 |
| semantic_cumulative_random | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.098, tok2:0.061 |

### Best Exact

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| format_L25_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| format_L23_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.268 |
| format_L24_layer_input_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.268 |
| repair_prompt | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.256, tok2:0.256 |
| format_L24_layer_out_semantic | 19/82 | 0/82 | 0.695 | tok0:0.232, tok1:0.988, tok2:0.268 |
| format_L25_layer_input_semantic | 19/82 | 0/82 | 0.695 | tok0:0.232, tok1:0.988, tok2:0.268 |
| format_L21_layer_out_semantic | 17/82 | 0/82 | 0.622 | tok0:0.207, tok1:0.988, tok2:0.220 |
| format_L22_layer_input_semantic | 17/82 | 0/82 | 0.622 | tok0:0.207, tok1:0.988, tok2:0.220 |
| format_L22_layer_out_semantic | 17/82 | 0/82 | 0.622 | tok0:0.207, tok1:0.988, tok2:0.232 |
| format_L23_layer_input_semantic | 17/82 | 0/82 | 0.622 | tok0:0.207, tok1:0.988, tok2:0.232 |
| format_L20_layer_out_semantic | 13/82 | 0/82 | 0.476 | tok0:0.159, tok1:0.988, tok2:0.183 |
| format_L21_layer_input_semantic | 13/82 | 0/82 | 0.476 | tok0:0.159, tok1:0.988, tok2:0.183 |
| format_L20_layer_input_semantic | 10/82 | 0/82 | 0.366 | tok0:0.122, tok1:0.988, tok2:0.134 |
| format_L20_layer_input | 4/82 | 6/82 | 0.220 | tok0:0.122, tok1:0.049, tok2:0.049 |
| format_L25_layer_out | 3/82 | 17/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.049 |
| format_L23_layer_out | 3/82 | 17/82 | 0.317 | tok0:0.244, tok1:0.037, tok2:0.037 |
| format_L24_layer_input | 3/82 | 17/82 | 0.317 | tok0:0.244, tok1:0.037, tok2:0.037 |
| format_L24_layer_out | 2/82 | 16/82 | 0.280 | tok0:0.232, tok1:0.024, tok2:0.037 |
| format_L25_layer_input | 2/82 | 16/82 | 0.280 | tok0:0.232, tok1:0.024, tok2:0.037 |
| format_L20_layer_out | 2/82 | 11/82 | 0.207 | tok0:0.159, tok1:0.024, tok2:0.024 |

### Best Tok0

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| format_L25_layer_out_semantic | 21/82 | 0/82 | 0.768 | tok0:0.256, tok1:0.988, tok2:0.305 |
| format_L25_layer_out | 3/82 | 17/82 | 0.329 | tok0:0.256, tok1:0.037, tok2:0.049 |
| format_L23_layer_out_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.268 |
| format_L24_layer_input_semantic | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.988, tok2:0.268 |
| repair_prompt | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.256, tok2:0.256 |
| format_L23_layer_out | 3/82 | 17/82 | 0.317 | tok0:0.244, tok1:0.037, tok2:0.037 |
| format_L24_layer_input | 3/82 | 17/82 | 0.317 | tok0:0.244, tok1:0.037, tok2:0.037 |
| format_L24_layer_out_semantic | 19/82 | 0/82 | 0.695 | tok0:0.232, tok1:0.988, tok2:0.268 |
| format_L25_layer_input_semantic | 19/82 | 0/82 | 0.695 | tok0:0.232, tok1:0.988, tok2:0.268 |
| format_L24_layer_out | 2/82 | 16/82 | 0.280 | tok0:0.232, tok1:0.024, tok2:0.037 |
| format_L25_layer_input | 2/82 | 16/82 | 0.280 | tok0:0.232, tok1:0.024, tok2:0.037 |
| format_L21_layer_out_semantic | 17/82 | 0/82 | 0.622 | tok0:0.207, tok1:0.988, tok2:0.220 |
| format_L22_layer_input_semantic | 17/82 | 0/82 | 0.622 | tok0:0.207, tok1:0.988, tok2:0.220 |
| format_L22_layer_out_semantic | 17/82 | 0/82 | 0.622 | tok0:0.207, tok1:0.988, tok2:0.232 |
| format_L23_layer_input_semantic | 17/82 | 0/82 | 0.622 | tok0:0.207, tok1:0.988, tok2:0.232 |
| format_L21_layer_out | 1/82 | 16/82 | 0.232 | tok0:0.207, tok1:0.012, tok2:0.012 |
| format_L22_layer_input | 1/82 | 16/82 | 0.232 | tok0:0.207, tok1:0.012, tok2:0.012 |
| format_L22_layer_out | 1/82 | 16/82 | 0.232 | tok0:0.207, tok1:0.012, tok2:0.012 |
| format_L23_layer_input | 1/82 | 16/82 | 0.232 | tok0:0.207, tok1:0.012, tok2:0.012 |
| format_L20_layer_out_semantic | 13/82 | 0/82 | 0.476 | tok0:0.159, tok1:0.988, tok2:0.183 |

### Examples

- o17 / r31 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - semantic_cumulative_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - format_L25_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - format_L23_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - format_L24_layer_input_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
- o29 / r31 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` ?

I think` [' ?\n\n', 'I', ' think']
  - semantic_cumulative_only: ` ?

21` [' ?\n\n', '2', '1']
  - format_L25_layer_out_semantic: ` ?

21` [' ?\n\n', '2', '1']
  - format_L23_layer_out_semantic: ` ?

21` [' ?\n\n', '2', '1']
  - format_L24_layer_input_semantic: ` ?

21` [' ?\n\n', '2', '1']
- o95 / r64 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - semantic_cumulative_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - format_L25_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - format_L23_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - format_L24_layer_input_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
- o06 / r64 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - semantic_cumulative_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - format_L25_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - format_L23_layer_out_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
  - format_L24_layer_input_semantic: ` ?

2
` [' ?\n\n', '2', '\n']
- o17 / r64 correct=v91 old_wrong=v05
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - repair_prompt: ` c77` [' c', '7', '7']
  - semantic_cumulative_only: ` ?

99` [' ?\n\n', '9', '9']
  - format_L25_layer_out_semantic: ` c91` [' c', '9', '1']
  - format_L23_layer_out_semantic: ` c91` [' c', '9', '1']
  - format_L24_layer_input_semantic: ` c91` [' c', '9', '1']

## glm4

- rows: 31 / raw 256
- target cases seen: 31
- format layers: [32, 33, 34, 35, 36, 37]
- downstream layers: [34, 35, 36, 37, 38, 39]
- components: ['layer_input', 'attn_out', 'mlp_out', 'layer_out']
- tokenization: `{'v05': {'ids': [348, 100002], 'tokens': [' v', '05']}, 'v91': {'ids': [348, 104327], 'tokens': [' v', '91']}, 'v22': {'ids': [348, 99241], 'tokens': [' v', '22']}, 'v48': {'ids': [348, 100933], 'tokens': [' v', '48']}}`

### Baselines

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| base | 2/31 | 9/31 | 0.419 | tok0:0.355, tok1:0.065 |
| repair_prompt | 28/31 | 1/31 | 1.839 | tok0:0.935, tok1:0.903 |
| semantic_cumulative_only | 11/31 | 0/31 | 0.710 | tok0:0.355, tok1:1.000 |
| semantic_cumulative_random | 2/31 | 9/31 | 0.419 | tok0:0.355, tok1:0.226 |

### Best Exact

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| format_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L33_layer_input_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L34_layer_input_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L35_layer_input_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L35_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L36_layer_input_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L32_layer_input_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| format_L36_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| format_L37_layer_input_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| format_L37_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| repair_prompt | 28/31 | 1/31 | 1.839 | tok0:0.935, tok1:0.903 |
| format_L32_attn_out_semantic | 19/31 | 0/31 | 1.226 | tok0:0.613, tok1:1.000 |
| format_L33_attn_out_semantic | 16/31 | 0/31 | 1.032 | tok0:0.516, tok1:1.000 |
| format_L35_mlp_out_semantic | 15/31 | 0/31 | 0.968 | tok0:0.484, tok1:1.000 |
| format_L34_mlp_out_semantic | 14/31 | 0/31 | 0.903 | tok0:0.452, tok1:1.000 |
| format_L33_layer_input_random_semantic | 13/31 | 0/31 | 0.839 | tok0:0.419, tok1:1.000 |
| format_L34_attn_out_semantic | 13/31 | 0/31 | 0.839 | tok0:0.419, tok1:1.000 |
| format_L35_attn_out_semantic | 13/31 | 0/31 | 0.839 | tok0:0.419, tok1:1.000 |

### Best Tok0

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| format_L32_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L33_layer_input_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L33_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L34_layer_input_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L34_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L35_layer_input_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L35_layer_out_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L36_layer_input_semantic | 30/31 | 0/31 | 1.935 | tok0:0.968, tok1:1.000 |
| format_L32_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| format_L33_layer_input | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| format_L33_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| format_L34_layer_input | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| format_L34_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| format_L35_layer_input | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| format_L35_layer_out | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| format_L36_layer_input | 5/31 | 25/31 | 1.129 | tok0:0.968, tok1:0.161 |
| format_L32_layer_input_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| format_L36_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| format_L37_layer_input_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |
| format_L37_layer_out_semantic | 29/31 | 0/31 | 1.871 | tok0:0.935, tok1:1.000 |

### Examples

- o43 / r31 correct=v05 old_wrong=v22
  - base: ` v22` [' v', '22']
  - repair_prompt: ` v05` [' v', '05']
  - semantic_cumulative_only: ` v05` [' v', '05']
  - format_L32_layer_out_semantic: ` v05` [' v', '05']
  - format_L33_layer_input_semantic: ` v05` [' v', '05']
  - format_L33_layer_out_semantic: ` v05` [' v', '05']
- o95 / r64 correct=v05 old_wrong=v91
  - base: ` o95` [' o', '95']
  - repair_prompt: ` v91` [' v', '91']
  - semantic_cumulative_only: ` o05` [' o', '05']
  - format_L32_layer_out_semantic: ` v05` [' v', '05']
  - format_L33_layer_input_semantic: ` v05` [' v', '05']
  - format_L33_layer_out_semantic: ` v05` [' v', '05']
- o43 / r31 correct=v05 old_wrong=v48
  - base: ` v48` [' v', '48']
  - repair_prompt: ` c77` [' c', '77']
  - semantic_cumulative_only: ` v05` [' v', '05']
  - format_L32_layer_out_semantic: ` c05` [' c', '05']
  - format_L33_layer_input_semantic: ` c05` [' c', '05']
  - format_L33_layer_out_semantic: ` c05` [' c', '05']
- o17 / r64 correct=v05 old_wrong=v91
  - base: ` o17` [' o', '17']
  - repair_prompt: ` v05` [' v', '05']
  - semantic_cumulative_only: ` o05` [' o', '05']
  - format_L32_layer_out_semantic: ` v05` [' v', '05']
  - format_L33_layer_input_semantic: ` v05` [' v', '05']
  - format_L33_layer_out_semantic: ` v05` [' v', '05']
- o43 / r64 correct=v05 old_wrong=v91
  - base: ` o43` [' o', '43']
  - repair_prompt: ` v05` [' v', '05']
  - semantic_cumulative_only: ` o05` [' o', '05']
  - format_L32_layer_out_semantic: ` v05` [' v', '05']
  - format_L33_layer_input_semantic: ` v05` [' v', '05']
  - format_L33_layer_out_semantic: ` v05` [' v', '05']

## qwen3

- rows: 17 / raw 256
- target cases seen: 17
- format layers: [27, 28, 29, 30, 31, 32]
- downstream layers: [29, 30, 31, 32, 33, 34, 35]
- components: ['layer_input', 'attn_out', 'mlp_out', 'layer_out']
- tokenization: `{'v05': {'ids': [348, 15, 20], 'tokens': [' v', '0', '5']}, 'v91': {'ids': [348, 24, 16], 'tokens': [' v', '9', '1']}, 'v22': {'ids': [348, 17, 17], 'tokens': [' v', '2', '2']}, 'v48': {'ids': [348, 19, 23], 'tokens': [' v', '4', '8']}}`

### Baselines

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| base | 1/17 | 9/17 | 0.706 | tok0:0.588, tok1:0.059, tok2:0.059 |
| repair_prompt | 11/17 | 3/17 | 2.118 | tok0:0.824, tok1:0.824, tok2:0.824 |
| semantic_cumulative_only | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.824 |
| semantic_cumulative_random | 2/17 | 8/17 | 0.824 | tok0:0.588, tok1:0.235, tok2:0.176 |

### Best Exact

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| format_L27_layer_out_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| format_L28_layer_input_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| format_L27_layer_input_semantic | 12/17 | 0/17 | 2.118 | tok0:0.706, tok1:1.000, tok2:1.000 |
| format_L28_attn_out_semantic | 12/17 | 0/17 | 2.118 | tok0:0.706, tok1:1.000, tok2:0.824 |
| repair_prompt | 11/17 | 3/17 | 2.118 | tok0:0.824, tok1:0.824, tok2:0.824 |
| format_L27_mlp_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L28_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| format_L29_layer_input_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| format_L30_attn_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L31_attn_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L31_mlp_out_random_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L31_mlp_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L32_attn_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L32_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| format_L27_attn_out_random_semantic | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.824 |
| format_L27_attn_out_semantic | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.824 |
| format_L27_mlp_out_random_semantic | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.824 |
| format_L28_attn_out_random_semantic | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.765 |
| format_L28_mlp_out_random_semantic | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.765 |
| format_L28_mlp_out_semantic | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.765 |

### Best Tok0

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| repair_prompt | 11/17 | 3/17 | 2.118 | tok0:0.824, tok1:0.824, tok2:0.824 |
| format_L27_layer_out_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| format_L28_layer_input_semantic | 13/17 | 0/17 | 2.294 | tok0:0.765, tok1:1.000, tok2:1.000 |
| format_L27_layer_out | 3/17 | 10/17 | 1.118 | tok0:0.765, tok1:0.235, tok2:0.235 |
| format_L28_layer_input | 3/17 | 10/17 | 1.118 | tok0:0.765, tok1:0.235, tok2:0.235 |
| format_L27_layer_input_semantic | 12/17 | 0/17 | 2.118 | tok0:0.706, tok1:1.000, tok2:1.000 |
| format_L28_attn_out_semantic | 12/17 | 0/17 | 2.118 | tok0:0.706, tok1:1.000, tok2:0.824 |
| format_L27_layer_input | 2/17 | 10/17 | 0.941 | tok0:0.706, tok1:0.176, tok2:0.176 |
| format_L28_attn_out | 2/17 | 10/17 | 0.941 | tok0:0.706, tok1:0.118, tok2:0.118 |
| format_L27_mlp_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L28_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| format_L29_layer_input_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| format_L30_attn_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L31_attn_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L31_mlp_out_random_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L31_mlp_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L32_attn_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:0.824 |
| format_L32_layer_out_semantic | 11/17 | 0/17 | 1.941 | tok0:0.647, tok1:1.000, tok2:1.000 |
| format_L27_mlp_out | 2/17 | 9/17 | 0.882 | tok0:0.647, tok1:0.118, tok2:0.118 |
| format_L28_layer_out | 2/17 | 9/17 | 0.882 | tok0:0.647, tok1:0.176, tok2:0.176 |

### Examples

- o58 / r31 correct=v05 old_wrong=v22
  - base: ` v22` [' v', '2', '2']
  - repair_prompt: ` v05` [' v', '0', '5']
  - semantic_cumulative_only: ` v05` [' v', '0', '5']
  - format_L27_layer_out_semantic: ` v05` [' v', '0', '5']
  - format_L28_layer_input_semantic: ` v05` [' v', '0', '5']
  - format_L27_layer_input_semantic: ` v05` [' v', '0', '5']
- o95 / r64 correct=v05 old_wrong=v91
  - base: ` v05` [' v', '0', '5']
  - repair_prompt: ` v05` [' v', '0', '5']
  - semantic_cumulative_only: ` v05` [' v', '0', '5']
  - format_L27_layer_out_semantic: ` v05` [' v', '0', '5']
  - format_L28_layer_input_semantic: ` v05` [' v', '0', '5']
  - format_L27_layer_input_semantic: ` v05` [' v', '0', '5']
- o06 / r31 correct=v05 old_wrong=v22
  - base: ` v22` [' v', '2', '2']
  - repair_prompt: ` v05` [' v', '0', '5']
  - semantic_cumulative_only: ` v05` [' v', '0', '5']
  - format_L27_layer_out_semantic: ` v05` [' v', '0', '5']
  - format_L28_layer_input_semantic: ` v05` [' v', '0', '5']
  - format_L27_layer_input_semantic: ` v05` [' v', '0', '5']
- o58 / r31 correct=v22 old_wrong=v48
  - base: ` v48` [' v', '4', '8']
  - repair_prompt: ` 22` [' ', '2', '2']
  - semantic_cumulative_only: ` v22` [' v', '2', '2']
  - format_L27_layer_out_semantic: ` 22` [' ', '2', '2']
  - format_L28_layer_input_semantic: ` 22` [' ', '2', '2']
  - format_L27_layer_input_semantic: ` 22` [' ', '2', '2']
- o06 / r31 correct=v05 old_wrong=v48
  - base: ` v48` [' v', '4', '8']
  - repair_prompt: ` v48` [' v', '4', '8']
  - semantic_cumulative_only: ` v05` [' v', '0', '5']
  - format_L27_layer_out_semantic: ` v05` [' v', '0', '5']
  - format_L28_layer_input_semantic: ` v05` [' v', '0', '5']
  - format_L27_layer_input_semantic: ` v05` [' v', '0', '5']
