# Phase 628 Cross-Model Summary

Prefix/format gate and semantic value gate integration in natural greedy generation.

## deepseek7b

- rows: 82 / raw 256
- target cases seen: 82
- result layers: [22]
- downstream layers: [22, 23, 24, 25, 26, 27]
- tokenization: `{'v05': {'ids': [348, 15, 20], 'tokens': [' v', '0', '5']}, 'v91': {'ids': [348, 24, 16], 'tokens': [' v', '9', '1']}, 'v22': {'ids': [348, 17, 17], 'tokens': [' v', '2', '2']}, 'v48': {'ids': [348, 19, 23], 'tokens': [' v', '4', '8']}}`

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| base | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.000, tok2:0.000 |
| repair_prompt | 20/82 | 0/82 | 0.732 | tok0:0.244, tok1:0.256, tok2:0.256 |
| prefix_forced_only | 3/82 | 79/82 | 1.073 | tok0:1.000, tok1:0.037, tok2:0.037 |
| result_only | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.902, tok2:0.049 |
| result_random | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.110, tok2:0.073 |
| prefix_forced_result_only | 74/82 | 7/82 | 2.805 | tok0:1.000, tok1:0.902, tok2:0.902 |
| prefix_forced_result_random | 9/82 | 71/82 | 1.220 | tok0:1.000, tok1:0.110, tok2:0.110 |
| cumulative_layer_out | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.988, tok2:0.024 |
| cumulative_layer_out_random | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.098, tok2:0.061 |
| prefix_forced_cumulative_layer_out | 81/82 | 1/82 | 2.976 | tok0:1.000, tok1:0.988, tok2:0.988 |
| prefix_forced_cumulative_layer_out_random | 8/82 | 62/82 | 1.195 | tok0:1.000, tok1:0.098, tok2:0.098 |
| final_output_all | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.000, tok2:0.293 |
| prefix_forced_final_output_all | 0/82 | 0/82 | 1.000 | tok0:1.000, tok1:0.000, tok2:0.293 |
| final_output_random_all | 0/82 | 0/82 | 0.000 | tok0:0.000, tok1:0.000, tok2:0.049 |

### Examples

- o17 / r31 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - prefix_forced_only: ` v48` [' v', '4', '8']
  - result_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - prefix_forced_result_only: ` v22` [' v', '2', '2']
  - cumulative_layer_out: ` ?

2
` [' ?\n\n', '2', '\n']
  - prefix_forced_cumulative_layer_out: ` v22` [' v', '2', '2']
  - final_output_all: ` ?

 ?

2` [' ?\n\n', ' ?\n\n', '2']
  - prefix_forced_final_output_all: ` v ?

2` [' v', ' ?\n\n', '2']
- o29 / r31 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - prefix_forced_only: ` v48` [' v', '4', '8']
  - result_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - prefix_forced_result_only: ` v22` [' v', '2', '2']
  - cumulative_layer_out: ` ?

21` [' ?\n\n', '2', '1']
  - prefix_forced_cumulative_layer_out: ` v22` [' v', '2', '2']
  - final_output_all: ` ?

 ?

2` [' ?\n\n', ' ?\n\n', '2']
  - prefix_forced_final_output_all: ` v ?

2` [' v', ' ?\n\n', '2']
- o95 / r64 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - prefix_forced_only: ` v48` [' v', '4', '8']
  - result_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - prefix_forced_result_only: ` v22` [' v', '2', '2']
  - cumulative_layer_out: ` ?

2
` [' ?\n\n', '2', '\n']
  - prefix_forced_cumulative_layer_out: ` v22` [' v', '2', '2']
  - final_output_all: ` ?

 ?

2` [' ?\n\n', ' ?\n\n', '2']
  - prefix_forced_final_output_all: ` v ?

2` [' v', ' ?\n\n', '2']
- o06 / r64 correct=v22 old_wrong=v48
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - prefix_forced_only: ` v48` [' v', '4', '8']
  - result_only: ` ?

2
` [' ?\n\n', '2', '\n']
  - prefix_forced_result_only: ` v22` [' v', '2', '2']
  - cumulative_layer_out: ` ?

2
` [' ?\n\n', '2', '\n']
  - prefix_forced_cumulative_layer_out: ` v22` [' v', '2', '2']
  - final_output_all: ` ?

 ?

2` [' ?\n\n', ' ?\n\n', '2']
  - prefix_forced_final_output_all: ` v ?

2` [' v', ' ?\n\n', '2']
- o17 / r64 correct=v91 old_wrong=v05
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - prefix_forced_only: ` v05` [' v', '0', '5']
  - result_only: ` ?

99` [' ?\n\n', '9', '9']
  - prefix_forced_result_only: ` v91` [' v', '9', '1']
  - cumulative_layer_out: ` ?

99` [' ?\n\n', '9', '9']
  - prefix_forced_cumulative_layer_out: ` v91` [' v', '9', '1']
  - final_output_all: ` ?

 c9` [' ?\n\n', ' c', '9']
  - prefix_forced_final_output_all: ` v c9` [' v', ' c', '9']
- o43 / r64 correct=v91 old_wrong=v05
  - base: ` ?

To solve` [' ?\n\n', 'To', ' solve']
  - prefix_forced_only: ` v05` [' v', '0', '5']
  - result_only: ` ?

99` [' ?\n\n', '9', '9']
  - prefix_forced_result_only: ` v91` [' v', '9', '1']
  - cumulative_layer_out: ` ?

99` [' ?\n\n', '9', '9']
  - prefix_forced_cumulative_layer_out: ` v91` [' v', '9', '1']
  - final_output_all: ` ?

 c9` [' ?\n\n', ' c', '9']
  - prefix_forced_final_output_all: ` v c9` [' v', ' c', '9']

## glm4

- rows: 31 / raw 256
- target cases seen: 31
- result layers: [34]
- downstream layers: [34, 35, 36, 37, 38, 39]
- tokenization: `{'v05': {'ids': [348, 100002], 'tokens': [' v', '05']}, 'v91': {'ids': [348, 104327], 'tokens': [' v', '91']}, 'v22': {'ids': [348, 99241], 'tokens': [' v', '22']}, 'v48': {'ids': [348, 100933], 'tokens': [' v', '48']}}`

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| base | 2/31 | 9/31 | 0.419 | tok0:0.355, tok1:0.065 |
| repair_prompt | 28/31 | 1/31 | 1.839 | tok0:0.935, tok1:0.903 |
| prefix_forced_only | 5/31 | 26/31 | 1.161 | tok0:1.000, tok1:0.161 |
| result_only | 10/31 | 0/31 | 0.677 | tok0:0.355, tok1:0.935 |
| result_random | 0/31 | 11/31 | 0.355 | tok0:0.355, tok1:0.065 |
| prefix_forced_result_only | 29/31 | 1/31 | 1.935 | tok0:1.000, tok1:0.935 |
| prefix_forced_result_random | 2/31 | 29/31 | 1.065 | tok0:1.000, tok1:0.065 |
| cumulative_layer_out | 11/31 | 0/31 | 0.710 | tok0:0.355, tok1:1.000 |
| cumulative_layer_out_random | 2/31 | 9/31 | 0.419 | tok0:0.355, tok1:0.226 |
| prefix_forced_cumulative_layer_out | 31/31 | 0/31 | 2.000 | tok0:1.000, tok1:1.000 |
| prefix_forced_cumulative_layer_out_random | 7/31 | 24/31 | 1.226 | tok0:1.000, tok1:0.226 |
| final_output_all | 0/31 | 0/31 | 0.355 | tok0:0.355, tok1:0.000 |
| prefix_forced_final_output_all | 0/31 | 0/31 | 1.000 | tok0:1.000, tok1:0.000 |
| final_output_random_all | 0/31 | 0/31 | 0.355 | tok0:0.355, tok1:0.000 |

### Examples

- o43 / r31 correct=v05 old_wrong=v22
  - base: ` v22` [' v', '22']
  - prefix_forced_only: ` v22` [' v', '22']
  - result_only: ` v05` [' v', '05']
  - prefix_forced_result_only: ` v05` [' v', '05']
  - cumulative_layer_out: ` v05` [' v', '05']
  - prefix_forced_cumulative_layer_out: ` v05` [' v', '05']
  - final_output_all: ` v v` [' v', ' v']
  - prefix_forced_final_output_all: ` v v` [' v', ' v']
- o95 / r64 correct=v05 old_wrong=v91
  - base: ` o95` [' o', '95']
  - prefix_forced_only: ` v91` [' v', '91']
  - result_only: ` o05` [' o', '05']
  - prefix_forced_result_only: ` v05` [' v', '05']
  - cumulative_layer_out: ` o05` [' o', '05']
  - prefix_forced_cumulative_layer_out: ` v05` [' v', '05']
  - final_output_all: ` o v` [' o', ' v']
  - prefix_forced_final_output_all: ` v v` [' v', ' v']
- o43 / r31 correct=v05 old_wrong=v48
  - base: ` v48` [' v', '48']
  - prefix_forced_only: ` v48` [' v', '48']
  - result_only: ` v05` [' v', '05']
  - prefix_forced_result_only: ` v05` [' v', '05']
  - cumulative_layer_out: ` v05` [' v', '05']
  - prefix_forced_cumulative_layer_out: ` v05` [' v', '05']
  - final_output_all: ` v c` [' v', ' c']
  - prefix_forced_final_output_all: ` v c` [' v', ' c']
- o17 / r64 correct=v05 old_wrong=v91
  - base: ` o17` [' o', '17']
  - prefix_forced_only: ` v91` [' v', '91']
  - result_only: ` o05` [' o', '05']
  - prefix_forced_result_only: ` v05` [' v', '05']
  - cumulative_layer_out: ` o05` [' o', '05']
  - prefix_forced_cumulative_layer_out: ` v05` [' v', '05']
  - final_output_all: ` o v` [' o', ' v']
  - prefix_forced_final_output_all: ` v v` [' v', ' v']
- o43 / r64 correct=v05 old_wrong=v91
  - base: ` o43` [' o', '43']
  - prefix_forced_only: ` v91` [' v', '91']
  - result_only: ` o05` [' o', '05']
  - prefix_forced_result_only: ` v05` [' v', '05']
  - cumulative_layer_out: ` o05` [' o', '05']
  - prefix_forced_cumulative_layer_out: ` v05` [' v', '05']
  - final_output_all: ` o v` [' o', ' v']
  - prefix_forced_final_output_all: ` v v` [' v', ' v']
- o82 / r31 correct=v48 old_wrong=v22
  - base: ` o82` [' o', '82']
  - prefix_forced_only: ` v22` [' v', '22']
  - result_only: ` o22` [' o', '22']
  - prefix_forced_result_only: ` v22` [' v', '22']
  - cumulative_layer_out: ` o48` [' o', '48']
  - prefix_forced_cumulative_layer_out: ` v48` [' v', '48']
  - final_output_all: ` o v` [' o', ' v']
  - prefix_forced_final_output_all: ` v v` [' v', ' v']

## qwen3

- rows: 17 / raw 256
- target cases seen: 17
- result layers: [29]
- downstream layers: [29, 30, 31, 32, 33, 34, 35]
- tokenization: `{'v05': {'ids': [348, 15, 20], 'tokens': [' v', '0', '5']}, 'v91': {'ids': [348, 24, 16], 'tokens': [' v', '9', '1']}, 'v22': {'ids': [348, 17, 17], 'tokens': [' v', '2', '2']}, 'v48': {'ids': [348, 19, 23], 'tokens': [' v', '4', '8']}}`

| mode | exact | wrong_exact | prefix_len | position_correct |
|---|---:|---:|---:|---|
| base | 1/17 | 9/17 | 0.706 | tok0:0.588, tok1:0.059, tok2:0.059 |
| repair_prompt | 11/17 | 3/17 | 2.118 | tok0:0.824, tok1:0.824, tok2:0.824 |
| prefix_forced_only | 3/17 | 14/17 | 1.353 | tok0:1.000, tok1:0.176, tok2:0.176 |
| result_only | 8/17 | 2/17 | 1.529 | tok0:0.588, tok1:0.882, tok2:0.647 |
| result_random | 0/17 | 10/17 | 0.588 | tok0:0.588, tok1:0.059, tok2:0.059 |
| prefix_forced_result_only | 15/17 | 2/17 | 2.765 | tok0:1.000, tok1:0.882, tok2:0.882 |
| prefix_forced_result_random | 1/17 | 16/17 | 1.118 | tok0:1.000, tok1:0.059, tok2:0.059 |
| cumulative_layer_out | 10/17 | 0/17 | 1.765 | tok0:0.588, tok1:1.000, tok2:0.824 |
| cumulative_layer_out_random | 2/17 | 8/17 | 0.824 | tok0:0.588, tok1:0.235, tok2:0.176 |
| prefix_forced_cumulative_layer_out | 17/17 | 0/17 | 3.000 | tok0:1.000, tok1:1.000, tok2:1.000 |
| prefix_forced_cumulative_layer_out_random | 4/17 | 13/17 | 1.471 | tok0:1.000, tok1:0.235, tok2:0.235 |
| final_output_all | 0/17 | 0/17 | 0.588 | tok0:0.588, tok1:0.000, tok2:0.235 |
| prefix_forced_final_output_all | 0/17 | 0/17 | 1.000 | tok0:1.000, tok1:0.000, tok2:0.235 |
| final_output_random_all | 0/17 | 0/17 | 0.588 | tok0:0.588, tok1:0.000, tok2:0.000 |

### Examples

- o58 / r31 correct=v05 old_wrong=v22
  - base: ` v22` [' v', '2', '2']
  - prefix_forced_only: ` v22` [' v', '2', '2']
  - result_only: ` v05` [' v', '0', '5']
  - prefix_forced_result_only: ` v05` [' v', '0', '5']
  - cumulative_layer_out: ` v05` [' v', '0', '5']
  - prefix_forced_cumulative_layer_out: ` v05` [' v', '0', '5']
  - final_output_all: ` v v0` [' v', ' v', '0']
  - prefix_forced_final_output_all: ` v v0` [' v', ' v', '0']
- o95 / r64 correct=v05 old_wrong=v91
  - base: ` v05` [' v', '0', '5']
  - prefix_forced_only: ` v05` [' v', '0', '5']
  - result_only: ` v05` [' v', '0', '5']
  - prefix_forced_result_only: ` v05` [' v', '0', '5']
  - cumulative_layer_out: ` v05` [' v', '0', '5']
  - prefix_forced_cumulative_layer_out: ` v05` [' v', '0', '5']
  - final_output_all: ` v v0` [' v', ' v', '0']
  - prefix_forced_final_output_all: ` v v0` [' v', ' v', '0']
- o06 / r31 correct=v05 old_wrong=v22
  - base: ` v22` [' v', '2', '2']
  - prefix_forced_only: ` v22` [' v', '2', '2']
  - result_only: ` v05` [' v', '0', '5']
  - prefix_forced_result_only: ` v05` [' v', '0', '5']
  - cumulative_layer_out: ` v05` [' v', '0', '5']
  - prefix_forced_cumulative_layer_out: ` v05` [' v', '0', '5']
  - final_output_all: ` v v0` [' v', ' v', '0']
  - prefix_forced_final_output_all: ` v v0` [' v', ' v', '0']
- o58 / r31 correct=v22 old_wrong=v48
  - base: ` v48` [' v', '4', '8']
  - prefix_forced_only: ` v48` [' v', '4', '8']
  - result_only: ` v22` [' v', '2', '2']
  - prefix_forced_result_only: ` v22` [' v', '2', '2']
  - cumulative_layer_out: ` v22` [' v', '2', '2']
  - prefix_forced_cumulative_layer_out: ` v22` [' v', '2', '2']
  - final_output_all: ` v 2` [' v', ' ', '2']
  - prefix_forced_final_output_all: ` v 2` [' v', ' ', '2']
- o06 / r31 correct=v05 old_wrong=v48
  - base: ` v48` [' v', '4', '8']
  - prefix_forced_only: ` v48` [' v', '4', '8']
  - result_only: ` v48` [' v', '4', '8']
  - prefix_forced_result_only: ` v48` [' v', '4', '8']
  - cumulative_layer_out: ` v05` [' v', '0', '5']
  - prefix_forced_cumulative_layer_out: ` v05` [' v', '0', '5']
  - final_output_all: ` v v0` [' v', ' v', '0']
  - prefix_forced_final_output_all: ` v v0` [' v', ' v', '0']
- o71 / r31 correct=v91 old_wrong=v05
  - base: ` ?

Okay,` [' ?\n\n', 'Okay', ',']
  - prefix_forced_only: ` v05` [' v', '0', '5']
  - result_only: ` ?

95` [' ?\n\n', '9', '5']
  - prefix_forced_result_only: ` v91` [' v', '9', '1']
  - cumulative_layer_out: ` ?

95` [' ?\n\n', '9', '5']
  - prefix_forced_cumulative_layer_out: ` v91` [' v', '9', '1']
  - final_output_all: ` ?

 9` [' ?\n\n', ' ', '9']
  - prefix_forced_final_output_all: ` v 9` [' v', ' ', '9']
