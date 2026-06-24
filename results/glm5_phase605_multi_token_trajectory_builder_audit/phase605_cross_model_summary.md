# Phase605 Cross-Model Summary

Multi-token trajectory builder audit.

## qwen3

cases=96, rows=7, target_cases_seen=7, time_min=0.50

Tokenization example:

```text
v05: [' v', '0', '5']
v91: [' v', '9', '1']
v22: [' v', '2', '2']
v48: [' v', '4', '8']
```

### Best Patch Modes

| key | kind | group | random | n | switch | margin_gain | correct_delta | old_wrong_delta | c_tok0 | c_tok1 | c_tok2 | w_tok0 | w_tok1 | w_tok2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `input_all` | input | all | False | 7 | 7/7 | 7.960 | 2.581 | -5.379 | 0.817 | 1.761 | 0.002 | 0.817 | -6.060 | -0.136 |
| `input_digits` | input | digits | False | 7 | 7/7 | 7.960 | 1.763 | -6.196 | -0.000 | 1.761 | 0.002 | -0.000 | -6.060 | -0.136 |
| `output_all` | output | all | False | 7 | 7/7 | 7.960 | 2.581 | -5.379 | 0.817 | 1.761 | 0.002 | 0.817 | -6.060 | -0.136 |
| `output_digits` | output | digits | False | 7 | 7/7 | 7.960 | 1.763 | -6.196 | -0.000 | 1.761 | 0.002 | -0.000 | -6.060 | -0.136 |
| `input_digit1` | input | digit1 | False | 7 | 7/7 | 7.821 | 1.761 | -6.060 | -0.000 | 1.761 | -0.000 | -0.000 | -6.060 | -0.000 |
| `output_digit1` | output | digit1 | False | 7 | 7/7 | 7.821 | 1.761 | -6.060 | -0.000 | 1.761 | -0.000 | -0.000 | -6.060 | -0.000 |
| `output_all_random` | output | all | True | 7 | 2/7 | 0.160 | -0.145 | -0.304 | -0.144 | 0.000 | -0.001 | -0.124 | -0.180 | -0.000 |
| `output_digits_random` | output | digits | True | 7 | 2/7 | 0.108 | 0.029 | -0.079 | -0.000 | 0.030 | -0.001 | -0.000 | -0.079 | 0.000 |
| `input_digits_random` | input | digits | True | 7 | 1/7 | 0.165 | 0.149 | -0.016 | -0.000 | 0.151 | -0.002 | -0.000 | -0.015 | -0.001 |
| `input_all_random` | input | all | True | 7 | 1/7 | -0.274 | -0.396 | -0.122 | -0.253 | -0.143 | -0.001 | -0.151 | 0.029 | -0.000 |
| `input_digit2` | input | digit2 | False | 7 | 0/7 | 0.138 | 0.002 | -0.136 | -0.000 | -0.000 | 0.002 | -0.000 | -0.000 | -0.136 |
| `output_digit2` | output | digit2 | False | 7 | 0/7 | 0.138 | 0.002 | -0.136 | -0.000 | -0.000 | 0.002 | -0.000 | -0.000 | -0.136 |
| `input_prefix0` | input | prefix0 | False | 7 | 0/7 | 0.000 | 0.817 | 0.817 | 0.817 | -0.000 | -0.000 | 0.817 | -0.000 | -0.000 |
| `output_prefix0` | output | prefix0 | False | 7 | 0/7 | 0.000 | 0.817 | 0.817 | 0.817 | -0.000 | -0.000 | 0.817 | -0.000 | -0.000 |

### Watched Patch Modes

| key | kind | group | random | n | switch | margin_gain | correct_delta | old_wrong_delta | c_tok0 | c_tok1 | c_tok2 | w_tok0 | w_tok1 | w_tok2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `input_prefix0` | input | prefix0 | False | 7 | 0/7 | 0.000 | 0.817 | 0.817 | 0.817 | -0.000 | -0.000 | 0.817 | -0.000 | -0.000 |
| `input_digit1` | input | digit1 | False | 7 | 7/7 | 7.821 | 1.761 | -6.060 | -0.000 | 1.761 | -0.000 | -0.000 | -6.060 | -0.000 |
| `input_digit2` | input | digit2 | False | 7 | 0/7 | 0.138 | 0.002 | -0.136 | -0.000 | -0.000 | 0.002 | -0.000 | -0.000 | -0.136 |
| `input_digits` | input | digits | False | 7 | 7/7 | 7.960 | 1.763 | -6.196 | -0.000 | 1.761 | 0.002 | -0.000 | -6.060 | -0.136 |
| `input_all` | input | all | False | 7 | 7/7 | 7.960 | 2.581 | -5.379 | 0.817 | 1.761 | 0.002 | 0.817 | -6.060 | -0.136 |
| `output_prefix0` | output | prefix0 | False | 7 | 0/7 | 0.000 | 0.817 | 0.817 | 0.817 | -0.000 | -0.000 | 0.817 | -0.000 | -0.000 |
| `output_digit1` | output | digit1 | False | 7 | 7/7 | 7.821 | 1.761 | -6.060 | -0.000 | 1.761 | -0.000 | -0.000 | -6.060 | -0.000 |
| `output_digit2` | output | digit2 | False | 7 | 0/7 | 0.138 | 0.002 | -0.136 | -0.000 | -0.000 | 0.002 | -0.000 | -0.000 | -0.136 |
| `output_digits` | output | digits | False | 7 | 7/7 | 7.960 | 1.763 | -6.196 | -0.000 | 1.761 | 0.002 | -0.000 | -6.060 | -0.136 |
| `output_all` | output | all | False | 7 | 7/7 | 7.960 | 2.581 | -5.379 | 0.817 | 1.761 | 0.002 | 0.817 | -6.060 | -0.136 |
| `input_digits_random` | input | digits | True | 7 | 1/7 | 0.165 | 0.149 | -0.016 | -0.000 | 0.151 | -0.002 | -0.000 | -0.015 | -0.001 |
| `output_digits_random` | output | digits | True | 7 | 2/7 | 0.108 | 0.029 | -0.079 | -0.000 | 0.030 | -0.001 | -0.000 | -0.079 | 0.000 |

## glm4

cases=96, rows=13, target_cases_seen=13, time_min=0.93

Tokenization example:

```text
v05: [' v', '05']
v91: [' v', '91']
v22: [' v', '22']
v48: [' v', '48']
```

### Best Patch Modes

| key | kind | group | random | n | switch | margin_gain | correct_delta | old_wrong_delta | c_tok0 | c_tok1 | c_tok2 | w_tok0 | w_tok1 | w_tok2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `input_all` | input | all | False | 13 | 13/13 | 2.913 | 0.330 | -2.584 | -0.671 | 1.001 | 0.000 | -0.671 | -1.913 | 0.000 |
| `input_digit1` | input | digit1 | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | -0.000 | 1.001 | 0.000 | -0.000 | -1.913 | 0.000 |
| `input_digits` | input | digits | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | -0.000 | 1.001 | 0.000 | -0.000 | -1.913 | 0.000 |
| `output_all` | output | all | False | 13 | 13/13 | 2.913 | 0.330 | -2.584 | -0.671 | 1.001 | 0.000 | -0.671 | -1.913 | 0.000 |
| `output_digit1` | output | digit1 | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | -0.000 | 1.001 | 0.000 | -0.000 | -1.913 | 0.000 |
| `output_digits` | output | digits | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | -0.000 | 1.001 | 0.000 | -0.000 | -1.913 | 0.000 |
| `input_digit2` | input | digit2 | False | 13 | 8/13 | 0.774 | -96.773 | -97.547 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `output_digit2` | output | digit2 | False | 13 | 8/13 | 0.774 | -96.773 | -97.547 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `input_all_random` | input | all | True | 13 | 4/13 | -0.118 | -0.727 | -0.609 | -0.624 | -0.103 | 0.000 | -0.397 | -0.212 | 0.000 |
| `input_digits_random` | input | digits | True | 13 | 2/13 | 0.217 | -0.043 | -0.261 | -0.000 | -0.043 | 0.000 | -0.000 | -0.261 | 0.000 |
| `output_digits_random` | output | digits | True | 13 | 2/13 | -0.037 | -0.099 | -0.062 | -0.000 | -0.099 | 0.000 | -0.000 | -0.062 | 0.000 |
| `output_all_random` | output | all | True | 13 | 2/13 | -0.266 | -0.365 | -0.099 | -0.215 | -0.150 | 0.000 | -0.023 | -0.076 | 0.000 |
| `input_prefix0` | input | prefix0 | False | 13 | 0/13 | 0.000 | -0.671 | -0.671 | -0.671 | -0.000 | 0.000 | -0.671 | -0.000 | 0.000 |
| `output_prefix0` | output | prefix0 | False | 13 | 0/13 | 0.000 | -0.671 | -0.671 | -0.671 | -0.000 | 0.000 | -0.671 | -0.000 | 0.000 |

### Watched Patch Modes

| key | kind | group | random | n | switch | margin_gain | correct_delta | old_wrong_delta | c_tok0 | c_tok1 | c_tok2 | w_tok0 | w_tok1 | w_tok2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `input_prefix0` | input | prefix0 | False | 13 | 0/13 | 0.000 | -0.671 | -0.671 | -0.671 | -0.000 | 0.000 | -0.671 | -0.000 | 0.000 |
| `input_digit1` | input | digit1 | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | -0.000 | 1.001 | 0.000 | -0.000 | -1.913 | 0.000 |
| `input_digit2` | input | digit2 | False | 13 | 8/13 | 0.774 | -96.773 | -97.547 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `input_digits` | input | digits | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | -0.000 | 1.001 | 0.000 | -0.000 | -1.913 | 0.000 |
| `input_all` | input | all | False | 13 | 13/13 | 2.913 | 0.330 | -2.584 | -0.671 | 1.001 | 0.000 | -0.671 | -1.913 | 0.000 |
| `output_prefix0` | output | prefix0 | False | 13 | 0/13 | 0.000 | -0.671 | -0.671 | -0.671 | -0.000 | 0.000 | -0.671 | -0.000 | 0.000 |
| `output_digit1` | output | digit1 | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | -0.000 | 1.001 | 0.000 | -0.000 | -1.913 | 0.000 |
| `output_digit2` | output | digit2 | False | 13 | 8/13 | 0.774 | -96.773 | -97.547 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `output_digits` | output | digits | False | 13 | 13/13 | 2.913 | 1.001 | -1.913 | -0.000 | 1.001 | 0.000 | -0.000 | -1.913 | 0.000 |
| `output_all` | output | all | False | 13 | 13/13 | 2.913 | 0.330 | -2.584 | -0.671 | 1.001 | 0.000 | -0.671 | -1.913 | 0.000 |
| `input_digits_random` | input | digits | True | 13 | 2/13 | 0.217 | -0.043 | -0.261 | -0.000 | -0.043 | 0.000 | -0.000 | -0.261 | 0.000 |
| `output_digits_random` | output | digits | True | 13 | 2/13 | -0.037 | -0.099 | -0.062 | -0.000 | -0.099 | 0.000 | -0.000 | -0.062 | 0.000 |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, time_min=1.64

Tokenization example:

```text
v05: [' v', '0', '5']
v91: [' v', '9', '1']
v22: [' v', '2', '2']
v48: [' v', '4', '8']
```

### Best Patch Modes

| key | kind | group | random | n | switch | margin_gain | correct_delta | old_wrong_delta | c_tok0 | c_tok1 | c_tok2 | w_tok0 | w_tok1 | w_tok2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `input_all` | input | all | False | 37 | 37/37 | 7.716 | 7.443 | -0.273 | 5.370 | 2.048 | 0.024 | 5.370 | -5.001 | -0.642 |
| `input_digits` | input | digits | False | 37 | 37/37 | 7.716 | 2.073 | -5.643 | -0.000 | 2.048 | 0.024 | -0.000 | -5.001 | -0.642 |
| `output_all` | output | all | False | 37 | 37/37 | 7.716 | 7.443 | -0.273 | 5.370 | 2.048 | 0.024 | 5.370 | -5.001 | -0.642 |
| `output_digits` | output | digits | False | 37 | 37/37 | 7.716 | 2.073 | -5.643 | -0.000 | 2.048 | 0.024 | -0.000 | -5.001 | -0.642 |
| `input_digit1` | input | digit1 | False | 37 | 37/37 | 7.049 | 2.048 | -5.001 | -0.000 | 2.048 | -0.000 | -0.000 | -5.001 | -0.000 |
| `output_digit1` | output | digit1 | False | 37 | 37/37 | 7.049 | 2.048 | -5.001 | -0.000 | 2.048 | -0.000 | -0.000 | -5.001 | -0.000 |
| `output_all_random` | output | all | True | 37 | 13/37 | 0.507 | -0.543 | -1.050 | -0.250 | -0.280 | -0.013 | -0.705 | -0.316 | -0.029 |
| `input_digit2` | input | digit2 | False | 37 | 7/37 | 0.667 | 0.024 | -0.642 | -0.000 | -0.000 | 0.024 | -0.000 | -0.000 | -0.642 |
| `output_digit2` | output | digit2 | False | 37 | 7/37 | 0.667 | 0.024 | -0.642 | -0.000 | -0.000 | 0.024 | -0.000 | -0.000 | -0.642 |
| `input_all_random` | input | all | True | 37 | 6/37 | -0.356 | -0.688 | -0.332 | -0.175 | -0.473 | -0.040 | 0.034 | -0.326 | -0.040 |
| `input_digits_random` | input | digits | True | 37 | 4/37 | 0.131 | -0.310 | -0.441 | -0.000 | -0.270 | -0.041 | -0.000 | -0.358 | -0.084 |
| `output_digits_random` | output | digits | True | 37 | 3/37 | -0.061 | -0.386 | -0.325 | -0.000 | -0.349 | -0.037 | -0.000 | -0.241 | -0.084 |
| `input_prefix0` | input | prefix0 | False | 37 | 0/37 | 0.000 | 5.370 | 5.370 | 5.370 | -0.000 | -0.000 | 5.370 | -0.000 | -0.000 |
| `output_prefix0` | output | prefix0 | False | 37 | 0/37 | 0.000 | 5.370 | 5.370 | 5.370 | -0.000 | -0.000 | 5.370 | -0.000 | -0.000 |

### Watched Patch Modes

| key | kind | group | random | n | switch | margin_gain | correct_delta | old_wrong_delta | c_tok0 | c_tok1 | c_tok2 | w_tok0 | w_tok1 | w_tok2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `input_prefix0` | input | prefix0 | False | 37 | 0/37 | 0.000 | 5.370 | 5.370 | 5.370 | -0.000 | -0.000 | 5.370 | -0.000 | -0.000 |
| `input_digit1` | input | digit1 | False | 37 | 37/37 | 7.049 | 2.048 | -5.001 | -0.000 | 2.048 | -0.000 | -0.000 | -5.001 | -0.000 |
| `input_digit2` | input | digit2 | False | 37 | 7/37 | 0.667 | 0.024 | -0.642 | -0.000 | -0.000 | 0.024 | -0.000 | -0.000 | -0.642 |
| `input_digits` | input | digits | False | 37 | 37/37 | 7.716 | 2.073 | -5.643 | -0.000 | 2.048 | 0.024 | -0.000 | -5.001 | -0.642 |
| `input_all` | input | all | False | 37 | 37/37 | 7.716 | 7.443 | -0.273 | 5.370 | 2.048 | 0.024 | 5.370 | -5.001 | -0.642 |
| `output_prefix0` | output | prefix0 | False | 37 | 0/37 | 0.000 | 5.370 | 5.370 | 5.370 | -0.000 | -0.000 | 5.370 | -0.000 | -0.000 |
| `output_digit1` | output | digit1 | False | 37 | 37/37 | 7.049 | 2.048 | -5.001 | -0.000 | 2.048 | -0.000 | -0.000 | -5.001 | -0.000 |
| `output_digit2` | output | digit2 | False | 37 | 7/37 | 0.667 | 0.024 | -0.642 | -0.000 | -0.000 | 0.024 | -0.000 | -0.000 | -0.642 |
| `output_digits` | output | digits | False | 37 | 37/37 | 7.716 | 2.073 | -5.643 | -0.000 | 2.048 | 0.024 | -0.000 | -5.001 | -0.642 |
| `output_all` | output | all | False | 37 | 37/37 | 7.716 | 7.443 | -0.273 | 5.370 | 2.048 | 0.024 | 5.370 | -5.001 | -0.642 |
| `input_digits_random` | input | digits | True | 37 | 4/37 | 0.131 | -0.310 | -0.441 | -0.000 | -0.270 | -0.041 | -0.000 | -0.358 | -0.084 |
| `output_digits_random` | output | digits | True | 37 | 3/37 | -0.061 | -0.386 | -0.325 | -0.000 | -0.349 | -0.037 | -0.000 | -0.241 | -0.084 |

