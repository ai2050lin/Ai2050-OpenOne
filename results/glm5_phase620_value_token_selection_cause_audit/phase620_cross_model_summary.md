# Phase 620 Cross Model Summary

Q/K cause audit for correct value-token attention selection.

## qwen3

rows=9, target_seen=9, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 119, 'empty_correct_value': 0}, layers=[27, 28, 29], heads={'27': [11, 23, 6, 14, 5, 2], '28': [11, 23, 6, 14, 5, 2], '29': [11, 23, 6, 14, 5, 2]}, time_min=0.58

### causal_patch

| mode | switch | margin | correct_delta | wrong_delta | positive_margin |
|---|---:|---:|---:|---:|---:|
| `q_only` | 4/9 | +1.182 | +0.757 | -0.425 | 9/9 |
| `qk_all_value_rule_lines` | 4/9 | +1.182 | +0.757 | -0.425 | 9/9 |
| `qk_correct_value` | 4/9 | +1.182 | +0.757 | -0.425 | 9/9 |
| `k_all_value_rule_lines` | 0/9 | +0.00000 | +0.00000 | +0.00000 | 0/9 |
| `k_correct_value` | 0/9 | +0.00000 | +0.00000 | +0.00000 | 0/9 |
| `q_random_same_norm` | 0/9 | +0.00000 | +0.00000 | +0.00000 | 0/9 |

### alpha_mass

| group | base | repair | q_only | q_random | repair-base | q-base | random-base |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct_value_token | +0.02953 | +0.08062 | +0.08076 | +0.03105 | +0.05108 | +0.05123 | +0.00152 |
| correct_rule_line | +0.03408 | +0.08722 | +0.08737 | +0.03592 | +0.05314 | +0.05329 | +0.00183 |
| all_value_rule_lines | +0.199 | +0.212 | +0.211 | +0.206 | +0.01326 | +0.01259 | +0.00696 |
| wrong_same_relation_lines | +0.110 | +0.07476 | +0.07431 | +0.112 | -0.03511 | -0.03556 | +0.00249 |
| wrong_same_category_lines | +0.01471 | +0.01683 | +0.01675 | +0.01550 | +0.00212 | +0.00204 | +0.00079 |

## glm4

rows=12, target_seen=12, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 116, 'empty_correct_value': 0}, layers=[32, 33, 34], heads={'32': [12, 8, 4, 28, 6, 7], '33': [12, 8, 4, 28, 6, 7], '34': [12, 8, 4, 28, 6, 7]}, time_min=0.94

### causal_patch

| mode | switch | margin | correct_delta | wrong_delta | positive_margin |
|---|---:|---:|---:|---:|---:|
| `q_only` | 1/12 | +0.01563 | -0.01563 | -0.03126 | 5/12 |
| `qk_all_value_rule_lines` | 1/12 | +0.01563 | -0.01563 | -0.03126 | 5/12 |
| `qk_correct_value` | 1/12 | +0.01563 | -0.01563 | -0.03126 | 5/12 |
| `k_all_value_rule_lines` | 0/12 | +0.00000 | +0.00000 | +0.00000 | 0/12 |
| `k_correct_value` | 0/12 | +0.00000 | +0.00000 | +0.00000 | 0/12 |
| `q_random_same_norm` | 0/12 | +0.00000 | +0.00000 | +0.00000 | 0/12 |

### alpha_mass

| group | base | repair | q_only | q_random | repair-base | q-base | random-base |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct_value_token | +0.04423 | +0.06613 | +0.06601 | +0.04528 | +0.02190 | +0.02178 | +0.00105 |
| correct_rule_line | +0.04908 | +0.07257 | +0.07240 | +0.05012 | +0.02348 | +0.02331 | +0.00103 |
| all_value_rule_lines | +0.393 | +0.360 | +0.359 | +0.389 | -0.03267 | -0.03336 | -0.00377 |
| wrong_same_relation_lines | +0.220 | +0.171 | +0.171 | +0.220 | -0.04903 | -0.04958 | -0.00045 |
| wrong_same_category_lines | +0.05084 | +0.04829 | +0.04824 | +0.04776 | -0.00255 | -0.00260 | -0.00307 |

## deepseek7b

rows=43, target_seen=43, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 85, 'empty_correct_value': 0}, layers=[20, 21, 22], heads={'20': [3, 1, 7, 24, 25, 13], '21': [3, 1, 7, 24, 25, 13], '22': [3, 1, 7, 24, 25, 13]}, time_min=1.38

### causal_patch

| mode | switch | margin | correct_delta | wrong_delta | positive_margin |
|---|---:|---:|---:|---:|---:|
| `q_only` | 33/43 | +1.769 | +1.113 | -0.656 | 43/43 |
| `qk_all_value_rule_lines` | 33/43 | +1.769 | +1.113 | -0.656 | 43/43 |
| `qk_correct_value` | 33/43 | +1.769 | +1.113 | -0.656 | 43/43 |
| `k_all_value_rule_lines` | 0/43 | +0.00000 | +0.00000 | +0.00000 | 0/43 |
| `k_correct_value` | 0/43 | +0.00000 | +0.00000 | +0.00000 | 0/43 |
| `q_random_same_norm` | 0/43 | +0.00000 | +0.00000 | +0.00000 | 0/43 |

### alpha_mass

| group | base | repair | q_only | q_random | repair-base | q-base | random-base |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct_value_token | +0.05860 | +0.158 | +0.160 | +0.05624 | +0.09897 | +0.102 | -0.00236 |
| correct_rule_line | +0.06576 | +0.177 | +0.181 | +0.06374 | +0.111 | +0.115 | -0.00202 |
| all_value_rule_lines | +0.309 | +0.395 | +0.406 | +0.305 | +0.08633 | +0.09741 | -0.00388 |
| wrong_same_relation_lines | +0.113 | +0.09707 | +0.100 | +0.114 | -0.01620 | -0.01321 | +0.00050 |
| wrong_same_category_lines | +0.05777 | +0.06793 | +0.06969 | +0.05535 | +0.01016 | +0.01192 | -0.00242 |
