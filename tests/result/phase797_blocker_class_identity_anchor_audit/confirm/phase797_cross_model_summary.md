# Phase 797 Blocker-Class Targeted Suppression and Identity-Anchor Separation (confirm)

- Source: Phase 796 saved top-k rows.
- Boundary: this is an oracle audit, not a new neural intervention.

## By Model

| model | rows | cases | target rank | global delta | exact top-k cover | identity fragmented | exact closure by class set | unobserved risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 480 | 6 | 1101.4604 | 1.5679 | 0.2771 | 1.0000 | 0.0479 | 0.7229 |
| glm4 | 264 | 6 | 264.7121 | 0.2032 | 0.2045 | 1.0000 | 0.0000 | 0.7955 |
| deepseek7b | 480 | 6 | 2527.2375 | 1.5139 | 0.1104 | 0.8917 | 0.0333 | 0.8896 |

## Single Class Suppression Audit

| model | blocker class | rows | mean count | required bias | observed top-k close | exact close |
|---|---|---:|---:|---:|---:|---:|
| glm4 | `semantic_or_lexical_competitor` | 264 | 17.2235 | 4.2755 | 0.0000 | 0.0000 |
| qwen3 | `semantic_or_lexical_competitor` | 480 | 8.6479 | 6.4852 | 0.0000 | 0.0000 |
| deepseek7b | `semantic_or_lexical_competitor` | 480 | 8.2438 | 6.3389 | 0.0000 | 0.0000 |
| deepseek7b | `whitespace_or_newline` | 480 | 7.6375 | 6.9369 | 0.0000 | 0.0000 |
| qwen3 | `echo_token` | 480 | 7.4417 | 7.3105 | 0.0000 | 0.0000 |
| deepseek7b | `punctuation` | 480 | 6.9333 | 7.5254 | 0.0000 | 0.0000 |
| deepseek7b | `echo_token` | 480 | 6.8187 | 7.1128 | 0.0000 | 0.0000 |
| glm4 | `candidate_list_or_case_value` | 264 | 6.4015 | 6.2754 | 0.0000 | 0.0000 |
| glm4 | `echo_token` | 264 | 6.2765 | 4.0957 | 0.0000 | 0.0000 |
| qwen3 | `candidate_list_or_case_value` | 480 | 5.7917 | 11.1391 | 0.0000 | 0.0000 |
| deepseek7b | `candidate_list_or_case_value` | 480 | 5.4667 | 9.2735 | 0.0000 | 0.0000 |
| qwen3 | `punctuation` | 480 | 4.6083 | 7.5850 | 0.0000 | 0.0000 |
| qwen3 | `high_frequency_or_format` | 480 | 3.9250 | 5.9123 | 0.0000 | 0.0000 |
| qwen3 | `whitespace_or_newline` | 480 | 3.6542 | 7.6510 | 0.0000 | 0.0000 |
| glm4 | `punctuation` | 264 | 3.3182 | 3.5060 | 0.0000 | 0.0000 |
| glm4 | `high_frequency_or_format` | 264 | 1.8523 | 2.8196 | 0.0000 | 0.0000 |
| deepseek7b | `high_frequency_or_format` | 480 | 1.7021 | 6.5349 | 0.0000 | 0.0000 |
| qwen3 | `designated_contrast` | 480 | 1.0000 | 11.8227 | 0.0000 | 0.0000 |
| glm4 | `designated_contrast` | 264 | 1.0000 | 6.4598 | 0.0000 | 0.0000 |
| deepseek7b | `designated_contrast` | 480 | 0.8917 | 9.6516 | 0.0000 | 0.0000 |
| glm4 | `whitespace_or_newline` | 264 | 0.7273 | 2.2702 | 0.0000 | 0.0000 |

## Identity Fragmentation Hotspots

| model | ladder | source group | rows | fragmented | surface variants | global delta | minimal sets |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | `kv_o_route` | `instruction` | 48 | 1.0000 | 2.9167 | 2.5612 | `{'not_found': 44, 'candidate_list_or_case_value+designated_contrast+semantic_or_lexical_competitor': 4}` |
| qwen3 | `route_answer` | `instruction` | 48 | 1.0000 | 2.9167 | 2.5599 | `{'not_found': 44, 'candidate_list_or_case_value+designated_contrast+semantic_or_lexical_competitor': 4}` |
| qwen3 | `route_answer` | `all_pre_answer` | 48 | 1.0000 | 2.9167 | 2.5599 | `{'not_found': 44, 'candidate_list_or_case_value+designated_contrast+semantic_or_lexical_competitor': 4}` |
| qwen3 | `kv_o_route` | `all_pre_answer` | 48 | 1.0000 | 2.9167 | 2.4069 | `{'not_found': 44, 'candidate_list_or_case_value+designated_contrast+semantic_or_lexical_competitor': 4}` |
| qwen3 | `kv_o` | `all_pre_answer` | 48 | 1.0000 | 3.0833 | 1.7822 | `{'not_found': 44, 'candidate_list_or_case_value+designated_contrast': 4}` |
| qwen3 | `kv_source` | `all_pre_answer` | 48 | 1.0000 | 3.1875 | 1.0706 | `{'not_found': 45, 'candidate_list_or_case_value+designated_contrast': 3}` |
| qwen3 | `o_only` | `instruction` | 48 | 1.0000 | 3.2917 | 0.9531 | `{'not_found': 48}` |
| qwen3 | `o_only` | `all_pre_answer` | 48 | 1.0000 | 3.2917 | 0.9531 | `{'not_found': 48}` |
| qwen3 | `kv_o` | `instruction` | 48 | 1.0000 | 3.2917 | 0.9186 | `{'not_found': 48}` |
| glm4 | `kv_source` | `all_pre_answer` | 24 | 1.0000 | 3.0000 | 0.5482 | `{'not_found': 24}` |
| glm4 | `kv_o` | `all_pre_answer` | 24 | 1.0000 | 3.0000 | 0.4427 | `{'not_found': 24}` |
| glm4 | `route_answer` | `instruction` | 36 | 1.0000 | 3.1667 | 0.2357 | `{'not_found': 36}` |
| glm4 | `route_answer` | `all_pre_answer` | 36 | 1.0000 | 3.1667 | 0.2357 | `{'not_found': 36}` |
| glm4 | `kv_o` | `instruction` | 24 | 1.0000 | 3.0000 | 0.1820 | `{'not_found': 24}` |
| glm4 | `o_only` | `instruction` | 24 | 1.0000 | 3.0000 | 0.1761 | `{'not_found': 24}` |
| glm4 | `o_only` | `all_pre_answer` | 24 | 1.0000 | 3.0000 | 0.1761 | `{'not_found': 24}` |
| glm4 | `kv_o_route` | `instruction` | 24 | 1.0000 | 3.1667 | 0.0247 | `{'not_found': 24}` |
| glm4 | `kv_o_route` | `all_pre_answer` | 24 | 1.0000 | 3.1667 | 0.0039 | `{'not_found': 24}` |
| glm4 | `kv_source` | `instruction` | 24 | 1.0000 | 3.0000 | -0.0260 | `{'not_found': 24}` |
| qwen3 | `kv_source` | `instruction` | 48 | 1.0000 | 3.3333 | -0.0866 | `{'not_found': 48}` |
| deepseek7b | `kv_o_route` | `instruction` | 48 | 0.9167 | 2.6250 | 2.4414 | `{'not_found': 44, 'candidate_list_or_case_value+designated_contrast+echo_token+punctuation': 4}` |
| deepseek7b | `route_answer` | `instruction` | 48 | 0.9167 | 2.7500 | 2.4271 | `{'not_found': 44, 'candidate_list_or_case_value+designated_contrast+echo_token+punctuation': 4}` |
| deepseek7b | `route_answer` | `all_pre_answer` | 48 | 0.9167 | 2.7500 | 2.4271 | `{'not_found': 44, 'candidate_list_or_case_value+designated_contrast+echo_token+punctuation': 4}` |
| deepseek7b | `kv_o_route` | `all_pre_answer` | 48 | 0.9167 | 2.5000 | 2.4089 | `{'not_found': 44, 'candidate_list_or_case_value+designated_contrast+echo_token+punctuation': 4}` |
| deepseek7b | `kv_o` | `all_pre_answer` | 48 | 0.8958 | 2.3958 | 1.2227 | `{'not_found': 48}` |
| deepseek7b | `kv_source` | `all_pre_answer` | 48 | 0.8958 | 2.3542 | 1.0418 | `{'not_found': 48}` |
| deepseek7b | `kv_o` | `instruction` | 48 | 0.8750 | 2.3958 | 1.0833 | `{'not_found': 48}` |
| deepseek7b | `o_only` | `instruction` | 48 | 0.8750 | 2.4167 | 0.6385 | `{'not_found': 48}` |
| deepseek7b | `o_only` | `all_pre_answer` | 48 | 0.8750 | 2.4167 | 0.6385 | `{'not_found': 48}` |
| deepseek7b | `kv_source` | `instruction` | 48 | 0.8333 | 2.2292 | 0.8095 | `{'not_found': 48}` |
