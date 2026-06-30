# Phase 797 Blocker-Class Targeted Suppression and Identity-Anchor Separation (main)

- Source: Phase 796 saved top-k rows.
- Boundary: this is an oracle audit, not a new neural intervention.

## By Model

| model | rows | cases | target rank | global delta | exact top-k cover | identity fragmented | exact closure by class set | unobserved risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 160 | 4 | 1803.8187 | 1.6772 | 0.1125 | 1.0000 | 0.0125 | 0.8875 |
| glm4 | 88 | 4 | 184.9773 | 0.0753 | 0.0909 | 1.0000 | 0.0000 | 0.9091 |
| deepseek7b | 160 | 4 | 2219.1937 | 1.6697 | 0.1000 | 0.8438 | 0.0500 | 0.9000 |

## Single Class Suppression Audit

| model | blocker class | rows | mean count | required bias | observed top-k close | exact close |
|---|---|---:|---:|---:|---:|---:|
| glm4 | `semantic_or_lexical_competitor` | 88 | 14.0227 | 4.2702 | 0.0000 | 0.0000 |
| deepseek7b | `whitespace_or_newline` | 160 | 6.2562 | 5.8865 | 0.0000 | 0.0000 |
| qwen3 | `echo_token` | 160 | 6.2062 | 8.2588 | 0.0000 | 0.0000 |
| qwen3 | `semantic_or_lexical_competitor` | 160 | 5.7250 | 7.4083 | 0.0000 | 0.0000 |
| qwen3 | `candidate_list_or_case_value` | 160 | 5.6500 | 12.7798 | 0.0000 | 0.0000 |
| deepseek7b | `echo_token` | 160 | 5.5125 | 6.1047 | 0.0000 | 0.0000 |
| deepseek7b | `punctuation` | 160 | 5.3125 | 6.3207 | 0.0000 | 0.0000 |
| glm4 | `candidate_list_or_case_value` | 88 | 5.1818 | 6.1776 | 0.0000 | 0.0000 |
| deepseek7b | `candidate_list_or_case_value` | 160 | 4.7812 | 7.8981 | 0.0000 | 0.0000 |
| glm4 | `echo_token` | 88 | 4.5568 | 4.3061 | 0.0000 | 0.0000 |
| deepseek7b | `semantic_or_lexical_competitor` | 160 | 4.3312 | 5.1509 | 0.0000 | 0.0000 |
| qwen3 | `punctuation` | 160 | 3.8937 | 8.9574 | 0.0000 | 0.0000 |
| qwen3 | `whitespace_or_newline` | 160 | 3.4750 | 8.9857 | 0.0000 | 0.0000 |
| qwen3 | `high_frequency_or_format` | 160 | 2.8375 | 7.0987 | 0.0000 | 0.0000 |
| glm4 | `punctuation` | 88 | 1.8182 | 3.5357 | 0.0000 | 0.0000 |
| glm4 | `high_frequency_or_format` | 88 | 1.5000 | 2.7341 | 0.0000 | 0.0000 |
| deepseek7b | `high_frequency_or_format` | 160 | 1.1000 | 5.9032 | 0.0000 | 0.0000 |
| qwen3 | `designated_contrast` | 160 | 1.0000 | 13.1243 | 0.0000 | 0.0000 |
| glm4 | `designated_contrast` | 88 | 1.0000 | 6.2429 | 0.0000 | 0.0000 |
| deepseek7b | `designated_contrast` | 160 | 0.8438 | 8.2366 | 0.0000 | 0.0000 |
| glm4 | `whitespace_or_newline` | 88 | 0.4659 | 2.1143 | 0.0000 | 0.0000 |

## Identity Fragmentation Hotspots

| model | ladder | source group | rows | fragmented | surface variants | global delta | minimal sets |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | `route_answer` | `all_pre_answer` | 32 | 1.0000 | 2.8750 | 2.6289 | `{'not_found': 32}` |
| qwen3 | `kv_o_route` | `all_pre_answer` | 32 | 1.0000 | 2.8750 | 2.4775 | `{'not_found': 32}` |
| qwen3 | `kv_o` | `all_pre_answer` | 32 | 1.0000 | 3.0000 | 1.3047 | `{'not_found': 31, 'candidate_list_or_case_value+designated_contrast': 1}` |
| qwen3 | `o_only` | `all_pre_answer` | 32 | 1.0000 | 3.1875 | 1.1602 | `{'not_found': 32}` |
| qwen3 | `kv_source` | `all_pre_answer` | 32 | 1.0000 | 3.1250 | 0.8149 | `{'not_found': 31, 'candidate_list_or_case_value+designated_contrast': 1}` |
| glm4 | `kv_source` | `all_pre_answer` | 16 | 1.0000 | 2.5000 | 0.3398 | `{'not_found': 12, 'candidate_list_or_case_value+designated_contrast+echo_token+semantic_or_lexical_competitor': 4}` |
| glm4 | `kv_o` | `all_pre_answer` | 16 | 1.0000 | 2.5000 | 0.2129 | `{'not_found': 14, 'candidate_list_or_case_value+designated_contrast+echo_token+semantic_or_lexical_competitor': 2}` |
| glm4 | `o_only` | `all_pre_answer` | 16 | 1.0000 | 2.7500 | 0.1016 | `{'not_found': 12, 'candidate_list_or_case_value+designated_contrast+echo_token+semantic_or_lexical_competitor': 4}` |
| glm4 | `route_answer` | `all_pre_answer` | 24 | 1.0000 | 2.6667 | 0.0013 | `{'not_found': 24}` |
| glm4 | `kv_o_route` | `all_pre_answer` | 16 | 1.0000 | 2.7500 | -0.2422 | `{'not_found': 16}` |
| deepseek7b | `route_answer` | `all_pre_answer` | 32 | 0.8750 | 2.7500 | 2.6582 | `{'not_found': 28, 'candidate_list_or_case_value+designated_contrast+echo_token+punctuation': 4}` |
| deepseek7b | `kv_o_route` | `all_pre_answer` | 32 | 0.8750 | 2.5312 | 2.6174 | `{'not_found': 28, 'candidate_list_or_case_value+designated_contrast+echo_token+punctuation': 4}` |
| deepseek7b | `kv_source` | `all_pre_answer` | 32 | 0.8438 | 2.1562 | 1.1526 | `{'not_found': 32}` |
| deepseek7b | `kv_o` | `all_pre_answer` | 32 | 0.8125 | 2.1250 | 1.2615 | `{'not_found': 32}` |
| deepseek7b | `o_only` | `all_pre_answer` | 32 | 0.8125 | 2.3438 | 0.6588 | `{'not_found': 32}` |
