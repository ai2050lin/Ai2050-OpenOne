# Phase 797 Blocker-Class Targeted Suppression and Identity-Anchor Separation (smoke)

- Source: Phase 796 saved top-k rows.
- Boundary: this is an oracle audit, not a new neural intervention.

## By Model

| model | rows | cases | target rank | global delta | exact top-k cover | identity fragmented | exact closure by class set | unobserved risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 20 | 1 | 83.2500 | 3.3188 | 0.1000 | 1.0000 | 0.1000 | 0.9000 |
| deepseek7b | 20 | 1 | 545.6500 | 1.5633 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| glm4 | 12 | 1 | 109.5833 | 1.1797 | 0.3333 | 1.0000 | 0.0000 | 0.6667 |

## Single Class Suppression Audit

| model | blocker class | rows | mean count | required bias | observed top-k close | exact close |
|---|---|---:|---:|---:|---:|---:|
| qwen3 | `candidate_list_or_case_value` | 20 | 6.7500 | 9.6625 | 0.0000 | 0.0000 |
| deepseek7b | `candidate_list_or_case_value` | 20 | 5.8000 | 9.9992 | 0.0000 | 0.0000 |
| glm4 | `candidate_list_or_case_value` | 12 | 5.2500 | 5.5286 | 0.0000 | 0.0000 |
| glm4 | `semantic_or_lexical_competitor` | 12 | 4.8333 | 3.3932 | 0.0000 | 0.0000 |
| deepseek7b | `whitespace_or_newline` | 20 | 3.8000 | 7.0898 | 0.0000 | 0.0000 |
| qwen3 | `echo_token` | 20 | 3.6000 | 4.7569 | 0.0000 | 0.0000 |
| glm4 | `high_frequency_or_format` | 12 | 3.2500 | 3.8411 | 0.0000 | 0.0000 |
| deepseek7b | `semantic_or_lexical_competitor` | 20 | 3.1500 | 6.2898 | 0.0000 | 0.0000 |
| deepseek7b | `echo_token` | 20 | 2.7000 | 6.1336 | 0.0000 | 0.0000 |
| qwen3 | `high_frequency_or_format` | 20 | 2.6500 | 5.5000 | 0.0000 | 0.0000 |
| glm4 | `echo_token` | 12 | 2.5833 | 3.0443 | 0.0000 | 0.0000 |
| deepseek7b | `punctuation` | 20 | 2.4000 | 7.9836 | 0.0000 | 0.0000 |
| qwen3 | `semantic_or_lexical_competitor` | 20 | 2.0500 | 4.0486 | 0.0000 | 0.0000 |
| qwen3 | `punctuation` | 20 | 1.5000 | 5.6597 | 0.0000 | 0.0000 |
| deepseek7b | `high_frequency_or_format` | 20 | 1.1000 | 6.3086 | 0.0000 | 0.0000 |
| qwen3 | `designated_contrast` | 20 | 1.0000 | 13.1813 | 0.0000 | 0.0000 |
| glm4 | `designated_contrast` | 12 | 1.0000 | 6.1641 | 0.0000 | 0.0000 |
| deepseek7b | `designated_contrast` | 20 | 1.0000 | 10.4586 | 0.0000 | 0.0000 |
| qwen3 | `whitespace_or_newline` | 20 | 0.9000 | 4.7014 | 0.0000 | 0.0000 |
| glm4 | `punctuation` | 12 | 0.8333 | 1.8021 | 0.0000 | 0.0000 |
| glm4 | `whitespace_or_newline` | 12 | 0.4167 | 2.5750 | 0.0000 | 0.0000 |

## Identity Fragmentation Hotspots

| model | ladder | source group | rows | fragmented | surface variants | global delta | minimal sets |
|---|---|---|---:|---:|---:|---:|---|
| qwen3 | `route_answer` | `all_pre_answer` | 4 | 1.0000 | 3.0000 | 5.0312 | `{'not_found': 4}` |
| qwen3 | `kv_o_route` | `all_pre_answer` | 4 | 1.0000 | 3.0000 | 4.5938 | `{'not_found': 4}` |
| qwen3 | `kv_o` | `all_pre_answer` | 4 | 1.0000 | 3.5000 | 3.0781 | `{'not_found': 3, 'candidate_list_or_case_value+designated_contrast': 1}` |
| qwen3 | `kv_source` | `all_pre_answer` | 4 | 1.0000 | 3.7500 | 2.8438 | `{'not_found': 3, 'candidate_list_or_case_value+designated_contrast': 1}` |
| deepseek7b | `kv_o_route` | `all_pre_answer` | 4 | 1.0000 | 2.7500 | 2.2500 | `{'not_found': 4}` |
| deepseek7b | `route_answer` | `all_pre_answer` | 4 | 1.0000 | 3.0000 | 2.0312 | `{'not_found': 4}` |
| deepseek7b | `kv_o` | `all_pre_answer` | 4 | 1.0000 | 2.5000 | 1.6250 | `{'not_found': 4}` |
| glm4 | `route_answer` | `all_pre_answer` | 4 | 1.0000 | 3.5000 | 1.4062 | `{'not_found': 4}` |
| glm4 | `kv_o_route` | `all_pre_answer` | 2 | 1.0000 | 4.0000 | 1.3438 | `{'not_found': 2}` |
| deepseek7b | `kv_source` | `all_pre_answer` | 4 | 1.0000 | 2.5000 | 1.2891 | `{'not_found': 4}` |
| glm4 | `kv_source` | `all_pre_answer` | 2 | 1.0000 | 3.0000 | 1.2188 | `{'not_found': 2}` |
| glm4 | `kv_o` | `all_pre_answer` | 2 | 1.0000 | 3.0000 | 1.1719 | `{'not_found': 2}` |
| qwen3 | `o_only` | `all_pre_answer` | 4 | 1.0000 | 3.7500 | 1.0469 | `{'not_found': 4}` |
| deepseek7b | `o_only` | `all_pre_answer` | 4 | 1.0000 | 2.5000 | 0.6211 | `{'not_found': 4}` |
| glm4 | `o_only` | `all_pre_answer` | 2 | 1.0000 | 2.5000 | 0.5312 | `{'not_found': 2}` |
