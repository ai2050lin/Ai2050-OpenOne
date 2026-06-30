# Phase 798 Full-Vocabulary Blocker and Identity-Anchor Audit (confirm)

- Status: `complete`
- Boundary: reruns Phase 796 intervention paths and extracts full-vocabulary blockers above the target token.
- This is still a logit-space audit; internal suppressor localization remains a later phase.

## By Model

| model | rows | cases | target rank | full blockers | hidden outside saved top-k | single-class close | identity fragmented | token gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 192 | 6 | 361.188 | 360.188 | 311.094 | 0.000 | 1.000 | 0.000 |
| glm4 | 120 | 6 | 120.633 | 119.633 | 63.567 | 0.000 | 1.000 | 0.000 |
| deepseek7b | 192 | 6 | 1927.422 | 1926.422 | 1864.255 | 0.000 | 1.000 | 0.000 |

## Full-Vocabulary Blocker Class Counts

| model | class | count |
|---|---|---:|
| qwen3 | `semantic_or_lexical_competitor` | 48083 |
| qwen3 | `echo_token` | 6949 |
| qwen3 | `punctuation` | 4892 |
| qwen3 | `whitespace_or_newline` | 4069 |
| qwen3 | `high_frequency_or_format` | 2196 |
| qwen3 | `candidate_list_or_case_value` | 1408 |
| qwen3 | `other_token` | 1319 |
| qwen3 | `designated_contrast` | 192 |
| qwen3 | `number_or_symbol` | 48 |
| glm4 | `semantic_or_lexical_competitor` | 8908 |
| glm4 | `echo_token` | 2024 |
| glm4 | `punctuation` | 1192 |
| glm4 | `candidate_list_or_case_value` | 1024 |
| glm4 | `high_frequency_or_format` | 908 |
| glm4 | `designated_contrast` | 120 |
| glm4 | `whitespace_or_newline` | 92 |
| glm4 | `other_token` | 88 |
| deepseek7b | `semantic_or_lexical_competitor` | 286869 |
| deepseek7b | `whitespace_or_newline` | 31676 |
| deepseek7b | `punctuation` | 25924 |
| deepseek7b | `echo_token` | 11745 |
| deepseek7b | `other_token` | 7270 |
| deepseek7b | `high_frequency_or_format` | 3845 |
| deepseek7b | `candidate_list_or_case_value` | 2145 |
| deepseek7b | `designated_contrast` | 192 |
| deepseek7b | `number_or_symbol` | 188 |
| deepseek7b | `special_token` | 19 |

## Top Full-Vocab Effects

| model | selection | ladder | subspace | source group | rows | full blockers | delta global | target gain | identity fragmented | single-class close |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `matched` | `kv_o_route` | `negative` | `instruction` | 12 | 361.167 | 2.565 | 2.857 | 1.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `negative` | `instruction` | 12 | 360.583 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `positive` | `instruction` | 12 | 360.667 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `positive` | `all_pre_answer` | 12 | 360.667 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `positive` | `instruction` | 12 | 360.667 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 12 | 360.667 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `negative` | `instruction` | 12 | 360.667 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `negative` | `all_pre_answer` | 12 | 360.667 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `negative` | `instruction` | 12 | 360.667 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 12 | 360.667 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `positive` | `instruction` | 12 | 361.083 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `instruction` | 12 | 361.167 | 2.560 | 2.852 | 1.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 12 | 348.417 | 2.424 | 2.977 | 1.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 12 | 364.417 | 2.438 | 2.958 | 1.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 12 | 360.333 | 2.383 | 3.049 | 1.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 12 | 360.500 | 2.383 | 3.029 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `instruction` | 6 | 254.500 | 0.647 | 0.397 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `all_pre_answer` | 6 | 254.500 | 0.647 | 0.397 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `instruction` | 6 | 254.500 | 0.647 | 0.397 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `all_pre_answer` | 6 | 254.500 | 0.647 | 0.397 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `instruction` | 6 | 84.333 | 0.030 | 1.895 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `all_pre_answer` | 6 | 84.333 | 0.030 | 1.895 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `instruction` | 6 | 84.333 | 0.030 | 1.895 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 6 | 84.333 | 0.030 | 1.895 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `negative` | `instruction` | 6 | 84.333 | 0.030 | 1.895 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `negative` | `all_pre_answer` | 6 | 84.333 | 0.030 | 1.895 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `negative` | `instruction` | 6 | 84.333 | 0.030 | 1.895 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 6 | 84.333 | 0.030 | 1.895 | 1.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `instruction` | 6 | 85.167 | 0.025 | 1.889 | 1.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `instruction` | 6 | 85.167 | 0.025 | 1.889 | 1.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `negative` | `instruction` | 6 | 85.167 | 0.025 | 1.889 | 1.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `negative` | `instruction` | 6 | 85.167 | 0.025 | 1.889 | 1.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 89.833 | 0.004 | 1.889 | 1.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 89.833 | 0.004 | 1.889 | 1.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 89.833 | 0.004 | 1.889 | 1.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 89.833 | 0.004 | 1.889 | 1.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `instruction` | 12 | 1878.750 | 2.436 | 3.004 | 1.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `negative` | `instruction` | 12 | 1894.000 | 2.452 | 3.035 | 1.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `negative` | `instruction` | 12 | 1885.333 | 2.439 | 2.975 | 1.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `instruction` | 12 | 1895.750 | 2.439 | 2.980 | 1.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `instruction` | 12 | 1899.250 | 2.427 | 3.016 | 1.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `all_pre_answer` | 12 | 1899.250 | 2.427 | 3.016 | 1.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `instruction` | 12 | 1899.250 | 2.427 | 3.016 | 1.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `all_pre_answer` | 12 | 1899.250 | 2.427 | 3.016 | 1.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `negative` | `instruction` | 12 | 1899.250 | 2.427 | 3.016 | 1.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `negative` | `all_pre_answer` | 12 | 1899.250 | 2.427 | 3.016 | 1.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `negative` | `instruction` | 12 | 1899.250 | 2.427 | 3.016 | 1.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `negative` | `all_pre_answer` | 12 | 1899.250 | 2.427 | 3.016 | 1.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 12 | 1955.750 | 2.400 | 2.843 | 1.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 12 | 2018.250 | 2.436 | 2.883 | 1.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 12 | 2076.000 | 2.455 | 2.856 | 1.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 12 | 2024.917 | 2.344 | 2.824 | 1.000 | 0.000 |
