# Phase 798 Full-Vocabulary Blocker and Identity-Anchor Audit (main)

- Status: `complete`
- Boundary: reruns Phase 796 intervention paths and extracts full-vocabulary blockers above the target token.
- This is still a logit-space audit; internal suppressor localization remains a later phase.

## By Model

| model | rows | cases | target rank | full blockers | hidden outside saved top-k | single-class close | identity fragmented | token gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 48 | 3 | 691.042 | 690.042 | 633.583 | 0.000 | 1.000 | 0.000 |
| glm4 | 30 | 3 | 82.467 | 81.467 | 36.933 | 0.000 | 1.000 | 0.000 |
| deepseek7b | 48 | 3 | 2549.917 | 2548.917 | 2495.521 | 0.000 | 1.000 | 0.000 |

## Full-Vocabulary Blocker Class Counts

| model | class | count |
|---|---|---:|
| qwen3 | `semantic_or_lexical_competitor` | 23716 |
| qwen3 | `echo_token` | 3082 |
| qwen3 | `punctuation` | 2277 |
| qwen3 | `whitespace_or_newline` | 1921 |
| qwen3 | `high_frequency_or_format` | 933 |
| qwen3 | `other_token` | 649 |
| qwen3 | `candidate_list_or_case_value` | 472 |
| qwen3 | `designated_contrast` | 48 |
| qwen3 | `number_or_symbol` | 24 |
| glm4 | `semantic_or_lexical_competitor` | 1416 |
| glm4 | `echo_token` | 370 |
| glm4 | `candidate_list_or_case_value` | 244 |
| glm4 | `punctuation` | 192 |
| glm4 | `high_frequency_or_format` | 164 |
| glm4 | `designated_contrast` | 30 |
| glm4 | `whitespace_or_newline` | 22 |
| glm4 | `other_token` | 6 |
| deepseek7b | `semantic_or_lexical_competitor` | 103949 |
| deepseek7b | `whitespace_or_newline` | 6591 |
| deepseek7b | `punctuation` | 5548 |
| deepseek7b | `echo_token` | 2900 |
| deepseek7b | `other_token` | 1919 |
| deepseek7b | `high_frequency_or_format` | 806 |
| deepseek7b | `candidate_list_or_case_value` | 493 |
| deepseek7b | `number_or_symbol` | 86 |
| deepseek7b | `designated_contrast` | 48 |
| deepseek7b | `special_token` | 8 |

## Top Full-Vocab Effects

| model | selection | ladder | subspace | source group | rows | full blockers | delta global | target gain | identity fragmented | single-class close |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `top` | `route_answer` | `positive` | `all_pre_answer` | 6 | 690.333 | 2.568 | 3.130 | 1.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 6 | 690.333 | 2.568 | 3.130 | 1.000 | 0.000 |
| qwen3 | `top` | `route_answer` | `negative` | `all_pre_answer` | 6 | 690.333 | 2.568 | 3.130 | 1.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 6 | 690.333 | 2.568 | 3.130 | 1.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 657.333 | 2.432 | 3.203 | 1.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 682.833 | 2.453 | 3.161 | 1.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 691.167 | 2.375 | 3.188 | 1.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 727.667 | 2.318 | 3.151 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `all_pre_answer` | 3 | 188.667 | 0.594 | 0.198 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `all_pre_answer` | 3 | 188.667 | 0.594 | 0.198 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `positive` | `all_pre_answer` | 3 | 53.333 | 0.141 | 1.911 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 3 | 53.333 | 0.141 | 1.911 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `negative` | `all_pre_answer` | 3 | 53.333 | 0.141 | 1.911 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `negative` | `all_pre_answer` | 3 | 53.333 | 0.141 | 1.911 | 1.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 3 | 56.000 | 0.130 | 1.922 | 1.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 3 | 56.000 | 0.130 | 1.922 | 1.000 | 0.000 |
| glm4 | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 3 | 56.000 | 0.130 | 1.922 | 1.000 | 0.000 |
| glm4 | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 3 | 56.000 | 0.130 | 1.922 | 1.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 2536.500 | 3.094 | 3.302 | 1.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `all_pre_answer` | 6 | 2539.500 | 3.076 | 3.315 | 1.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `all_pre_answer` | 6 | 2539.500 | 3.076 | 3.315 | 1.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `negative` | `all_pre_answer` | 6 | 2539.500 | 3.076 | 3.315 | 1.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `negative` | `all_pre_answer` | 6 | 2539.500 | 3.076 | 3.315 | 1.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 2529.500 | 2.986 | 3.204 | 1.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `negative` | `all_pre_answer` | 6 | 2529.333 | 2.975 | 3.257 | 1.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 6 | 2638.000 | 3.009 | 3.145 | 1.000 | 0.000 |
