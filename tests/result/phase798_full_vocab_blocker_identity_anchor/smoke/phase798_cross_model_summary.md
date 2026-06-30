# Phase 798 Full-Vocabulary Blocker and Identity-Anchor Audit (smoke)

- Status: `complete`
- Boundary: reruns Phase 796 intervention paths and extracts full-vocabulary blockers above the target token.
- This is still a logit-space audit; internal suppressor localization remains a later phase.

## By Model

| model | rows | cases | target rank | full blockers | hidden outside saved top-k | single-class close | identity fragmented | token gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 4 | 1 | 32.000 | 31.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| glm4 | 2 | 1 | 194.000 | 193.000 | 153.000 | 0.000 | 1.000 | 0.000 |
| deepseek7b | 4 | 1 | 410.000 | 409.000 | 369.000 | 0.000 | 1.000 | 0.000 |

## Full-Vocabulary Blocker Class Counts

| model | class | count |
|---|---|---:|
| qwen3 | `candidate_list_or_case_value` | 32 |
| qwen3 | `echo_token` | 24 |
| qwen3 | `high_frequency_or_format` | 20 |
| qwen3 | `semantic_or_lexical_competitor` | 20 |
| qwen3 | `punctuation` | 16 |
| qwen3 | `whitespace_or_newline` | 8 |
| qwen3 | `designated_contrast` | 4 |
| glm4 | `semantic_or_lexical_competitor` | 236 |
| glm4 | `echo_token` | 62 |
| glm4 | `high_frequency_or_format` | 44 |
| glm4 | `punctuation` | 20 |
| glm4 | `candidate_list_or_case_value` | 18 |
| glm4 | `designated_contrast` | 2 |
| glm4 | `other_token` | 2 |
| glm4 | `whitespace_or_newline` | 2 |
| deepseek7b | `semantic_or_lexical_competitor` | 666 |
| deepseek7b | `whitespace_or_newline` | 360 |
| deepseek7b | `punctuation` | 211 |
| deepseek7b | `echo_token` | 176 |
| deepseek7b | `high_frequency_or_format` | 87 |
| deepseek7b | `other_token` | 79 |
| deepseek7b | `candidate_list_or_case_value` | 50 |
| deepseek7b | `designated_contrast` | 4 |
| deepseek7b | `number_or_symbol` | 3 |

## Top Full-Vocab Effects

| model | selection | ladder | subspace | source group | rows | full blockers | delta global | target gain | identity fragmented | single-class close |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `top` | `route_answer` | `positive` | `all_pre_answer` | 1 | 31.000 | 4.312 | 2.938 | 1.000 | 0.000 |
| qwen3 | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 31.000 | 4.312 | 2.938 | 1.000 | 0.000 |
| qwen3 | `matched` | `route_answer` | `positive` | `all_pre_answer` | 1 | 31.000 | 4.312 | 2.938 | 1.000 | 0.000 |
| qwen3 | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 31.000 | 4.312 | 2.938 | 1.000 | 0.000 |
| glm4 | `top` | `route_answer` | `route` | `all_pre_answer` | 1 | 193.000 | 1.469 | -0.219 | 1.000 | 0.000 |
| glm4 | `matched` | `route_answer` | `route` | `all_pre_answer` | 1 | 193.000 | 1.469 | -0.219 | 1.000 | 0.000 |
| deepseek7b | `matched` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 402.000 | 3.109 | 3.172 | 1.000 | 0.000 |
| deepseek7b | `top` | `route_answer` | `positive` | `all_pre_answer` | 1 | 379.000 | 2.422 | 3.172 | 1.000 | 0.000 |
| deepseek7b | `matched` | `route_answer` | `positive` | `all_pre_answer` | 1 | 379.000 | 2.422 | 3.172 | 1.000 | 0.000 |
| deepseek7b | `top` | `kv_o_route` | `positive` | `all_pre_answer` | 1 | 476.000 | 2.609 | 2.859 | 1.000 | 0.000 |
