# Phase 805 Residual Closure Blocker Audit (smoke)

- Status: `complete`
- Boundary: audits residual full-vocabulary blockers after semantic-blocker projection.
- It diagnoses why token closure still fails; it is not a new neuron-level closure proof.

## By Target Alpha And Semantic Beta

| model | target alpha | semantic beta | rows | cases | old suppress | new rate | sem suppress | sem still | residual blockers | required bias | sem share | format/echo share | anchor frag | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 0.000 | 1 | 1 | 1.242 | 0.374 | 0.000 | 1.000 | 171.000 | 13.312 | 0.497 | 0.409 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 1}` |
| qwen3 | 0.000 | 1.000 | 1 | 1 | 2.237 | 0.283 | 2.673 | 0.094 | 106.000 | 11.562 | 0.245 | 0.642 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 1}` |
| qwen3 | 0.750 | 0.000 | 1 | 1 | 1.143 | 0.121 | -0.174 | 0.031 | 33.000 | 12.562 | 0.152 | 0.576 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 1}` |
| qwen3 | 0.750 | 1.000 | 1 | 1 | 2.123 | 0.107 | 2.462 | 0.000 | 28.000 | 10.938 | 0.000 | 0.679 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 1}` |
| glm4 | 0.000 | 0.000 | 1 | 1 | 0.790 | 0.242 | 0.000 | 1.000 | 128.000 | 5.844 | 0.586 | 0.336 | 1.000 | 0.000 | `{"residual_semantic_still_dominant": 1}` |
| glm4 | 0.000 | 1.000 | 1 | 1 | 1.166 | 0.068 | 1.571 | 0.000 | 74.000 | 5.406 | 0.459 | 0.405 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 1}` |
| glm4 | 0.750 | 0.000 | 1 | 1 | 0.784 | 0.281 | -0.019 | 1.000 | 153.000 | 5.969 | 0.588 | 0.346 | 1.000 | 0.000 | `{"residual_semantic_still_dominant": 1}` |
| glm4 | 0.750 | 1.000 | 1 | 1 | 1.169 | 0.096 | 1.572 | 0.000 | 94.000 | 5.562 | 0.500 | 0.394 | 1.000 | 0.000 | `{"residual_semantic_still_dominant": 1}` |
| deepseek7b | 0.000 | 0.000 | 1 | 1 | -1.100 | 0.501 | 0.000 | 1.000 | 1892.000 | 11.016 | 0.621 | 0.344 | 1.000 | 0.000 | `{"residual_semantic_still_dominant": 1}` |
| deepseek7b | 0.000 | 1.000 | 1 | 1 | 0.007 | 0.217 | 2.695 | 0.500 | 825.000 | 11.438 | 0.472 | 0.467 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 1}` |
| deepseek7b | 0.750 | 0.000 | 1 | 1 | -1.139 | 0.160 | -0.030 | 1.000 | 619.000 | 10.094 | 0.460 | 0.472 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 1}` |
| deepseek7b | 0.750 | 1.000 | 1 | 1 | -0.019 | 0.053 | 2.665 | 0.031 | 321.000 | 10.344 | 0.302 | 0.623 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 1}` |

## Residual Class Counts

| model | target alpha | semantic beta | class | count |
|---|---:|---:|---|---:|
| qwen3 | 0.750 | 1.000 | `candidate_list_or_case_value` | 8 |
| qwen3 | 0.750 | 1.000 | `echo_token` | 6 |
| qwen3 | 0.750 | 1.000 | `high_frequency_or_format` | 6 |
| qwen3 | 0.750 | 1.000 | `punctuation` | 4 |
| qwen3 | 0.750 | 1.000 | `whitespace_or_newline` | 3 |
| qwen3 | 0.750 | 1.000 | `designated_contrast` | 1 |
| qwen3 | 0.750 | 0.000 | `candidate_list_or_case_value` | 8 |
| qwen3 | 0.750 | 0.000 | `echo_token` | 7 |
| qwen3 | 0.750 | 0.000 | `high_frequency_or_format` | 6 |
| qwen3 | 0.750 | 0.000 | `semantic_or_lexical_competitor` | 5 |
| qwen3 | 0.750 | 0.000 | `punctuation` | 4 |
| qwen3 | 0.750 | 0.000 | `whitespace_or_newline` | 2 |
| qwen3 | 0.750 | 0.000 | `designated_contrast` | 1 |
| qwen3 | 0.000 | 1.000 | `echo_token` | 31 |
| qwen3 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 26 |
| qwen3 | 0.000 | 1.000 | `high_frequency_or_format` | 19 |
| qwen3 | 0.000 | 1.000 | `punctuation` | 14 |
| qwen3 | 0.000 | 1.000 | `candidate_list_or_case_value` | 10 |
| qwen3 | 0.000 | 1.000 | `whitespace_or_newline` | 4 |
| qwen3 | 0.000 | 1.000 | `designated_contrast` | 1 |
| qwen3 | 0.000 | 1.000 | `other_token` | 1 |
| qwen3 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 85 |
| qwen3 | 0.000 | 0.000 | `echo_token` | 33 |
| qwen3 | 0.000 | 0.000 | `high_frequency_or_format` | 20 |
| qwen3 | 0.000 | 0.000 | `punctuation` | 13 |
| qwen3 | 0.000 | 0.000 | `candidate_list_or_case_value` | 12 |
| qwen3 | 0.000 | 0.000 | `whitespace_or_newline` | 4 |
| qwen3 | 0.000 | 0.000 | `other_token` | 3 |
| qwen3 | 0.000 | 0.000 | `designated_contrast` | 1 |
| glm4 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 34 |
| glm4 | 0.000 | 1.000 | `echo_token` | 13 |
| glm4 | 0.000 | 1.000 | `candidate_list_or_case_value` | 9 |
| glm4 | 0.000 | 1.000 | `high_frequency_or_format` | 9 |
| glm4 | 0.000 | 1.000 | `punctuation` | 7 |
| glm4 | 0.000 | 1.000 | `designated_contrast` | 1 |
| glm4 | 0.000 | 1.000 | `whitespace_or_newline` | 1 |
| glm4 | 0.750 | 1.000 | `semantic_or_lexical_competitor` | 47 |
| glm4 | 0.750 | 1.000 | `echo_token` | 17 |
| glm4 | 0.750 | 1.000 | `high_frequency_or_format` | 11 |
| glm4 | 0.750 | 1.000 | `candidate_list_or_case_value` | 9 |
| glm4 | 0.750 | 1.000 | `punctuation` | 8 |
| glm4 | 0.750 | 1.000 | `designated_contrast` | 1 |
| glm4 | 0.750 | 1.000 | `whitespace_or_newline` | 1 |
| glm4 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 75 |
| glm4 | 0.000 | 0.000 | `echo_token` | 20 |
| glm4 | 0.000 | 0.000 | `high_frequency_or_format` | 14 |
| glm4 | 0.000 | 0.000 | `candidate_list_or_case_value` | 9 |
| glm4 | 0.000 | 0.000 | `punctuation` | 8 |
| glm4 | 0.000 | 0.000 | `designated_contrast` | 1 |
| glm4 | 0.000 | 0.000 | `whitespace_or_newline` | 1 |
| glm4 | 0.750 | 0.000 | `semantic_or_lexical_competitor` | 90 |
| glm4 | 0.750 | 0.000 | `echo_token` | 26 |
| glm4 | 0.750 | 0.000 | `high_frequency_or_format` | 18 |
| glm4 | 0.750 | 0.000 | `candidate_list_or_case_value` | 9 |
| glm4 | 0.750 | 0.000 | `punctuation` | 8 |
| glm4 | 0.750 | 0.000 | `designated_contrast` | 1 |
| glm4 | 0.750 | 0.000 | `whitespace_or_newline` | 1 |
| deepseek7b | 0.750 | 1.000 | `semantic_or_lexical_competitor` | 97 |
| deepseek7b | 0.750 | 1.000 | `whitespace_or_newline` | 70 |
| deepseek7b | 0.750 | 1.000 | `echo_token` | 51 |
| deepseek7b | 0.750 | 1.000 | `punctuation` | 51 |
| deepseek7b | 0.750 | 1.000 | `high_frequency_or_format` | 23 |
| deepseek7b | 0.750 | 1.000 | `other_token` | 12 |
| deepseek7b | 0.750 | 1.000 | `candidate_list_or_case_value` | 11 |
| deepseek7b | 0.750 | 1.000 | `number_or_symbol` | 5 |
| deepseek7b | 0.750 | 1.000 | `designated_contrast` | 1 |
| deepseek7b | 0.750 | 0.000 | `semantic_or_lexical_competitor` | 285 |
| deepseek7b | 0.750 | 0.000 | `whitespace_or_newline` | 134 |
| deepseek7b | 0.750 | 0.000 | `punctuation` | 79 |
| deepseek7b | 0.750 | 0.000 | `echo_token` | 51 |
| deepseek7b | 0.750 | 0.000 | `other_token` | 28 |
| deepseek7b | 0.750 | 0.000 | `high_frequency_or_format` | 26 |
| deepseek7b | 0.750 | 0.000 | `candidate_list_or_case_value` | 13 |
| deepseek7b | 0.750 | 0.000 | `number_or_symbol` | 2 |
| deepseek7b | 0.750 | 0.000 | `designated_contrast` | 1 |
| deepseek7b | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 389 |
| deepseek7b | 0.000 | 1.000 | `whitespace_or_newline` | 140 |
| deepseek7b | 0.000 | 1.000 | `punctuation` | 115 |
| deepseek7b | 0.000 | 1.000 | `echo_token` | 82 |
| deepseek7b | 0.000 | 1.000 | `high_frequency_or_format` | 39 |
| deepseek7b | 0.000 | 1.000 | `other_token` | 36 |
| deepseek7b | 0.000 | 1.000 | `candidate_list_or_case_value` | 13 |
| deepseek7b | 0.000 | 1.000 | `number_or_symbol` | 9 |
| deepseek7b | 0.000 | 1.000 | `designated_contrast` | 1 |
| deepseek7b | 0.000 | 1.000 | `special_token` | 1 |
| deepseek7b | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 1174 |
| deepseek7b | 0.000 | 0.000 | `whitespace_or_newline` | 307 |
| deepseek7b | 0.000 | 0.000 | `punctuation` | 205 |
| deepseek7b | 0.000 | 0.000 | `echo_token` | 97 |
| deepseek7b | 0.000 | 0.000 | `other_token` | 53 |
| deepseek7b | 0.000 | 0.000 | `high_frequency_or_format` | 39 |
| deepseek7b | 0.000 | 0.000 | `candidate_list_or_case_value` | 13 |
| deepseek7b | 0.000 | 0.000 | `number_or_symbol` | 3 |
| deepseek7b | 0.000 | 0.000 | `designated_contrast` | 1 |
