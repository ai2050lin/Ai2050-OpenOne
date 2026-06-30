# Phase 805 Residual Closure Blocker Audit (main)

- Status: `complete`
- Boundary: audits residual full-vocabulary blockers after semantic-blocker projection.
- It diagnoses why token closure still fails; it is not a new neuron-level closure proof.

## By Target Alpha And Semantic Beta

| model | target alpha | semantic beta | rows | cases | old suppress | new rate | sem suppress | sem still | residual blockers | required bias | sem share | format/echo share | anchor frag | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 0.000 | 6 | 3 | 0.851 | 0.380 | 0.000 | 1.000 | 5102.833 | 16.792 | 0.760 | 0.199 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 1, "residual_semantic_still_dominant": 5}` |
| qwen3 | 0.000 | 1.000 | 6 | 3 | 2.485 | 0.172 | 3.142 | 0.346 | 1957.333 | 16.646 | 0.598 | 0.343 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 2, "residual_semantic_still_dominant": 4}` |
| qwen3 | 0.750 | 0.000 | 6 | 3 | 0.854 | 0.065 | -0.042 | 0.379 | 1081.333 | 15.047 | 0.591 | 0.326 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 1, "residual_identity_anchor_fragmented": 1, "residual_semantic_still_dominant": 4}` |
| qwen3 | 0.750 | 1.000 | 6 | 3 | 2.493 | 0.031 | 3.095 | 0.017 | 535.500 | 14.792 | 0.414 | 0.475 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 2, "residual_identity_anchor_fragmented": 1, "residual_semantic_still_dominant": 3}` |
| glm4 | 0.000 | 0.000 | 6 | 3 | 0.470 | 0.202 | 0.000 | 1.000 | 225.167 | 7.424 | 0.567 | 0.360 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 3, "residual_semantic_still_dominant": 3}` |
| glm4 | 0.000 | 1.000 | 6 | 3 | 0.614 | 0.152 | 0.804 | 0.318 | 215.333 | 7.258 | 0.536 | 0.381 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 2, "residual_identity_anchor_fragmented": 2, "residual_semantic_still_dominant": 2}` |
| glm4 | 0.750 | 0.000 | 6 | 3 | 0.471 | 0.103 | 0.001 | 0.487 | 136.000 | 6.812 | 0.510 | 0.350 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 3, "residual_semantic_still_dominant": 3}` |
| glm4 | 0.750 | 1.000 | 6 | 3 | 0.616 | 0.070 | 0.808 | 0.223 | 131.167 | 6.648 | 0.487 | 0.364 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 3, "residual_semantic_still_dominant": 3}` |
| deepseek7b | 0.000 | 0.000 | 6 | 3 | -0.439 | 0.429 | 0.000 | 1.000 | 14493.667 | 11.062 | 0.598 | 0.323 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 1, "residual_identity_anchor_fragmented": 1, "residual_semantic_still_dominant": 4}` |
| deepseek7b | 0.000 | 1.000 | 6 | 3 | 1.035 | 0.205 | 2.956 | 0.359 | 2222.167 | 13.305 | 0.479 | 0.417 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 2, "residual_identity_anchor_fragmented": 2, "residual_semantic_still_dominant": 2}` |
| deepseek7b | 0.750 | 0.000 | 6 | 3 | -0.546 | 0.136 | -0.083 | 0.618 | 2183.167 | 9.264 | 0.468 | 0.399 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 1, "residual_identity_anchor_fragmented": 2, "residual_semantic_still_dominant": 3}` |
| deepseek7b | 0.750 | 1.000 | 6 | 3 | 0.926 | 0.096 | 2.871 | 0.181 | 601.500 | 11.152 | 0.324 | 0.531 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 5, "residual_semantic_still_dominant": 1}` |

## Residual Class Counts

| model | target alpha | semantic beta | class | count |
|---|---:|---:|---|---:|
| qwen3 | 0.750 | 1.000 | `semantic_or_lexical_competitor` | 1931 |
| qwen3 | 0.750 | 1.000 | `echo_token` | 439 |
| qwen3 | 0.750 | 1.000 | `punctuation` | 327 |
| qwen3 | 0.750 | 1.000 | `whitespace_or_newline` | 214 |
| qwen3 | 0.750 | 1.000 | `high_frequency_or_format` | 125 |
| qwen3 | 0.750 | 1.000 | `other_token` | 95 |
| qwen3 | 0.750 | 1.000 | `candidate_list_or_case_value` | 55 |
| qwen3 | 0.750 | 1.000 | `number_or_symbol` | 21 |
| qwen3 | 0.750 | 1.000 | `designated_contrast` | 6 |
| qwen3 | 0.750 | 0.000 | `semantic_or_lexical_competitor` | 4925 |
| qwen3 | 0.750 | 0.000 | `echo_token` | 488 |
| qwen3 | 0.750 | 0.000 | `punctuation` | 378 |
| qwen3 | 0.750 | 0.000 | `whitespace_or_newline` | 354 |
| qwen3 | 0.750 | 0.000 | `high_frequency_or_format` | 134 |
| qwen3 | 0.750 | 0.000 | `other_token` | 132 |
| qwen3 | 0.750 | 0.000 | `candidate_list_or_case_value` | 65 |
| qwen3 | 0.750 | 0.000 | `designated_contrast` | 6 |
| qwen3 | 0.750 | 0.000 | `number_or_symbol` | 6 |
| qwen3 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 8886 |
| qwen3 | 0.000 | 1.000 | `echo_token` | 814 |
| qwen3 | 0.000 | 1.000 | `whitespace_or_newline` | 699 |
| qwen3 | 0.000 | 1.000 | `punctuation` | 694 |
| qwen3 | 0.000 | 1.000 | `other_token` | 346 |
| qwen3 | 0.000 | 1.000 | `high_frequency_or_format` | 195 |
| qwen3 | 0.000 | 1.000 | `candidate_list_or_case_value` | 72 |
| qwen3 | 0.000 | 1.000 | `number_or_symbol` | 32 |
| qwen3 | 0.000 | 1.000 | `designated_contrast` | 6 |
| qwen3 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 22304 |
| qwen3 | 0.000 | 0.000 | `whitespace_or_newline` | 1149 |
| qwen3 | 0.000 | 0.000 | `punctuation` | 980 |
| qwen3 | 0.000 | 0.000 | `echo_token` | 940 |
| qwen3 | 0.000 | 0.000 | `other_token` | 550 |
| qwen3 | 0.000 | 0.000 | `high_frequency_or_format` | 208 |
| qwen3 | 0.000 | 0.000 | `candidate_list_or_case_value` | 87 |
| qwen3 | 0.000 | 0.000 | `number_or_symbol` | 22 |
| qwen3 | 0.000 | 0.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 1.000 | `semantic_or_lexical_competitor` | 489 |
| glm4 | 0.750 | 1.000 | `echo_token` | 119 |
| glm4 | 0.750 | 1.000 | `punctuation` | 60 |
| glm4 | 0.750 | 1.000 | `candidate_list_or_case_value` | 54 |
| glm4 | 0.750 | 1.000 | `high_frequency_or_format` | 50 |
| glm4 | 0.750 | 1.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 1.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 1.000 | `other_token` | 2 |
| glm4 | 0.750 | 0.000 | `semantic_or_lexical_competitor` | 512 |
| glm4 | 0.750 | 0.000 | `echo_token` | 122 |
| glm4 | 0.750 | 0.000 | `high_frequency_or_format` | 59 |
| glm4 | 0.750 | 0.000 | `punctuation` | 55 |
| glm4 | 0.750 | 0.000 | `candidate_list_or_case_value` | 53 |
| glm4 | 0.750 | 0.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 0.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 0.000 | `other_token` | 2 |
| glm4 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 822 |
| glm4 | 0.000 | 1.000 | `echo_token` | 199 |
| glm4 | 0.000 | 1.000 | `punctuation` | 105 |
| glm4 | 0.000 | 1.000 | `high_frequency_or_format` | 87 |
| glm4 | 0.000 | 1.000 | `candidate_list_or_case_value` | 57 |
| glm4 | 0.000 | 1.000 | `whitespace_or_newline` | 9 |
| glm4 | 0.000 | 1.000 | `other_token` | 7 |
| glm4 | 0.000 | 1.000 | `designated_contrast` | 6 |
| glm4 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 867 |
| glm4 | 0.000 | 0.000 | `echo_token` | 206 |
| glm4 | 0.000 | 0.000 | `punctuation` | 105 |
| glm4 | 0.000 | 0.000 | `high_frequency_or_format` | 95 |
| glm4 | 0.000 | 0.000 | `candidate_list_or_case_value` | 57 |
| glm4 | 0.000 | 0.000 | `whitespace_or_newline` | 8 |
| glm4 | 0.000 | 0.000 | `other_token` | 7 |
| glm4 | 0.000 | 0.000 | `designated_contrast` | 6 |
| deepseek7b | 0.750 | 1.000 | `semantic_or_lexical_competitor` | 2239 |
| deepseek7b | 0.750 | 1.000 | `punctuation` | 367 |
| deepseek7b | 0.750 | 1.000 | `whitespace_or_newline` | 367 |
| deepseek7b | 0.750 | 1.000 | `echo_token` | 344 |
| deepseek7b | 0.750 | 1.000 | `high_frequency_or_format` | 116 |
| deepseek7b | 0.750 | 1.000 | `other_token` | 82 |
| deepseek7b | 0.750 | 1.000 | `candidate_list_or_case_value` | 56 |
| deepseek7b | 0.750 | 1.000 | `number_or_symbol` | 29 |
| deepseek7b | 0.750 | 1.000 | `designated_contrast` | 6 |
| deepseek7b | 0.750 | 1.000 | `special_token` | 3 |
| deepseek7b | 0.750 | 0.000 | `semantic_or_lexical_competitor` | 7795 |
| deepseek7b | 0.750 | 0.000 | `whitespace_or_newline` | 768 |
| deepseek7b | 0.750 | 0.000 | `punctuation` | 619 |
| deepseek7b | 0.750 | 0.000 | `echo_token` | 350 |
| deepseek7b | 0.750 | 0.000 | `other_token` | 184 |
| deepseek7b | 0.750 | 0.000 | `high_frequency_or_format` | 110 |
| deepseek7b | 0.750 | 0.000 | `candidate_list_or_case_value` | 67 |
| deepseek7b | 0.750 | 0.000 | `number_or_symbol` | 8 |
| deepseek7b | 0.750 | 0.000 | `designated_contrast` | 6 |
| deepseek7b | 0.750 | 0.000 | `special_token` | 1 |
| deepseek7b | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 8247 |
| deepseek7b | 0.000 | 1.000 | `punctuation` | 779 |
| deepseek7b | 0.000 | 1.000 | `whitespace_or_newline` | 678 |
| deepseek7b | 0.000 | 1.000 | `echo_token` | 664 |
| deepseek7b | 0.000 | 1.000 | `other_token` | 274 |
| deepseek7b | 0.000 | 1.000 | `high_frequency_or_format` | 194 |
| deepseek7b | 0.000 | 1.000 | `candidate_list_or_case_value` | 68 |
| deepseek7b | 0.000 | 1.000 | `number_or_symbol` | 39 |
| deepseek7b | 0.000 | 1.000 | `designated_contrast` | 6 |
| deepseek7b | 0.000 | 1.000 | `special_token` | 4 |
| deepseek7b | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 13588 |
| deepseek7b | 0.000 | 0.000 | `whitespace_or_newline` | 1437 |
| deepseek7b | 0.000 | 0.000 | `punctuation` | 1213 |
| deepseek7b | 0.000 | 0.000 | `echo_token` | 583 |
| deepseek7b | 0.000 | 0.000 | `other_token` | 360 |
| deepseek7b | 0.000 | 0.000 | `high_frequency_or_format` | 183 |
| deepseek7b | 0.000 | 0.000 | `candidate_list_or_case_value` | 74 |
| deepseek7b | 0.000 | 0.000 | `number_or_symbol` | 17 |
| deepseek7b | 0.000 | 0.000 | `designated_contrast` | 6 |
| deepseek7b | 0.000 | 0.000 | `special_token` | 1 |
