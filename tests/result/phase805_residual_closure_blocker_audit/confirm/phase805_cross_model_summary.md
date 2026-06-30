# Phase 805 Residual Closure Blocker Audit (confirm)

- Status: `complete`
- Boundary: audits residual full-vocabulary blockers after semantic-blocker projection.
- It diagnoses why token closure still fails; it is not a new neuron-level closure proof.

## By Target Alpha And Semantic Beta

| model | target alpha | semantic beta | rows | cases | old suppress | new rate | sem suppress | sem still | residual blockers | required bias | sem share | format/echo share | anchor frag | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 0.000 | 10 | 5 | 0.802 | 0.363 | 0.000 | 1.000 | 3103.600 | 14.225 | 0.638 | 0.299 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 1, "residual_identity_anchor_fragmented": 3, "residual_semantic_still_dominant": 6}` |
| qwen3 | 0.000 | 1.000 | 10 | 5 | 2.093 | 0.165 | 2.637 | 0.221 | 1144.000 | 13.909 | 0.463 | 0.449 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 6, "residual_semantic_still_dominant": 4}` |
| qwen3 | 0.750 | 0.000 | 10 | 5 | 0.764 | 0.068 | -0.073 | 0.259 | 663.100 | 12.859 | 0.479 | 0.378 | 1.000 | 0.000 | `{"near_closure_small_residual": 1, "residual_format_echo_dominant": 3, "residual_identity_anchor_fragmented": 2, "residual_semantic_still_dominant": 4}` |
| qwen3 | 0.750 | 1.000 | 10 | 5 | 2.059 | 0.036 | 2.552 | 0.009 | 318.300 | 12.434 | 0.322 | 0.510 | 1.000 | 0.000 | `{"near_closure_small_residual": 1, "residual_format_echo_dominant": 5, "residual_identity_anchor_fragmented": 1, "residual_semantic_still_dominant": 3}` |
| glm4 | 0.000 | 0.000 | 6 | 3 | 0.470 | 0.202 | 0.000 | 1.000 | 225.167 | 7.424 | 0.567 | 0.360 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 3, "residual_semantic_still_dominant": 3}` |
| glm4 | 0.000 | 1.000 | 6 | 3 | 0.612 | 0.151 | 0.795 | 0.318 | 214.500 | 7.253 | 0.535 | 0.381 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 2, "residual_identity_anchor_fragmented": 2, "residual_semantic_still_dominant": 2}` |
| glm4 | 0.750 | 0.000 | 6 | 3 | 0.471 | 0.103 | 0.001 | 0.454 | 136.000 | 6.812 | 0.510 | 0.350 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 3, "residual_semantic_still_dominant": 3}` |
| glm4 | 0.750 | 1.000 | 6 | 3 | 0.613 | 0.070 | 0.798 | 0.214 | 131.833 | 6.656 | 0.488 | 0.364 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 3, "residual_semantic_still_dominant": 3}` |
| deepseek7b | 0.000 | 0.000 | 4 | 2 | -0.741 | 0.545 | 0.000 | 1.000 | 21676.000 | 12.516 | 0.736 | 0.237 | 1.000 | 0.000 | `{"residual_semantic_still_dominant": 4}` |
| deepseek7b | 0.000 | 1.000 | 4 | 2 | 1.493 | 0.245 | 3.035 | 0.508 | 3007.000 | 16.371 | 0.599 | 0.354 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 2, "residual_semantic_still_dominant": 2}` |
| deepseek7b | 0.750 | 0.000 | 4 | 2 | -0.878 | 0.162 | -0.087 | 0.633 | 3255.000 | 10.193 | 0.583 | 0.365 | 1.000 | 0.000 | `{"residual_identity_anchor_fragmented": 1, "residual_semantic_still_dominant": 3}` |
| deepseek7b | 0.750 | 1.000 | 4 | 2 | 1.355 | 0.120 | 2.953 | 0.273 | 864.250 | 13.506 | 0.402 | 0.538 | 1.000 | 0.000 | `{"residual_format_echo_dominant": 3, "residual_semantic_still_dominant": 1}` |

## Residual Class Counts

| model | target alpha | semantic beta | class | count |
|---|---:|---:|---|---:|
| qwen3 | 0.750 | 1.000 | `semantic_or_lexical_competitor` | 1821 |
| qwen3 | 0.750 | 1.000 | `echo_token` | 460 |
| qwen3 | 0.750 | 1.000 | `punctuation` | 339 |
| qwen3 | 0.750 | 1.000 | `whitespace_or_newline` | 221 |
| qwen3 | 0.750 | 1.000 | `high_frequency_or_format` | 143 |
| qwen3 | 0.750 | 1.000 | `other_token` | 96 |
| qwen3 | 0.750 | 1.000 | `candidate_list_or_case_value` | 71 |
| qwen3 | 0.750 | 1.000 | `number_or_symbol` | 22 |
| qwen3 | 0.750 | 1.000 | `designated_contrast` | 10 |
| qwen3 | 0.750 | 0.000 | `semantic_or_lexical_competitor` | 4972 |
| qwen3 | 0.750 | 0.000 | `echo_token` | 517 |
| qwen3 | 0.750 | 0.000 | `punctuation` | 390 |
| qwen3 | 0.750 | 0.000 | `whitespace_or_newline` | 366 |
| qwen3 | 0.750 | 0.000 | `high_frequency_or_format` | 154 |
| qwen3 | 0.750 | 0.000 | `other_token` | 133 |
| qwen3 | 0.750 | 0.000 | `candidate_list_or_case_value` | 83 |
| qwen3 | 0.750 | 0.000 | `designated_contrast` | 10 |
| qwen3 | 0.750 | 0.000 | `number_or_symbol` | 6 |
| qwen3 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 8441 |
| qwen3 | 0.000 | 1.000 | `echo_token` | 868 |
| qwen3 | 0.000 | 1.000 | `punctuation` | 717 |
| qwen3 | 0.000 | 1.000 | `whitespace_or_newline` | 703 |
| qwen3 | 0.000 | 1.000 | `other_token` | 343 |
| qwen3 | 0.000 | 1.000 | `high_frequency_or_format` | 232 |
| qwen3 | 0.000 | 1.000 | `candidate_list_or_case_value` | 93 |
| qwen3 | 0.000 | 1.000 | `number_or_symbol` | 33 |
| qwen3 | 0.000 | 1.000 | `designated_contrast` | 10 |
| qwen3 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 22511 |
| qwen3 | 0.000 | 0.000 | `whitespace_or_newline` | 1170 |
| qwen3 | 0.000 | 0.000 | `echo_token` | 1015 |
| qwen3 | 0.000 | 0.000 | `punctuation` | 1014 |
| qwen3 | 0.000 | 0.000 | `other_token` | 556 |
| qwen3 | 0.000 | 0.000 | `high_frequency_or_format` | 255 |
| qwen3 | 0.000 | 0.000 | `candidate_list_or_case_value` | 112 |
| qwen3 | 0.000 | 0.000 | `number_or_symbol` | 22 |
| qwen3 | 0.000 | 0.000 | `designated_contrast` | 10 |
| glm4 | 0.750 | 1.000 | `semantic_or_lexical_competitor` | 493 |
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
| glm4 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 817 |
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
| deepseek7b | 0.750 | 1.000 | `semantic_or_lexical_competitor` | 2154 |
| deepseek7b | 0.750 | 1.000 | `punctuation` | 355 |
| deepseek7b | 0.750 | 1.000 | `whitespace_or_newline` | 345 |
| deepseek7b | 0.750 | 1.000 | `echo_token` | 337 |
| deepseek7b | 0.750 | 1.000 | `high_frequency_or_format` | 113 |
| deepseek7b | 0.750 | 1.000 | `other_token` | 77 |
| deepseek7b | 0.750 | 1.000 | `candidate_list_or_case_value` | 38 |
| deepseek7b | 0.750 | 1.000 | `number_or_symbol` | 31 |
| deepseek7b | 0.750 | 1.000 | `designated_contrast` | 4 |
| deepseek7b | 0.750 | 1.000 | `special_token` | 3 |
| deepseek7b | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 8178 |
| deepseek7b | 0.000 | 1.000 | `punctuation` | 744 |
| deepseek7b | 0.000 | 1.000 | `echo_token` | 644 |
| deepseek7b | 0.000 | 1.000 | `whitespace_or_newline` | 629 |
| deepseek7b | 0.000 | 1.000 | `other_token` | 266 |
| deepseek7b | 0.000 | 1.000 | `high_frequency_or_format` | 179 |
| deepseek7b | 0.000 | 1.000 | `candidate_list_or_case_value` | 44 |
| deepseek7b | 0.000 | 1.000 | `number_or_symbol` | 40 |
| deepseek7b | 0.000 | 1.000 | `designated_contrast` | 4 |
| deepseek7b | 0.000 | 1.000 | `special_token` | 4 |
| deepseek7b | 0.750 | 0.000 | `semantic_or_lexical_competitor` | 7776 |
| deepseek7b | 0.750 | 0.000 | `whitespace_or_newline` | 755 |
| deepseek7b | 0.750 | 0.000 | `punctuation` | 610 |
| deepseek7b | 0.750 | 0.000 | `echo_token` | 338 |
| deepseek7b | 0.750 | 0.000 | `other_token` | 182 |
| deepseek7b | 0.750 | 0.000 | `high_frequency_or_format` | 107 |
| deepseek7b | 0.750 | 0.000 | `candidate_list_or_case_value` | 48 |
| deepseek7b | 0.750 | 0.000 | `number_or_symbol` | 8 |
| deepseek7b | 0.750 | 0.000 | `designated_contrast` | 4 |
| deepseek7b | 0.750 | 0.000 | `special_token` | 1 |
| deepseek7b | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 13497 |
| deepseek7b | 0.000 | 0.000 | `whitespace_or_newline` | 1386 |
| deepseek7b | 0.000 | 0.000 | `punctuation` | 1178 |
| deepseek7b | 0.000 | 0.000 | `echo_token` | 558 |
| deepseek7b | 0.000 | 0.000 | `other_token` | 347 |
| deepseek7b | 0.000 | 0.000 | `high_frequency_or_format` | 165 |
| deepseek7b | 0.000 | 0.000 | `candidate_list_or_case_value` | 51 |
| deepseek7b | 0.000 | 0.000 | `number_or_symbol` | 17 |
| deepseek7b | 0.000 | 0.000 | `designated_contrast` | 4 |
| deepseek7b | 0.000 | 0.000 | `special_token` | 1 |
