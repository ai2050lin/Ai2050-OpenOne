# Phase 806 Format/Echo and Identity-Anchor Residual Suppressor Search (main)

- Status: `complete`
- Boundary: direction-level projection after semantic suppression, not neuron-level suppressor discovery.
- Baseline for deltas is semantic-only projection with semantic beta fixed.

## By Projection

| model | target alpha | sem beta | fmt beta | id beta | rows | cases | base blockers | blockers | blocker delta | bias delta | fmt supp | fmt still | id supp | id still | fmt share | fmt share delta | anchor frag | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.750 | 1.000 | 0.000 | 0.000 | 6 | 3 | 535.500 | 535.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.475 | 0.000 | 1.000 | 0.000 | `{"semantic_only_baseline": 6}` |
| qwen3 | 0.750 | 1.000 | 0.000 | 1.000 | 6 | 3 | 535.500 | 543.167 | 7.667 | 0.661 | 0.018 | 0.991 | -0.895 | 1.000 | 0.446 | -0.029 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 4, "identity_direction_effective_no_closure": 1, "residual_count_reduced_but_class_target_unclear": 1}` |
| qwen3 | 0.750 | 1.000 | 1.000 | 0.000 | 6 | 3 | 535.500 | 432.333 | -103.167 | -0.708 | 1.013 | 0.927 | 0.053 | 1.000 | 0.454 | -0.021 | 1.000 | 0.000 | `{"format_echo_direction_effective_no_closure": 3, "residual_count_reduced_but_class_target_unclear": 3}` |
| qwen3 | 0.750 | 1.000 | 1.000 | 1.000 | 6 | 3 | 535.500 | 447.000 | -88.500 | -0.177 | 1.024 | 0.935 | -0.860 | 1.000 | 0.433 | -0.041 | 1.000 | 0.000 | `{"combined_residual_direction_reduces_blockers": 4, "format_echo_direction_effective_no_closure": 2}` |
| glm4 | 0.750 | 1.000 | 0.000 | 0.000 | 6 | 3 | 131.167 | 131.167 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.364 | 0.000 | 1.000 | 0.000 | `{"semantic_only_baseline": 6}` |
| glm4 | 0.750 | 1.000 | 0.000 | 1.000 | 6 | 3 | 131.167 | 136.000 | 4.833 | -0.008 | 0.004 | 1.000 | 0.082 | 1.000 | 0.349 | -0.015 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 3, "identity_direction_effective_no_closure": 2, "residual_count_reduced_but_class_target_unclear": 1}` |
| glm4 | 0.750 | 1.000 | 1.000 | 0.000 | 6 | 3 | 131.167 | 142.167 | 11.000 | 0.135 | -0.081 | 0.958 | -0.023 | 1.000 | 0.374 | 0.009 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 4, "format_echo_direction_effective_no_closure": 1, "residual_count_reduced_but_class_target_unclear": 1}` |
| glm4 | 0.750 | 1.000 | 1.000 | 1.000 | 6 | 3 | 131.167 | 149.333 | 18.167 | 0.112 | -0.077 | 0.956 | 0.044 | 1.000 | 0.365 | 0.001 | 1.000 | 0.000 | `{"combined_residual_direction_reduces_blockers": 1, "direction_projection_weak_or_backfires": 3, "identity_direction_effective_no_closure": 2}` |
| deepseek7b | 0.750 | 1.000 | 0.000 | 0.000 | 6 | 3 | 601.500 | 601.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.531 | 0.000 | 1.000 | 0.000 | `{"semantic_only_baseline": 6}` |
| deepseek7b | 0.750 | 1.000 | 0.000 | 1.000 | 6 | 3 | 601.500 | 602.667 | 1.167 | -0.099 | -0.002 | 1.000 | 0.742 | 0.903 | 0.540 | 0.009 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 1, "identity_direction_effective_no_closure": 5}` |
| deepseek7b | 0.750 | 1.000 | 1.000 | 0.000 | 6 | 3 | 601.500 | 980.000 | 378.500 | -1.368 | 1.455 | 0.879 | 0.442 | 1.000 | 0.482 | -0.049 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 1, "format_echo_direction_effective_no_closure": 3, "residual_count_reduced_but_class_target_unclear": 2}` |
| deepseek7b | 0.750 | 1.000 | 1.000 | 1.000 | 6 | 3 | 601.500 | 973.333 | 371.833 | -1.592 | 1.459 | 0.879 | 1.161 | 0.903 | 0.486 | -0.045 | 1.000 | 0.000 | `{"combined_residual_direction_reduces_blockers": 4, "format_echo_direction_effective_no_closure": 1, "identity_direction_effective_no_closure": 1}` |

## Direction Token Class Counts

| model | target alpha | fmt beta | id beta | class | count |
|---|---:|---:|---:|---|---:|
| qwen3 | 0.750 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 1931 |
| qwen3 | 0.750 | 1.000 | 0.000 | `echo_token` | 439 |
| qwen3 | 0.750 | 1.000 | 0.000 | `punctuation` | 327 |
| qwen3 | 0.750 | 1.000 | 0.000 | `whitespace_or_newline` | 214 |
| qwen3 | 0.750 | 1.000 | 0.000 | `high_frequency_or_format` | 125 |
| qwen3 | 0.750 | 1.000 | 0.000 | `other_token` | 95 |
| qwen3 | 0.750 | 1.000 | 0.000 | `candidate_list_or_case_value` | 55 |
| qwen3 | 0.750 | 1.000 | 0.000 | `number_or_symbol` | 21 |
| qwen3 | 0.750 | 1.000 | 0.000 | `designated_contrast` | 6 |
| qwen3 | 0.750 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 1931 |
| qwen3 | 0.750 | 1.000 | 1.000 | `echo_token` | 439 |
| qwen3 | 0.750 | 1.000 | 1.000 | `punctuation` | 327 |
| qwen3 | 0.750 | 1.000 | 1.000 | `whitespace_or_newline` | 214 |
| qwen3 | 0.750 | 1.000 | 1.000 | `high_frequency_or_format` | 125 |
| qwen3 | 0.750 | 1.000 | 1.000 | `other_token` | 95 |
| qwen3 | 0.750 | 1.000 | 1.000 | `candidate_list_or_case_value` | 55 |
| qwen3 | 0.750 | 1.000 | 1.000 | `number_or_symbol` | 21 |
| qwen3 | 0.750 | 1.000 | 1.000 | `designated_contrast` | 6 |
| qwen3 | 0.750 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 1931 |
| qwen3 | 0.750 | 0.000 | 0.000 | `echo_token` | 439 |
| qwen3 | 0.750 | 0.000 | 0.000 | `punctuation` | 327 |
| qwen3 | 0.750 | 0.000 | 0.000 | `whitespace_or_newline` | 214 |
| qwen3 | 0.750 | 0.000 | 0.000 | `high_frequency_or_format` | 125 |
| qwen3 | 0.750 | 0.000 | 0.000 | `other_token` | 95 |
| qwen3 | 0.750 | 0.000 | 0.000 | `candidate_list_or_case_value` | 55 |
| qwen3 | 0.750 | 0.000 | 0.000 | `number_or_symbol` | 21 |
| qwen3 | 0.750 | 0.000 | 0.000 | `designated_contrast` | 6 |
| qwen3 | 0.750 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 1931 |
| qwen3 | 0.750 | 0.000 | 1.000 | `echo_token` | 439 |
| qwen3 | 0.750 | 0.000 | 1.000 | `punctuation` | 327 |
| qwen3 | 0.750 | 0.000 | 1.000 | `whitespace_or_newline` | 214 |
| qwen3 | 0.750 | 0.000 | 1.000 | `high_frequency_or_format` | 125 |
| qwen3 | 0.750 | 0.000 | 1.000 | `other_token` | 95 |
| qwen3 | 0.750 | 0.000 | 1.000 | `candidate_list_or_case_value` | 55 |
| qwen3 | 0.750 | 0.000 | 1.000 | `number_or_symbol` | 21 |
| qwen3 | 0.750 | 0.000 | 1.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 489 |
| glm4 | 0.750 | 0.000 | 0.000 | `echo_token` | 119 |
| glm4 | 0.750 | 0.000 | 0.000 | `punctuation` | 60 |
| glm4 | 0.750 | 0.000 | 0.000 | `candidate_list_or_case_value` | 54 |
| glm4 | 0.750 | 0.000 | 0.000 | `high_frequency_or_format` | 50 |
| glm4 | 0.750 | 0.000 | 0.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 0.000 | 0.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 0.000 | 0.000 | `other_token` | 2 |
| glm4 | 0.750 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 489 |
| glm4 | 0.750 | 0.000 | 1.000 | `echo_token` | 119 |
| glm4 | 0.750 | 0.000 | 1.000 | `punctuation` | 60 |
| glm4 | 0.750 | 0.000 | 1.000 | `candidate_list_or_case_value` | 54 |
| glm4 | 0.750 | 0.000 | 1.000 | `high_frequency_or_format` | 50 |
| glm4 | 0.750 | 0.000 | 1.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 0.000 | 1.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 0.000 | 1.000 | `other_token` | 2 |
| glm4 | 0.750 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 489 |
| glm4 | 0.750 | 1.000 | 0.000 | `echo_token` | 119 |
| glm4 | 0.750 | 1.000 | 0.000 | `punctuation` | 60 |
| glm4 | 0.750 | 1.000 | 0.000 | `candidate_list_or_case_value` | 54 |
| glm4 | 0.750 | 1.000 | 0.000 | `high_frequency_or_format` | 50 |
| glm4 | 0.750 | 1.000 | 0.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 1.000 | 0.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 1.000 | 0.000 | `other_token` | 2 |
| glm4 | 0.750 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 489 |
| glm4 | 0.750 | 1.000 | 1.000 | `echo_token` | 119 |
| glm4 | 0.750 | 1.000 | 1.000 | `punctuation` | 60 |
| glm4 | 0.750 | 1.000 | 1.000 | `candidate_list_or_case_value` | 54 |
| glm4 | 0.750 | 1.000 | 1.000 | `high_frequency_or_format` | 50 |
| glm4 | 0.750 | 1.000 | 1.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 1.000 | 1.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 1.000 | 1.000 | `other_token` | 2 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 2239 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `punctuation` | 367 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `whitespace_or_newline` | 367 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `echo_token` | 344 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `high_frequency_or_format` | 116 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `other_token` | 82 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `candidate_list_or_case_value` | 56 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `number_or_symbol` | 29 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `designated_contrast` | 6 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `special_token` | 3 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 2239 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `punctuation` | 367 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `whitespace_or_newline` | 367 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `echo_token` | 344 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `high_frequency_or_format` | 116 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `other_token` | 82 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `candidate_list_or_case_value` | 56 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `number_or_symbol` | 29 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `designated_contrast` | 6 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `special_token` | 3 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 2239 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `punctuation` | 367 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `whitespace_or_newline` | 367 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `echo_token` | 344 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `high_frequency_or_format` | 116 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `other_token` | 82 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `candidate_list_or_case_value` | 56 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `number_or_symbol` | 29 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `designated_contrast` | 6 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `special_token` | 3 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 2239 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `punctuation` | 367 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `whitespace_or_newline` | 367 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `echo_token` | 344 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `high_frequency_or_format` | 116 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `other_token` | 82 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `candidate_list_or_case_value` | 56 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `number_or_symbol` | 29 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `designated_contrast` | 6 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `special_token` | 3 |
