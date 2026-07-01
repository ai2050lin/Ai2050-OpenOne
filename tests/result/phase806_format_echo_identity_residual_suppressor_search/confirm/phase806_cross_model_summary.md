# Phase 806 Format/Echo and Identity-Anchor Residual Suppressor Search (confirm)

- Status: `complete`
- Boundary: direction-level projection after semantic suppression, not neuron-level suppressor discovery.
- Baseline for deltas is semantic-only projection with semantic beta fixed.

## By Projection

| model | target alpha | sem beta | fmt beta | id beta | rows | cases | base blockers | blockers | blocker delta | bias delta | fmt supp | fmt still | id supp | id still | fmt share | fmt share delta | anchor frag | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.750 | 1.000 | 0.000 | 0.000 | 10 | 5 | 318.300 | 318.300 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.510 | 0.000 | 1.000 | 0.000 | `{"semantic_only_baseline": 10}` |
| qwen3 | 0.750 | 1.000 | 0.000 | 1.000 | 10 | 5 | 318.300 | 331.600 | 13.300 | 0.838 | 0.009 | 0.994 | -0.968 | 1.000 | 0.485 | -0.025 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 9, "identity_direction_effective_no_closure": 1}` |
| qwen3 | 0.750 | 1.000 | 1.000 | 0.000 | 10 | 5 | 318.300 | 254.400 | -63.900 | -0.447 | 0.863 | 0.921 | 0.075 | 1.000 | 0.489 | -0.021 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 1, "format_echo_direction_effective_no_closure": 5, "residual_count_reduced_but_class_target_unclear": 4}` |
| qwen3 | 0.750 | 1.000 | 1.000 | 1.000 | 10 | 5 | 318.300 | 264.500 | -53.800 | 0.281 | 0.865 | 0.926 | -0.917 | 1.000 | 0.472 | -0.039 | 1.000 | 0.000 | `{"combined_residual_direction_reduces_blockers": 5, "direction_projection_weak_or_backfires": 1, "format_echo_direction_effective_no_closure": 4}` |
| glm4 | 0.750 | 1.000 | 0.000 | 0.000 | 6 | 3 | 131.833 | 131.833 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.364 | 0.000 | 1.000 | 0.000 | `{"semantic_only_baseline": 6}` |
| glm4 | 0.750 | 1.000 | 0.000 | 1.000 | 6 | 3 | 131.833 | 136.667 | 4.833 | -0.008 | 0.001 | 1.000 | 0.082 | 1.000 | 0.349 | -0.015 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 3, "identity_direction_effective_no_closure": 2, "residual_count_reduced_but_class_target_unclear": 1}` |
| glm4 | 0.750 | 1.000 | 1.000 | 0.000 | 6 | 3 | 131.833 | 142.833 | 11.000 | 0.138 | -0.082 | 0.958 | -0.023 | 1.000 | 0.373 | 0.009 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 4, "format_echo_direction_effective_no_closure": 1, "residual_count_reduced_but_class_target_unclear": 1}` |
| glm4 | 0.750 | 1.000 | 1.000 | 1.000 | 6 | 3 | 131.833 | 149.833 | 18.000 | 0.102 | -0.079 | 0.958 | 0.055 | 1.000 | 0.364 | 0.001 | 1.000 | 0.000 | `{"combined_residual_direction_reduces_blockers": 1, "direction_projection_weak_or_backfires": 3, "identity_direction_effective_no_closure": 2}` |
| deepseek7b | 0.750 | 1.000 | 0.000 | 0.000 | 4 | 2 | 864.250 | 864.250 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.538 | 0.000 | 1.000 | 0.000 | `{"semantic_only_baseline": 4}` |
| deepseek7b | 0.750 | 1.000 | 0.000 | 1.000 | 4 | 2 | 864.250 | 884.000 | 19.750 | -0.061 | -0.004 | 1.000 | 0.945 | 0.917 | 0.537 | -0.001 | 1.000 | 0.000 | `{"identity_direction_effective_no_closure": 4}` |
| deepseek7b | 0.750 | 1.000 | 1.000 | 0.000 | 4 | 2 | 864.250 | 1374.750 | 510.500 | -2.303 | 1.978 | 0.973 | 0.670 | 1.000 | 0.507 | -0.031 | 1.000 | 0.000 | `{"format_echo_direction_effective_no_closure": 2, "residual_count_reduced_but_class_target_unclear": 2}` |
| deepseek7b | 0.750 | 1.000 | 1.000 | 1.000 | 4 | 2 | 864.250 | 1417.750 | 553.500 | -2.490 | 1.978 | 0.973 | 1.609 | 0.917 | 0.511 | -0.026 | 1.000 | 0.000 | `{"combined_residual_direction_reduces_blockers": 3, "format_echo_direction_effective_no_closure": 1}` |

## Direction Token Class Counts

| model | target alpha | fmt beta | id beta | class | count |
|---|---:|---:|---:|---|---:|
| qwen3 | 0.750 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 1821 |
| qwen3 | 0.750 | 1.000 | 0.000 | `echo_token` | 460 |
| qwen3 | 0.750 | 1.000 | 0.000 | `punctuation` | 339 |
| qwen3 | 0.750 | 1.000 | 0.000 | `whitespace_or_newline` | 221 |
| qwen3 | 0.750 | 1.000 | 0.000 | `high_frequency_or_format` | 143 |
| qwen3 | 0.750 | 1.000 | 0.000 | `other_token` | 96 |
| qwen3 | 0.750 | 1.000 | 0.000 | `candidate_list_or_case_value` | 71 |
| qwen3 | 0.750 | 1.000 | 0.000 | `number_or_symbol` | 22 |
| qwen3 | 0.750 | 1.000 | 0.000 | `designated_contrast` | 10 |
| qwen3 | 0.750 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 1821 |
| qwen3 | 0.750 | 1.000 | 1.000 | `echo_token` | 460 |
| qwen3 | 0.750 | 1.000 | 1.000 | `punctuation` | 339 |
| qwen3 | 0.750 | 1.000 | 1.000 | `whitespace_or_newline` | 221 |
| qwen3 | 0.750 | 1.000 | 1.000 | `high_frequency_or_format` | 143 |
| qwen3 | 0.750 | 1.000 | 1.000 | `other_token` | 96 |
| qwen3 | 0.750 | 1.000 | 1.000 | `candidate_list_or_case_value` | 71 |
| qwen3 | 0.750 | 1.000 | 1.000 | `number_or_symbol` | 22 |
| qwen3 | 0.750 | 1.000 | 1.000 | `designated_contrast` | 10 |
| qwen3 | 0.750 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 1821 |
| qwen3 | 0.750 | 0.000 | 0.000 | `echo_token` | 460 |
| qwen3 | 0.750 | 0.000 | 0.000 | `punctuation` | 339 |
| qwen3 | 0.750 | 0.000 | 0.000 | `whitespace_or_newline` | 221 |
| qwen3 | 0.750 | 0.000 | 0.000 | `high_frequency_or_format` | 143 |
| qwen3 | 0.750 | 0.000 | 0.000 | `other_token` | 96 |
| qwen3 | 0.750 | 0.000 | 0.000 | `candidate_list_or_case_value` | 71 |
| qwen3 | 0.750 | 0.000 | 0.000 | `number_or_symbol` | 22 |
| qwen3 | 0.750 | 0.000 | 0.000 | `designated_contrast` | 10 |
| qwen3 | 0.750 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 1821 |
| qwen3 | 0.750 | 0.000 | 1.000 | `echo_token` | 460 |
| qwen3 | 0.750 | 0.000 | 1.000 | `punctuation` | 339 |
| qwen3 | 0.750 | 0.000 | 1.000 | `whitespace_or_newline` | 221 |
| qwen3 | 0.750 | 0.000 | 1.000 | `high_frequency_or_format` | 143 |
| qwen3 | 0.750 | 0.000 | 1.000 | `other_token` | 96 |
| qwen3 | 0.750 | 0.000 | 1.000 | `candidate_list_or_case_value` | 71 |
| qwen3 | 0.750 | 0.000 | 1.000 | `number_or_symbol` | 22 |
| qwen3 | 0.750 | 0.000 | 1.000 | `designated_contrast` | 10 |
| glm4 | 0.750 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 493 |
| glm4 | 0.750 | 0.000 | 0.000 | `echo_token` | 119 |
| glm4 | 0.750 | 0.000 | 0.000 | `punctuation` | 60 |
| glm4 | 0.750 | 0.000 | 0.000 | `candidate_list_or_case_value` | 54 |
| glm4 | 0.750 | 0.000 | 0.000 | `high_frequency_or_format` | 50 |
| glm4 | 0.750 | 0.000 | 0.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 0.000 | 0.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 0.000 | 0.000 | `other_token` | 2 |
| glm4 | 0.750 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 493 |
| glm4 | 0.750 | 0.000 | 1.000 | `echo_token` | 119 |
| glm4 | 0.750 | 0.000 | 1.000 | `punctuation` | 60 |
| glm4 | 0.750 | 0.000 | 1.000 | `candidate_list_or_case_value` | 54 |
| glm4 | 0.750 | 0.000 | 1.000 | `high_frequency_or_format` | 50 |
| glm4 | 0.750 | 0.000 | 1.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 0.000 | 1.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 0.000 | 1.000 | `other_token` | 2 |
| glm4 | 0.750 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 493 |
| glm4 | 0.750 | 1.000 | 0.000 | `echo_token` | 119 |
| glm4 | 0.750 | 1.000 | 0.000 | `punctuation` | 60 |
| glm4 | 0.750 | 1.000 | 0.000 | `candidate_list_or_case_value` | 54 |
| glm4 | 0.750 | 1.000 | 0.000 | `high_frequency_or_format` | 50 |
| glm4 | 0.750 | 1.000 | 0.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 1.000 | 0.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 1.000 | 0.000 | `other_token` | 2 |
| glm4 | 0.750 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 493 |
| glm4 | 0.750 | 1.000 | 1.000 | `echo_token` | 119 |
| glm4 | 0.750 | 1.000 | 1.000 | `punctuation` | 60 |
| glm4 | 0.750 | 1.000 | 1.000 | `candidate_list_or_case_value` | 54 |
| glm4 | 0.750 | 1.000 | 1.000 | `high_frequency_or_format` | 50 |
| glm4 | 0.750 | 1.000 | 1.000 | `whitespace_or_newline` | 7 |
| glm4 | 0.750 | 1.000 | 1.000 | `designated_contrast` | 6 |
| glm4 | 0.750 | 1.000 | 1.000 | `other_token` | 2 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 2154 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `punctuation` | 355 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `whitespace_or_newline` | 345 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `echo_token` | 337 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `high_frequency_or_format` | 113 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `other_token` | 77 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `candidate_list_or_case_value` | 38 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `number_or_symbol` | 31 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `designated_contrast` | 4 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `special_token` | 3 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 2154 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `punctuation` | 355 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `whitespace_or_newline` | 345 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `echo_token` | 337 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `high_frequency_or_format` | 113 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `other_token` | 77 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `candidate_list_or_case_value` | 38 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `number_or_symbol` | 31 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `designated_contrast` | 4 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `special_token` | 3 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 2154 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `punctuation` | 355 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `whitespace_or_newline` | 345 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `echo_token` | 337 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `high_frequency_or_format` | 113 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `other_token` | 77 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `candidate_list_or_case_value` | 38 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `number_or_symbol` | 31 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `designated_contrast` | 4 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `special_token` | 3 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 2154 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `punctuation` | 355 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `whitespace_or_newline` | 345 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `echo_token` | 337 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `high_frequency_or_format` | 113 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `other_token` | 77 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `candidate_list_or_case_value` | 38 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `number_or_symbol` | 31 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `designated_contrast` | 4 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `special_token` | 3 |
