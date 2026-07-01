# Phase 806 Format/Echo and Identity-Anchor Residual Suppressor Search (smoke)

- Status: `complete`
- Boundary: direction-level projection after semantic suppression, not neuron-level suppressor discovery.
- Baseline for deltas is semantic-only projection with semantic beta fixed.

## By Projection

| model | target alpha | sem beta | fmt beta | id beta | rows | cases | base blockers | blockers | blocker delta | bias delta | fmt supp | fmt still | id supp | id still | fmt share | fmt share delta | anchor frag | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.750 | 1.000 | 0.000 | 0.000 | 1 | 1 | 28.000 | 28.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.679 | 0.000 | 1.000 | 0.000 | `{"semantic_only_baseline": 1}` |
| qwen3 | 0.750 | 1.000 | 0.000 | 1.000 | 1 | 1 | 28.000 | 29.000 | 1.000 | 1.250 | 0.003 | 1.000 | -1.469 | 1.000 | 0.655 | -0.023 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 1}` |
| qwen3 | 0.750 | 1.000 | 1.000 | 0.000 | 1 | 1 | 28.000 | 20.000 | -8.000 | -0.125 | 1.355 | 0.526 | 0.016 | 1.000 | 0.500 | -0.179 | 1.000 | 0.000 | `{"format_echo_direction_effective_no_closure": 1}` |
| qwen3 | 0.750 | 1.000 | 1.000 | 1.000 | 1 | 1 | 28.000 | 20.000 | -8.000 | 1.062 | 1.326 | 0.526 | -1.469 | 1.000 | 0.500 | -0.179 | 1.000 | 0.000 | `{"combined_residual_direction_reduces_blockers": 1}` |
| glm4 | 0.750 | 1.000 | 0.000 | 0.000 | 1 | 1 | 94.000 | 94.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.394 | 0.000 | 1.000 | 0.000 | `{"semantic_only_baseline": 1}` |
| glm4 | 0.750 | 1.000 | 0.000 | 1.000 | 1 | 1 | 94.000 | 95.000 | 1.000 | 0.750 | 0.015 | 1.000 | -0.711 | 1.000 | 0.389 | -0.004 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 1}` |
| glm4 | 0.750 | 1.000 | 1.000 | 0.000 | 1 | 1 | 94.000 | 117.000 | 23.000 | 0.219 | -0.671 | 1.000 | -0.039 | 1.000 | 0.470 | 0.076 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 1}` |
| glm4 | 0.750 | 1.000 | 1.000 | 1.000 | 1 | 1 | 94.000 | 116.000 | 22.000 | 0.938 | -0.680 | 1.000 | -0.773 | 1.000 | 0.474 | 0.081 | 1.000 | 0.000 | `{"direction_projection_weak_or_backfires": 1}` |
| deepseek7b | 0.750 | 1.000 | 0.000 | 0.000 | 1 | 1 | 321.000 | 321.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.623 | 0.000 | 1.000 | 0.000 | `{"semantic_only_baseline": 1}` |
| deepseek7b | 0.750 | 1.000 | 0.000 | 1.000 | 1 | 1 | 321.000 | 311.000 | -10.000 | -0.688 | 0.015 | 1.000 | 0.625 | 1.000 | 0.624 | 0.001 | 1.000 | 0.000 | `{"identity_direction_effective_no_closure": 1}` |
| deepseek7b | 0.750 | 1.000 | 1.000 | 0.000 | 1 | 1 | 321.000 | 407.000 | 86.000 | -1.375 | 2.315 | 0.938 | 0.891 | 1.000 | 0.501 | -0.122 | 1.000 | 0.000 | `{"format_echo_direction_effective_no_closure": 1}` |
| deepseek7b | 0.750 | 1.000 | 1.000 | 1.000 | 1 | 1 | 321.000 | 393.000 | 72.000 | -2.156 | 2.276 | 0.938 | 1.500 | 1.000 | 0.517 | -0.107 | 1.000 | 0.000 | `{"format_echo_direction_effective_no_closure": 1}` |

## Direction Token Class Counts

| model | target alpha | fmt beta | id beta | class | count |
|---|---:|---:|---:|---|---:|
| qwen3 | 0.750 | 1.000 | 0.000 | `candidate_list_or_case_value` | 8 |
| qwen3 | 0.750 | 1.000 | 0.000 | `echo_token` | 6 |
| qwen3 | 0.750 | 1.000 | 0.000 | `high_frequency_or_format` | 6 |
| qwen3 | 0.750 | 1.000 | 0.000 | `punctuation` | 4 |
| qwen3 | 0.750 | 1.000 | 0.000 | `whitespace_or_newline` | 3 |
| qwen3 | 0.750 | 1.000 | 0.000 | `designated_contrast` | 1 |
| qwen3 | 0.750 | 1.000 | 1.000 | `candidate_list_or_case_value` | 8 |
| qwen3 | 0.750 | 1.000 | 1.000 | `echo_token` | 6 |
| qwen3 | 0.750 | 1.000 | 1.000 | `high_frequency_or_format` | 6 |
| qwen3 | 0.750 | 1.000 | 1.000 | `punctuation` | 4 |
| qwen3 | 0.750 | 1.000 | 1.000 | `whitespace_or_newline` | 3 |
| qwen3 | 0.750 | 1.000 | 1.000 | `designated_contrast` | 1 |
| qwen3 | 0.750 | 0.000 | 0.000 | `candidate_list_or_case_value` | 8 |
| qwen3 | 0.750 | 0.000 | 0.000 | `echo_token` | 6 |
| qwen3 | 0.750 | 0.000 | 0.000 | `high_frequency_or_format` | 6 |
| qwen3 | 0.750 | 0.000 | 0.000 | `punctuation` | 4 |
| qwen3 | 0.750 | 0.000 | 0.000 | `whitespace_or_newline` | 3 |
| qwen3 | 0.750 | 0.000 | 0.000 | `designated_contrast` | 1 |
| qwen3 | 0.750 | 0.000 | 1.000 | `candidate_list_or_case_value` | 8 |
| qwen3 | 0.750 | 0.000 | 1.000 | `echo_token` | 6 |
| qwen3 | 0.750 | 0.000 | 1.000 | `high_frequency_or_format` | 6 |
| qwen3 | 0.750 | 0.000 | 1.000 | `punctuation` | 4 |
| qwen3 | 0.750 | 0.000 | 1.000 | `whitespace_or_newline` | 3 |
| qwen3 | 0.750 | 0.000 | 1.000 | `designated_contrast` | 1 |
| glm4 | 0.750 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 47 |
| glm4 | 0.750 | 0.000 | 0.000 | `echo_token` | 17 |
| glm4 | 0.750 | 0.000 | 0.000 | `high_frequency_or_format` | 11 |
| glm4 | 0.750 | 0.000 | 0.000 | `candidate_list_or_case_value` | 9 |
| glm4 | 0.750 | 0.000 | 0.000 | `punctuation` | 8 |
| glm4 | 0.750 | 0.000 | 0.000 | `designated_contrast` | 1 |
| glm4 | 0.750 | 0.000 | 0.000 | `whitespace_or_newline` | 1 |
| glm4 | 0.750 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 47 |
| glm4 | 0.750 | 0.000 | 1.000 | `echo_token` | 17 |
| glm4 | 0.750 | 0.000 | 1.000 | `high_frequency_or_format` | 11 |
| glm4 | 0.750 | 0.000 | 1.000 | `candidate_list_or_case_value` | 9 |
| glm4 | 0.750 | 0.000 | 1.000 | `punctuation` | 8 |
| glm4 | 0.750 | 0.000 | 1.000 | `designated_contrast` | 1 |
| glm4 | 0.750 | 0.000 | 1.000 | `whitespace_or_newline` | 1 |
| glm4 | 0.750 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 47 |
| glm4 | 0.750 | 1.000 | 1.000 | `echo_token` | 17 |
| glm4 | 0.750 | 1.000 | 1.000 | `high_frequency_or_format` | 11 |
| glm4 | 0.750 | 1.000 | 1.000 | `candidate_list_or_case_value` | 9 |
| glm4 | 0.750 | 1.000 | 1.000 | `punctuation` | 8 |
| glm4 | 0.750 | 1.000 | 1.000 | `designated_contrast` | 1 |
| glm4 | 0.750 | 1.000 | 1.000 | `whitespace_or_newline` | 1 |
| glm4 | 0.750 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 47 |
| glm4 | 0.750 | 1.000 | 0.000 | `echo_token` | 17 |
| glm4 | 0.750 | 1.000 | 0.000 | `high_frequency_or_format` | 11 |
| glm4 | 0.750 | 1.000 | 0.000 | `candidate_list_or_case_value` | 9 |
| glm4 | 0.750 | 1.000 | 0.000 | `punctuation` | 8 |
| glm4 | 0.750 | 1.000 | 0.000 | `designated_contrast` | 1 |
| glm4 | 0.750 | 1.000 | 0.000 | `whitespace_or_newline` | 1 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 97 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `whitespace_or_newline` | 70 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `echo_token` | 51 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `punctuation` | 51 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `high_frequency_or_format` | 23 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `other_token` | 12 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `candidate_list_or_case_value` | 11 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `number_or_symbol` | 5 |
| deepseek7b | 0.750 | 0.000 | 1.000 | `designated_contrast` | 1 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `semantic_or_lexical_competitor` | 97 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `whitespace_or_newline` | 70 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `echo_token` | 51 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `punctuation` | 51 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `high_frequency_or_format` | 23 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `other_token` | 12 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `candidate_list_or_case_value` | 11 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `number_or_symbol` | 5 |
| deepseek7b | 0.750 | 0.000 | 0.000 | `designated_contrast` | 1 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 97 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `whitespace_or_newline` | 70 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `echo_token` | 51 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `punctuation` | 51 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `high_frequency_or_format` | 23 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `other_token` | 12 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `candidate_list_or_case_value` | 11 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `number_or_symbol` | 5 |
| deepseek7b | 0.750 | 1.000 | 1.000 | `designated_contrast` | 1 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 97 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `whitespace_or_newline` | 70 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `echo_token` | 51 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `punctuation` | 51 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `high_frequency_or_format` | 23 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `other_token` | 12 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `candidate_list_or_case_value` | 11 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `number_or_symbol` | 5 |
| deepseek7b | 0.750 | 1.000 | 0.000 | `designated_contrast` | 1 |
