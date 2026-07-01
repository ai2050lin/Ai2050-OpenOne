# Phase 807 Readout Geometry and New-Blocker Emergence Audit (smoke)

- Status: `complete`
- Boundary: compares semantic-only blocker sets against residual projections.
- It separates resolved old blockers from emerged new blockers; token closure remains a separate criterion.

## By Projection

| model | fmt beta | id beta | rows | cases | base | after | net | resolved | emerged | emergence rate | emergence share | target delta | base supp | gap red | new token delta | new gap delta | bias delta | fmt supp | id supp | anchor frag | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 0.000 | 1 | 1 | 28.000 | 28.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | null | null | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | `{"mixed_transition": 1}` |
| qwen3 | 0.000 | 1.000 | 1 | 1 | 28.000 | 29.000 | 1.000 | 0.000 | 1.000 | 0.036 | 0.034 | 0.000 | -0.203 | -0.203 | 0.562 | 0.562 | 1.250 | 0.003 | -1.469 | 1.000 | 0.000 | `{"new_blocker_emergence_dominant": 1}` |
| qwen3 | 1.000 | 0.000 | 1 | 1 | 28.000 | 20.000 | -8.000 | 9.000 | 1.000 | 0.036 | 0.050 | 0.000 | 0.922 | 0.922 | 0.250 | 0.250 | -0.125 | 1.355 | 0.016 | 1.000 | 0.000 | `{"readout_closer_candidate_no_closure": 1}` |
| qwen3 | 1.000 | 1.000 | 1 | 1 | 28.000 | 20.000 | -8.000 | 9.000 | 1.000 | 0.036 | 0.050 | 0.062 | 0.699 | 0.761 | 0.812 | 0.750 | 1.062 | 1.326 | -1.469 | 1.000 | 0.000 | `{"net_blocker_reduction_with_emergence": 1}` |
| glm4 | 0.000 | 0.000 | 1 | 1 | 94.000 | 94.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | null | null | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | `{"mixed_transition": 1}` |
| glm4 | 0.000 | 1.000 | 1 | 1 | 94.000 | 95.000 | 1.000 | 0.000 | 1.000 | 0.011 | 0.011 | 0.000 | -0.053 | -0.053 | 0.062 | 0.062 | 0.750 | 0.015 | -0.711 | 1.000 | 0.000 | `{"new_blocker_emergence_dominant": 1}` |
| glm4 | 1.000 | 0.000 | 1 | 1 | 94.000 | 117.000 | 23.000 | 0.000 | 23.000 | 0.245 | 0.197 | -0.031 | -0.305 | -0.336 | 0.454 | 0.485 | 0.219 | -0.671 | -0.039 | 1.000 | 0.000 | `{"new_blocker_emergence_dominant": 1}` |
| glm4 | 1.000 | 1.000 | 1 | 1 | 94.000 | 116.000 | 22.000 | 0.000 | 22.000 | 0.234 | 0.190 | 0.000 | -0.370 | -0.370 | 0.482 | 0.482 | 0.938 | -0.680 | -0.773 | 1.000 | 0.000 | `{"new_blocker_emergence_dominant": 1}` |
| deepseek7b | 0.000 | 0.000 | 1 | 1 | 321.000 | 321.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | null | null | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | `{"mixed_transition": 1}` |
| deepseek7b | 0.000 | 1.000 | 1 | 1 | 321.000 | 311.000 | -10.000 | 10.000 | 0.000 | 0.000 | 0.000 | 0.062 | 0.015 | 0.077 | null | null | -0.688 | 0.015 | 0.625 | 1.000 | 0.000 | `{"readout_closer_candidate_no_closure": 1}` |
| deepseek7b | 1.000 | 0.000 | 1 | 1 | 321.000 | 407.000 | 86.000 | 81.000 | 167.000 | 0.520 | 0.410 | -0.062 | 0.913 | 0.850 | 1.626 | 1.688 | -1.375 | 2.315 | 0.891 | 1.000 | 0.000 | `{"local_suppression_global_field_deformation": 1}` |
| deepseek7b | 1.000 | 1.000 | 1 | 1 | 321.000 | 393.000 | 72.000 | 82.000 | 154.000 | 0.480 | 0.392 | -0.031 | 0.909 | 0.878 | 1.647 | 1.678 | -2.156 | 2.276 | 1.500 | 1.000 | 0.000 | `{"local_suppression_global_field_deformation": 1}` |

## Emerged Class Counts

| model | fmt beta | id beta | class | count |
|---|---:|---:|---|---:|
| qwen3 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 1 |
| qwen3 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 1 |
| qwen3 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 1 |
| glm4 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 1 |
| glm4 | 1.000 | 1.000 | `echo_token` | 11 |
| glm4 | 1.000 | 1.000 | `high_frequency_or_format` | 5 |
| glm4 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 3 |
| glm4 | 1.000 | 1.000 | `punctuation` | 2 |
| glm4 | 1.000 | 1.000 | `other_token` | 1 |
| glm4 | 1.000 | 0.000 | `echo_token` | 11 |
| glm4 | 1.000 | 0.000 | `high_frequency_or_format` | 5 |
| glm4 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 4 |
| glm4 | 1.000 | 0.000 | `punctuation` | 2 |
| glm4 | 1.000 | 0.000 | `other_token` | 1 |
| deepseek7b | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 80 |
| deepseek7b | 1.000 | 1.000 | `whitespace_or_newline` | 36 |
| deepseek7b | 1.000 | 1.000 | `punctuation` | 20 |
| deepseek7b | 1.000 | 1.000 | `other_token` | 12 |
| deepseek7b | 1.000 | 1.000 | `echo_token` | 4 |
| deepseek7b | 1.000 | 1.000 | `candidate_list_or_case_value` | 1 |
| deepseek7b | 1.000 | 1.000 | `high_frequency_or_format` | 1 |
| deepseek7b | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 92 |
| deepseek7b | 1.000 | 0.000 | `whitespace_or_newline` | 37 |
| deepseek7b | 1.000 | 0.000 | `punctuation` | 20 |
| deepseek7b | 1.000 | 0.000 | `other_token` | 12 |
| deepseek7b | 1.000 | 0.000 | `echo_token` | 4 |
| deepseek7b | 1.000 | 0.000 | `candidate_list_or_case_value` | 1 |
| deepseek7b | 1.000 | 0.000 | `high_frequency_or_format` | 1 |

## Resolved Class Counts

| model | fmt beta | id beta | class | count |
|---|---:|---:|---|---:|
| qwen3 | 1.000 | 0.000 | `echo_token` | 3 |
| qwen3 | 1.000 | 0.000 | `high_frequency_or_format` | 2 |
| qwen3 | 1.000 | 0.000 | `punctuation` | 2 |
| qwen3 | 1.000 | 0.000 | `whitespace_or_newline` | 2 |
| qwen3 | 1.000 | 1.000 | `echo_token` | 3 |
| qwen3 | 1.000 | 1.000 | `high_frequency_or_format` | 2 |
| qwen3 | 1.000 | 1.000 | `punctuation` | 2 |
| qwen3 | 1.000 | 1.000 | `whitespace_or_newline` | 2 |
| deepseek7b | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 3 |
| deepseek7b | 0.000 | 1.000 | `echo_token` | 2 |
| deepseek7b | 0.000 | 1.000 | `punctuation` | 2 |
| deepseek7b | 0.000 | 1.000 | `high_frequency_or_format` | 1 |
| deepseek7b | 0.000 | 1.000 | `number_or_symbol` | 1 |
| deepseek7b | 0.000 | 1.000 | `other_token` | 1 |
| deepseek7b | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 20 |
| deepseek7b | 1.000 | 1.000 | `punctuation` | 19 |
| deepseek7b | 1.000 | 1.000 | `echo_token` | 16 |
| deepseek7b | 1.000 | 1.000 | `whitespace_or_newline` | 14 |
| deepseek7b | 1.000 | 1.000 | `high_frequency_or_format` | 6 |
| deepseek7b | 1.000 | 1.000 | `other_token` | 4 |
| deepseek7b | 1.000 | 1.000 | `number_or_symbol` | 3 |
| deepseek7b | 1.000 | 0.000 | `punctuation` | 19 |
| deepseek7b | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 19 |
| deepseek7b | 1.000 | 0.000 | `echo_token` | 16 |
| deepseek7b | 1.000 | 0.000 | `whitespace_or_newline` | 14 |
| deepseek7b | 1.000 | 0.000 | `high_frequency_or_format` | 6 |
| deepseek7b | 1.000 | 0.000 | `other_token` | 4 |
| deepseek7b | 1.000 | 0.000 | `number_or_symbol` | 3 |
