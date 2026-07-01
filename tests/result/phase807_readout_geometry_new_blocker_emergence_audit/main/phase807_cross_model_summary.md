# Phase 807 Readout Geometry and New-Blocker Emergence Audit (main)

- Status: `complete`
- Boundary: compares semantic-only blocker sets against residual projections.
- It separates resolved old blockers from emerged new blockers; token closure remains a separate criterion.

## By Projection

| model | fmt beta | id beta | rows | cases | base | after | net | resolved | emerged | emergence rate | emergence share | target delta | base supp | gap red | new token delta | new gap delta | bias delta | fmt supp | id supp | anchor frag | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 0.000 | 6 | 3 | 535.500 | 535.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | null | null | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | `{"mixed_transition": 6}` |
| qwen3 | 0.000 | 1.000 | 6 | 3 | 535.500 | 543.167 | 7.667 | 13.667 | 21.333 | 0.092 | 0.075 | -0.120 | -0.047 | -0.167 | 0.253 | 0.373 | 0.661 | 0.018 | -0.895 | 1.000 | 0.000 | `{"mixed_transition": 1, "new_blocker_emergence_dominant": 3, "readout_closer_candidate_no_closure": 2}` |
| qwen3 | 1.000 | 0.000 | 6 | 3 | 535.500 | 432.333 | -103.167 | 110.000 | 6.833 | 0.013 | 0.016 | 0.146 | 0.457 | 0.602 | 0.447 | 0.332 | -0.708 | 1.013 | 0.053 | 1.000 | 0.000 | `{"net_blocker_reduction_with_emergence": 1, "readout_closer_candidate_no_closure": 5}` |
| qwen3 | 1.000 | 1.000 | 6 | 3 | 535.500 | 447.000 | -88.500 | 110.000 | 21.500 | 0.068 | 0.065 | 0.010 | 0.398 | 0.409 | 0.547 | 0.647 | -0.177 | 1.024 | -0.860 | 1.000 | 0.000 | `{"local_suppression_global_field_deformation": 2, "net_blocker_reduction_with_emergence": 1, "readout_closer_candidate_no_closure": 3}` |
| glm4 | 0.000 | 0.000 | 6 | 3 | 131.167 | 131.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | null | null | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | `{"mixed_transition": 6}` |
| glm4 | 0.000 | 1.000 | 6 | 3 | 131.167 | 136.000 | 4.833 | 1.000 | 5.833 | 0.076 | 0.065 | -0.065 | -0.010 | -0.075 | 0.039 | 0.117 | -0.008 | 0.004 | 0.082 | 1.000 | 0.000 | `{"bias_reduced_but_blocker_count_expands": 1, "new_blocker_emergence_dominant": 4, "readout_closer_candidate_no_closure": 1}` |
| glm4 | 1.000 | 0.000 | 6 | 3 | 131.167 | 142.167 | 11.000 | 6.833 | 17.833 | 0.127 | 0.106 | -0.062 | -0.055 | -0.117 | 0.180 | 0.261 | 0.135 | -0.081 | -0.023 | 1.000 | 0.000 | `{"net_blocker_reduction_with_emergence": 2, "new_blocker_emergence_dominant": 4}` |
| glm4 | 1.000 | 1.000 | 6 | 3 | 131.167 | 149.333 | 18.167 | 7.333 | 25.500 | 0.212 | 0.160 | -0.122 | -0.066 | -0.188 | 0.162 | 0.285 | 0.112 | -0.077 | 0.044 | 1.000 | 0.000 | `{"net_blocker_reduction_with_emergence": 1, "new_blocker_emergence_dominant": 5}` |
| deepseek7b | 0.000 | 0.000 | 6 | 3 | 601.500 | 601.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | null | null | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | `{"mixed_transition": 6}` |
| deepseek7b | 0.000 | 1.000 | 6 | 3 | 601.500 | 602.667 | 1.167 | 6.167 | 7.333 | 0.017 | 0.017 | 0.005 | 0.018 | 0.023 | 0.016 | 0.078 | -0.099 | -0.002 | 0.742 | 1.000 | 0.000 | `{"mixed_transition": 1, "new_blocker_emergence_dominant": 3, "readout_closer_candidate_no_closure": 2}` |
| deepseek7b | 1.000 | 0.000 | 6 | 3 | 601.500 | 980.000 | 378.500 | 64.167 | 442.667 | 0.233 | 0.160 | -0.111 | 0.556 | 0.445 | 0.698 | 0.808 | -1.368 | 1.455 | 0.442 | 1.000 | 0.000 | `{"local_suppression_global_field_deformation": 1, "net_blocker_reduction_with_emergence": 1, "new_blocker_emergence_dominant": 1, "readout_closer_candidate_no_closure": 3}` |
| deepseek7b | 1.000 | 1.000 | 6 | 3 | 601.500 | 973.333 | 371.833 | 67.167 | 439.000 | 0.230 | 0.160 | -0.074 | 0.572 | 0.498 | 0.812 | 0.930 | -1.592 | 1.459 | 1.161 | 1.000 | 0.000 | `{"local_suppression_global_field_deformation": 1, "net_blocker_reduction_with_emergence": 1, "new_blocker_emergence_dominant": 1, "readout_closer_candidate_no_closure": 3}` |

## Emerged Class Counts

| model | fmt beta | id beta | class | count |
|---|---:|---:|---|---:|
| qwen3 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 36 |
| qwen3 | 1.000 | 0.000 | `whitespace_or_newline` | 2 |
| qwen3 | 1.000 | 0.000 | `candidate_list_or_case_value` | 1 |
| qwen3 | 1.000 | 0.000 | `other_token` | 1 |
| qwen3 | 1.000 | 0.000 | `punctuation` | 1 |
| qwen3 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 111 |
| qwen3 | 1.000 | 1.000 | `candidate_list_or_case_value` | 4 |
| qwen3 | 1.000 | 1.000 | `whitespace_or_newline` | 4 |
| qwen3 | 1.000 | 1.000 | `echo_token` | 3 |
| qwen3 | 1.000 | 1.000 | `punctuation` | 3 |
| qwen3 | 1.000 | 1.000 | `high_frequency_or_format` | 2 |
| qwen3 | 1.000 | 1.000 | `other_token` | 2 |
| qwen3 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 98 |
| qwen3 | 0.000 | 1.000 | `echo_token` | 9 |
| qwen3 | 0.000 | 1.000 | `punctuation` | 8 |
| qwen3 | 0.000 | 1.000 | `whitespace_or_newline` | 5 |
| qwen3 | 0.000 | 1.000 | `candidate_list_or_case_value` | 3 |
| qwen3 | 0.000 | 1.000 | `high_frequency_or_format` | 3 |
| qwen3 | 0.000 | 1.000 | `other_token` | 2 |
| glm4 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 29 |
| glm4 | 0.000 | 1.000 | `echo_token` | 4 |
| glm4 | 0.000 | 1.000 | `high_frequency_or_format` | 2 |
| glm4 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 63 |
| glm4 | 1.000 | 0.000 | `echo_token` | 24 |
| glm4 | 1.000 | 0.000 | `punctuation` | 9 |
| glm4 | 1.000 | 0.000 | `high_frequency_or_format` | 8 |
| glm4 | 1.000 | 0.000 | `other_token` | 3 |
| glm4 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 95 |
| glm4 | 1.000 | 1.000 | `echo_token` | 29 |
| glm4 | 1.000 | 1.000 | `high_frequency_or_format` | 14 |
| glm4 | 1.000 | 1.000 | `punctuation` | 11 |
| glm4 | 1.000 | 1.000 | `other_token` | 3 |
| glm4 | 1.000 | 1.000 | `whitespace_or_newline` | 1 |
| deepseek7b | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 32 |
| deepseek7b | 0.000 | 1.000 | `echo_token` | 5 |
| deepseek7b | 0.000 | 1.000 | `punctuation` | 4 |
| deepseek7b | 0.000 | 1.000 | `whitespace_or_newline` | 3 |
| deepseek7b | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 2384 |
| deepseek7b | 1.000 | 1.000 | `punctuation` | 116 |
| deepseek7b | 1.000 | 1.000 | `whitespace_or_newline` | 87 |
| deepseek7b | 1.000 | 1.000 | `other_token` | 33 |
| deepseek7b | 1.000 | 1.000 | `echo_token` | 11 |
| deepseek7b | 1.000 | 1.000 | `high_frequency_or_format` | 2 |
| deepseek7b | 1.000 | 1.000 | `candidate_list_or_case_value` | 1 |
| deepseek7b | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 2408 |
| deepseek7b | 1.000 | 0.000 | `punctuation` | 113 |
| deepseek7b | 1.000 | 0.000 | `whitespace_or_newline` | 88 |
| deepseek7b | 1.000 | 0.000 | `other_token` | 33 |
| deepseek7b | 1.000 | 0.000 | `echo_token` | 11 |
| deepseek7b | 1.000 | 0.000 | `high_frequency_or_format` | 2 |
| deepseek7b | 1.000 | 0.000 | `candidate_list_or_case_value` | 1 |

## Resolved Class Counts

| model | fmt beta | id beta | class | count |
|---|---:|---:|---|---:|
| qwen3 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 448 |
| qwen3 | 1.000 | 0.000 | `echo_token` | 62 |
| qwen3 | 1.000 | 0.000 | `punctuation` | 61 |
| qwen3 | 1.000 | 0.000 | `whitespace_or_newline` | 39 |
| qwen3 | 1.000 | 0.000 | `other_token` | 28 |
| qwen3 | 1.000 | 0.000 | `high_frequency_or_format` | 15 |
| qwen3 | 1.000 | 0.000 | `number_or_symbol` | 7 |
| qwen3 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 462 |
| qwen3 | 1.000 | 1.000 | `echo_token` | 58 |
| qwen3 | 1.000 | 1.000 | `punctuation` | 57 |
| qwen3 | 1.000 | 1.000 | `whitespace_or_newline` | 36 |
| qwen3 | 1.000 | 1.000 | `other_token` | 26 |
| qwen3 | 1.000 | 1.000 | `high_frequency_or_format` | 14 |
| qwen3 | 1.000 | 1.000 | `number_or_symbol` | 7 |
| qwen3 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 66 |
| qwen3 | 0.000 | 1.000 | `whitespace_or_newline` | 6 |
| qwen3 | 0.000 | 1.000 | `echo_token` | 4 |
| qwen3 | 0.000 | 1.000 | `punctuation` | 3 |
| qwen3 | 0.000 | 1.000 | `other_token` | 2 |
| qwen3 | 0.000 | 1.000 | `high_frequency_or_format` | 1 |
| glm4 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 4 |
| glm4 | 0.000 | 1.000 | `echo_token` | 2 |
| glm4 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 21 |
| glm4 | 1.000 | 0.000 | `echo_token` | 11 |
| glm4 | 1.000 | 0.000 | `high_frequency_or_format` | 5 |
| glm4 | 1.000 | 0.000 | `punctuation` | 2 |
| glm4 | 1.000 | 0.000 | `other_token` | 1 |
| glm4 | 1.000 | 0.000 | `whitespace_or_newline` | 1 |
| glm4 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 23 |
| glm4 | 1.000 | 1.000 | `echo_token` | 11 |
| glm4 | 1.000 | 1.000 | `high_frequency_or_format` | 6 |
| glm4 | 1.000 | 1.000 | `punctuation` | 2 |
| glm4 | 1.000 | 1.000 | `other_token` | 1 |
| glm4 | 1.000 | 1.000 | `whitespace_or_newline` | 1 |
| deepseek7b | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 21 |
| deepseek7b | 0.000 | 1.000 | `echo_token` | 5 |
| deepseek7b | 0.000 | 1.000 | `whitespace_or_newline` | 5 |
| deepseek7b | 0.000 | 1.000 | `punctuation` | 3 |
| deepseek7b | 0.000 | 1.000 | `candidate_list_or_case_value` | 2 |
| deepseek7b | 0.000 | 1.000 | `other_token` | 1 |
| deepseek7b | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 190 |
| deepseek7b | 1.000 | 1.000 | `echo_token` | 71 |
| deepseek7b | 1.000 | 1.000 | `punctuation` | 57 |
| deepseek7b | 1.000 | 1.000 | `whitespace_or_newline` | 43 |
| deepseek7b | 1.000 | 1.000 | `high_frequency_or_format` | 19 |
| deepseek7b | 1.000 | 1.000 | `other_token` | 13 |
| deepseek7b | 1.000 | 1.000 | `number_or_symbol` | 6 |
| deepseek7b | 1.000 | 1.000 | `candidate_list_or_case_value` | 2 |
| deepseek7b | 1.000 | 1.000 | `special_token` | 2 |
| deepseek7b | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 181 |
| deepseek7b | 1.000 | 0.000 | `echo_token` | 65 |
| deepseek7b | 1.000 | 0.000 | `punctuation` | 54 |
| deepseek7b | 1.000 | 0.000 | `whitespace_or_newline` | 43 |
| deepseek7b | 1.000 | 0.000 | `high_frequency_or_format` | 21 |
| deepseek7b | 1.000 | 0.000 | `other_token` | 12 |
| deepseek7b | 1.000 | 0.000 | `number_or_symbol` | 6 |
| deepseek7b | 1.000 | 0.000 | `special_token` | 2 |
| deepseek7b | 1.000 | 0.000 | `candidate_list_or_case_value` | 1 |
