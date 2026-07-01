# Phase 807 Readout Geometry and New-Blocker Emergence Audit (confirm)

- Status: `complete`
- Boundary: compares semantic-only blocker sets against residual projections.
- It separates resolved old blockers from emerged new blockers; token closure remains a separate criterion.

## By Projection

| model | fmt beta | id beta | rows | cases | base | after | net | resolved | emerged | emergence rate | emergence share | target delta | base supp | gap red | new token delta | new gap delta | bias delta | fmt supp | id supp | anchor frag | closure | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 0.000 | 0.000 | 10 | 5 | 318.300 | 318.300 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | null | null | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | `{"mixed_transition": 10}` |
| qwen3 | 0.000 | 1.000 | 10 | 5 | 318.300 | 331.600 | 13.300 | 4.200 | 17.500 | 0.086 | 0.071 | -0.125 | -0.076 | -0.201 | 0.294 | 0.478 | 0.838 | 0.009 | -0.968 | 1.000 | 0.000 | `{"bias_reduced_but_blocker_count_expands": 1, "mixed_transition": 1, "new_blocker_emergence_dominant": 7, "readout_closer_candidate_no_closure": 1}` |
| qwen3 | 1.000 | 0.000 | 10 | 5 | 318.300 | 254.400 | -63.900 | 68.000 | 4.100 | 0.012 | 0.014 | 0.059 | 0.394 | 0.454 | 0.309 | 0.270 | -0.447 | 0.863 | 0.075 | 1.000 | 0.000 | `{"bias_reduced_but_blocker_count_expands": 1, "new_blocker_emergence_dominant": 1, "readout_closer_candidate_no_closure": 8}` |
| qwen3 | 1.000 | 1.000 | 10 | 5 | 318.300 | 264.500 | -53.800 | 66.200 | 12.400 | 0.075 | 0.067 | -0.044 | 0.303 | 0.259 | 0.466 | 0.595 | 0.281 | 0.865 | -0.917 | 1.000 | 0.000 | `{"bias_reduced_but_blocker_count_expands": 1, "local_suppression_global_field_deformation": 2, "mixed_transition": 1, "net_blocker_reduction_with_emergence": 2, "new_blocker_emergence_dominant": 1, "readout_closer_candidate_no_closure": 3}` |
| glm4 | 0.000 | 0.000 | 6 | 3 | 131.833 | 131.833 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | null | null | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | `{"mixed_transition": 6}` |
| glm4 | 0.000 | 1.000 | 6 | 3 | 131.833 | 136.667 | 4.833 | 0.667 | 5.500 | 0.075 | 0.064 | -0.065 | -0.010 | -0.075 | 0.040 | 0.142 | -0.008 | 0.001 | 0.082 | 1.000 | 0.000 | `{"bias_reduced_but_blocker_count_expands": 1, "new_blocker_emergence_dominant": 4, "readout_closer_candidate_no_closure": 1}` |
| glm4 | 1.000 | 0.000 | 6 | 3 | 131.833 | 142.833 | 11.000 | 6.667 | 17.667 | 0.126 | 0.106 | -0.065 | -0.054 | -0.120 | 0.186 | 0.271 | 0.138 | -0.082 | -0.023 | 1.000 | 0.000 | `{"net_blocker_reduction_with_emergence": 2, "new_blocker_emergence_dominant": 4}` |
| glm4 | 1.000 | 1.000 | 6 | 3 | 131.833 | 149.833 | 18.000 | 7.500 | 25.500 | 0.212 | 0.160 | -0.122 | -0.066 | -0.188 | 0.167 | 0.289 | 0.102 | -0.079 | 0.055 | 1.000 | 0.000 | `{"net_blocker_reduction_with_emergence": 1, "new_blocker_emergence_dominant": 5}` |
| deepseek7b | 0.000 | 0.000 | 4 | 2 | 864.250 | 864.250 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | null | null | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | `{"mixed_transition": 4}` |
| deepseek7b | 0.000 | 1.000 | 4 | 2 | 864.250 | 884.000 | 19.750 | 4.500 | 24.250 | 0.025 | 0.023 | 0.014 | 0.005 | 0.019 | 0.017 | 0.099 | -0.061 | -0.004 | 0.945 | 1.000 | 0.000 | `{"new_blocker_emergence_dominant": 2, "readout_closer_candidate_no_closure": 2}` |
| deepseek7b | 1.000 | 0.000 | 4 | 2 | 864.250 | 1374.750 | 510.500 | 103.750 | 614.250 | 0.303 | 0.199 | -0.135 | 0.721 | 0.586 | 0.805 | 0.939 | -2.303 | 1.978 | 0.670 | 1.000 | 0.000 | `{"local_suppression_global_field_deformation": 1, "net_blocker_reduction_with_emergence": 1, "readout_closer_candidate_no_closure": 2}` |
| deepseek7b | 1.000 | 1.000 | 4 | 2 | 864.250 | 1417.750 | 553.500 | 99.750 | 653.250 | 0.315 | 0.200 | -0.119 | 0.737 | 0.618 | 0.979 | 1.190 | -2.490 | 1.978 | 1.609 | 1.000 | 0.000 | `{"local_suppression_global_field_deformation": 1, "net_blocker_reduction_with_emergence": 1, "readout_closer_candidate_no_closure": 2}` |

## Emerged Class Counts

| model | fmt beta | id beta | class | count |
|---|---:|---:|---|---:|
| qwen3 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 34 |
| qwen3 | 1.000 | 0.000 | `whitespace_or_newline` | 4 |
| qwen3 | 1.000 | 0.000 | `candidate_list_or_case_value` | 1 |
| qwen3 | 1.000 | 0.000 | `other_token` | 1 |
| qwen3 | 1.000 | 0.000 | `punctuation` | 1 |
| qwen3 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 101 |
| qwen3 | 1.000 | 1.000 | `echo_token` | 7 |
| qwen3 | 1.000 | 1.000 | `whitespace_or_newline` | 6 |
| qwen3 | 1.000 | 1.000 | `candidate_list_or_case_value` | 3 |
| qwen3 | 1.000 | 1.000 | `other_token` | 3 |
| qwen3 | 1.000 | 1.000 | `high_frequency_or_format` | 2 |
| qwen3 | 1.000 | 1.000 | `punctuation` | 2 |
| qwen3 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 127 |
| qwen3 | 0.000 | 1.000 | `echo_token` | 18 |
| qwen3 | 0.000 | 1.000 | `punctuation` | 11 |
| qwen3 | 0.000 | 1.000 | `whitespace_or_newline` | 11 |
| qwen3 | 0.000 | 1.000 | `candidate_list_or_case_value` | 3 |
| qwen3 | 0.000 | 1.000 | `other_token` | 3 |
| qwen3 | 0.000 | 1.000 | `high_frequency_or_format` | 2 |
| glm4 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 28 |
| glm4 | 0.000 | 1.000 | `echo_token` | 3 |
| glm4 | 0.000 | 1.000 | `high_frequency_or_format` | 2 |
| glm4 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 62 |
| glm4 | 1.000 | 0.000 | `echo_token` | 24 |
| glm4 | 1.000 | 0.000 | `punctuation` | 9 |
| glm4 | 1.000 | 0.000 | `high_frequency_or_format` | 8 |
| glm4 | 1.000 | 0.000 | `other_token` | 3 |
| glm4 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 94 |
| glm4 | 1.000 | 1.000 | `echo_token` | 29 |
| glm4 | 1.000 | 1.000 | `high_frequency_or_format` | 14 |
| glm4 | 1.000 | 1.000 | `punctuation` | 11 |
| glm4 | 1.000 | 1.000 | `other_token` | 4 |
| glm4 | 1.000 | 1.000 | `whitespace_or_newline` | 1 |
| deepseek7b | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 79 |
| deepseek7b | 0.000 | 1.000 | `echo_token` | 7 |
| deepseek7b | 0.000 | 1.000 | `punctuation` | 5 |
| deepseek7b | 0.000 | 1.000 | `number_or_symbol` | 2 |
| deepseek7b | 0.000 | 1.000 | `other_token` | 2 |
| deepseek7b | 0.000 | 1.000 | `whitespace_or_newline` | 2 |
| deepseek7b | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 2214 |
| deepseek7b | 1.000 | 0.000 | `punctuation` | 110 |
| deepseek7b | 1.000 | 0.000 | `whitespace_or_newline` | 86 |
| deepseek7b | 1.000 | 0.000 | `other_token` | 34 |
| deepseek7b | 1.000 | 0.000 | `echo_token` | 10 |
| deepseek7b | 1.000 | 0.000 | `high_frequency_or_format` | 2 |
| deepseek7b | 1.000 | 0.000 | `candidate_list_or_case_value` | 1 |
| deepseek7b | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 2362 |
| deepseek7b | 1.000 | 1.000 | `punctuation` | 115 |
| deepseek7b | 1.000 | 1.000 | `whitespace_or_newline` | 87 |
| deepseek7b | 1.000 | 1.000 | `other_token` | 35 |
| deepseek7b | 1.000 | 1.000 | `echo_token` | 10 |
| deepseek7b | 1.000 | 1.000 | `high_frequency_or_format` | 3 |
| deepseek7b | 1.000 | 1.000 | `candidate_list_or_case_value` | 1 |

## Resolved Class Counts

| model | fmt beta | id beta | class | count |
|---|---:|---:|---|---:|
| qwen3 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 448 |
| qwen3 | 1.000 | 0.000 | `echo_token` | 66 |
| qwen3 | 1.000 | 0.000 | `punctuation` | 63 |
| qwen3 | 1.000 | 0.000 | `whitespace_or_newline` | 47 |
| qwen3 | 1.000 | 0.000 | `other_token` | 29 |
| qwen3 | 1.000 | 0.000 | `high_frequency_or_format` | 19 |
| qwen3 | 1.000 | 0.000 | `number_or_symbol` | 8 |
| qwen3 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 449 |
| qwen3 | 1.000 | 1.000 | `echo_token` | 62 |
| qwen3 | 1.000 | 1.000 | `punctuation` | 61 |
| qwen3 | 1.000 | 1.000 | `whitespace_or_newline` | 39 |
| qwen3 | 1.000 | 1.000 | `other_token` | 27 |
| qwen3 | 1.000 | 1.000 | `high_frequency_or_format` | 17 |
| qwen3 | 1.000 | 1.000 | `number_or_symbol` | 7 |
| qwen3 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 33 |
| qwen3 | 0.000 | 1.000 | `other_token` | 3 |
| qwen3 | 0.000 | 1.000 | `punctuation` | 3 |
| qwen3 | 0.000 | 1.000 | `high_frequency_or_format` | 1 |
| qwen3 | 0.000 | 1.000 | `number_or_symbol` | 1 |
| qwen3 | 0.000 | 1.000 | `whitespace_or_newline` | 1 |
| glm4 | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 4 |
| glm4 | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 19 |
| glm4 | 1.000 | 0.000 | `echo_token` | 11 |
| glm4 | 1.000 | 0.000 | `high_frequency_or_format` | 6 |
| glm4 | 1.000 | 0.000 | `punctuation` | 2 |
| glm4 | 1.000 | 0.000 | `other_token` | 1 |
| glm4 | 1.000 | 0.000 | `whitespace_or_newline` | 1 |
| glm4 | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 24 |
| glm4 | 1.000 | 1.000 | `echo_token` | 11 |
| glm4 | 1.000 | 1.000 | `high_frequency_or_format` | 6 |
| glm4 | 1.000 | 1.000 | `punctuation` | 2 |
| glm4 | 1.000 | 1.000 | `other_token` | 1 |
| glm4 | 1.000 | 1.000 | `whitespace_or_newline` | 1 |
| deepseek7b | 0.000 | 1.000 | `semantic_or_lexical_competitor` | 7 |
| deepseek7b | 0.000 | 1.000 | `whitespace_or_newline` | 4 |
| deepseek7b | 0.000 | 1.000 | `punctuation` | 3 |
| deepseek7b | 0.000 | 1.000 | `echo_token` | 2 |
| deepseek7b | 0.000 | 1.000 | `candidate_list_or_case_value` | 1 |
| deepseek7b | 0.000 | 1.000 | `high_frequency_or_format` | 1 |
| deepseek7b | 1.000 | 0.000 | `semantic_or_lexical_competitor` | 211 |
| deepseek7b | 1.000 | 0.000 | `echo_token` | 66 |
| deepseek7b | 1.000 | 0.000 | `punctuation` | 53 |
| deepseek7b | 1.000 | 0.000 | `whitespace_or_newline` | 41 |
| deepseek7b | 1.000 | 0.000 | `high_frequency_or_format` | 21 |
| deepseek7b | 1.000 | 0.000 | `other_token` | 13 |
| deepseek7b | 1.000 | 0.000 | `number_or_symbol` | 8 |
| deepseek7b | 1.000 | 0.000 | `special_token` | 2 |
| deepseek7b | 1.000 | 1.000 | `semantic_or_lexical_competitor` | 196 |
| deepseek7b | 1.000 | 1.000 | `echo_token` | 70 |
| deepseek7b | 1.000 | 1.000 | `punctuation` | 52 |
| deepseek7b | 1.000 | 1.000 | `whitespace_or_newline` | 35 |
| deepseek7b | 1.000 | 1.000 | `high_frequency_or_format` | 21 |
| deepseek7b | 1.000 | 1.000 | `other_token` | 14 |
| deepseek7b | 1.000 | 1.000 | `number_or_symbol` | 8 |
| deepseek7b | 1.000 | 1.000 | `special_token` | 2 |
| deepseek7b | 1.000 | 1.000 | `candidate_list_or_case_value` | 1 |
