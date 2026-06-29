# Phase 758 Late Carrier Rewrite Relabel Test (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: source removal followed by primary, late-candidate, joint, and true-late-control component restores.

## Combo Kind Baseline

| model | combo kind | groups | restore rate | recovered | recovery frac | release reduced | roles |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `late_candidate_all` | 8 | 0.068 | -0.063 | 0.186 | 0.033 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| qwen3 | `primary_multisite_all` | 8 | 0.057 | -0.072 | -0.260 | -0.002 | `{'anti_restore_or_off_path': 4, 'weak_or_unclear': 4}` |
| qwen3 | `primary_plus_late_all` | 8 | 0.057 | -0.162 | -0.529 | 0.042 | `{'anti_restore_or_off_path': 4, 'weak_or_unclear': 4}` |
| qwen3 | `single_late_candidate_site` | 16 | 0.081 | -0.019 | 0.325 | 0.007 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| qwen3 | `single_primary_site` | 16 | 0.057 | -0.022 | 0.158 | -0.005 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| glm4 | `late_candidate_all` | 8 | 0.010 | -0.029 | 0.778 | 0.042 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| glm4 | `primary_multisite_all` | 8 | 0.000 | -0.011 | 0.028 | 0.009 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | 8 | 0.000 | -0.038 | 0.519 | 0.049 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| glm4 | `same_layer_late_candidate_pair` | 16 | 0.005 | -0.028 | 0.438 | 0.026 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| glm4 | `same_layer_primary_pair` | 16 | 0.000 | -0.013 | 0.155 | 0.008 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| glm4 | `single_late_candidate_site` | 8 | 0.000 | -0.004 | 0.215 | 0.008 | `{'weak_or_unclear': 8}` |
| glm4 | `single_primary_site` | 32 | 0.001 | -0.008 | 0.212 | 0.005 | `{'anti_restore_or_off_path': 1, 'weak_or_unclear': 31}` |
| deepseek7b | `late_candidate_all` | 8 | 0.271 | 0.098 | 0.453 | 0.019 | `{'late_target_rewrite_candidate': 2, 'late_writer_guard_closure_candidate': 2, 'weak_or_unclear': 4}` |
| deepseek7b | `primary_multisite_all` | 8 | 0.198 | 0.035 | 0.293 | 0.034 | `{'partial_primary_carrier_candidate': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `primary_plus_late_all` | 8 | 0.365 | 0.142 | 0.679 | 0.001 | `{'primary_late_joint_closure_candidate': 1, 'primary_late_joint_target_candidate': 3, 'weak_or_unclear': 4}` |
| deepseek7b | `same_layer_late_candidate_pair` | 16 | 0.201 | 0.049 | 0.366 | -0.002 | `{'late_target_rewrite_candidate': 2, 'late_writer_guard_closure_candidate': 1, 'partial_late_rewrite_candidate': 1, 'weak_or_unclear': 12}` |
| deepseek7b | `same_layer_primary_pair` | 16 | 0.128 | 0.016 | 0.258 | 0.002 | `{'partial_primary_carrier_candidate': 2, 'weak_or_unclear': 14}` |
| deepseek7b | `single_primary_site` | 32 | 0.121 | 0.007 | 0.265 | -0.003 | `{'anti_restore_or_off_path': 2, 'partial_primary_carrier_candidate': 2, 'weak_or_unclear': 28}` |
| deepseek7b | `true_late_control` | 8 | 0.203 | 0.023 | 0.548 | 0.039 | `{'true_late_control_suspicious': 2, 'weak_or_unclear': 6}` |

## Top Multi-Site Restores

| model | kind | writer | source | combo | sites | n | restore rate | erase drop | recovered | frac | release reduced | guess |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L35:attn_out` | `['L35:attn_out']` | 24 | 0.208 | 0.156 | 0.057 | 0.532 | -0.047 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | target_record_line | `L35:attn_out` | `['L35:attn_out']` | 24 | 0.208 | 0.130 | 0.036 | 0.266 | -0.010 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L35:attn_out` | `['L35:attn_out']` | 24 | 0.125 | 0.031 | 0.052 | 0.786 | -0.068 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `primary_plus_late_all` | `['L34:attn_out', 'L34:mlp_out', 'L35:attn_out', 'L35:mlp_out']` | 24 | 0.125 | 0.031 | 0.026 | 0.857 | 0.036 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L35:mlp_out` | `['L35:mlp_out']` | 24 | 0.125 | 0.031 | 0.026 | 0.714 | 0.016 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `late_candidate_all` | `['L35:attn_out', 'L35:mlp_out']` | 24 | 0.125 | 0.031 | 0.021 | 0.714 | 0.010 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.125 | 0.031 | -0.010 | 0.357 | -0.026 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `primary_all` | `['L34:attn_out', 'L34:mlp_out']` | 24 | 0.125 | 0.031 | -0.021 | 0.143 | 0.005 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `primary_plus_late_all` | `['L34:attn_out', 'L34:mlp_out', 'L35:attn_out', 'L35:mlp_out']` | 24 | 0.083 | 0.052 | 0.094 | 0.917 | 0.010 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `primary_plus_late_all` | `['L34:attn_out', 'L34:mlp_out', 'L35:attn_out', 'L35:mlp_out']` | 24 | 0.083 | 0.031 | 0.078 | 1.150 | 0.036 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L35:mlp_out` | `['L35:mlp_out']` | 24 | 0.083 | 0.052 | 0.057 | 0.417 | -0.016 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `primary_all` | `['L34:attn_out', 'L34:mlp_out']` | 24 | 0.083 | 0.052 | 0.047 | 0.625 | 0.021 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `L35:mlp_out` | `['L35:mlp_out']` | 24 | 0.083 | -0.016 | 0.021 | 1.167 | 0.073 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.083 | 0.031 | 0.021 | 0.900 | 0.016 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `L35:mlp_out` | `['L35:mlp_out']` | 24 | 0.083 | 0.031 | 0.021 | 0.300 | 0.010 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `late_candidate_all` | `['L35:attn_out', 'L35:mlp_out']` | 24 | 0.083 | 0.052 | 0.021 | 0.167 | 0.010 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.083 | 0.156 | 0.021 | 0.148 | -0.026 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `late_candidate_all` | `['L35:attn_out', 'L35:mlp_out']` | 24 | 0.083 | -0.016 | 0.016 | 1.167 | 0.083 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | target_record_line | `L35:mlp_out` | `['L35:mlp_out']` | 24 | 0.083 | -0.021 | 0.010 | 0.750 | 0.068 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `primary_all` | `['L34:attn_out', 'L34:mlp_out']` | 24 | 0.083 | 0.031 | 0.010 | 0.650 | 0.021 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.042 | 0.034 | 0.023 | 0.792 | 0.034 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `L36:attn_out` | `['L36:attn_out']` | 24 | 0.042 | -0.044 | -0.008 | 0.333 | 0.010 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | target_record_line | `L38:late_attn+mlp` | `['L38:attn_out', 'L38:mlp_out']` | 24 | 0.042 | -0.042 | -0.031 | 0.467 | 0.053 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `L38:late_attn+mlp` | `['L38:attn_out', 'L38:mlp_out']` | 24 | 0.042 | -0.044 | -0.039 | 0.333 | 0.052 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.042 | -0.044 | -0.156 | 0.333 | 0.120 | `anti_restore_or_off_path` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L39:late_attn+mlp` | `['L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.034 | 0.031 | 0.771 | -0.001 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `L39:late_attn+mlp` | `['L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.013 | 0.016 | 1.000 | 0.001 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `primary_plus_late_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out', 'L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.034 | 0.016 | 0.688 | 0.033 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L37:attn_out` | `['L37:attn_out']` | 24 | 0.000 | 0.013 | 0.016 | 0.667 | -0.013 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `primary_plus_late_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out', 'L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.013 | 0.013 | 1.000 | 0.027 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `primary_plus_late_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out', 'L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.013 | 0.013 | 1.000 | 0.018 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.010 | 0.013 | 1.000 | -0.003 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L37:attn+mlp` | `['L37:attn_out', 'L37:mlp_out']` | 24 | 0.000 | 0.034 | 0.013 | 0.521 | 0.008 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `primary_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out']` | 24 | 0.000 | 0.013 | 0.013 | 0.333 | -0.001 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.013 | 0.010 | 1.000 | 0.020 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | `primary_plus_late_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out', 'L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.010 | 0.010 | 1.000 | 0.008 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L37:attn+mlp` | `['L37:attn_out', 'L37:mlp_out']` | 24 | 0.000 | 0.013 | 0.010 | 1.000 | -0.001 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | target_record_line | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.023 | 0.010 | 0.700 | 0.030 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.013 | 0.010 | 0.667 | 0.016 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L36:mlp_out` | `['L36:mlp_out']` | 24 | 0.000 | 0.013 | 0.008 | 0.667 | -0.003 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.750 | 0.495 | 0.378 | 0.842 | -0.146 | `primary_late_joint_target_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.750 | 0.419 | 0.289 | 0.781 | -0.141 | `primary_late_joint_target_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.542 | 0.523 | 0.271 | 0.480 | -0.003 | `primary_late_joint_target_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.542 | 0.424 | 0.211 | 0.509 | 0.062 | `primary_late_joint_closure_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L26:late_attn+mlp` | `['L26:attn_out', 'L26:mlp_out']` | 24 | 0.542 | 0.495 | 0.190 | 0.434 | -0.055 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.500 | 0.495 | 0.273 | 0.669 | -0.156 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.500 | 0.523 | 0.185 | 0.363 | 0.062 | `late_writer_guard_closure_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `L26:late_attn+mlp` | `['L26:attn_out', 'L26:mlp_out']` | 24 | 0.458 | 0.523 | 0.177 | 0.396 | -0.010 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.458 | 0.419 | 0.164 | 0.441 | -0.034 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.458 | 0.424 | 0.154 | 0.381 | 0.055 | `late_writer_guard_closure_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L26:late_attn+mlp` | `['L26:attn_out', 'L26:mlp_out']` | 24 | 0.458 | 0.424 | 0.122 | 0.273 | 0.039 | `late_writer_guard_closure_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L26:late_attn+mlp` | `['L26:attn_out', 'L26:mlp_out']` | 24 | 0.375 | 0.419 | 0.120 | 0.336 | -0.042 | `partial_late_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 24 | 0.375 | 0.523 | 0.107 | 0.224 | -0.023 | `partial_primary_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `late_control_same_count` | `['L27:attn_out', 'L27:mlp_out']` | 24 | 0.375 | 0.495 | 0.052 | 0.088 | 0.036 | `true_late_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `late_control_same_count` | `['L27:attn_out', 'L27:mlp_out']` | 24 | 0.375 | 0.523 | 0.042 | 0.078 | -0.154 | `true_late_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 24 | 0.333 | 0.424 | 0.078 | 0.240 | 0.128 | `partial_primary_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 24 | 0.333 | 0.495 | 0.068 | 0.042 | -0.003 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L24:mlp_out` | `['L24:mlp_out']` | 24 | 0.333 | 0.419 | 0.039 | 0.118 | -0.049 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L23:attn_out` | `['L23:attn_out']` | 24 | 0.333 | 0.495 | 0.029 | 0.012 | -0.049 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `L24:mlp_out` | `['L24:mlp_out']` | 24 | 0.292 | 0.523 | 0.089 | 0.297 | -0.049 | `partial_primary_carrier_candidate` |

## Strict Interpretation

- Phase 758 relabels Phase 757 off-path recovery as a late carrier / rewrite candidate.
- Strong evidence requires late_candidate groups to beat primary path and true_late_control groups.
- If target recovery rises but route release is not reduced, the mechanism is target rewrite rather than route closure.
