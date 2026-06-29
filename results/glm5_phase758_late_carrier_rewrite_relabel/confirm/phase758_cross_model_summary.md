# Phase 758 Late Carrier Rewrite Relabel Test (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: source removal followed by primary, late-candidate, joint, and true-late-control component restores.

## Combo Kind Baseline

| model | combo kind | groups | restore rate | recovered | recovery frac | release reduced | roles |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `late_candidate_all` | 8 | 0.070 | -0.033 | 0.515 | 0.032 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| qwen3 | `primary_multisite_all` | 8 | 0.055 | -0.076 | -0.094 | 0.012 | `{'anti_restore_or_off_path': 4, 'weak_or_unclear': 4}` |
| qwen3 | `primary_plus_late_all` | 8 | 0.060 | -0.140 | -0.125 | 0.054 | `{'anti_restore_or_off_path': 4, 'weak_or_unclear': 4}` |
| qwen3 | `single_late_candidate_site` | 16 | 0.076 | -0.005 | 0.464 | 0.006 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| qwen3 | `single_primary_site` | 16 | 0.052 | -0.029 | 0.166 | 0.001 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| glm4 | `late_candidate_all` | 8 | 0.010 | -0.011 | 0.793 | 0.032 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| glm4 | `primary_multisite_all` | 8 | 0.000 | -0.013 | 0.230 | 0.010 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | 8 | 0.000 | -0.020 | 0.727 | 0.040 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| glm4 | `same_layer_late_candidate_pair` | 16 | 0.003 | -0.012 | 0.516 | 0.015 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| glm4 | `same_layer_primary_pair` | 16 | 0.000 | -0.010 | 0.293 | 0.003 | `{'anti_restore_or_off_path': 1, 'weak_or_unclear': 15}` |
| glm4 | `single_late_candidate_site` | 8 | 0.000 | -0.001 | 0.329 | 0.003 | `{'weak_or_unclear': 8}` |
| glm4 | `single_primary_site` | 32 | 0.001 | -0.006 | 0.292 | 0.002 | `{'weak_or_unclear': 32}` |
| deepseek7b | `late_candidate_all` | 8 | 0.297 | 0.104 | 0.532 | -0.012 | `{'late_target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `primary_multisite_all` | 8 | 0.237 | 0.052 | 0.387 | 0.011 | `{'partial_primary_carrier_candidate': 3, 'weak_or_unclear': 5}` |
| deepseek7b | `primary_plus_late_all` | 8 | 0.378 | 0.145 | 0.666 | -0.003 | `{'primary_late_joint_target_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `same_layer_late_candidate_pair` | 16 | 0.228 | 0.056 | 0.389 | -0.019 | `{'late_target_rewrite_candidate': 3, 'partial_late_rewrite_candidate': 1, 'weak_or_unclear': 12}` |
| deepseek7b | `same_layer_primary_pair` | 16 | 0.148 | 0.027 | 0.276 | -0.017 | `{'weak_or_unclear': 16}` |
| deepseek7b | `single_primary_site` | 32 | 0.128 | 0.014 | 0.300 | -0.016 | `{'partial_primary_carrier_candidate': 1, 'weak_or_unclear': 31}` |
| deepseek7b | `true_late_control` | 8 | 0.193 | 0.022 | 0.544 | 0.048 | `{'true_late_control_suspicious': 2, 'weak_or_unclear': 6}` |

## Top Multi-Site Restores

| model | kind | writer | source | combo | sites | n | restore rate | erase drop | recovered | frac | release reduced | guess |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L35:attn_out` | `['L35:attn_out']` | 48 | 0.188 | 0.047 | 0.073 | 0.833 | -0.086 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `primary_plus_late_all` | `['L34:attn_out', 'L34:mlp_out', 'L35:attn_out', 'L35:mlp_out']` | 48 | 0.188 | 0.047 | 0.036 | 0.898 | 0.044 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `late_candidate_all` | `['L35:attn_out', 'L35:mlp_out']` | 48 | 0.188 | 0.047 | 0.036 | 0.787 | 0.029 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L35:attn_out` | `['L35:attn_out']` | 48 | 0.167 | 0.125 | 0.039 | 0.371 | -0.018 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L35:mlp_out` | `['L35:mlp_out']` | 48 | 0.167 | 0.047 | 0.026 | 0.630 | 0.044 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `primary_all` | `['L34:attn_out', 'L34:mlp_out']` | 48 | 0.146 | 0.047 | -0.023 | 0.324 | 0.055 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | target_record_line | `L35:attn_out` | `['L35:attn_out']` | 48 | 0.125 | 0.107 | 0.039 | 0.301 | -0.016 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 48 | 0.125 | 0.047 | 0.010 | 0.519 | -0.005 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L34:attn_out` | `['L34:attn_out']` | 48 | 0.125 | 0.047 | 0.005 | 0.426 | 0.008 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `late_candidate_all` | `['L35:attn_out', 'L35:mlp_out']` | 48 | 0.104 | 0.125 | -0.190 | -0.108 | 0.013 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L34:attn_out` | `['L34:attn_out']` | 48 | 0.083 | 0.125 | 0.008 | 0.183 | -0.034 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 48 | 0.083 | 0.125 | -0.206 | -2.188 | -0.042 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L35:mlp_out` | `['L35:mlp_out']` | 48 | 0.083 | 0.125 | -0.211 | -0.261 | 0.029 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `primary_all` | `['L34:attn_out', 'L34:mlp_out']` | 48 | 0.083 | 0.125 | -0.224 | -2.367 | -0.081 | `anti_restore_or_off_path` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `primary_plus_late_all` | `['L34:attn_out', 'L34:mlp_out', 'L35:attn_out', 'L35:mlp_out']` | 48 | 0.062 | 0.049 | 0.083 | 1.208 | 0.013 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `primary_plus_late_all` | `['L34:attn_out', 'L34:mlp_out', 'L35:attn_out', 'L35:mlp_out']` | 48 | 0.062 | 0.029 | 0.060 | 1.062 | 0.055 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `primary_all` | `['L34:attn_out', 'L34:mlp_out']` | 48 | 0.062 | 0.049 | 0.044 | 0.750 | -0.005 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L35:mlp_out` | `['L35:mlp_out']` | 48 | 0.062 | 0.049 | 0.036 | 0.542 | -0.023 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | target_record_line | `L35:mlp_out` | `['L35:mlp_out']` | 48 | 0.062 | 0.005 | 0.023 | 0.875 | 0.052 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `L35:mlp_out` | `['L35:mlp_out']` | 48 | 0.062 | 0.029 | 0.023 | 0.625 | 0.013 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.042 | 0.025 | 0.017 | 0.829 | 0.025 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | target_record_line | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.021 | 0.016 | 0.007 | 0.618 | 0.028 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `L36:attn_out` | `['L36:attn_out']` | 48 | 0.021 | -0.016 | -0.005 | 0.190 | 0.003 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `L38:late_attn+mlp` | `['L38:attn_out', 'L38:mlp_out']` | 48 | 0.021 | -0.016 | -0.012 | 0.433 | 0.035 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | target_record_line | `L38:late_attn+mlp` | `['L38:attn_out', 'L38:mlp_out']` | 48 | 0.021 | -0.017 | -0.013 | 0.400 | 0.030 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.021 | -0.016 | -0.069 | 0.719 | 0.076 | `anti_restore_or_off_path` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `L39:late_attn+mlp` | `['L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.010 | 0.020 | 1.000 | -0.008 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L37:attn_out` | `['L37:attn_out']` | 48 | 0.000 | 0.013 | 0.018 | 0.333 | -0.015 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `primary_plus_late_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out', 'L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.013 | 0.017 | 1.000 | 0.020 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L39:late_attn+mlp` | `['L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.025 | 0.017 | 0.711 | 0.001 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `primary_plus_late_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out', 'L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.010 | 0.012 | 1.000 | 0.026 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.010 | 0.012 | 1.000 | 0.018 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L39:late_attn+mlp` | `['L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.013 | 0.012 | 0.667 | -0.001 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `late_candidate_all` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.013 | 0.010 | 0.667 | 0.017 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `primary_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out']` | 48 | 0.000 | 0.013 | 0.010 | 0.444 | 0.006 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L37:attn+mlp` | `['L37:attn_out', 'L37:mlp_out']` | 48 | 0.000 | 0.013 | 0.009 | 0.778 | -0.003 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `L36:mlp_out` | `['L36:mlp_out']` | 48 | 0.000 | 0.010 | 0.009 | 0.625 | -0.007 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L36:mlp_out` | `['L36:mlp_out']` | 48 | 0.000 | 0.013 | 0.009 | 0.556 | -0.005 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `L38:attn_out` | `['L38:attn_out']` | 48 | 0.000 | 0.010 | 0.008 | 0.500 | 0.006 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | `L36:attn_out` | `['L36:attn_out']` | 48 | 0.000 | 0.003 | 0.008 | 0.200 | -0.021 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.792 | 0.500 | 0.363 | 0.739 | -0.107 | `primary_late_joint_target_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.771 | 0.428 | 0.285 | 0.734 | -0.055 | `primary_late_joint_target_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L26:late_attn+mlp` | `['L26:attn_out', 'L26:mlp_out']` | 48 | 0.625 | 0.500 | 0.203 | 0.419 | -0.036 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.604 | 0.500 | 0.262 | 0.614 | -0.147 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.583 | 0.480 | 0.273 | 0.497 | -0.044 | `primary_late_joint_target_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `primary_plus_late_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out', 'L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.583 | 0.385 | 0.207 | 0.472 | 0.009 | `primary_late_joint_target_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.542 | 0.480 | 0.178 | 0.274 | 0.010 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L26:late_attn+mlp` | `['L26:attn_out', 'L26:mlp_out']` | 48 | 0.542 | 0.428 | 0.167 | 0.434 | -0.057 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `L26:late_attn+mlp` | `['L26:attn_out', 'L26:mlp_out']` | 48 | 0.521 | 0.480 | 0.161 | 0.274 | -0.016 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.500 | 0.428 | 0.190 | 0.466 | -0.055 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `late_candidate_all` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.500 | 0.385 | 0.147 | 0.373 | -0.017 | `late_target_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 48 | 0.479 | 0.500 | 0.115 | 0.139 | -0.020 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 48 | 0.438 | 0.428 | 0.098 | 0.297 | 0.020 | `partial_primary_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L26:late_attn+mlp` | `['L26:attn_out', 'L26:mlp_out']` | 48 | 0.417 | 0.385 | 0.116 | 0.303 | 0.005 | `partial_late_rewrite_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 48 | 0.396 | 0.480 | 0.128 | 0.262 | -0.027 | `partial_primary_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `late_control_same_count` | `['L27:attn_out', 'L27:mlp_out']` | 48 | 0.396 | 0.500 | 0.051 | 0.105 | 0.023 | `true_late_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 48 | 0.375 | 0.428 | 0.040 | 0.124 | 0.020 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 48 | 0.354 | 0.385 | 0.074 | 0.218 | 0.064 | `partial_primary_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L25:late_attn+mlp` | `['L25:attn_out', 'L25:mlp_out']` | 48 | 0.354 | 0.428 | 0.060 | 0.143 | -0.056 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L25:late_attn+mlp` | `['L25:attn_out', 'L25:mlp_out']` | 48 | 0.354 | 0.500 | 0.059 | 0.131 | -0.066 | `weak_or_unclear` |

## Strict Interpretation

- Phase 758 relabels Phase 757 off-path recovery as a late carrier / rewrite candidate.
- Strong evidence requires late_candidate groups to beat primary path and true_late_control groups.
- If target recovery rises but route release is not reduced, the mechanism is target rewrite rather than route closure.
