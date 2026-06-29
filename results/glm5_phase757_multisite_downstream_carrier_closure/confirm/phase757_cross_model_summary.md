# Phase 757 Multi-Site Downstream Carrier Closure Test (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: source removal followed by single-site, primary multi-site, and off-path multi-site component restores.

## Combo Kind Baseline

| model | combo kind | groups | restore rate | recovered | recovery frac | release reduced | roles |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `off_path_control` | 8 | 0.070 | -0.033 | 0.515 | 0.032 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| qwen3 | `same_layer_primary_pair` | 8 | 0.055 | -0.076 | -0.094 | 0.012 | `{'anti_restore_or_off_path': 4, 'weak_or_unclear': 4}` |
| qwen3 | `single_primary_site` | 16 | 0.052 | -0.029 | 0.166 | 0.001 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| glm4 | `off_path_control` | 8 | 0.010 | -0.011 | 0.793 | 0.032 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| glm4 | `primary_multisite_all` | 8 | 0.000 | -0.013 | 0.230 | 0.010 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_primary_pair` | 16 | 0.000 | -0.010 | 0.293 | 0.003 | `{'anti_restore_or_off_path': 1, 'weak_or_unclear': 15}` |
| glm4 | `single_primary_site` | 32 | 0.001 | -0.006 | 0.292 | 0.002 | `{'weak_or_unclear': 32}` |
| deepseek7b | `off_path_control` | 8 | 0.297 | 0.104 | 0.532 | -0.012 | `{'off_path_control_suspicious': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `primary_multisite_all` | 8 | 0.237 | 0.052 | 0.387 | 0.011 | `{'partial_multisite_carrier_candidate': 3, 'weak_or_unclear': 5}` |
| deepseek7b | `same_layer_primary_pair` | 16 | 0.148 | 0.027 | 0.276 | -0.017 | `{'weak_or_unclear': 16}` |
| deepseek7b | `single_primary_site` | 32 | 0.128 | 0.014 | 0.300 | -0.016 | `{'partial_multisite_carrier_candidate': 1, 'weak_or_unclear': 31}` |

## Top Multi-Site Restores

| model | kind | writer | source | combo | sites | n | restore rate | erase drop | recovered | frac | release reduced | guess |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 48 | 0.188 | 0.047 | 0.036 | 0.787 | 0.029 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L34:attn+mlp` | `['L34:attn_out', 'L34:mlp_out']` | 48 | 0.146 | 0.047 | -0.023 | 0.324 | 0.055 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 48 | 0.125 | 0.047 | 0.010 | 0.519 | -0.005 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L34:attn_out` | `['L34:attn_out']` | 48 | 0.125 | 0.047 | 0.005 | 0.426 | 0.008 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 48 | 0.104 | 0.125 | -0.190 | -0.108 | 0.013 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L34:attn_out` | `['L34:attn_out']` | 48 | 0.083 | 0.125 | 0.008 | 0.183 | -0.034 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 48 | 0.083 | 0.125 | -0.206 | -2.188 | -0.042 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L34:attn+mlp` | `['L34:attn_out', 'L34:mlp_out']` | 48 | 0.083 | 0.125 | -0.224 | -2.367 | -0.081 | `anti_restore_or_off_path` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:attn+mlp` | `['L34:attn_out', 'L34:mlp_out']` | 48 | 0.062 | 0.049 | 0.044 | 0.750 | -0.005 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `L34:attn+mlp` | `['L34:attn_out', 'L34:mlp_out']` | 48 | 0.062 | 0.029 | 0.021 | 0.781 | 0.031 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 48 | 0.062 | 0.049 | 0.021 | 0.542 | 0.003 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 48 | 0.062 | 0.029 | 0.018 | 0.719 | 0.026 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `L34:mlp_out` | `['L34:mlp_out']` | 48 | 0.062 | 0.029 | 0.005 | 0.500 | 0.000 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `L34:attn_out` | `['L34:attn_out']` | 48 | 0.062 | 0.029 | 0.003 | 0.719 | 0.026 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 48 | 0.042 | -0.005 | 0.034 | 1.274 | 0.057 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | target_record_line | `L34:attn_out` | `['L34:attn_out']` | 48 | 0.042 | 0.107 | 0.021 | 0.372 | -0.023 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | target_record_line | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 48 | 0.042 | 0.005 | 0.016 | 0.688 | 0.057 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:attn_out` | `['L34:attn_out']` | 48 | 0.042 | 0.049 | 0.005 | 0.646 | -0.005 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | target_record_line | `L34:attn_out` | `['L34:attn_out']` | 48 | 0.042 | 0.005 | 0.000 | 0.688 | 0.018 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 48 | 0.042 | 0.049 | -0.005 | 0.375 | -0.005 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.042 | 0.025 | 0.017 | 0.829 | 0.025 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | target_record_line | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.021 | 0.016 | 0.007 | 0.618 | 0.028 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `L36:attn_out` | `['L36:attn_out']` | 48 | 0.021 | -0.016 | -0.005 | 0.190 | 0.003 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.021 | -0.016 | -0.069 | 0.719 | 0.076 | `anti_restore_or_off_path` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L37:attn_out` | `['L37:attn_out']` | 48 | 0.000 | 0.013 | 0.018 | 0.333 | -0.015 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.010 | 0.012 | 1.000 | 0.018 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.013 | 0.010 | 0.667 | 0.017 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `primary_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out']` | 48 | 0.000 | 0.013 | 0.010 | 0.444 | 0.006 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L37:attn+mlp` | `['L37:attn_out', 'L37:mlp_out']` | 48 | 0.000 | 0.013 | 0.009 | 0.778 | -0.003 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `L36:mlp_out` | `['L36:mlp_out']` | 48 | 0.000 | 0.010 | 0.009 | 0.625 | -0.007 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L36:mlp_out` | `['L36:mlp_out']` | 48 | 0.000 | 0.013 | 0.009 | 0.556 | -0.005 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | `L36:attn_out` | `['L36:attn_out']` | 48 | 0.000 | 0.003 | 0.008 | 0.200 | -0.021 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `L37:mlp_out` | `['L37:mlp_out']` | 48 | 0.000 | 0.010 | 0.007 | 0.625 | -0.005 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L36:attn_out` | `['L36:attn_out']` | 48 | 0.000 | 0.013 | 0.007 | 0.444 | -0.002 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 48 | 0.000 | 0.003 | 0.005 | 0.800 | 0.011 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L36:attn+mlp` | `['L36:attn_out', 'L36:mlp_out']` | 48 | 0.000 | 0.013 | 0.005 | 0.556 | 0.002 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `primary_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out']` | 48 | 0.000 | 0.010 | 0.003 | 0.750 | 0.001 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L37:attn+mlp` | `['L37:attn_out', 'L37:mlp_out']` | 48 | 0.000 | 0.025 | 0.003 | 0.511 | 0.003 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `L36:attn_out` | `['L36:attn_out']` | 48 | 0.000 | 0.010 | 0.003 | 0.375 | -0.007 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L36:attn_out` | `['L36:attn_out']` | 48 | 0.000 | 0.025 | 0.003 | 0.167 | -0.002 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `off_path_same_count` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.604 | 0.500 | 0.262 | 0.614 | -0.147 | `off_path_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `off_path_same_count` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.542 | 0.480 | 0.178 | 0.274 | 0.010 | `off_path_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `off_path_same_count` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.500 | 0.428 | 0.190 | 0.466 | -0.055 | `off_path_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `off_path_same_count` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 48 | 0.500 | 0.385 | 0.147 | 0.373 | -0.017 | `off_path_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 48 | 0.479 | 0.500 | 0.115 | 0.139 | -0.020 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 48 | 0.438 | 0.428 | 0.098 | 0.297 | 0.020 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 48 | 0.396 | 0.480 | 0.128 | 0.262 | -0.027 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 48 | 0.375 | 0.428 | 0.040 | 0.124 | 0.020 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 48 | 0.354 | 0.385 | 0.074 | 0.218 | 0.064 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L23:attn_out` | `['L23:attn_out']` | 48 | 0.354 | 0.500 | 0.034 | 0.020 | -0.010 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L23:attn_out` | `['L23:attn_out']` | 48 | 0.312 | 0.428 | 0.046 | 0.267 | -0.039 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `L24:attn+mlp` | `['L24:attn_out', 'L24:mlp_out']` | 48 | 0.292 | 0.480 | 0.076 | 0.143 | -0.007 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 48 | 0.292 | 0.500 | 0.047 | 0.069 | -0.008 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L24:mlp_out` | `['L24:mlp_out']` | 48 | 0.292 | 0.428 | 0.040 | 0.067 | -0.036 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 48 | 0.271 | 0.480 | 0.059 | 0.064 | -0.010 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 48 | 0.250 | 0.385 | 0.070 | 0.141 | -0.042 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L23:mlp_out` | `['L23:mlp_out']` | 48 | 0.250 | 0.385 | 0.052 | 0.197 | -0.044 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L23:attn_out` | `['L23:attn_out']` | 48 | 0.250 | 0.385 | 0.039 | 0.098 | -0.018 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L24:mlp_out` | `['L24:mlp_out']` | 48 | 0.229 | 0.385 | 0.043 | 0.119 | 0.008 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L24:attn+mlp` | `['L24:attn_out', 'L24:mlp_out']` | 48 | 0.229 | 0.428 | 0.016 | -0.006 | -0.010 | `weak_or_unclear` |

## Strict Interpretation

- Multi-site restore is stronger than Phase 756 only if primary path combos beat off-path controls.
- If off-path controls recover similarly, the result is not a localized carrier.
- Weak multi-site restore points the next bottleneck toward readout threshold / phrase likelihood / generation closure.
