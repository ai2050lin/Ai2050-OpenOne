# Phase 757 Multi-Site Downstream Carrier Closure Test (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence: source removal followed by single-site, primary multi-site, and off-path multi-site component restores.

## Combo Kind Baseline

| model | combo kind | groups | restore rate | recovered | recovery frac | release reduced | roles |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `off_path_control` | 8 | 0.068 | -0.063 | 0.186 | 0.033 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| qwen3 | `same_layer_primary_pair` | 8 | 0.057 | -0.072 | -0.260 | -0.002 | `{'anti_restore_or_off_path': 4, 'weak_or_unclear': 4}` |
| qwen3 | `single_primary_site` | 16 | 0.057 | -0.022 | 0.158 | -0.005 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| glm4 | `off_path_control` | 8 | 0.010 | -0.029 | 0.778 | 0.042 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 6}` |
| glm4 | `primary_multisite_all` | 8 | 0.000 | -0.011 | 0.028 | 0.009 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_primary_pair` | 16 | 0.000 | -0.013 | 0.155 | 0.008 | `{'anti_restore_or_off_path': 2, 'weak_or_unclear': 14}` |
| glm4 | `single_primary_site` | 32 | 0.001 | -0.008 | 0.212 | 0.005 | `{'anti_restore_or_off_path': 1, 'weak_or_unclear': 31}` |
| deepseek7b | `off_path_control` | 8 | 0.271 | 0.098 | 0.453 | 0.019 | `{'off_path_control_suspicious': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `primary_multisite_all` | 8 | 0.198 | 0.035 | 0.293 | 0.034 | `{'partial_multisite_carrier_candidate': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `same_layer_primary_pair` | 16 | 0.128 | 0.016 | 0.258 | 0.002 | `{'partial_multisite_carrier_candidate': 2, 'weak_or_unclear': 14}` |
| deepseek7b | `single_primary_site` | 32 | 0.121 | 0.007 | 0.265 | -0.003 | `{'anti_restore_or_off_path': 2, 'partial_multisite_carrier_candidate': 2, 'weak_or_unclear': 28}` |

## Top Multi-Site Restores

| model | kind | writer | source | combo | sites | n | restore rate | erase drop | recovered | frac | release reduced | guess |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 24 | 0.125 | 0.031 | 0.021 | 0.714 | 0.010 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.125 | 0.031 | -0.010 | 0.357 | -0.026 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L34:attn+mlp` | `['L34:attn_out', 'L34:mlp_out']` | 24 | 0.125 | 0.031 | -0.021 | 0.143 | 0.005 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:attn+mlp` | `['L34:attn_out', 'L34:mlp_out']` | 24 | 0.083 | 0.052 | 0.047 | 0.625 | 0.021 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.083 | 0.031 | 0.021 | 0.900 | 0.016 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 24 | 0.083 | 0.052 | 0.021 | 0.167 | 0.010 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.083 | 0.156 | 0.021 | 0.148 | -0.026 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 24 | 0.083 | -0.016 | 0.016 | 1.167 | 0.083 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `L34:attn+mlp` | `['L34:attn_out', 'L34:mlp_out']` | 24 | 0.083 | 0.031 | 0.010 | 0.650 | 0.021 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 24 | 0.083 | 0.031 | 0.010 | 0.450 | 0.021 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 24 | 0.083 | 0.052 | 0.005 | 0.333 | -0.016 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H4 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 24 | 0.083 | 0.031 | 0.000 | 0.571 | -0.031 | `weak_or_unclear` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | target_record_line | `L34:mlp_out` | `['L34:mlp_out']` | 24 | 0.083 | 0.031 | 0.000 | 0.400 | 0.000 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `L34:mlp_out` | `['L34:mlp_out']` | 24 | 0.083 | 0.156 | -0.203 | -1.925 | -0.026 | `anti_restore_or_off_path` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | records_all | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 24 | 0.083 | 0.156 | -0.286 | -1.000 | 0.021 | `anti_restore_or_off_path` |
| qwen3 | `same_layer_control_head` | L33:attn_out:H28 | records_all | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.042 | 0.052 | 0.031 | 0.833 | -0.005 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H23 | target_record_line | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.042 | 0.130 | 0.016 | 0.316 | -0.042 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | records_all | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.042 | -0.016 | 0.010 | 0.500 | 0.031 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | target_record_line | `off_path_same_count` | `['L35:attn_out', 'L35:mlp_out']` | 24 | 0.042 | -0.021 | 0.000 | 0.688 | 0.062 | `weak_or_unclear` |
| qwen3 | `phase755_top_candidate` | L33:attn_out:H15 | target_record_line | `L34:attn_out` | `['L34:attn_out']` | 24 | 0.042 | -0.021 | -0.005 | 0.750 | 0.057 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.042 | 0.034 | 0.023 | 0.792 | 0.034 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `L36:attn_out` | `['L36:attn_out']` | 24 | 0.042 | -0.044 | -0.008 | 0.333 | 0.010 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L34:attn_out:H4 | records_all | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.042 | -0.044 | -0.156 | 0.333 | 0.120 | `anti_restore_or_off_path` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L37:attn_out` | `['L37:attn_out']` | 24 | 0.000 | 0.013 | 0.016 | 0.667 | -0.013 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.010 | 0.013 | 1.000 | -0.003 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L37:attn+mlp` | `['L37:attn_out', 'L37:mlp_out']` | 24 | 0.000 | 0.034 | 0.013 | 0.521 | 0.008 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `primary_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out']` | 24 | 0.000 | 0.013 | 0.013 | 0.333 | -0.001 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.013 | 0.010 | 1.000 | 0.020 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L37:attn+mlp` | `['L37:attn_out', 'L37:mlp_out']` | 24 | 0.000 | 0.013 | 0.010 | 1.000 | -0.001 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | target_record_line | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.023 | 0.010 | 0.700 | 0.030 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.013 | 0.010 | 0.667 | 0.016 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L36:mlp_out` | `['L36:mlp_out']` | 24 | 0.000 | 0.013 | 0.008 | 0.667 | -0.003 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | `L36:attn_out` | `['L36:attn_out']` | 24 | 0.000 | 0.010 | 0.008 | 0.333 | -0.020 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | records_all | `off_path_same_count` | `['L38:attn_out', 'L38:mlp_out', 'L39:attn_out', 'L39:mlp_out']` | 24 | 0.000 | 0.003 | 0.005 | 1.000 | 0.016 | `weak_or_unclear` |
| glm4 | `phase755_top_candidate` | L35:attn_out:H29 | records_all | `L37:mlp_out` | `['L37:mlp_out']` | 24 | 0.000 | 0.034 | 0.005 | 0.271 | 0.005 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `primary_all` | `['L36:attn_out', 'L36:mlp_out', 'L37:attn_out', 'L37:mlp_out']` | 24 | 0.000 | 0.013 | 0.003 | 0.800 | -0.005 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L36:attn+mlp` | `['L36:attn_out', 'L36:mlp_out']` | 24 | 0.000 | 0.013 | 0.003 | 0.667 | 0.010 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | records_all | `L36:attn_out` | `['L36:attn_out']` | 24 | 0.000 | 0.013 | 0.003 | 0.667 | 0.003 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L35:attn_out:H10 | target_record_line | `L37:attn_out` | `['L37:attn_out']` | 24 | 0.000 | 0.010 | 0.003 | 0.667 | -0.022 | `weak_or_unclear` |
| glm4 | `same_layer_control_head` | L34:attn_out:H17 | target_record_line | `L36:mlp_out` | `['L36:mlp_out']` | 24 | 0.000 | 0.013 | 0.003 | 0.600 | 0.010 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `off_path_same_count` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.500 | 0.495 | 0.273 | 0.669 | -0.156 | `off_path_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `off_path_same_count` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.500 | 0.523 | 0.185 | 0.363 | 0.062 | `off_path_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `off_path_same_count` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.458 | 0.419 | 0.164 | 0.441 | -0.034 | `off_path_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `off_path_same_count` | `['L25:attn_out', 'L25:mlp_out', 'L26:attn_out', 'L26:mlp_out']` | 24 | 0.458 | 0.424 | 0.154 | 0.381 | 0.055 | `off_path_control_suspicious` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 24 | 0.375 | 0.523 | 0.107 | 0.224 | -0.023 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 24 | 0.333 | 0.424 | 0.078 | 0.240 | 0.128 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 24 | 0.333 | 0.495 | 0.068 | 0.042 | -0.003 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L24:mlp_out` | `['L24:mlp_out']` | 24 | 0.333 | 0.419 | 0.039 | 0.118 | -0.049 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L23:attn_out` | `['L23:attn_out']` | 24 | 0.333 | 0.495 | 0.029 | 0.012 | -0.049 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `L24:mlp_out` | `['L24:mlp_out']` | 24 | 0.292 | 0.523 | 0.089 | 0.297 | -0.049 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `primary_all` | `['L23:attn_out', 'L23:mlp_out', 'L24:attn_out', 'L24:mlp_out']` | 24 | 0.292 | 0.419 | 0.052 | 0.113 | -0.029 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L23:attn_out` | `['L23:attn_out']` | 24 | 0.292 | 0.419 | 0.000 | 0.139 | -0.044 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `L24:attn+mlp` | `['L24:attn_out', 'L24:mlp_out']` | 24 | 0.250 | 0.523 | 0.091 | 0.248 | -0.018 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L24:attn+mlp` | `['L24:attn_out', 'L24:mlp_out']` | 24 | 0.250 | 0.424 | 0.065 | 0.156 | 0.026 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L23:mlp_out` | `['L23:mlp_out']` | 24 | 0.250 | 0.424 | 0.062 | 0.196 | -0.029 | `partial_multisite_carrier_candidate` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L23:attn_out` | `['L23:attn_out']` | 24 | 0.250 | 0.424 | 0.029 | 0.012 | -0.010 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | target_record_line | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 24 | 0.250 | 0.419 | -0.005 | -0.072 | -0.005 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | target_record_line | `L24:mlp_out` | `['L24:mlp_out']` | 24 | 0.208 | 0.424 | 0.070 | 0.178 | 0.013 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H1 | records_all | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 24 | 0.208 | 0.523 | 0.039 | 0.055 | 0.023 | `weak_or_unclear` |
| deepseek7b | `phase755_top_candidate` | L22:attn_out:H24 | records_all | `L23:attn+mlp` | `['L23:attn_out', 'L23:mlp_out']` | 24 | 0.208 | 0.495 | 0.003 | 0.036 | -0.057 | `weak_or_unclear` |

## Strict Interpretation

- Multi-site restore is stronger than Phase 756 only if primary path combos beat off-path controls.
- If off-path controls recover similarly, the result is not a localized carrier.
- Weak multi-site restore points the next bottleneck toward readout threshold / phrase likelihood / generation closure.
