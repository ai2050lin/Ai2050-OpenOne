# Phase 759 Rewrite vs Route Suppression Split (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Source: Phase 758 restore rows; no model inference in this phase.

## Combo Kind Split

| model | combo kind | n | target rate | route rate | target recovered | route reduced | role |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | `late_candidate_all` | 384 | 0.073 | 0.268 | -0.033 | 0.032 | `weak_split_signal` |
| qwen3 | `primary_multisite_all` | 384 | 0.055 | 0.242 | -0.076 | 0.012 | `weak_split_signal` |
| qwen3 | `primary_plus_late_all` | 384 | 0.060 | 0.323 | -0.140 | 0.054 | `route_suppression_only_candidate` |
| qwen3 | `single_late_candidate_site` | 768 | 0.077 | 0.204 | -0.005 | 0.006 | `weak_split_signal` |
| qwen3 | `single_primary_site` | 768 | 0.053 | 0.212 | -0.029 | 0.001 | `weak_split_signal` |
| glm4 | `late_candidate_all` | 384 | 0.010 | 0.159 | -0.011 | 0.032 | `weak_split_signal` |
| glm4 | `primary_multisite_all` | 384 | 0.000 | 0.120 | -0.013 | 0.010 | `weak_or_unclear` |
| glm4 | `primary_plus_late_all` | 384 | 0.000 | 0.161 | -0.020 | 0.040 | `weak_split_signal` |
| glm4 | `same_layer_late_candidate_pair` | 768 | 0.003 | 0.124 | -0.012 | 0.015 | `weak_or_unclear` |
| glm4 | `same_layer_primary_pair` | 768 | 0.000 | 0.102 | -0.010 | 0.003 | `weak_or_unclear` |
| glm4 | `single_late_candidate_site` | 384 | 0.000 | 0.089 | -0.001 | 0.003 | `weak_or_unclear` |
| glm4 | `single_primary_site` | 1536 | 0.001 | 0.098 | -0.006 | 0.002 | `weak_or_unclear` |
| deepseek7b | `late_candidate_all` | 384 | 0.315 | 0.250 | 0.104 | -0.012 | `weak_split_signal` |
| deepseek7b | `primary_multisite_all` | 384 | 0.250 | 0.289 | 0.052 | 0.011 | `weak_split_signal` |
| deepseek7b | `primary_plus_late_all` | 384 | 0.385 | 0.302 | 0.145 | -0.003 | `joint_target_and_route_candidate` |
| deepseek7b | `same_layer_late_candidate_pair` | 768 | 0.240 | 0.257 | 0.056 | -0.019 | `weak_split_signal` |
| deepseek7b | `same_layer_primary_pair` | 768 | 0.159 | 0.271 | 0.027 | -0.017 | `weak_split_signal` |
| deepseek7b | `single_primary_site` | 1536 | 0.141 | 0.254 | 0.014 | -0.016 | `weak_split_signal` |
| deepseek7b | `true_late_control` | 384 | 0.206 | 0.323 | 0.022 | 0.048 | `weak_split_signal` |

## Top Split Candidates

| model | writer | source | combo | kind | n | target rate | route rate | recovered | route reduced | role |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out:H4 | records_all | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.188 | 0.208 | 0.036 | 0.044 | `weak_split_signal` |
| qwen3 | L33:attn_out:H4 | records_all | `late_candidate_all` | `late_candidate_all` | 48 | 0.188 | 0.188 | 0.036 | 0.029 | `weak_split_signal` |
| qwen3 | L33:attn_out:H4 | records_all | `L35:attn_out` | `single_late_candidate_site` | 48 | 0.188 | 0.083 | 0.073 | -0.086 | `weak_split_signal` |
| qwen3 | L33:attn_out:H4 | records_all | `L35:mlp_out` | `single_late_candidate_site` | 48 | 0.167 | 0.167 | 0.026 | 0.044 | `weak_split_signal` |
| qwen3 | L33:attn_out:H23 | records_all | `L35:attn_out` | `single_late_candidate_site` | 48 | 0.167 | 0.146 | 0.039 | -0.018 | `weak_split_signal` |
| qwen3 | L33:attn_out:H4 | records_all | `primary_all` | `primary_multisite_all` | 48 | 0.146 | 0.229 | -0.023 | 0.055 | `weak_split_signal` |
| qwen3 | L33:attn_out:H4 | records_all | `L34:mlp_out` | `single_primary_site` | 48 | 0.125 | 0.188 | 0.010 | -0.005 | `weak_split_signal` |
| qwen3 | L33:attn_out:H23 | target_record_line | `L35:attn_out` | `single_late_candidate_site` | 48 | 0.125 | 0.125 | 0.039 | -0.016 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H4 | records_all | `L34:attn_out` | `single_primary_site` | 48 | 0.125 | 0.125 | 0.005 | 0.008 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H23 | records_all | `late_candidate_all` | `late_candidate_all` | 48 | 0.104 | 0.188 | -0.190 | 0.013 | `weak_split_signal` |
| qwen3 | L33:attn_out:H23 | records_all | `L35:mlp_out` | `single_late_candidate_site` | 48 | 0.083 | 0.250 | -0.211 | 0.029 | `weak_split_signal` |
| qwen3 | L33:attn_out:H23 | records_all | `L34:attn_out` | `single_primary_site` | 48 | 0.083 | 0.188 | 0.008 | -0.034 | `weak_split_signal` |
| qwen3 | L33:attn_out:H23 | records_all | `primary_all` | `primary_multisite_all` | 48 | 0.083 | 0.167 | -0.224 | -0.081 | `weak_split_signal` |
| qwen3 | L33:attn_out:H23 | records_all | `L34:mlp_out` | `single_primary_site` | 48 | 0.083 | 0.104 | -0.206 | -0.042 | `anti_target_restore` |
| qwen3 | L33:attn_out:H15 | target_record_line | `late_candidate_all` | `late_candidate_all` | 48 | 0.062 | 0.354 | 0.016 | 0.057 | `route_suppression_only_candidate` |
| qwen3 | L33:attn_out:H28 | target_record_line | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.062 | 0.312 | 0.060 | 0.055 | `route_suppression_only_candidate` |
| qwen3 | L33:attn_out:H15 | target_record_line | `L35:mlp_out` | `single_late_candidate_site` | 48 | 0.062 | 0.312 | 0.023 | 0.052 | `route_suppression_only_candidate` |
| qwen3 | L33:attn_out:H23 | records_all | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.062 | 0.271 | -0.549 | 0.010 | `weak_split_signal` |
| qwen3 | L33:attn_out:H28 | target_record_line | `late_candidate_all` | `late_candidate_all` | 48 | 0.062 | 0.250 | 0.018 | 0.026 | `weak_split_signal` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.062 | 0.229 | 0.083 | 0.013 | `weak_split_signal` |
| glm4 | L35:attn_out:H29 | records_all | `late_candidate_all` | `late_candidate_all` | 48 | 0.042 | 0.167 | 0.017 | 0.025 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | records_all | `late_candidate_all` | `late_candidate_all` | 48 | 0.021 | 0.333 | -0.069 | 0.076 | `route_suppression_only_candidate` |
| glm4 | L34:attn_out:H4 | records_all | `L38:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.021 | 0.229 | -0.012 | 0.035 | `weak_split_signal` |
| glm4 | L35:attn_out:H29 | target_record_line | `late_candidate_all` | `late_candidate_all` | 48 | 0.021 | 0.167 | 0.007 | 0.028 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | target_record_line | `L38:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.021 | 0.167 | -0.013 | 0.030 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | records_all | `L36:attn_out` | `single_primary_site` | 48 | 0.021 | 0.146 | -0.005 | 0.003 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.000 | 0.271 | -0.091 | 0.082 | `route_suppression_only_candidate` |
| glm4 | L34:attn_out:H4 | records_all | `L38:attn_out` | `single_late_candidate_site` | 48 | 0.000 | 0.250 | -0.010 | 0.021 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | records_all | `L36:mlp_out` | `single_primary_site` | 48 | 0.000 | 0.250 | -0.036 | 0.021 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | records_all | `L36:attn+mlp` | `same_layer_primary_pair` | 48 | 0.000 | 0.250 | -0.044 | 0.018 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | records_all | `primary_all` | `primary_multisite_all` | 48 | 0.000 | 0.229 | -0.038 | 0.021 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | target_record_line | `late_candidate_all` | `late_candidate_all` | 48 | 0.000 | 0.229 | -0.072 | 0.065 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | target_record_line | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.000 | 0.229 | -0.094 | 0.082 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | target_record_line | `L39:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.000 | 0.208 | -0.103 | 0.057 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | records_all | `L39:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.000 | 0.208 | -0.103 | 0.048 | `weak_split_signal` |
| glm4 | L35:attn_out:H29 | records_all | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.000 | 0.188 | 0.005 | 0.027 | `weak_split_signal` |
| glm4 | L35:attn_out:H29 | target_record_line | `L38:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.000 | 0.188 | -0.003 | 0.021 | `weak_split_signal` |
| glm4 | L35:attn_out:H29 | target_record_line | `L39:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.000 | 0.188 | -0.008 | 0.017 | `weak_split_signal` |
| glm4 | L35:attn_out:H29 | target_record_line | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.000 | 0.188 | -0.012 | 0.033 | `weak_split_signal` |
| glm4 | L34:attn_out:H4 | target_record_line | `L36:attn+mlp` | `same_layer_primary_pair` | 48 | 0.000 | 0.188 | -0.057 | 0.021 | `weak_split_signal` |
| deepseek7b | L22:attn_out:H24 | records_all | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.833 | 0.208 | 0.363 | -0.107 | `target_rewrite_only_candidate` |
| deepseek7b | L22:attn_out:H24 | target_record_line | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.771 | 0.271 | 0.285 | -0.055 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H24 | records_all | `late_candidate_all` | `late_candidate_all` | 48 | 0.667 | 0.104 | 0.262 | -0.147 | `target_rewrite_only_candidate` |
| deepseek7b | L22:attn_out:H24 | records_all | `L26:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.625 | 0.208 | 0.203 | -0.036 | `target_rewrite_only_candidate` |
| deepseek7b | L22:attn_out:H1 | records_all | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.604 | 0.271 | 0.273 | -0.044 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `primary_plus_late_all` | `primary_plus_late_all` | 48 | 0.583 | 0.312 | 0.207 | 0.009 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H1 | records_all | `late_candidate_all` | `late_candidate_all` | 48 | 0.583 | 0.271 | 0.178 | 0.010 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H24 | target_record_line | `L26:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.583 | 0.229 | 0.167 | -0.057 | `target_rewrite_only_candidate` |
| deepseek7b | L22:attn_out:H1 | records_all | `L26:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.562 | 0.292 | 0.161 | -0.016 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H24 | records_all | `primary_all` | `primary_multisite_all` | 48 | 0.542 | 0.250 | 0.115 | -0.020 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H24 | target_record_line | `late_candidate_all` | `late_candidate_all` | 48 | 0.542 | 0.229 | 0.190 | -0.055 | `target_rewrite_only_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `late_candidate_all` | `late_candidate_all` | 48 | 0.500 | 0.292 | 0.147 | -0.017 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `L26:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.438 | 0.375 | 0.116 | 0.005 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H1 | records_all | `primary_all` | `primary_multisite_all` | 48 | 0.438 | 0.312 | 0.128 | -0.027 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H24 | target_record_line | `primary_all` | `primary_multisite_all` | 48 | 0.438 | 0.250 | 0.098 | 0.020 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H24 | records_all | `late_control_same_count` | `true_late_control` | 48 | 0.417 | 0.292 | 0.051 | 0.023 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H24 | records_all | `L23:attn_out` | `single_primary_site` | 48 | 0.396 | 0.229 | 0.034 | -0.010 | `weak_split_signal` |
| deepseek7b | L22:attn_out:H24 | target_record_line | `L23:attn+mlp` | `same_layer_primary_pair` | 48 | 0.375 | 0.312 | 0.040 | 0.020 | `joint_target_and_route_candidate` |
| deepseek7b | L22:attn_out:H24 | records_all | `L25:late_attn+mlp` | `same_layer_late_candidate_pair` | 48 | 0.375 | 0.188 | 0.059 | -0.066 | `weak_split_signal` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `primary_all` | `primary_multisite_all` | 48 | 0.354 | 0.333 | 0.074 | 0.064 | `joint_target_and_route_candidate` |

## Strict Interpretation

- Target recovery and route suppression are scored separately.
- A target-rewrite candidate is not a suppressor unless route success also rises.
- This is an offline split of Phase 758 rows, not new causal intervention.
