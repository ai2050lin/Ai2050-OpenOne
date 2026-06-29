# Phase 760 Route Suppression Matrix Atlas (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Route groups: contrast_answer / object_relation_echo / other_record_value / format_schema / generic_answer / top_non_target / dynamic top classes.

## Combo Kind x Route Group

| model | combo kind | route group | groups | route-only | route rate | target rate | route reduced | target recovered | roles |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `primary_plus_late_all` | `top_class:recipient_answer` | 8 | 0.500 | 0.500 | 0.000 | 0.078 | -0.125 | `{'route_only_suppressor_candidate': 4, 'weak_or_unclear': 4}` |
| qwen3 | `late_candidate_all` | `top_class:recipient_answer` | 8 | 0.500 | 0.500 | 0.000 | 0.062 | -0.094 | `{'route_only_suppressor_candidate': 4, 'weak_or_unclear': 4}` |
| qwen3 | `primary_multisite_all` | `top_class:recipient_answer` | 8 | 0.500 | 0.500 | 0.000 | 0.047 | -0.062 | `{'route_only_suppressor_candidate': 4, 'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 3}` |
| qwen3 | `single_late_candidate_site` | `top_class:recipient_answer` | 16 | 0.312 | 0.312 | 0.000 | 0.031 | -0.031 | `{'route_only_suppressor_candidate': 5, 'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 10}` |
| qwen3 | `primary_plus_late_all` | `top_class:format_or_schema` | 8 | 0.219 | 0.219 | 0.057 | -0.006 | -0.162 | `{'route_only_suppressor_candidate': 2, 'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 5}` |
| qwen3 | `primary_plus_late_all` | `top_non_target` | 8 | 0.193 | 0.193 | 0.057 | 0.023 | -0.162 | `{'weak_or_unclear': 8}` |
| qwen3 | `single_primary_site` | `top_class:recipient_answer` | 16 | 0.188 | 0.188 | 0.000 | 0.016 | 0.016 | `{'route_only_suppressor_candidate': 3, 'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 11}` |
| qwen3 | `late_candidate_all` | `top_class:format_or_schema` | 8 | 0.177 | 0.177 | 0.073 | -0.007 | -0.063 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `other_record_value` | 8 | 0.172 | 0.172 | 0.057 | 0.013 | -0.162 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `generic_answer` | 8 | 0.167 | 0.167 | 0.057 | 0.010 | -0.162 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `top_non_target` | 8 | 0.161 | 0.161 | 0.073 | 0.003 | -0.063 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `format_schema` | 8 | 0.151 | 0.151 | 0.057 | -0.018 | -0.162 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `contrast_answer` | 8 | 0.146 | 0.146 | 0.057 | 0.041 | -0.162 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `contrast_answer` | 8 | 0.146 | 0.146 | 0.073 | 0.022 | -0.063 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `top_class:punctuation_or_stop` | 8 | 0.146 | 0.146 | 0.057 | -0.024 | -0.162 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| qwen3 | `primary_multisite_all` | `top_class:format_or_schema` | 8 | 0.141 | 0.141 | 0.057 | -0.009 | -0.072 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| qwen3 | `primary_plus_late_all` | `object_relation_echo` | 8 | 0.135 | 0.141 | 0.057 | -0.007 | -0.162 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| qwen3 | `primary_multisite_all` | `generic_answer` | 8 | 0.135 | 0.135 | 0.057 | 0.005 | -0.072 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `format_schema` | 8 | 0.135 | 0.135 | 0.073 | -0.004 | -0.063 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `other_record_value` | 8 | 0.130 | 0.130 | 0.073 | 0.007 | -0.063 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `top_class:other_semantic_value` | 8 | 0.125 | 0.125 | 0.083 | 0.227 | -0.231 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_multisite_all` | `top_non_target` | 8 | 0.125 | 0.125 | 0.057 | 0.020 | -0.072 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_multisite_all` | `other_record_value` | 8 | 0.125 | 0.125 | 0.057 | 0.004 | -0.072 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `top_class:echo_object_or_relation` | 8 | 0.125 | 0.125 | 0.000 | -0.005 | 0.005 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 3, 'weak_or_unclear': 4}` |
| qwen3 | `single_late_candidate_site` | `top_class:echo_object_or_relation` | 16 | 0.125 | 0.125 | 0.000 | -0.013 | 0.018 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 4, 'weak_or_unclear': 11}` |
| qwen3 | `primary_plus_late_all` | `top_class:echo_object_or_relation` | 8 | 0.125 | 0.125 | 0.000 | -0.016 | 0.042 | `{'route_release_amplifier_or_nonclosure': 3, 'weak_or_unclear': 5}` |
| qwen3 | `single_primary_site` | `top_class:format_or_schema` | 16 | 0.122 | 0.122 | 0.060 | -0.001 | -0.022 | `{'weak_or_unclear': 16}` |
| qwen3 | `late_candidate_all` | `generic_answer` | 8 | 0.120 | 0.120 | 0.073 | 0.003 | -0.063 | `{'weak_or_unclear': 8}` |
| qwen3 | `single_primary_site` | `top_non_target` | 16 | 0.120 | 0.120 | 0.060 | 0.010 | -0.022 | `{'weak_or_unclear': 16}` |
| qwen3 | `single_late_candidate_site` | `top_class:format_or_schema` | 16 | 0.117 | 0.117 | 0.083 | -0.012 | -0.019 | `{'weak_or_unclear': 16}` |
| qwen3 | `primary_multisite_all` | `top_class:other_semantic_value` | 8 | 0.117 | 0.117 | 0.083 | 0.092 | -0.097 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `object_relation_echo` | 8 | 0.115 | 0.120 | 0.073 | 0.002 | -0.063 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_late_candidate_pair` | `top_class:punctuation_or_stop` | 16 | 0.062 | 0.062 | 0.000 | 0.009 | -0.104 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 15}` |
| glm4 | `same_layer_primary_pair` | `top_class:punctuation_or_stop` | 16 | 0.052 | 0.052 | 0.000 | 0.010 | -0.018 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 15}` |
| glm4 | `single_primary_site` | `top_class:punctuation_or_stop` | 24 | 0.049 | 0.049 | 0.000 | 0.006 | -0.010 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 23}` |
| glm4 | `primary_plus_late_all` | `top_class:other_vocab` | 8 | 0.042 | 0.042 | 0.000 | 0.013 | -0.046 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `top_class:punctuation_or_stop` | 8 | 0.042 | 0.042 | 0.000 | 0.009 | -0.141 | `{'weak_or_unclear': 8}` |
| glm4 | `late_candidate_all` | `top_class:punctuation_or_stop` | 8 | 0.042 | 0.042 | 0.000 | 0.008 | -0.141 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_late_candidate_pair` | `top_class:other_vocab` | 16 | 0.042 | 0.042 | 0.007 | 0.005 | -0.032 | `{'weak_or_unclear': 16}` |
| glm4 | `late_candidate_all` | `top_class:other_vocab` | 8 | 0.042 | 0.042 | 0.014 | 0.005 | -0.036 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_multisite_all` | `top_class:echo_object_or_relation` | 8 | 0.036 | 0.036 | 0.000 | 0.008 | -0.016 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `top_class:echo_object_or_relation` | 8 | 0.036 | 0.036 | 0.000 | -0.001 | -0.067 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `top_class:other_semantic_value` | 8 | 0.033 | 0.033 | 0.000 | 0.071 | -0.063 | `{'weak_or_unclear': 8}` |
| glm4 | `late_candidate_all` | `top_class:other_semantic_value` | 8 | 0.033 | 0.033 | 0.017 | 0.060 | -0.053 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_late_candidate_pair` | `top_class:other_semantic_value` | 16 | 0.033 | 0.033 | 0.008 | 0.041 | -0.041 | `{'weak_or_unclear': 16}` |
| glm4 | `primary_multisite_all` | `top_class:other_vocab` | 8 | 0.028 | 0.028 | 0.000 | 0.014 | -0.015 | `{'weak_or_unclear': 8}` |
| glm4 | `late_candidate_all` | `top_class:echo_object_or_relation` | 8 | 0.027 | 0.036 | 0.018 | -0.001 | -0.054 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `top_non_target` | 8 | 0.026 | 0.026 | 0.000 | 0.045 | -0.038 | `{'weak_or_unclear': 8}` |
| glm4 | `late_candidate_all` | `top_non_target` | 8 | 0.026 | 0.026 | 0.010 | 0.035 | -0.029 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_late_candidate_pair` | `top_non_target` | 16 | 0.026 | 0.026 | 0.005 | 0.029 | -0.028 | `{'weak_or_unclear': 16}` |
| glm4 | `same_layer_late_candidate_pair` | `top_class:echo_object_or_relation` | 16 | 0.022 | 0.031 | 0.009 | 0.001 | -0.041 | `{'weak_or_unclear': 16}` |
| glm4 | `same_layer_primary_pair` | `top_class:echo_object_or_relation` | 16 | 0.022 | 0.022 | 0.000 | 0.009 | -0.016 | `{'weak_or_unclear': 16}` |
| glm4 | `same_layer_primary_pair` | `top_class:other_semantic_value` | 16 | 0.021 | 0.021 | 0.000 | 0.017 | -0.015 | `{'weak_or_unclear': 16}` |
| glm4 | `same_layer_primary_pair` | `top_class:other_vocab` | 16 | 0.021 | 0.021 | 0.000 | 0.008 | -0.016 | `{'weak_or_unclear': 16}` |
| glm4 | `single_primary_site` | `top_class:recipient_answer` | 24 | 0.021 | 0.021 | 0.000 | 0.008 | -0.010 | `{'weak_or_unclear': 24}` |
| glm4 | `primary_multisite_all` | `top_class:recipient_answer` | 8 | 0.021 | 0.021 | 0.000 | 0.005 | -0.001 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_multisite_all` | `object_relation_echo` | 8 | 0.021 | 0.021 | 0.000 | 0.005 | -0.011 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_late_candidate_pair` | `top_class:recipient_answer` | 16 | 0.021 | 0.021 | 0.000 | 0.004 | -0.013 | `{'weak_or_unclear': 16}` |
| glm4 | `primary_multisite_all` | `generic_answer` | 8 | 0.021 | 0.021 | 0.000 | 0.003 | -0.011 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_primary_pair` | `top_class:recipient_answer` | 16 | 0.021 | 0.021 | 0.000 | 0.003 | -0.004 | `{'weak_or_unclear': 16}` |
| glm4 | `primary_plus_late_all` | `object_relation_echo` | 8 | 0.021 | 0.021 | 0.000 | 0.000 | -0.038 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `top_class:recipient_answer` | 8 | 0.021 | 0.021 | 0.000 | -0.001 | -0.009 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `generic_answer` | 8 | 0.021 | 0.021 | 0.000 | -0.001 | -0.038 | `{'weak_or_unclear': 8}` |
| glm4 | `late_candidate_all` | `generic_answer` | 8 | 0.021 | 0.021 | 0.010 | -0.003 | -0.029 | `{'weak_or_unclear': 8}` |
| deepseek7b | `true_late_control` | `top_class:recipient_answer` | 8 | 0.438 | 0.438 | 0.188 | 0.012 | -0.016 | `{'joint_rewrite_suppressor_candidate': 1, 'route_only_suppressor_candidate': 4, 'route_release_amplifier_or_nonclosure': 1, 'target_rewrite_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `primary_multisite_all` | `top_class:recipient_answer` | 8 | 0.375 | 0.375 | 0.188 | -0.027 | 0.043 | `{'route_only_suppressor_candidate': 3, 'route_release_amplifier_or_nonclosure': 3, 'target_rewrite_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `same_layer_late_candidate_pair` | `top_class:recipient_answer` | 16 | 0.312 | 0.312 | 0.156 | -0.033 | 0.039 | `{'joint_rewrite_suppressor_candidate': 1, 'route_only_suppressor_candidate': 6, 'route_release_amplifier_or_nonclosure': 6, 'target_rewrite_candidate': 2, 'weak_or_unclear': 1}` |
| deepseek7b | `late_candidate_all` | `top_class:recipient_answer` | 8 | 0.312 | 0.312 | 0.188 | -0.062 | 0.062 | `{'route_only_suppressor_candidate': 3, 'route_release_amplifier_or_nonclosure': 2, 'target_rewrite_candidate': 2, 'weak_or_unclear': 1}` |
| deepseek7b | `single_primary_site` | `top_class:recipient_answer` | 16 | 0.281 | 0.281 | 0.188 | -0.020 | 0.020 | `{'joint_rewrite_suppressor_candidate': 1, 'route_only_suppressor_candidate': 6, 'route_release_amplifier_or_nonclosure': 5, 'target_rewrite_candidate': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `same_layer_primary_pair` | `top_class:recipient_answer` | 16 | 0.250 | 0.250 | 0.125 | -0.041 | 0.037 | `{'route_only_suppressor_candidate': 2, 'route_release_amplifier_or_nonclosure': 6, 'target_rewrite_candidate': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `primary_plus_late_all` | `top_class:recipient_answer` | 8 | 0.250 | 0.250 | 0.188 | -0.078 | 0.086 | `{'route_only_suppressor_candidate': 3, 'route_release_amplifier_or_nonclosure': 2, 'target_rewrite_candidate': 2, 'weak_or_unclear': 1}` |
| deepseek7b | `true_late_control` | `top_non_target` | 8 | 0.177 | 0.188 | 0.214 | -0.028 | 0.023 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `late_candidate_all` | `top_non_target` | 8 | 0.161 | 0.198 | 0.286 | -0.002 | 0.098 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `primary_multisite_all` | `top_non_target` | 8 | 0.161 | 0.172 | 0.219 | -0.007 | 0.035 | `{'target_rewrite_candidate': 1, 'weak_or_unclear': 7}` |
| deepseek7b | `true_late_control` | `top_class:format_or_schema` | 8 | 0.156 | 0.167 | 0.214 | -0.027 | 0.023 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `true_late_control` | `top_class:other_vocab` | 8 | 0.151 | 0.151 | 0.214 | -0.042 | 0.023 | `{'route_release_amplifier_or_nonclosure': 3, 'weak_or_unclear': 5}` |
| deepseek7b | `same_layer_primary_pair` | `top_non_target` | 16 | 0.146 | 0.151 | 0.141 | -0.008 | 0.016 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 15}` |
| deepseek7b | `primary_multisite_all` | `top_class:other_vocab` | 8 | 0.146 | 0.146 | 0.219 | 0.022 | 0.035 | `{'target_rewrite_candidate': 1, 'weak_or_unclear': 7}` |
| deepseek7b | `primary_plus_late_all` | `top_class:echo_object_or_relation` | 8 | 0.146 | 0.174 | 0.347 | -0.013 | 0.127 | `{'joint_rewrite_suppressor_candidate': 1, 'target_rewrite_candidate': 3, 'weak_or_unclear': 4}` |
| deepseek7b | `primary_plus_late_all` | `top_non_target` | 8 | 0.141 | 0.193 | 0.375 | -0.025 | 0.142 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `same_layer_late_candidate_pair` | `top_non_target` | 16 | 0.141 | 0.156 | 0.224 | -0.004 | 0.049 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 12}` |
| deepseek7b | `true_late_control` | `top_class:punctuation_or_stop` | 8 | 0.141 | 0.151 | 0.214 | -0.038 | 0.023 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `primary_multisite_all` | `generic_answer` | 8 | 0.141 | 0.146 | 0.219 | 0.000 | 0.035 | `{'target_rewrite_candidate': 1, 'weak_or_unclear': 7}` |
| deepseek7b | `same_layer_primary_pair` | `top_class:other_vocab` | 16 | 0.141 | 0.141 | 0.141 | 0.007 | 0.016 | `{'weak_or_unclear': 16}` |
| deepseek7b | `true_late_control` | `generic_answer` | 8 | 0.141 | 0.141 | 0.214 | -0.051 | 0.023 | `{'route_release_amplifier_or_nonclosure': 3, 'weak_or_unclear': 5}` |
| deepseek7b | `primary_multisite_all` | `top_class:echo_object_or_relation` | 8 | 0.139 | 0.146 | 0.167 | 0.040 | 0.008 | `{'weak_or_unclear': 8}` |
| deepseek7b | `same_layer_primary_pair` | `top_class:echo_object_or_relation` | 16 | 0.139 | 0.142 | 0.111 | 0.026 | -0.002 | `{'weak_or_unclear': 16}` |
| deepseek7b | `true_late_control` | `top_class:echo_object_or_relation` | 8 | 0.139 | 0.139 | 0.167 | -0.040 | -0.007 | `{'route_release_amplifier_or_nonclosure': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `late_candidate_all` | `top_class:format_or_schema` | 8 | 0.135 | 0.151 | 0.286 | -0.029 | 0.098 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `same_layer_primary_pair` | `object_relation_echo` | 16 | 0.135 | 0.143 | 0.141 | 0.013 | 0.016 | `{'weak_or_unclear': 16}` |
| deepseek7b | `single_primary_site` | `generic_answer` | 16 | 0.135 | 0.143 | 0.143 | 0.008 | 0.009 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 14}` |
| deepseek7b | `late_candidate_all` | `top_class:other_vocab` | 8 | 0.135 | 0.135 | 0.286 | -0.022 | 0.098 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `same_layer_primary_pair` | `generic_answer` | 16 | 0.133 | 0.133 | 0.141 | -0.002 | 0.016 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 14}` |
| deepseek7b | `late_candidate_all` | `top_class:echo_object_or_relation` | 8 | 0.132 | 0.167 | 0.257 | -0.004 | 0.086 | `{'joint_rewrite_suppressor_candidate': 1, 'target_rewrite_candidate': 3, 'weak_or_unclear': 4}` |
| deepseek7b | `primary_plus_late_all` | `top_class:format_or_schema` | 8 | 0.130 | 0.156 | 0.375 | -0.056 | 0.142 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `primary_multisite_all` | `top_class:format_or_schema` | 8 | 0.130 | 0.135 | 0.219 | -0.017 | 0.035 | `{'target_rewrite_candidate': 1, 'weak_or_unclear': 7}` |

## Top Route Suppression Cells

| model | writer | source | combo | route | n | route-only | route rate | target rate | route reduced | recovered | role |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out:H15 | records_all | `primary_plus_late_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.250 | -0.250 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `primary_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `late_candidate_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `L35:mlp_out` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | target_record_line | `primary_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.250 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | target_record_line | `late_candidate_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.250 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.375 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | target_record_line | `L34:attn_out` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | target_record_line | `L35:mlp_out` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | target_record_line | `primary_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.250 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | target_record_line | `late_candidate_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.250 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | target_record_line | `L34:attn_out` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | target_record_line | `L35:attn_out` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | target_record_line | `L35:mlp_out` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `primary_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `late_candidate_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `primary_plus_late_all` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `L34:mlp_out` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `L35:mlp_out` | `top_class:recipient_answer` | 1 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `primary_plus_late_all` | `top_class:format_or_schema` | 24 | 0.458 | 0.458 | 0.042 | 0.057 | -0.073 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `primary_plus_late_all` | `top_non_target` | 24 | 0.375 | 0.375 | 0.042 | 0.042 | -0.073 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | records_all | `late_candidate_all` | `top_class:echo_object_or_relation` | 3 | 0.333 | 0.333 | 0.000 | 0.083 | -0.042 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `L35:attn_out` | `top_class:echo_object_or_relation` | 3 | 0.333 | 0.333 | 0.000 | 0.083 | 0.000 | `route_only_suppressor_candidate` |
| glm4 | L34:attn_out:H4 | records_all | `L37:attn+mlp` | `top_class:punctuation_or_stop` | 6 | 0.500 | 0.500 | 0.000 | 0.052 | -0.031 | `route_only_suppressor_candidate` |
| glm4 | L34:attn_out:H4 | records_all | `L38:late_attn+mlp` | `top_class:punctuation_or_stop` | 6 | 0.500 | 0.500 | 0.000 | 0.052 | -0.115 | `route_only_suppressor_candidate` |
| glm4 | L34:attn_out:H4 | records_all | `L36:mlp_out` | `top_class:punctuation_or_stop` | 6 | 0.500 | 0.500 | 0.000 | 0.052 | -0.031 | `route_only_suppressor_candidate` |
| glm4 | L34:attn_out:H4 | records_all | `L39:late_attn+mlp` | `top_class:punctuation_or_stop` | 6 | 0.500 | 0.500 | 0.000 | 0.042 | -0.750 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `late_candidate_all` | `top_class:punctuation_or_stop` | 6 | 0.333 | 0.333 | 0.000 | 0.042 | -0.625 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `primary_plus_late_all` | `top_class:punctuation_or_stop` | 6 | 0.333 | 0.333 | 0.000 | 0.042 | -0.625 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L37:attn_out` | `top_class:punctuation_or_stop` | 6 | 0.333 | 0.333 | 0.000 | 0.042 | 0.000 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L36:attn+mlp` | `top_class:punctuation_or_stop` | 6 | 0.333 | 0.333 | 0.000 | 0.031 | -0.031 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L36:attn_out` | `top_class:punctuation_or_stop` | 6 | 0.333 | 0.333 | 0.000 | 0.021 | -0.010 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L39:late_attn+mlp` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.031 | -0.021 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L36:mlp_out` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.031 | 0.010 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.021 | 0.010 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L37:attn+mlp` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.021 | 0.021 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L38:late_attn+mlp` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.021 | 0.010 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L37:attn_out` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.021 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `primary_all` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.010 | 0.021 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `late_candidate_all` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.010 | 0.021 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L36:attn+mlp` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.010 | 0.010 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L36:attn_out` | `top_class:recipient_answer` | 6 | 0.167 | 0.167 | 0.000 | 0.010 | 0.021 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `primary_all` | `top_class:punctuation_or_stop` | 6 | 0.167 | 0.167 | 0.000 | 0.010 | -0.010 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | target_record_line | `L37:attn+mlp` | `top_class:echo_object_or_relation` | 14 | 0.143 | 0.143 | 0.000 | 0.031 | -0.040 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | target_record_line | `primary_all` | `top_class:echo_object_or_relation` | 14 | 0.143 | 0.143 | 0.000 | 0.027 | -0.049 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | target_record_line | `primary_plus_late_all` | `top_class:echo_object_or_relation` | 14 | 0.143 | 0.143 | 0.000 | 0.018 | -0.308 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `primary_all` | `top_class:echo_object_or_relation` | 14 | 0.143 | 0.143 | 0.000 | 0.013 | -0.049 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `late_candidate_all` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.219 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `L25:late_attn+mlp` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.188 | -0.156 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `L23:attn+mlp` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.188 | -0.156 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `primary_all` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.156 | -0.094 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `primary_all` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.156 | -0.156 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `late_control_same_count` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.156 | -0.094 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `L26:late_attn+mlp` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.156 | -0.188 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `late_control_same_count` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.125 | -0.219 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.125 | -0.156 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `L23:attn_out` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.094 | -0.094 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `L24:attn+mlp` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.062 | -0.094 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H24 | records_all | `L23:attn_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.500 | 0.188 | -0.062 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H14 | records_all | `L25:late_attn+mlp` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.156 | -0.094 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `late_candidate_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H14 | records_all | `L23:mlp_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.125 | -0.094 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.094 | -0.062 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `L25:late_attn+mlp` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.094 | -0.062 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `L23:mlp_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.094 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H14 | records_all | `late_candidate_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.094 | -0.062 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H14 | records_all | `late_control_same_count` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.094 | -0.062 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H14 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.094 | -0.156 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H14 | target_record_line | `L25:late_attn+mlp` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.094 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `L26:late_attn+mlp` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `L23:attn_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.031 | `route_only_suppressor_candidate` |

## Strict Interpretation

- A route-only cell means restore reduces a route group after source removal while target recovery does not satisfy the target threshold.
- Negative route reduced means the restored component amplifies or fails to close that route release.
- This phase separates route classes explicitly, but still works at component/head level and is not a neuron atlas.
