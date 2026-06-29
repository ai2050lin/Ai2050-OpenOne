# Phase 760 Route Suppression Matrix Atlas (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Route groups: contrast_answer / object_relation_echo / other_record_value / format_schema / generic_answer / top_non_target / dynamic top classes.

## Combo Kind x Route Group

| model | combo kind | route group | groups | route-only | route rate | target rate | route reduced | target recovered | roles |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `late_candidate_all` | `top_class:recipient_answer` | 8 | 0.250 | 0.250 | 0.062 | 0.008 | -0.016 | `{'route_only_suppressor_candidate': 3, 'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 3}` |
| qwen3 | `primary_plus_late_all` | `top_class:recipient_answer` | 8 | 0.250 | 0.250 | 0.062 | 0.008 | -0.055 | `{'route_only_suppressor_candidate': 3, 'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 3}` |
| qwen3 | `primary_multisite_all` | `top_class:recipient_answer` | 8 | 0.250 | 0.250 | 0.062 | -0.031 | 0.000 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 3, 'weak_or_unclear': 4}` |
| qwen3 | `primary_plus_late_all` | `top_class:format_or_schema` | 8 | 0.203 | 0.203 | 0.060 | 0.002 | -0.140 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 6}` |
| qwen3 | `primary_plus_late_all` | `top_non_target` | 8 | 0.190 | 0.190 | 0.060 | 0.015 | -0.140 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 7}` |
| qwen3 | `single_late_candidate_site` | `top_class:recipient_answer` | 16 | 0.188 | 0.188 | 0.031 | 0.012 | -0.008 | `{'route_only_suppressor_candidate': 4, 'route_release_amplifier_or_nonclosure': 2, 'target_rewrite_candidate': 1, 'weak_or_unclear': 9}` |
| qwen3 | `primary_plus_late_all` | `format_schema` | 8 | 0.182 | 0.182 | 0.060 | -0.011 | -0.140 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `top_class:punctuation_or_stop` | 8 | 0.161 | 0.161 | 0.060 | -0.012 | -0.140 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `top_class:other_vocab` | 8 | 0.159 | 0.159 | 0.062 | 0.000 | -0.147 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 7}` |
| qwen3 | `late_candidate_all` | `top_class:format_or_schema` | 8 | 0.156 | 0.156 | 0.073 | -0.012 | -0.033 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `top_class:echo_object_or_relation` | 8 | 0.156 | 0.156 | 0.062 | -0.027 | 0.074 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| qwen3 | `primary_plus_late_all` | `object_relation_echo` | 8 | 0.151 | 0.154 | 0.060 | 0.006 | -0.140 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 7}` |
| qwen3 | `late_candidate_all` | `format_schema` | 8 | 0.151 | 0.151 | 0.073 | -0.006 | -0.033 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `other_record_value` | 8 | 0.148 | 0.148 | 0.060 | 0.006 | -0.140 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_multisite_all` | `format_schema` | 8 | 0.148 | 0.148 | 0.055 | -0.003 | -0.076 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 7}` |
| qwen3 | `late_candidate_all` | `top_non_target` | 8 | 0.148 | 0.148 | 0.073 | -0.008 | -0.033 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_multisite_all` | `top_class:format_or_schema` | 8 | 0.141 | 0.141 | 0.055 | -0.002 | -0.076 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 7}` |
| qwen3 | `primary_plus_late_all` | `contrast_answer` | 8 | 0.138 | 0.138 | 0.060 | 0.034 | -0.140 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_multisite_all` | `top_non_target` | 8 | 0.133 | 0.133 | 0.055 | 0.012 | -0.076 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `top_class:other_vocab` | 8 | 0.131 | 0.131 | 0.072 | -0.008 | -0.028 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `generic_answer` | 8 | 0.130 | 0.130 | 0.060 | 0.004 | -0.140 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_plus_late_all` | `top_class:other_semantic_value` | 8 | 0.125 | 0.125 | 0.081 | 0.176 | -0.181 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `contrast_answer` | 8 | 0.122 | 0.122 | 0.073 | 0.013 | -0.033 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_multisite_all` | `object_relation_echo` | 8 | 0.122 | 0.128 | 0.055 | -0.003 | -0.076 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 7}` |
| qwen3 | `late_candidate_all` | `top_class:punctuation_or_stop` | 8 | 0.122 | 0.122 | 0.073 | -0.003 | -0.033 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `object_relation_echo` | 8 | 0.117 | 0.120 | 0.073 | 0.000 | -0.033 | `{'weak_or_unclear': 8}` |
| qwen3 | `late_candidate_all` | `other_record_value` | 8 | 0.117 | 0.117 | 0.073 | 0.003 | -0.033 | `{'weak_or_unclear': 8}` |
| qwen3 | `single_primary_site` | `format_schema` | 16 | 0.117 | 0.117 | 0.053 | -0.002 | -0.029 | `{'weak_or_unclear': 16}` |
| qwen3 | `primary_multisite_all` | `top_class:other_vocab` | 8 | 0.116 | 0.116 | 0.059 | 0.009 | -0.086 | `{'weak_or_unclear': 8}` |
| qwen3 | `primary_multisite_all` | `top_class:punctuation_or_stop` | 8 | 0.115 | 0.115 | 0.055 | -0.015 | -0.076 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| qwen3 | `single_primary_site` | `top_class:format_or_schema` | 16 | 0.113 | 0.113 | 0.053 | -0.001 | -0.029 | `{'weak_or_unclear': 16}` |
| qwen3 | `primary_multisite_all` | `top_class:other_semantic_value` | 8 | 0.113 | 0.113 | 0.073 | 0.080 | -0.093 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `top_class:other_vocab` | 8 | 0.019 | 0.019 | 0.000 | 0.010 | -0.022 | `{'weak_or_unclear': 8}` |
| glm4 | `late_candidate_all` | `top_class:other_vocab` | 8 | 0.019 | 0.019 | 0.013 | 0.003 | -0.012 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_late_candidate_pair` | `top_class:other_vocab` | 16 | 0.019 | 0.019 | 0.003 | 0.002 | -0.014 | `{'weak_or_unclear': 16}` |
| glm4 | `primary_plus_late_all` | `top_class:other_semantic_value` | 8 | 0.019 | 0.019 | 0.000 | 0.029 | -0.024 | `{'weak_or_unclear': 8}` |
| glm4 | `late_candidate_all` | `top_class:other_semantic_value` | 8 | 0.019 | 0.019 | 0.015 | 0.016 | -0.012 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_late_candidate_pair` | `top_class:other_semantic_value` | 16 | 0.019 | 0.019 | 0.004 | 0.013 | -0.014 | `{'weak_or_unclear': 16}` |
| glm4 | `primary_plus_late_all` | `top_non_target` | 8 | 0.016 | 0.016 | 0.000 | 0.022 | -0.020 | `{'weak_or_unclear': 8}` |
| glm4 | `late_candidate_all` | `top_non_target` | 8 | 0.016 | 0.016 | 0.010 | 0.013 | -0.011 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_late_candidate_pair` | `top_non_target` | 16 | 0.016 | 0.016 | 0.003 | 0.012 | -0.012 | `{'weak_or_unclear': 16}` |
| glm4 | `primary_multisite_all` | `top_class:echo_object_or_relation` | 8 | 0.014 | 0.014 | 0.000 | 0.007 | -0.014 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `top_class:echo_object_or_relation` | 8 | 0.014 | 0.014 | 0.000 | -0.001 | -0.026 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_late_candidate_pair` | `top_class:punctuation_or_stop` | 16 | 0.013 | 0.013 | 0.004 | 0.001 | -0.019 | `{'weak_or_unclear': 16}` |
| glm4 | `primary_multisite_all` | `top_class:other_vocab` | 8 | 0.013 | 0.013 | 0.000 | 0.014 | -0.014 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_primary_pair` | `top_class:echo_object_or_relation` | 16 | 0.012 | 0.012 | 0.000 | 0.006 | -0.011 | `{'weak_or_unclear': 16}` |
| glm4 | `single_primary_site` | `top_class:echo_object_or_relation` | 24 | 0.012 | 0.012 | 0.001 | 0.003 | -0.007 | `{'weak_or_unclear': 24}` |
| glm4 | `same_layer_primary_pair` | `top_class:punctuation_or_stop` | 16 | 0.011 | 0.011 | 0.000 | 0.004 | -0.011 | `{'weak_or_unclear': 16}` |
| glm4 | `late_candidate_all` | `top_class:echo_object_or_relation` | 8 | 0.011 | 0.014 | 0.014 | -0.003 | -0.012 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_multisite_all` | `object_relation_echo` | 8 | 0.010 | 0.010 | 0.000 | 0.006 | -0.013 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `object_relation_echo` | 8 | 0.010 | 0.010 | 0.000 | 0.000 | -0.020 | `{'weak_or_unclear': 8}` |
| glm4 | `same_layer_primary_pair` | `top_class:other_vocab` | 16 | 0.010 | 0.010 | 0.000 | 0.008 | -0.012 | `{'weak_or_unclear': 16}` |
| glm4 | `same_layer_primary_pair` | `top_class:other_semantic_value` | 16 | 0.009 | 0.009 | 0.000 | 0.010 | -0.011 | `{'weak_or_unclear': 16}` |
| glm4 | `same_layer_primary_pair` | `top_non_target` | 16 | 0.009 | 0.009 | 0.000 | 0.007 | -0.010 | `{'weak_or_unclear': 16}` |
| glm4 | `same_layer_primary_pair` | `object_relation_echo` | 16 | 0.009 | 0.009 | 0.000 | 0.005 | -0.010 | `{'weak_or_unclear': 16}` |
| glm4 | `same_layer_late_candidate_pair` | `top_class:echo_object_or_relation` | 16 | 0.009 | 0.012 | 0.004 | 0.001 | -0.014 | `{'weak_or_unclear': 16}` |
| glm4 | `single_primary_site` | `top_class:punctuation_or_stop` | 24 | 0.009 | 0.009 | 0.001 | 0.003 | -0.004 | `{'weak_or_unclear': 24}` |
| glm4 | `late_candidate_all` | `top_class:punctuation_or_stop` | 8 | 0.009 | 0.009 | 0.009 | 0.001 | -0.017 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `top_class:punctuation_or_stop` | 8 | 0.009 | 0.009 | 0.000 | 0.001 | -0.025 | `{'weak_or_unclear': 8}` |
| glm4 | `single_primary_site` | `object_relation_echo` | 24 | 0.009 | 0.009 | 0.001 | 0.002 | -0.006 | `{'weak_or_unclear': 24}` |
| glm4 | `single_primary_site` | `top_class:other_vocab` | 24 | 0.009 | 0.009 | 0.001 | 0.005 | -0.006 | `{'weak_or_unclear': 24}` |
| glm4 | `late_candidate_all` | `object_relation_echo` | 8 | 0.008 | 0.010 | 0.010 | -0.001 | -0.011 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_multisite_all` | `top_non_target` | 8 | 0.008 | 0.008 | 0.000 | 0.011 | -0.013 | `{'weak_or_unclear': 8}` |
| glm4 | `primary_plus_late_all` | `top_class:recipient_answer` | 8 | 0.008 | 0.008 | 0.000 | 0.005 | -0.005 | `{'weak_or_unclear': 8}` |
| deepseek7b | `true_late_control` | `top_class:recipient_answer` | 8 | 0.194 | 0.194 | 0.181 | -0.052 | 0.050 | `{'target_rewrite_candidate': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `true_late_control` | `top_class:format_or_schema` | 8 | 0.190 | 0.198 | 0.206 | -0.024 | 0.022 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `true_late_control` | `top_non_target` | 8 | 0.190 | 0.198 | 0.206 | -0.029 | 0.022 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `primary_multisite_all` | `top_class:recipient_answer` | 8 | 0.181 | 0.181 | 0.153 | -0.035 | 0.046 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 1, 'target_rewrite_candidate': 2, 'weak_or_unclear': 4}` |
| deepseek7b | `true_late_control` | `generic_answer` | 8 | 0.174 | 0.180 | 0.206 | -0.044 | 0.022 | `{'route_release_amplifier_or_nonclosure': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `true_late_control` | `top_class:other_vocab` | 8 | 0.173 | 0.178 | 0.207 | -0.043 | 0.023 | `{'route_release_amplifier_or_nonclosure': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `true_late_control` | `top_class:other_semantic_value` | 8 | 0.170 | 0.170 | 0.241 | -0.003 | 0.032 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 7}` |
| deepseek7b | `late_candidate_all` | `top_class:recipient_answer` | 8 | 0.153 | 0.153 | 0.208 | -0.088 | 0.097 | `{'route_release_amplifier_or_nonclosure': 1, 'target_rewrite_candidate': 2, 'weak_or_unclear': 5}` |
| deepseek7b | `primary_multisite_all` | `generic_answer` | 8 | 0.146 | 0.151 | 0.250 | -0.015 | 0.052 | `{'target_rewrite_candidate': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `true_late_control` | `other_record_value` | 8 | 0.146 | 0.151 | 0.206 | -0.046 | 0.022 | `{'route_release_amplifier_or_nonclosure': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `same_layer_late_candidate_pair` | `top_class:recipient_answer` | 16 | 0.146 | 0.146 | 0.188 | -0.059 | 0.068 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 3, 'target_rewrite_candidate': 4, 'weak_or_unclear': 8}` |
| deepseek7b | `true_late_control` | `top_class:echo_object_or_relation` | 8 | 0.144 | 0.151 | 0.183 | -0.030 | 0.005 | `{'route_release_amplifier_or_nonclosure': 3, 'weak_or_unclear': 5}` |
| deepseek7b | `primary_plus_late_all` | `top_non_target` | 8 | 0.143 | 0.182 | 0.385 | -0.035 | 0.145 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `late_candidate_all` | `top_non_target` | 8 | 0.143 | 0.174 | 0.315 | -0.019 | 0.104 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `true_late_control` | `top_class:punctuation_or_stop` | 8 | 0.143 | 0.154 | 0.206 | -0.037 | 0.022 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `same_layer_primary_pair` | `generic_answer` | 16 | 0.143 | 0.146 | 0.159 | -0.013 | 0.027 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 15}` |
| deepseek7b | `primary_plus_late_all` | `top_class:other_semantic_value` | 8 | 0.143 | 0.170 | 0.473 | -0.076 | 0.166 | `{'route_release_amplifier_or_nonclosure': 1, 'target_rewrite_candidate': 4, 'weak_or_unclear': 3}` |
| deepseek7b | `single_primary_site` | `generic_answer` | 16 | 0.142 | 0.147 | 0.156 | -0.000 | 0.017 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 14}` |
| deepseek7b | `primary_multisite_all` | `top_non_target` | 8 | 0.141 | 0.159 | 0.250 | -0.019 | 0.052 | `{'target_rewrite_candidate': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `true_late_control` | `format_schema` | 8 | 0.141 | 0.154 | 0.206 | -0.027 | 0.022 | `{'route_release_amplifier_or_nonclosure': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `primary_plus_late_all` | `top_class:recipient_answer` | 8 | 0.139 | 0.139 | 0.208 | -0.079 | 0.097 | `{'route_release_amplifier_or_nonclosure': 1, 'target_rewrite_candidate': 2, 'weak_or_unclear': 5}` |
| deepseek7b | `primary_multisite_all` | `top_class:other_vocab` | 8 | 0.138 | 0.152 | 0.253 | 0.006 | 0.054 | `{'target_rewrite_candidate': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `primary_plus_late_all` | `top_class:format_or_schema` | 8 | 0.138 | 0.161 | 0.385 | -0.062 | 0.145 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `late_candidate_all` | `top_class:format_or_schema` | 8 | 0.138 | 0.159 | 0.315 | -0.043 | 0.104 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `same_layer_late_candidate_pair` | `top_class:other_vocab` | 16 | 0.137 | 0.145 | 0.242 | -0.024 | 0.057 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 12}` |
| deepseek7b | `same_layer_late_candidate_pair` | `generic_answer` | 16 | 0.137 | 0.143 | 0.240 | -0.031 | 0.056 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 12}` |
| deepseek7b | `same_layer_primary_pair` | `top_class:other_vocab` | 16 | 0.136 | 0.140 | 0.161 | -0.006 | 0.027 | `{'weak_or_unclear': 16}` |
| deepseek7b | `late_candidate_all` | `generic_answer` | 8 | 0.135 | 0.154 | 0.315 | -0.054 | 0.104 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `same_layer_primary_pair` | `object_relation_echo` | 16 | 0.134 | 0.150 | 0.159 | 0.001 | 0.027 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 15}` |
| deepseek7b | `late_candidate_all` | `top_class:other_semantic_value` | 8 | 0.134 | 0.179 | 0.366 | -0.013 | 0.099 | `{'target_rewrite_candidate': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `same_layer_primary_pair` | `top_class:echo_object_or_relation` | 16 | 0.133 | 0.147 | 0.149 | 0.010 | 0.020 | `{'weak_or_unclear': 16}` |
| deepseek7b | `single_primary_site` | `top_class:recipient_answer` | 16 | 0.132 | 0.132 | 0.132 | -0.027 | 0.033 | `{'route_release_amplifier_or_nonclosure': 4, 'target_rewrite_candidate': 1, 'weak_or_unclear': 11}` |

## Top Route Suppression Cells

| model | writer | source | combo | route | n | route-only | route rate | target rate | route reduced | recovered | role |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out:H15 | target_record_line | `L35:mlp_out` | `top_class:recipient_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.125 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `primary_plus_late_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.125 | -0.188 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `late_candidate_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.062 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `L35:mlp_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.125 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | target_record_line | `late_candidate_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.188 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.312 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | target_record_line | `L34:attn_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.062 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | target_record_line | `L34:attn_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.062 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | target_record_line | `L35:attn_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.062 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `L35:mlp_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.062 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `primary_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.500 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `late_candidate_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.500 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `primary_plus_late_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.500 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `L34:mlp_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.500 | 0.062 | 0.062 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `primary_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | -0.062 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | target_record_line | `primary_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | -0.188 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | target_record_line | `primary_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | -0.062 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | target_record_line | `late_candidate_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | -0.125 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | target_record_line | `L35:mlp_out` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | records_all | `primary_plus_late_all` | `top_class:format_or_schema` | 48 | 0.375 | 0.375 | 0.021 | 0.086 | -0.091 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `primary_plus_late_all` | `top_class:echo_object_or_relation` | 8 | 0.375 | 0.375 | 0.000 | 0.000 | 0.016 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | records_all | `primary_plus_late_all` | `top_non_target` | 48 | 0.333 | 0.333 | 0.021 | 0.081 | -0.091 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H4 | records_all | `late_candidate_all` | `other_record_value` | 48 | 0.312 | 0.312 | 0.188 | 0.048 | 0.036 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L39:late_attn+mlp` | `top_class:other_semantic_value` | 33 | 0.091 | 0.091 | 0.000 | 0.138 | -0.142 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `primary_plus_late_all` | `top_class:other_semantic_value` | 33 | 0.091 | 0.091 | 0.000 | 0.127 | -0.121 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `late_candidate_all` | `top_class:other_semantic_value` | 33 | 0.091 | 0.091 | 0.030 | 0.097 | -0.093 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L38:late_attn+mlp` | `top_class:other_semantic_value` | 33 | 0.091 | 0.091 | 0.030 | 0.008 | -0.011 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L39:late_attn+mlp` | `top_class:punctuation_or_stop` | 28 | 0.071 | 0.071 | 0.000 | 0.013 | -0.170 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L36:mlp_out` | `top_class:punctuation_or_stop` | 28 | 0.071 | 0.071 | 0.000 | 0.013 | -0.027 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L37:attn+mlp` | `top_class:punctuation_or_stop` | 28 | 0.071 | 0.071 | 0.000 | 0.011 | -0.004 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L38:late_attn+mlp` | `top_class:punctuation_or_stop` | 28 | 0.071 | 0.071 | 0.036 | 0.004 | -0.013 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L39:late_attn+mlp` | `top_non_target` | 48 | 0.062 | 0.062 | 0.000 | 0.102 | -0.103 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `primary_plus_late_all` | `top_non_target` | 48 | 0.062 | 0.062 | 0.000 | 0.092 | -0.091 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `late_candidate_all` | `top_non_target` | 48 | 0.062 | 0.062 | 0.021 | 0.070 | -0.069 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L36:mlp_out` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.020 | 0.008 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `primary_all` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.016 | -0.004 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.016 | -0.004 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L37:attn+mlp` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.016 | 0.008 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L38:late_attn+mlp` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.016 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L37:attn_out` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.016 | -0.008 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `late_candidate_all` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.012 | 0.000 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | records_all | `L38:late_attn+mlp` | `top_non_target` | 48 | 0.062 | 0.062 | 0.021 | 0.010 | -0.012 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L36:attn+mlp` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.008 | 0.004 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L39:late_attn+mlp` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.008 | -0.004 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | target_record_line | `L36:attn_out` | `top_class:recipient_answer` | 16 | 0.062 | 0.062 | 0.000 | 0.008 | 0.020 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | target_record_line | `L39:late_attn+mlp` | `top_class:other_semantic_value` | 33 | 0.061 | 0.061 | 0.000 | 0.133 | -0.140 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | target_record_line | `primary_plus_late_all` | `top_class:other_semantic_value` | 33 | 0.061 | 0.061 | 0.000 | 0.117 | -0.131 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `late_candidate_all` | `top_class:other_semantic_value` | 14 | 0.429 | 0.429 | 0.214 | 0.049 | -0.040 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `primary_plus_late_all` | `top_class:other_semantic_value` | 14 | 0.429 | 0.429 | 0.214 | 0.018 | 0.004 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `late_control_same_count` | `top_class:other_semantic_value` | 14 | 0.429 | 0.429 | 0.214 | 0.013 | 0.000 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `L23:attn_out` | `top_class:other_semantic_value` | 14 | 0.357 | 0.357 | 0.143 | 0.049 | -0.062 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `L23:attn+mlp` | `top_class:other_semantic_value` | 14 | 0.357 | 0.357 | 0.143 | 0.018 | -0.013 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `L26:late_attn+mlp` | `top_class:other_semantic_value` | 14 | 0.357 | 0.357 | 0.214 | 0.004 | -0.004 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `L24:attn+mlp` | `top_class:other_semantic_value` | 14 | 0.357 | 0.357 | 0.143 | -0.018 | 0.004 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `primary_all` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.000 | 0.076 | -0.049 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `L24:attn+mlp` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.000 | 0.076 | -0.076 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `L25:late_attn+mlp` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.111 | 0.056 | -0.028 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `late_control_same_count` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.000 | 0.042 | -0.083 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H14 | target_record_line | `primary_all` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.000 | 0.021 | -0.021 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H14 | target_record_line | `primary_plus_late_all` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.000 | 0.014 | -0.042 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H14 | target_record_line | `late_control_same_count` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.000 | 0.014 | -0.014 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H14 | target_record_line | `late_candidate_all` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H14 | target_record_line | `L25:late_attn+mlp` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.000 | 0.000 | 0.007 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H14 | target_record_line | `L23:attn+mlp` | `top_class:recipient_answer` | 9 | 0.333 | 0.333 | 0.000 | -0.007 | -0.007 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H1 | target_record_line | `L24:attn+mlp` | `object_relation_echo` | 48 | 0.312 | 0.333 | 0.208 | 0.027 | 0.029 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `primary_plus_late_all` | `top_non_target` | 48 | 0.312 | 0.312 | 0.104 | 0.016 | 0.009 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `primary_all` | `top_non_target` | 48 | 0.292 | 0.292 | 0.104 | 0.012 | 0.016 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `primary_plus_late_all` | `top_class:format_or_schema` | 48 | 0.292 | 0.292 | 0.104 | 0.004 | 0.009 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `late_control_same_count` | `top_non_target` | 48 | 0.292 | 0.292 | 0.104 | -0.003 | 0.034 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H14 | target_record_line | `late_candidate_all` | `top_class:other_semantic_value` | 14 | 0.286 | 0.286 | 0.071 | 0.004 | 0.000 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | target_record_line | `primary_all` | `top_class:other_semantic_value` | 14 | 0.286 | 0.286 | 0.214 | 0.000 | 0.004 | `weak_or_unclear` |

## Strict Interpretation

- A route-only cell means restore reduces a route group after source removal while target recovery does not satisfy the target threshold.
- Negative route reduced means the restored component amplifies or fails to close that route release.
- This phase separates route classes explicitly, but still works at component/head level and is not a neuron atlas.
