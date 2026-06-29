# Phase 760 Route Suppression Matrix Atlas (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Route groups: contrast_answer / object_relation_echo / other_record_value / format_schema / generic_answer / top_non_target / dynamic top classes.

## Combo Kind x Route Group

| model | combo kind | route group | groups | route-only | route rate | target rate | route reduced | target recovered | roles |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `late_candidate_all` | `top_non_target` | 2 | 0.250 | 0.250 | 0.250 | 0.094 | 0.000 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| qwen3 | `late_candidate_all` | `format_schema` | 2 | 0.250 | 0.250 | 0.250 | 0.031 | 0.000 | `{'joint_rewrite_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| qwen3 | `late_candidate_all` | `top_class:punctuation_or_stop` | 2 | 0.250 | 0.250 | 0.250 | 0.031 | 0.000 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| qwen3 | `single_primary_site` | `top_class:punctuation_or_stop` | 2 | 0.250 | 0.250 | 0.000 | 0.031 | 0.031 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| qwen3 | `late_candidate_all` | `top_class:format_or_schema` | 2 | 0.250 | 0.250 | 0.250 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| qwen3 | `primary_multisite_all` | `format_schema` | 2 | 0.250 | 0.250 | 0.000 | 0.000 | -0.344 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 1}` |
| qwen3 | `primary_multisite_all` | `top_non_target` | 2 | 0.250 | 0.250 | 0.000 | 0.000 | -0.344 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 1}` |
| qwen3 | `primary_plus_late_all` | `format_schema` | 2 | 0.250 | 0.250 | 0.000 | 0.000 | -0.344 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 1}` |
| qwen3 | `primary_plus_late_all` | `top_non_target` | 2 | 0.250 | 0.250 | 0.000 | 0.000 | -0.344 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 1}` |
| qwen3 | `single_primary_site` | `object_relation_echo` | 2 | 0.250 | 0.250 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| qwen3 | `single_primary_site` | `top_non_target` | 2 | 0.250 | 0.250 | 0.000 | 0.000 | 0.031 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 1}` |
| qwen3 | `primary_multisite_all` | `top_class:format_or_schema` | 2 | 0.250 | 0.250 | 0.000 | -0.031 | -0.344 | `{'route_only_suppressor_candidate': 1, 'route_release_amplifier_or_nonclosure': 1}` |
| qwen3 | `primary_plus_late_all` | `top_class:format_or_schema` | 2 | 0.250 | 0.250 | 0.000 | -0.062 | -0.344 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 1}` |
| qwen3 | `single_primary_site` | `top_class:format_or_schema` | 2 | 0.250 | 0.250 | 0.000 | -0.062 | 0.031 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 1}` |
| qwen3 | `late_candidate_all` | `object_relation_echo` | 2 | 0.000 | 0.250 | 0.250 | 0.062 | 0.000 | `{'joint_rewrite_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| qwen3 | `late_candidate_all` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | `{'weak_or_unclear': 2}` |
| qwen3 | `primary_plus_late_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | -0.344 | `{'weak_or_unclear': 2}` |
| qwen3 | `primary_plus_late_all` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | -0.344 | `{'weak_or_unclear': 2}` |
| qwen3 | `single_primary_site` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.031 | `{'weak_or_unclear': 2}` |
| qwen3 | `single_primary_site` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.031 | `{'weak_or_unclear': 2}` |
| qwen3 | `late_candidate_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.250 | 0.031 | 0.000 | `{'weak_or_unclear': 2}` |
| qwen3 | `late_candidate_all` | `other_record_value` | 2 | 0.000 | 0.000 | 0.250 | 0.031 | 0.000 | `{'weak_or_unclear': 2}` |
| qwen3 | `primary_multisite_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | -0.344 | `{'weak_or_unclear': 2}` |
| qwen3 | `primary_multisite_all` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | -0.344 | `{'weak_or_unclear': 2}` |
| qwen3 | `primary_multisite_all` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | -0.344 | `{'weak_or_unclear': 2}` |
| qwen3 | `primary_multisite_all` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | `{'weak_or_unclear': 2}` |
| qwen3 | `primary_plus_late_all` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | `{'weak_or_unclear': 2}` |
| qwen3 | `single_primary_site` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 | `{'weak_or_unclear': 2}` |
| qwen3 | `primary_plus_late_all` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | -0.008 | -0.344 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 1}` |
| qwen3 | `primary_multisite_all` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | -0.016 | -0.344 | `{'weak_or_unclear': 2}` |
| qwen3 | `single_primary_site` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | -0.016 | 0.031 | `{'weak_or_unclear': 2}` |
| qwen3 | `late_candidate_all` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.250 | -0.023 | 0.000 | `{'route_release_amplifier_or_nonclosure': 1, 'weak_or_unclear': 1}` |
| glm4 | `primary_plus_late_all` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_primary_pair` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `late_candidate_all` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_multisite_all` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_multisite_all` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_plus_late_all` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_primary_pair` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `late_candidate_all` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.016 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_plus_late_all` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.002 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `late_candidate_all` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `late_candidate_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `late_candidate_all` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `late_candidate_all` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `late_candidate_all` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_multisite_all` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_multisite_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_multisite_all` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_multisite_all` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_plus_late_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_plus_late_all` | `object_relation_echo` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_plus_late_all` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_plus_late_all` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_plus_late_all` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_primary_pair` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_primary_pair` | `object_relation_echo` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_primary_pair` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_primary_pair` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_primary_pair` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `same_layer_primary_pair` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | -0.002 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `late_candidate_all` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | -0.008 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_multisite_all` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | -0.008 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `primary_plus_late_all` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | -0.008 | 0.031 | `{'weak_or_unclear': 2}` |
| deepseek7b | `late_candidate_all` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.500 | 0.000 | 0.188 | -0.094 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `late_candidate_all` | `top_class:other_vocab` | 2 | 0.500 | 0.500 | 0.250 | 0.156 | -0.047 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `true_late_control` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.500 | 0.500 | 0.156 | -0.031 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `true_late_control` | `top_non_target` | 2 | 0.500 | 0.500 | 0.500 | 0.141 | 0.016 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `late_candidate_all` | `top_non_target` | 2 | 0.500 | 0.500 | 0.250 | 0.125 | -0.047 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `true_late_control` | `format_schema` | 2 | 0.500 | 0.500 | 0.500 | 0.125 | 0.016 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `true_late_control` | `object_relation_echo` | 2 | 0.500 | 0.500 | 0.500 | 0.125 | 0.016 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `true_late_control` | `top_class:echo_object_or_relation` | 2 | 0.500 | 0.500 | 0.500 | 0.125 | 0.016 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `true_late_control` | `top_class:other_vocab` | 2 | 0.500 | 0.500 | 0.500 | 0.125 | 0.016 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `late_candidate_all` | `generic_answer` | 2 | 0.500 | 0.500 | 0.250 | 0.109 | -0.047 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `late_candidate_all` | `other_record_value` | 2 | 0.500 | 0.500 | 0.250 | 0.109 | -0.047 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `late_candidate_all` | `format_schema` | 2 | 0.500 | 0.500 | 0.250 | 0.094 | -0.047 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `late_candidate_all` | `object_relation_echo` | 2 | 0.500 | 0.500 | 0.250 | 0.094 | -0.047 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `late_candidate_all` | `top_class:echo_object_or_relation` | 2 | 0.500 | 0.500 | 0.250 | 0.094 | -0.047 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `late_candidate_all` | `top_class:format_or_schema` | 2 | 0.500 | 0.500 | 0.500 | 0.062 | 0.000 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `true_late_control` | `generic_answer` | 2 | 0.500 | 0.500 | 0.500 | 0.062 | 0.016 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `true_late_control` | `other_record_value` | 2 | 0.500 | 0.500 | 0.500 | 0.062 | 0.016 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `true_late_control` | `top_class:format_or_schema` | 2 | 0.500 | 0.500 | 0.500 | 0.062 | 0.062 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_plus_late_all` | `top_class:other_vocab` | 2 | 0.500 | 0.500 | 0.500 | 0.047 | 0.078 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_plus_late_all` | `top_non_target` | 2 | 0.500 | 0.500 | 0.500 | 0.047 | 0.078 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_multisite_all` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.500 | 0.500 | 0.031 | 0.031 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_plus_late_all` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.500 | 0.500 | 0.031 | 0.094 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_plus_late_all` | `format_schema` | 2 | 0.500 | 0.500 | 0.500 | 0.016 | 0.078 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_plus_late_all` | `object_relation_echo` | 2 | 0.500 | 0.500 | 0.500 | -0.016 | 0.078 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_plus_late_all` | `top_class:echo_object_or_relation` | 2 | 0.500 | 0.500 | 0.500 | -0.016 | 0.078 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_plus_late_all` | `generic_answer` | 2 | 0.500 | 0.500 | 0.500 | -0.031 | 0.078 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_plus_late_all` | `other_record_value` | 2 | 0.500 | 0.500 | 0.500 | -0.031 | 0.078 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_plus_late_all` | `top_class:format_or_schema` | 2 | 0.500 | 0.500 | 0.500 | -0.062 | 0.062 | `{'route_only_suppressor_candidate': 1, 'target_rewrite_candidate': 1}` |
| deepseek7b | `primary_multisite_all` | `top_class:other_vocab` | 2 | 0.250 | 0.250 | 0.250 | 0.094 | -0.016 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `primary_multisite_all` | `format_schema` | 2 | 0.250 | 0.250 | 0.250 | 0.078 | -0.016 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `primary_multisite_all` | `generic_answer` | 2 | 0.250 | 0.250 | 0.250 | 0.078 | -0.016 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `primary_multisite_all` | `other_record_value` | 2 | 0.250 | 0.250 | 0.250 | 0.078 | -0.016 | `{'route_only_suppressor_candidate': 1, 'weak_or_unclear': 1}` |

## Top Route Suppression Cells

| model | writer | source | combo | route | n | route-only | route rate | target rate | route reduced | recovered | role |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out:H28 | records_all | `late_candidate_all` | `top_non_target` | 2 | 0.500 | 0.500 | 0.000 | 0.125 | -0.062 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_all` | `format_schema` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_all` | `top_class:format_or_schema` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_all` | `top_non_target` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `late_candidate_all` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | -0.062 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_plus_late_all` | `format_schema` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_plus_late_all` | `top_non_target` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `L34:attn_out` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `L34:attn_out` | `top_non_target` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.000 | `route_only_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `late_candidate_all` | `top_class:format_or_schema` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | -0.062 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_plus_late_all` | `top_class:format_or_schema` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | records_all | `L34:attn_out` | `object_relation_echo` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | records_all | `L34:attn_out` | `top_class:format_or_schema` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | records_all | `late_candidate_all` | `format_schema` | 2 | 0.500 | 0.500 | 0.500 | 0.000 | 0.062 | `joint_rewrite_suppressor_candidate` |
| qwen3 | L33:attn_out:H15 | records_all | `late_candidate_all` | `object_relation_echo` | 2 | 0.000 | 0.500 | 0.500 | 0.125 | 0.062 | `joint_rewrite_suppressor_candidate` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_plus_late_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_plus_late_all` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | records_all | `late_candidate_all` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | records_all | `late_candidate_all` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | records_all | `L34:attn_out` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.062 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | records_all | `L34:attn_out` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.062 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_all` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | records_all | `primary_all` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `primary_plus_late_all` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `L36:attn+mlp` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `primary_all` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `late_candidate_all` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `primary_plus_late_all` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `L36:attn+mlp` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `primary_all` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `primary_plus_late_all` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.062 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `L36:attn+mlp` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.062 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `primary_all` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `late_candidate_all` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `primary_plus_late_all` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `L36:attn+mlp` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `late_candidate_all` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | records_all | `primary_plus_late_all` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `primary_all` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `primary_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `primary_all` | `object_relation_echo` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `primary_all` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `primary_all` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `late_candidate_all` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `late_candidate_all` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `late_candidate_all` | `object_relation_echo` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | records_all | `late_candidate_all` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_candidate_all` | `top_class:punctuation_or_stop` | 1 | 1.000 | 1.000 | 0.000 | 0.312 | -0.250 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_control_same_count` | `top_class:punctuation_or_stop` | 1 | 1.000 | 1.000 | 0.000 | 0.312 | -0.250 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_control_same_count` | `top_class:other_vocab` | 2 | 1.000 | 1.000 | 0.000 | 0.281 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_candidate_all` | `top_class:format_or_schema` | 1 | 1.000 | 1.000 | 0.000 | 0.250 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_candidate_all` | `top_class:other_vocab` | 2 | 1.000 | 1.000 | 0.000 | 0.250 | -0.188 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `primary_plus_late_all` | `top_class:punctuation_or_stop` | 1 | 1.000 | 1.000 | 0.000 | 0.250 | -0.188 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_candidate_all` | `format_schema` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.188 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_candidate_all` | `generic_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.188 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_candidate_all` | `other_record_value` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.188 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_candidate_all` | `top_non_target` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.188 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `primary_plus_late_all` | `top_class:other_vocab` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.156 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_control_same_count` | `format_schema` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_control_same_count` | `generic_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_control_same_count` | `other_record_value` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_control_same_count` | `top_non_target` | 2 | 1.000 | 1.000 | 0.000 | 0.219 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_candidate_all` | `object_relation_echo` | 2 | 1.000 | 1.000 | 0.000 | 0.188 | -0.188 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_candidate_all` | `top_class:echo_object_or_relation` | 2 | 1.000 | 1.000 | 0.000 | 0.188 | -0.188 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `primary_plus_late_all` | `generic_answer` | 2 | 1.000 | 1.000 | 0.000 | 0.188 | -0.156 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `primary_plus_late_all` | `other_record_value` | 2 | 1.000 | 1.000 | 0.000 | 0.188 | -0.156 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `primary_plus_late_all` | `top_non_target` | 2 | 1.000 | 1.000 | 0.000 | 0.188 | -0.156 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_control_same_count` | `object_relation_echo` | 2 | 1.000 | 1.000 | 0.000 | 0.188 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_control_same_count` | `top_class:echo_object_or_relation` | 2 | 1.000 | 1.000 | 0.000 | 0.188 | -0.125 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `late_control_same_count` | `top_class:format_or_schema` | 1 | 1.000 | 1.000 | 0.000 | 0.188 | 0.000 | `route_only_suppressor_candidate` |
| deepseek7b | L22:attn_out:H9 | records_all | `primary_all` | `top_class:punctuation_or_stop` | 1 | 1.000 | 1.000 | 0.000 | 0.188 | -0.188 | `route_only_suppressor_candidate` |

## Strict Interpretation

- A route-only cell means restore reduces a route group after source removal while target recovery does not satisfy the target threshold.
- Negative route reduced means the restored component amplifies or fails to close that route release.
- This phase separates route classes explicitly, but still works at component/head level and is not a neuron atlas.
