# Phase 761 Route Source Target Disentanglement (smoke)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Test: remove the same candidate head contribution from target-record sources and route-token sources, then measure target drop and route release separately.

## Source Family x Route Group

| model | source family | route group | groups | route rate | target drop rate | target boost rate | route release | target drop | roles |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `broad_record_source` | `object_relation_echo` | 2 | 0.500 | 0.250 | 0.250 | 0.125 | 0.094 | `{'negative_target_drop_route_artifact': 1, 'route_release_unclear_source': 1}` |
| qwen3 | `format_source` | `top_class:echo_object_or_relation` | 2 | 0.500 | 0.000 | 1.000 | 0.062 | -0.125 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| qwen3 | `route_token_source` | `object_relation_echo` | 2 | 0.500 | 0.000 | 0.500 | 0.062 | -0.094 | `{'negative_target_drop_route_artifact': 2}` |
| qwen3 | `broad_record_source` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.250 | 0.250 | 0.062 | 0.094 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| qwen3 | `format_source` | `format_schema` | 2 | 0.500 | 0.000 | 0.750 | 0.031 | -0.094 | `{'negative_target_drop_route_artifact': 2}` |
| qwen3 | `route_token_source` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.000 | 0.500 | 0.031 | -0.094 | `{'negative_target_drop_route_artifact': 2}` |
| qwen3 | `broad_record_source` | `format_schema` | 2 | 0.500 | 0.250 | 0.250 | 0.031 | 0.094 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| qwen3 | `route_source_union` | `top_class:format_or_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `{'negative_target_drop_route_artifact': 2}` |
| qwen3 | `route_source_union` | `top_non_target` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `{'negative_target_drop_route_artifact': 2}` |
| qwen3 | `route_token_source` | `top_class:format_or_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.094 | `{'negative_target_drop_route_artifact': 2}` |
| qwen3 | `route_token_source` | `top_non_target` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.094 | `{'negative_target_drop_route_artifact': 2}` |
| qwen3 | `target_source` | `object_relation_echo` | 4 | 0.375 | 0.250 | 0.250 | 0.078 | 0.094 | `{'negative_target_drop_route_artifact': 1, 'route_release_unclear_source': 1, 'target_source_with_route_release': 1, 'weak_or_unclear': 1}` |
| qwen3 | `format_source` | `object_relation_echo` | 2 | 0.250 | 0.000 | 0.750 | 0.031 | -0.094 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| qwen3 | `route_source_union` | `object_relation_echo` | 2 | 0.250 | 0.000 | 0.500 | 0.031 | -0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| qwen3 | `broad_record_source` | `top_non_target` | 2 | 0.250 | 0.250 | 0.250 | 0.000 | 0.094 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| qwen3 | `broad_record_source` | `top_class:format_or_schema` | 2 | 0.250 | 0.250 | 0.250 | -0.031 | 0.094 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| qwen3 | `target_source` | `format_schema` | 4 | 0.125 | 0.250 | 0.250 | -0.062 | 0.094 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1, 'weak_or_unclear': 2}` |
| qwen3 | `broad_record_source` | `top_class:echo_object_or_relation` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | `{'weak_or_unclear': 2}` |
| qwen3 | `format_source` | `generic_answer` | 2 | 0.000 | 0.000 | 0.750 | 0.000 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `format_source` | `other_record_value` | 2 | 0.000 | 0.000 | 0.750 | 0.000 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `format_source` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.750 | 0.000 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `format_source` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 1.000 | 0.000 | -0.125 | `{'weak_or_unclear': 2}` |
| qwen3 | `format_source` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.750 | 0.000 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `format_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.750 | 0.000 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_source_union` | `generic_answer` | 2 | 0.000 | 0.000 | 0.500 | 0.000 | -0.062 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_source_union` | `other_record_value` | 2 | 0.000 | 0.000 | 0.500 | 0.000 | -0.062 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_source_union` | `top_class:echo_object_or_relation` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_source_union` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.500 | 0.000 | -0.062 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_token_source` | `format_schema` | 2 | 0.000 | 0.000 | 0.500 | 0.000 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_token_source` | `top_class:echo_object_or_relation` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| qwen3 | `format_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.750 | -0.016 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_token_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.500 | -0.016 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `target_source` | `top_class:punctuation_or_stop` | 4 | 0.000 | 0.250 | 0.250 | -0.016 | 0.094 | `{'target_source_writer': 1, 'weak_or_unclear': 3}` |
| qwen3 | `broad_record_source` | `contrast_answer` | 2 | 0.000 | 0.250 | 0.250 | -0.023 | 0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_source_union` | `format_schema` | 2 | 0.000 | 0.000 | 0.500 | -0.031 | -0.062 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_token_source` | `generic_answer` | 2 | 0.000 | 0.000 | 0.500 | -0.031 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `route_token_source` | `other_record_value` | 2 | 0.000 | 0.000 | 0.500 | -0.031 | -0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `target_source` | `top_class:echo_object_or_relation` | 4 | 0.000 | 0.000 | 0.250 | -0.031 | 0.000 | `{'weak_or_unclear': 4}` |
| qwen3 | `broad_record_source` | `generic_answer` | 2 | 0.000 | 0.250 | 0.250 | -0.031 | 0.094 | `{'weak_or_unclear': 2}` |
| qwen3 | `broad_record_source` | `other_record_value` | 2 | 0.000 | 0.250 | 0.250 | -0.031 | 0.094 | `{'weak_or_unclear': 2}` |
| glm4 | `broad_record_source` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `format_source` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `broad_record_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `format_source` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `format_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_source_union` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `target_source` | `top_class:other_semantic_value` | 4 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `{'weak_or_unclear': 4}` |
| glm4 | `format_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.023 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `target_source` | `top_class:format_or_schema` | 4 | 0.000 | 0.000 | 0.000 | 0.023 | 0.000 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `top_non_target` | 4 | 0.000 | 0.000 | 0.000 | 0.023 | 0.000 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.016 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_source_union` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.016 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `format_source` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `format_source` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `target_source` | `contrast_answer` | 4 | 0.000 | 0.000 | 0.000 | 0.007 | 0.000 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `broad_record_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `broad_record_source` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `broad_record_source` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `broad_record_source` | `top_class:echo_object_or_relation` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `broad_record_source` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `broad_record_source` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `broad_record_source` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | `{'weak_or_unclear': 2}` |
| glm4 | `format_source` | `top_class:echo_object_or_relation` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `format_source` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `format_source` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_source_union` | `generic_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_source_union` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_source_union` | `top_class:echo_object_or_relation` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_source_union` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_source_union` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_source_union` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_token_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_token_source` | `top_class:echo_object_or_relation` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_token_source` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_token_source` | `top_class:other_semantic_value` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_token_source` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_token_source` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `route_token_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 2}` |
| glm4 | `target_source` | `top_class:echo_object_or_relation` | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `{'weak_or_unclear': 4}` |
| deepseek7b | `route_token_source` | `top_class:other_vocab` | 2 | 0.750 | 0.000 | 0.750 | 0.156 | -0.125 | `{'negative_target_drop_route_artifact': 2}` |
| deepseek7b | `route_token_source` | `top_class:punctuation_or_stop` | 2 | 0.750 | 0.000 | 0.750 | 0.141 | -0.125 | `{'negative_target_drop_route_artifact': 2}` |
| deepseek7b | `route_token_source` | `top_class:format_or_schema` | 2 | 0.750 | 0.000 | 0.750 | 0.109 | -0.125 | `{'negative_target_drop_route_artifact': 2}` |
| deepseek7b | `route_token_source` | `top_non_target` | 2 | 0.750 | 0.000 | 0.750 | 0.109 | -0.125 | `{'negative_target_drop_route_artifact': 2}` |
| deepseek7b | `broad_record_source` | `top_class:other_vocab` | 2 | 0.500 | 0.500 | 0.500 | 0.125 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `route_token_source` | `format_schema` | 2 | 0.500 | 0.000 | 0.750 | 0.109 | -0.125 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `broad_record_source` | `format_schema` | 2 | 0.500 | 0.500 | 0.500 | 0.109 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `broad_record_source` | `top_class:format_or_schema` | 2 | 0.500 | 0.500 | 0.500 | 0.109 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `broad_record_source` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.500 | 0.500 | 0.094 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `broad_record_source` | `top_non_target` | 2 | 0.500 | 0.500 | 0.500 | 0.094 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `route_token_source` | `object_relation_echo` | 2 | 0.500 | 0.000 | 0.750 | 0.078 | -0.125 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `route_token_source` | `top_class:echo_object_or_relation` | 2 | 0.500 | 0.000 | 0.750 | 0.078 | -0.125 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `format_source` | `contrast_answer` | 2 | 0.500 | 0.000 | 0.250 | 0.059 | -0.031 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 1}` |
| deepseek7b | `broad_record_source` | `generic_answer` | 2 | 0.500 | 0.500 | 0.500 | 0.031 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `broad_record_source` | `other_record_value` | 2 | 0.500 | 0.500 | 0.500 | 0.031 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `format_source` | `generic_answer` | 2 | 0.500 | 0.000 | 0.250 | 0.016 | -0.031 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 1}` |
| deepseek7b | `format_source` | `other_record_value` | 2 | 0.500 | 0.000 | 0.250 | 0.016 | -0.031 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 1}` |
| deepseek7b | `route_token_source` | `contrast_answer` | 2 | 0.500 | 0.000 | 0.750 | 0.012 | -0.125 | `{'negative_target_drop_route_artifact': 2}` |
| deepseek7b | `broad_record_source` | `object_relation_echo` | 2 | 0.500 | 0.500 | 0.500 | 0.000 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `broad_record_source` | `top_class:echo_object_or_relation` | 2 | 0.500 | 0.500 | 0.500 | 0.000 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `target_source` | `top_non_target` | 4 | 0.375 | 0.250 | 0.375 | 0.023 | 0.062 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `target_source` | `object_relation_echo` | 4 | 0.375 | 0.250 | 0.375 | 0.000 | 0.062 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `target_source` | `top_class:echo_object_or_relation` | 4 | 0.375 | 0.250 | 0.375 | 0.000 | 0.062 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `route_token_source` | `generic_answer` | 2 | 0.250 | 0.000 | 0.750 | 0.078 | -0.125 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `route_token_source` | `other_record_value` | 2 | 0.250 | 0.000 | 0.750 | 0.078 | -0.125 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `route_source_union` | `format_schema` | 2 | 0.250 | 0.000 | 0.500 | 0.062 | -0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `route_source_union` | `top_class:punctuation_or_stop` | 2 | 0.250 | 0.000 | 0.500 | 0.047 | -0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `route_source_union` | `top_non_target` | 2 | 0.250 | 0.000 | 0.500 | 0.047 | -0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `format_source` | `top_non_target` | 2 | 0.250 | 0.000 | 0.250 | 0.016 | -0.031 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `route_source_union` | `object_relation_echo` | 2 | 0.250 | 0.000 | 0.500 | 0.016 | -0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `route_source_union` | `top_class:echo_object_or_relation` | 2 | 0.250 | 0.000 | 0.500 | 0.016 | -0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `route_source_union` | `top_class:format_or_schema` | 2 | 0.250 | 0.000 | 0.500 | 0.016 | -0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `route_source_union` | `top_class:other_vocab` | 2 | 0.250 | 0.000 | 0.500 | 0.016 | -0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `target_source` | `top_class:punctuation_or_stop` | 4 | 0.250 | 0.250 | 0.375 | 0.008 | 0.062 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `format_source` | `object_relation_echo` | 2 | 0.250 | 0.000 | 0.250 | 0.000 | -0.031 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `format_source` | `top_class:echo_object_or_relation` | 2 | 0.250 | 0.000 | 0.250 | 0.000 | -0.031 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 1}` |
| deepseek7b | `target_source` | `top_class:other_vocab` | 4 | 0.250 | 0.250 | 0.375 | 0.000 | 0.062 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `target_source` | `top_class:format_or_schema` | 4 | 0.250 | 0.250 | 0.375 | -0.016 | 0.062 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `broad_record_source` | `contrast_answer` | 2 | 0.250 | 0.500 | 0.500 | -0.074 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 1}` |
| deepseek7b | `target_source` | `format_schema` | 4 | 0.125 | 0.250 | 0.375 | 0.008 | 0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |

## Top Cells

| model | head | kind | source | family | route | n | route rate | target drop rate | target boost rate | route release | target drop | role |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `top_class:echo_object_or_relation` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_record_line` | `target_source` | `object_relation_echo` | 2 | 0.500 | 0.500 | 0.000 | 0.188 | 0.250 | `target_source_with_route_release` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `object_relation_echo` | 2 | 0.500 | 0.500 | 0.000 | 0.188 | 0.188 | `route_release_unclear_source` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `object_relation_echo` | 2 | 0.500 | 0.500 | 0.000 | 0.125 | 0.188 | `route_release_unclear_source` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `object_relation_echo` | 2 | 0.500 | 0.000 | 0.500 | 0.062 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.000 | 0.500 | 0.062 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `records_all` | `broad_record_source` | `format_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.062 | 0.000 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `records_all` | `broad_record_source` | `object_relation_echo` | 2 | 0.500 | 0.000 | 0.500 | 0.062 | 0.000 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.000 | 0.500 | 0.062 | 0.000 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:any_route` | `route_source_union` | `object_relation_echo` | 2 | 0.500 | 0.000 | 0.500 | 0.062 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `object_relation_echo` | 2 | 0.500 | 0.000 | 0.500 | 0.062 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `format_schema` | 2 | 0.500 | 0.000 | 1.000 | 0.062 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `object_relation_echo` | 2 | 0.500 | 0.000 | 1.000 | 0.062 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `object_relation_echo` | 2 | 0.500 | 0.000 | 1.000 | 0.062 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.188 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:format_or_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_non_target` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `top_class:format_or_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `top_non_target` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_class:format_or_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | 0.000 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_non_target` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | 0.000 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:any_route` | `route_source_union` | `top_class:format_or_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:any_route` | `route_source_union` | `top_non_target` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:format_schema` | `format_source` | `format_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_class:format_or_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_non_target` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `format_schema` | 2 | 0.500 | 0.000 | 1.000 | 0.000 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `format_schema` | 2 | 0.500 | 0.500 | 0.000 | 0.000 | 0.188 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `records_all` | `broad_record_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.500 | 0.016 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:echo_object_or_relation` | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:echo_object_or_relation` | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:echo_object_or_relation` | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:echo_object_or_relation` | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `route_src:format_schema` | `format_source` | `top_class:other_semantic_value` | 1 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.062 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `route_src:format_schema` | `format_source` | `top_class:format_or_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `route_src:format_schema` | `format_source` | `top_non_target` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.023 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.023 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `route_src:format_schema` | `format_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.023 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.016 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.016 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `route_src:format_schema` | `format_source` | `other_record_value` | 2 | 0.000 | 0.000 | 0.000 | 0.016 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `target_record_line` | `target_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `records_all` | `broad_record_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `route_src:any_route` | `route_source_union` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `contrast_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `format_schema` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `object_relation_echo` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_vocab` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:punctuation_or_stop` | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `records_all` | `broad_record_source` | `generic_answer` | 2 | 1.000 | 0.000 | 1.000 | 0.250 | -0.219 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `records_all` | `broad_record_source` | `other_record_value` | 2 | 1.000 | 0.000 | 1.000 | 0.250 | -0.219 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_class:other_vocab` | 2 | 1.000 | 0.000 | 1.000 | 0.250 | -0.219 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `top_class:other_vocab` | 2 | 1.000 | 0.000 | 1.000 | 0.219 | -0.188 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `records_all` | `broad_record_source` | `format_schema` | 2 | 1.000 | 0.000 | 1.000 | 0.219 | -0.219 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_class:format_or_schema` | 2 | 1.000 | 0.000 | 1.000 | 0.219 | -0.219 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_class:punctuation_or_stop` | 2 | 1.000 | 0.000 | 1.000 | 0.219 | -0.219 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_non_target` | 2 | 1.000 | 0.000 | 1.000 | 0.219 | -0.219 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `records_all` | `broad_record_source` | `object_relation_echo` | 2 | 1.000 | 0.000 | 1.000 | 0.188 | -0.219 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_class:echo_object_or_relation` | 2 | 1.000 | 0.000 | 1.000 | 0.188 | -0.219 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `format_schema` | 2 | 1.000 | 0.000 | 1.000 | 0.156 | -0.188 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `top_class:format_or_schema` | 2 | 1.000 | 0.000 | 1.000 | 0.156 | -0.188 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 2 | 1.000 | 0.000 | 1.000 | 0.156 | -0.188 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `object_relation_echo` | 2 | 1.000 | 0.000 | 1.000 | 0.156 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `top_class:echo_object_or_relation` | 2 | 1.000 | 0.000 | 1.000 | 0.156 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `object_relation_echo` | 2 | 1.000 | 0.000 | 1.000 | 0.125 | -0.188 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `top_class:echo_object_or_relation` | 2 | 1.000 | 0.000 | 1.000 | 0.125 | -0.188 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `top_non_target` | 2 | 1.000 | 0.000 | 1.000 | 0.125 | -0.188 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `top_non_target` | 2 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.000 | 0.500 | 0.125 | -0.062 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `generic_answer` | 2 | 0.500 | 0.000 | 1.000 | 0.125 | -0.188 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `other_record_value` | 2 | 0.500 | 0.000 | 1.000 | 0.125 | -0.188 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_class:other_vocab` | 2 | 0.500 | 0.000 | 0.500 | 0.094 | -0.062 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_non_target` | 2 | 0.500 | 0.000 | 0.500 | 0.094 | -0.062 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `format_schema` | 2 | 0.500 | 0.000 | 1.000 | 0.094 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:other_vocab` | 2 | 0.500 | 0.000 | 1.000 | 0.094 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_non_target` | 2 | 0.500 | 0.000 | 1.000 | 0.094 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `format_schema` | 2 | 0.500 | 0.000 | 1.000 | 0.094 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `top_class:format_or_schema` | 2 | 0.500 | 0.000 | 1.000 | 0.094 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `top_class:other_vocab` | 2 | 0.500 | 0.000 | 1.000 | 0.094 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `top_class:punctuation_or_stop` | 2 | 0.500 | 0.000 | 1.000 | 0.094 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:format_schema` | `format_source` | `contrast_answer` | 2 | 0.500 | 0.000 | 0.000 | 0.062 | 0.000 | `route_source_release_without_target_drop` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_class:format_or_schema` | 2 | 0.500 | 0.000 | 0.500 | 0.062 | -0.062 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `object_relation_echo` | 2 | 0.500 | 0.000 | 1.000 | 0.062 | -0.125 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:echo_object_or_relation` | 2 | 0.500 | 0.000 | 1.000 | 0.062 | -0.125 | `negative_target_drop_route_artifact` |

## Strict Interpretation

- If route-token sources release routes without target drop, route competition has source-family evidence distinct from target writer evidence.
- If target sources and route sources both release routes, the result supports a mixed or distributed route field.
- If same-layer control heads match the candidate heads, the effect is not specific enough to call a global suppressor.
