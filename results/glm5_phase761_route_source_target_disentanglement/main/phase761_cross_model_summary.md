# Phase 761 Route Source Target Disentanglement (main)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Test: remove the same candidate head contribution from target-record sources and route-token sources, then measure target drop and route release separately.

## Source Family x Route Group

| model | source family | route group | groups | route rate | target drop rate | target boost rate | route release | target drop | roles |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `broad_record_source` | `top_class:recipient_answer` | 4 | 0.500 | 0.000 | 0.250 | 0.094 | 0.000 | `{'negative_target_drop_route_artifact': 1, 'route_release_unclear_source': 1, 'weak_or_unclear': 2}` |
| qwen3 | `target_source` | `top_class:recipient_answer` | 8 | 0.500 | 0.000 | 0.375 | 0.062 | 0.000 | `{'negative_target_drop_route_artifact': 3, 'route_release_unclear_source': 1, 'weak_or_unclear': 4}` |
| qwen3 | `route_token_source` | `top_class:format_or_schema` | 20 | 0.361 | 0.006 | 0.255 | 0.027 | -0.005 | `{'negative_target_drop_route_artifact': 4, 'route_source_release_without_target_drop': 5, 'weak_or_unclear': 11}` |
| qwen3 | `route_token_source` | `top_non_target` | 20 | 0.332 | 0.006 | 0.255 | 0.030 | -0.005 | `{'negative_target_drop_route_artifact': 2, 'route_source_release_without_target_drop': 4, 'weak_or_unclear': 14}` |
| qwen3 | `route_source_union` | `top_class:format_or_schema` | 4 | 0.292 | 0.010 | 0.271 | 0.012 | -0.021 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 2}` |
| qwen3 | `format_source` | `top_class:format_or_schema` | 4 | 0.292 | 0.010 | 0.281 | 0.008 | -0.014 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_token_source` | `top_class:punctuation_or_stop` | 20 | 0.284 | 0.006 | 0.255 | 0.013 | -0.005 | `{'negative_target_drop_route_artifact': 3, 'route_source_release_without_target_drop': 2, 'weak_or_unclear': 15}` |
| qwen3 | `echo_source` | `top_class:punctuation_or_stop` | 4 | 0.272 | 0.022 | 0.261 | 0.015 | -0.008 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_token_source` | `format_schema` | 20 | 0.267 | 0.006 | 0.255 | 0.013 | -0.005 | `{'negative_target_drop_route_artifact': 3, 'route_source_release_without_target_drop': 2, 'weak_or_unclear': 15}` |
| qwen3 | `broad_record_source` | `other_record_value` | 4 | 0.260 | 0.146 | 0.198 | -0.002 | 0.056 | `{'weak_or_unclear': 4}` |
| qwen3 | `broad_record_source` | `object_relation_echo` | 4 | 0.260 | 0.146 | 0.198 | -0.052 | 0.056 | `{'weak_or_unclear': 4}` |
| qwen3 | `format_source` | `top_class:recipient_answer` | 4 | 0.250 | 0.000 | 0.000 | 0.031 | 0.031 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_token_source` | `top_class:recipient_answer` | 16 | 0.250 | 0.000 | 0.188 | 0.023 | 0.008 | `{'negative_target_drop_route_artifact': 3, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 12}` |
| qwen3 | `format_source` | `top_class:punctuation_or_stop` | 4 | 0.250 | 0.010 | 0.281 | 0.016 | -0.014 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_source_union` | `top_non_target` | 4 | 0.250 | 0.010 | 0.271 | 0.008 | -0.021 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_source_union` | `format_schema` | 4 | 0.240 | 0.010 | 0.271 | 0.008 | -0.021 | `{'weak_or_unclear': 4}` |
| qwen3 | `format_source` | `top_non_target` | 4 | 0.240 | 0.010 | 0.281 | 0.004 | -0.014 | `{'weak_or_unclear': 4}` |
| qwen3 | `target_source` | `top_class:format_or_schema` | 8 | 0.240 | 0.099 | 0.266 | 0.000 | 0.030 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 6}` |
| qwen3 | `broad_record_source` | `top_class:punctuation_or_stop` | 4 | 0.240 | 0.146 | 0.198 | -0.017 | 0.056 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `target_source` | `top_non_target` | 8 | 0.234 | 0.099 | 0.266 | -0.010 | 0.030 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 6}` |
| qwen3 | `broad_record_source` | `generic_answer` | 4 | 0.229 | 0.146 | 0.198 | 0.003 | 0.056 | `{'weak_or_unclear': 4}` |
| qwen3 | `broad_record_source` | `top_class:format_or_schema` | 4 | 0.229 | 0.146 | 0.198 | -0.022 | 0.056 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `target_source` | `top_class:punctuation_or_stop` | 8 | 0.224 | 0.099 | 0.266 | -0.003 | 0.030 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 7}` |
| qwen3 | `route_source_union` | `top_class:punctuation_or_stop` | 4 | 0.219 | 0.010 | 0.271 | 0.007 | -0.021 | `{'weak_or_unclear': 4}` |
| qwen3 | `route_token_source` | `top_class:other_vocab` | 16 | 0.218 | 0.010 | 0.235 | -0.004 | -0.000 | `{'negative_target_drop_route_artifact': 3, 'route_source_release_without_target_drop': 2, 'weak_or_unclear': 11}` |
| qwen3 | `echo_source` | `format_schema` | 4 | 0.217 | 0.022 | 0.261 | 0.003 | -0.008 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| qwen3 | `echo_source` | `top_class:format_or_schema` | 4 | 0.217 | 0.022 | 0.261 | -0.008 | -0.008 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| qwen3 | `target_source` | `format_schema` | 8 | 0.214 | 0.099 | 0.266 | -0.005 | 0.030 | `{'weak_or_unclear': 8}` |
| qwen3 | `route_source_union` | `top_class:other_vocab` | 4 | 0.212 | 0.013 | 0.225 | -0.003 | -0.011 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `broad_record_source` | `contrast_answer` | 4 | 0.208 | 0.146 | 0.198 | 0.038 | 0.056 | `{'weak_or_unclear': 4}` |
| qwen3 | `target_source` | `top_class:other_vocab` | 8 | 0.206 | 0.106 | 0.225 | -0.020 | 0.038 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 6}` |
| qwen3 | `echo_source` | `top_class:echo_object_or_relation` | 4 | 0.200 | 0.000 | 0.250 | 0.006 | -0.006 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| qwen3 | `broad_record_source` | `top_class:echo_object_or_relation` | 4 | 0.200 | 0.100 | 0.200 | -0.031 | 0.038 | `{'weak_or_unclear': 4}` |
| qwen3 | `broad_record_source` | `top_non_target` | 4 | 0.198 | 0.146 | 0.198 | -0.039 | 0.056 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `target_source` | `object_relation_echo` | 8 | 0.193 | 0.099 | 0.266 | -0.017 | 0.030 | `{'weak_or_unclear': 8}` |
| qwen3 | `format_source` | `format_schema` | 4 | 0.188 | 0.010 | 0.281 | 0.007 | -0.014 | `{'weak_or_unclear': 4}` |
| qwen3 | `target_source` | `top_class:other_semantic_value` | 8 | 0.183 | 0.142 | 0.250 | -0.073 | 0.045 | `{'negative_target_drop_route_artifact': 2, 'target_source_writer': 2, 'weak_or_unclear': 4}` |
| qwen3 | `route_source_union` | `top_class:other_semantic_value` | 4 | 0.183 | 0.017 | 0.267 | -0.010 | -0.013 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `target_source` | `other_record_value` | 8 | 0.182 | 0.099 | 0.266 | 0.010 | 0.030 | `{'weak_or_unclear': 8}` |
| qwen3 | `route_token_source` | `other_record_value` | 20 | 0.181 | 0.006 | 0.255 | 0.007 | -0.005 | `{'negative_target_drop_route_artifact': 2, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 17}` |
| glm4 | `broad_record_source` | `top_class:punctuation_or_stop` | 4 | 0.050 | 0.033 | 0.033 | 0.010 | 0.005 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `top_class:other_vocab` | 8 | 0.050 | 0.025 | 0.050 | -0.003 | 0.004 | `{'weak_or_unclear': 8}` |
| glm4 | `broad_record_source` | `top_class:other_vocab` | 4 | 0.050 | 0.025 | 0.050 | -0.007 | 0.004 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `top_class:recipient_answer` | 4 | 0.036 | 0.000 | 0.000 | -0.004 | -0.013 | `{'weak_or_unclear': 4}` |
| glm4 | `broad_record_source` | `top_class:other_semantic_value` | 4 | 0.031 | 0.031 | 0.062 | 0.010 | 0.005 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `top_class:other_semantic_value` | 8 | 0.031 | 0.031 | 0.055 | 0.009 | 0.005 | `{'weak_or_unclear': 8}` |
| glm4 | `target_source` | `top_class:echo_object_or_relation` | 8 | 0.029 | 0.029 | 0.059 | 0.003 | 0.001 | `{'weak_or_unclear': 8}` |
| glm4 | `broad_record_source` | `top_class:echo_object_or_relation` | 4 | 0.029 | 0.029 | 0.059 | 0.000 | 0.000 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `top_non_target` | 8 | 0.026 | 0.021 | 0.042 | 0.006 | 0.002 | `{'weak_or_unclear': 8}` |
| glm4 | `broad_record_source` | `top_non_target` | 4 | 0.021 | 0.021 | 0.042 | 0.008 | 0.001 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `object_relation_echo` | 8 | 0.021 | 0.021 | 0.042 | 0.000 | 0.002 | `{'weak_or_unclear': 8}` |
| glm4 | `broad_record_source` | `object_relation_echo` | 4 | 0.021 | 0.021 | 0.042 | -0.001 | 0.001 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `generic_answer` | 8 | 0.021 | 0.021 | 0.042 | -0.004 | 0.002 | `{'weak_or_unclear': 8}` |
| glm4 | `target_source` | `top_class:recipient_answer` | 8 | 0.018 | 0.000 | 0.000 | -0.006 | -0.001 | `{'weak_or_unclear': 8}` |
| glm4 | `target_source` | `contrast_answer` | 8 | 0.016 | 0.021 | 0.042 | -0.003 | 0.002 | `{'weak_or_unclear': 8}` |
| glm4 | `route_source_union` | `top_non_target` | 4 | 0.010 | 0.000 | 0.021 | 0.005 | 0.005 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `generic_answer` | 4 | 0.010 | 0.000 | 0.021 | 0.002 | 0.005 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `contrast_answer` | 4 | 0.010 | 0.000 | 0.021 | -0.000 | 0.005 | `{'weak_or_unclear': 4}` |
| glm4 | `broad_record_source` | `generic_answer` | 4 | 0.010 | 0.021 | 0.042 | -0.001 | 0.001 | `{'weak_or_unclear': 4}` |
| glm4 | `broad_record_source` | `contrast_answer` | 4 | 0.010 | 0.021 | 0.042 | -0.002 | 0.001 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `top_class:punctuation_or_stop` | 8 | 0.008 | 0.033 | 0.033 | 0.006 | 0.005 | `{'weak_or_unclear': 8}` |
| glm4 | `target_source` | `format_schema` | 8 | 0.005 | 0.021 | 0.042 | 0.002 | 0.002 | `{'weak_or_unclear': 8}` |
| glm4 | `broad_record_source` | `other_record_value` | 4 | 0.000 | 0.021 | 0.042 | 0.005 | 0.001 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `other_record_value` | 4 | 0.000 | 0.000 | 0.021 | 0.005 | 0.005 | `{'weak_or_unclear': 4}` |
| glm4 | `echo_source` | `other_record_value` | 4 | 0.000 | 0.000 | 0.045 | 0.004 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `route_token_source` | `top_class:format_or_schema` | 20 | 0.000 | 0.000 | 0.020 | 0.003 | 0.011 | `{'weak_or_unclear': 20}` |
| glm4 | `broad_record_source` | `top_class:format_or_schema` | 4 | 0.000 | 0.021 | 0.042 | 0.003 | 0.001 | `{'weak_or_unclear': 4}` |
| glm4 | `format_source` | `top_class:punctuation_or_stop` | 4 | 0.000 | 0.000 | 0.000 | 0.003 | 0.011 | `{'weak_or_unclear': 4}` |
| glm4 | `echo_source` | `top_class:other_semantic_value` | 4 | 0.000 | 0.000 | 0.047 | 0.003 | 0.013 | `{'weak_or_unclear': 4}` |
| glm4 | `broad_record_source` | `format_schema` | 4 | 0.000 | 0.021 | 0.042 | 0.003 | 0.001 | `{'weak_or_unclear': 4}` |
| glm4 | `route_token_source` | `top_non_target` | 20 | 0.000 | 0.000 | 0.020 | 0.003 | 0.011 | `{'weak_or_unclear': 20}` |
| glm4 | `echo_source` | `format_schema` | 4 | 0.000 | 0.000 | 0.045 | 0.002 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `other_record_value` | 8 | 0.000 | 0.021 | 0.042 | 0.002 | 0.002 | `{'weak_or_unclear': 8}` |
| glm4 | `route_token_source` | `top_class:other_semantic_value` | 16 | 0.000 | 0.000 | 0.027 | 0.002 | 0.019 | `{'weak_or_unclear': 16}` |
| glm4 | `format_source` | `top_class:other_semantic_value` | 4 | 0.000 | 0.000 | 0.000 | 0.002 | 0.014 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `top_class:format_or_schema` | 4 | 0.000 | 0.000 | 0.021 | 0.002 | 0.005 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `top_class:other_semantic_value` | 4 | 0.000 | 0.000 | 0.031 | 0.002 | 0.013 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `format_schema` | 4 | 0.000 | 0.000 | 0.021 | 0.002 | 0.005 | `{'weak_or_unclear': 4}` |
| glm4 | `echo_source` | `top_non_target` | 4 | 0.000 | 0.000 | 0.045 | 0.001 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `route_token_source` | `contrast_answer` | 20 | 0.000 | 0.000 | 0.020 | 0.001 | 0.011 | `{'weak_or_unclear': 20}` |
| deepseek7b | `route_token_source` | `top_class:recipient_answer` | 20 | 0.492 | 0.000 | 0.629 | 0.138 | -0.142 | `{'negative_target_drop_route_artifact': 13, 'weak_or_unclear': 7}` |
| deepseek7b | `route_source_union` | `top_class:recipient_answer` | 4 | 0.375 | 0.000 | 0.500 | 0.066 | -0.086 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `format_source` | `top_class:recipient_answer` | 4 | 0.312 | 0.000 | 0.312 | 0.051 | -0.074 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `route_token_source` | `object_relation_echo` | 24 | 0.302 | 0.051 | 0.352 | -0.002 | -0.027 | `{'negative_target_drop_route_artifact': 4, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 19}` |
| deepseek7b | `target_source` | `top_class:recipient_answer` | 8 | 0.281 | 0.094 | 0.250 | 0.041 | -0.029 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 6}` |
| deepseek7b | `target_source` | `top_non_target` | 8 | 0.276 | 0.359 | 0.156 | -0.025 | 0.197 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 4, 'weak_or_unclear': 3}` |
| deepseek7b | `route_token_source` | `top_non_target` | 24 | 0.275 | 0.051 | 0.352 | 0.015 | -0.027 | `{'negative_target_drop_route_artifact': 5, 'weak_or_unclear': 19}` |
| deepseek7b | `target_source` | `object_relation_echo` | 8 | 0.271 | 0.359 | 0.156 | -0.004 | 0.197 | `{'target_source_writer': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `route_token_source` | `top_class:other_vocab` | 24 | 0.271 | 0.051 | 0.352 | 0.036 | -0.027 | `{'negative_target_drop_route_artifact': 6, 'weak_or_unclear': 18}` |
| deepseek7b | `target_source` | `top_class:echo_object_or_relation` | 8 | 0.269 | 0.350 | 0.163 | -0.013 | 0.182 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 4, 'weak_or_unclear': 3}` |
| deepseek7b | `route_token_source` | `format_schema` | 24 | 0.264 | 0.051 | 0.352 | -0.007 | -0.027 | `{'negative_target_drop_route_artifact': 3, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 20}` |
| deepseek7b | `route_source_union` | `top_class:echo_object_or_relation` | 4 | 0.263 | 0.062 | 0.350 | -0.013 | -0.021 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `route_token_source` | `generic_answer` | 24 | 0.256 | 0.051 | 0.352 | 0.053 | -0.027 | `{'negative_target_drop_route_artifact': 4, 'weak_or_unclear': 20}` |
| deepseek7b | `route_token_source` | `top_class:format_or_schema` | 24 | 0.253 | 0.051 | 0.352 | 0.012 | -0.027 | `{'negative_target_drop_route_artifact': 5, 'weak_or_unclear': 19}` |
| deepseek7b | `route_token_source` | `top_class:punctuation_or_stop` | 24 | 0.251 | 0.051 | 0.352 | 0.027 | -0.027 | `{'negative_target_drop_route_artifact': 4, 'weak_or_unclear': 20}` |
| deepseek7b | `echo_source` | `top_class:recipient_answer` | 4 | 0.250 | 0.000 | 0.250 | 0.068 | -0.078 | `{'negative_target_drop_route_artifact': 3, 'weak_or_unclear': 1}` |
| deepseek7b | `route_source_union` | `top_class:punctuation_or_stop` | 4 | 0.250 | 0.094 | 0.323 | 0.010 | -0.005 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `route_source_union` | `object_relation_echo` | 4 | 0.229 | 0.094 | 0.323 | -0.018 | -0.005 | `{'weak_or_unclear': 4}` |
| deepseek7b | `broad_record_source` | `top_non_target` | 4 | 0.229 | 0.448 | 0.146 | -0.032 | 0.253 | `{'target_source_writer': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `echo_source` | `top_class:other_vocab` | 4 | 0.228 | 0.076 | 0.261 | -0.005 | -0.011 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 2}` |
| deepseek7b | `route_token_source` | `contrast_answer` | 24 | 0.222 | 0.051 | 0.352 | 0.025 | -0.027 | `{'negative_target_drop_route_artifact': 3, 'weak_or_unclear': 21}` |
| deepseek7b | `format_source` | `top_non_target` | 4 | 0.219 | 0.062 | 0.198 | -0.004 | 0.010 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `route_source_union` | `top_non_target` | 4 | 0.219 | 0.094 | 0.323 | -0.005 | -0.005 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `echo_source` | `top_non_target` | 4 | 0.217 | 0.076 | 0.261 | -0.013 | -0.011 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `target_source` | `top_class:format_or_schema` | 8 | 0.214 | 0.359 | 0.156 | -0.041 | 0.197 | `{'target_source_writer': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `route_token_source` | `top_class:echo_object_or_relation` | 20 | 0.211 | 0.056 | 0.187 | -0.042 | 0.034 | `{'mixed_route_and_target_source': 1, 'negative_target_drop_route_artifact': 2, 'route_source_release_without_target_drop': 2, 'weak_or_unclear': 15}` |
| deepseek7b | `route_token_source` | `top_class:other_semantic_value` | 12 | 0.210 | 0.224 | 0.126 | -0.034 | 0.084 | `{'negative_target_drop_route_artifact': 2, 'route_source_release_without_target_drop': 2, 'weak_or_unclear': 8}` |
| deepseek7b | `route_source_union` | `format_schema` | 4 | 0.208 | 0.094 | 0.323 | 0.003 | -0.005 | `{'weak_or_unclear': 4}` |
| deepseek7b | `route_source_union` | `top_class:format_or_schema` | 4 | 0.208 | 0.094 | 0.323 | -0.022 | -0.005 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `echo_source` | `top_class:echo_object_or_relation` | 4 | 0.200 | 0.075 | 0.288 | -0.041 | -0.023 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `target_source` | `top_class:other_vocab` | 8 | 0.198 | 0.359 | 0.156 | -0.042 | 0.197 | `{'target_source_writer': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `format_source` | `generic_answer` | 4 | 0.198 | 0.062 | 0.198 | 0.008 | 0.010 | `{'weak_or_unclear': 4}` |
| deepseek7b | `route_source_union` | `top_class:other_vocab` | 4 | 0.198 | 0.094 | 0.323 | -0.001 | -0.005 | `{'weak_or_unclear': 4}` |
| deepseek7b | `broad_record_source` | `top_class:format_or_schema` | 4 | 0.198 | 0.448 | 0.146 | -0.040 | 0.253 | `{'target_source_writer': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `target_source` | `generic_answer` | 8 | 0.193 | 0.359 | 0.156 | -0.025 | 0.197 | `{'target_source_writer': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `format_source` | `top_class:format_or_schema` | 4 | 0.188 | 0.062 | 0.198 | -0.007 | 0.010 | `{'weak_or_unclear': 4}` |
| deepseek7b | `route_source_union` | `generic_answer` | 4 | 0.188 | 0.094 | 0.323 | -0.010 | -0.005 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `route_token_source` | `other_record_value` | 24 | 0.180 | 0.051 | 0.352 | -0.006 | -0.027 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 22}` |
| deepseek7b | `route_source_union` | `top_class:other_semantic_value` | 4 | 0.179 | 0.250 | 0.214 | -0.004 | 0.067 | `{'weak_or_unclear': 4}` |
| deepseek7b | `format_source` | `object_relation_echo` | 4 | 0.177 | 0.062 | 0.198 | -0.005 | 0.010 | `{'weak_or_unclear': 4}` |

## Top Cells

| model | head | kind | source | family | route | n | route rate | target drop rate | target boost rate | route release | target drop | role |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:format_or_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_non_target` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:other_record_value` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `format_schema` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `other_record_value` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:format_or_schema` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_non_target` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:format_or_schema` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_non_target` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_release_unclear_source` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_release_unclear_source` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `target_record_line` | `target_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `contrast_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `format_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `generic_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `other_record_value` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_non_target` | 4 | 0.500 | 0.000 | 0.250 | 0.031 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_class:other_semantic_value` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:top_non_target` | `route_token_source` | `top_class:punctuation_or_stop` | 13 | 0.462 | 0.000 | 0.385 | 0.058 | -0.029 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:format_or_schema` | 24 | 0.458 | 0.083 | 0.375 | 0.026 | -0.016 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_non_target` | 24 | 0.417 | 0.042 | 0.458 | 0.057 | -0.052 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:format_or_schema` | 24 | 0.417 | 0.042 | 0.458 | 0.052 | -0.052 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `top_class:echo_object_or_relation` | 5 | 0.400 | 0.000 | 0.200 | 0.050 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `top_class:echo_object_or_relation` | 5 | 0.400 | 0.000 | 0.600 | 0.050 | -0.050 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_vocab` | 20 | 0.400 | 0.000 | 0.450 | 0.044 | -0.069 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_semantic_value` | 15 | 0.400 | 0.000 | 0.467 | 0.042 | -0.067 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:object_relation_echo` | `echo_source` | `top_class:echo_object_or_relation` | 5 | 0.400 | 0.000 | 0.200 | 0.025 | 0.000 | `route_source_release_without_target_drop` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:punctuation_or_stop` | 15 | 0.200 | 0.067 | 0.133 | 0.033 | -0.058 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_vocab` | 20 | 0.150 | 0.050 | 0.150 | 0.031 | -0.034 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:other_vocab` | 20 | 0.150 | 0.050 | 0.150 | 0.028 | -0.050 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:other_vocab` | 20 | 0.150 | 0.050 | 0.150 | 0.025 | -0.050 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:recipient_answer` | 7 | 0.143 | 0.000 | 0.000 | 0.018 | 0.009 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:recipient_answer` | 7 | 0.143 | 0.000 | 0.000 | 0.000 | -0.027 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:other_semantic_value` | 16 | 0.125 | 0.062 | 0.188 | 0.070 | -0.062 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_semantic_value` | 16 | 0.125 | 0.062 | 0.188 | 0.062 | -0.043 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:other_semantic_value` | 16 | 0.125 | 0.062 | 0.188 | 0.055 | -0.062 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:echo_object_or_relation` | 17 | 0.118 | 0.059 | 0.176 | 0.020 | -0.044 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:echo_object_or_relation` | 17 | 0.118 | 0.059 | 0.176 | 0.017 | -0.059 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:echo_object_or_relation` | 17 | 0.118 | 0.059 | 0.176 | 0.011 | -0.062 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_non_target` | 24 | 0.083 | 0.042 | 0.125 | 0.047 | -0.044 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_non_target` | 24 | 0.083 | 0.042 | 0.125 | 0.039 | -0.026 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_non_target` | 24 | 0.083 | 0.042 | 0.125 | 0.036 | -0.042 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `object_relation_echo` | 24 | 0.083 | 0.042 | 0.125 | 0.010 | -0.042 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `object_relation_echo` | 24 | 0.083 | 0.042 | 0.125 | 0.010 | -0.026 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `object_relation_echo` | 24 | 0.083 | 0.042 | 0.125 | 0.005 | -0.044 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `generic_answer` | 24 | 0.083 | 0.042 | 0.125 | -0.008 | -0.042 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:punctuation_or_stop` | 15 | 0.067 | 0.067 | 0.133 | 0.029 | -0.033 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:other_vocab` | 20 | 0.050 | 0.050 | 0.050 | -0.019 | 0.028 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:other_vocab` | 20 | 0.050 | 0.050 | 0.050 | -0.025 | 0.044 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_vocab` | 20 | 0.050 | 0.050 | 0.000 | -0.028 | 0.031 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_non_target` | 24 | 0.042 | 0.000 | 0.083 | 0.008 | -0.005 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `generic_answer` | 24 | 0.042 | 0.000 | 0.083 | 0.007 | -0.005 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `generic_answer` | 24 | 0.042 | 0.042 | 0.042 | 0.007 | 0.023 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `contrast_answer` | 24 | 0.042 | 0.042 | 0.042 | 0.006 | 0.023 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `contrast_answer` | 24 | 0.042 | 0.000 | 0.083 | 0.002 | -0.005 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `format_schema` | 24 | 0.042 | 0.042 | 0.125 | -0.001 | -0.026 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `generic_answer` | 24 | 0.042 | 0.042 | 0.125 | -0.003 | -0.044 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_non_target` | 24 | 0.042 | 0.042 | 0.042 | -0.005 | 0.023 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `contrast_answer` | 24 | 0.042 | 0.042 | 0.125 | -0.006 | -0.044 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `contrast_answer` | 24 | 0.042 | 0.042 | 0.125 | -0.008 | -0.042 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `generic_answer` | 24 | 0.042 | 0.042 | 0.125 | -0.010 | -0.026 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `contrast_answer` | 24 | 0.042 | 0.042 | 0.125 | -0.013 | -0.026 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_class:recipient_answer` | 2 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | `weak_or_unclear` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.438 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `contrast_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.438 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `generic_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.438 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.438 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:other_vocab` | 1 | 1.000 | 0.000 | 1.000 | 0.375 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:contrast_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.312 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `contrast_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.312 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `generic_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.312 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:other_vocab` | 1 | 1.000 | 0.000 | 1.000 | 0.312 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.312 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:format_or_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.312 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_non_target` | 1 | 1.000 | 0.000 | 1.000 | 0.312 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:format_or_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_non_target` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `contrast_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `generic_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:format_or_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:other_vocab` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_non_target` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `format_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `object_relation_echo` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `object_relation_echo` | 1 | 1.000 | 0.000 | 1.000 | 0.188 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `other_record_value` | 1 | 1.000 | 0.000 | 1.000 | 0.188 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 1 | 1.000 | 0.000 | 1.000 | 0.188 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 1 | 1.000 | 0.000 | 1.000 | 0.188 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `other_record_value` | 1 | 1.000 | 0.000 | 1.000 | 0.188 | -0.375 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:recipient_answer` | 4 | 1.000 | 0.000 | 1.000 | 0.156 | -0.156 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `format_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `format_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `object_relation_echo` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H14 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_class:recipient_answer` | 4 | 0.750 | 0.000 | 1.000 | 0.141 | -0.141 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `object_relation_echo` | 4 | 0.750 | 0.000 | 0.500 | 0.078 | -0.094 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_class:echo_object_or_relation` | 3 | 0.667 | 0.000 | 0.333 | 0.062 | -0.042 | `negative_target_drop_route_artifact` |

## Strict Interpretation

- If route-token sources release routes without target drop, route competition has source-family evidence distinct from target writer evidence.
- If target sources and route sources both release routes, the result supports a mixed or distributed route field.
- If same-layer control heads match the candidate heads, the effect is not specific enough to call a global suppressor.
