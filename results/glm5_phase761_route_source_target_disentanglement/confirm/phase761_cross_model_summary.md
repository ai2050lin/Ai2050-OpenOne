# Phase 761 Route Source Target Disentanglement (confirm)

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Test: remove the same candidate head contribution from target-record sources and route-token sources, then measure target drop and route release separately.

## Source Family x Route Group

| model | source family | route group | groups | route rate | target drop rate | target boost rate | route release | target drop | roles |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | `target_source` | `top_class:recipient_answer` | 8 | 0.375 | 0.000 | 0.312 | 0.031 | 0.016 | `{'negative_target_drop_route_artifact': 3, 'weak_or_unclear': 5}` |
| qwen3 | `route_token_source` | `top_class:format_or_schema` | 20 | 0.352 | 0.006 | 0.227 | 0.031 | 0.003 | `{'negative_target_drop_route_artifact': 2, 'route_source_release_without_target_drop': 5, 'weak_or_unclear': 13}` |
| qwen3 | `route_token_source` | `top_non_target` | 20 | 0.335 | 0.006 | 0.227 | 0.033 | 0.003 | `{'negative_target_drop_route_artifact': 2, 'route_source_release_without_target_drop': 4, 'weak_or_unclear': 14}` |
| qwen3 | `target_source` | `top_class:echo_object_or_relation` | 8 | 0.266 | 0.031 | 0.234 | -0.002 | 0.014 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 7}` |
| qwen3 | `route_source_union` | `top_class:format_or_schema` | 4 | 0.260 | 0.021 | 0.219 | 0.007 | 0.003 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_token_source` | `top_class:punctuation_or_stop` | 20 | 0.260 | 0.006 | 0.227 | 0.010 | 0.003 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 2, 'weak_or_unclear': 17}` |
| qwen3 | `target_source` | `top_class:format_or_schema` | 8 | 0.253 | 0.083 | 0.219 | 0.005 | 0.032 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 6}` |
| qwen3 | `format_source` | `top_class:recipient_answer` | 4 | 0.250 | 0.000 | 0.000 | 0.031 | 0.031 | `{'route_source_release_without_target_drop': 2, 'weak_or_unclear': 2}` |
| qwen3 | `echo_source` | `top_class:echo_object_or_relation` | 4 | 0.250 | 0.031 | 0.219 | 0.016 | 0.004 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| qwen3 | `broad_record_source` | `top_class:recipient_answer` | 4 | 0.250 | 0.125 | 0.125 | 0.016 | 0.062 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_source_union` | `top_class:echo_object_or_relation` | 4 | 0.250 | 0.062 | 0.344 | 0.012 | 0.000 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `broad_record_source` | `top_class:echo_object_or_relation` | 4 | 0.250 | 0.094 | 0.156 | -0.004 | 0.039 | `{'weak_or_unclear': 4}` |
| qwen3 | `target_source` | `top_non_target` | 8 | 0.247 | 0.083 | 0.219 | -0.001 | 0.032 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 6}` |
| qwen3 | `format_source` | `top_class:format_or_schema` | 4 | 0.245 | 0.005 | 0.219 | 0.010 | 0.001 | `{'weak_or_unclear': 4}` |
| qwen3 | `route_source_union` | `format_schema` | 4 | 0.245 | 0.021 | 0.219 | 0.002 | 0.003 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_token_source` | `format_schema` | 20 | 0.242 | 0.006 | 0.227 | 0.013 | 0.003 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 18}` |
| qwen3 | `target_source` | `format_schema` | 8 | 0.237 | 0.083 | 0.219 | 0.003 | 0.032 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 7}` |
| qwen3 | `route_source_union` | `top_non_target` | 4 | 0.234 | 0.021 | 0.219 | 0.003 | 0.003 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `broad_record_source` | `other_record_value` | 4 | 0.234 | 0.141 | 0.214 | -0.002 | 0.054 | `{'weak_or_unclear': 4}` |
| qwen3 | `broad_record_source` | `format_schema` | 4 | 0.234 | 0.141 | 0.214 | -0.014 | 0.054 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `broad_record_source` | `top_class:format_or_schema` | 4 | 0.229 | 0.141 | 0.214 | -0.017 | 0.054 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `target_source` | `top_class:other_vocab` | 8 | 0.225 | 0.084 | 0.203 | -0.008 | 0.036 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 6}` |
| qwen3 | `target_source` | `object_relation_echo` | 8 | 0.224 | 0.083 | 0.219 | -0.008 | 0.032 | `{'weak_or_unclear': 8}` |
| qwen3 | `broad_record_source` | `object_relation_echo` | 4 | 0.224 | 0.141 | 0.214 | -0.049 | 0.054 | `{'weak_or_unclear': 4}` |
| qwen3 | `target_source` | `top_class:punctuation_or_stop` | 8 | 0.221 | 0.083 | 0.219 | -0.004 | 0.032 | `{'weak_or_unclear': 8}` |
| qwen3 | `route_token_source` | `top_class:recipient_answer` | 16 | 0.219 | 0.000 | 0.188 | 0.020 | 0.012 | `{'negative_target_drop_route_artifact': 3, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 12}` |
| qwen3 | `format_source` | `top_class:punctuation_or_stop` | 4 | 0.219 | 0.005 | 0.219 | 0.010 | 0.001 | `{'weak_or_unclear': 4}` |
| qwen3 | `route_source_union` | `top_class:punctuation_or_stop` | 4 | 0.219 | 0.021 | 0.219 | 0.002 | 0.003 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_source_union` | `top_class:other_vocab` | 4 | 0.219 | 0.013 | 0.188 | -0.002 | 0.008 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `broad_record_source` | `top_class:punctuation_or_stop` | 4 | 0.219 | 0.141 | 0.214 | -0.017 | 0.054 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `echo_source` | `top_class:punctuation_or_stop` | 4 | 0.218 | 0.032 | 0.207 | 0.003 | 0.009 | `{'weak_or_unclear': 4}` |
| qwen3 | `format_source` | `top_non_target` | 4 | 0.214 | 0.005 | 0.219 | 0.007 | 0.001 | `{'weak_or_unclear': 4}` |
| qwen3 | `broad_record_source` | `top_non_target` | 4 | 0.214 | 0.141 | 0.214 | -0.027 | 0.054 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `route_token_source` | `top_class:other_vocab` | 16 | 0.209 | 0.005 | 0.213 | -0.000 | 0.005 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 14}` |
| qwen3 | `route_source_union` | `object_relation_echo` | 4 | 0.208 | 0.021 | 0.219 | -0.031 | 0.003 | `{'weak_or_unclear': 4}` |
| qwen3 | `broad_record_source` | `top_class:other_vocab` | 4 | 0.206 | 0.156 | 0.219 | -0.034 | 0.059 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| qwen3 | `echo_source` | `top_class:format_or_schema` | 4 | 0.202 | 0.032 | 0.207 | -0.005 | 0.009 | `{'weak_or_unclear': 4}` |
| qwen3 | `broad_record_source` | `contrast_answer` | 4 | 0.193 | 0.141 | 0.214 | 0.022 | 0.054 | `{'weak_or_unclear': 4}` |
| qwen3 | `format_source` | `top_class:other_vocab` | 4 | 0.188 | 0.006 | 0.200 | -0.002 | 0.005 | `{'weak_or_unclear': 4}` |
| qwen3 | `format_source` | `top_class:echo_object_or_relation` | 4 | 0.188 | 0.000 | 0.219 | -0.004 | 0.012 | `{'route_source_release_without_target_drop': 1, 'weak_or_unclear': 3}` |
| glm4 | `target_source` | `top_class:other_semantic_value` | 8 | 0.027 | 0.030 | 0.076 | -0.004 | 0.008 | `{'weak_or_unclear': 8}` |
| glm4 | `broad_record_source` | `top_class:other_semantic_value` | 4 | 0.023 | 0.030 | 0.091 | -0.008 | 0.012 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `top_non_target` | 8 | 0.021 | 0.021 | 0.055 | -0.002 | 0.003 | `{'weak_or_unclear': 8}` |
| glm4 | `target_source` | `top_class:other_vocab` | 8 | 0.019 | 0.026 | 0.064 | -0.006 | 0.005 | `{'weak_or_unclear': 8}` |
| glm4 | `broad_record_source` | `top_class:other_vocab` | 4 | 0.019 | 0.026 | 0.090 | -0.010 | 0.007 | `{'weak_or_unclear': 4}` |
| glm4 | `broad_record_source` | `top_class:punctuation_or_stop` | 4 | 0.019 | 0.019 | 0.056 | 0.003 | 0.007 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `top_class:punctuation_or_stop` | 4 | 0.019 | 0.000 | 0.046 | 0.000 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `top_class:echo_object_or_relation` | 8 | 0.018 | 0.029 | 0.070 | -0.000 | 0.004 | `{'weak_or_unclear': 8}` |
| glm4 | `route_source_union` | `top_class:recipient_answer` | 4 | 0.016 | 0.000 | 0.031 | 0.008 | -0.013 | `{'weak_or_unclear': 4}` |
| glm4 | `broad_record_source` | `top_non_target` | 4 | 0.016 | 0.021 | 0.073 | -0.002 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `object_relation_echo` | 8 | 0.013 | 0.021 | 0.055 | 0.000 | 0.003 | `{'weak_or_unclear': 8}` |
| glm4 | `route_source_union` | `top_class:format_or_schema` | 4 | 0.010 | 0.000 | 0.062 | 0.005 | -0.002 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `top_non_target` | 4 | 0.010 | 0.000 | 0.062 | 0.004 | -0.002 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `top_class:punctuation_or_stop` | 8 | 0.009 | 0.019 | 0.051 | 0.003 | 0.003 | `{'weak_or_unclear': 8}` |
| glm4 | `target_source` | `top_class:recipient_answer` | 8 | 0.008 | 0.000 | 0.008 | 0.004 | -0.002 | `{'weak_or_unclear': 8}` |
| glm4 | `echo_source` | `top_class:other_semantic_value` | 4 | 0.008 | 0.000 | 0.062 | -0.003 | 0.010 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `top_class:other_semantic_value` | 4 | 0.008 | 0.000 | 0.076 | 0.000 | 0.003 | `{'weak_or_unclear': 4}` |
| glm4 | `format_source` | `top_class:other_semantic_value` | 4 | 0.008 | 0.000 | 0.045 | -0.001 | 0.009 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `top_class:echo_object_or_relation` | 4 | 0.007 | 0.000 | 0.066 | -0.002 | -0.000 | `{'weak_or_unclear': 4}` |
| glm4 | `broad_record_source` | `top_class:echo_object_or_relation` | 4 | 0.007 | 0.029 | 0.088 | -0.005 | 0.007 | `{'weak_or_unclear': 4}` |
| glm4 | `echo_source` | `top_non_target` | 4 | 0.006 | 0.000 | 0.050 | 0.001 | 0.004 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `contrast_answer` | 4 | 0.005 | 0.000 | 0.062 | 0.004 | -0.002 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `generic_answer` | 4 | 0.005 | 0.000 | 0.062 | 0.004 | -0.002 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `format_schema` | 4 | 0.005 | 0.000 | 0.062 | 0.004 | -0.002 | `{'weak_or_unclear': 4}` |
| glm4 | `format_source` | `top_class:format_or_schema` | 4 | 0.005 | 0.000 | 0.031 | 0.002 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `generic_answer` | 8 | 0.005 | 0.021 | 0.055 | 0.001 | 0.003 | `{'weak_or_unclear': 8}` |
| glm4 | `broad_record_source` | `top_class:format_or_schema` | 4 | 0.005 | 0.021 | 0.073 | 0.001 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `top_class:format_or_schema` | 8 | 0.005 | 0.021 | 0.055 | 0.001 | 0.003 | `{'weak_or_unclear': 8}` |
| glm4 | `format_source` | `top_non_target` | 4 | 0.005 | 0.000 | 0.031 | 0.000 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `target_source` | `format_schema` | 8 | 0.005 | 0.021 | 0.055 | -0.000 | 0.003 | `{'weak_or_unclear': 8}` |
| glm4 | `broad_record_source` | `format_schema` | 4 | 0.005 | 0.021 | 0.073 | -0.001 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `route_source_union` | `object_relation_echo` | 4 | 0.005 | 0.000 | 0.062 | -0.001 | -0.002 | `{'weak_or_unclear': 4}` |
| glm4 | `broad_record_source` | `object_relation_echo` | 4 | 0.005 | 0.021 | 0.073 | -0.003 | 0.006 | `{'weak_or_unclear': 4}` |
| glm4 | `route_token_source` | `top_class:other_semantic_value` | 20 | 0.005 | 0.000 | 0.042 | 0.005 | 0.018 | `{'weak_or_unclear': 20}` |
| glm4 | `route_token_source` | `top_class:format_or_schema` | 20 | 0.005 | 0.000 | 0.031 | -0.000 | 0.012 | `{'weak_or_unclear': 20}` |
| glm4 | `route_token_source` | `top_non_target` | 20 | 0.004 | 0.000 | 0.031 | -0.004 | 0.012 | `{'weak_or_unclear': 20}` |
| glm4 | `target_source` | `contrast_answer` | 8 | 0.003 | 0.021 | 0.055 | 0.001 | 0.003 | `{'weak_or_unclear': 8}` |
| glm4 | `route_token_source` | `format_schema` | 20 | 0.002 | 0.000 | 0.031 | -0.003 | 0.012 | `{'weak_or_unclear': 20}` |
| glm4 | `format_source` | `top_class:recipient_answer` | 4 | 0.000 | 0.000 | 0.000 | 0.005 | -0.001 | `{'weak_or_unclear': 4}` |
| glm4 | `broad_record_source` | `top_class:recipient_answer` | 4 | 0.000 | 0.000 | 0.031 | 0.004 | -0.002 | `{'weak_or_unclear': 4}` |
| deepseek7b | `route_token_source` | `top_class:recipient_answer` | 20 | 0.444 | 0.007 | 0.544 | 0.123 | -0.118 | `{'negative_target_drop_route_artifact': 9, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 10}` |
| deepseek7b | `route_source_union` | `top_class:recipient_answer` | 4 | 0.306 | 0.000 | 0.333 | 0.033 | -0.036 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `route_token_source` | `object_relation_echo` | 24 | 0.294 | 0.088 | 0.361 | 0.000 | -0.024 | `{'negative_target_drop_route_artifact': 5, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 18}` |
| deepseek7b | `route_token_source` | `top_non_target` | 24 | 0.292 | 0.088 | 0.361 | 0.014 | -0.024 | `{'negative_target_drop_route_artifact': 8, 'weak_or_unclear': 16}` |
| deepseek7b | `route_token_source` | `top_class:other_vocab` | 24 | 0.289 | 0.089 | 0.363 | 0.034 | -0.025 | `{'negative_target_drop_route_artifact': 7, 'weak_or_unclear': 17}` |
| deepseek7b | `route_source_union` | `top_class:other_semantic_value` | 4 | 0.286 | 0.196 | 0.286 | 0.025 | 0.032 | `{'negative_target_drop_route_artifact': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `route_token_source` | `top_class:format_or_schema` | 24 | 0.285 | 0.088 | 0.361 | 0.015 | -0.024 | `{'negative_target_drop_route_artifact': 6, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 17}` |
| deepseek7b | `target_source` | `top_class:echo_object_or_relation` | 8 | 0.282 | 0.359 | 0.163 | -0.008 | 0.172 | `{'target_source_with_route_release': 1, 'target_source_writer': 3, 'weak_or_unclear': 4}` |
| deepseek7b | `route_token_source` | `top_class:other_semantic_value` | 12 | 0.279 | 0.166 | 0.237 | 0.004 | 0.047 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 3, 'weak_or_unclear': 8}` |
| deepseek7b | `target_source` | `object_relation_echo` | 8 | 0.276 | 0.362 | 0.154 | -0.007 | 0.181 | `{'target_source_with_route_release': 1, 'target_source_writer': 3, 'weak_or_unclear': 4}` |
| deepseek7b | `route_source_union` | `top_class:format_or_schema` | 4 | 0.276 | 0.094 | 0.307 | -0.008 | -0.003 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 2}` |
| deepseek7b | `route_token_source` | `format_schema` | 24 | 0.273 | 0.088 | 0.361 | -0.007 | -0.024 | `{'negative_target_drop_route_artifact': 6, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 17}` |
| deepseek7b | `target_source` | `top_non_target` | 8 | 0.268 | 0.362 | 0.154 | -0.024 | 0.181 | `{'target_source_writer': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `route_token_source` | `top_class:punctuation_or_stop` | 24 | 0.266 | 0.088 | 0.361 | 0.017 | -0.024 | `{'negative_target_drop_route_artifact': 6, 'weak_or_unclear': 18}` |
| deepseek7b | `echo_source` | `top_class:other_vocab` | 4 | 0.261 | 0.065 | 0.315 | 0.011 | -0.015 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `route_source_union` | `top_non_target` | 4 | 0.260 | 0.094 | 0.307 | -0.003 | -0.003 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `route_token_source` | `generic_answer` | 24 | 0.254 | 0.088 | 0.361 | 0.045 | -0.024 | `{'negative_target_drop_route_artifact': 6, 'weak_or_unclear': 18}` |
| deepseek7b | `route_source_union` | `top_class:punctuation_or_stop` | 4 | 0.250 | 0.094 | 0.307 | 0.012 | -0.003 | `{'negative_target_drop_route_artifact': 1, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 2}` |
| deepseek7b | `route_token_source` | `contrast_answer` | 24 | 0.246 | 0.088 | 0.361 | 0.034 | -0.024 | `{'negative_target_drop_route_artifact': 4, 'route_source_release_without_target_drop': 1, 'weak_or_unclear': 19}` |
| deepseek7b | `broad_record_source` | `top_non_target` | 4 | 0.240 | 0.432 | 0.146 | -0.032 | 0.244 | `{'target_source_writer': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `route_source_union` | `generic_answer` | 4 | 0.240 | 0.094 | 0.307 | 0.001 | -0.003 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `route_source_union` | `top_class:other_vocab` | 4 | 0.234 | 0.096 | 0.303 | 0.009 | -0.002 | `{'weak_or_unclear': 4}` |
| deepseek7b | `echo_source` | `top_non_target` | 4 | 0.234 | 0.064 | 0.314 | -0.014 | -0.014 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `broad_record_source` | `top_class:format_or_schema` | 4 | 0.229 | 0.432 | 0.146 | -0.034 | 0.244 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 2, 'weak_or_unclear': 1}` |
| deepseek7b | `route_source_union` | `format_schema` | 4 | 0.229 | 0.094 | 0.307 | 0.007 | -0.003 | `{'weak_or_unclear': 4}` |
| deepseek7b | `target_source` | `top_class:format_or_schema` | 8 | 0.224 | 0.362 | 0.154 | -0.032 | 0.181 | `{'target_source_writer': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `echo_source` | `generic_answer` | 4 | 0.223 | 0.064 | 0.314 | 0.019 | -0.014 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `echo_source` | `top_class:punctuation_or_stop` | 4 | 0.223 | 0.064 | 0.314 | 0.013 | -0.014 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `format_source` | `top_class:recipient_answer` | 4 | 0.222 | 0.000 | 0.222 | 0.030 | -0.028 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `target_source` | `top_class:other_vocab` | 8 | 0.221 | 0.370 | 0.157 | -0.017 | 0.184 | `{'target_source_writer': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `route_source_union` | `other_record_value` | 4 | 0.219 | 0.094 | 0.307 | -0.006 | -0.003 | `{'weak_or_unclear': 4}` |
| deepseek7b | `route_source_union` | `top_class:echo_object_or_relation` | 4 | 0.218 | 0.083 | 0.340 | -0.036 | -0.015 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `echo_source` | `top_class:other_semantic_value` | 4 | 0.214 | 0.107 | 0.232 | -0.011 | 0.037 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 3}` |
| deepseek7b | `target_source` | `top_class:other_semantic_value` | 8 | 0.214 | 0.518 | 0.152 | -0.146 | 0.287 | `{'negative_target_drop_route_artifact': 1, 'target_source_writer': 4, 'weak_or_unclear': 3}` |
| deepseek7b | `target_source` | `generic_answer` | 8 | 0.214 | 0.362 | 0.154 | -0.016 | 0.181 | `{'target_source_writer': 4, 'weak_or_unclear': 4}` |
| deepseek7b | `echo_source` | `other_record_value` | 4 | 0.213 | 0.064 | 0.314 | 0.013 | -0.014 | `{'weak_or_unclear': 4}` |
| deepseek7b | `broad_record_source` | `object_relation_echo` | 4 | 0.203 | 0.432 | 0.146 | -0.062 | 0.244 | `{'target_source_writer': 2, 'weak_or_unclear': 2}` |
| deepseek7b | `route_token_source` | `other_record_value` | 24 | 0.203 | 0.088 | 0.361 | 0.004 | -0.024 | `{'negative_target_drop_route_artifact': 5, 'weak_or_unclear': 19}` |
| deepseek7b | `echo_source` | `top_class:format_or_schema` | 4 | 0.202 | 0.064 | 0.314 | -0.013 | -0.014 | `{'weak_or_unclear': 4}` |
| deepseek7b | `route_token_source` | `top_class:echo_object_or_relation` | 20 | 0.201 | 0.104 | 0.223 | -0.036 | 0.030 | `{'negative_target_drop_route_artifact': 1, 'weak_or_unclear': 19}` |

## Top Cells

| model | head | kind | source | family | route | n | route rate | target drop rate | target boost rate | route release | target drop | role |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:format_or_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_non_target` | 1 | 1.000 | 0.000 | 1.000 | 0.250 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `format_schema` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `other_record_value` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:format_or_schema` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_non_target` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:format_or_schema` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_non_target` | 1 | 1.000 | 0.000 | 0.000 | 0.125 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:recipient_answer` | 2 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:recipient_answer` | 2 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `contrast_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `format_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `generic_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `other_record_value` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:punctuation_or_stop` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `route_src:generic_answer` | `route_token_source` | `top_class:recipient_answer` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.125 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:recipient_answer` | 2 | 0.500 | 0.000 | 0.500 | 0.125 | -0.062 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:recipient_answer` | 2 | 0.500 | 0.000 | 0.000 | 0.062 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:other_record_value` | `route_token_source` | `top_class:recipient_answer` | 2 | 0.500 | 0.000 | 0.000 | 0.062 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `top_class:recipient_answer` | 2 | 0.500 | 0.000 | 0.000 | 0.062 | 0.000 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H23 | `phase755_top_candidate` | `route_src:format_schema` | `format_source` | `top_class:recipient_answer` | 2 | 0.500 | 0.000 | 0.000 | 0.062 | 0.062 | `route_source_release_without_target_drop` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `top_class:recipient_answer` | 2 | 0.500 | 0.000 | 0.000 | 0.062 | 0.000 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `target_value_tokens` | `target_source` | `top_class:echo_object_or_relation` | 8 | 0.500 | 0.000 | 0.375 | 0.062 | -0.016 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H4 | `same_layer_control_head` | `records_all` | `broad_record_source` | `top_class:recipient_answer` | 2 | 0.500 | 0.500 | 0.000 | 0.062 | 0.125 | `weak_or_unclear` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_non_target` | 6 | 0.500 | 0.000 | 0.333 | 0.042 | -0.021 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `target_record_line` | `target_source` | `top_class:recipient_answer` | 2 | 0.500 | 0.000 | 0.500 | 0.000 | 0.000 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:other_vocab` | 40 | 0.425 | 0.050 | 0.400 | 0.028 | -0.016 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:format_or_schema` | 48 | 0.417 | 0.062 | 0.375 | 0.042 | -0.005 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_vocab` | 40 | 0.400 | 0.025 | 0.425 | 0.031 | -0.034 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H28 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_class:other_vocab` | 5 | 0.400 | 0.000 | 0.400 | 0.025 | -0.025 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_non_target` | 48 | 0.396 | 0.062 | 0.417 | 0.047 | -0.021 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:format_or_schema` | 48 | 0.396 | 0.062 | 0.417 | 0.044 | -0.021 | `negative_target_drop_route_artifact` |
| qwen3 | L33:attn_out:H15 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `format_schema` | 48 | 0.396 | 0.062 | 0.375 | 0.039 | -0.005 | `negative_target_drop_route_artifact` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:other_semantic_value` | 33 | 0.091 | 0.061 | 0.152 | 0.017 | -0.013 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_semantic_value` | 33 | 0.091 | 0.061 | 0.212 | 0.015 | -0.017 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:punctuation_or_stop` | 27 | 0.074 | 0.037 | 0.111 | 0.016 | -0.030 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:recipient_answer` | 16 | 0.062 | 0.000 | 0.000 | 0.016 | 0.004 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_non_target` | 48 | 0.062 | 0.042 | 0.125 | 0.016 | -0.016 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_non_target` | 48 | 0.062 | 0.042 | 0.146 | 0.009 | -0.013 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:recipient_answer` | 16 | 0.062 | 0.000 | 0.000 | 0.000 | -0.004 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:other_semantic_value` | 33 | 0.061 | 0.061 | 0.182 | 0.004 | -0.019 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_semantic_value` | 33 | 0.061 | 0.061 | 0.061 | -0.019 | 0.023 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:echo_object_or_relation` | 34 | 0.059 | 0.059 | 0.176 | 0.005 | -0.022 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:echo_object_or_relation` | 34 | 0.059 | 0.059 | 0.206 | 0.003 | -0.022 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:other_vocab` | 39 | 0.051 | 0.051 | 0.154 | 0.005 | -0.018 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:other_vocab` | 39 | 0.051 | 0.051 | 0.154 | 0.002 | -0.019 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_vocab` | 39 | 0.051 | 0.051 | 0.179 | 0.001 | -0.018 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_non_target` | 48 | 0.042 | 0.042 | 0.125 | 0.007 | -0.017 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_record_line` | `target_source` | `object_relation_echo` | 48 | 0.042 | 0.042 | 0.125 | 0.005 | -0.017 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `object_relation_echo` | 48 | 0.042 | 0.042 | 0.146 | 0.002 | -0.013 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_non_target` | 48 | 0.042 | 0.042 | 0.042 | -0.014 | 0.014 | `weak_or_unclear` |
| glm4 | L34:attn_out:H17 | `same_layer_control_head` | `route_src:top_non_target` | `route_token_source` | `top_class:other_semantic_value` | 25 | 0.040 | 0.000 | 0.040 | -0.005 | 0.013 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:punctuation_or_stop` | 27 | 0.037 | 0.037 | 0.185 | 0.016 | -0.025 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:punctuation_or_stop` | 27 | 0.037 | 0.000 | 0.000 | 0.014 | 0.002 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `target_record_line` | `target_source` | `top_class:punctuation_or_stop` | 27 | 0.037 | 0.000 | 0.000 | 0.000 | 0.009 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:punctuation_or_stop` | 27 | 0.037 | 0.000 | 0.074 | -0.007 | 0.007 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `route_src:top_non_target` | `route_token_source` | `top_class:format_or_schema` | 31 | 0.032 | 0.000 | 0.097 | 0.000 | 0.008 | `weak_or_unclear` |
| glm4 | L34:attn_out:H17 | `same_layer_control_head` | `route_src:top_non_target` | `route_token_source` | `top_non_target` | 31 | 0.032 | 0.000 | 0.032 | -0.004 | 0.008 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `route_src:object_relation_echo` | `echo_source` | `top_class:other_semantic_value` | 32 | 0.031 | 0.000 | 0.094 | 0.002 | 0.006 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:other_semantic_value` | 33 | 0.030 | 0.000 | 0.061 | 0.002 | 0.000 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `route_src:other_record_value` | `route_token_source` | `top_class:other_semantic_value` | 33 | 0.030 | 0.000 | 0.091 | 0.002 | 0.008 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `route_src:generic_answer` | `route_token_source` | `top_class:other_semantic_value` | 33 | 0.030 | 0.000 | 0.091 | 0.000 | 0.017 | `weak_or_unclear` |
| glm4 | L34:attn_out:H17 | `same_layer_control_head` | `route_src:format_schema` | `format_source` | `top_class:other_semantic_value` | 33 | 0.030 | 0.000 | 0.091 | 0.000 | 0.009 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:echo_object_or_relation` | 34 | 0.029 | 0.000 | 0.059 | 0.006 | -0.006 | `weak_or_unclear` |
| glm4 | L34:attn_out:H4 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:echo_object_or_relation` | 34 | 0.029 | 0.059 | 0.147 | 0.002 | -0.020 | `weak_or_unclear` |
| glm4 | L35:attn_out:H10 | `same_layer_control_head` | `target_record_line` | `target_source` | `top_class:echo_object_or_relation` | 34 | 0.029 | 0.000 | 0.000 | -0.006 | 0.009 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_record_line` | `target_source` | `top_class:other_vocab` | 39 | 0.026 | 0.051 | 0.077 | -0.018 | 0.019 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `top_class:other_vocab` | 39 | 0.026 | 0.051 | 0.051 | -0.019 | 0.018 | `weak_or_unclear` |
| glm4 | L35:attn_out:H29 | `phase755_top_candidate` | `records_all` | `broad_record_source` | `top_class:other_vocab` | 39 | 0.026 | 0.051 | 0.128 | -0.030 | 0.029 | `weak_or_unclear` |
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
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `route_src:top_class:recipient_answer` | `route_token_source` | `format_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.312 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `format_schema` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:top_class:recipient_answer` | `route_token_source` | `object_relation_echo` | 1 | 1.000 | 0.000 | 1.000 | 0.125 | -0.250 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H24 | `phase755_top_candidate` | `route_src:any_route` | `route_source_union` | `top_class:recipient_answer` | 9 | 0.667 | 0.000 | 0.556 | 0.090 | -0.076 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `object_relation_echo` | 6 | 0.667 | 0.000 | 0.500 | 0.052 | -0.062 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H9 | `same_layer_control_head` | `route_src:contrast_answer` | `route_token_source` | `top_class:echo_object_or_relation` | 5 | 0.600 | 0.000 | 0.400 | 0.037 | -0.025 | `negative_target_drop_route_artifact` |
| deepseek7b | L22:attn_out:H1 | `phase755_top_candidate` | `target_value_tokens` | `target_source` | `object_relation_echo` | 48 | 0.500 | 0.583 | 0.042 | 0.130 | 0.353 | `target_source_with_route_release` |

## Strict Interpretation

- If route-token sources release routes without target drop, route competition has source-family evidence distinct from target writer evidence.
- If target sources and route sources both release routes, the result supports a mixed or distributed route field.
- If same-layer control heads match the candidate heads, the effect is not specific enough to call a global suppressor.
