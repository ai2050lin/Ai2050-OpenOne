# Phase 775 Semantic Latent Route vs Output Closure (confirm)

- Status: `complete`
- Test: separate value-pool latent selection from open-vocabulary output closure.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Prompt Observation Summary

| model | variant | rows | cases | base top1 | latent pool hit | pool top1 | base rank | pool rank | base margin | pool margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | 3 | 3 | 0.000 | 0.667 | 0.667 | 1941.667 | 3.000 | 3.031 | 2.760 |
| qwen3 | `with_candidate_list` | 3 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 8.333 | 5.958 |
| qwen3 | `constrained_free_prompt` | 3 | 3 | 0.000 | 0.667 | 0.667 | 828.333 | 2.667 | 2.479 | 2.375 |
| glm4 | `without_candidate_list` | 3 | 3 | 0.000 | 0.667 | 0.667 | 72.667 | 1.333 | 4.619 | 2.271 |
| glm4 | `with_candidate_list` | 3 | 3 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 2.698 | 1.875 |
| glm4 | `constrained_free_prompt` | 3 | 3 | 0.000 | 1.000 | 1.000 | 65.667 | 1.000 | 4.676 | 2.646 |
| deepseek7b | `without_candidate_list` | 3 | 3 | 0.000 | 0.667 | 0.667 | 6958.667 | 1.333 | 3.947 | 2.394 |
| deepseek7b | `with_candidate_list` | 3 | 3 | 0.333 | 0.333 | 0.667 | 3.000 | 1.333 | 4.552 | 2.792 |
| deepseek7b | `constrained_free_prompt` | 3 | 3 | 0.000 | 0.667 | 0.667 | 2363.000 | 1.333 | 4.255 | 2.240 |

## Component Effect Summary

| model | variant | kind | rows | cases | target drop | margin drop | target rank delta | pool rank delta | pool margin drop | pool top1 loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `source_group_top_component` | 21 | 3 | 0.058 | 0.097 | 62.048 | 0.000 | 0.100 | 0.000 |
| qwen3 | `without_candidate_list` | `same_layer_control_head` | 21 | 3 | 0.003 | 0.039 | 32.952 | 0.000 | 0.042 | 0.000 |
| qwen3 | `with_candidate_list` | `source_group_top_component` | 24 | 3 | 0.214 | 0.268 | 0.000 | 0.000 | 0.281 | 0.000 |
| qwen3 | `with_candidate_list` | `same_layer_control_head` | 24 | 3 | 0.036 | -0.018 | 0.000 | 0.000 | -0.005 | 0.000 |
| qwen3 | `constrained_free_prompt` | `source_group_top_component` | 21 | 3 | -0.158 | -0.024 | -15.286 | -0.048 | -0.015 | 0.000 |
| qwen3 | `constrained_free_prompt` | `same_layer_control_head` | 21 | 3 | -0.045 | -0.033 | 25.619 | 0.000 | -0.030 | 0.000 |
| glm4 | `without_candidate_list` | `source_group_top_component` | 21 | 3 | -0.010 | -0.002 | -1.238 | 0.000 | 0.010 | 0.000 |
| glm4 | `without_candidate_list` | `same_layer_control_head` | 21 | 3 | -0.003 | -0.003 | -0.429 | 0.000 | 0.006 | 0.000 |
| glm4 | `with_candidate_list` | `same_layer_control_head` | 24 | 3 | 0.010 | 0.026 | 0.000 | 0.000 | 0.026 | 0.000 |
| glm4 | `with_candidate_list` | `source_group_top_component` | 24 | 3 | 0.010 | 0.039 | 0.000 | 0.000 | 0.026 | 0.000 |
| glm4 | `constrained_free_prompt` | `same_layer_control_head` | 21 | 3 | -0.003 | 0.007 | -0.381 | 0.000 | 0.007 | 0.000 |
| glm4 | `constrained_free_prompt` | `source_group_top_component` | 21 | 3 | -0.010 | -0.002 | -0.952 | 0.000 | -0.004 | 0.000 |
| deepseek7b | `without_candidate_list` | `source_group_top_component` | 21 | 3 | 0.091 | 0.285 | -1920.429 | 0.143 | 0.243 | 0.000 |
| deepseek7b | `without_candidate_list` | `same_layer_control_head` | 21 | 3 | 0.057 | 0.018 | -2402.381 | 0.000 | 0.103 | 0.000 |
| deepseek7b | `with_candidate_list` | `source_group_top_component` | 24 | 3 | 0.109 | 0.049 | 0.167 | 0.000 | 0.031 | 0.000 |
| deepseek7b | `with_candidate_list` | `same_layer_control_head` | 24 | 3 | 0.042 | 0.017 | 0.000 | 0.000 | 0.013 | 0.000 |
| deepseek7b | `constrained_free_prompt` | `source_group_top_component` | 21 | 3 | 0.101 | 0.171 | -254.952 | 0.000 | 0.120 | 0.000 |
| deepseek7b | `constrained_free_prompt` | `same_layer_control_head` | 21 | 3 | -0.001 | -0.006 | -24.190 | 0.000 | -0.001 | 0.000 |

## Prompt Source Family Summary

| model | variant | family | kind | rows | cases | target drop | pool margin drop | direct boost | route suppression |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | 0.094 | 0.177 | 0.072 | 0.314 |
| qwen3 | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.052 | 0.156 | 0.031 | 0.361 |
| qwen3 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | 0.052 | 0.115 | -0.038 | 0.470 |
| qwen3 | `without_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.021 | 0.104 | 0.208 | 0.254 |
| qwen3 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | 0.000 | 0.083 | 0.108 | 0.221 |
| qwen3 | `without_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.010 | 0.073 | -0.052 | 0.000 |
| qwen3 | `without_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.031 | 0.052 | 0.011 | 0.028 |
| qwen3 | `without_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.031 | 0.052 | 0.027 | 0.078 |
| qwen3 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.021 | 0.042 | 0.026 | 0.240 |
| qwen3 | `without_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.042 | -0.018 | 0.045 |
| qwen3 | `without_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | -0.010 | 0.031 | 0.000 | 0.001 |
| qwen3 | `without_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | -0.010 | 0.031 | 0.000 | 0.001 |
| qwen3 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.167 | 0.021 | 0.477 | 0.206 |
| qwen3 | `without_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | -0.031 | 0.010 | -0.006 | 0.003 |
| qwen3 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 3 | 3 | 1.250 | 2.125 | 18.892 | 1.183 |
| qwen3 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | 0.042 | 0.125 | -0.046 | 0.460 |
| qwen3 | `with_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.125 | 0.083 | 0.179 | 0.240 |
| qwen3 | `with_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.083 | 0.042 | -0.006 | 0.021 |
| qwen3 | `with_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.042 | 0.042 | -0.042 | 0.018 |
| qwen3 | `with_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.042 | 0.042 | -0.009 | 0.001 |
| qwen3 | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.042 | 0.042 | 0.056 | 0.187 |
| qwen3 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | 0.042 | 0.042 | 0.063 | 0.220 |
| qwen3 | `with_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.083 | 0.000 | 0.001 | 0.000 |
| qwen3 | `with_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.042 | 0.000 | -0.004 | 0.008 |
| qwen3 | `with_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.000 | 0.010 | 0.003 |
| qwen3 | `with_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.000 | 0.000 | 0.008 |
| qwen3 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.292 | -0.042 | 0.800 | 0.116 |
| qwen3 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | -0.042 | -0.042 | 0.026 | 0.142 |
| qwen3 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | -0.042 | -0.083 | 0.149 | 0.127 |
| qwen3 | `with_candidate_list` | `candidate_protocol` | `same_layer_control_head` | 3 | 3 | 0.000 | -0.167 | -1.055 | 0.140 |
| qwen3 | `constrained_free_prompt` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.000 | 0.042 | 0.219 | 0.204 |
| qwen3 | `constrained_free_prompt` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | -0.042 | 0.042 | 0.046 | 0.550 |
| qwen3 | `constrained_free_prompt` | `protocol_format` | `source_group_top_component` | 3 | 3 | 0.000 | 0.021 | 0.131 | 0.054 |
| qwen3 | `constrained_free_prompt` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.021 | -0.012 | 0.007 |
| qwen3 | `constrained_free_prompt` | `semantic_relation` | `source_group_top_component` | 3 | 3 | -0.083 | 0.021 | 0.061 | 0.223 |
| qwen3 | `constrained_free_prompt` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | -0.021 | 0.000 | 0.000 | 0.001 |
| qwen3 | `constrained_free_prompt` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | -0.083 | 0.000 | -0.045 | 0.460 |
| qwen3 | `constrained_free_prompt` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | -0.021 | -0.021 | -0.008 | 0.006 |
| qwen3 | `constrained_free_prompt` | `protocol_format` | `same_layer_control_head` | 3 | 3 | -0.062 | -0.021 | -0.039 | 0.015 |
| qwen3 | `constrained_free_prompt` | `query_frame` | `same_layer_control_head` | 3 | 3 | -0.042 | -0.042 | 0.014 | 0.049 |
| qwen3 | `constrained_free_prompt` | `semantic_object` | `source_group_top_component` | 3 | 3 | -0.083 | -0.042 | 0.022 | 0.496 |
| qwen3 | `constrained_free_prompt` | `output_prefix` | `same_layer_control_head` | 3 | 3 | -0.083 | -0.062 | -0.000 | 0.002 |
| qwen3 | `constrained_free_prompt` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | -0.083 | -0.083 | -0.004 | 0.025 |
| qwen3 | `constrained_free_prompt` | `output_prefix` | `source_group_top_component` | 3 | 3 | -0.812 | -0.188 | 0.108 | 0.578 |
| glm4 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.000 | 0.031 | 0.007 | 0.052 |
| glm4 | `without_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.021 | 0.001 | 0.003 |
| glm4 | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.000 | 0.021 | 0.031 | 0.056 |
| glm4 | `without_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.021 | -0.000 | 0.001 |
| glm4 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | -0.021 | 0.021 | 0.030 | 0.032 |
| glm4 | `without_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.010 | -0.008 | 0.158 |
| glm4 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.000 | 0.010 | 0.030 | 0.040 |
| glm4 | `without_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.010 | 0.001 | 0.003 |
| glm4 | `without_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.010 | 0.006 | 0.003 |
| glm4 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | -0.010 | 0.010 | -0.019 | 0.228 |
| glm4 | `without_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | -0.010 | 0.000 | 0.008 | 0.037 |
| glm4 | `without_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | -0.010 | -0.010 | -0.001 | 0.004 |
| glm4 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | -0.031 | -0.021 | -0.002 | 0.033 |
| glm4 | `without_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | -0.031 | -0.021 | -0.002 | 0.005 |
| glm4 | `with_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.062 | -0.000 | 0.004 |
| glm4 | `with_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.042 | 0.042 | 0.027 | 0.026 |
| glm4 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | 0.021 | 0.042 | -0.019 | 0.225 |
| glm4 | `with_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.042 | -0.000 | 0.000 |
| glm4 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.000 | 0.042 | 0.061 | 0.000 |
| glm4 | `with_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.042 | -0.015 | 0.209 |
| glm4 | `with_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.042 | 0.011 | 0.001 |
| glm4 | `with_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.021 | -0.000 | 0.001 |
| glm4 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.021 | 0.021 | 0.011 | 0.037 |
| glm4 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | 0.021 | 0.021 | 0.023 | 0.022 |
| glm4 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | 0.000 | 0.021 | -0.003 | 0.040 |
| glm4 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 3 | 3 | -0.021 | 0.021 | 0.255 | 0.182 |
| glm4 | `with_candidate_list` | `candidate_protocol` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.000 | 0.002 | 0.008 |
| glm4 | `with_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.000 | 0.002 | 0.001 |
| glm4 | `with_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.000 | -0.000 | 0.003 |
| glm4 | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.000 | 0.000 | 0.031 | 0.051 |
| glm4 | `constrained_free_prompt` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.010 | 0.042 | 0.009 | 0.067 |
| glm4 | `constrained_free_prompt` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.021 | -0.001 | 0.004 |
| glm4 | `constrained_free_prompt` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.010 | 0.005 | 0.007 |
| glm4 | `constrained_free_prompt` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.010 | 0.001 | 0.002 |
| glm4 | `constrained_free_prompt` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.010 | -0.001 | 0.003 |
| glm4 | `constrained_free_prompt` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.010 | -0.000 | 0.001 |
| glm4 | `constrained_free_prompt` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | 0.000 | 0.000 | -0.023 | 0.230 |
| glm4 | `constrained_free_prompt` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.000 | 0.005 | 0.002 |
| glm4 | `constrained_free_prompt` | `query_frame` | `source_group_top_component` | 3 | 3 | -0.010 | 0.000 | 0.013 | 0.037 |
| glm4 | `constrained_free_prompt` | `output_prefix` | `source_group_top_component` | 3 | 3 | -0.010 | -0.010 | 0.039 | 0.020 |
| glm4 | `constrained_free_prompt` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | -0.010 | -0.010 | 0.023 | 0.068 |
| glm4 | `constrained_free_prompt` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | -0.021 | -0.010 | -0.010 | 0.098 |
| glm4 | `constrained_free_prompt` | `semantic_relation` | `source_group_top_component` | 3 | 3 | -0.021 | -0.021 | 0.021 | 0.032 |
| glm4 | `constrained_free_prompt` | `protocol_format` | `source_group_top_component` | 3 | 3 | -0.031 | -0.031 | 0.009 | 0.026 |
| deepseek7b | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.916 | 0.960 | 3.360 | 1.081 |
| deepseek7b | `without_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.779 | 0.438 | -0.014 | 0.318 |
| deepseek7b | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.155 | 0.358 | 3.346 | 1.024 |
| deepseek7b | `without_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | 0.222 | 0.287 | 0.290 | 0.736 |
| deepseek7b | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | -0.016 | 0.283 | 1.106 | 1.380 |
| deepseek7b | `without_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.138 | 0.138 | 0.140 | 0.387 |
| deepseek7b | `without_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | -0.069 | 0.090 | 0.580 | 0.788 |
| deepseek7b | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | -0.054 | 0.032 | 0.208 | 0.980 |
| deepseek7b | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | -0.156 | 0.013 | 0.341 | 0.337 |
| deepseek7b | `without_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | -0.017 | -0.004 | -0.002 | 0.102 |
| deepseek7b | `without_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | -0.247 | -0.031 | 0.016 | 0.083 |
| deepseek7b | `without_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | -0.285 | -0.032 | -0.069 | 0.260 |
| deepseek7b | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | -0.140 | -0.038 | -2.584 | 23.666 |
| deepseek7b | `without_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | -0.188 | -0.076 | 0.044 | 0.197 |
| deepseek7b | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 3 | 3 | 0.521 | 0.292 | 19.145 | 0.428 |
| deepseek7b | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.083 | 0.062 | 2.344 | 3.321 |
| deepseek7b | `with_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.062 | 0.062 | 0.014 | 0.239 |
| deepseek7b | `with_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.062 | 0.042 | 0.128 | 0.197 |
| deepseek7b | `with_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.125 | 0.021 | 0.722 | 0.941 |
| deepseek7b | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | 0.062 | 0.021 | 0.233 | 0.540 |
| deepseek7b | `with_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.042 | 0.021 | -0.023 | 0.140 |
| deepseek7b | `with_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | 0.042 | 0.021 | 0.071 | 0.269 |
| deepseek7b | `with_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | 0.062 | 0.000 | 0.046 | 0.016 |
| deepseek7b | `with_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.000 | 0.043 | 0.149 |
| deepseek7b | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.021 | 0.000 | 2.380 | 3.323 |
| deepseek7b | `with_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.062 | -0.021 | -0.052 | 0.482 |
| deepseek7b | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | 0.042 | -0.021 | 0.193 | 0.458 |
| deepseek7b | `with_candidate_list` | `candidate_protocol` | `same_layer_control_head` | 3 | 3 | -0.021 | -0.021 | -0.637 | 1.225 |
| deepseek7b | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | 0.000 | -0.042 | 0.326 | 0.316 |
| deepseek7b | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.021 | -0.083 | 0.919 | 1.183 |
| deepseek7b | `constrained_free_prompt` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.365 | 0.594 | 2.654 | 0.902 |
| deepseek7b | `constrained_free_prompt` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.156 | 0.417 | 2.810 | 0.748 |
| deepseek7b | `constrained_free_prompt` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.241 | 0.116 | 0.686 | 0.747 |
| deepseek7b | `constrained_free_prompt` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.062 | 0.031 | 0.105 | 0.261 |
| deepseek7b | `constrained_free_prompt` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.062 | 0.026 | -0.066 | 0.242 |
| deepseek7b | `constrained_free_prompt` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.018 | 0.013 | -0.011 | 0.051 |
| deepseek7b | `constrained_free_prompt` | `output_prefix` | `same_layer_control_head` | 3 | 3 | -0.003 | 0.013 | -0.027 | 0.299 |
| deepseek7b | `constrained_free_prompt` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | -0.065 | -0.003 | 0.207 | 0.604 |
| deepseek7b | `constrained_free_prompt` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | -0.049 | -0.039 | 0.014 | 0.017 |
| deepseek7b | `constrained_free_prompt` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | -0.008 | -0.049 | -0.634 | 6.885 |
| deepseek7b | `constrained_free_prompt` | `semantic_relation` | `source_group_top_component` | 3 | 3 | -0.013 | -0.049 | 0.270 | 0.530 |
| deepseek7b | `constrained_free_prompt` | `protocol_format` | `same_layer_control_head` | 3 | 3 | -0.034 | -0.049 | 0.032 | 0.240 |
| deepseek7b | `constrained_free_prompt` | `output_prefix` | `source_group_top_component` | 3 | 3 | -0.023 | -0.091 | 1.012 | 1.514 |
| deepseek7b | `constrained_free_prompt` | `protocol_format` | `source_group_top_component` | 3 | 3 | -0.013 | -0.096 | 0.160 | 0.773 |

## Strict Interpretation

- `base top1` measures open-vocabulary output closure.
- `pool top1` measures whether the target wins inside the relation value pool without using that pool as prompt evidence.
- A high `pool top1` with low `base top1` suggests latent semantic selection without readout closure.
- Component removal remains head/source-level and does not prove neuron-level coding.
