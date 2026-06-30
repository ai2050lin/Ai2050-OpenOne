# Phase 774 Candidate-List Ablation (confirm)

- Status: `complete`
- Test: compare prompts with and without allowed-values candidate list.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Prompt Variant Summary

| model | variant | kind | rows | cases | base top1 | base rank | base margin | target drop | margin drop | top1 loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `source_group_top_component` | 21 | 3 | 0.000 | 1941.667 | 3.031 | 0.058 | 0.097 | 0.000 |
| qwen3 | `without_candidate_list` | `same_layer_control_head` | 21 | 3 | 0.000 | 1941.667 | 3.031 | 0.003 | 0.039 | 0.000 |
| qwen3 | `with_candidate_list` | `source_group_top_component` | 24 | 3 | 1.000 | 1.000 | 8.333 | 0.214 | 0.268 | 0.000 |
| qwen3 | `with_candidate_list` | `same_layer_control_head` | 24 | 3 | 1.000 | 1.000 | 8.333 | 0.036 | -0.018 | 0.000 |
| glm4 | `without_candidate_list` | `same_layer_control_head` | 21 | 3 | 0.000 | 72.667 | 4.619 | -0.003 | -0.003 | 0.000 |
| glm4 | `without_candidate_list` | `source_group_top_component` | 21 | 3 | 0.000 | 72.667 | 4.619 | -0.010 | -0.002 | 0.000 |
| glm4 | `with_candidate_list` | `source_group_top_component` | 24 | 3 | 1.000 | 1.000 | 2.698 | 0.010 | 0.039 | 0.000 |
| glm4 | `with_candidate_list` | `same_layer_control_head` | 24 | 3 | 1.000 | 1.000 | 2.698 | 0.010 | 0.026 | 0.000 |
| deepseek7b | `without_candidate_list` | `source_group_top_component` | 21 | 3 | 0.000 | 6958.667 | 3.947 | 0.091 | 0.285 | 0.000 |
| deepseek7b | `without_candidate_list` | `same_layer_control_head` | 21 | 3 | 0.000 | 6958.667 | 3.947 | 0.057 | 0.018 | 0.000 |
| deepseek7b | `with_candidate_list` | `source_group_top_component` | 24 | 3 | 0.333 | 3.000 | 4.552 | 0.109 | 0.049 | 0.000 |
| deepseek7b | `with_candidate_list` | `same_layer_control_head` | 24 | 3 | 0.333 | 3.000 | 4.552 | 0.042 | 0.017 | 0.000 |

## Prompt Source Family Summary

| model | variant | family | kind | rows | cases | target drop | margin drop | direct boost | route suppression |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.167 | 0.000 | 0.477 | 0.206 |
| qwen3 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | 0.094 | 0.156 | 0.072 | 0.314 |
| qwen3 | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.052 | 0.156 | 0.031 | 0.361 |
| qwen3 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | 0.052 | 0.115 | -0.038 | 0.470 |
| qwen3 | `without_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.031 | 0.052 | 0.011 | 0.028 |
| qwen3 | `without_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.031 | 0.052 | 0.027 | 0.078 |
| qwen3 | `without_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.021 | 0.104 | 0.208 | 0.254 |
| qwen3 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.021 | 0.042 | 0.026 | 0.240 |
| qwen3 | `without_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.010 | 0.052 | -0.052 | 0.000 |
| qwen3 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | 0.000 | 0.104 | 0.108 | 0.221 |
| qwen3 | `without_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.042 | -0.018 | 0.045 |
| qwen3 | `without_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | -0.010 | 0.031 | 0.000 | 0.001 |
| qwen3 | `without_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | -0.010 | 0.031 | 0.000 | 0.001 |
| qwen3 | `without_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | -0.031 | 0.010 | -0.006 | 0.003 |
| qwen3 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 3 | 3 | 1.250 | 2.062 | 18.892 | 1.183 |
| qwen3 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.292 | -0.042 | 0.800 | 0.116 |
| qwen3 | `with_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.125 | 0.125 | 0.179 | 0.240 |
| qwen3 | `with_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.083 | 0.042 | -0.006 | 0.021 |
| qwen3 | `with_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.083 | -0.021 | 0.001 | 0.000 |
| qwen3 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | 0.042 | 0.125 | -0.046 | 0.460 |
| qwen3 | `with_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.042 | 0.042 | -0.009 | 0.001 |
| qwen3 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | 0.042 | 0.021 | 0.063 | 0.220 |
| qwen3 | `with_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.042 | 0.000 | -0.042 | 0.018 |
| qwen3 | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.042 | 0.000 | 0.056 | 0.187 |
| qwen3 | `with_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.042 | -0.021 | -0.004 | 0.008 |
| qwen3 | `with_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.000 | 0.010 | 0.003 |
| qwen3 | `with_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.000 | 0.000 | 0.008 |
| qwen3 | `with_candidate_list` | `candidate_protocol` | `same_layer_control_head` | 3 | 3 | 0.000 | -0.188 | -1.055 | 0.140 |
| qwen3 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | -0.042 | -0.062 | 0.026 | 0.142 |
| qwen3 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | -0.042 | -0.083 | 0.149 | 0.127 |
| glm4 | `without_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.003 | -0.008 | 0.158 |
| glm4 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.000 | 0.020 | 0.007 | 0.052 |
| glm4 | `without_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.012 | 0.006 | 0.003 |
| glm4 | `without_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.010 | 0.001 | 0.003 |
| glm4 | `without_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.003 | -0.000 | 0.001 |
| glm4 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.000 | 0.003 | 0.030 | 0.040 |
| glm4 | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.000 | 0.002 | 0.031 | 0.056 |
| glm4 | `without_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.000 | -0.001 | 0.001 | 0.003 |
| glm4 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | -0.010 | 0.001 | -0.019 | 0.228 |
| glm4 | `without_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | -0.010 | -0.010 | 0.008 | 0.037 |
| glm4 | `without_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | -0.010 | -0.019 | -0.001 | 0.004 |
| glm4 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | -0.021 | -0.003 | 0.030 | 0.032 |
| glm4 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | -0.031 | -0.027 | -0.002 | 0.033 |
| glm4 | `without_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | -0.031 | -0.031 | -0.002 | 0.005 |
| glm4 | `with_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.042 | 0.052 | 0.027 | 0.026 |
| glm4 | `with_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.052 | -0.000 | 0.004 |
| glm4 | `with_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.042 | -0.000 | 0.000 |
| glm4 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | 0.021 | 0.031 | 0.023 | 0.022 |
| glm4 | `with_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.021 | 0.021 | -0.000 | 0.001 |
| glm4 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | 0.021 | 0.021 | -0.019 | 0.225 |
| glm4 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.021 | 0.021 | 0.011 | 0.037 |
| glm4 | `with_candidate_list` | `candidate_protocol` | `same_layer_control_head` | 3 | 3 | 0.021 | -0.010 | 0.002 | 0.008 |
| glm4 | `with_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.052 | -0.015 | 0.209 |
| glm4 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.000 | 0.042 | 0.061 | 0.000 |
| glm4 | `with_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.031 | 0.011 | 0.001 |
| glm4 | `with_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.010 | 0.002 | 0.001 |
| glm4 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | 0.000 | 0.010 | -0.003 | 0.040 |
| glm4 | `with_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | 0.000 | 0.010 | -0.000 | 0.003 |
| glm4 | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.000 | 0.000 | 0.031 | 0.051 |
| glm4 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 3 | 3 | -0.021 | 0.135 | 0.255 | 0.182 |
| deepseek7b | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.916 | 1.190 | 3.360 | 1.081 |
| deepseek7b | `without_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.779 | 0.251 | -0.014 | 0.318 |
| deepseek7b | `without_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | 0.222 | 0.123 | 0.290 | 0.736 |
| deepseek7b | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.155 | 0.649 | 3.346 | 1.024 |
| deepseek7b | `without_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.138 | 0.082 | 0.140 | 0.387 |
| deepseek7b | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | -0.016 | 0.240 | 1.106 | 1.380 |
| deepseek7b | `without_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | -0.017 | -0.006 | -0.002 | 0.102 |
| deepseek7b | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | -0.054 | 0.101 | 0.208 | 0.980 |
| deepseek7b | `without_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | -0.069 | 0.021 | 0.580 | 0.788 |
| deepseek7b | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | -0.140 | -0.128 | -2.584 | 23.666 |
| deepseek7b | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | -0.156 | -0.076 | 0.341 | 0.337 |
| deepseek7b | `without_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | -0.188 | -0.158 | 0.044 | 0.197 |
| deepseek7b | `without_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | -0.247 | -0.098 | 0.016 | 0.083 |
| deepseek7b | `without_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | -0.285 | -0.066 | -0.069 | 0.260 |
| deepseek7b | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 3 | 3 | 0.521 | 0.406 | 19.145 | 0.428 |
| deepseek7b | `with_candidate_list` | `query_frame` | `source_group_top_component` | 3 | 3 | 0.125 | 0.031 | 0.722 | 0.941 |
| deepseek7b | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 3 | 3 | 0.083 | -0.021 | 2.344 | 3.321 |
| deepseek7b | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 3 | 3 | 0.062 | 0.083 | 0.233 | 0.540 |
| deepseek7b | `with_candidate_list` | `semantic_object` | `same_layer_control_head` | 3 | 3 | 0.062 | 0.062 | 0.014 | 0.239 |
| deepseek7b | `with_candidate_list` | `query_frame` | `same_layer_control_head` | 3 | 3 | 0.062 | 0.042 | 0.128 | 0.197 |
| deepseek7b | `with_candidate_list` | `semantic_relation` | `same_layer_control_head` | 3 | 3 | 0.062 | 0.000 | 0.046 | 0.016 |
| deepseek7b | `with_candidate_list` | `output_prefix` | `same_layer_control_head` | 3 | 3 | 0.062 | -0.010 | -0.052 | 0.482 |
| deepseek7b | `with_candidate_list` | `protocol_instruction` | `same_layer_control_head` | 3 | 3 | 0.042 | 0.042 | -0.023 | 0.140 |
| deepseek7b | `with_candidate_list` | `semantic_mixed` | `same_layer_control_head` | 3 | 3 | 0.042 | 0.021 | 0.071 | 0.269 |
| deepseek7b | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 3 | 3 | 0.042 | -0.010 | 0.193 | 0.458 |
| deepseek7b | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 3 | 3 | 0.021 | 0.021 | 0.919 | 1.183 |
| deepseek7b | `with_candidate_list` | `protocol_format` | `same_layer_control_head` | 3 | 3 | 0.021 | -0.010 | 0.043 | 0.149 |
| deepseek7b | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 3 | 3 | 0.021 | -0.073 | 2.380 | 3.323 |
| deepseek7b | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 3 | 3 | 0.000 | -0.042 | 0.326 | 0.316 |
| deepseek7b | `with_candidate_list` | `candidate_protocol` | `same_layer_control_head` | 3 | 3 | -0.021 | -0.010 | -0.637 | 1.225 |

## Strict Interpretation

- If removing the candidate list collapses base output and object/relation effects do not rise, the previous atlas mostly describes candidate-conditioned closure.
- If object/relation sources strengthen without the candidate list, the route can move toward free semantic closure.
- Causal removal remains head/source-level, not neuron/channel-level.
