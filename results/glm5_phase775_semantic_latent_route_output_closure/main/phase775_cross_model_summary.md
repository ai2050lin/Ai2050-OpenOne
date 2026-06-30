# Phase 775 Semantic Latent Route vs Output Closure (main)

- Status: `complete`
- Test: separate value-pool latent selection from open-vocabulary output closure.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Prompt Observation Summary

| model | variant | rows | cases | base top1 | latent pool hit | pool top1 | base rank | pool rank | base margin | pool margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | 5 | 5 | 0.000 | 0.800 | 0.800 | 1168.800 | 2.200 | 3.181 | 3.019 |
| qwen3 | `with_candidate_list` | 5 | 5 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 6.750 | 4.425 |
| qwen3 | `constrained_free_prompt` | 5 | 5 | 0.000 | 0.800 | 0.800 | 498.200 | 2.000 | 4.875 | 4.525 |
| glm4 | `without_candidate_list` | 5 | 5 | 0.000 | 0.800 | 0.800 | 45.800 | 1.200 | 2.934 | 1.525 |
| glm4 | `with_candidate_list` | 5 | 5 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 2.056 | 1.562 |
| glm4 | `constrained_free_prompt` | 5 | 5 | 0.000 | 0.600 | 0.600 | 42.000 | 1.400 | 2.612 | 1.394 |
| deepseek7b | `without_candidate_list` | 5 | 5 | 0.000 | 0.800 | 0.800 | 4177.000 | 1.200 | 5.243 | 3.442 |
| deepseek7b | `with_candidate_list` | 5 | 5 | 0.600 | 0.200 | 0.800 | 2.200 | 1.200 | 3.931 | 2.875 |
| deepseek7b | `constrained_free_prompt` | 5 | 5 | 0.000 | 0.800 | 0.800 | 1420.000 | 1.200 | 5.594 | 3.506 |

## Component Effect Summary

| model | variant | kind | rows | cases | target drop | margin drop | target rank delta | pool rank delta | pool margin drop | pool top1 loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `source_group_top_component` | 35 | 5 | 0.031 | 0.063 | 37.200 | 0.086 | 0.072 | 0.086 |
| qwen3 | `with_candidate_list` | `source_group_top_component` | 40 | 5 | 0.138 | 0.222 | 0.025 | 0.025 | 0.231 | 0.025 |
| qwen3 | `constrained_free_prompt` | `source_group_top_component` | 35 | 5 | -0.120 | -0.004 | -9.171 | -0.029 | -0.011 | 0.000 |
| glm4 | `without_candidate_list` | `source_group_top_component` | 35 | 5 | -0.004 | -0.003 | -0.714 | 0.000 | 0.004 | 0.000 |
| glm4 | `with_candidate_list` | `source_group_top_component` | 40 | 5 | 0.005 | 0.023 | 0.000 | 0.000 | 0.016 | 0.000 |
| glm4 | `constrained_free_prompt` | `source_group_top_component` | 35 | 5 | -0.009 | 0.005 | -0.543 | 0.000 | 0.004 | 0.000 |
| deepseek7b | `without_candidate_list` | `source_group_top_component` | 35 | 5 | 0.033 | 0.222 | -1152.257 | 0.086 | 0.147 | 0.000 |
| deepseek7b | `with_candidate_list` | `source_group_top_component` | 40 | 5 | 0.102 | 0.105 | 0.100 | 0.000 | 0.094 | 0.000 |
| deepseek7b | `constrained_free_prompt` | `source_group_top_component` | 35 | 5 | 0.076 | 0.141 | -153.171 | 0.000 | 0.049 | 0.000 |

## Prompt Source Family Summary

| model | variant | family | kind | rows | cases | target drop | pool margin drop | direct boost | route suppression |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.006 | 0.131 | 0.318 | 0.342 |
| qwen3 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.050 | 0.087 | 0.131 | 0.365 |
| qwen3 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.037 | 0.050 | 0.310 | 0.240 |
| qwen3 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.131 | 0.131 | -0.033 | 0.489 |
| qwen3 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | 0.006 | 0.081 | 0.047 | 0.359 |
| qwen3 | `without_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.037 | 0.037 | 0.262 | 0.166 |
| qwen3 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.050 | -0.013 | 0.497 | 0.696 |
| qwen3 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 5 | 5 | 0.975 | 1.550 | 16.360 | 0.882 |
| qwen3 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.075 | 0.125 | 0.042 | 0.390 |
| qwen3 | `with_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.075 | 0.075 | 0.274 | 0.144 |
| qwen3 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | 0.025 | 0.075 | 0.052 | 0.199 |
| qwen3 | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.025 | 0.050 | 0.209 | 0.345 |
| qwen3 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.000 | 0.000 | 0.144 | 0.257 |
| qwen3 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.050 | 0.000 | 0.886 | 0.069 |
| qwen3 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | -0.025 | -0.025 | 0.182 | 0.317 |
| qwen3 | `constrained_free_prompt` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | -0.025 | 0.062 | -0.148 | 0.833 |
| qwen3 | `constrained_free_prompt` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.050 | 0.025 | -0.037 | 0.480 |
| qwen3 | `constrained_free_prompt` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.000 | 0.013 | 0.086 | 0.131 |
| qwen3 | `constrained_free_prompt` | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.075 | 0.013 | 0.028 | 0.294 |
| qwen3 | `constrained_free_prompt` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.000 | -0.013 | 0.267 | 0.130 |
| qwen3 | `constrained_free_prompt` | `semantic_object` | `source_group_top_component` | 5 | 5 | -0.050 | -0.013 | -0.143 | 0.761 |
| qwen3 | `constrained_free_prompt` | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.738 | -0.163 | 0.228 | 0.775 |
| glm4 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.013 | 0.025 | 0.004 | 0.031 |
| glm4 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.013 | 0.013 | 0.017 | 0.019 |
| glm4 | `without_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.006 | 0.006 | 0.001 | 0.024 |
| glm4 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | 0.000 | 0.006 | 0.008 | 0.026 |
| glm4 | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.000 | 0.000 | 0.017 | 0.034 |
| glm4 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | -0.019 | -0.006 | 0.014 | 0.146 |
| glm4 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | -0.019 | -0.013 | -0.002 | 0.020 |
| glm4 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 5 | 5 | 0.000 | 0.037 | 0.376 | 0.109 |
| glm4 | `with_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.025 | 0.025 | 0.011 | 0.018 |
| glm4 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.013 | 0.025 | 0.007 | 0.141 |
| glm4 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.013 | 0.013 | 0.005 | 0.022 |
| glm4 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | 0.013 | 0.013 | 0.015 | 0.013 |
| glm4 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.013 | 0.013 | 0.029 | 0.000 |
| glm4 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.000 | 0.000 | -0.002 | 0.026 |
| glm4 | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | -0.013 | 0.000 | 0.019 | 0.031 |
| glm4 | `constrained_free_prompt` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.006 | 0.037 | 0.001 | 0.040 |
| glm4 | `constrained_free_prompt` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.000 | 0.019 | 0.002 | 0.023 |
| glm4 | `constrained_free_prompt` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.000 | 0.000 | 0.012 | 0.147 |
| glm4 | `constrained_free_prompt` | `protocol_format` | `source_group_top_component` | 5 | 5 | -0.013 | 0.000 | 0.009 | 0.016 |
| glm4 | `constrained_free_prompt` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | -0.019 | -0.006 | 0.018 | 0.043 |
| glm4 | `constrained_free_prompt` | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.013 | -0.013 | 0.024 | 0.012 |
| glm4 | `constrained_free_prompt` | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.025 | -0.013 | 0.017 | 0.021 |
| deepseek7b | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.437 | 0.570 | 3.419 | 0.867 |
| deepseek7b | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.056 | 0.190 | 3.048 | 0.857 |
| deepseek7b | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | 0.003 | 0.182 | 0.935 | 1.004 |
| deepseek7b | `without_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | -0.004 | 0.091 | 0.586 | 0.697 |
| deepseek7b | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | -0.033 | 0.019 | 0.260 | 0.924 |
| deepseek7b | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.156 | -0.004 | 0.738 | 0.296 |
| deepseek7b | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | -0.071 | -0.017 | -1.421 | 14.215 |
| deepseek7b | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 5 | 5 | 0.650 | 0.388 | 18.232 | 0.353 |
| deepseek7b | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.037 | 0.237 | 2.189 | 2.337 |
| deepseek7b | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | -0.013 | 0.188 | 2.312 | 2.507 |
| deepseek7b | `with_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.062 | 0.025 | 0.619 | 0.758 |
| deepseek7b | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.062 | 0.013 | 0.335 | 0.376 |
| deepseek7b | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.025 | -0.013 | 0.128 | 0.410 |
| deepseek7b | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.037 | -0.013 | 0.459 | 0.398 |
| deepseek7b | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | 0.025 | -0.075 | 0.813 | 0.990 |
| deepseek7b | `constrained_free_prompt` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.256 | 0.331 | 2.678 | 0.825 |
| deepseek7b | `constrained_free_prompt` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.069 | 0.144 | 3.106 | 0.710 |
| deepseek7b | `constrained_free_prompt` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.170 | 0.076 | 0.692 | 0.676 |
| deepseek7b | `constrained_free_prompt` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.020 | -0.030 | -0.275 | 4.144 |
| deepseek7b | `constrained_free_prompt` | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.020 | -0.036 | 0.687 | 0.415 |
| deepseek7b | `constrained_free_prompt` | `output_prefix` | `source_group_top_component` | 5 | 5 | 0.011 | -0.061 | 0.914 | 0.924 |
| deepseek7b | `constrained_free_prompt` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.030 | -0.083 | 0.233 | 0.792 |

## Strict Interpretation

- `base top1` measures open-vocabulary output closure.
- `pool top1` measures whether the target wins inside the relation value pool without using that pool as prompt evidence.
- A high `pool top1` with low `base top1` suggests latent semantic selection without readout closure.
- Component removal remains head/source-level and does not prove neuron-level coding.
