# Phase 774 Candidate-List Ablation (main)

- Status: `complete`
- Test: compare prompts with and without allowed-values candidate list.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Prompt Variant Summary

| model | variant | kind | rows | cases | base top1 | base rank | base margin | target drop | margin drop | top1 loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `source_group_top_component` | 35 | 5 | 0.000 | 1168.800 | 3.181 | 0.031 | 0.063 | 0.000 |
| qwen3 | `with_candidate_list` | `source_group_top_component` | 40 | 5 | 1.000 | 1.000 | 6.750 | 0.138 | 0.222 | 0.025 |
| glm4 | `without_candidate_list` | `source_group_top_component` | 35 | 5 | 0.000 | 45.800 | 2.934 | -0.004 | -0.003 | 0.000 |
| glm4 | `with_candidate_list` | `source_group_top_component` | 40 | 5 | 1.000 | 1.000 | 2.056 | 0.005 | 0.023 | 0.000 |
| deepseek7b | `without_candidate_list` | `source_group_top_component` | 35 | 5 | 0.000 | 4177.000 | 5.243 | 0.033 | 0.222 | 0.000 |
| deepseek7b | `with_candidate_list` | `source_group_top_component` | 40 | 5 | 0.600 | 2.200 | 3.931 | 0.102 | 0.105 | 0.000 |

## Prompt Source Family Summary

| model | variant | family | kind | rows | cases | target drop | margin drop | direct boost | route suppression |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.131 | 0.131 | -0.033 | 0.489 |
| qwen3 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.050 | 0.100 | 0.131 | 0.365 |
| qwen3 | `without_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.037 | 0.037 | 0.262 | 0.166 |
| qwen3 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.037 | 0.025 | 0.310 | 0.240 |
| qwen3 | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.006 | 0.106 | 0.318 | 0.342 |
| qwen3 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | 0.006 | 0.069 | 0.047 | 0.359 |
| qwen3 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.050 | -0.025 | 0.497 | 0.696 |
| qwen3 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 5 | 5 | 0.975 | 1.512 | 16.360 | 0.882 |
| qwen3 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.075 | 0.138 | 0.042 | 0.390 |
| qwen3 | `with_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.075 | 0.062 | 0.274 | 0.144 |
| qwen3 | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.025 | 0.062 | 0.209 | 0.345 |
| qwen3 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | 0.025 | 0.037 | 0.052 | 0.199 |
| qwen3 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.000 | 0.000 | 0.144 | 0.257 |
| qwen3 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | -0.025 | 0.013 | 0.182 | 0.317 |
| qwen3 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.050 | -0.050 | 0.886 | 0.069 |
| glm4 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.013 | 0.018 | 0.004 | 0.031 |
| glm4 | `without_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.006 | 0.000 | 0.001 | 0.024 |
| glm4 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | 0.000 | 0.002 | 0.008 | 0.026 |
| glm4 | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.000 | -0.011 | 0.017 | 0.034 |
| glm4 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.013 | -0.002 | 0.017 | 0.019 |
| glm4 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | -0.019 | -0.012 | 0.014 | 0.146 |
| glm4 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | -0.019 | -0.016 | -0.002 | 0.020 |
| glm4 | `with_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.025 | 0.031 | 0.011 | 0.018 |
| glm4 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | 0.013 | 0.019 | 0.015 | 0.013 |
| glm4 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.013 | 0.013 | 0.007 | 0.141 |
| glm4 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.013 | 0.013 | 0.005 | 0.022 |
| glm4 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 5 | 5 | 0.000 | 0.106 | 0.376 | 0.109 |
| glm4 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.000 | -0.006 | -0.002 | 0.026 |
| glm4 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.013 | 0.013 | 0.029 | 0.000 |
| glm4 | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | -0.013 | 0.000 | 0.019 | 0.031 |
| deepseek7b | `without_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.437 | 0.905 | 3.419 | 0.867 |
| deepseek7b | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.056 | 0.539 | 3.048 | 0.857 |
| deepseek7b | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | 0.003 | 0.153 | 0.935 | 1.004 |
| deepseek7b | `without_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | -0.004 | 0.037 | 0.586 | 0.697 |
| deepseek7b | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | -0.033 | 0.046 | 0.260 | 0.924 |
| deepseek7b | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | -0.071 | -0.067 | -1.421 | 14.215 |
| deepseek7b | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.156 | -0.058 | 0.738 | 0.296 |
| deepseek7b | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 5 | 5 | 0.650 | 0.456 | 18.232 | 0.353 |
| deepseek7b | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.062 | 0.050 | 0.335 | 0.376 |
| deepseek7b | `with_candidate_list` | `query_frame` | `source_group_top_component` | 5 | 5 | 0.062 | 0.031 | 0.619 | 0.758 |
| deepseek7b | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.037 | 0.188 | 2.189 | 2.337 |
| deepseek7b | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.025 | -0.006 | 0.128 | 0.410 |
| deepseek7b | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 5 | 5 | 0.025 | -0.013 | 0.813 | 0.990 |
| deepseek7b | `with_candidate_list` | `semantic_mixed` | `source_group_top_component` | 5 | 5 | -0.013 | 0.144 | 2.312 | 2.507 |
| deepseek7b | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.037 | -0.013 | 0.459 | 0.398 |

## Strict Interpretation

- If removing the candidate list collapses base output and object/relation effects do not rise, the previous atlas mostly describes candidate-conditioned closure.
- If object/relation sources strengthen without the candidate list, the route can move toward free semantic closure.
- Causal removal remains head/source-level, not neuron/channel-level.
