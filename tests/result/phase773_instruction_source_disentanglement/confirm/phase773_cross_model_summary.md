# Phase 773 Instruction Source Disentanglement (confirm)

- Status: `complete`
- Test: split instruction/protocol sources from object/relation semantic sources, then scan and causally remove per-source components.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Candidate Kind Summary

| model | kind | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `source_group_top_component` | 32 | 4 | 0.176 | 0.248 | 0.000 | 2.145 | 0.414 |
| qwen3 | `same_layer_control_head` | 32 | 4 | 0.020 | -0.004 | 0.000 | 0.018 | 0.064 |
| glm4 | `source_group_top_component` | 32 | 4 | 0.021 | 0.025 | 0.000 | 0.091 | 0.029 |
| glm4 | `same_layer_control_head` | 32 | 4 | 0.008 | 0.008 | 0.000 | 0.000 | 0.013 |
| deepseek7b | `source_group_top_component` | 32 | 4 | 0.139 | 0.137 | 0.000 | 2.982 | 0.630 |
| deepseek7b | `same_layer_control_head` | 32 | 4 | 0.016 | 0.012 | 0.000 | -0.026 | 0.308 |

## Source Family Summary

| model | family | kind | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `candidate_protocol` | `source_group_top_component` | 4 | 4 | 1.000 | 1.750 | 0.000 | 15.230 | 1.130 |
| qwen3 | `output_prefix` | `source_group_top_component` | 4 | 4 | 0.188 | -0.125 | 0.000 | 0.891 | 0.087 |
| qwen3 | `protocol_instruction` | `source_group_top_component` | 4 | 4 | 0.125 | 0.188 | 0.000 | -0.024 | 0.621 |
| qwen3 | `query_frame` | `source_group_top_component` | 4 | 4 | 0.062 | 0.078 | 0.000 | 0.157 | 0.274 |
| qwen3 | `semantic_relation` | `source_group_top_component` | 4 | 4 | 0.062 | 0.062 | 0.000 | 0.085 | 0.251 |
| qwen3 | `protocol_instruction` | `same_layer_control_head` | 4 | 4 | 0.062 | 0.047 | 0.000 | -0.004 | 0.035 |
| qwen3 | `protocol_format` | `same_layer_control_head` | 4 | 4 | 0.062 | 0.031 | 0.000 | -0.030 | 0.068 |
| qwen3 | `semantic_object` | `same_layer_control_head` | 4 | 4 | 0.062 | 0.000 | 0.000 | 0.038 | 0.050 |
| qwen3 | `semantic_mixed` | `source_group_top_component` | 4 | 4 | 0.031 | 0.078 | 0.000 | 0.356 | 0.339 |
| qwen3 | `query_frame` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.031 | 0.000 | -0.011 | 0.006 |
| qwen3 | `output_prefix` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.000 | 0.000 | -0.003 | 0.006 |
| qwen3 | `semantic_mixed` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.000 | 0.000 | 0.048 | 0.052 |
| qwen3 | `semantic_relation` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.000 | 0.000 | -0.001 | 0.006 |
| qwen3 | `semantic_object` | `source_group_top_component` | 4 | 4 | -0.031 | 0.047 | 0.000 | 0.334 | 0.305 |
| qwen3 | `protocol_format` | `source_group_top_component` | 4 | 4 | -0.031 | -0.094 | 0.000 | 0.134 | 0.302 |
| qwen3 | `candidate_protocol` | `same_layer_control_head` | 4 | 4 | -0.094 | -0.141 | 0.000 | 0.109 | 0.290 |
| glm4 | `candidate_protocol` | `source_group_top_component` | 4 | 4 | 0.125 | 0.141 | 0.000 | 0.664 | 0.023 |
| glm4 | `query_frame` | `source_group_top_component` | 4 | 4 | 0.031 | 0.031 | 0.000 | 0.008 | 0.018 |
| glm4 | `semantic_object` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.031 | 0.000 | -0.000 | 0.003 |
| glm4 | `protocol_instruction` | `source_group_top_component` | 4 | 4 | 0.016 | 0.016 | 0.000 | 0.018 | 0.104 |
| glm4 | `semantic_relation` | `source_group_top_component` | 4 | 4 | 0.016 | 0.016 | 0.000 | 0.005 | 0.016 |
| glm4 | `protocol_format` | `same_layer_control_head` | 4 | 4 | 0.016 | 0.016 | 0.000 | -0.000 | 0.001 |
| glm4 | `candidate_protocol` | `same_layer_control_head` | 4 | 4 | 0.016 | -0.016 | 0.000 | 0.000 | 0.000 |
| glm4 | `protocol_instruction` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.031 | 0.000 | 0.009 | 0.096 |
| glm4 | `output_prefix` | `source_group_top_component` | 4 | 4 | 0.000 | 0.016 | 0.000 | 0.025 | 0.000 |
| glm4 | `query_frame` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.016 | 0.000 | -0.005 | 0.001 |
| glm4 | `semantic_relation` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.016 | 0.000 | -0.000 | 0.001 |
| glm4 | `semantic_object` | `source_group_top_component` | 4 | 4 | 0.000 | 0.000 | 0.000 | 0.003 | 0.024 |
| glm4 | `protocol_format` | `source_group_top_component` | 4 | 4 | 0.000 | -0.016 | 0.000 | -0.001 | 0.012 |
| glm4 | `output_prefix` | `same_layer_control_head` | 4 | 4 | 0.000 | -0.016 | 0.000 | -0.001 | 0.001 |
| glm4 | `semantic_mixed` | `same_layer_control_head` | 4 | 4 | 0.000 | -0.016 | 0.000 | -0.000 | 0.003 |
| glm4 | `semantic_mixed` | `source_group_top_component` | 4 | 4 | -0.016 | 0.000 | 0.000 | 0.007 | 0.039 |
| deepseek7b | `candidate_protocol` | `source_group_top_component` | 4 | 4 | 0.844 | 0.547 | 0.000 | 17.786 | 0.150 |
| deepseek7b | `semantic_object` | `source_group_top_component` | 4 | 4 | 0.078 | 0.328 | 0.000 | 1.709 | 1.041 |
| deepseek7b | `query_frame` | `source_group_top_component` | 4 | 4 | 0.078 | 0.031 | 0.000 | 0.734 | 0.674 |
| deepseek7b | `protocol_format` | `source_group_top_component` | 4 | 4 | 0.078 | 0.000 | 0.000 | 0.389 | 0.250 |
| deepseek7b | `output_prefix` | `source_group_top_component` | 4 | 4 | 0.062 | -0.031 | 0.000 | 0.784 | 0.857 |
| deepseek7b | `query_frame` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.016 | 0.000 | -0.054 | 0.160 |
| deepseek7b | `semantic_relation` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.000 | 0.000 | -0.008 | 0.037 |
| deepseek7b | `protocol_instruction` | `source_group_top_component` | 4 | 4 | 0.031 | -0.016 | 0.000 | 0.146 | 0.321 |
| deepseek7b | `output_prefix` | `same_layer_control_head` | 4 | 4 | 0.031 | -0.016 | 0.000 | -0.046 | 0.355 |
| deepseek7b | `semantic_mixed` | `source_group_top_component` | 4 | 4 | 0.016 | 0.266 | 0.000 | 1.863 | 1.252 |
| deepseek7b | `semantic_mixed` | `same_layer_control_head` | 4 | 4 | 0.016 | 0.062 | 0.000 | 0.105 | 0.230 |
| deepseek7b | `semantic_object` | `same_layer_control_head` | 4 | 4 | 0.016 | 0.016 | 0.000 | 0.056 | 0.190 |
| deepseek7b | `protocol_format` | `same_layer_control_head` | 4 | 4 | 0.016 | 0.000 | 0.000 | 0.034 | 0.092 |
| deepseek7b | `candidate_protocol` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.031 | 0.000 | -0.268 | 1.320 |
| deepseek7b | `protocol_instruction` | `same_layer_control_head` | 4 | 4 | -0.016 | -0.016 | 0.000 | -0.025 | 0.078 |
| deepseek7b | `semantic_relation` | `source_group_top_component` | 4 | 4 | -0.078 | -0.031 | 0.000 | 0.448 | 0.498 |

## Source Group Summary

| model | source | kind | rows | cases | target drop | margin drop | top1 loss | attention |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `candidate_list` | `source_group_top_component` | 4 | 4 | 1.000 | 1.750 | 0.000 | 0.428 |
| qwen3 | `answer_prefix` | `source_group_top_component` | 4 | 4 | 0.188 | -0.125 | 0.000 | 0.342 |
| qwen3 | `instruction_core` | `source_group_top_component` | 4 | 4 | 0.125 | 0.188 | 0.000 | 0.905 |
| qwen3 | `task_frame_without_semantic` | `source_group_top_component` | 4 | 4 | 0.062 | 0.078 | 0.000 | 0.240 |
| qwen3 | `relation_tokens` | `source_group_top_component` | 4 | 4 | 0.062 | 0.062 | 0.000 | 0.044 |
| qwen3 | `instruction_core` | `same_layer_control_head` | 4 | 4 | 0.062 | 0.047 | 0.000 | 0.401 |
| qwen3 | `format_cue` | `same_layer_control_head` | 4 | 4 | 0.062 | 0.031 | 0.000 | 0.044 |
| qwen3 | `object_tokens` | `same_layer_control_head` | 4 | 4 | 0.062 | 0.000 | 0.000 | 0.011 |
| qwen3 | `semantic_pair` | `source_group_top_component` | 4 | 4 | 0.031 | 0.078 | 0.000 | 0.071 |
| qwen3 | `task_frame_without_semantic` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.031 | 0.000 | 0.029 |
| qwen3 | `answer_prefix` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.000 | 0.000 | 0.008 |
| qwen3 | `semantic_pair` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.000 | 0.000 | 0.015 |
| qwen3 | `relation_tokens` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.000 | 0.000 | 0.004 |
| qwen3 | `object_tokens` | `source_group_top_component` | 4 | 4 | -0.031 | 0.047 | 0.000 | 0.065 |
| qwen3 | `format_cue` | `source_group_top_component` | 4 | 4 | -0.031 | -0.094 | 0.000 | 0.095 |
| qwen3 | `candidate_list` | `same_layer_control_head` | 4 | 4 | -0.094 | -0.141 | 0.000 | 0.209 |
| glm4 | `candidate_list` | `source_group_top_component` | 4 | 4 | 0.125 | 0.141 | 0.000 | 0.456 |
| glm4 | `task_frame_without_semantic` | `source_group_top_component` | 4 | 4 | 0.031 | 0.031 | 0.000 | 0.540 |
| glm4 | `object_tokens` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.031 | 0.000 | 0.018 |
| glm4 | `instruction_core` | `source_group_top_component` | 4 | 4 | 0.016 | 0.016 | 0.000 | 0.901 |
| glm4 | `relation_tokens` | `source_group_top_component` | 4 | 4 | 0.016 | 0.016 | 0.000 | 0.373 |
| glm4 | `format_cue` | `same_layer_control_head` | 4 | 4 | 0.016 | 0.016 | 0.000 | 0.042 |
| glm4 | `candidate_list` | `same_layer_control_head` | 4 | 4 | 0.016 | -0.016 | 0.000 | 0.075 |
| glm4 | `instruction_core` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.031 | 0.000 | 0.462 |
| glm4 | `answer_prefix` | `source_group_top_component` | 4 | 4 | 0.000 | 0.016 | 0.000 | 0.754 |
| glm4 | `task_frame_without_semantic` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.016 | 0.000 | 0.119 |
| glm4 | `relation_tokens` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.016 | 0.000 | 0.020 |
| glm4 | `object_tokens` | `source_group_top_component` | 4 | 4 | 0.000 | 0.000 | 0.000 | 0.374 |
| glm4 | `format_cue` | `source_group_top_component` | 4 | 4 | 0.000 | -0.016 | 0.000 | 0.324 |
| glm4 | `answer_prefix` | `same_layer_control_head` | 4 | 4 | 0.000 | -0.016 | 0.000 | 0.120 |
| glm4 | `semantic_pair` | `same_layer_control_head` | 4 | 4 | 0.000 | -0.016 | 0.000 | 0.033 |
| glm4 | `semantic_pair` | `source_group_top_component` | 4 | 4 | -0.016 | 0.000 | 0.000 | 0.416 |
| deepseek7b | `candidate_list` | `source_group_top_component` | 4 | 4 | 0.844 | 0.547 | 0.000 | 0.634 |
| deepseek7b | `object_tokens` | `source_group_top_component` | 4 | 4 | 0.078 | 0.328 | 0.000 | 0.340 |
| deepseek7b | `task_frame_without_semantic` | `source_group_top_component` | 4 | 4 | 0.078 | 0.031 | 0.000 | 0.435 |
| deepseek7b | `format_cue` | `source_group_top_component` | 4 | 4 | 0.078 | 0.000 | 0.000 | 0.188 |
| deepseek7b | `answer_prefix` | `source_group_top_component` | 4 | 4 | 0.062 | -0.031 | 0.000 | 0.267 |
| deepseek7b | `task_frame_without_semantic` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.016 | 0.000 | 0.199 |
| deepseek7b | `relation_tokens` | `same_layer_control_head` | 4 | 4 | 0.031 | 0.000 | 0.000 | 0.057 |
| deepseek7b | `instruction_core` | `source_group_top_component` | 4 | 4 | 0.031 | -0.016 | 0.000 | 0.296 |
| deepseek7b | `answer_prefix` | `same_layer_control_head` | 4 | 4 | 0.031 | -0.016 | 0.000 | 0.048 |
| deepseek7b | `semantic_pair` | `source_group_top_component` | 4 | 4 | 0.016 | 0.266 | 0.000 | 0.584 |
| deepseek7b | `semantic_pair` | `same_layer_control_head` | 4 | 4 | 0.016 | 0.062 | 0.000 | 0.143 |
| deepseek7b | `object_tokens` | `same_layer_control_head` | 4 | 4 | 0.016 | 0.016 | 0.000 | 0.057 |
| deepseek7b | `format_cue` | `same_layer_control_head` | 4 | 4 | 0.016 | 0.000 | 0.000 | 0.091 |
| deepseek7b | `candidate_list` | `same_layer_control_head` | 4 | 4 | 0.000 | 0.031 | 0.000 | 0.250 |
| deepseek7b | `instruction_core` | `same_layer_control_head` | 4 | 4 | -0.016 | -0.016 | 0.000 | 0.156 |
| deepseek7b | `relation_tokens` | `source_group_top_component` | 4 | 4 | -0.078 | -0.031 | 0.000 | 0.281 |

## Strict Interpretation

- This phase tests whether Phase 772's instruction dominance is protocol/candidate-list driven or semantic-source driven.
- Per-source selection prevents instruction from hiding object/relation candidates in global ranking.
- Causal removal remains the evidence; direct score remains a filter.
- This is still not neuron/channel-level evidence.
