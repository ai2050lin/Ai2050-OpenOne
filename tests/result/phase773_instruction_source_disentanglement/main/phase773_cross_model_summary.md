# Phase 773 Instruction Source Disentanglement (main)

- Status: `complete`
- Test: split instruction/protocol sources from object/relation semantic sources, then scan and causally remove per-source components.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Candidate Kind Summary

| model | kind | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `source_group_top_component` | 40 | 5 | 0.138 | 0.222 | 0.025 | 2.269 | 0.325 |
| glm4 | `source_group_top_component` | 40 | 5 | 0.005 | 0.023 | 0.000 | 0.058 | 0.045 |
| deepseek7b | `source_group_top_component` | 40 | 5 | 0.102 | 0.105 | 0.000 | 3.136 | 1.016 |

## Source Family Summary

| model | family | kind | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `candidate_protocol` | `source_group_top_component` | 5 | 5 | 0.975 | 1.512 | 0.200 | 16.360 | 0.882 |
| qwen3 | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.075 | 0.138 | 0.000 | 0.042 | 0.390 |
| qwen3 | `query_frame` | `source_group_top_component` | 5 | 5 | 0.075 | 0.062 | 0.000 | 0.274 | 0.144 |
| qwen3 | `semantic_mixed` | `source_group_top_component` | 5 | 5 | 0.025 | 0.062 | 0.000 | 0.209 | 0.345 |
| qwen3 | `semantic_relation` | `source_group_top_component` | 5 | 5 | 0.025 | 0.037 | 0.000 | 0.052 | 0.199 |
| qwen3 | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.000 | 0.000 | 0.000 | 0.144 | 0.257 |
| qwen3 | `semantic_object` | `source_group_top_component` | 5 | 5 | -0.025 | 0.013 | 0.000 | 0.182 | 0.317 |
| qwen3 | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.050 | -0.050 | 0.000 | 0.886 | 0.069 |
| glm4 | `query_frame` | `source_group_top_component` | 5 | 5 | 0.025 | 0.031 | 0.000 | 0.011 | 0.018 |
| glm4 | `semantic_relation` | `source_group_top_component` | 5 | 5 | 0.013 | 0.019 | 0.000 | 0.015 | 0.013 |
| glm4 | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.013 | 0.013 | 0.000 | 0.007 | 0.141 |
| glm4 | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.013 | 0.013 | 0.000 | 0.005 | 0.022 |
| glm4 | `candidate_protocol` | `source_group_top_component` | 5 | 5 | 0.000 | 0.106 | 0.000 | 0.376 | 0.109 |
| glm4 | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.000 | -0.006 | 0.000 | -0.002 | 0.026 |
| glm4 | `output_prefix` | `source_group_top_component` | 5 | 5 | -0.013 | 0.013 | 0.000 | 0.029 | 0.000 |
| glm4 | `semantic_mixed` | `source_group_top_component` | 5 | 5 | -0.013 | 0.000 | 0.000 | 0.019 | 0.031 |
| deepseek7b | `candidate_protocol` | `source_group_top_component` | 5 | 5 | 0.650 | 0.456 | 0.000 | 18.232 | 0.353 |
| deepseek7b | `protocol_format` | `source_group_top_component` | 5 | 5 | 0.062 | 0.050 | 0.000 | 0.335 | 0.376 |
| deepseek7b | `query_frame` | `source_group_top_component` | 5 | 5 | 0.062 | 0.031 | 0.000 | 0.619 | 0.758 |
| deepseek7b | `semantic_object` | `source_group_top_component` | 5 | 5 | 0.037 | 0.188 | 0.000 | 2.189 | 2.337 |
| deepseek7b | `protocol_instruction` | `source_group_top_component` | 5 | 5 | 0.025 | -0.006 | 0.000 | 0.128 | 0.410 |
| deepseek7b | `output_prefix` | `source_group_top_component` | 5 | 5 | 0.025 | -0.013 | 0.000 | 0.813 | 0.990 |
| deepseek7b | `semantic_mixed` | `source_group_top_component` | 5 | 5 | -0.013 | 0.144 | 0.000 | 2.312 | 2.507 |
| deepseek7b | `semantic_relation` | `source_group_top_component` | 5 | 5 | -0.037 | -0.013 | 0.000 | 0.459 | 0.398 |

## Source Group Summary

| model | source | kind | rows | cases | target drop | margin drop | top1 loss | attention |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `candidate_list` | `source_group_top_component` | 5 | 5 | 0.975 | 1.512 | 0.200 | 0.399 |
| qwen3 | `instruction_core` | `source_group_top_component` | 5 | 5 | 0.075 | 0.138 | 0.000 | 0.825 |
| qwen3 | `task_frame_without_semantic` | `source_group_top_component` | 5 | 5 | 0.075 | 0.062 | 0.000 | 0.225 |
| qwen3 | `semantic_pair` | `source_group_top_component` | 5 | 5 | 0.025 | 0.062 | 0.000 | 0.087 |
| qwen3 | `relation_tokens` | `source_group_top_component` | 5 | 5 | 0.025 | 0.037 | 0.000 | 0.062 |
| qwen3 | `format_cue` | `source_group_top_component` | 5 | 5 | 0.000 | 0.000 | 0.000 | 0.100 |
| qwen3 | `object_tokens` | `source_group_top_component` | 5 | 5 | -0.025 | 0.013 | 0.000 | 0.062 |
| qwen3 | `answer_prefix` | `source_group_top_component` | 5 | 5 | -0.050 | -0.050 | 0.000 | 0.431 |
| glm4 | `task_frame_without_semantic` | `source_group_top_component` | 5 | 5 | 0.025 | 0.031 | 0.000 | 0.572 |
| glm4 | `relation_tokens` | `source_group_top_component` | 5 | 5 | 0.013 | 0.019 | 0.000 | 0.389 |
| glm4 | `instruction_core` | `source_group_top_component` | 5 | 5 | 0.013 | 0.013 | 0.000 | 0.913 |
| glm4 | `object_tokens` | `source_group_top_component` | 5 | 5 | 0.013 | 0.013 | 0.000 | 0.311 |
| glm4 | `candidate_list` | `source_group_top_component` | 5 | 5 | 0.000 | 0.106 | 0.000 | 0.556 |
| glm4 | `format_cue` | `source_group_top_component` | 5 | 5 | 0.000 | -0.006 | 0.000 | 0.350 |
| glm4 | `answer_prefix` | `source_group_top_component` | 5 | 5 | -0.013 | 0.013 | 0.000 | 0.682 |
| glm4 | `semantic_pair` | `source_group_top_component` | 5 | 5 | -0.013 | 0.000 | 0.000 | 0.435 |
| deepseek7b | `candidate_list` | `source_group_top_component` | 5 | 5 | 0.650 | 0.456 | 0.000 | 0.651 |
| deepseek7b | `format_cue` | `source_group_top_component` | 5 | 5 | 0.062 | 0.050 | 0.000 | 0.218 |
| deepseek7b | `task_frame_without_semantic` | `source_group_top_component` | 5 | 5 | 0.062 | 0.031 | 0.000 | 0.432 |
| deepseek7b | `object_tokens` | `source_group_top_component` | 5 | 5 | 0.037 | 0.188 | 0.000 | 0.413 |
| deepseek7b | `instruction_core` | `source_group_top_component` | 5 | 5 | 0.025 | -0.006 | 0.000 | 0.263 |
| deepseek7b | `answer_prefix` | `source_group_top_component` | 5 | 5 | 0.025 | -0.013 | 0.000 | 0.396 |
| deepseek7b | `semantic_pair` | `source_group_top_component` | 5 | 5 | -0.013 | 0.144 | 0.000 | 0.608 |
| deepseek7b | `relation_tokens` | `source_group_top_component` | 5 | 5 | -0.037 | -0.013 | 0.000 | 0.261 |

## Strict Interpretation

- This phase tests whether Phase 772's instruction dominance is protocol/candidate-list driven or semantic-source driven.
- Per-source selection prevents instruction from hiding object/relation candidates in global ranking.
- Causal removal remains the evidence; direct score remains a filter.
- This is still not neuron/channel-level evidence.
