# Phase 773 Instruction Source Disentanglement (smoke)

- Status: `complete`
- Test: split instruction/protocol sources from object/relation semantic sources, then scan and causally remove per-source components.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Candidate Kind Summary

| model | kind | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `source_group_top_component` | 8 | 2 | 0.234 | 0.180 | 0.000 | 3.841 | 0.483 |
| glm4 | `source_group_top_component` | 4 | 1 | 0.062 | 0.078 | 0.000 | 0.211 | 0.039 |
| deepseek7b | `source_group_top_component` | 8 | 2 | 0.023 | -0.102 | 0.000 | 1.911 | 0.288 |

## Source Family Summary

| model | family | kind | rows | cases | target drop | margin drop | top1 loss | direct boost | route suppression |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `candidate_protocol` | `source_group_top_component` | 2 | 2 | 0.750 | 0.969 | 0.000 | 14.955 | 0.950 |
| qwen3 | `output_prefix` | `source_group_top_component` | 2 | 2 | 0.125 | -0.281 | 0.000 | 0.292 | 0.623 |
| qwen3 | `protocol_format` | `source_group_top_component` | 2 | 2 | 0.062 | -0.031 | 0.000 | 0.115 | 0.059 |
| qwen3 | `protocol_instruction` | `source_group_top_component` | 2 | 2 | 0.000 | 0.062 | 0.000 | 0.005 | 0.299 |
| glm4 | `candidate_protocol` | `source_group_top_component` | 1 | 1 | 0.312 | 0.312 | 0.000 | 0.839 | 0.065 |
| glm4 | `protocol_instruction` | `source_group_top_component` | 1 | 1 | 0.000 | 0.062 | 0.000 | -0.011 | 0.068 |
| glm4 | `protocol_format` | `source_group_top_component` | 1 | 1 | 0.000 | 0.000 | 0.000 | -0.004 | 0.005 |
| glm4 | `output_prefix` | `source_group_top_component` | 1 | 1 | -0.062 | -0.062 | 0.000 | 0.021 | 0.018 |
| deepseek7b | `output_prefix` | `source_group_top_component` | 2 | 2 | 0.156 | 0.031 | 0.000 | 0.095 | 0.326 |
| deepseek7b | `candidate_protocol` | `source_group_top_component` | 2 | 2 | 0.062 | -0.281 | 0.000 | 7.164 | 0.255 |
| deepseek7b | `protocol_instruction` | `source_group_top_component` | 2 | 2 | -0.031 | -0.062 | 0.000 | 0.247 | 0.303 |
| deepseek7b | `protocol_format` | `source_group_top_component` | 2 | 2 | -0.094 | -0.094 | 0.000 | 0.139 | 0.268 |

## Source Group Summary

| model | source | kind | rows | cases | target drop | margin drop | top1 loss | attention |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `candidate_list` | `source_group_top_component` | 2 | 2 | 0.750 | 0.969 | 0.000 | 0.482 |
| qwen3 | `answer_prefix` | `source_group_top_component` | 2 | 2 | 0.125 | -0.281 | 0.000 | 0.457 |
| qwen3 | `format_cue` | `source_group_top_component` | 2 | 2 | 0.062 | -0.031 | 0.000 | 0.061 |
| qwen3 | `instruction_core` | `source_group_top_component` | 2 | 2 | 0.000 | 0.062 | 0.000 | 0.635 |
| glm4 | `candidate_list` | `source_group_top_component` | 1 | 1 | 0.312 | 0.312 | 0.000 | 0.638 |
| glm4 | `instruction_core` | `source_group_top_component` | 1 | 1 | 0.000 | 0.062 | 0.000 | 0.898 |
| glm4 | `format_cue` | `source_group_top_component` | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.247 |
| glm4 | `answer_prefix` | `source_group_top_component` | 1 | 1 | -0.062 | -0.062 | 0.000 | 0.348 |
| deepseek7b | `answer_prefix` | `source_group_top_component` | 2 | 2 | 0.156 | 0.031 | 0.000 | 0.105 |
| deepseek7b | `candidate_list` | `source_group_top_component` | 2 | 2 | 0.062 | -0.281 | 0.000 | 0.469 |
| deepseek7b | `instruction_core` | `source_group_top_component` | 2 | 2 | -0.031 | -0.062 | 0.000 | 0.366 |
| deepseek7b | `format_cue` | `source_group_top_component` | 2 | 2 | -0.094 | -0.094 | 0.000 | 0.193 |

## Strict Interpretation

- This phase tests whether Phase 772's instruction dominance is protocol/candidate-list driven or semantic-source driven.
- Per-source selection prevents instruction from hiding object/relation candidates in global ranking.
- Causal removal remains the evidence; direct score remains a filter.
- This is still not neuron/channel-level evidence.
