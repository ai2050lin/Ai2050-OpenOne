# Phase 774 Candidate-List Ablation (smoke)

- Status: `complete`
- Test: compare prompts with and without allowed-values candidate list.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Prompt Variant Summary

| model | variant | kind | rows | cases | base top1 | base rank | base margin | target drop | margin drop | top1 loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `source_group_top_component` | 10 | 2 | 0.000 | 2911.000 | 1.547 | 0.037 | 0.050 | 0.000 |
| qwen3 | `with_candidate_list` | `source_group_top_component` | 12 | 2 | 1.000 | 1.000 | 8.688 | 0.177 | 0.130 | 0.000 |
| glm4 | `without_candidate_list` | `source_group_top_component` | 5 | 1 | 0.000 | 214.000 | 4.045 | -0.013 | -0.005 | 0.000 |
| glm4 | `with_candidate_list` | `source_group_top_component` | 6 | 1 | 1.000 | 1.000 | 0.438 | 0.042 | 0.062 | 0.000 |
| deepseek7b | `without_candidate_list` | `source_group_top_component` | 10 | 2 | 0.000 | 10435.000 | 1.272 | -0.420 | -0.168 | 0.000 |
| deepseek7b | `with_candidate_list` | `source_group_top_component` | 12 | 2 | 0.000 | 4.000 | 1.188 | -0.010 | -0.089 | 0.000 |

## Prompt Source Family Summary

| model | variant | family | kind | rows | cases | target drop | margin drop | direct boost | route suppression |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 2 | 2 | 0.109 | 0.016 | 0.445 | 0.082 |
| qwen3 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 2 | 2 | 0.047 | 0.078 | 0.120 | 0.070 |
| qwen3 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | 0.031 | 0.094 | 0.146 | 0.074 |
| qwen3 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 2 | 2 | 0.031 | 0.094 | 0.031 | 0.233 |
| qwen3 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 2 | 2 | -0.031 | -0.031 | 0.062 | 0.000 |
| qwen3 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 2 | 2 | 0.750 | 0.969 | 14.955 | 0.950 |
| qwen3 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 2 | 2 | 0.125 | -0.281 | 0.292 | 0.623 |
| qwen3 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 2 | 2 | 0.062 | 0.094 | 0.013 | 0.095 |
| qwen3 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 2 | 2 | 0.062 | -0.031 | 0.115 | 0.059 |
| qwen3 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 2 | 2 | 0.062 | -0.031 | 0.020 | 0.127 |
| qwen3 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | 0.000 | 0.062 | 0.005 | 0.299 |
| glm4 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 1 | 1 | 0.000 | 0.021 | 0.005 | 0.002 |
| glm4 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 1 | 1 | 0.000 | 0.006 | 0.000 | 0.013 |
| glm4 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 1 | 1 | 0.000 | 0.004 | -0.005 | 0.045 |
| glm4 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 1 | 1 | 0.000 | -0.020 | 0.008 | 0.064 |
| glm4 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 1 | 1 | -0.062 | -0.039 | 0.014 | 0.000 |
| glm4 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 1 | 1 | 0.312 | 0.312 | 0.839 | 0.065 |
| glm4 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 1 | 1 | 0.000 | 0.062 | -0.011 | 0.068 |
| glm4 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 1 | 1 | 0.000 | 0.062 | 0.007 | 0.000 |
| glm4 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 1 | 1 | 0.000 | 0.000 | -0.004 | 0.005 |
| glm4 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 1 | 1 | 0.000 | 0.000 | 0.016 | 0.076 |
| glm4 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 1 | 1 | -0.062 | -0.062 | 0.021 | 0.018 |
| deepseek7b | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | -0.378 | -0.147 | 0.116 | 0.255 |
| deepseek7b | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 2 | 2 | -0.396 | -0.173 | 0.195 | 0.014 |
| deepseek7b | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 2 | 2 | -0.416 | -0.377 | 0.253 | 0.991 |
| deepseek7b | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 2 | 2 | -0.453 | -0.123 | 0.251 | 0.259 |
| deepseek7b | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 2 | 2 | -0.457 | -0.021 | 0.162 | 0.789 |
| deepseek7b | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 2 | 2 | 0.156 | 0.031 | 0.095 | 0.326 |
| deepseek7b | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 2 | 2 | 0.062 | -0.281 | 7.164 | 0.255 |
| deepseek7b | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | -0.031 | -0.062 | 0.247 | 0.303 |
| deepseek7b | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 2 | 2 | -0.062 | 0.031 | 0.482 | 0.612 |
| deepseek7b | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 2 | 2 | -0.094 | -0.094 | 0.139 | 0.268 |
| deepseek7b | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 2 | 2 | -0.094 | -0.156 | 0.016 | 0.519 |

## Strict Interpretation

- If removing the candidate list collapses base output and object/relation effects do not rise, the previous atlas mostly describes candidate-conditioned closure.
- If object/relation sources strengthen without the candidate list, the route can move toward free semantic closure.
- Causal removal remains head/source-level, not neuron/channel-level.
