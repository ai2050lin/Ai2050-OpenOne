# Phase 775 Semantic Latent Route vs Output Closure (smoke)

- Status: `complete`
- Test: separate value-pool latent selection from open-vocabulary output closure.
- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.

## Prompt Observation Summary

| model | variant | rows | cases | base top1 | latent pool hit | pool top1 | base rank | pool rank | base margin | pool margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | 2 | 2 | 0.000 | 0.500 | 0.500 | 2911.000 | 4.000 | 1.547 | 1.141 |
| qwen3 | `with_candidate_list` | 2 | 2 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 8.688 | 5.125 |
| qwen3 | `constrained_free_prompt` | 2 | 2 | 0.000 | 0.500 | 0.500 | 1241.500 | 3.500 | 1.219 | 1.062 |
| glm4 | `without_candidate_list` | 1 | 1 | 0.000 | 0.000 | 0.000 | 214.000 | 2.000 | 4.045 | -0.031 |
| glm4 | `with_candidate_list` | 1 | 1 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.438 | 0.438 |
| glm4 | `constrained_free_prompt` | 1 | 1 | 0.000 | 1.000 | 1.000 | 189.000 | 1.000 | 3.996 | 0.594 |
| deepseek7b | `without_candidate_list` | 2 | 2 | 0.000 | 0.500 | 0.500 | 10435.000 | 1.500 | 1.272 | 0.090 |
| deepseek7b | `with_candidate_list` | 2 | 2 | 0.000 | 0.500 | 0.500 | 4.000 | 1.500 | 1.188 | 1.188 |
| deepseek7b | `constrained_free_prompt` | 2 | 2 | 0.000 | 0.500 | 0.500 | 3543.500 | 1.500 | 0.914 | -0.609 |

## Component Effect Summary

| model | variant | kind | rows | cases | target drop | margin drop | target rank delta | pool rank delta | pool margin drop | pool top1 loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `source_group_top_component` | 10 | 2 | 0.037 | 0.050 | 103.700 | 0.000 | 0.062 | 0.000 |
| qwen3 | `with_candidate_list` | `source_group_top_component` | 12 | 2 | 0.177 | 0.130 | 0.000 | 0.000 | 0.198 | 0.000 |
| qwen3 | `constrained_free_prompt` | `source_group_top_component` | 10 | 2 | -0.156 | -0.037 | -27.100 | -0.100 | -0.025 | 0.000 |
| glm4 | `without_candidate_list` | `source_group_top_component` | 5 | 1 | -0.013 | -0.005 | -3.000 | 0.000 | 0.019 | 0.000 |
| glm4 | `with_candidate_list` | `source_group_top_component` | 6 | 1 | 0.042 | 0.062 | 0.000 | 0.000 | 0.062 | 0.000 |
| glm4 | `constrained_free_prompt` | `source_group_top_component` | 5 | 1 | -0.006 | -0.024 | -3.400 | 0.000 | 0.000 | 0.000 |
| deepseek7b | `without_candidate_list` | `source_group_top_component` | 10 | 2 | -0.420 | -0.168 | -6245.600 | 0.000 | -0.085 | 0.000 |
| deepseek7b | `with_candidate_list` | `source_group_top_component` | 12 | 2 | -0.010 | -0.089 | -0.083 | 0.000 | -0.089 | 0.000 |
| deepseek7b | `constrained_free_prompt` | `source_group_top_component` | 10 | 2 | 0.009 | -0.063 | -842.300 | 0.000 | -0.066 | 0.000 |

## Prompt Source Family Summary

| model | variant | family | kind | rows | cases | target drop | pool margin drop | direct boost | route suppression |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | 0.031 | 0.094 | 0.146 | 0.074 |
| qwen3 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 2 | 2 | 0.031 | 0.094 | 0.031 | 0.233 |
| qwen3 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 2 | 2 | 0.109 | 0.047 | 0.445 | 0.082 |
| qwen3 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 2 | 2 | 0.047 | 0.047 | 0.120 | 0.070 |
| qwen3 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 2 | 2 | -0.031 | 0.031 | 0.062 | 0.000 |
| qwen3 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 2 | 2 | 0.750 | 1.062 | 14.955 | 0.950 |
| qwen3 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 2 | 2 | 0.062 | 0.125 | 0.013 | 0.095 |
| qwen3 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 2 | 2 | 0.062 | 0.062 | 0.115 | 0.059 |
| qwen3 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 2 | 2 | 0.125 | 0.000 | 0.292 | 0.623 |
| qwen3 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | 0.000 | 0.000 | 0.005 | 0.299 |
| qwen3 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 2 | 2 | 0.062 | -0.062 | 0.020 | 0.127 |
| qwen3 | `constrained_free_prompt` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | 0.031 | 0.094 | -0.032 | 0.356 |
| qwen3 | `constrained_free_prompt` | `protocol_format` | `source_group_top_component` | 2 | 2 | 0.000 | 0.031 | 0.036 | 0.041 |
| qwen3 | `constrained_free_prompt` | `semantic_relation` | `source_group_top_component` | 2 | 2 | -0.031 | -0.031 | 0.033 | 0.041 |
| qwen3 | `constrained_free_prompt` | `semantic_object` | `source_group_top_component` | 2 | 2 | -0.062 | -0.062 | 0.042 | 0.653 |
| qwen3 | `constrained_free_prompt` | `output_prefix` | `source_group_top_component` | 2 | 2 | -0.719 | -0.156 | -0.212 | 0.763 |
| glm4 | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 1 | 1 | 0.000 | 0.031 | 0.000 | 0.013 |
| glm4 | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 1 | 1 | 0.000 | 0.031 | 0.005 | 0.002 |
| glm4 | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 1 | 1 | 0.000 | 0.031 | 0.008 | 0.064 |
| glm4 | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 1 | 1 | 0.000 | 0.000 | -0.005 | 0.045 |
| glm4 | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 1 | 1 | -0.062 | 0.000 | 0.014 | 0.000 |
| glm4 | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 1 | 1 | 0.312 | 0.312 | 0.839 | 0.065 |
| glm4 | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 1 | 1 | 0.000 | 0.062 | -0.011 | 0.068 |
| glm4 | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 1 | 1 | 0.000 | 0.062 | 0.007 | 0.000 |
| glm4 | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 1 | 1 | 0.000 | 0.000 | -0.004 | 0.005 |
| glm4 | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 1 | 1 | 0.000 | 0.000 | 0.016 | 0.076 |
| glm4 | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 1 | 1 | -0.062 | -0.062 | 0.021 | 0.018 |
| glm4 | `constrained_free_prompt` | `semantic_object` | `source_group_top_component` | 1 | 1 | 0.031 | 0.031 | 0.014 | 0.098 |
| glm4 | `constrained_free_prompt` | `protocol_instruction` | `source_group_top_component` | 1 | 1 | 0.000 | 0.031 | 0.011 | 0.015 |
| glm4 | `constrained_free_prompt` | `semantic_relation` | `source_group_top_component` | 1 | 1 | 0.000 | 0.000 | 0.013 | 0.000 |
| glm4 | `constrained_free_prompt` | `output_prefix` | `source_group_top_component` | 1 | 1 | -0.031 | 0.000 | 0.006 | 0.007 |
| glm4 | `constrained_free_prompt` | `protocol_format` | `source_group_top_component` | 1 | 1 | -0.031 | -0.062 | -0.001 | 0.003 |
| deepseek7b | `without_candidate_list` | `semantic_relation` | `source_group_top_component` | 2 | 2 | -0.453 | 0.020 | 0.251 | 0.259 |
| deepseek7b | `without_candidate_list` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | -0.378 | -0.039 | 0.116 | 0.255 |
| deepseek7b | `without_candidate_list` | `protocol_format` | `source_group_top_component` | 2 | 2 | -0.457 | -0.062 | 0.162 | 0.789 |
| deepseek7b | `without_candidate_list` | `output_prefix` | `source_group_top_component` | 2 | 2 | -0.396 | -0.095 | 0.195 | 0.014 |
| deepseek7b | `without_candidate_list` | `semantic_object` | `source_group_top_component` | 2 | 2 | -0.416 | -0.248 | 0.253 | 0.991 |
| deepseek7b | `with_candidate_list` | `output_prefix` | `source_group_top_component` | 2 | 2 | 0.156 | 0.031 | 0.095 | 0.326 |
| deepseek7b | `with_candidate_list` | `semantic_object` | `source_group_top_component` | 2 | 2 | -0.062 | 0.031 | 0.482 | 0.612 |
| deepseek7b | `with_candidate_list` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | -0.031 | -0.062 | 0.247 | 0.303 |
| deepseek7b | `with_candidate_list` | `protocol_format` | `source_group_top_component` | 2 | 2 | -0.094 | -0.094 | 0.139 | 0.268 |
| deepseek7b | `with_candidate_list` | `semantic_relation` | `source_group_top_component` | 2 | 2 | -0.094 | -0.156 | 0.016 | 0.519 |
| deepseek7b | `with_candidate_list` | `candidate_protocol` | `source_group_top_component` | 2 | 2 | 0.062 | -0.281 | 7.164 | 0.255 |
| deepseek7b | `constrained_free_prompt` | `protocol_instruction` | `source_group_top_component` | 2 | 2 | -0.012 | 0.043 | 0.118 | 0.260 |
| deepseek7b | `constrained_free_prompt` | `semantic_relation` | `source_group_top_component` | 2 | 2 | -0.020 | -0.012 | 0.249 | 0.264 |
| deepseek7b | `constrained_free_prompt` | `protocol_format` | `source_group_top_component` | 2 | 2 | -0.020 | -0.020 | 0.132 | 0.655 |
| deepseek7b | `constrained_free_prompt` | `output_prefix` | `source_group_top_component` | 2 | 2 | 0.062 | -0.086 | 0.257 | 0.057 |
| deepseek7b | `constrained_free_prompt` | `semantic_object` | `source_group_top_component` | 2 | 2 | 0.035 | -0.254 | 0.225 | 0.951 |

## Strict Interpretation

- `base top1` measures open-vocabulary output closure.
- `pool top1` measures whether the target wins inside the relation value pool without using that pool as prompt evidence.
- A high `pool top1` with low `base top1` suggests latent semantic selection without readout closure.
- Component removal remains head/source-level and does not prove neuron-level coding.
