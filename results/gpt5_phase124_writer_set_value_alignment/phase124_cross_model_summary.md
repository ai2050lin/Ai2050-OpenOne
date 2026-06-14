# Phase 124 Cross-model Writer Set Value Alignment

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, plant; train/test objects per category: 8/16; templates: 4; prompts/category: 64
- layers: qwen3: L32-L35 monitor L35; glm4: L15-L18 monitor L18; deepseek7b: L24-L27 monitor L27; rank: 16; set sizes: [1, 2, 4, 8, 16]; candidate pool: 24

| model | category | attention set | value set | target-discovered set | projection set | object control | random control | pre-MLP subspace | class |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 | number | attention_mass k4 T-0.28 R+0.00 A+4.16 | value_aligned k8 T-0.03 R+0.02 A-0.37 | target_discovered k16 T-0.64 R+0.00 A+8.92 | projection_discovered k16 T-0.47 R+0.00 A-2.94 | object_control k16 T-0.19 R+0.11 A+8.42 | random_control k1 T+0.04 R+0.03 A+1.83 | pre_answer L35 T-0.00 R+0.10 A+0.00 | weak_or_control_like |
| qwen3 | container | attention_mass k2 T-0.08 R+0.00 A+0.80 | value_aligned k16 T+0.01 R+0.67 A-55.83 | target_discovered k16 T-0.34 R+0.00 A-29.17 | projection_discovered k8 T-0.11 R+0.38 A-40.36 | object_control k4 T-0.01 R+0.02 A-2.77 | random_control k16 T-0.07 R+0.00 A-8.71 | pre_answer L32 T-0.05 R+0.06 A-1.78 | weak_or_control_like |
| qwen3 | plant | attention_mass k2 T-0.49 R+0.00 A-0.73 | abs_value_aligned k16 T-0.28 R+0.17 A+51.96 | target_discovered k16 T-0.93 R+0.00 A+18.59 | projection_discovered k1 T-0.55 R+0.00 A-1.77 | object_control k16 T-0.09 R+0.23 A+1.54 | random_control k16 T-0.61 R+0.00 A+4.79 | pre_answer L32 T-0.01 R+0.08 A+2.75 | weak_or_control_like |
| glm4 | number | attention_mass k16 T-0.09 R+0.03 A+0.15 | abs_value_aligned k16 T-0.16 R+0.06 A+0.11 | target_discovered k16 T-0.39 R+0.09 A+0.17 | projection_discovered k1 T-0.01 R+0.02 A-0.04 | object_control k16 T-0.06 R+0.02 A+0.04 | random_control k2 T-0.03 R+0.00 A+0.01 | pre_answer L18 T-0.06 R+0.08 A+0.00 | weak_or_control_like |
| glm4 | container | attention_mass k8 T-0.02 R+0.01 A+0.03 | value_aligned k4 T-0.02 R+0.01 A-0.03 | target_discovered k16 T-0.09 R+0.04 A-0.04 | projection_discovered k16 T-0.04 R+0.00 A-0.13 | object_control k8 T-0.01 R+0.00 A-0.02 | random_control k16 T-0.04 R+0.00 A-0.04 | pre_answer L16 T-0.06 R+0.04 A-0.01 | weak_or_control_like |
| glm4 | plant | attention_mass k1 T+0.00 R+0.01 A+0.02 | value_aligned k1 T+0.00 R+0.04 A-0.04 | target_discovered k16 T-0.08 R+0.00 A+0.17 | projection_discovered k1 T+0.01 R+0.07 A-0.04 | object_control k8 T-0.02 R+0.03 A+0.03 | random_control k8 T-0.01 R+0.01 A+0.03 | pre_answer L17 T-0.01 R+0.05 A+0.04 | weak_or_control_like |
| deepseek7b | number | attention_mass k16 T-0.16 R+0.55 A-71.98 | value_aligned k16 T-0.64 R+0.19 A-132.77 | target_discovered k16 T-0.67 R+0.00 A-102.78 | projection_discovered k16 T-0.75 R+0.24 A-192.08 | object_control k16 T-0.32 R+0.25 A-45.32 | random_control k2 T-0.04 R+0.10 A-0.38 | pre_answer L24 T-0.54 R+0.00 A-19.50 | weak_pre_mlp_subspace |
| deepseek7b | container | attention_mass k1 T-0.03 R+0.04 A+10.42 | abs_value_aligned k2 T-0.25 R+0.00 A+17.29 | target_discovered k16 T-0.44 R+0.00 A+89.55 | projection_discovered k1 T-0.07 R+0.13 A-3.99 | object_control k2 T-0.15 R+0.05 A-2.61 | random_control k2 T-0.09 R+0.00 A+11.55 | pre_answer L27 T-0.40 R+0.00 A+0.00 | weak_or_control_like |
| deepseek7b | plant | attention_mass k16 T-0.60 R+0.00 A-86.17 | value_aligned k16 T-0.67 R+0.02 A-126.17 | target_discovered k16 T-1.25 R+0.00 A-132.29 | projection_discovered k16 T-0.96 R+0.00 A-204.73 | object_control k1 T-0.03 R+0.13 A+2.94 | random_control k16 T-0.32 R+0.00 A-49.09 | pre_answer L24 T-0.67 R+0.00 A-16.14 | head_set_candidate |

## Reading Rules
- attention set is ranked by answer-token attention mass to pre-answer tokens.
- value set is ranked by attention head output alignment with the answer monitor axis.
- target-discovered set is ranked by measured single-head target_delta within the candidate pool.
- A is answer projection delta at the peak answer site.
