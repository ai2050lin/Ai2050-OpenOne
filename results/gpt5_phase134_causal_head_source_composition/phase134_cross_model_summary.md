# Phase 134 Cross-model Causal Head Source Composition Summary

## qwen3

Peak layer: L35; true last layer: L36; causal heads: [11, 10, 28, 3, 31, 2, 5, 20]

| category | audit | reference | best | pre-object | object | bridge | structural | tail | all-pre |
|---|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-0.07 R+0.15 A+0.38 | all_pre_answer T-0.25 R+0.00 A-0.49 | pre_object T-0.15 R+0.00 A-0.53 | object_span T+0.00 R+0.01 A-0.01 | object_to_template_bridge T-0.03 R+0.02 A+0.01 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.08 R+0.01 A+0.09 | all_pre_answer T-0.25 R+0.00 A-0.49 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.07 R+0.22 A+0.37 | pre_object T-0.15 R+0.00 A-0.39 | pre_object T-0.15 R+0.00 A-0.39 | object_span T+0.00 R+0.01 A-0.03 | object_to_template_bridge T+0.01 R+0.03 A-0.01 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T+0.00 R+0.01 A+0.06 | all_pre_answer T-0.14 R+0.00 A-0.37 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.24 R+0.26 A+0.51 | pre_object T-0.20 R+0.00 A-0.59 | pre_object T-0.20 R+0.00 A-0.59 | object_span T+0.01 R+0.01 A-0.02 | object_to_template_bridge T+0.03 R+0.04 A+0.05 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T+0.01 R+0.02 A+0.13 | all_pre_answer T-0.18 R+0.00 A-0.49 |
| time | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-0.03 R+0.17 A+0.39 | all_pre_answer T-0.29 R+0.00 A-0.54 | pre_object T-0.19 R+0.00 A-0.55 | object_span T+0.00 R+0.01 A-0.01 | object_to_template_bridge T-0.02 R+0.02 A+0.00 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.08 R+0.00 A+0.06 | all_pre_answer T-0.29 R+0.00 A-0.54 |
| clothing | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.14 R+0.23 A+0.23 | all_pre_answer T-0.31 R+0.00 A-0.24 | pre_object T-0.22 R+0.00 A-0.31 | object_span T-0.00 R+0.01 A-0.02 | object_to_template_bridge T-0.04 R+0.03 A-0.01 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.06 R+0.01 A+0.09 | all_pre_answer T-0.31 R+0.00 A-0.24 |
| furniture | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.06 R+0.28 A+0.22 | all_pre_answer T-0.30 R+0.00 A-0.01 | pre_object T-0.20 R+0.00 A-0.13 | object_span T-0.00 R+0.01 A-0.02 | object_to_template_bridge T-0.04 R+0.03 A+0.02 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.07 R+0.02 A+0.11 | all_pre_answer T-0.30 R+0.00 A-0.01 |

## glm4

Peak layer: L18; true last layer: L40; causal heads: [1, 28, 0, 18, 11, 27, 23, 4]

| category | audit | reference | best | pre-object | object | bridge | structural | tail | all-pre |
|---|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=32, mean_pre=3.2 | last_input_pre_answer T-0.05 R+0.05 A+0.81 | all_pre_answer T-0.11 R+0.11 A-2.17 | pre_object T-0.04 R+0.00 A-0.83 | object_span T-0.00 R+0.00 A-0.29 | object_to_template_bridge T-0.02 R+0.01 A-0.42 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.02 R+0.01 A-0.50 | all_pre_answer T-0.11 R+0.11 A-2.17 |
| container | old_mismatch=62, mean_pre=3.2 | last_input_pre_answer T+0.02 R+0.08 A+0.66 | all_pre_answer T-0.16 R+0.10 A-1.17 | pre_object T-0.05 R+0.00 A-0.54 | object_span T-0.01 R+0.00 A-0.20 | object_to_template_bridge T-0.02 R+0.01 A-0.27 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.02 R+0.02 A-0.31 | all_pre_answer T-0.16 R+0.10 A-1.17 |
| plant | old_mismatch=52, mean_pre=3.2 | last_input_pre_answer T-0.15 R+0.08 A+0.23 | pre_object T-0.03 R+0.00 A-0.57 | pre_object T-0.03 R+0.00 A-0.57 | object_span T-0.02 R+0.00 A-0.67 | object_to_template_bridge T-0.01 R+0.02 A-0.22 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.00 R+0.02 A-0.28 | all_pre_answer T+0.14 R+0.17 A-1.46 |
| time | old_mismatch=32, mean_pre=3.2 | last_input_pre_answer T-0.00 R+0.10 A+1.03 | all_pre_answer T-0.13 R+0.09 A-1.22 | pre_object T-0.05 R+0.00 A-0.58 | object_span T-0.01 R+0.00 A-0.21 | object_to_template_bridge T-0.02 R+0.02 A-0.25 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.03 R+0.01 A-0.32 | all_pre_answer T-0.13 R+0.09 A-1.22 |
| clothing | old_mismatch=60, mean_pre=3.2 | last_input_pre_answer T+0.10 R+0.05 A+0.04 | all_pre_answer T-0.17 R+0.09 A-1.01 | pre_object T-0.06 R+0.00 A-0.51 | object_span T-0.02 R+0.00 A-0.31 | object_to_template_bridge T-0.02 R+0.01 A-0.17 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.02 R+0.01 A-0.23 | all_pre_answer T-0.17 R+0.09 A-1.01 |
| furniture | old_mismatch=52, mean_pre=3.2 | last_input_pre_answer T-0.01 R+0.08 A-0.53 | all_pre_answer T-0.13 R+0.11 A-1.36 | pre_object T-0.04 R+0.00 A-0.39 | object_span T-0.01 R+0.00 A-0.47 | object_to_template_bridge T-0.01 R+0.01 A-0.18 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.02 R+0.02 A-0.27 | all_pre_answer T-0.13 R+0.11 A-1.36 |

## deepseek7b

Peak layer: L27; true last layer: L28; causal heads: [13, 12, 11, 8, 25, 10, 26, 24]

| category | audit | reference | best | pre-object | object | bridge | structural | tail | all-pre |
|---|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.56 R+0.93 A-36.45 | all_pre_answer T-1.97 R+0.00 A-40.87 | pre_object T-1.88 R+0.00 A-41.07 | object_span T-0.04 R+0.00 A-0.32 | object_to_template_bridge T-0.17 R+0.02 A-1.25 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.19 R+0.17 A-1.21 | all_pre_answer T-1.97 R+0.00 A-40.87 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.85 R+0.90 A-40.09 | all_pre_answer T-2.30 R+0.00 A-40.88 | pre_object T-2.10 R+0.00 A-42.99 | object_span T-0.07 R+0.02 A-0.21 | object_to_template_bridge T-0.12 R+0.01 A-0.90 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.15 R+0.11 A-0.80 | all_pre_answer T-2.30 R+0.00 A-40.88 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.25 R+1.43 A-31.98 | all_pre_answer T-1.77 R+0.00 A-35.60 | pre_object T-1.29 R+0.00 A-33.89 | object_span T-0.07 R+0.04 A+0.09 | object_to_template_bridge T-0.26 R+0.00 A-1.14 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.30 R+0.21 A-0.64 | all_pre_answer T-1.77 R+0.00 A-35.60 |
| time | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.62 R+1.69 A-32.67 | all_pre_answer T-2.02 R+0.00 A-36.07 | pre_object T-1.76 R+0.00 A-35.88 | object_span T-0.03 R+0.07 A+0.01 | object_to_template_bridge T-0.21 R+0.00 A-1.36 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T-0.23 R+0.09 A-1.42 | all_pre_answer T-2.02 R+0.00 A-36.07 |
| clothing | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+1.87 R+1.84 A-27.32 | pre_object T-0.90 R+0.00 A-34.97 | pre_object T-0.90 R+0.00 A-34.97 | object_span T+0.06 R+0.06 A-0.21 | object_to_template_bridge T-0.02 R+0.00 A-1.06 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T+0.13 R+0.11 A-0.73 | all_pre_answer T-0.55 R+0.00 A-33.92 |
| furniture | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+1.49 R+1.33 A-30.85 | pre_object T-0.77 R+0.00 A-33.41 | pre_object T-0.77 R+0.00 A-33.41 | object_span T-0.02 R+0.00 A-0.25 | object_to_template_bridge T-0.04 R+0.00 A-1.35 | post_object_structural T+0.00 R+0.00 A+0.00 | answer_prompt_tail T+0.13 R+0.14 A-0.89 | all_pre_answer T-0.59 R+0.00 A-34.48 |

