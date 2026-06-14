# Phase 132 Cross-model Source-specific Value Contribution Summary

## qwen3

Peak layer: L35; true last layer: L36; heads: 32; kv_heads: 8

| category | audit | reference | best | object all | post-object all | all-pre all | self all | all-pre top |
|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-0.07 R+0.15 A+0.38 | post_object_pre_answer:top_heads T-0.02 R+0.06 A+0.17 | object_span:all_heads T+0.02 R+0.02 A-0.01 | post_object_pre_answer:all_heads T+0.03 R+0.16 A+0.28 | all_pre_answer:all_heads T+0.24 R+0.26 A+0.35 | self:all_heads T+0.86 R+1.01 A+1.03 | all_pre_answer:top_heads T+0.19 R+0.23 A+0.37 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.07 R+0.22 A+0.36 | object_span:top_heads T+0.00 R+0.03 A-0.02 | object_span:all_heads T+0.02 R+0.04 A+0.00 | post_object_pre_answer:all_heads T+0.08 R+0.22 A+0.24 | all_pre_answer:all_heads T+0.13 R+0.37 A+0.19 | self:all_heads T+0.46 R+0.72 A+0.25 | all_pre_answer:top_heads T+0.11 R+0.28 A+0.18 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.24 R+0.26 A+0.50 | object_span:top_heads T+0.01 R+0.05 A+0.01 | object_span:all_heads T+0.04 R+0.06 A+0.06 | post_object_pre_answer:all_heads T+0.23 R+0.25 A+0.51 | all_pre_answer:all_heads T+0.30 R+0.41 A+0.33 | self:all_heads T+0.14 R+0.78 A+1.11 | all_pre_answer:top_heads T+0.27 R+0.44 A+0.44 |

## glm4

Peak layer: L18; true last layer: L40; heads: 32; kv_heads: 2

| category | audit | reference | best | object all | post-object all | all-pre all | self all | all-pre top |
|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=32, mean_pre=3.2 | last_input_pre_answer T-0.05 R+0.05 A+0.94 | all_pre_answer:all_heads T-0.05 R+0.28 A-0.71 | object_span:all_heads T-0.00 R+0.05 A-0.22 | post_object_pre_answer:all_heads T-0.04 R+0.15 A-0.78 | all_pre_answer:all_heads T-0.05 R+0.28 A-0.71 | self:all_heads T-0.05 R+0.10 A+4.32 | all_pre_answer:top_heads T+0.01 R+0.07 A-0.57 |
| container | old_mismatch=62, mean_pre=3.2 | last_input_pre_answer T+0.02 R+0.08 A+0.57 | self:all_heads T-0.03 R+0.05 A+2.49 | object_span:all_heads T+0.04 R+0.04 A-0.18 | post_object_pre_answer:all_heads T+0.11 R+0.12 A-0.81 | all_pre_answer:all_heads T+0.21 R+0.26 A-0.89 | self:all_heads T-0.03 R+0.05 A+2.49 | all_pre_answer:top_heads T+0.06 R+0.05 A-0.26 |
| plant | old_mismatch=52, mean_pre=3.2 | last_input_pre_answer T-0.14 R+0.10 A+0.42 | object_span:all_heads T-0.01 R+0.09 A-0.48 | object_span:all_heads T-0.01 R+0.09 A-0.48 | post_object_pre_answer:all_heads T+0.02 R+0.17 A-0.22 | all_pre_answer:all_heads T+0.22 R+0.36 A-0.49 | self:all_heads T+0.04 R+0.06 A+1.18 | all_pre_answer:top_heads T+0.02 R+0.08 A-0.03 |

## deepseek7b

Peak layer: L27; true last layer: L28; heads: 28; kv_heads: 4

| category | audit | reference | best | object all | post-object all | all-pre all | self all | all-pre top |
|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.36 R+0.57 A-32.68 | all_pre_answer:all_heads T-1.86 R+0.00 A-41.52 | object_span:all_heads T-0.07 R+0.00 A-0.25 | post_object_pre_answer:all_heads T-0.34 R+0.19 A-1.78 | all_pre_answer:all_heads T-1.86 R+0.00 A-41.52 | self:all_heads T-0.36 R+0.69 A-1.81 | all_pre_answer:top_heads T-0.27 R+0.14 A-1.33 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.44 R+0.58 A-30.58 | all_pre_answer:all_heads T-2.65 R+0.00 A-38.17 | object_span:all_heads T-0.02 R+0.06 A-0.02 | post_object_pre_answer:all_heads T-0.21 R+0.11 A-0.95 | all_pre_answer:all_heads T-2.65 R+0.00 A-38.17 | self:all_heads T-0.10 R+0.73 A-0.79 | all_pre_answer:top_heads T-0.10 R+0.00 A-0.33 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.17 R+2.10 A-35.67 | all_pre_answer:all_heads T-2.01 R+0.00 A-39.74 | object_span:all_heads T-0.05 R+0.08 A+1.72 | post_object_pre_answer:all_heads T-0.50 R+0.21 A+0.30 | all_pre_answer:all_heads T-2.01 R+0.00 A-39.74 | self:all_heads T-0.10 R+0.90 A+1.80 | all_pre_answer:top_heads T-0.38 R+0.50 A-2.60 |

