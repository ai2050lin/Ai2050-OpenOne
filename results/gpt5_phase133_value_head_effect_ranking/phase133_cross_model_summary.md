# Phase 133 Cross-model Value Head Effect Ranking Summary

## qwen3

Peak layer: L35; true last layer: L36; heads: 32; kv_heads: 8

| category | audit | reference | best head | top1 | top2 | top4 | top8 | all heads |
|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-0.07 R+0.15 A+0.38 | H11 T-0.08 R+0.00 A-0.28 | top_causal_1 T-0.08 R+0.00 A-0.28 | top_causal_2 T-0.13 R+0.00 A-0.14 | top_causal_4 T-0.19 R+0.00 A-0.33 | top_causal_8 T-0.25 R+0.00 A-0.49 | all_heads T+0.24 R+0.26 A+0.35 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.07 R+0.22 A+0.36 | H11 T-0.07 R+0.00 A-0.12 | top_causal_1 T-0.07 R+0.00 A-0.12 | top_causal_2 T-0.10 R+0.00 A-0.29 | top_causal_4 T-0.13 R+0.00 A-0.24 | top_causal_8 T-0.19 R+0.00 A-0.21 | all_heads T+0.13 R+0.37 A+0.19 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.24 R+0.26 A+0.50 | H11 T-0.09 R+0.00 A-0.18 | top_causal_1 T-0.09 R+0.00 A-0.18 | top_causal_2 T-0.14 R+0.00 A-0.46 | top_causal_4 T-0.20 R+0.00 A-0.62 | top_causal_8 T-0.26 R+0.00 A-1.00 | all_heads T+0.30 R+0.41 A+0.33 |

## glm4

Peak layer: L18; true last layer: L40; heads: 32; kv_heads: 2

| category | audit | reference | best head | top1 | top2 | top4 | top8 | all heads |
|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=32, mean_pre=3.2 | last_input_pre_answer T-0.05 R+0.05 A+0.94 | H1 T-0.03 R+0.06 A+0.05 | top_causal_1 T-0.03 R+0.06 A+0.05 | top_causal_2 T-0.06 R+0.04 A-0.72 | top_causal_4 T-0.09 R+0.02 A-1.20 | top_causal_8 T-0.11 R+0.11 A-2.18 | all_heads T-0.05 R+0.28 A-0.71 |
| container | old_mismatch=62, mean_pre=3.2 | last_input_pre_answer T+0.02 R+0.08 A+0.57 | H0 T-0.05 R+0.14 A-0.06 | top_causal_1 T-0.05 R+0.14 A-0.06 | top_causal_2 T-0.09 R+0.14 A-0.09 | top_causal_4 T-0.15 R+0.16 A-0.42 | top_causal_8 T-0.19 R+0.14 A-0.81 | all_heads T+0.21 R+0.26 A-0.89 |
| plant | old_mismatch=52, mean_pre=3.2 | last_input_pre_answer T-0.14 R+0.10 A+0.42 | H18 T-0.03 R+0.02 A-0.06 | top_causal_1 T-0.03 R+0.02 A-0.06 | top_causal_2 T-0.07 R+0.04 A-0.11 | top_causal_4 T-0.10 R+0.00 A-1.11 | top_causal_8 T-0.15 R+0.02 A-1.64 | all_heads T+0.22 R+0.36 A-0.49 |

## deepseek7b

Peak layer: L27; true last layer: L28; heads: 28; kv_heads: 4

| category | audit | reference | best head | top1 | top2 | top4 | top8 | all heads |
|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.36 R+0.57 A-32.68 | H13 T-0.28 R+0.00 A-6.15 | top_causal_1 T-0.28 R+0.00 A-6.15 | top_causal_2 T-0.91 R+0.00 A-19.12 | top_causal_4 T-1.41 R+0.00 A-26.74 | top_causal_8 T-2.16 R+0.00 A-42.31 | all_heads T-1.86 R+0.00 A-41.52 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.44 R+0.58 A-30.58 | H13 T-0.41 R+0.13 A-4.72 | top_causal_1 T-0.41 R+0.13 A-4.72 | top_causal_2 T-0.74 R+0.21 A-10.13 | top_causal_4 T-1.90 R+0.00 A-37.44 | top_causal_8 T-2.31 R+0.00 A-40.01 | all_heads T-2.65 R+0.00 A-38.17 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.17 R+2.10 A-35.67 | H13 T-0.29 R+0.16 A-9.89 | top_causal_1 T-0.29 R+0.16 A-9.89 | top_causal_2 T-0.53 R+0.00 A-11.76 | top_causal_4 T-1.15 R+0.00 A-26.77 | top_causal_8 T-1.52 R+0.00 A-26.00 | all_heads T-2.01 R+0.00 A-39.74 |

