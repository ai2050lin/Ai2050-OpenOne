# Phase 136 Cross-model Long-template Head Re-ranking Summary

## qwen3

Peak layer: L35; true last layer: L36; heads: 32; kv_heads: 8; short core: [11, 10, 28, 3, 31, 2, 5, 20]

| category | audit | reference | best head | top4 | top8 | short core | all heads |
|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=28.8 | last_input_pre_answer T-2.80 R+0.57 A+3.62 | H11 T-0.07 R+0.00 A-0.26 | long_top_4 [11, 3, 10, 5] T-0.16 R+0.00 A-0.41 | long_top_8 [11, 3, 10, 5, 28, 6, 4, 30] T-0.20 R+0.00 A-0.65 | short_template_core [11, 10, 28, 3, 31, 2, 5, 20] T-0.12 R+0.00 A-0.64 | all_heads T+1.76 R+2.74 A+2.92 |
| container | old_mismatch=0, mean_pre=28.9 | last_input_pre_answer T-0.70 R+0.16 A+0.15 | H11 T-0.04 R+0.00 A-0.20 | long_top_4 [11, 28, 10, 6] T-0.12 R+0.00 A-0.60 | long_top_8 [11, 28, 10, 6, 12, 20, 30, 16] T-0.16 R+0.00 A-0.46 | short_template_core [11, 10, 28, 3, 31, 2, 5, 20] T-0.09 R+0.00 A-0.61 | all_heads T+2.47 R+2.61 A+3.99 |
| plant | old_mismatch=0, mean_pre=29.2 | last_input_pre_answer T-0.41 R+0.08 A+5.56 | H11 T-0.08 R+0.00 A-0.20 | long_top_4 [11, 28, 2, 10] T-0.17 R+0.00 A-0.58 | long_top_8 [11, 28, 2, 10, 6, 31, 30, 22] T-0.21 R+0.00 A-0.81 | short_template_core [11, 10, 28, 3, 31, 2, 5, 20] T-0.15 R+0.00 A-0.59 | all_heads T+1.09 R+1.70 A+1.57 |
| time | old_mismatch=0, mean_pre=28.8 | last_input_pre_answer T-1.90 R+0.28 A-0.53 | H11 T-0.05 R+0.00 A-0.20 | long_top_4 [11, 10, 3, 5] T-0.18 R+0.00 A-0.32 | long_top_8 [11, 10, 3, 5, 28, 16, 4, 6] T-0.22 R+0.00 A-0.39 | short_template_core [11, 10, 28, 3, 31, 2, 5, 20] T-0.14 R+0.00 A-0.55 | all_heads T+1.66 R+2.75 A+3.16 |
| clothing | old_mismatch=0, mean_pre=29.0 | last_input_pre_answer T-1.02 R+0.00 A+2.61 | H3 T-0.08 R+0.01 A-0.04 | long_top_4 [3, 28, 11, 10] T-0.21 R+0.00 A-0.46 | long_top_8 [3, 28, 11, 10, 22, 20, 6, 21] T-0.25 R+0.00 A-0.46 | short_template_core [11, 10, 28, 3, 31, 2, 5, 20] T-0.13 R+0.00 A-0.46 | all_heads T+0.87 R+2.41 A+2.09 |
| furniture | old_mismatch=0, mean_pre=29.2 | last_input_pre_answer T-2.67 R+0.00 A+5.46 | H3 T-0.08 R+0.02 A-0.03 | long_top_4 [3, 28, 11, 10] T-0.20 R+0.00 A-0.50 | long_top_8 [3, 28, 11, 10, 20, 6, 22, 21] T-0.25 R+0.00 A-0.44 | short_template_core [11, 10, 28, 3, 31, 2, 5, 20] T-0.14 R+0.00 A-0.51 | all_heads T+0.86 R+1.82 A+2.25 |

## glm4

Peak layer: L18; true last layer: L40; heads: 32; kv_heads: 2; short core: [1, 28, 0, 18, 11, 27, 23, 4]

| category | audit | reference | best head | top4 | top8 | short core | all heads |
|---|---|---|---|---|---|---|---|
| number | old_mismatch=80, mean_pre=30.8 | last_input_pre_answer T-0.04 R+0.67 A+2.82 | H28 T-0.03 R+0.01 A-0.51 | long_top_4 [28, 3, 0, 4] T-0.07 R+0.02 A-0.93 | long_top_8 [28, 3, 0, 4, 11, 18, 2, 14] T-0.11 R+0.06 A-1.81 | short_template_core [1, 28, 0, 18, 11, 27, 23, 4] T-0.08 R+0.02 A-1.75 | all_heads T-0.03 R+0.39 A-2.98 |
| container | old_mismatch=95, mean_pre=30.9 | last_input_pre_answer T+0.09 R+0.60 A+4.74 | H0 T-0.05 R+0.05 A-0.10 | long_top_4 [0, 15, 11, 28] T-0.07 R+0.05 A-0.84 | long_top_8 [0, 15, 11, 28, 23, 21, 4, 1] T-0.07 R+0.06 A-0.98 | short_template_core [1, 28, 0, 18, 11, 27, 23, 4] T-0.03 R+0.02 A-1.86 | all_heads T+0.34 R+0.40 A-2.30 |
| plant | old_mismatch=90, mean_pre=31.2 | last_input_pre_answer T-0.26 R+0.58 A+4.09 | H2 T-0.06 R+0.04 A-0.34 | long_top_4 [2, 30, 27, 28] T-0.12 R+0.08 A-1.53 | long_top_8 [2, 30, 27, 28, 4, 14, 10, 20] T-0.17 R+0.13 A-1.99 | short_template_core [1, 28, 0, 18, 11, 27, 23, 4] T-0.00 R+0.03 A-1.70 | all_heads T-0.04 R+0.39 A-2.75 |
| time | old_mismatch=80, mean_pre=30.8 | last_input_pre_answer T+0.11 R+0.58 A+3.73 | H28 T-0.03 R+0.01 A-0.60 | long_top_4 [28, 0, 4, 2] T-0.09 R+0.02 A-1.22 | long_top_8 [28, 0, 4, 2, 3, 11, 1, 8] T-0.13 R+0.05 A-1.82 | short_template_core [1, 28, 0, 18, 11, 27, 23, 4] T-0.10 R+0.02 A-1.72 | all_heads T-0.07 R+0.38 A-3.25 |
| clothing | old_mismatch=94, mean_pre=31.0 | last_input_pre_answer T+0.33 R+0.28 A+0.37 | H28 T-0.03 R+0.01 A-0.40 | long_top_4 [28, 4, 2, 29] T-0.11 R+0.13 A-1.78 | long_top_8 [28, 4, 2, 29, 14, 31, 8, 1] T-0.19 R+0.21 A-2.49 | short_template_core [1, 28, 0, 18, 11, 27, 23, 4] T-0.11 R+0.02 A-1.50 | all_heads T-0.12 R+0.40 A-3.93 |
| furniture | old_mismatch=90, mean_pre=31.2 | last_input_pre_answer T+0.42 R+0.69 A+3.56 | H31 T-0.03 R+0.04 A+0.07 | long_top_4 [31, 28, 2, 4] T-0.10 R+0.09 A-0.79 | long_top_8 [31, 28, 2, 4, 29, 1, 12, 14] T-0.16 R+0.21 A-0.94 | short_template_core [1, 28, 0, 18, 11, 27, 23, 4] T-0.07 R+0.02 A-1.42 | all_heads T-0.07 R+0.43 A-1.58 |

## deepseek7b

Peak layer: L27; true last layer: L28; heads: 28; kv_heads: 4; short core: [13, 12, 11, 8, 25, 10, 26, 24]

| category | audit | reference | best head | top4 | top8 | short core | all heads |
|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=28.8 | last_input_pre_answer T-2.66 R+0.49 A-28.80 | H11 T-0.15 R+0.00 A-2.06 | long_top_4 [11, 13, 10, 8] T-0.60 R+0.00 A-8.77 | long_top_8 [11, 13, 10, 8, 19, 7, 25, 27] T-0.90 R+0.00 A-9.03 | short_template_core [13, 12, 11, 8, 25, 10, 26, 24] T-0.64 R+0.00 A-10.13 | all_heads T-0.39 R+0.49 A-7.53 |
| container | old_mismatch=0, mean_pre=28.9 | last_input_pre_answer T-2.67 R+0.00 A-31.88 | H10 T-0.19 R+0.00 A-1.62 | long_top_4 [10, 11, 13, 8] T-0.68 R+0.00 A-8.38 | long_top_8 [10, 11, 13, 8, 24, 21, 26, 25] T-1.05 R+0.00 A-9.18 | short_template_core [13, 12, 11, 8, 25, 10, 26, 24] T-1.05 R+0.00 A-10.52 | all_heads T-1.04 R+0.57 A-8.08 |
| plant | old_mismatch=0, mean_pre=29.3 | last_input_pre_answer T-2.70 R+0.59 A-23.81 | H13 T-0.19 R+0.00 A-2.52 | long_top_4 [13, 10, 11, 8] T-0.66 R+0.00 A-7.08 | long_top_8 [13, 10, 11, 8, 26, 22, 24, 21] T-1.12 R+0.00 A-7.86 | short_template_core [13, 12, 11, 8, 25, 10, 26, 24] T-1.08 R+0.00 A-8.96 | all_heads T-1.03 R+0.67 A-4.87 |
| time | old_mismatch=0, mean_pre=28.8 | last_input_pre_answer T-3.13 R+0.84 A-34.96 | H13 T-0.20 R+0.00 A-2.58 | long_top_4 [13, 10, 11, 8] T-0.70 R+0.00 A-7.80 | long_top_8 [13, 10, 11, 8, 12, 7, 0, 21] T-0.94 R+0.00 A-9.54 | short_template_core [13, 12, 11, 8, 25, 10, 26, 24] T-0.82 R+0.02 A-9.94 | all_heads T-0.49 R+0.59 A-6.87 |
| clothing | old_mismatch=0, mean_pre=29.0 | last_input_pre_answer T+0.74 R+0.88 A-21.67 | H0 T-0.07 R+0.06 A-0.19 | long_top_4 [0, 25, 7, 22] T-0.24 R+0.00 A-0.91 | long_top_8 [0, 25, 7, 22, 8, 6, 13, 21] T-0.33 R+0.00 A-5.17 | short_template_core [13, 12, 11, 8, 25, 10, 26, 24] T+0.02 R+0.06 A-8.95 | all_heads T+0.57 R+0.67 A-5.61 |
| furniture | old_mismatch=0, mean_pre=29.3 | last_input_pre_answer T+1.15 R+1.03 A-19.69 | H0 T-0.08 R+0.05 A+0.11 | long_top_4 [0, 8, 7, 10] T-0.26 R+0.00 A-3.81 | long_top_8 [0, 8, 7, 10, 22, 13, 25, 6] T-0.41 R+0.00 A-6.69 | short_template_core [13, 12, 11, 8, 25, 10, 26, 24] T-0.09 R+0.00 A-9.58 | all_heads T+0.64 R+0.43 A-5.36 |

