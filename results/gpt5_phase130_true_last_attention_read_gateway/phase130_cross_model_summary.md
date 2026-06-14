# Phase 130 Cross-model True-last Attention Read Gateway Summary

## qwen3

Peak layer: L35; true last layer: L36; heads: 32

| category | audit | reference | attention answer | mlp input answer | mlp output answer | block output answer | final norm answer | best head ablation |
|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-0.07 R+0.15 A+0.38 | last_attention_output_answer T+0.75 R+1.01 A+1.64 | last_mlp_input_answer T-1.34 R+0.00 A-7.92 | last_mlp_output_answer T-8.91 R+0.00 A-32.72 | last_block_output_answer T-6.08 R+0.00 A+23.49 | final_norm_output_answer T-0.64 R+1.04 A-6.78 | H5 pre0.408 T-0.06 R+0.02 A+0.16 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.07 R+0.22 A+0.37 | last_attention_output_answer T+0.71 R+0.68 A+1.16 | last_mlp_input_answer T-0.01 R+0.00 A-3.78 | last_mlp_output_answer T-4.54 R+0.00 A-15.01 | last_block_output_answer T-3.72 R+0.65 A+23.05 | final_norm_output_answer T-0.70 R+0.74 A-6.51 | H16 pre0.228 T-0.02 R+0.05 A+0.04 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.24 R+0.26 A+0.51 | last_attention_output_answer T+0.08 R+0.62 A+1.50 | last_mlp_input_answer T-0.66 R+0.01 A-2.84 | last_mlp_output_answer T-8.35 R+0.00 A-17.54 | last_block_output_answer T-5.60 R+0.00 A+19.35 | final_norm_output_answer T-2.39 R+0.00 A-8.61 | H2 pre0.308 T-0.01 R+0.05 A+0.16 |

## glm4

Peak layer: L18; true last layer: L40; heads: 32

| category | audit | reference | attention answer | mlp input answer | mlp output answer | block output answer | final norm answer | best head ablation |
|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=32, mean_pre=3.2 | last_input_pre_answer T-0.05 R+0.05 A+0.81 | last_attention_output_answer T-0.11 R+0.16 A-4.06 | last_mlp_input_answer T-0.44 R+0.16 A-37.52 | last_mlp_output_answer T-1.13 R+1.53 A-64.01 | last_block_output_answer T+0.91 R+0.88 A-8.52 | final_norm_output_answer T-0.53 R+0.85 A-26.69 | H26 pre0.724 T-0.01 R+0.01 A-0.31 |
| container | old_mismatch=62, mean_pre=3.2 | last_input_pre_answer T+0.02 R+0.08 A+0.66 | last_attention_output_answer T-0.00 R+0.09 A-4.03 | last_mlp_input_answer T-0.26 R+0.55 A-27.78 | last_mlp_output_answer T+1.31 R+1.22 A-53.63 | last_block_output_answer T+0.43 R+1.23 A-4.76 | final_norm_output_answer T+0.62 R+1.26 A-18.81 | H27 pre0.475 T-0.01 R+0.00 A-0.39 |
| plant | old_mismatch=52, mean_pre=3.2 | last_input_pre_answer T-0.15 R+0.08 A+0.23 | last_attention_output_answer T+0.20 R+0.29 A-8.19 | last_mlp_input_answer T+0.78 R+1.23 A-18.36 | last_mlp_output_answer T-0.11 R+1.21 A-51.69 | last_block_output_answer T-0.53 R+0.57 A-18.23 | final_norm_output_answer T-0.40 R+0.49 A-25.41 | H8 pre0.454 T-0.02 R+0.03 A-0.20 |

## deepseek7b

Peak layer: L27; true last layer: L28; heads: 28

| category | audit | reference | attention answer | mlp input answer | mlp output answer | block output answer | final norm answer | best head ablation |
|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.51 R+0.54 A-35.14 | last_attention_output_answer T-5.09 R+0.00 A-87.69 | last_mlp_input_answer T-2.78 R+0.00 A-93.60 | last_mlp_output_answer T-9.40 R+0.00 A-123.23 | last_block_output_answer T-11.98 R+0.00 A-67.04 | final_norm_output_answer T-7.82 R+0.00 A-142.86 | H8 pre0.615 T-0.25 R+0.00 A-2.38 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.66 R+0.88 A-35.01 | last_attention_output_answer T-4.83 R+0.00 A-92.44 | last_mlp_input_answer T-8.21 R+0.00 A-133.17 | last_mlp_output_answer T-11.45 R+0.00 A-97.89 | last_block_output_answer T-11.33 R+0.00 A-75.49 | final_norm_output_answer T-6.44 R+0.00 A-128.25 | H25 pre0.413 T-0.11 R+0.00 A+0.13 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.42 R+1.56 A-34.38 | last_attention_output_answer T-4.16 R+0.00 A-77.39 | last_mlp_input_answer T-3.63 R+0.00 A-115.90 | last_mlp_output_answer T-7.67 R+0.00 A-94.87 | last_block_output_answer T-9.62 R+0.00 A-75.94 | final_norm_output_answer T-5.63 R+0.00 A-122.69 | H8 pre0.580 T-0.28 R+0.00 A-2.44 |

