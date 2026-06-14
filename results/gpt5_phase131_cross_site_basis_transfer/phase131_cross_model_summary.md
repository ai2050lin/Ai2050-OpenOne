# Phase 131 Cross-model Cross-site Basis Transfer Summary

## qwen3

Peak layer: L35; true last layer: L36

| category | audit | reference | attention answer | mlp input answer | mlp output answer | block output answer | final norm answer |
|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-0.07 R+0.15 A-0.11 | last_attention_output_answer T-0.10 R+0.20 A-4.36 | last_mlp_input_answer T-0.30 R+0.00 A+0.06 | last_mlp_output_answer T+0.34 R+0.82 A-9.54 | last_block_output_answer T+1.00 R+6.05 A+47.13 | final_norm_output_answer T+0.93 R+2.52 A-16.49 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.07 R+0.22 A+0.06 | last_attention_output_answer T-0.02 R+0.22 A-2.74 | last_mlp_input_answer T+0.86 R+0.70 A-1.75 | last_mlp_output_answer T+0.35 R+0.59 A-4.62 | last_block_output_answer T+2.81 R+6.63 A+37.93 | final_norm_output_answer T+0.27 R+2.82 A-20.84 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.24 R+0.26 A-0.03 | last_attention_output_answer T-0.19 R+0.17 A-1.67 | last_mlp_input_answer T-0.28 R+0.64 A-0.87 | last_mlp_output_answer T-0.15 R+0.42 A-0.20 | last_block_output_answer T-0.13 R+5.27 A+33.96 | final_norm_output_answer T+0.20 R+2.34 A-19.50 |

## glm4

Peak layer: L18; true last layer: L40

| category | audit | reference | attention answer | mlp input answer | mlp output answer | block output answer | final norm answer |
|---|---|---|---|---|---|---|---|
| number | old_mismatch=32, mean_pre=3.2 | last_input_pre_answer T-0.05 R+0.05 A+0.90 | last_attention_output_answer T-0.10 R+0.00 A-2.90 | last_mlp_input_answer T+0.01 R+0.69 A-16.77 | last_mlp_output_answer T-0.94 R+0.25 A-0.04 | last_block_output_answer T+1.53 R+1.76 A+16.50 | final_norm_output_answer T+0.45 R+0.59 A-27.96 |
| container | old_mismatch=62, mean_pre=3.2 | last_input_pre_answer T+0.02 R+0.08 A+1.04 | last_attention_output_answer T-0.17 R+0.00 A-2.40 | last_mlp_input_answer T+0.36 R+0.53 A-15.80 | last_mlp_output_answer T-0.34 R+0.40 A-9.64 | last_block_output_answer T+1.21 R+1.60 A+13.77 | final_norm_output_answer T+0.20 R+0.84 A-33.57 |
| plant | old_mismatch=52, mean_pre=3.2 | last_input_pre_answer T-0.15 R+0.08 A+0.67 | last_attention_output_answer T-0.08 R+0.00 A-2.85 | last_mlp_input_answer T+0.44 R+0.95 A-14.93 | last_mlp_output_answer T-0.66 R+0.43 A-19.21 | last_block_output_answer T-0.49 R+2.29 A+0.94 | final_norm_output_answer T-2.28 R+1.65 A-38.13 |

## deepseek7b

Peak layer: L27; true last layer: L28

| category | audit | reference | attention answer | mlp input answer | mlp output answer | block output answer | final norm answer |
|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.51 R+0.54 A-5.42 | last_attention_output_answer T+1.26 R+1.72 A+7.98 | last_mlp_input_answer T-0.43 R+0.10 A-0.60 | last_mlp_output_answer T+0.14 R+0.34 A-4.43 | last_block_output_answer T-0.48 R+0.91 A-17.97 | final_norm_output_answer T+1.59 R+2.16 A-18.13 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.66 R+0.88 A-3.12 | last_attention_output_answer T+0.26 R+2.70 A+11.39 | last_mlp_input_answer T-0.03 R+2.58 A+1.13 | last_mlp_output_answer T+0.38 R+0.31 A-5.24 | last_block_output_answer T+0.85 R+0.92 A-20.14 | final_norm_output_answer T-0.05 R+2.96 A-19.15 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-2.42 R+1.56 A-3.37 | last_attention_output_answer T+1.44 R+2.58 A+3.02 | last_mlp_input_answer T+1.72 R+2.49 A-0.57 | last_mlp_output_answer T-0.24 R+0.30 A-4.81 | last_block_output_answer T-0.69 R+1.25 A-16.15 | final_norm_output_answer T+1.06 R+2.44 A-21.24 |

