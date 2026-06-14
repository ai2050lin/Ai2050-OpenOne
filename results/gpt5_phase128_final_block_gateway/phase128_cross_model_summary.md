# Phase 128 Cross-model Final Block Gateway Summary

## qwen3

Peak layer: L35
Available components: block_input, attention_output, post_attention_norm_input, mlp_input, mlp_output, block_output, final_norm_input, final_norm_output

| category | audit | best | block input | post-attn norm input | mlp input | mlp output | block output | final norm input | final norm output |
|---|---|---|---|---|---|---|---|---|---|
| number | answer_in_pre=0, mean_pre_len=3.2 | block_output T-0.07 R+0.15 A+0.00 | block_input T-0.05 R+0.27 A+10.58 | post_attention_norm_input T+0.21 R+0.31 A+0.00 | mlp_input T+0.08 R+0.09 A+0.00 | mlp_output T+0.05 R+0.11 A+0.00 | block_output T-0.07 R+0.15 A+0.00 | final_norm_input T+0.00 R+0.00 A+0.00 | final_norm_output T+0.00 R+0.00 A+0.00 |
| container | answer_in_pre=0, mean_pre_len=3.2 | block_input T-0.04 R+0.41 A-8.84 | block_input T-0.04 R+0.41 A-8.84 | post_attention_norm_input T+0.14 R+0.38 A+0.00 | mlp_input T+0.05 R+0.09 A+0.00 | mlp_output T+0.07 R+0.12 A+0.00 | block_output T+0.07 R+0.22 A+0.00 | final_norm_input T+0.00 R+0.00 A+0.00 | final_norm_output T+0.00 R+0.00 A+0.00 |
| plant | answer_in_pre=0, mean_pre_len=3.2 | final_norm_input T+0.00 R+0.00 A+0.00 | block_input T+0.27 R+0.45 A+10.15 | post_attention_norm_input T+0.21 R+0.44 A+0.00 | mlp_input T+0.04 R+0.07 A+0.00 | mlp_output T+0.08 R+0.17 A+0.00 | block_output T+0.24 R+0.26 A+0.00 | final_norm_input T+0.00 R+0.00 A+0.00 | final_norm_output T+0.00 R+0.00 A+0.00 |

## glm4

Peak layer: L18
Available components: block_input, attention_output, post_attention_norm_input, mlp_input, mlp_output, block_output, final_norm_input, final_norm_output

| category | audit | best | block input | post-attn norm input | mlp input | mlp output | block output | final norm input | final norm output |
|---|---|---|---|---|---|---|---|---|---|
| number | answer_in_pre=0, mean_pre_len=5.2 | block_input T-0.26 R+0.27 A+0.07 | block_input T-0.26 R+0.27 A+0.07 | post_attention_norm_input T+0.10 R+0.16 A+0.00 | mlp_input T-0.00 R+0.09 A+0.00 | mlp_output T-0.06 R+0.04 A+0.00 | block_output T-0.23 R+0.29 A+0.00 | final_norm_input T+0.00 R+0.00 A+0.00 | final_norm_output T+0.00 R+0.00 A+0.00 |
| container | answer_in_pre=0, mean_pre_len=5.2 | mlp_output T-0.03 R+0.02 A+0.00 | block_input T-0.01 R+0.70 A-0.01 | post_attention_norm_input T+0.01 R+0.10 A+0.00 | mlp_input T-0.00 R+0.06 A+0.00 | mlp_output T-0.03 R+0.02 A+0.00 | block_output T+0.00 R+0.70 A+0.00 | final_norm_input T+0.00 R+0.00 A+0.00 | final_norm_output T+0.00 R+0.00 A+0.00 |
| plant | answer_in_pre=0, mean_pre_len=5.2 | block_input T-0.02 R+0.41 A+0.11 | block_input T-0.02 R+0.41 A+0.11 | post_attention_norm_input T+0.02 R+0.16 A+0.00 | mlp_input T+0.04 R+0.09 A+0.00 | mlp_output T+0.02 R+0.11 A+0.00 | block_output T-0.01 R+0.39 A+0.00 | final_norm_input T+0.00 R+0.00 A+0.00 | final_norm_output T+0.00 R+0.00 A+0.00 |

## deepseek7b

Peak layer: L27
Available components: block_input, attention_output, post_attention_norm_input, mlp_input, mlp_output, block_output, final_norm_input, final_norm_output

| category | audit | best | block input | post-attn norm input | mlp input | mlp output | block output | final norm input | final norm output |
|---|---|---|---|---|---|---|---|---|---|
| number | answer_in_pre=0, mean_pre_len=3.2 | block_output T-2.51 R+0.54 A+0.00 | block_input T-0.96 R+0.00 A-8.98 | post_attention_norm_input T+0.29 R+0.40 A+0.00 | mlp_input T+0.05 R+0.06 A+0.00 | mlp_output T+0.33 R+0.32 A+0.00 | block_output T-2.51 R+0.54 A+0.00 | final_norm_input T+0.00 R+0.00 A+0.00 | final_norm_output T+0.00 R+0.00 A+0.00 |
| container | answer_in_pre=0, mean_pre_len=3.2 | block_output T-2.66 R+0.88 A+0.00 | block_input T-0.98 R+0.00 A+8.08 | post_attention_norm_input T+0.06 R+0.24 A+0.00 | mlp_input T+0.04 R+0.08 A+0.00 | mlp_output T+0.20 R+0.36 A+0.00 | block_output T-2.66 R+0.88 A+0.00 | final_norm_input T+0.00 R+0.00 A+0.00 | final_norm_output T+0.00 R+0.00 A+0.00 |
| plant | answer_in_pre=0, mean_pre_len=3.2 | block_output T-2.42 R+1.56 A+0.00 | block_input T-1.21 R+0.00 A-2.30 | post_attention_norm_input T+0.22 R+0.29 A+0.00 | mlp_input T+0.15 R+0.15 A+0.00 | mlp_output T+0.50 R+0.49 A+0.00 | block_output T-2.42 R+1.56 A+0.00 | final_norm_input T+0.00 R+0.00 A+0.00 | final_norm_output T+0.00 R+0.00 A+0.00 |

