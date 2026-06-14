# Phase 131 Cross-site Basis Transfer: qwen3

Generated: 2026-06-14 21:30:07
Peak layer: L35; true last layer: L36

| category | audit | reference | best same-basis answer component | attention answer | block output answer | final norm answer |
|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T-0.07 R+0.15 A-0.11 | last_mlp_input_answer T-0.30 R+0.00 A+0.06 | last_attention_output_answer T-0.10 R+0.20 A-4.36 | last_block_output_answer T+1.00 R+6.05 A+47.13 | final_norm_output_answer T+0.93 R+2.52 A-16.49 |
| container | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.07 R+0.22 A+0.06 | last_attention_output_answer T-0.02 R+0.22 A-2.74 | last_attention_output_answer T-0.02 R+0.22 A-2.74 | last_block_output_answer T+2.81 R+6.63 A+37.93 | final_norm_output_answer T+0.27 R+2.82 A-20.84 |
| plant | old_mismatch=0, mean_pre=3.2 | last_input_pre_answer T+0.24 R+0.26 A-0.03 | last_mlp_input_answer T-0.28 R+0.64 A-0.87 | last_attention_output_answer T-0.19 R+0.17 A-1.67 | last_block_output_answer T-0.13 R+5.27 A+33.96 | final_norm_output_answer T+0.20 R+2.34 A-19.50 |
