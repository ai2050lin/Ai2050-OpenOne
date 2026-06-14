# Phase 130 True-last Attention Read Gateway: glm4

Generated: 2026-06-14 21:27:21
Peak layer: L18; true last layer: L40; heads: 32

| category | audit | reference pre-answer | best answer component | top head by mass | best head ablation |
|---|---|---|---|---|---|
| number | answer_in_pre=0, old_mismatch=32, mean_pre=3.2 | last_input_pre_answer T-0.05 R+0.05 A+0.81 | last_mlp_output_answer T-1.13 R+1.53 A-64.01 | H6 pre0.744 self0.108 | H26 T-0.01 R+0.01 A-0.31 |
| container | answer_in_pre=0, old_mismatch=62, mean_pre=3.2 | last_input_pre_answer T+0.02 R+0.08 A+0.66 | last_mlp_input_answer T-0.26 R+0.55 A-27.78 | H6 pre0.771 self0.134 | H27 T-0.01 R+0.00 A-0.39 |
| plant | answer_in_pre=0, old_mismatch=52, mean_pre=3.2 | last_input_pre_answer T-0.15 R+0.08 A+0.23 | last_block_output_answer T-0.53 R+0.57 A-18.23 | H6 pre0.774 self0.098 | H8 T-0.02 R+0.03 A-0.20 |
