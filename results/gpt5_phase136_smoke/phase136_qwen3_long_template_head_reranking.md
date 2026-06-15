# Phase 136 Long-template Head Re-ranking: qwen3

Generated: 2026-06-14 23:17:47
Peak layer: L35; true last layer: L36; heads: 32; kv_heads: 8; short core: [11, 10, 28, 3, 31, 2, 5, 20]

| category | audit | reference | best head | top1 | top2 | top4 | top8 | short core | all heads |
|---|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=28.8 | last_input_pre_answer T-3.00 R+0.00 A+1.51 | H11 T-0.08 R+0.00 A-0.24 | long_top_1 T-0.08 R+0.00 A-0.24 | long_top_2 T-0.11 R+0.00 A-0.45 | long_top_4 [11, 10, 3, 5] T-0.18 R+0.00 A-0.36 | long_top_8 [11, 10, 3, 5, 28, 6, 30, 4] T-0.21 R+0.00 A-0.56 | short_template_core [11, 10, 28, 3, 31, 2, 5, 20] T-0.14 R+0.00 A-0.64 | all_heads T+1.94 R+2.82 A+2.49 |
