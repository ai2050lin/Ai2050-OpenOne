# Phase 674 Synthetic Value Readout Competitor Source Localization

- generated: `2026-06-26 10:51:22`

| model | cases | top1_rate | mean_rank | expected_minus_comp | norm_shift | top1_category | source_diag |
|---|---:|---:|---:|---:|---:|---|---|
| deepseek7b | 72 | 0.125 | 469.79 | -6.301 | 73.982 | {'word_or_explanation': 40, 'other': 17, 'expected': 9, 'newline': 6} | {'direction_alignment': 63, 'expected_wins': 9} |
| glm4 | 72 | 1.000 | 1.00 | 3.329 | 34.829 | {'expected': 72} | {'expected_wins': 72} |
| qwen3 | 72 | 0.931 | 1.49 | 5.341 | 60.114 | {'expected': 67, 'other': 5} | {'expected_wins': 67, 'direction_alignment': 4, 'projection_norm_advantage': 1} |