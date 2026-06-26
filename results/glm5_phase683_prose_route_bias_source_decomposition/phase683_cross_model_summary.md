# Phase 683 Prose-Route Bias Source Decomposition

- generated: `2026-06-26 12:23:40`

| model | rows | value_top1 | value_final_pmv | short_top1 | short_final_pmv | terse_final_pmv | bare_final_pmv | sentence_final_pmv | explanation_final_pmv | json_final_pmv | label_final_pmv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek7b | 1008 | 0.368 | 1.429 | 0.083 | 4.677 | -0.930 | 0.540 | 10.178 | 6.419 | 7.616 | 9.279 |
| glm4 | 1008 | 0.597 | -2.075 | 0.889 | -5.917 | -4.135 | 3.828 | 4.441 | 7.926 | 5.437 | 0.258 |
| qwen3 | 1008 | 0.650 | -2.923 | 0.965 | -6.504 | -7.421 | 5.157 | 14.580 | 14.903 | 8.655 | 9.834 |

## Variant Details

### deepseek7b

| variant | n | top1 | mean_rank | protocol_pmv | final_norm_input_pmv | final_pmv | final_target_margin | failure_best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bare_answer | 144 | 0.438 | 2.08 | -2.034 | -92.613 | 0.540 | -0.540 | {'prose': 81} |
| explanation | 144 | 0.993 | 1.01 | -4.280 | -40.946 | 6.419 | 3.334 | {'continuation': 1} |
| json | 144 | 0.896 | 1.10 | -3.548 | -31.214 | 7.616 | 2.616 | {'prose': 15} |
| label | 144 | 0.986 | 1.01 | -5.211 | -26.493 | 9.279 | 1.916 | {'prose': 2} |
| sentence | 144 | 1.000 | 1.00 | -4.632 | -30.424 | 10.178 | 5.384 | {} |
| short_only | 144 | 0.083 | 277.51 | -4.651 | -51.511 | 4.677 | -4.677 | {'prose': 132} |
| terse_no_explain | 144 | 0.583 | 3.04 | -5.263 | -86.988 | -0.930 | 0.930 | {'prose': 60} |

### glm4

| variant | n | top1 | mean_rank | protocol_pmv | final_norm_input_pmv | final_pmv | final_target_margin | failure_best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bare_answer | 144 | 0.049 | 4.44 | 0.121 | -10.048 | 3.828 | -3.828 | {'prose': 137} |
| explanation | 144 | 1.000 | 1.00 | 0.132 | -6.556 | 7.926 | 6.630 | {} |
| json | 144 | 1.000 | 1.00 | 0.090 | 1.162 | 5.437 | 4.913 | {} |
| label | 144 | 1.000 | 1.00 | 0.037 | -8.709 | 0.258 | 2.406 | {} |
| sentence | 144 | 1.000 | 1.00 | 0.197 | -8.551 | 4.441 | 4.430 | {} |
| short_only | 144 | 0.889 | 1.17 | 0.038 | -15.184 | -5.917 | 3.245 | {'continuation': 16} |
| terse_no_explain | 144 | 0.854 | 1.17 | 0.078 | -16.762 | -4.135 | 4.059 | {'continuation': 1, 'prose': 20} |

### qwen3

| variant | n | top1 | mean_rank | protocol_pmv | final_norm_input_pmv | final_pmv | final_target_margin | failure_best_other |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bare_answer | 144 | 0.000 | 5.14 | 1.030 | -17.174 | 5.157 | -5.157 | {'prose': 144} |
| explanation | 144 | 1.000 | 1.00 | 1.356 | 16.170 | 14.903 | 11.918 | {} |
| json | 144 | 1.000 | 1.00 | 0.807 | 16.879 | 8.655 | -0.112 | {} |
| label | 144 | 1.000 | 1.00 | 1.047 | 3.684 | 9.834 | 7.323 | {} |
| sentence | 144 | 1.000 | 1.00 | 1.762 | 19.965 | 14.580 | 8.604 | {} |
| short_only | 144 | 0.965 | 1.24 | 0.726 | -64.462 | -6.504 | 5.480 | {'continuation': 4, 'prose': 1} |
| terse_no_explain | 144 | 0.986 | 1.05 | 1.030 | -58.030 | -7.421 | 6.491 | {'continuation': 1, 'prose': 1} |

