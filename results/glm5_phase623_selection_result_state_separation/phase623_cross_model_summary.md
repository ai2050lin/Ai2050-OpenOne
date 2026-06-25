# Phase 623 Cross-Model Summary

Selection state and result state patch combinations.

## deepseek7b

- rows: 82 / raw 256
- target cases seen: 82
- patch layers: [20, 21, 22]
- selection layers: [20, 21, 22]

| mode | switch | margin | correct_delta | wrong_delta | qproj | alpha_cv | alpha_wrong_rel | norm_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| result_only | 75/82 | 2.890 | 1.537 | -1.352 | 0.000 | 0.00000 | 0.00000 | 0.921 |
| result_random_norm | 2/82 | -0.092 | -0.094 | -0.002 | 0.000 | 0.00000 | 0.00000 | 1.000 |
| selection_both | 63/82 | 1.907 | 1.192 | -0.715 | 0.538 | 0.07563 | -0.00636 | 0.465 |
| selection_both_plus_result | 75/82 | 2.892 | 1.540 | -1.352 | 0.538 | 0.07563 | -0.00636 | 0.617 |
| selection_early | 64/82 | 1.979 | 1.240 | -0.739 | 0.545 | 0.07503 | -0.00636 | 0.472 |
| selection_late | 62/82 | 1.904 | 1.187 | -0.716 | 0.242 | 0.05826 | -0.01191 | 0.457 |
| selection_late_plus_result | 75/82 | 2.889 | 1.536 | -1.353 | 0.242 | 0.05826 | -0.01191 | 0.689 |
| selection_random_norm | 8/82 | -0.031 | -0.118 | -0.087 | 0.011 | -0.00582 | -0.00129 | 1.000 |

Top modes:
- selection_both_plus_result: switch=75/82, margin=2.892, qproj=0.538, alpha_cv=0.07563
- result_only: switch=75/82, margin=2.890, qproj=0.000, alpha_cv=0.00000
- selection_late_plus_result: switch=75/82, margin=2.889, qproj=0.242, alpha_cv=0.05826
- selection_early: switch=64/82, margin=1.979, qproj=0.545, alpha_cv=0.07503
- selection_both: switch=63/82, margin=1.907, qproj=0.538, alpha_cv=0.07563
- selection_late: switch=62/82, margin=1.904, qproj=0.242, alpha_cv=0.05826

## glm4

- rows: 31 / raw 256
- target cases seen: 31
- patch layers: [31, 32, 34]
- selection layers: [32, 33, 34]

| mode | switch | margin | correct_delta | wrong_delta | qproj | alpha_cv | alpha_wrong_rel | norm_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| result_only | 29/31 | 2.131 | 0.974 | -1.157 | 0.000 | 0.00000 | 0.00000 | 0.969 |
| result_random_norm | 3/31 | -0.069 | -0.066 | 0.003 | 0.000 | 0.00000 | 0.00000 | 1.000 |
| selection_both | 7/31 | 0.347 | 0.192 | -0.155 | 0.886 | 0.03123 | -0.04563 | 0.320 |
| selection_both_plus_result | 29/31 | 2.131 | 0.974 | -1.157 | 0.886 | 0.03123 | -0.04563 | 0.537 |
| selection_early | 9/31 | 0.476 | 0.248 | -0.227 | 0.882 | 0.02395 | -0.03972 | 0.340 |
| selection_late | 7/31 | 0.347 | 0.192 | -0.155 | 0.588 | 0.01520 | -0.02870 | 0.301 |
| selection_late_plus_result | 29/31 | 2.131 | 0.974 | -1.157 | 0.588 | 0.01520 | -0.02870 | 0.635 |
| selection_random_norm | 1/31 | -0.082 | -0.051 | 0.031 | 0.030 | 0.00065 | -0.00515 | 1.000 |

Top modes:
- selection_both_plus_result: switch=29/31, margin=2.131, qproj=0.886, alpha_cv=0.03123
- selection_late_plus_result: switch=29/31, margin=2.131, qproj=0.588, alpha_cv=0.01520
- result_only: switch=29/31, margin=2.131, qproj=0.000, alpha_cv=0.00000
- selection_early: switch=9/31, margin=0.476, qproj=0.882, alpha_cv=0.02395
- selection_both: switch=7/31, margin=0.347, qproj=0.886, alpha_cv=0.03123
- selection_late: switch=7/31, margin=0.347, qproj=0.588, alpha_cv=0.01520

## qwen3

- rows: 17 / raw 256
- target cases seen: 17
- patch layers: [26, 27, 29]
- selection layers: [27, 28, 29]

| mode | switch | margin | correct_delta | wrong_delta | qproj | alpha_cv | alpha_wrong_rel | norm_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| result_only | 15/17 | 4.407 | 1.814 | -2.593 | 0.000 | 0.00000 | 0.00000 | 0.930 |
| result_random_norm | 2/17 | 0.088 | 0.060 | -0.029 | 0.000 | 0.00000 | 0.00000 | 1.000 |
| selection_both | 9/17 | 1.384 | 0.850 | -0.534 | 0.847 | 0.03001 | 0.00748 | 0.495 |
| selection_both_plus_result | 15/17 | 4.407 | 1.814 | -2.593 | 0.847 | 0.03001 | 0.00748 | 0.640 |
| selection_early | 9/17 | 1.553 | 0.893 | -0.660 | 0.803 | 0.02526 | -0.00562 | 0.499 |
| selection_late | 9/17 | 1.384 | 0.850 | -0.534 | 0.534 | 0.02966 | 0.00539 | 0.491 |
| selection_late_plus_result | 15/17 | 4.407 | 1.814 | -2.593 | 0.534 | 0.02966 | 0.00539 | 0.710 |
| selection_random_norm | 2/17 | 0.131 | 0.054 | -0.078 | 0.020 | -0.00065 | 0.00054 | 1.000 |

Top modes:
- selection_both_plus_result: switch=15/17, margin=4.407, qproj=0.847, alpha_cv=0.03001
- result_only: switch=15/17, margin=4.407, qproj=0.000, alpha_cv=0.00000
- selection_late_plus_result: switch=15/17, margin=4.407, qproj=0.534, alpha_cv=0.02966
- selection_early: switch=9/17, margin=1.553, qproj=0.803, alpha_cv=0.02526
- selection_late: switch=9/17, margin=1.384, qproj=0.534, alpha_cv=0.02966
- selection_both: switch=9/17, margin=1.384, qproj=0.847, alpha_cv=0.03001
