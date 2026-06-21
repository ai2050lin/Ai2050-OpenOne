# Phase563 Hidden Trajectory Distance Cross-Model Summary

Metrics: hidden deltas are measured against same-seed baseline trajectories.
hidden_relax_step uses first step where delta_ratio <= epsilon_ratio.

## Route: forbidden_sentence_completion:temperature<-forbidden_definition

### clean_non_object_rate

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.54 | 0.47 | 0.40 | 0.44 | 0.49 | 0.49 | 0.53 |
| glm4 | 0.25 | 0.39 | 0.46 | 0.32 | 0.31 | 0.36 | 0.39 |
| deepseek7b | 0.22 | 0.22 | 0.17 | 0.18 | 0.14 | 0.14 | 0.18 |

### hidden_relax_step

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 1.00 | 2.83 | 1.00 | 12.00 | 6.50 | 2.83 |
| glm4 | 0.00 | 12.00 | 12.00 | 10.17 | 12.00 | 12.00 | 8.33 |
| deepseek7b | 0.00 | 12.00 | 12.00 | 10.17 | 12.00 | 10.17 | 8.33 |

### finite_time_log_growth

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | -0.00 | -0.01 | -0.00 | -0.03 | 0.09 | 0.09 |
| glm4 | 0.00 | 0.02 | 0.01 | 0.02 | -0.03 | 0.08 | 0.06 |
| deepseek7b | 0.00 | 0.01 | 0.01 | 0.01 | -0.03 | 0.24 | 0.24 |

### trajectory_distance

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 411.15 | 383.98 | 408.49 | 556.53 | 325.45 | 283.20 |
| glm4 | 0.00 | 390.81 | 377.78 | 370.31 | 449.90 | 377.82 | 287.96 |
| deepseek7b | 0.00 | 2817.47 | 2889.80 | 2807.70 | 3400.50 | 1488.60 | 1406.53 |

### last_delta_ratio

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 0.97 | 0.93 | 1.00 | 0.70 | 2.67 | 2.73 |
| glm4 | 0.00 | 1.21 | 1.14 | 1.21 | 0.70 | 2.38 | 2.04 |
| deepseek7b | 0.00 | 1.08 | 1.11 | 1.15 | 0.75 | 14.22 | 14.33 |

### tangent vs normal

| model | baseline | tangent clean | normal clean | tangent growth | normal growth | tangent traj | normal traj |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.54 | 0.49 | 0.53 | 0.09 | 0.09 | 325.45 | 283.20 |
| glm4 | 0.25 | 0.36 | 0.39 | 0.08 | 0.06 | 377.82 | 287.96 |
| deepseek7b | 0.22 | 0.14 | 0.18 | 0.24 | 0.24 | 1488.60 | 1406.53 |

## Route: forbidden_definition:top_p<-forbidden_definition

### clean_non_object_rate

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.28 | 0.24 | 0.31 | 0.24 | 0.32 | 0.28 | 0.28 |
| glm4 | 0.39 | 0.36 | 0.35 | 0.36 | 0.38 | 0.33 | 0.42 |
| deepseek7b | 0.17 | 0.17 | 0.19 | 0.21 | 0.14 | 0.17 | 0.17 |

### hidden_relax_step

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 4.67 | 4.67 | 4.67 | 12.00 | 12.00 | 6.50 |
| glm4 | 0.00 | 12.00 | 12.00 | 12.00 | 12.00 | 12.00 | 12.00 |
| deepseek7b | 0.00 | 12.00 | 12.00 | 12.00 | 12.00 | 12.00 | 6.50 |

### finite_time_log_growth

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 0.07 | 0.08 | 0.09 | -0.02 | 0.11 | 0.10 |
| glm4 | 0.00 | 0.04 | 0.05 | 0.04 | -0.03 | 0.09 | 0.07 |
| deepseek7b | 0.00 | 0.08 | 0.08 | 0.11 | -0.03 | 0.23 | 0.22 |

### trajectory_distance

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 264.85 | 282.99 | 267.24 | 573.78 | 338.74 | 267.06 |
| glm4 | 0.00 | 334.60 | 349.67 | 315.15 | 473.29 | 324.10 | 299.24 |
| deepseek7b | 0.00 | 1826.77 | 2021.58 | 1949.92 | 3418.92 | 1152.61 | 1068.71 |

### last_delta_ratio

| model | base | r2 | r4 | mean | rand | tan_r2 | norm_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 2.29 | 2.52 | 2.81 | 0.78 | 3.33 | 2.99 |
| glm4 | 0.00 | 1.60 | 1.66 | 1.48 | 0.75 | 2.80 | 2.10 |
| deepseek7b | 0.00 | 2.41 | 2.48 | 3.22 | 0.73 | 13.06 | 11.46 |

### tangent vs normal

| model | baseline | tangent clean | normal clean | tangent growth | normal growth | tangent traj | normal traj |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.28 | 0.28 | 0.28 | 0.11 | 0.10 | 338.74 | 267.06 |
| glm4 | 0.39 | 0.33 | 0.42 | 0.09 | 0.07 | 324.10 | 299.24 |
| deepseek7b | 0.17 | 0.17 | 0.17 | 0.23 | 0.22 | 1152.61 | 1068.71 |

## Timing

| model | minutes | seeds | max tokens | epsilon |
|---|---:|---|---:|---:|
| qwen3 | 9.0 | [101, 103, 107, 109, 113, 127] | 12 | 0.25 |
| glm4 | 16.08 | [101, 103, 107, 109, 113, 127] | 12 | 0.25 |
| deepseek7b | 12.38 | [101, 103, 107, 109, 113, 127] | 12 | 0.25 |
