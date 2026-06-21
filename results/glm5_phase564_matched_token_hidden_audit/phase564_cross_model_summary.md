# Phase564 Matched-Token Hidden Audit Cross-Model Summary

free = baseline free generation vs intervention free generation.
matched_base = both runs forced through baseline generated tokens.
matched_condition = both runs forced through intervention generated tokens.

## Route: forbidden_sentence_completion:temperature<-forbidden_definition

### clean_non_object_rate

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.54 | 0.47 | 0.40 | 0.49 | 0.53 |
| glm4 | 0.25 | 0.39 | 0.46 | 0.31 | 0.39 |
| deepseek7b | 0.22 | 0.22 | 0.17 | 0.14 | 0.18 |

### exact_sequence_match_rate

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 1.00 | 0.04 | 0.08 | 0.00 | 0.22 |
| glm4 | 1.00 | 0.01 | 0.01 | 0.00 | 0.15 |
| deepseek7b | 1.00 | 0.01 | 0.01 | 0.00 | 0.28 |

### avg_first_divergence_step

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 12.00 | 2.29 | 2.90 | 0.01 | 4.71 |
| glm4 | 12.00 | 0.76 | 0.89 | 0.00 | 2.81 |
| deepseek7b | 12.00 | 1.44 | 1.31 | 0.17 | 5.79 |

### free trajectory_distance

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 411.15 | 383.98 | 556.53 | 283.20 |
| glm4 | 0.00 | 390.81 | 377.78 | 449.90 | 287.96 |
| deepseek7b | 0.00 | 2817.47 | 2889.80 | 3400.50 | 1406.53 |

### matched_base trajectory_distance

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 55.76 | 55.94 | 75.89 | 16.25 |
| glm4 | 0.00 | 43.00 | 45.90 | 69.71 | 17.85 |
| deepseek7b | 0.00 | 353.23 | 358.69 | 527.61 | 24.24 |

### matched_condition trajectory_distance

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 56.58 | 56.96 | 80.16 | 16.35 |
| glm4 | 0.00 | 42.63 | 45.25 | 70.68 | 17.76 |
| deepseek7b | 0.00 | 352.87 | 362.07 | 534.87 | 24.27 |

### free hidden_relax_step

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 1.00 | 2.83 | 12.00 | 2.83 |
| glm4 | 0.00 | 12.00 | 12.00 | 12.00 | 8.33 |
| deepseek7b | 0.00 | 12.00 | 12.00 | 12.00 | 8.33 |

### matched_base hidden_relax_step

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| glm4 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| deepseek7b | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### matched_condition hidden_relax_step

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| glm4 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| deepseek7b | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### free finite_time_growth

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | -0.00 | -0.01 | -0.03 | 0.09 |
| glm4 | 0.00 | 0.02 | 0.01 | -0.03 | 0.06 |
| deepseek7b | 0.00 | 0.01 | 0.01 | -0.03 | 0.24 |

### matched_base finite_time_growth

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | -0.33 | -0.33 | -0.39 | -0.33 |
| glm4 | 0.00 | -0.30 | -0.29 | -0.33 | -0.32 |
| deepseek7b | 0.00 | -0.34 | -0.33 | -0.36 | -0.23 |

### matched_condition finite_time_growth

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | -0.32 | -0.31 | -0.41 | -0.33 |
| glm4 | 0.00 | -0.35 | -0.33 | -0.34 | -0.33 |
| deepseek7b | 0.00 | -0.34 | -0.33 | -0.36 | -0.23 |

### free vs matched_base retention

| model | condition | clean_delta | free_traj | matched_base_traj | retention | seq_match | first_div |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | r2 | -0.07 | 411.15 | 55.76 | 0.14 | 0.04 | 2.29 |
| qwen3 | r4 | -0.14 | 383.98 | 55.94 | 0.15 | 0.08 | 2.90 |
| qwen3 | rand | -0.06 | 556.53 | 75.89 | 0.14 | 0.00 | 0.01 |
| qwen3 | norm_r2 | -0.01 | 283.20 | 16.25 | 0.06 | 0.22 | 4.71 |
| glm4 | r2 | +0.14 | 390.81 | 43.00 | 0.11 | 0.01 | 0.76 |
| glm4 | r4 | +0.21 | 377.78 | 45.90 | 0.12 | 0.01 | 0.89 |
| glm4 | rand | +0.06 | 449.90 | 69.71 | 0.15 | 0.00 | 0.00 |
| glm4 | norm_r2 | +0.14 | 287.96 | 17.85 | 0.06 | 0.15 | 2.81 |
| deepseek7b | r2 | +0.00 | 2817.47 | 353.23 | 0.13 | 0.01 | 1.44 |
| deepseek7b | r4 | -0.06 | 2889.80 | 358.69 | 0.12 | 0.01 | 1.31 |
| deepseek7b | rand | -0.08 | 3400.50 | 527.61 | 0.16 | 0.00 | 0.17 |
| deepseek7b | norm_r2 | -0.04 | 1406.53 | 24.24 | 0.02 | 0.28 | 5.79 |

## Route: forbidden_definition:top_p<-forbidden_definition

### clean_non_object_rate

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.28 | 0.24 | 0.31 | 0.32 | 0.28 |
| glm4 | 0.39 | 0.36 | 0.35 | 0.38 | 0.42 |
| deepseek7b | 0.17 | 0.17 | 0.19 | 0.14 | 0.17 |

### exact_sequence_match_rate

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 1.00 | 0.33 | 0.26 | 0.01 | 0.32 |
| glm4 | 1.00 | 0.11 | 0.11 | 0.00 | 0.19 |
| deepseek7b | 1.00 | 0.22 | 0.19 | 0.04 | 0.47 |

### avg_first_divergence_step

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 12.00 | 5.83 | 5.26 | 0.24 | 5.79 |
| glm4 | 12.00 | 2.36 | 2.21 | 0.00 | 3.39 |
| deepseek7b | 12.00 | 4.89 | 4.04 | 0.61 | 7.69 |

### free trajectory_distance

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 264.85 | 282.99 | 573.78 | 267.06 |
| glm4 | 0.00 | 334.60 | 349.67 | 473.29 | 299.24 |
| deepseek7b | 0.00 | 1826.77 | 2021.58 | 3418.92 | 1068.71 |

### matched_base trajectory_distance

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 18.90 | 19.73 | 75.83 | 14.88 |
| glm4 | 0.00 | 25.31 | 26.03 | 64.13 | 17.46 |
| deepseek7b | 0.00 | 133.02 | 134.89 | 512.76 | 24.68 |

### matched_condition trajectory_distance

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 18.83 | 19.77 | 78.25 | 14.74 |
| glm4 | 0.00 | 25.15 | 25.93 | 65.96 | 17.43 |
| deepseek7b | 0.00 | 132.40 | 135.99 | 527.46 | 24.55 |

### free hidden_relax_step

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 4.67 | 4.67 | 12.00 | 6.50 |
| glm4 | 0.00 | 12.00 | 12.00 | 12.00 | 12.00 |
| deepseek7b | 0.00 | 12.00 | 12.00 | 12.00 | 6.50 |

### matched_base hidden_relax_step

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| glm4 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| deepseek7b | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### matched_condition hidden_relax_step

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| glm4 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| deepseek7b | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### free finite_time_growth

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | 0.07 | 0.08 | -0.02 | 0.10 |
| glm4 | 0.00 | 0.04 | 0.05 | -0.03 | 0.07 |
| deepseek7b | 0.00 | 0.08 | 0.08 | -0.03 | 0.22 |

### matched_base finite_time_growth

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | -0.31 | -0.31 | -0.39 | -0.31 |
| glm4 | 0.00 | -0.41 | -0.40 | -0.40 | -0.40 |
| deepseek7b | 0.00 | -0.32 | -0.31 | -0.36 | -0.23 |

### matched_condition finite_time_growth

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.00 | -0.32 | -0.31 | -0.40 | -0.32 |
| glm4 | 0.00 | -0.43 | -0.40 | -0.39 | -0.40 |
| deepseek7b | 0.00 | -0.32 | -0.30 | -0.36 | -0.23 |

### free vs matched_base retention

| model | condition | clean_delta | free_traj | matched_base_traj | retention | seq_match | first_div |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | r2 | -0.04 | 264.85 | 18.90 | 0.07 | 0.33 | 5.83 |
| qwen3 | r4 | +0.03 | 282.99 | 19.73 | 0.07 | 0.26 | 5.26 |
| qwen3 | rand | +0.04 | 573.78 | 75.83 | 0.13 | 0.01 | 0.24 |
| qwen3 | norm_r2 | +0.00 | 267.06 | 14.88 | 0.06 | 0.32 | 5.79 |
| glm4 | r2 | -0.03 | 334.60 | 25.31 | 0.08 | 0.11 | 2.36 |
| glm4 | r4 | -0.04 | 349.67 | 26.03 | 0.07 | 0.11 | 2.21 |
| glm4 | rand | -0.01 | 473.29 | 64.13 | 0.14 | 0.00 | 0.00 |
| glm4 | norm_r2 | +0.03 | 299.24 | 17.46 | 0.06 | 0.19 | 3.39 |
| deepseek7b | r2 | +0.00 | 1826.77 | 133.02 | 0.07 | 0.22 | 4.89 |
| deepseek7b | r4 | +0.03 | 2021.58 | 134.89 | 0.07 | 0.19 | 4.04 |
| deepseek7b | rand | -0.03 | 3418.92 | 512.76 | 0.15 | 0.04 | 0.61 |
| deepseek7b | norm_r2 | +0.00 | 1068.71 | 24.68 | 0.02 | 0.47 | 7.69 |

## Timing

| model | minutes | seeds | max tokens | epsilon |
|---|---:|---|---:|---:|
| qwen3 | 16.03 | [101, 103, 107, 109, 113, 127] | 12 | 0.25 |
| glm4 | 28.85 | [101, 103, 107, 109, 113, 127] | 12 | 0.25 |
| deepseek7b | 22.18 | [101, 103, 107, 109, 113, 127] | 12 | 0.25 |
