# Phase565 Early Gate and Token Fork Cross-Model Summary

bfi = baseline forced to use intervention first token.
ifb = intervention forced to use baseline first token.

## Route: forbidden_sentence_completion:temperature<-forbidden_definition

### Clean Rates

| model | base | intervention | free | bfi | ifb | bfi_transfer | ifb_transfer | first_div |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.54 | r2 | 0.47 | 0.56 | 0.40 | -0.20 | -1.00 | 2.29 |
| qwen3 | 0.54 | r4 | 0.40 | 0.57 | 0.39 | -0.20 | -0.10 | 2.90 |
| qwen3 | 0.54 | rand | 0.49 | 0.56 | 0.51 | -0.25 | 0.50 | 0.01 |
| qwen3 | 0.54 | norm_r2 | 0.53 | 0.57 | 0.36 | -2.00 | -12.00 | 4.71 |
| glm4 | 0.25 | r2 | 0.39 | 0.31 | 0.35 | 0.40 | 0.30 | 0.76 |
| glm4 | 0.25 | r4 | 0.46 | 0.31 | 0.44 | 0.27 | 0.07 | 0.89 |
| glm4 | 0.25 | rand | 0.31 | 0.31 | 0.33 | 1.00 | -0.50 | 0.00 |
| glm4 | 0.25 | norm_r2 | 0.39 | 0.35 | 0.39 | 0.70 | -0.00 | 2.81 |
| deepseek7b | 0.22 | r2 | 0.22 | 0.15 | 0.17 | 0.00 | 0.00 | 1.44 |
| deepseek7b | 0.22 | r4 | 0.17 | 0.14 | 0.18 | 1.50 | 0.25 | 1.31 |
| deepseek7b | 0.22 | rand | 0.14 | 0.15 | 0.25 | 0.83 | 1.33 | 0.17 |
| deepseek7b | 0.22 | norm_r2 | 0.18 | 0.15 | 0.18 | 1.67 | 0.00 | 5.79 |

### Step0 Target-Competitor Margin

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 0.45 | 1.39 | 1.35 | 1.25 | 0.61 |
| glm4 | -1.28 | 3.64 | 3.61 | -1.44 | 2.09 |
| deepseek7b | 0.81 | -1.76 | -1.29 | 0.46 | 0.53 |

### Step1 Target-Competitor Margin

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | -0.25 | 0.97 | 1.01 | 1.62 | 0.06 |
| glm4 | -2.29 | 0.07 | -0.26 | -0.73 | -1.64 |
| deepseek7b | 0.96 | 1.39 | 1.15 | 2.43 | 0.83 |

### Fork Buckets For Free Intervention

| model | intervention | early_rate | early_clean | middle_rate | middle_clean | late_rate | late_clean |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | r2 | 0.67 | 0.44 | 0.21 | 0.33 | 0.12 | 0.89 |
| qwen3 | r4 | 0.57 | 0.34 | 0.26 | 0.32 | 0.17 | 0.75 |
| qwen3 | rand | 1.00 | 0.49 | 0.00 | 0.00 | 0.00 | 0.00 |
| qwen3 | norm_r2 | 0.44 | 0.41 | 0.18 | 0.46 | 0.38 | 0.70 |
| glm4 | r2 | 0.90 | 0.43 | 0.08 | 0.00 | 0.01 | 0.00 |
| glm4 | r4 | 0.93 | 0.49 | 0.06 | 0.00 | 0.01 | 0.00 |
| glm4 | rand | 1.00 | 0.31 | 0.00 | 0.00 | 0.00 | 0.00 |
| glm4 | norm_r2 | 0.74 | 0.42 | 0.10 | 0.43 | 0.17 | 0.25 |
| deepseek7b | r2 | 0.76 | 0.22 | 0.19 | 0.29 | 0.04 | 0.00 |
| deepseek7b | r4 | 0.76 | 0.15 | 0.21 | 0.27 | 0.03 | 0.00 |
| deepseek7b | rand | 0.96 | 0.14 | 0.04 | 0.00 | 0.00 | 0.00 |
| deepseek7b | norm_r2 | 0.29 | 0.10 | 0.24 | 0.24 | 0.47 | 0.21 |

## Route: forbidden_definition:top_p<-forbidden_definition

### Clean Rates

| model | base | intervention | free | bfi | ifb | bfi_transfer | ifb_transfer | first_div |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| qwen3 | 0.28 | r2 | 0.24 | 0.18 | 0.25 | 2.33 | 0.33 | 5.83 |
| qwen3 | 0.28 | r4 | 0.31 | 0.19 | 0.25 | -3.00 | 2.00 | 5.26 |
| qwen3 | 0.28 | rand | 0.32 | 0.22 | 0.15 | -1.33 | 4.00 | 0.24 |
| qwen3 | 0.28 | norm_r2 | 0.28 | 0.18 | 0.25 | 0.00 | 0.00 | 5.79 |
| glm4 | 0.39 | r2 | 0.36 | 0.35 | 0.35 | 1.50 | -0.50 | 2.36 |
| glm4 | 0.39 | r4 | 0.35 | 0.42 | 0.36 | -0.67 | 0.33 | 2.21 |
| glm4 | 0.39 | rand | 0.38 | 0.31 | 0.42 | 6.00 | 3.00 | 0.00 |
| glm4 | 0.39 | norm_r2 | 0.42 | 0.33 | 0.29 | -2.00 | 4.50 | 3.39 |
| deepseek7b | 0.17 | r2 | 0.17 | 0.14 | 0.19 | 0.00 | 0.00 | 4.89 |
| deepseek7b | 0.17 | r4 | 0.19 | 0.15 | 0.12 | -0.50 | 2.50 | 4.04 |
| deepseek7b | 0.17 | rand | 0.14 | 0.18 | 0.19 | -0.50 | 2.00 | 0.61 |
| deepseek7b | 0.17 | norm_r2 | 0.17 | 0.14 | 0.17 | 0.00 | 0.00 | 7.69 |

### Step0 Target-Competitor Margin

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 2.26 | 3.26 | 2.98 | 2.04 | 3.07 |
| glm4 | 1.14 | 4.41 | 3.98 | -1.53 | 3.28 |
| deepseek7b | 0.55 | -0.08 | 0.02 | 0.08 | 0.47 |

### Step1 Target-Competitor Margin

| model | base | r2 | r4 | rand | norm_r2 |
|---|---:|---:|---:|---:|---:|
| qwen3 | 4.89 | 5.11 | 4.95 | 2.21 | 5.24 |
| glm4 | 1.19 | 1.76 | 1.62 | 1.30 | 1.51 |
| deepseek7b | 2.30 | 2.83 | 1.92 | 3.29 | 2.80 |

### Fork Buckets For Free Intervention

| model | intervention | early_rate | early_clean | middle_rate | middle_clean | late_rate | late_clean |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | r2 | 0.36 | 0.27 | 0.18 | 0.23 | 0.46 | 0.21 |
| qwen3 | r4 | 0.40 | 0.28 | 0.17 | 0.33 | 0.43 | 0.32 |
| qwen3 | rand | 0.99 | 0.32 | 0.00 | 0.00 | 0.01 | 0.00 |
| qwen3 | norm_r2 | 0.33 | 0.25 | 0.24 | 0.47 | 0.43 | 0.19 |
| glm4 | r2 | 0.72 | 0.38 | 0.10 | 0.29 | 0.18 | 0.31 |
| glm4 | r4 | 0.72 | 0.29 | 0.11 | 0.38 | 0.17 | 0.58 |
| glm4 | rand | 1.00 | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 |
| glm4 | norm_r2 | 0.62 | 0.38 | 0.12 | 0.44 | 0.25 | 0.50 |
| deepseek7b | r2 | 0.39 | 0.14 | 0.24 | 0.24 | 0.38 | 0.15 |
| deepseek7b | r4 | 0.49 | 0.23 | 0.21 | 0.27 | 0.31 | 0.09 |
| deepseek7b | rand | 0.93 | 0.12 | 0.03 | 0.50 | 0.04 | 0.33 |
| deepseek7b | norm_r2 | 0.18 | 0.00 | 0.15 | 0.36 | 0.67 | 0.17 |

## Timing

| model | minutes | seeds | max tokens |
|---|---:|---|---:|
| qwen3 | 16.63 | [101, 103, 107, 109, 113, 127] | 12 |
| glm4 | 29.73 | [101, 103, 107, 109, 113, 127] | 12 |
| deepseek7b | 22.84 | [101, 103, 107, 109, 113, 127] | 12 |
