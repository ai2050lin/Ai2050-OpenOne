# Phase 684 Late Readout Prose Amplification Causal Audit

- generated: `2026-06-26 12:28:06`

| model | short_failures | best_condition | repair_rate | patched_top1 | mean_rank_delta | patched_pmv | patched_best_other |
|---|---:|---|---:|---:|---:|---:|---|
| deepseek7b | 132 | logit|final_logits|add_value|a2.0 | 0.992 | 0.992 | 301.64 | -2.400 | {'prose': 132} |
| glm4 | 16 | hidden_direction|final_norm_input|add_value_minus_prose|r0.1 | 1.000 | 1.000 | 1.56 | -14.148 | {'continuation': 16} |
| qwen3 | 5 | hidden_direction|final_norm_output|add_value_minus_prose|r0.1 | 1.000 | 1.000 | 7.00 | -17.363 | {'continuation': 5} |

## Best Conditions

### deepseek7b

| condition | repair_rate | patched_top1 | mean_patched_rank | rank_delta | baseline_pmv | patched_pmv |
|---|---:|---:|---:|---:|---:|---:|
| logit|final_logits|add_value|a2.0 | 0.992 | 0.992 | 1.01 | 301.64 | 5.192 | -2.400 |
| logit|final_logits|remove_prose_add_value|a2.0 | 0.992 | 0.992 | 1.01 | 301.64 | 5.192 | -12.785 |
| hidden_direction|final_norm_output|add_value_minus_prose|r0.1 | 0.917 | 0.917 | 1.20 | 301.45 | 5.192 | -9.990 |
| hidden_direction|final_norm_input|add_value_minus_prose|r0.1 | 0.909 | 0.909 | 1.27 | 301.37 | 5.192 | -9.830 |
| logit|final_logits|add_value|a1.0 | 0.667 | 0.667 | 1.44 | 301.20 | 5.192 | -0.038 |
| logit|final_logits|remove_prose_add_value|a1.0 | 0.644 | 0.644 | 1.64 | 301.00 | 5.192 | -5.195 |
| hidden_direction|final_norm_output|add_value_minus_prose|r0.05 | 0.568 | 0.568 | 12.39 | 290.25 | 5.192 | -2.338 |
| hidden_direction|final_norm_input|add_value_minus_prose|r0.05 | 0.568 | 0.568 | 12.54 | 290.11 | 5.192 | -2.295 |
| hidden_direction|final_norm_output|add_value_minus_prose|r0.02 | 0.273 | 0.273 | 97.55 | 205.10 | 5.192 | 2.231 |
| hidden_direction|final_norm_input|add_value_minus_prose|r0.02 | 0.258 | 0.258 | 97.04 | 205.61 | 5.192 | 2.256 |

### glm4

| condition | repair_rate | patched_top1 | mean_patched_rank | rank_delta | baseline_pmv | patched_pmv |
|---|---:|---:|---:|---:|---:|---:|
| hidden_direction|final_norm_input|add_value_minus_prose|r0.1 | 1.000 | 1.000 | 1.00 | 1.56 | -1.709 | -14.148 |
| hidden_direction|final_norm_output|add_value_minus_prose|r0.1 | 1.000 | 1.000 | 1.00 | 1.56 | -1.709 | -12.137 |
| hidden_direction|final_norm_input|add_value_minus_prose|r0.05 | 0.938 | 0.938 | 1.06 | 1.50 | -1.709 | -7.951 |
| hidden_direction|final_norm_output|add_value_minus_prose|r0.05 | 0.938 | 0.938 | 1.06 | 1.50 | -1.709 | -6.930 |
| logit|final_logits|add_value|a2.0 | 0.875 | 0.875 | 1.12 | 1.44 | -1.709 | -3.709 |
| logit|final_logits|remove_prose_add_value|a2.0 | 0.875 | 0.875 | 1.12 | 1.44 | -1.709 | -5.709 |
| hidden_direction|final_norm_input|add_value_minus_prose|r0.02 | 0.812 | 0.812 | 1.31 | 1.25 | -1.709 | -4.211 |
| hidden_direction|final_norm_output|add_value_minus_prose|r0.02 | 0.812 | 0.812 | 1.31 | 1.25 | -1.709 | -3.801 |
| logit|final_logits|add_value|a1.0 | 0.688 | 0.688 | 1.31 | 1.25 | -1.709 | -2.709 |
| logit|final_logits|remove_prose_add_value|a1.0 | 0.688 | 0.688 | 1.31 | 1.25 | -1.709 | -3.709 |

### qwen3

| condition | repair_rate | patched_top1 | mean_patched_rank | rank_delta | baseline_pmv | patched_pmv |
|---|---:|---:|---:|---:|---:|---:|
| hidden_direction|final_norm_output|add_value_minus_prose|r0.1 | 1.000 | 1.000 | 1.00 | 7.00 | 0.537 | -17.363 |
| logit|final_logits|add_value|a2.0 | 0.800 | 0.800 | 1.20 | 6.80 | 0.537 | -3.025 |
| logit|final_logits|remove_prose_add_value|a2.0 | 0.800 | 0.800 | 1.20 | 6.80 | 0.537 | -6.963 |
| hidden_direction|final_norm_input|add_value_minus_prose|r0.1 | 0.800 | 0.800 | 1.20 | 6.80 | 0.537 | -13.475 |
| hidden_direction|final_norm_output|add_value_minus_prose|r0.05 | 0.800 | 0.800 | 1.40 | 6.60 | 0.537 | -8.400 |
| hidden_direction|final_norm_input|add_value_minus_prose|r0.05 | 0.800 | 0.800 | 1.80 | 6.20 | 0.537 | -6.525 |
| hidden_direction|final_norm_output|add_value_minus_prose|r0.02 | 0.800 | 0.800 | 2.40 | 5.60 | 0.537 | -3.075 |
| hidden_direction|final_norm_input|add_value_minus_prose|r0.02 | 0.600 | 0.600 | 3.00 | 5.00 | 0.537 | -2.275 |
| logit|final_logits|add_value|a1.0 | 0.400 | 0.400 | 2.00 | 6.00 | 0.537 | -1.425 |
| logit|final_logits|remove_prose_add_value|a1.0 | 0.400 | 0.400 | 2.00 | 6.00 | 0.537 | -3.388 |

