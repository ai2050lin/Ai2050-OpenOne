# Phase 729 Full-Head vs Channel-Cluster Residual Propagation

- Status: `complete`
- Models: `['qwen3', 'glm4', 'deepseek7b']`
- Evidence type: residual/component propagation.

| model | intervention | max layer amp | mean MLP/input | mean attn/input | top site | top delta |
|---|---|---:|---:|---:|---|---:|
| qwen3 | category_cluster | 7.112 | 0.536 | 0.461 | hidden_33 | 5.891 |
| qwen3 | category_full_head | 2.652 | 0.517 | 0.407 | hidden_33 | 27.987 |
| qwen3 | category_plus_fruit_cluster | 8.020 | 0.605 | 0.715 | hidden_33 | 8.080 |
| glm4 | category_cluster | 10.784 | 0.465 | 0.389 | hidden_40 | 2.599 |
| glm4 | category_full_head | 3.978 | 0.405 | 0.343 | hidden_40 | 5.406 |
| glm4 | category_plus_fruit_cluster | 12.441 | 0.465 | 0.389 | hidden_40 | 3.392 |
| deepseek7b | category_cluster | 4.105 | 0.683 | 0.329 | hidden_27 | 31.437 |
| deepseek7b | category_full_head | 3.042 | 0.626 | 0.255 | hidden_27 | 97.070 |
| deepseek7b | category_plus_fruit_cluster | 4.105 | 0.683 | 0.329 | hidden_27 | 31.437 |

## Strict Interpretation

- This is a propagation measurement, not a generation closure test.
- A large full-head residual trajectory with weak cluster trajectory supports the Phase 727 boundary.
- MLP/input and attention/input ratios are diagnostic ratios, not a proof of module-level causality.
