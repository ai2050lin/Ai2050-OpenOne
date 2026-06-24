# Phase 612 Cross Model Summary

Source-aligned strict pattern/content split. Prompts are filtered to equal token length.

## qwen3

rows=128, target_seen=9, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 0}, layers=[29], top_k=4, top_heads=[11, 23, 6, 14, 5, 2], time_min=3.02

### all_rows

| mode | switch | margin | correct_delta | wrong_delta | pos_margin | heads |
|---|---:|---:|---:|---:|---:|---|
| `actual` | 123/128 | +0.429 | +0.114 | -0.315 | 113/128 | `[11, 23, 6, 14]` |
| `rr_pattern_content` | 122/128 | +0.441 | +0.114 | -0.327 | 115/128 | `[11, 23, 6, 14]` |
| `rb_pattern` | 122/128 | +0.427 | +0.113 | -0.314 | 116/128 | `[11, 23, 6, 14]` |
| `br_content` | 118/128 | +0.008 | -0.000 | -0.008 | 62/128 | `[11, 23, 6, 14]` |
| `bb` | 118/128 | -0.000 | +0.000 | +0.000 | 20/128 | `[11, 23, 6, 14]` |
| `random_actual_norm` | 118/128 | -0.001 | +0.005 | +0.006 | 62/128 | `[11, 23, 6, 14]` |

### target_rows

| mode | switch | margin | correct_delta | wrong_delta | pos_margin | heads |
|---|---:|---:|---:|---:|---:|---|
| `actual` | 6/9 | +1.557 | +0.944 | -0.614 | 9/9 | `[11, 23, 6, 14]` |
| `rr_pattern_content` | 5/9 | +1.557 | +0.940 | -0.617 | 9/9 | `[11, 23, 6, 14]` |
| `rb_pattern` | 5/9 | +1.529 | +0.924 | -0.605 | 9/9 | `[11, 23, 6, 14]` |
| `br_content` | 1/9 | -0.014 | -0.015 | -0.001 | 2/9 | `[11, 23, 6, 14]` |
| `bb` | 1/9 | +0.014 | +0.006 | -0.007 | 3/9 | `[11, 23, 6, 14]` |
| `random_actual_norm` | 1/9 | +0.105 | +0.096 | -0.009 | 7/9 | `[11, 23, 6, 14]` |

## glm4

rows=128, target_seen=12, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 0}, layers=[34], top_k=4, top_heads=[12, 8, 4, 28, 6, 7], time_min=4.08

### all_rows

| mode | switch | margin | correct_delta | wrong_delta | pos_margin | heads |
|---|---:|---:|---:|---:|---:|---|
| `actual` | 104/128 | +0.003 | +0.003 | +0.000 | 47/128 | `[12, 8, 4, 28]` |
| `rr_pattern_content` | 104/128 | +0.006 | +0.005 | -0.002 | 54/128 | `[12, 8, 4, 28]` |
| `rb_pattern` | 104/128 | +0.009 | +0.004 | -0.005 | 58/128 | `[12, 8, 4, 28]` |
| `br_content` | 103/128 | -0.003 | -0.001 | +0.002 | 49/128 | `[12, 8, 4, 28]` |
| `bb` | 103/128 | +0.000 | +0.000 | -0.000 | 20/128 | `[12, 8, 4, 28]` |
| `random_actual_norm` | 103/128 | +0.001 | -0.001 | -0.001 | 66/128 | `[12, 8, 4, 28]` |

### target_rows

| mode | switch | margin | correct_delta | wrong_delta | pos_margin | heads |
|---|---:|---:|---:|---:|---:|---|
| `actual` | 1/12 | +0.062 | +0.032 | -0.031 | 6/12 | `[12, 8, 4, 28]` |
| `rr_pattern_content` | 1/12 | +0.089 | +0.043 | -0.045 | 8/12 | `[12, 8, 4, 28]` |
| `rb_pattern` | 1/12 | +0.063 | +0.031 | -0.032 | 7/12 | `[12, 8, 4, 28]` |
| `br_content` | 0/12 | -0.021 | -0.010 | +0.011 | 2/12 | `[12, 8, 4, 28]` |
| `bb` | 0/12 | +0.000 | +0.000 | +0.000 | 1/12 | `[12, 8, 4, 28]` |
| `random_actual_norm` | 1/12 | +0.011 | +0.009 | -0.002 | 9/12 | `[12, 8, 4, 28]` |

## deepseek7b

rows=128, target_seen=43, raw=128, filtered={'token_len_mismatch': 0, 'not_target': 0}, layers=[22], top_k=4, top_heads=[3, 1, 7, 24, 25, 13], time_min=3.48

### all_rows

| mode | switch | margin | correct_delta | wrong_delta | pos_margin | heads |
|---|---:|---:|---:|---:|---:|---|
| `actual` | 109/128 | +1.053 | +0.521 | -0.532 | 121/128 | `[3, 1, 7, 24]` |
| `rr_pattern_content` | 109/128 | +1.062 | +0.525 | -0.537 | 121/128 | `[3, 1, 7, 24]` |
| `rb_pattern` | 110/128 | +1.053 | +0.523 | -0.530 | 120/128 | `[3, 1, 7, 24]` |
| `br_content` | 78/128 | -0.006 | -0.010 | -0.004 | 59/128 | `[3, 1, 7, 24]` |
| `bb` | 77/128 | +0.003 | -0.001 | -0.003 | 28/128 | `[3, 1, 7, 24]` |
| `random_actual_norm` | 79/128 | -0.013 | -0.013 | +0.001 | 66/128 | `[3, 1, 7, 24]` |

### target_rows

| mode | switch | margin | correct_delta | wrong_delta | pos_margin | heads |
|---|---:|---:|---:|---:|---:|---|
| `actual` | 31/43 | +1.709 | +1.076 | -0.634 | 43/43 | `[3, 1, 7, 24]` |
| `rr_pattern_content` | 31/43 | +1.708 | +1.074 | -0.634 | 43/43 | `[3, 1, 7, 24]` |
| `rb_pattern` | 32/43 | +1.709 | +1.076 | -0.633 | 43/43 | `[3, 1, 7, 24]` |
| `br_content` | 1/43 | -0.013 | -0.016 | -0.004 | 20/43 | `[3, 1, 7, 24]` |
| `bb` | 0/43 | -0.004 | -0.004 | +0.001 | 7/43 | `[3, 1, 7, 24]` |
| `random_actual_norm` | 1/43 | -0.038 | -0.027 | +0.010 | 19/43 | `[3, 1, 7, 24]` |
