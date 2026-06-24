# Phase610 Cross-Model Summary

Cumulative head-slot mixture audit.

## qwen3

cases=96, rows=7, target_cases_seen=7, layers=[29], heads={'29': 32}, top_heads=[11, 23, 6, 14, 5, 2], time_min=0.56

### Best Patches

| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---|---:|---:|---:|---:|
| `L29|top6_delta` | L29 | top6_delta | top_delta | [11, 23, 6, 14, 5, 2] | 7/7 | 2.305 | 1.317 | -0.988 |
| `L29|top3_delta` | L29 | top3_delta | top_delta | [11, 23, 6] | 7/7 | 2.269 | 1.306 | -0.963 |
| `L29|top4_delta` | L29 | top4_delta | top_delta | [11, 23, 6, 14] | 7/7 | 2.251 | 1.301 | -0.950 |
| `L29|all_delta` | L29 | all_delta | all_delta | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 6/7 | 2.055 | 1.173 | -0.882 |
| `L29|top1_delta` | L29 | top1_delta | top_delta | [11] | 5/7 | 1.894 | 1.149 | -0.745 |
| `L29|top2_delta` | L29 | top2_delta | top_delta | [11, 23] | 5/7 | 1.876 | 1.138 | -0.738 |
| `L29|top6_random_slots` | L29 | top6_random_slots | top_random_slots | [11, 23, 6, 14, 5, 2] | 2/7 | 0.120 | 0.064 | -0.056 |
| `L29|top3_random_slots` | L29 | top3_random_slots | top_random_slots | [11, 23, 6] | 2/7 | 0.078 | 0.021 | -0.057 |
| `L29|top4_random_slots` | L29 | top4_random_slots | top_random_slots | [11, 23, 6, 14] | 2/7 | 0.077 | 0.021 | -0.056 |
| `L29|top1_random_slots` | L29 | top1_random_slots | top_random_slots | [11] | 2/7 | 0.018 | -0.021 | -0.039 |
| `L29|top2_random_slots` | L29 | top2_random_slots | top_random_slots | [11, 23] | 1/7 | 0.039 | 0.006 | -0.034 |
| `L29|randheads1_delta` | L29 | randheads1_delta | random_heads_delta | [31] | 1/7 | 0.036 | 0.020 | -0.016 |
| `L29|randheads6_delta` | L29 | randheads6_delta | random_heads_delta | [3, 13, 21, 24, 25, 29] | 1/7 | 0.018 | -0.000 | -0.018 |
| `L29|randheads2_delta` | L29 | randheads2_delta | random_heads_delta | [13, 27] | 0/7 | 0.304 | 0.240 | -0.064 |
| `L29|randheads4_delta` | L29 | randheads4_delta | random_heads_delta | [1, 7, 9, 25] | 0/7 | -0.018 | -0.023 | -0.006 |
| `L29|randheads3_delta` | L29 | randheads3_delta | random_heads_delta | [9, 16, 26] | 0/7 | -0.071 | -0.071 | 0.001 |
| `L29|all_random_slots` | L29 | all_random_slots | all_random_slots | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 0/7 | -0.098 | -0.045 | 0.053 |
| `L29|weak1_delta` | L29 | weak1_delta | weak_delta | [9] | 0/7 | -0.179 | -0.145 | 0.033 |
| `L29|weak2_delta` | L29 | weak2_delta | weak_delta | [9, 0] | 0/7 | -0.197 | -0.162 | 0.034 |
| `L29|weak3_delta` | L29 | weak3_delta | weak_delta | [9, 0, 19] | 0/7 | -0.232 | -0.183 | 0.050 |
| `L29|weak4_delta` | L29 | weak4_delta | weak_delta | [9, 0, 19, 28] | 0/7 | -0.250 | -0.200 | 0.050 |
| `L29|weak6_delta` | L29 | weak6_delta | weak_delta | [9, 0, 19, 28, 15, 8] | 0/7 | -0.286 | -0.226 | 0.060 |

### Top Cumulative Curve

| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---|---:|---:|---:|---:|
| `L29|top1_delta` | L29 | top1_delta | top_delta | [11] | 5/7 | 1.894 | 1.149 | -0.745 |
| `L29|top2_delta` | L29 | top2_delta | top_delta | [11, 23] | 5/7 | 1.876 | 1.138 | -0.738 |
| `L29|top3_delta` | L29 | top3_delta | top_delta | [11, 23, 6] | 7/7 | 2.269 | 1.306 | -0.963 |
| `L29|top4_delta` | L29 | top4_delta | top_delta | [11, 23, 6, 14] | 7/7 | 2.251 | 1.301 | -0.950 |
| `L29|top6_delta` | L29 | top6_delta | top_delta | [11, 23, 6, 14, 5, 2] | 7/7 | 2.305 | 1.317 | -0.988 |
| `L29|all_delta` | L29 | all_delta | all_delta | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 6/7 | 2.055 | 1.173 | -0.882 |

### Controls

| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---|---:|---:|---:|---:|
| `L29|top1_random_slots` | L29 | top1_random_slots | top_random_slots | [11] | 2/7 | 0.018 | -0.021 | -0.039 |
| `L29|top2_random_slots` | L29 | top2_random_slots | top_random_slots | [11, 23] | 1/7 | 0.039 | 0.006 | -0.034 |
| `L29|top3_random_slots` | L29 | top3_random_slots | top_random_slots | [11, 23, 6] | 2/7 | 0.078 | 0.021 | -0.057 |
| `L29|top4_random_slots` | L29 | top4_random_slots | top_random_slots | [11, 23, 6, 14] | 2/7 | 0.077 | 0.021 | -0.056 |
| `L29|top6_random_slots` | L29 | top6_random_slots | top_random_slots | [11, 23, 6, 14, 5, 2] | 2/7 | 0.120 | 0.064 | -0.056 |
| `L29|weak1_delta` | L29 | weak1_delta | weak_delta | [9] | 0/7 | -0.179 | -0.145 | 0.033 |
| `L29|weak2_delta` | L29 | weak2_delta | weak_delta | [9, 0] | 0/7 | -0.197 | -0.162 | 0.034 |
| `L29|weak3_delta` | L29 | weak3_delta | weak_delta | [9, 0, 19] | 0/7 | -0.232 | -0.183 | 0.050 |
| `L29|weak4_delta` | L29 | weak4_delta | weak_delta | [9, 0, 19, 28] | 0/7 | -0.250 | -0.200 | 0.050 |
| `L29|weak6_delta` | L29 | weak6_delta | weak_delta | [9, 0, 19, 28, 15, 8] | 0/7 | -0.286 | -0.226 | 0.060 |
| `L29|all_random_slots` | L29 | all_random_slots | all_random_slots | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 0/7 | -0.098 | -0.045 | 0.053 |

## glm4

cases=96, rows=13, target_cases_seen=13, layers=[34], heads={'34': 32}, top_heads=[12, 8, 4, 28, 6, 7], time_min=1.20

### Best Patches

| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---|---:|---:|---:|---:|
| `L34|top6_delta` | L34 | top6_delta | top_delta | [12, 8, 4, 28, 6, 7] | 3/13 | 0.308 | 0.150 | -0.158 |
| `L34|top3_delta` | L34 | top3_delta | top_delta | [12, 8, 4] | 3/13 | 0.288 | 0.140 | -0.149 |
| `L34|top4_delta` | L34 | top4_delta | top_delta | [12, 8, 4, 28] | 3/13 | 0.279 | 0.134 | -0.145 |
| `L34|all_delta` | L34 | all_delta | all_delta | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 3/13 | 0.173 | 0.090 | -0.083 |
| `L34|top2_delta` | L34 | top2_delta | top_delta | [12, 8] | 2/13 | 0.202 | 0.097 | -0.105 |
| `L34|top1_delta` | L34 | top1_delta | top_delta | [12] | 1/13 | 0.125 | 0.066 | -0.059 |
| `L34|randheads6_delta` | L34 | randheads6_delta | random_heads_delta | [2, 5, 13, 16, 18, 19] | 1/13 | 0.087 | 0.044 | -0.043 |
| `L34|randheads3_delta` | L34 | randheads3_delta | random_heads_delta | [8, 24, 25] | 1/13 | 0.053 | 0.032 | -0.021 |
| `L34|all_random_slots` | L34 | all_random_slots | all_random_slots | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 1/13 | 0.018 | 0.018 | 0.000 |
| `L34|randheads2_delta` | L34 | randheads2_delta | random_heads_delta | [13, 24] | 1/13 | 0.014 | 0.006 | -0.008 |
| `L34|top4_random_slots` | L34 | top4_random_slots | top_random_slots | [12, 8, 4, 28] | 1/13 | -0.017 | -0.003 | 0.014 |
| `L34|top6_random_slots` | L34 | top6_random_slots | top_random_slots | [12, 8, 4, 28, 6, 7] | 1/13 | -0.017 | -0.005 | 0.012 |
| `L34|top3_random_slots` | L34 | top3_random_slots | top_random_slots | [12, 8, 4] | 1/13 | -0.017 | -0.003 | 0.014 |
| `L34|randheads4_delta` | L34 | randheads4_delta | random_heads_delta | [5, 11, 12, 15] | 0/13 | 0.034 | 0.027 | -0.006 |
| `L34|top2_random_slots` | L34 | top2_random_slots | top_random_slots | [12, 8] | 0/13 | 0.004 | 0.007 | 0.003 |
| `L34|randheads1_delta` | L34 | randheads1_delta | random_heads_delta | [28] | 0/13 | -0.019 | -0.008 | 0.011 |
| `L34|top1_random_slots` | L34 | top1_random_slots | top_random_slots | [12] | 0/13 | -0.023 | -0.016 | 0.008 |
| `L34|weak1_delta` | L34 | weak1_delta | weak_delta | [2] | 0/13 | -0.192 | -0.129 | 0.063 |
| `L34|weak2_delta` | L34 | weak2_delta | weak_delta | [2, 26] | 0/13 | -0.216 | -0.145 | 0.072 |
| `L34|weak4_delta` | L34 | weak4_delta | weak_delta | [2, 26, 3, 25] | 0/13 | -0.221 | -0.151 | 0.070 |
| `L34|weak3_delta` | L34 | weak3_delta | weak_delta | [2, 26, 3] | 0/13 | -0.221 | -0.149 | 0.073 |
| `L34|weak6_delta` | L34 | weak6_delta | weak_delta | [2, 26, 3, 25, 23, 9] | 0/13 | -0.226 | -0.152 | 0.073 |

### Top Cumulative Curve

| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---|---:|---:|---:|---:|
| `L34|top1_delta` | L34 | top1_delta | top_delta | [12] | 1/13 | 0.125 | 0.066 | -0.059 |
| `L34|top2_delta` | L34 | top2_delta | top_delta | [12, 8] | 2/13 | 0.202 | 0.097 | -0.105 |
| `L34|top3_delta` | L34 | top3_delta | top_delta | [12, 8, 4] | 3/13 | 0.288 | 0.140 | -0.149 |
| `L34|top4_delta` | L34 | top4_delta | top_delta | [12, 8, 4, 28] | 3/13 | 0.279 | 0.134 | -0.145 |
| `L34|top6_delta` | L34 | top6_delta | top_delta | [12, 8, 4, 28, 6, 7] | 3/13 | 0.308 | 0.150 | -0.158 |
| `L34|all_delta` | L34 | all_delta | all_delta | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 3/13 | 0.173 | 0.090 | -0.083 |

### Controls

| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---|---:|---:|---:|---:|
| `L34|top1_random_slots` | L34 | top1_random_slots | top_random_slots | [12] | 0/13 | -0.023 | -0.016 | 0.008 |
| `L34|top2_random_slots` | L34 | top2_random_slots | top_random_slots | [12, 8] | 0/13 | 0.004 | 0.007 | 0.003 |
| `L34|top3_random_slots` | L34 | top3_random_slots | top_random_slots | [12, 8, 4] | 1/13 | -0.017 | -0.003 | 0.014 |
| `L34|top4_random_slots` | L34 | top4_random_slots | top_random_slots | [12, 8, 4, 28] | 1/13 | -0.017 | -0.003 | 0.014 |
| `L34|top6_random_slots` | L34 | top6_random_slots | top_random_slots | [12, 8, 4, 28, 6, 7] | 1/13 | -0.017 | -0.005 | 0.012 |
| `L34|weak1_delta` | L34 | weak1_delta | weak_delta | [2] | 0/13 | -0.192 | -0.129 | 0.063 |
| `L34|weak2_delta` | L34 | weak2_delta | weak_delta | [2, 26] | 0/13 | -0.216 | -0.145 | 0.072 |
| `L34|weak3_delta` | L34 | weak3_delta | weak_delta | [2, 26, 3] | 0/13 | -0.221 | -0.149 | 0.073 |
| `L34|weak4_delta` | L34 | weak4_delta | weak_delta | [2, 26, 3, 25] | 0/13 | -0.221 | -0.151 | 0.070 |
| `L34|weak6_delta` | L34 | weak6_delta | weak_delta | [2, 26, 3, 25, 23, 9] | 0/13 | -0.226 | -0.152 | 0.073 |
| `L34|all_random_slots` | L34 | all_random_slots | all_random_slots | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] | 1/13 | 0.018 | 0.018 | 0.000 |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, layers=[22], heads={'22': 28}, top_heads=[3, 1, 7, 24, 25, 13], time_min=2.09

### Best Patches

| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---|---:|---:|---:|---:|
| `L22|all_delta` | L22 | all_delta | all_delta | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] | 33/37 | 3.428 | 1.585 | -1.843 |
| `L22|top6_delta` | L22 | top6_delta | top_delta | [3, 1, 7, 24, 25, 13] | 32/37 | 3.085 | 1.554 | -1.531 |
| `L22|top4_delta` | L22 | top4_delta | top_delta | [3, 1, 7, 24] | 32/37 | 2.932 | 1.478 | -1.453 |
| `L22|top3_delta` | L22 | top3_delta | top_delta | [3, 1, 7] | 31/37 | 2.670 | 1.351 | -1.319 |
| `L22|top2_delta` | L22 | top2_delta | top_delta | [3, 1] | 28/37 | 2.223 | 1.131 | -1.092 |
| `L22|top1_delta` | L22 | top1_delta | top_delta | [3] | 16/37 | 1.516 | 0.796 | -0.721 |
| `L22|randheads6_delta` | L22 | randheads6_delta | random_heads_delta | [3, 9, 13, 15, 18, 21] | 13/37 | 0.828 | 0.494 | -0.335 |
| `L22|randheads4_delta` | L22 | randheads4_delta | random_heads_delta | [11, 14, 18, 20] | 7/37 | 0.646 | 0.310 | -0.336 |
| `L22|randheads3_delta` | L22 | randheads3_delta | random_heads_delta | [3, 11, 12] | 5/37 | 0.393 | 0.298 | -0.095 |
| `L22|randheads1_delta` | L22 | randheads1_delta | random_heads_delta | [26] | 3/37 | 0.184 | 0.115 | -0.069 |
| `L22|randheads2_delta` | L22 | randheads2_delta | random_heads_delta | [11, 27] | 3/37 | 0.125 | 0.094 | -0.030 |
| `L22|top4_random_slots` | L22 | top4_random_slots | top_random_slots | [3, 1, 7, 24] | 2/37 | 0.086 | 0.024 | -0.062 |
| `L22|top3_random_slots` | L22 | top3_random_slots | top_random_slots | [3, 1, 7] | 1/37 | 0.086 | 0.027 | -0.060 |
| `L22|top6_random_slots` | L22 | top6_random_slots | top_random_slots | [3, 1, 7, 24, 25, 13] | 1/37 | 0.074 | -0.010 | -0.084 |
| `L22|all_random_slots` | L22 | all_random_slots | all_random_slots | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] | 1/37 | -0.154 | -0.176 | -0.022 |
| `L22|top2_random_slots` | L22 | top2_random_slots | top_random_slots | [3, 1] | 0/37 | 0.086 | 0.026 | -0.060 |
| `L22|top1_random_slots` | L22 | top1_random_slots | top_random_slots | [3] | 0/37 | 0.049 | 0.010 | -0.040 |
| `L22|weak2_delta` | L22 | weak2_delta | weak_delta | [19, 18] | 0/37 | -0.005 | -0.011 | -0.006 |
| `L22|weak6_delta` | L22 | weak6_delta | weak_delta | [19, 18, 26, 8, 6, 14] | 0/37 | -0.005 | -0.017 | -0.012 |
| `L22|weak3_delta` | L22 | weak3_delta | weak_delta | [19, 18, 26] | 0/37 | -0.005 | -0.010 | -0.004 |
| `L22|weak1_delta` | L22 | weak1_delta | weak_delta | [19] | 0/37 | -0.014 | -0.014 | -0.000 |
| `L22|weak4_delta` | L22 | weak4_delta | weak_delta | [19, 18, 26, 8] | 0/37 | -0.023 | -0.027 | -0.004 |

### Top Cumulative Curve

| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---|---:|---:|---:|---:|
| `L22|top1_delta` | L22 | top1_delta | top_delta | [3] | 16/37 | 1.516 | 0.796 | -0.721 |
| `L22|top2_delta` | L22 | top2_delta | top_delta | [3, 1] | 28/37 | 2.223 | 1.131 | -1.092 |
| `L22|top3_delta` | L22 | top3_delta | top_delta | [3, 1, 7] | 31/37 | 2.670 | 1.351 | -1.319 |
| `L22|top4_delta` | L22 | top4_delta | top_delta | [3, 1, 7, 24] | 32/37 | 2.932 | 1.478 | -1.453 |
| `L22|top6_delta` | L22 | top6_delta | top_delta | [3, 1, 7, 24, 25, 13] | 32/37 | 3.085 | 1.554 | -1.531 |
| `L22|all_delta` | L22 | all_delta | all_delta | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] | 33/37 | 3.428 | 1.585 | -1.843 |

### Controls

| key | layer | name | kind | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---|---:|---:|---:|---:|
| `L22|top1_random_slots` | L22 | top1_random_slots | top_random_slots | [3] | 0/37 | 0.049 | 0.010 | -0.040 |
| `L22|top2_random_slots` | L22 | top2_random_slots | top_random_slots | [3, 1] | 0/37 | 0.086 | 0.026 | -0.060 |
| `L22|top3_random_slots` | L22 | top3_random_slots | top_random_slots | [3, 1, 7] | 1/37 | 0.086 | 0.027 | -0.060 |
| `L22|top4_random_slots` | L22 | top4_random_slots | top_random_slots | [3, 1, 7, 24] | 2/37 | 0.086 | 0.024 | -0.062 |
| `L22|top6_random_slots` | L22 | top6_random_slots | top_random_slots | [3, 1, 7, 24, 25, 13] | 1/37 | 0.074 | -0.010 | -0.084 |
| `L22|weak1_delta` | L22 | weak1_delta | weak_delta | [19] | 0/37 | -0.014 | -0.014 | -0.000 |
| `L22|weak2_delta` | L22 | weak2_delta | weak_delta | [19, 18] | 0/37 | -0.005 | -0.011 | -0.006 |
| `L22|weak3_delta` | L22 | weak3_delta | weak_delta | [19, 18, 26] | 0/37 | -0.005 | -0.010 | -0.004 |
| `L22|weak4_delta` | L22 | weak4_delta | weak_delta | [19, 18, 26, 8] | 0/37 | -0.023 | -0.027 | -0.004 |
| `L22|weak6_delta` | L22 | weak6_delta | weak_delta | [19, 18, 26, 8, 6, 14] | 0/37 | -0.005 | -0.017 | -0.012 |
| `L22|all_random_slots` | L22 | all_random_slots | all_random_slots | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] | 1/37 | -0.154 | -0.176 | -0.022 |

