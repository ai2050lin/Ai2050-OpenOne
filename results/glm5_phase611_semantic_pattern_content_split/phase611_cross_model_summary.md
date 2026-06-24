# Phase611 Cross-Model Summary

Semantic source-group pattern/content split.

## qwen3

cases=96, rows=7, target_cases_seen=7, layers=[29], top_k=4, top_heads=[11, 23, 6, 14, 5, 2], time_min=0.44

### Best Patches

| key | layer | mode | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L29|top4|actual` | L29 | actual | [11, 23, 6, 14] | 7/7 | 2.251 | 1.301 | -0.950 |
| `L29|top4|content` | L29 | content | [11, 23, 6, 14] | 1/7 | 0.161 | 0.088 | -0.073 |
| `L29|top4|pattern_content` | L29 | pattern_content | [11, 23, 6, 14] | 1/7 | 0.143 | 0.068 | -0.075 |
| `L29|top4|random` | L29 | random | [11, 23, 6, 14] | 1/7 | 0.141 | 0.088 | -0.053 |
| `L29|top4|pattern` | L29 | pattern | [11, 23, 6, 14] | 0/7 | 0.089 | 0.063 | -0.026 |

### Mode Grid

| key | layer | mode | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L29|top4|actual` | L29 | actual | [11, 23, 6, 14] | 7/7 | 2.251 | 1.301 | -0.950 |
| `L29|top4|content` | L29 | content | [11, 23, 6, 14] | 1/7 | 0.161 | 0.088 | -0.073 |
| `L29|top4|pattern` | L29 | pattern | [11, 23, 6, 14] | 0/7 | 0.089 | 0.063 | -0.026 |
| `L29|top4|pattern_content` | L29 | pattern_content | [11, 23, 6, 14] | 1/7 | 0.143 | 0.068 | -0.075 |
| `L29|top4|random` | L29 | random | [11, 23, 6, 14] | 1/7 | 0.141 | 0.088 | -0.053 |

## glm4

cases=96, rows=13, target_cases_seen=13, layers=[34], top_k=4, top_heads=[12, 8, 4, 28, 6, 7], time_min=0.79

### Best Patches

| key | layer | mode | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L34|top4|pattern_content` | L34 | pattern_content | [12, 8, 4, 28] | 10/13 | 1.202 | 0.424 | -0.777 |
| `L34|top4|content` | L34 | content | [12, 8, 4, 28] | 10/13 | 1.192 | 0.424 | -0.769 |
| `L34|top4|actual` | L34 | actual | [12, 8, 4, 28] | 3/13 | 0.279 | 0.134 | -0.145 |
| `L34|top4|pattern` | L34 | pattern | [12, 8, 4, 28] | 1/13 | 0.053 | 0.021 | -0.032 |
| `L34|top4|random` | L34 | random | [12, 8, 4, 28] | 0/13 | -0.028 | -0.031 | -0.003 |

### Mode Grid

| key | layer | mode | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L34|top4|actual` | L34 | actual | [12, 8, 4, 28] | 3/13 | 0.279 | 0.134 | -0.145 |
| `L34|top4|content` | L34 | content | [12, 8, 4, 28] | 10/13 | 1.192 | 0.424 | -0.769 |
| `L34|top4|pattern` | L34 | pattern | [12, 8, 4, 28] | 1/13 | 0.053 | 0.021 | -0.032 |
| `L34|top4|pattern_content` | L34 | pattern_content | [12, 8, 4, 28] | 10/13 | 1.202 | 0.424 | -0.777 |
| `L34|top4|random` | L34 | random | [12, 8, 4, 28] | 0/13 | -0.028 | -0.031 | -0.003 |

## deepseek7b

cases=96, rows=37, target_cases_seen=37, layers=[22], top_k=4, top_heads=[3, 1, 7, 24, 25, 13], time_min=1.19

### Best Patches

| key | layer | mode | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L22|top4|actual` | L22 | actual | [3, 1, 7, 24] | 32/37 | 2.932 | 1.478 | -1.453 |
| `L22|top4|pattern_content` | L22 | pattern_content | [3, 1, 7, 24] | 17/37 | 1.215 | 0.238 | -0.977 |
| `L22|top4|content` | L22 | content | [3, 1, 7, 24] | 9/37 | 0.705 | -0.208 | -0.913 |
| `L22|top4|pattern` | L22 | pattern | [3, 1, 7, 24] | 5/37 | 0.480 | 0.391 | -0.089 |
| `L22|top4|random` | L22 | random | [3, 1, 7, 24] | 0/37 | -0.056 | -0.086 | -0.030 |

### Mode Grid

| key | layer | mode | heads | switch | margin_gain | correct_delta | old_wrong_delta |
|---|---:|---|---|---:|---:|---:|---:|
| `L22|top4|actual` | L22 | actual | [3, 1, 7, 24] | 32/37 | 2.932 | 1.478 | -1.453 |
| `L22|top4|content` | L22 | content | [3, 1, 7, 24] | 9/37 | 0.705 | -0.208 | -0.913 |
| `L22|top4|pattern` | L22 | pattern | [3, 1, 7, 24] | 5/37 | 0.480 | 0.391 | -0.089 |
| `L22|top4|pattern_content` | L22 | pattern_content | [3, 1, 7, 24] | 17/37 | 1.215 | 0.238 | -0.977 |
| `L22|top4|random` | L22 | random | [3, 1, 7, 24] | 0/37 | -0.056 | -0.086 | -0.030 |

