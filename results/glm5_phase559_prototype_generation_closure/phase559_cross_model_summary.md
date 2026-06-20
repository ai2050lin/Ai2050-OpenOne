# Phase 559: Prototype Generation Closure Cross-Model Summary

## Object Audit (test_n=12, vehicle category)

| repeat_idx | object | tokens | chars |
|---|---|---|---|
| repeat0 | tram | 1 | 4 |
| repeat1 | subway | 2 | 6 |
| repeat2 | helicopter | 2 | 10 |
| repeat3 | tractor | 1 | 7 |
| repeat4 | rocket | 1 | 6 |
| repeat5 | canoe | 2 | 5 |
| repeat6 | ferry | 2 | 5 |
| repeat7 | jeep | 2 | 4 |
| repeat8 | ambulance | 2 | 9 |
| repeat9 | cart | 1 | 4 |
| repeat10 | sled | 1 | 4 |
| repeat11 | wagon | 1 | 5 |

**Key:** repeat2 = helicopter (2 tokens), repeat4 = rocket (1 token)

## Route: sent_comp<-def

Surgery mode: one_shot_step0_then_free_kv_cache

### clean_non_object_rate

| model | baseline | add_perp | remove_perp | same | shuffle | repeat2(heli) | repeat4(rocket) | mean_cache | pca1_cache | random_cache | tool_same |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.57 |   0.60 |   0.62 |   0.46 |   0.42 |   0.47 |   0.39 |   0.51 |   0.47 |   0.49 |   0.40 |
| glm4 |   0.26 |   0.35 |   0.25 |   0.25 |   0.22 |   0.39 |   0.42 |   0.25 |   0.29 |   0.32 |   0.26 |
| deepseek7b |   0.19 |   0.17 |   0.19 |   0.25 |   0.17 |   0.19 |   0.15 |   0.14 |   0.26 |   0.15 |   0.25 |

### Necessity Drop (remove_perp - baseline) and Restore Gain (donor - remove_perp)

| model | nec_drop | remove_perp | same_rgain | shuffle_rgain | repeat2(heli)_rgain | repeat4(rocket)_rgain | mean_cache_rgain | pca1_cache_rgain | random_cache_rgain | tool_same_rgain |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 |  +0.06 |  +0.00 |  -0.17 |  -0.21 |  -0.15 |  -0.24 |  -0.11 |  -0.15 |  -0.14 |  -0.22 |
| glm4 |  -0.01 |  +0.00 |  +0.00 |  -0.03 |  +0.14 |  +0.17 |  +0.00 |  +0.04 |  +0.07 |  +0.01 |
| deepseek7b |  +0.00 |  +0.00 |  +0.06 |  -0.03 |  +0.00 |  -0.04 |  -0.06 |  +0.07 |  -0.04 |  +0.06 |

### object_echo_rate

| model | baseline | add_perp | remove_perp | same | shuffle | repeat2(heli) | repeat4(rocket) | mean_cache | pca1_cache | random_cache | tool_same |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.53 |   0.60 |   0.62 |   0.60 |   0.65 |   0.61 |   0.75 |   0.61 |   0.62 |   0.69 |   0.69 |
| glm4 |   0.74 |   0.74 |   0.72 |   0.65 |   0.75 |   0.69 |   0.51 |   0.68 |   0.71 |   0.54 |   0.71 |
| deepseek7b |   0.51 |   0.57 |   0.60 |   0.61 |   0.69 |   0.72 |   0.75 |   0.65 |   0.64 |   0.50 |   0.61 |

### clean_non_object_score

| model | baseline | add_perp | remove_perp | same | shuffle | repeat2(heli) | repeat4(rocket) | mean_cache | pca1_cache | random_cache | tool_same |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.53 |   0.54 |   0.57 |   0.44 |   0.39 |   0.46 |   0.36 |   0.49 |   0.44 |   0.43 |   0.31 |
| glm4 |   0.24 |   0.33 |   0.25 |   0.21 |   0.22 |   0.35 |   0.35 |   0.22 |   0.26 |   0.26 |   0.21 |
| deepseek7b |   0.17 |   0.12 |   0.17 |   0.21 |   0.14 |   0.08 |   0.12 |   0.10 |   0.19 |  -0.01 |   0.17 |

## Route: def<-def

Surgery mode: one_shot_step0_then_free_kv_cache

### clean_non_object_rate

| model | baseline | add_perp | remove_perp | same | shuffle | repeat2(heli) | repeat4(rocket) | mean_cache | pca1_cache | random_cache | tool_same |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.18 |   0.24 |   0.22 |   0.22 |   0.22 |   0.26 |   0.22 |   0.26 |   0.24 |   0.28 |   0.25 |
| glm4 |   0.38 |   0.43 |   0.39 |   0.42 |   0.43 |   0.46 |   0.46 |   0.35 |   0.46 |   0.35 |   0.43 |
| deepseek7b |   0.15 |   0.21 |   0.25 |   0.17 |   0.21 |   0.18 |   0.11 |   0.17 |   0.24 |   0.14 |   0.10 |

### Necessity Drop (remove_perp - baseline) and Restore Gain (donor - remove_perp)

| model | nec_drop | remove_perp | same_rgain | shuffle_rgain | repeat2(heli)_rgain | repeat4(rocket)_rgain | mean_cache_rgain | pca1_cache_rgain | random_cache_rgain | tool_same_rgain |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 |  +0.04 |  +0.00 |  +0.00 |  +0.00 |  +0.04 |  +0.00 |  +0.04 |  +0.01 |  +0.06 |  +0.03 |
| glm4 |  +0.01 |  +0.00 |  +0.03 |  +0.04 |  +0.07 |  +0.07 |  -0.04 |  +0.07 |  -0.04 |  +0.04 |
| deepseek7b |  +0.10 |  +0.00 |  -0.08 |  -0.04 |  -0.07 |  -0.14 |  -0.08 |  -0.01 |  -0.11 |  -0.15 |

### object_echo_rate

| model | baseline | add_perp | remove_perp | same | shuffle | repeat2(heli) | repeat4(rocket) | mean_cache | pca1_cache | random_cache | tool_same |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.29 |   0.31 |   0.31 |   0.32 |   0.35 |   0.29 |   0.39 |   0.28 |   0.33 |   0.28 |   0.29 |
| glm4 |   0.07 |   0.10 |   0.04 |   0.11 |   0.06 |   0.06 |   0.10 |   0.01 |   0.07 |   0.18 |   0.10 |
| deepseek7b |   0.24 |   0.24 |   0.22 |   0.26 |   0.15 |   0.22 |   0.25 |   0.18 |   0.28 |   0.25 |   0.14 |

### clean_non_object_score

| model | baseline | add_perp | remove_perp | same | shuffle | repeat2(heli) | repeat4(rocket) | mean_cache | pca1_cache | random_cache | tool_same |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.08 |   0.17 |   0.18 |   0.17 |   0.12 |   0.22 |   0.12 |   0.19 |   0.17 |   0.17 |   0.14 |
| glm4 |   0.26 |   0.28 |   0.26 |   0.29 |   0.28 |   0.31 |   0.38 |   0.15 |   0.32 |   0.26 |   0.31 |
| deepseek7b |  -0.10 |  -0.03 |   0.06 |  -0.04 |  -0.03 |  -0.08 |  -0.08 |  -0.08 |   0.04 |  -0.10 |  -0.15 |

## Phase 558 (next-token margin) vs Phase 559 (generation closure) — GLM4 all L24/26/28

| condition | P558 margin (sent) | P559 clean_no (sent) | P558 margin (def) | P559 clean_no (def) |
|---|---|---|---|---|
| baseline |  -1.28 |   0.26 |  +0.29 |   0.38 |
| add_perp |  +0.92 |   0.35 |  +3.16 |   0.43 |
| remove_perp |  -1.59 |   0.25 |  -0.38 |   0.39 |
| same |  +2.95 |   0.25 |  +3.16 |   0.42 |
| shuffle |  +2.99 |   0.22 |  +3.28 |   0.43 |
| repeat2(heli) |  +3.64 |   0.39 |  +4.38 |   0.46 |
| repeat4(rocket) |  +3.61 |   0.42 |  +3.92 |   0.46 |
| mean_cache |  +4.06 |   0.25 |  +4.37 |   0.35 |
| pca1_cache |  +3.15 |   0.29 |  +3.45 |   0.46 |
| random_cache |  -1.44 |   0.32 |  -1.44 |   0.35 |
| tool_same |  +2.74 |   0.26 |  +3.09 |   0.43 |

## Timing

| model | time (min) | attn |
|---|---|---|
| qwen3 | 1.33 | sdpa |
| glm4 | 24.44 | sdpa |
| deepseek7b | 10.5 | sdpa |
