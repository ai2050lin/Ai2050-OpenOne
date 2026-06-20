# Phase 561: Gated Continuous Injection Cross-Model Summary

Surgery: h' = (1-beta)*h_current + beta*donor_cache, applied every step
Beta sweep: 0.05, 0.10, 0.25, 0.50

## Route: sent<-def

### clean_non_object_rate

| model | baseline | remove | repeat2(heli)__b0p05 | repeat2(heli)__b0p1 | repeat2(heli)__b0p25 | repeat2(heli)__b0p5 | repeat4(rocket)__b0p05 | repeat4(rocket)__b0p1 | repeat4(rocket)__b0p25 | repeat4(rocket)__b0p5 | mean_cache__b0p05 | mean_cache__b0p1 | mean_cache__b0p25 | mean_cache__b0p5 | random_cache__b0p05 | random_cache__b0p1 | random_cache__b0p25 | random_cache__b0p5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.48 |   0.61 |   0.55 |   0.53 |   0.47 |   0.01 |   0.60 |   0.50 |   0.51 |   0.00 |   0.61 |   0.50 |   0.51 |   0.00 |   0.58 |   0.57 |   0.67 |   0.04 |
| glm4 |   0.25 |   0.22 |   0.15 |   0.29 |   0.28 |   0.03 |   0.20 |   0.34 |   0.19 |   0.01 |   0.21 |   0.20 |   0.45 |   0.02 |   0.16 |   0.23 |   0.21 |   0.03 |
| deepseek7b |   0.22 |   0.25 |   0.20 |   0.26 |   0.06 |   0.00 |   0.19 |   0.21 |   0.08 |   0.00 |   0.15 |   0.20 |   0.12 |   0.00 |   0.14 |   0.29 |   0.25 |   0.02 |

### steering gain (condition - baseline)

| model | baseline | remove | repeat2(heli)__b0p05 | repeat2(heli)__b0p1 | repeat2(heli)__b0p25 | repeat2(heli)__b0p5 | repeat4(rocket)__b0p05 | repeat4(rocket)__b0p1 | repeat4(rocket)__b0p25 | repeat4(rocket)__b0p5 | mean_cache__b0p05 | mean_cache__b0p1 | mean_cache__b0p25 | mean_cache__b0p5 | random_cache__b0p05 | random_cache__b0p1 | random_cache__b0p25 | random_cache__b0p5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |  +0.00 |  +0.14 |  +0.07 |  +0.05 |  -0.01 |  -0.47 |  +0.12 |  +0.02 |  +0.03 |  -0.48 |  +0.14 |  +0.02 |  +0.03 |  -0.48 |  +0.10 |  +0.09 |  +0.19 |  -0.44 |
| glm4 |  +0.00 |  -0.03 |  -0.10 |  +0.04 |  +0.03 |  -0.22 |  -0.05 |  +0.09 |  -0.06 |  -0.24 |  -0.04 |  -0.05 |  +0.20 |  -0.23 |  -0.09 |  -0.02 |  -0.04 |  -0.22 |
| deepseek7b |  +0.00 |  +0.03 |  -0.02 |  +0.04 |  -0.16 |  -0.22 |  -0.03 |  -0.01 |  -0.14 |  -0.22 |  -0.07 |  -0.02 |  -0.09 |  -0.22 |  -0.08 |  +0.07 |  +0.03 |  -0.20 |

### object_echo_rate

| model | baseline | remove | repeat2(heli)__b0p05 | repeat2(heli)__b0p1 | repeat2(heli)__b0p25 | repeat2(heli)__b0p5 | repeat4(rocket)__b0p05 | repeat4(rocket)__b0p1 | repeat4(rocket)__b0p25 | repeat4(rocket)__b0p5 | mean_cache__b0p05 | mean_cache__b0p1 | mean_cache__b0p25 | mean_cache__b0p5 | random_cache__b0p05 | random_cache__b0p1 | random_cache__b0p25 | random_cache__b0p5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.64 |   0.51 |   0.61 |   0.66 |   0.00 |   0.00 |   0.57 |   0.71 |   0.00 |   0.00 |   0.57 |   0.76 |   0.00 |   0.00 |   0.65 |   0.56 |   0.38 |   0.02 |
| glm4 |   0.76 |   0.72 |   0.58 |   0.66 |   0.02 |   0.00 |   0.70 |   0.59 |   0.08 |   0.00 |   0.72 |   0.64 |   0.35 |   0.00 |   0.70 |   0.66 |   0.60 |   0.01 |
| deepseek7b |   0.50 |   0.61 |   0.68 |   0.80 |   0.23 |   0.00 |   0.71 |   0.80 |   0.62 |   0.00 |   0.72 |   0.80 |   0.44 |   0.00 |   0.64 |   0.54 |   0.38 |   0.21 |

## Route: def<-def

### clean_non_object_rate

| model | baseline | remove | repeat2(heli)__b0p05 | repeat2(heli)__b0p1 | repeat2(heli)__b0p25 | repeat2(heli)__b0p5 | repeat4(rocket)__b0p05 | repeat4(rocket)__b0p1 | repeat4(rocket)__b0p25 | repeat4(rocket)__b0p5 | mean_cache__b0p05 | mean_cache__b0p1 | mean_cache__b0p25 | mean_cache__b0p5 | random_cache__b0p05 | random_cache__b0p1 | random_cache__b0p25 | random_cache__b0p5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.26 |   0.28 |   0.31 |   0.39 |   0.31 |   0.00 |   0.29 |   0.35 |   0.33 |   0.00 |   0.32 |   0.38 |   0.29 |   0.00 |   0.23 |   0.20 |   0.24 |   0.00 |
| glm4 |   0.41 |   0.29 |   0.38 |   0.32 |   0.16 |   0.00 |   0.39 |   0.31 |   0.20 |   0.00 |   0.28 |   0.35 |   0.31 |   0.00 |   0.29 |   0.25 |   0.27 |   0.00 |
| deepseek7b |   0.17 |   0.21 |   0.27 |   0.30 |   0.06 |   0.00 |   0.23 |   0.25 |   0.09 |   0.00 |   0.24 |   0.25 |   0.12 |   0.00 |   0.21 |   0.22 |   0.25 |   0.03 |

### steering gain (condition - baseline)

| model | baseline | remove | repeat2(heli)__b0p05 | repeat2(heli)__b0p1 | repeat2(heli)__b0p25 | repeat2(heli)__b0p5 | repeat4(rocket)__b0p05 | repeat4(rocket)__b0p1 | repeat4(rocket)__b0p25 | repeat4(rocket)__b0p5 | mean_cache__b0p05 | mean_cache__b0p1 | mean_cache__b0p25 | mean_cache__b0p5 | random_cache__b0p05 | random_cache__b0p1 | random_cache__b0p25 | random_cache__b0p5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |  +0.00 |  +0.02 |  +0.05 |  +0.12 |  +0.05 |  -0.26 |  +0.03 |  +0.09 |  +0.07 |  -0.26 |  +0.06 |  +0.11 |  +0.03 |  -0.26 |  -0.03 |  -0.06 |  -0.02 |  -0.26 |
| glm4 |  +0.00 |  -0.11 |  -0.03 |  -0.08 |  -0.25 |  -0.41 |  -0.02 |  -0.09 |  -0.21 |  -0.41 |  -0.12 |  -0.05 |  -0.09 |  -0.41 |  -0.11 |  -0.16 |  -0.14 |  -0.41 |
| deepseek7b |  +0.00 |  +0.04 |  +0.10 |  +0.14 |  -0.10 |  -0.17 |  +0.06 |  +0.08 |  -0.07 |  -0.17 |  +0.07 |  +0.08 |  -0.04 |  -0.17 |  +0.04 |  +0.05 |  +0.08 |  -0.14 |

### object_echo_rate

| model | baseline | remove | repeat2(heli)__b0p05 | repeat2(heli)__b0p1 | repeat2(heli)__b0p25 | repeat2(heli)__b0p5 | repeat4(rocket)__b0p05 | repeat4(rocket)__b0p1 | repeat4(rocket)__b0p25 | repeat4(rocket)__b0p5 | mean_cache__b0p05 | mean_cache__b0p1 | mean_cache__b0p25 | mean_cache__b0p5 | random_cache__b0p05 | random_cache__b0p1 | random_cache__b0p25 | random_cache__b0p5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 |   0.38 |   0.34 |   0.28 |   0.19 |   0.00 |   0.00 |   0.28 |   0.19 |   0.00 |   0.00 |   0.27 |   0.22 |   0.00 |   0.00 |   0.29 |   0.29 |   0.25 |   0.01 |
| glm4 |   0.08 |   0.03 |   0.06 |   0.05 |   0.01 |   0.00 |   0.05 |   0.02 |   0.00 |   0.00 |   0.05 |   0.04 |   0.00 |   0.00 |   0.06 |   0.10 |   0.11 |   0.00 |
| deepseek7b |   0.22 |   0.24 |   0.20 |   0.28 |   0.01 |   0.00 |   0.25 |   0.34 |   0.10 |   0.00 |   0.24 |   0.31 |   0.07 |   0.00 |   0.20 |   0.33 |   0.46 |   0.50 |

## Timing & Metadata

| model | time (min) | seeds | max tokens |
|---|---|---|---|
| qwen3 | 38.84 | [101, 103, 107, 109, 113, 127, 131, 137] | 12 |
| glm4 | 68.83 | [101, 103, 107, 109, 113, 127, 131, 137] | 12 |
| deepseek7b | 52.71 | [101, 103, 107, 109, 113, 127, 131, 137] | 12 |

## Key Findings Summary

### Low-beta window (beta=0.05):

**qwen3** (sent<-def):
  - repeat2(heli): clean_no=0.55 (gain +0.07)
  - repeat4(rocket): clean_no=0.60 (gain +0.12)
  - mean_cache: clean_no=0.61 (gain +0.14)
  - random_cache: clean_no=0.58 (gain +0.10)

**glm4** (sent<-def):
  - repeat2(heli): clean_no=0.15 (gain -0.10)
  - repeat4(rocket): clean_no=0.20 (gain -0.05)
  - mean_cache: clean_no=0.21 (gain -0.04)
  - random_cache: clean_no=0.16 (gain -0.09)

**deepseek7b** (sent<-def):
  - repeat2(heli): clean_no=0.20 (gain -0.02)
  - repeat4(rocket): clean_no=0.19 (gain -0.03)
  - mean_cache: clean_no=0.15 (gain -0.07)
  - random_cache: clean_no=0.14 (gain -0.08)
