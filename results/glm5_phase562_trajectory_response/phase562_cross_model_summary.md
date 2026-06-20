# Phase 562: Trajectory Response Audit Cross-Model Summary

Surgery: one-shot at step 0, then free KV-cache generation (16 tokens)
Conditions: baseline, one_shot repeat2/4/mean/random, add_tangent/normal

## Route: sent<-def

### clean_non_object_rate

| model | baseline | repeat2(heli) | repeat4(rocket) | mean_cache | random_cache | tangent_r2 | normal_r2 |
|---|---|---|---|---|---|---|---|
| qwen3 |   0.58 |   0.56 |   0.51 |   0.60 |   0.49 |   0.53 |   0.56 |
| glm4 |   0.33 |   0.39 |   0.54 |   0.29 |   0.38 |   0.44 |   0.38 |
| deepseek7b |   0.22 |   0.25 |   0.24 |   0.17 |   0.18 |   0.21 |   0.29 |

### mean_relaxation_length (steps until target token disappears)

| model | baseline | repeat2(heli) | repeat4(rocket) | mean_cache | random_cache | tangent_r2 | normal_r2 |
|---|---|---|---|---|---|---|---|
| qwen3 |   0.97 |   0.60 |   0.92 |   0.57 |   0.15 |   0.68 |   1.10 |
| glm4 |   0.00 |   0.00 |   0.00 |   0.00 |   0.00 |   0.00 |   0.00 |
| deepseek7b |   0.72 |   1.10 |   0.72 |   0.50 |   0.65 |   0.96 |   0.82 |

### semantic_specificity (condition - random_cache)

| model | baseline | repeat2(heli) | repeat4(rocket) | mean_cache | random_cache | tangent_r2 | normal_r2 |
|---|---|---|---|---|---|---|---|
| qwen3 |  +0.10 |  +0.07 |  +0.03 |  +0.11 |  +0.00 |  +0.04 |  +0.07 |
| glm4 |  -0.04 |  +0.01 |  +0.17 |  -0.08 |  +0.00 |  +0.07 |  +0.00 |
| deepseek7b |  +0.04 |  +0.07 |  +0.06 |  -0.01 |  +0.00 |  +0.03 |  +0.11 |

### tangent vs normal (add_tangent vs add_normal)

| model | baseline | tangent_r2 | normal_r2 | tangent-baseline | normal-baseline | tangent-normal |
|---|---|---|---|---|---|---|
| qwen3 | 0.58 | 0.53 | 0.56 | -0.06 | -0.03 | -0.03 |
| glm4 | 0.33 | 0.44 | 0.38 | +0.11 | +0.04 | +0.07 |
| deepseek7b | 0.22 | 0.21 | 0.29 | -0.01 | +0.07 | -0.08 |

### degeneration_distribution (normal / repetition / garbage)

| model | baseline | repeat2(heli) | repeat4(rocket) | mean_cache | random_cache | tangent_r2 | normal_r2 |
|---|---|---|---|---|---|---|---|
| qwen3 | 0.94n/0.06r/0.00g | 0.94n/0.06r/0.00g | 0.96n/0.04r/0.00g | 0.96n/0.04r/0.00g | 1.00n/0.00r/0.00g | 0.97n/0.03r/0.00g | 0.92n/0.08r/0.00g |
| glm4 | 0.94n/0.04r/0.00g | 0.97n/0.03r/0.00g | 0.96n/0.03r/0.01g | 0.97n/0.03r/0.00g | 1.00n/0.00r/0.00g | 0.99n/0.00r/0.00g | 1.00n/0.00r/0.00g |
| deepseek7b | 0.96n/0.03r/0.00g | 0.94n/0.01r/0.01g | 0.94n/0.03r/0.00g | 0.97n/0.01r/0.00g | 0.99n/0.00r/0.00g | 0.97n/0.01r/0.00g | 0.97n/0.01r/0.00g |

### per-step target_rate (first 8 steps)

**qwen3:**
| condition | step0 | step1 | step2 | step3 | step4 | step5 | step6 | step7 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| repeat2(heli) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| repeat4(rocket) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mean_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random_cache | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| tangent_r2 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| normal_r2 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**glm4:**
| condition | step0 | step1 | step2 | step3 | step4 | step5 | step6 | step7 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| repeat2(heli) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| repeat4(rocket) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mean_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| tangent_r2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| normal_r2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**deepseek7b:**
| condition | step0 | step1 | step2 | step3 | step4 | step5 | step6 | step7 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| repeat2(heli) | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| repeat4(rocket) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mean_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| random_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| tangent_r2 | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| normal_r2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Route: def<-def

### clean_non_object_rate

| model | baseline | repeat2(heli) | repeat4(rocket) | mean_cache | random_cache | tangent_r2 | normal_r2 |
|---|---|---|---|---|---|---|---|
| qwen3 |   0.26 |   0.36 |   0.31 |   0.31 |   0.35 |   0.25 |   0.33 |
| glm4 |   0.42 |   0.44 |   0.44 |   0.35 |   0.36 |   0.42 |   0.43 |
| deepseek7b |   0.15 |   0.17 |   0.11 |   0.15 |   0.15 |   0.22 |   0.19 |

### mean_relaxation_length (steps until target token disappears)

| model | baseline | repeat2(heli) | repeat4(rocket) | mean_cache | random_cache | tangent_r2 | normal_r2 |
|---|---|---|---|---|---|---|---|
| qwen3 |   0.49 |   0.58 |   0.53 |   0.64 |   0.50 |   0.65 |   0.58 |
| glm4 |   0.00 |   0.07 |   0.00 |   0.10 |   0.00 |   0.15 |   0.07 |
| deepseek7b |   1.35 |   1.68 |   1.26 |   2.21 |   2.04 |   1.31 |   1.21 |

### semantic_specificity (condition - random_cache)

| model | baseline | repeat2(heli) | repeat4(rocket) | mean_cache | random_cache | tangent_r2 | normal_r2 |
|---|---|---|---|---|---|---|---|
| qwen3 |  -0.08 |  +0.01 |  -0.04 |  -0.04 |  +0.00 |  -0.10 |  -0.01 |
| glm4 |  +0.06 |  +0.08 |  +0.08 |  -0.01 |  +0.00 |  +0.06 |  +0.07 |
| deepseek7b |  +0.00 |  +0.01 |  -0.04 |  +0.00 |  +0.00 |  +0.07 |  +0.04 |

### tangent vs normal (add_tangent vs add_normal)

| model | baseline | tangent_r2 | normal_r2 | tangent-baseline | normal-baseline | tangent-normal |
|---|---|---|---|---|---|---|
| qwen3 | 0.26 | 0.25 | 0.33 | -0.01 | +0.07 | -0.08 |
| glm4 | 0.42 | 0.42 | 0.43 | +0.00 | +0.01 | -0.01 |
| deepseek7b | 0.15 | 0.22 | 0.19 | +0.07 | +0.04 | +0.03 |

### degeneration_distribution (normal / repetition / garbage)

| model | baseline | repeat2(heli) | repeat4(rocket) | mean_cache | random_cache | tangent_r2 | normal_r2 |
|---|---|---|---|---|---|---|---|
| qwen3 | 0.99n/0.01r/0.00g | 0.99n/0.01r/0.00g | 0.96n/0.03r/0.00g | 0.99n/0.01r/0.00g | 0.97n/0.01r/0.00g | 0.97n/0.01r/0.00g | 0.99n/0.01r/0.00g |
| glm4 | 1.00n/0.00r/0.00g | 0.99n/0.00r/0.01g | 0.99n/0.01r/0.00g | 1.00n/0.00r/0.00g | 1.00n/0.00r/0.00g | 1.00n/0.00r/0.00g | 1.00n/0.00r/0.00g |
| deepseek7b | 0.97n/0.01r/0.00g | 0.99n/0.01r/0.00g | 0.97n/0.03r/0.00g | 1.00n/0.00r/0.00g | 0.96n/0.01r/0.00g | 0.94n/0.06r/0.00g | 0.99n/0.00r/0.00g |

### per-step target_rate (first 8 steps)

**qwen3:**
| condition | step0 | step1 | step2 | step3 | step4 | step5 | step6 | step7 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.01 |
| repeat2(heli) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 |
| repeat4(rocket) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 |
| mean_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| random_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 |
| tangent_r2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| normal_r2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |

**glm4:**
| condition | step0 | step1 | step2 | step3 | step4 | step5 | step6 | step7 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| repeat2(heli) | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| repeat4(rocket) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mean_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| random_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| tangent_r2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| normal_r2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |

**deepseek7b:**
| condition | step0 | step1 | step2 | step3 | step4 | step5 | step6 | step7 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.00 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.04 |
| repeat2(heli) | 0.00 | 0.01 | 0.00 | 0.01 | 0.03 | 0.01 | 0.03 | 0.06 |
| repeat4(rocket) | 0.00 | 0.01 | 0.00 | 0.00 | 0.04 | 0.00 | 0.01 | 0.03 |
| mean_cache | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 | 0.07 |
| random_cache | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 |
| tangent_r2 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 | 0.04 | 0.01 | 0.06 |
| normal_r2 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.03 | 0.00 | 0.04 |

## Timing

| model | time (min) |
|---|---|
| qwen3 | 1.03 |
| glm4 | 19.9 |
| deepseek7b | 8.58 |

## Direction Decomposition (tangent vs normal component norms)

GLM4 sent<-def: |u|≈25-53, |tangent|≈5-9, |normal|≈21-39 → donor direction mostly NORMAL to trajectory
GLM4 def<-def: |tangent|≈0.5-0.9, |normal|≈12-30 → even more normal-dominated
DS7B: |u|≈215-356 (much larger), |tangent|≈7-9, |normal|≈69-111 → also normal-dominated
