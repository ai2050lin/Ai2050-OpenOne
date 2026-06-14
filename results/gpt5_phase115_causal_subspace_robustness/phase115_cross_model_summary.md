# Phase 115 Cross-model Causal Subspace Robustness

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, clothing, plant; train/test objects per category: 8/16; templates: 4; full prompts/category: 64
- ranks: [8, 16]; scales: [0.25, 0.5, 1.0, 1.5]; layer: model-specific causal peak
- robustness: full-template basis, leave-template-out basis, matched-spectrum random, release-excluded subspace

## Cross-model Table
| model | category | full subspace | matched random | release-excluded | LTO mean | LTO random mean | class |
|---|---|---|---|---|---|---|---|
| qwen3 | number | r8 s1.5 T-1.83 R+2.37 | r8 s0.25 T+0.05 R+0.09 | r8 s1.5 T-2.72 R+1.99 | rfold_best sfold_best T-2.18 R+2.12 | rfold_best sfold_best T-0.19 R+0.13 | mixed |
| qwen3 | container | r16 s1.5 T-2.53 R+1.90 | r16 s0.25 T+0.03 R+0.07 | r16 s1.5 T-3.06 R+1.97 | rfold_best sfold_best T-2.54 R+1.54 | rfold_best sfold_best T-0.07 R+0.09 | robust_moderate |
| qwen3 | clothing | r8 s0.25 T+0.22 R+0.51 | r16 s1.5 T-0.38 R+0.41 | r8 s1.5 T+0.13 R+0.91 | rfold_best sfold_best T+0.24 R+0.81 | rfold_best sfold_best T-0.30 R+0.23 | control_sensitive |
| qwen3 | plant | r16 s1.5 T-1.24 R+1.59 | r16 s1.5 T-0.15 R+0.68 | r16 s1.5 T-1.17 R+1.77 | rfold_best sfold_best T-1.74 R+1.41 | rfold_best sfold_best T-0.10 R+0.05 | mixed |
| glm4 | number | r16 s1.5 T-0.90 R+0.68 | r8 s0.5 T-0.02 R+0.00 | r16 s1.5 T-0.89 R+0.66 | rfold_best sfold_best T-0.58 R+0.37 | rfold_best sfold_best T-0.03 R+0.03 | mixed |
| glm4 | container | r8 s1.5 T-0.32 R+0.06 | r16 s1.5 T-0.01 R+0.04 | r8 s1.5 T-0.32 R+0.06 | rfold_best sfold_best T-0.36 R+0.21 | rfold_best sfold_best T-0.02 R+0.06 | control_sensitive |
| glm4 | clothing | r8 s1.5 T-0.28 R+0.20 | r16 s0.25 T-0.00 R+0.02 | r8 s1.5 T-0.27 R+0.20 | rfold_best sfold_best T-0.19 R+0.16 | rfold_best sfold_best T-0.03 R+0.02 | control_sensitive |
| glm4 | plant | r16 s1.5 T-0.13 R+0.00 | r8 s1.0 T-0.01 R+0.03 | NA | rfold_best sfold_best T-0.07 R+0.04 | rfold_best sfold_best T-0.01 R+0.04 | control_sensitive |
| deepseek7b | number | r16 s1.5 T-12.58 R+0.00 | r8 s0.25 T-0.07 R+0.08 | NA | rfold_best sfold_best T-11.59 R+0.00 | rfold_best sfold_best T-0.20 R+0.09 | robust_strong |
| deepseek7b | container | r16 s1.5 T-12.52 R+0.00 | r8 s1.5 T-0.24 R+0.12 | NA | rfold_best sfold_best T-11.45 R+0.00 | rfold_best sfold_best T-0.37 R+0.08 | robust_strong |
| deepseek7b | clothing | r8 s1.0 T-4.20 R+0.00 | r16 s0.25 T-0.06 R+0.07 | NA | rfold_best sfold_best T-5.07 R+0.00 | rfold_best sfold_best T-0.37 R+0.22 | robust_moderate |
| deepseek7b | plant | r8 s1.5 T-9.40 R+0.00 | r16 s1.5 T-0.29 R+0.16 | NA | rfold_best sfold_best T-8.71 R+0.00 | rfold_best sfold_best T-0.22 R+0.13 | robust_strong |

## Objective Reading Rules
- LTO mean averages the best heldout-template result across the four heldout templates.
- robust_strong requires full-template target_delta <= -5 and LTO mean <= -3, with controls weaker.
- release-excluded removes the strongest release category from contrast construction.

## Hard Limits
- Matched-spectrum random is implemented through synthetic contrast matrices, but the intervention still uses orthonormal bases.
- Release decomposition only excludes the strongest observed release category; it is not a full support/release factorization.
- This phase still uses DCF logits, not open generation.
