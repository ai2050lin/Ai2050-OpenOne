# Phase 121 Cross-model Pre-answer and Answer Additivity

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, plant; train/test objects per category: 8/16; templates: 4; prompts/category: 64
- layers: peak-3 ... peak; rank: 16; scale: 1.5; axis types: local_varimax_best, local_svd_subspace

## Cross-model Table
| model | category | axis | best pre-only | best answer-only | best combined | combined-answer | class |
|---|---|---|---|---|---|---|---|
| qwen3 | number | local_varimax_best | L33 T-0.22 R+0.43 | L35 T-1.41 R+2.53 | L35 T-1.58 R+2.20 | -0.17 | answer_absorbs_pre |
| qwen3 | number | local_svd_subspace | L33 T-0.13 R+0.45 | L35 T-1.82 R+2.76 | L35 T-2.00 R+2.49 | -0.18 | answer_absorbs_pre |
| qwen3 | container | local_varimax_best | L34 T-0.14 R+0.40 | L35 T-2.64 R+1.33 | L35 T-2.64 R+1.31 | +0.00 | answer_absorbs_pre |
| qwen3 | container | local_svd_subspace | L32 T-0.04 R+0.26 | L35 T-2.53 R+1.90 | L32 T-2.29 R+1.27 | +0.24 | answer_absorbs_pre |
| qwen3 | plant | local_varimax_best | L32 T-0.15 R+0.48 | L35 T-0.94 R+1.36 | L32 T-0.98 R+0.46 | -0.04 | answer_absorbs_pre |
| qwen3 | plant | local_svd_subspace | L32 T-0.22 R+0.29 | L33 T-1.28 R+1.00 | L32 T-2.02 R+0.37 | -0.74 | answer_absorbs_pre |
| glm4 | number | local_varimax_best | L17 T-0.12 R+0.26 | L18 T-0.38 R+0.26 | L17 T-0.44 R+0.56 | -0.06 | answer_absorbs_pre |
| glm4 | number | local_svd_subspace | L17 T-0.26 R+0.27 | L18 T-0.90 R+0.68 | L18 T-0.87 R+1.00 | +0.03 | answer_absorbs_pre |
| glm4 | container | local_varimax_best | L17 T-0.29 R+0.32 | L17 T-0.15 R+0.16 | L17 T-0.44 R+0.28 | -0.28 | answer_absorbs_pre |
| glm4 | container | local_svd_subspace | L17 T-0.01 R+0.70 | L18 T-0.22 R+0.21 | L17 T-0.17 R+0.63 | +0.05 | answer_absorbs_pre |
| glm4 | plant | local_varimax_best | L18 T-0.24 R+0.31 | L15 T-0.04 R+0.03 | L18 T-0.26 R+0.13 | -0.22 | answer_absorbs_pre |
| glm4 | plant | local_svd_subspace | L17 T-0.02 R+0.41 | L18 T-0.13 R+0.00 | L17 T-0.29 R+0.00 | -0.16 | answer_absorbs_pre |
| deepseek7b | number | local_varimax_best | L27 T-2.35 R+0.57 | L27 T-12.24 R+0.00 | L27 T-13.51 R+0.00 | -1.27 | additive_or_independent |
| deepseek7b | number | local_svd_subspace | L27 T-2.51 R+0.54 | L27 T-12.58 R+0.00 | L27 T-13.71 R+0.00 | -1.13 | additive_or_independent |
| deepseek7b | container | local_varimax_best | L27 T-2.79 R+0.78 | L27 T-11.53 R+0.00 | L27 T-12.85 R+0.00 | -1.33 | additive_or_independent |
| deepseek7b | container | local_svd_subspace | L27 T-2.66 R+0.88 | L27 T-12.52 R+0.00 | L27 T-13.69 R+0.00 | -1.17 | additive_or_independent |
| deepseek7b | plant | local_varimax_best | L27 T-2.64 R+1.45 | L27 T-8.63 R+0.00 | L27 T-10.15 R+0.00 | -1.52 | additive_or_independent |
| deepseek7b | plant | local_svd_subspace | L27 T-2.42 R+1.56 | L27 T-7.87 R+0.00 | L27 T-9.32 R+0.00 | -1.45 | additive_or_independent |

## Reading Rules
- combined-answer is target_delta(combined) minus target_delta(answer-only). Negative means combined is stronger than answer-only.
- answer_absorbs_pre means combined is close to answer-only, so pre-answer adds little under this patch.
- additive_or_independent means combined is at least 1 logit stronger than answer-only and pre-only.

## Hard Limits
- Pre-answer and answer axes are selected independently at the same layer.
- This does not identify the attention/MLP writer of either field.
- Results are DCF logits, not open generation.
