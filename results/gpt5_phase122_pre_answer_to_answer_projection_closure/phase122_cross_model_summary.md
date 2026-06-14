# Phase 122 Cross-model Pre-answer to Answer Projection Closure

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, plant; train/test objects per category: 8/16; templates: 4; prompts/category: 64
- patch layers: [32, 33, 34, 35]; monitor layer: L35; rank: 16; scale: 1.5

## Cross-model Table
| model | category | axis | best pre | best answer | best combined | combo-answer | strongest pre answer-proj drop | class |
|---|---|---|---|---|---|---|---|---|
| qwen3 | number | local_varimax_best | L33 pre_remove T-0.22 R+0.43 Aproj+10.86 | L35 answer_remove T-1.41 R+2.53 Aproj+0.00 | L35 pre_plus_answer T-1.58 R+2.20 Aproj+0.00 | -0.17 | L35 pre_remove T-0.08 R+0.15 Aproj+0.00 | answer_absorbs_pre |
| qwen3 | number | local_svd_subspace | L33 pre_remove T-0.13 R+0.45 Aproj-11.13 | L35 answer_remove T-1.82 R+2.76 Aproj+0.00 | L35 pre_plus_answer T-2.00 R+2.49 Aproj+0.00 | -0.18 | L32 pre_remove T-0.06 R+0.26 Aproj-13.59 | answer_absorbs_pre |
| qwen3 | container | local_varimax_best | L34 pre_remove T-0.14 R+0.40 Aproj-10.67 | L35 answer_remove T-2.64 R+1.33 Aproj+0.00 | L35 pre_plus_answer T-2.64 R+1.31 Aproj+0.00 | +0.00 | L32 pre_remove T-0.14 R+0.40 Aproj-11.43 | answer_absorbs_pre |
| qwen3 | container | local_svd_subspace | L32 pre_remove T-0.04 R+0.26 Aproj-8.59 | L35 answer_remove T-2.53 R+1.90 Aproj+0.00 | L32 pre_plus_answer T-2.29 R+1.27 Aproj-276.21 | +0.24 | L34 pre_remove T-0.04 R+0.41 Aproj-9.50 | answer_absorbs_pre |
| qwen3 | plant | local_varimax_best | L32 pre_remove T-0.15 R+0.48 Aproj+17.40 | L35 answer_remove T-0.94 R+1.36 Aproj+0.00 | L32 pre_plus_answer T-0.98 R+0.46 Aproj+334.01 | -0.04 | L33 pre_remove T-0.04 R+0.00 Aproj-0.09 | answer_absorbs_pre |
| qwen3 | plant | local_svd_subspace | L32 pre_remove T-0.22 R+0.29 Aproj-13.05 | L33 answer_remove T-1.28 R+1.00 Aproj-237.59 | L32 pre_plus_answer T-2.02 R+0.37 Aproj-229.20 | -0.74 | L32 pre_remove T-0.22 R+0.29 Aproj-13.05 | answer_absorbs_pre |
| glm4 | number | local_varimax_best | L17 pre_remove T-0.12 R+0.26 Aproj+0.01 | L18 answer_remove T-0.38 R+0.26 Aproj+0.00 | L17 pre_plus_answer T-0.44 R+0.56 Aproj+1.78 | -0.06 | L18 pre_remove T-0.07 R+0.34 Aproj+0.00 | answer_absorbs_pre |
| glm4 | number | local_svd_subspace | L17 pre_remove T-0.26 R+0.27 Aproj-0.08 | L18 answer_remove T-0.90 R+0.68 Aproj+0.00 | L18 pre_plus_answer T-0.87 R+1.00 Aproj+0.00 | +0.03 | L15 pre_remove T-0.11 R+0.29 Aproj-0.14 | answer_absorbs_pre |
| glm4 | container | local_varimax_best | L17 pre_remove T-0.29 R+0.32 Aproj-0.02 | L17 answer_remove T-0.15 R+0.16 Aproj-0.26 | L17 pre_plus_answer T-0.44 R+0.28 Aproj-0.28 | -0.28 | L16 pre_remove T-0.20 R+0.42 Aproj-0.03 | answer_absorbs_pre |
| glm4 | container | local_svd_subspace | L17 pre_remove T-0.01 R+0.70 Aproj-0.06 | L18 answer_remove T-0.22 R+0.21 Aproj+0.00 | L17 pre_plus_answer T-0.17 R+0.63 Aproj-1.89 | +0.05 | L15 pre_remove T+0.03 R+0.55 Aproj-0.22 | answer_absorbs_pre |
| glm4 | plant | local_varimax_best | L18 pre_remove T-0.24 R+0.31 Aproj+0.00 | L15 answer_remove T-0.04 R+0.03 Aproj+0.03 | L18 pre_plus_answer T-0.26 R+0.13 Aproj+0.00 | -0.22 | L18 pre_remove T-0.24 R+0.31 Aproj+0.00 | answer_absorbs_pre |
| glm4 | plant | local_svd_subspace | L17 pre_remove T-0.02 R+0.41 Aproj-0.13 | L18 answer_remove T-0.13 R+0.00 Aproj+0.00 | L17 pre_plus_answer T-0.29 R+0.00 Aproj-2.16 | -0.16 | L15 pre_remove T+0.02 R+0.38 Aproj-0.40 | answer_absorbs_pre |
| deepseek7b | number | local_varimax_best | L27 pre_remove T-2.35 R+0.57 Aproj+0.00 | L27 answer_remove T-12.24 R+0.00 Aproj+0.00 | L27 pre_plus_answer T-13.51 R+0.00 Aproj+0.00 | -1.27 | L24 pre_remove T-2.06 R+0.00 Aproj-55.89 | pre_writes_answer_projection |
| deepseek7b | number | local_svd_subspace | L27 pre_remove T-2.51 R+0.54 Aproj+0.00 | L27 answer_remove T-12.58 R+0.00 Aproj+0.00 | L27 pre_plus_answer T-13.71 R+0.00 Aproj+0.00 | -1.13 | L24 pre_remove T-2.05 R+0.00 Aproj-51.27 | pre_writes_answer_projection |
| deepseek7b | container | local_varimax_best | L27 pre_remove T-2.79 R+0.78 Aproj+0.00 | L27 answer_remove T-11.53 R+0.00 Aproj+0.00 | L27 pre_plus_answer T-12.85 R+0.00 Aproj+0.00 | -1.33 | L27 pre_remove T-2.79 R+0.78 Aproj+0.00 | pre_adds_without_projection_drop |
| deepseek7b | container | local_svd_subspace | L27 pre_remove T-2.66 R+0.88 Aproj+0.00 | L27 answer_remove T-12.52 R+0.00 Aproj+0.00 | L27 pre_plus_answer T-13.69 R+0.00 Aproj+0.00 | -1.17 | L24 pre_remove T-1.19 R+0.00 Aproj-44.30 | pre_writes_answer_projection |
| deepseek7b | plant | local_varimax_best | L27 pre_remove T-2.64 R+1.45 Aproj+0.00 | L27 answer_remove T-8.63 R+0.00 Aproj+0.00 | L27 pre_plus_answer T-10.15 R+0.00 Aproj+0.00 | -1.52 | L24 pre_remove T-2.36 R+0.00 Aproj-49.09 | pre_writes_answer_projection |
| deepseek7b | plant | local_svd_subspace | L27 pre_remove T-2.42 R+1.56 Aproj+0.00 | L27 answer_remove T-7.87 R+0.00 Aproj+0.00 | L27 pre_plus_answer T-9.32 R+0.00 Aproj+0.00 | -1.45 | L24 pre_remove T-2.28 R+0.00 Aproj-44.10 | pre_writes_answer_projection |

## Reading Rules
- Aproj is the peak answer_last projection delta on the answer-site axis/subspace.
- pre_writes_answer_projection requires combined to beat answer-only and pre_remove to lower answer projection.
- pre_adds_without_projection_drop means extra logit effect exists but was not visible as mean answer-axis projection loss.
