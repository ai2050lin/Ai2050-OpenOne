# Phase 118 Cross-model Causal Axis Transport Closure

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, plant; train/test objects per category: 8/16; templates: 4; prompts/category: 64
- monitor layer: model boundary layer; patch layers: monitor-layer-3 ... monitor-layer; rank: 16; scale: 1.5

## Cross-model Table
| model | category | axis | selected varimax single | best object_last | best answer_last | best both | class |
|---|---|---|---|---|---|---|---|
| qwen3 | number | varimax_best | b0 T-1.41 R+2.53 | L32 object_last T-0.02 R+0.04 Aproj+1.69 | L35 answer_last T-1.41 R+2.53 Aproj+0.00 | L35 both T-1.41 R+2.57 Aproj+0.00 | weak_or_no_closure |
| qwen3 | number | svd_subspace | b0 T-1.41 R+2.53 | L32 object_last T-0.01 R+0.04 Aproj+1.68 | L35 answer_last T-1.82 R+2.76 Aproj+0.00 | L35 both T-1.78 R+2.79 Aproj+0.00 | weak_or_no_closure |
| qwen3 | number | random_in_subspace | b0 T-1.41 R+2.53 | L35 object_last T-0.01 R+0.01 Aproj+0.00 | L35 answer_last T-1.35 R+0.30 Aproj+0.00 | L35 both T-1.37 R+0.28 Aproj+0.00 | weak_or_no_closure |
| qwen3 | container | varimax_best | b2 T-2.64 R+1.33 | L33 object_last T-0.07 R+0.05 Aproj-2.55 | L35 answer_last T-2.64 R+1.33 Aproj+0.00 | L35 both T-2.73 R+1.28 Aproj+0.00 | answer_site_dominant |
| qwen3 | container | svd_subspace | b2 T-2.64 R+1.33 | L33 object_last T-0.06 R+0.06 Aproj-2.47 | L35 answer_last T-2.53 R+1.90 Aproj+0.00 | L35 both T-2.57 R+1.88 Aproj+0.00 | answer_site_dominant |
| qwen3 | container | random_in_subspace | b2 T-2.64 R+1.33 | L35 object_last T+0.00 R+0.00 Aproj+0.00 | L32 answer_last T-0.01 R+0.02 Aproj-0.57 | L32 both T-0.01 R+0.03 Aproj-0.28 | weak_or_no_closure |
| qwen3 | plant | varimax_best | b0 T-0.94 R+1.36 | L32 object_last T+0.01 R+0.08 Aproj+1.64 | L35 answer_last T-0.94 R+1.36 Aproj+0.00 | L35 both T-1.00 R+1.31 Aproj+0.00 | weak_or_no_closure |
| qwen3 | plant | svd_subspace | b0 T-0.94 R+1.36 | L33 object_last T+0.03 R+0.10 Aproj+2.10 | L35 answer_last T-1.24 R+1.59 Aproj+0.00 | L35 both T-1.28 R+1.56 Aproj+0.00 | weak_or_no_closure |
| qwen3 | plant | random_in_subspace | b0 T-0.94 R+1.36 | L35 object_last T+0.02 R+0.02 Aproj+0.00 | L33 answer_last T+0.15 R+0.23 Aproj-76.14 | L33 both T+0.17 R+0.26 Aproj-78.08 | weak_or_no_closure |
| glm4 | number | varimax_best | b2 T-0.38 R+0.26 | L18 object_last T+0.00 R+0.06 Aproj+0.00 | L18 answer_last T-0.38 R+0.26 Aproj+0.00 | L18 both T-0.37 R+0.27 Aproj+0.00 | weak_or_no_closure |
| glm4 | number | svd_subspace | b2 T-0.38 R+0.26 | L15 object_last T-0.31 R+0.17 Aproj+0.02 | L18 answer_last T-0.90 R+0.68 Aproj+0.00 | L18 both T-1.13 R+0.88 Aproj+0.00 | weak_or_no_closure |
| glm4 | number | random_in_subspace | b2 T-0.38 R+0.26 | L18 object_last T-0.03 R+0.07 Aproj+0.00 | L15 answer_last T-0.03 R+0.00 Aproj+0.18 | L16 both T-0.04 R+0.11 Aproj+0.20 | weak_or_no_closure |
| glm4 | container | varimax_best | b3 T-0.15 R+0.09 | L16 object_last T-0.01 R+0.02 Aproj+0.00 | L18 answer_last T-0.15 R+0.09 Aproj+0.00 | L18 both T-0.15 R+0.09 Aproj+0.00 | weak_or_no_closure |
| glm4 | container | svd_subspace | b3 T-0.15 R+0.09 | L15 object_last T-0.18 R+0.24 Aproj-0.00 | L18 answer_last T-0.22 R+0.21 Aproj+0.00 | L18 both T-0.50 R+0.43 Aproj+0.00 | weak_or_no_closure |
| glm4 | container | random_in_subspace | b3 T-0.15 R+0.09 | L15 object_last T-0.10 R+0.09 Aproj-0.01 | L17 answer_last T-0.02 R+0.14 Aproj-0.76 | L17 both T-0.09 R+0.24 Aproj-0.75 | weak_or_no_closure |
| glm4 | plant | varimax_best | b1 T-0.04 R+0.00 | L16 object_last T-0.02 R+0.03 Aproj+0.00 | L18 answer_last T-0.04 R+0.00 Aproj+0.00 | L16 both T-0.05 R+0.00 Aproj+0.75 | weak_or_no_closure |
| glm4 | plant | svd_subspace | b1 T-0.04 R+0.00 | L15 object_last T+0.06 R+0.21 Aproj+0.03 | L18 answer_last T-0.13 R+0.00 Aproj+0.00 | L18 both T-0.17 R+0.00 Aproj+0.00 | weak_or_no_closure |
| glm4 | plant | random_in_subspace | b1 T-0.04 R+0.00 | L15 object_last T+0.01 R+0.07 Aproj-0.01 | L17 answer_last T-0.03 R+0.11 Aproj-0.61 | L17 both T-0.01 R+0.15 Aproj-0.61 | weak_or_no_closure |
| deepseek7b | number | varimax_best | b1 T-12.24 R+0.00 | L27 object_last T-0.74 R+0.00 Aproj+0.00 | L27 answer_last T-12.24 R+0.00 Aproj+0.00 | L27 both T-12.46 R+0.00 Aproj+0.00 | answer_site_assembled |
| deepseek7b | number | svd_subspace | b1 T-12.24 R+0.00 | L27 object_last T-0.79 R+0.00 Aproj+0.00 | L27 answer_last T-12.58 R+0.00 Aproj+0.00 | L27 both T-12.78 R+0.00 Aproj+0.00 | answer_site_assembled |
| deepseek7b | number | random_in_subspace | b1 T-12.24 R+0.00 | L25 object_last T-0.09 R+0.05 Aproj-1.37 | L27 answer_last T-2.38 R+0.00 Aproj+0.00 | L27 both T-2.42 R+0.00 Aproj+0.00 | answer_site_dominant |
| deepseek7b | container | varimax_best | b0 T-11.53 R+0.00 | L27 object_last T-0.47 R+0.00 Aproj+0.00 | L27 answer_last T-11.53 R+0.00 Aproj+0.00 | L27 both T-11.70 R+0.00 Aproj+0.00 | answer_site_assembled |
| deepseek7b | container | svd_subspace | b0 T-11.53 R+0.00 | L25 object_last T-0.48 R+0.00 Aproj+4.69 | L27 answer_last T-12.52 R+0.00 Aproj+0.00 | L27 both T-12.68 R+0.00 Aproj+0.00 | answer_site_assembled |
| deepseek7b | container | random_in_subspace | b0 T-11.53 R+0.00 | L25 object_last T-0.24 R+0.03 Aproj-0.11 | L26 answer_last T-0.20 R+0.00 Aproj-86.95 | L26 both T-0.22 R+0.00 Aproj-87.05 | weak_or_no_closure |
| deepseek7b | plant | varimax_best | b0 T-8.63 R+0.00 | L27 object_last T-0.95 R+0.00 Aproj+0.00 | L27 answer_last T-8.63 R+0.00 Aproj+0.00 | L27 both T-8.91 R+0.00 Aproj+0.00 | answer_site_assembled |
| deepseek7b | plant | svd_subspace | b0 T-8.63 R+0.00 | L27 object_last T-0.90 R+0.00 Aproj+0.00 | L27 answer_last T-7.87 R+0.00 Aproj+0.00 | L27 both T-8.16 R+0.00 Aproj+0.00 | answer_site_assembled |
| deepseek7b | plant | random_in_subspace | b0 T-8.63 R+0.00 | L24 object_last T-0.19 R+0.00 Aproj+7.67 | L27 answer_last T-2.30 R+0.00 Aproj+0.00 | L27 both T-2.37 R+0.00 Aproj+0.00 | answer_site_dominant |

## Reading Rules
- object_last tests whether the answer-site axis already has upstream causal leverage at the object token.
- answer_last is the direct answer-site removal baseline.
- both tests whether source and answer removals add or interfere.
- Aproj is the mean answer-layer projection delta on the selected varimax axis.

## Hard Limits
- Axes are built at the monitor layer, then reused across nearby layers; layer-wise bases are not refit.
- Projection closure is measured on DCF logits, not open generation.
- A weak object_last effect does not prove absence of upstream encoding; it may use a different coordinate before the answer layer.
