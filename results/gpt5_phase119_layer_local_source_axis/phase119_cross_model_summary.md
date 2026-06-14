# Phase 119 Cross-model Layer-local Source Axis Discovery

## Test Scope
- models: qwen3, glm4, deepseek7b; categories: number, container, plant; train/test objects per category: 8/16; templates: 4; prompts/category: 64
- layers: peak-3 ... peak; sites: object_last, object_span_mean, post_object_mean, answer_last; rank: 16; scale: 1.5

## Cross-model Table
| model | category | axis | object_last | object_span | post_object | best source | answer_last | class |
|---|---|---|---|---|---|---|---|---|
| qwen3 | number | local_varimax_best | L34 object_last T-0.00 R+0.00 | L34 object_span_mean T-0.00 R+0.00 | L35 post_object_mean T-4.43 R+1.93 | L35 post_object_mean T-4.43 R+1.93 | L35 answer_last T-1.41 R+2.53 | local_source_axis_found |
| qwen3 | number | local_svd_subspace | L35 object_last T+0.00 R+0.01 | L35 object_span_mean T+0.00 R+0.01 | L35 post_object_mean T-4.41 R+2.30 | L35 post_object_mean T-4.41 R+2.30 | L35 answer_last T-1.82 R+2.76 | local_source_axis_found |
| qwen3 | number | random_in_local_subspace | L35 object_last T-0.01 R+0.00 | L33 object_span_mean T-0.02 R+0.00 | L34 post_object_mean T-0.08 R+0.23 | L34 post_object_mean T-0.08 R+0.23 | L33 answer_last T-0.14 R+0.49 | weak_or_no_local_source |
| qwen3 | container | local_varimax_best | L32 object_last T-0.07 R+0.06 | L32 object_span_mean T-0.07 R+0.04 | L32 post_object_mean T-1.23 R+1.86 | L32 post_object_mean T-1.23 R+1.86 | L35 answer_last T-2.64 R+1.33 | answer_site_dominant |
| qwen3 | container | local_svd_subspace | L32 object_last T-0.04 R+0.03 | L32 object_span_mean T-0.05 R+0.03 | L32 post_object_mean T-1.73 R+3.61 | L32 post_object_mean T-1.73 R+3.61 | L35 answer_last T-2.53 R+1.90 | answer_site_dominant |
| qwen3 | container | random_in_local_subspace | L32 object_last T-0.00 R+0.00 | L35 object_span_mean T-0.01 R+0.00 | L34 post_object_mean T-0.08 R+0.26 | L34 post_object_mean T-0.08 R+0.26 | L32 answer_last T-0.12 R+0.69 | weak_or_no_local_source |
| qwen3 | plant | local_varimax_best | L32 object_last T-0.02 R+0.01 | L32 object_span_mean T-0.05 R+0.07 | L35 post_object_mean T-5.29 R+1.37 | L35 post_object_mean T-5.29 R+1.37 | L35 answer_last T-0.94 R+1.36 | local_source_axis_found |
| qwen3 | plant | local_svd_subspace | L35 object_last T+0.02 R+0.04 | L32 object_span_mean T-0.04 R+0.13 | L35 post_object_mean T-4.66 R+1.83 | L35 post_object_mean T-4.66 R+1.83 | L33 answer_last T-1.28 R+1.00 | local_source_axis_found |
| qwen3 | plant | random_in_local_subspace | L34 object_last T-0.01 R+0.03 | L33 object_span_mean T+0.00 R+0.02 | L35 post_object_mean T-0.71 R+0.67 | L35 post_object_mean T-0.71 R+0.67 | L35 answer_last T-1.64 R+0.00 | weak_or_no_local_source |
| glm4 | number | local_varimax_best | L16 object_last T-0.16 R+0.05 | L16 object_span_mean T-0.20 R+0.06 | L18 post_object_mean T-0.38 R+0.00 | L18 post_object_mean T-0.38 R+0.00 | L18 answer_last T-0.38 R+0.26 | weak_or_no_local_source |
| glm4 | number | local_svd_subspace | L15 object_last T-0.27 R+0.19 | L15 object_span_mean T-0.20 R+0.23 | L18 post_object_mean T-1.11 R+0.05 | L18 post_object_mean T-1.11 R+0.05 | L18 answer_last T-0.90 R+0.68 | weak_local_source_axis |
| glm4 | number | random_in_local_subspace | L17 object_last T-0.15 R+0.11 | L18 object_span_mean T-0.14 R+0.39 | L17 post_object_mean T-0.14 R+0.18 | L17 object_last T-0.15 R+0.11 | L18 answer_last T-0.10 R+0.05 | weak_or_no_local_source |
| glm4 | container | local_varimax_best | L18 object_last T-0.14 R+0.08 | L18 object_span_mean T-0.09 R+0.06 | L18 post_object_mean T-0.48 R+0.57 | L18 post_object_mean T-0.48 R+0.57 | L17 answer_last T-0.15 R+0.16 | weak_or_no_local_source |
| glm4 | container | local_svd_subspace | L18 object_last T+0.02 R+0.40 | L18 object_span_mean T+0.04 R+0.40 | L18 post_object_mean T-0.01 R+0.79 | L18 post_object_mean T-0.01 R+0.79 | L18 answer_last T-0.22 R+0.21 | weak_or_no_local_source |
| glm4 | container | random_in_local_subspace | L16 object_last T-0.05 R+0.09 | L16 object_span_mean T-0.15 R+0.31 | L18 post_object_mean T-0.02 R+0.04 | L16 object_span_mean T-0.15 R+0.31 | L18 answer_last T-0.05 R+0.02 | weak_or_no_local_source |
| glm4 | plant | local_varimax_best | L16 object_last T-0.06 R+0.09 | L15 object_span_mean T-0.09 R+0.19 | L18 post_object_mean T-0.29 R+0.31 | L18 post_object_mean T-0.29 R+0.31 | L15 answer_last T-0.04 R+0.03 | weak_or_no_local_source |
| glm4 | plant | local_svd_subspace | L17 object_last T-0.11 R+0.28 | L15 object_span_mean T-0.08 R+0.47 | L17 post_object_mean T-0.06 R+0.47 | L17 object_last T-0.11 R+0.28 | L18 answer_last T-0.13 R+0.00 | weak_or_no_local_source |
| glm4 | plant | random_in_local_subspace | L18 object_last T-0.03 R+0.14 | L18 object_span_mean T-0.05 R+0.14 | L17 post_object_mean T-0.05 R+0.10 | L18 object_span_mean T-0.05 R+0.14 | L18 answer_last T-0.06 R+0.02 | weak_or_no_local_source |
| deepseek7b | number | local_varimax_best | L27 object_last T-0.78 R+0.00 | L27 object_span_mean T-0.81 R+0.00 | L27 post_object_mean T-11.74 R+0.00 | L27 post_object_mean T-11.74 R+0.00 | L27 answer_last T-12.24 R+0.00 | local_source_and_answer_axes |
| deepseek7b | number | local_svd_subspace | L27 object_last T-0.76 R+0.00 | L27 object_span_mean T-0.84 R+0.00 | L27 post_object_mean T-12.03 R+0.00 | L27 post_object_mean T-12.03 R+0.00 | L27 answer_last T-12.58 R+0.00 | local_source_and_answer_axes |
| deepseek7b | number | random_in_local_subspace | L26 object_last T-0.05 R+0.08 | L25 object_span_mean T-0.05 R+0.04 | L25 post_object_mean T-4.07 R+0.00 | L25 post_object_mean T-4.07 R+0.00 | L24 answer_last T-3.82 R+0.00 | local_source_and_answer_axes |
| deepseek7b | container | local_varimax_best | L27 object_last T-0.90 R+0.00 | L27 object_span_mean T-0.95 R+0.00 | L27 post_object_mean T-13.24 R+0.00 | L27 post_object_mean T-13.24 R+0.00 | L27 answer_last T-11.53 R+0.00 | local_source_and_answer_axes |
| deepseek7b | container | local_svd_subspace | L27 object_last T-0.93 R+0.00 | L27 object_span_mean T-1.03 R+0.00 | L27 post_object_mean T-12.74 R+0.00 | L27 post_object_mean T-12.74 R+0.00 | L27 answer_last T-12.52 R+0.00 | local_source_and_answer_axes |
| deepseek7b | container | random_in_local_subspace | L25 object_last T-0.22 R+0.00 | L25 object_span_mean T-0.34 R+0.00 | L27 post_object_mean T-1.48 R+0.00 | L27 post_object_mean T-1.48 R+0.00 | L27 answer_last T-1.74 R+0.00 | weak_local_source_axis |
| deepseek7b | plant | local_varimax_best | L27 object_last T-0.97 R+0.00 | L27 object_span_mean T-1.44 R+0.00 | L27 post_object_mean T-10.58 R+0.00 | L27 post_object_mean T-10.58 R+0.00 | L27 answer_last T-8.63 R+0.00 | local_source_and_answer_axes |
| deepseek7b | plant | local_svd_subspace | L27 object_last T-0.72 R+0.00 | L27 object_span_mean T-1.05 R+0.00 | L27 post_object_mean T-9.57 R+0.00 | L27 post_object_mean T-9.57 R+0.00 | L27 answer_last T-7.87 R+0.00 | local_source_and_answer_axes |
| deepseek7b | plant | random_in_local_subspace | L24 object_last T-0.04 R+0.04 | L27 object_span_mean T-0.15 R+0.00 | L24 post_object_mean T-2.61 R+0.00 | L24 post_object_mean T-2.61 R+0.00 | L24 answer_last T-1.54 R+0.00 | weak_local_source_axis |

## Reading Rules
- Each local axis is fit at its own layer and site, then patched at that same layer and site.
- Source sites are object_last, object_span_mean, and post_object_mean.
- answer_last remains the readout-site baseline.

## Hard Limits
- Local source axes are selected by single-site removal, not by an explicit source-to-answer transform fit.
- object_span_mean and post_object_mean patch all tokens in the group with one local mean-derived axis.
- Results are DCF logits, not open generation.
