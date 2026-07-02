# Phase 847 Context Gated Residual Audit (confirm)

- Source: Phase 845 pair/triplet interaction residual rows.
- Method: compare global residual means with prompt/object conditioned residual means.
- Boundary: residual transfer diagnostics; not token closure.

## Residual Prediction

| model | split | predictor | n | MAE | RMSE | sign acc | class acc | mean MAE gain vs global |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `in_sample` | `global_combo` | 570 | 0.1612 | 0.3705 | 0.5221 | 0.9351 | NA |
| qwen3 | `in_sample` | `object_combo` | 570 | 0.1124 | 0.2728 | 0.7065 | 0.9526 | 0.0488 |
| qwen3 | `in_sample` | `object_prompt_combo` | 570 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1612 |
| qwen3 | `in_sample` | `prompt_combo` | 570 | 0.1384 | 0.3268 | 0.6131 | 0.9526 | 0.0229 |
| qwen3 | `object_holdout` | `global_combo` | 570 | 0.1813 | 0.4154 | 0.4491 | 0.9351 | NA |
| qwen3 | `object_holdout` | `object_combo` | 570 | 0.1813 | 0.4154 | 0.4491 | 0.9351 | 0.0000 |
| qwen3 | `object_holdout` | `object_prompt_combo` | 570 | 0.1813 | 0.4154 | 0.4491 | 0.9351 | 0.0000 |
| qwen3 | `object_holdout` | `prompt_combo` | 570 | 0.1730 | 0.4085 | 0.4753 | 0.9526 | 0.0083 |
| qwen3 | `prompt_holdout` | `global_combo` | 570 | 0.1802 | 0.4187 | 0.4886 | 0.9351 | NA |
| qwen3 | `prompt_holdout` | `object_combo` | 570 | 0.1686 | 0.4092 | 0.5364 | 0.9298 | 0.0116 |
| qwen3 | `prompt_holdout` | `object_prompt_combo` | 570 | 0.1802 | 0.4187 | 0.4886 | 0.9351 | 0.0000 |
| qwen3 | `prompt_holdout` | `prompt_combo` | 570 | 0.1802 | 0.4187 | 0.4886 | 0.9351 | 0.0000 |
| glm4 | `in_sample` | `global_combo` | 570 | 0.0406 | 0.0556 | 0.5702 | 1.0000 | NA |
| glm4 | `in_sample` | `object_combo` | 570 | 0.0324 | 0.0432 | 0.6912 | 1.0000 | 0.0081 |
| glm4 | `in_sample` | `object_prompt_combo` | 570 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0406 |
| glm4 | `in_sample` | `prompt_combo` | 570 | 0.0388 | 0.0519 | 0.6263 | 1.0000 | 0.0017 |
| glm4 | `object_holdout` | `global_combo` | 570 | 0.0447 | 0.0615 | 0.4474 | 1.0000 | NA |
| glm4 | `object_holdout` | `object_combo` | 570 | 0.0447 | 0.0615 | 0.4474 | 1.0000 | 0.0000 |
| glm4 | `object_holdout` | `object_prompt_combo` | 570 | 0.0447 | 0.0615 | 0.4474 | 1.0000 | 0.0000 |
| glm4 | `object_holdout` | `prompt_combo` | 570 | 0.0486 | 0.0648 | 0.4947 | 1.0000 | -0.0039 |
| glm4 | `prompt_holdout` | `global_combo` | 570 | 0.0438 | 0.0600 | 0.5175 | 1.0000 | NA |
| glm4 | `prompt_holdout` | `object_combo` | 570 | 0.0487 | 0.0648 | 0.5140 | 1.0000 | -0.0049 |
| glm4 | `prompt_holdout` | `object_prompt_combo` | 570 | 0.0438 | 0.0600 | 0.5175 | 1.0000 | 0.0000 |
| glm4 | `prompt_holdout` | `prompt_combo` | 570 | 0.0438 | 0.0600 | 0.5175 | 1.0000 | 0.0000 |
| deepseek7b | `in_sample` | `global_combo` | 570 | 0.0403 | 0.0633 | 0.4509 | 0.9947 | NA |
| deepseek7b | `in_sample` | `object_combo` | 570 | 0.0341 | 0.0555 | 0.6443 | 0.9947 | 0.0062 |
| deepseek7b | `in_sample` | `object_prompt_combo` | 570 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0403 |
| deepseek7b | `in_sample` | `prompt_combo` | 570 | 0.0369 | 0.0569 | 0.5644 | 0.9947 | 0.0034 |
| deepseek7b | `object_holdout` | `global_combo` | 570 | 0.0432 | 0.0672 | 0.4482 | 0.9947 | NA |
| deepseek7b | `object_holdout` | `object_combo` | 570 | 0.0432 | 0.0672 | 0.4482 | 0.9947 | 0.0000 |
| deepseek7b | `object_holdout` | `object_prompt_combo` | 570 | 0.0432 | 0.0672 | 0.4482 | 0.9947 | 0.0000 |
| deepseek7b | `object_holdout` | `prompt_combo` | 570 | 0.0462 | 0.0711 | 0.4446 | 0.9947 | -0.0030 |
| deepseek7b | `prompt_holdout` | `global_combo` | 570 | 0.0448 | 0.0704 | 0.4208 | 0.9947 | NA |
| deepseek7b | `prompt_holdout` | `object_combo` | 570 | 0.0512 | 0.0833 | 0.4430 | 0.9947 | -0.0064 |
| deepseek7b | `prompt_holdout` | `object_prompt_combo` | 570 | 0.0448 | 0.0704 | 0.4208 | 0.9947 | 0.0000 |
| deepseek7b | `prompt_holdout` | `prompt_combo` | 570 | 0.0448 | 0.0704 | 0.4208 | 0.9947 | 0.0000 |

## Context Strength

| model | residual rows | prompt shift | object shift | object+prompt shift | terms |
|---|---:|---:|---:|---:|---|
| qwen3 | 570 | 0.0747 | 0.1174 | 0.1612 | `{"global": 38, "prompt": 114, "object": 190, "object_prompt": 570}` |
| glm4 | 570 | 0.0157 | 0.0255 | 0.0406 | `{"global": 38, "prompt": 114, "object": 190, "object_prompt": 570}` |
| deepseek7b | 570 | 0.0215 | 0.0217 | 0.0403 | `{"global": 38, "prompt": 114, "object": 190, "object_prompt": 570}` |
