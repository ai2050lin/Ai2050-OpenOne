# Phase 847 Context Gated Residual Audit (main)

- Source: Phase 845 pair/triplet interaction residual rows.
- Method: compare global residual means with prompt/object conditioned residual means.
- Boundary: residual transfer diagnostics; not token closure.

## Residual Prediction

| model | split | predictor | n | MAE | RMSE | sign acc | class acc | mean MAE gain vs global |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `in_sample` | `global_combo` | 304 | 0.1799 | 0.3745 | 0.5921 | 0.9243 | NA |
| qwen3 | `in_sample` | `object_combo` | 304 | 0.0800 | 0.1354 | 0.7801 | 0.9671 | 0.1000 |
| qwen3 | `in_sample` | `object_prompt_combo` | 304 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1799 |
| qwen3 | `in_sample` | `prompt_combo` | 304 | 0.1763 | 0.3707 | 0.6054 | 0.9243 | 0.0037 |
| qwen3 | `object_holdout` | `global_combo` | 304 | 0.2302 | 0.4849 | 0.5050 | 0.9243 | NA |
| qwen3 | `object_holdout` | `object_combo` | 304 | 0.2302 | 0.4849 | 0.5050 | 0.9243 | 0.0000 |
| qwen3 | `object_holdout` | `object_prompt_combo` | 304 | 0.2302 | 0.4849 | 0.5050 | 0.9243 | 0.0000 |
| qwen3 | `object_holdout` | `prompt_combo` | 304 | 0.2350 | 0.4942 | 0.4796 | 0.9243 | -0.0048 |
| qwen3 | `prompt_holdout` | `global_combo` | 304 | 0.1960 | 0.3859 | 0.5452 | 0.9243 | NA |
| qwen3 | `prompt_holdout` | `object_combo` | 304 | 0.1600 | 0.2708 | 0.5603 | 0.9342 | 0.0361 |
| qwen3 | `prompt_holdout` | `object_prompt_combo` | 304 | 0.1960 | 0.3859 | 0.5452 | 0.9243 | 0.0000 |
| qwen3 | `prompt_holdout` | `prompt_combo` | 304 | 0.1960 | 0.3859 | 0.5452 | 0.9243 | 0.0000 |
| glm4 | `in_sample` | `global_combo` | 304 | 0.0381 | 0.0506 | 0.5888 | 1.0000 | NA |
| glm4 | `in_sample` | `object_combo` | 304 | 0.0240 | 0.0290 | 0.7796 | 1.0000 | 0.0141 |
| glm4 | `in_sample` | `object_prompt_combo` | 304 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0381 |
| glm4 | `in_sample` | `prompt_combo` | 304 | 0.0368 | 0.0490 | 0.6020 | 1.0000 | 0.0014 |
| glm4 | `object_holdout` | `global_combo` | 304 | 0.0469 | 0.0624 | 0.4112 | 1.0000 | NA |
| glm4 | `object_holdout` | `object_combo` | 304 | 0.0469 | 0.0624 | 0.4112 | 1.0000 | 0.0000 |
| glm4 | `object_holdout` | `object_prompt_combo` | 304 | 0.0469 | 0.0624 | 0.4112 | 1.0000 | 0.0000 |
| glm4 | `object_holdout` | `prompt_combo` | 304 | 0.0490 | 0.0654 | 0.4243 | 1.0000 | -0.0021 |
| glm4 | `prompt_holdout` | `global_combo` | 304 | 0.0416 | 0.0549 | 0.5296 | 1.0000 | NA |
| glm4 | `prompt_holdout` | `object_combo` | 304 | 0.0480 | 0.0579 | 0.5592 | 1.0000 | -0.0064 |
| glm4 | `prompt_holdout` | `object_prompt_combo` | 304 | 0.0416 | 0.0549 | 0.5296 | 1.0000 | 0.0000 |
| glm4 | `prompt_holdout` | `prompt_combo` | 304 | 0.0416 | 0.0549 | 0.5296 | 1.0000 | 0.0000 |
| deepseek7b | `in_sample` | `global_combo` | 304 | 0.0536 | 0.0785 | 0.5526 | 0.9901 | NA |
| deepseek7b | `in_sample` | `object_combo` | 304 | 0.0463 | 0.0642 | 0.6689 | 0.9901 | 0.0073 |
| deepseek7b | `in_sample` | `object_prompt_combo` | 304 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0536 |
| deepseek7b | `in_sample` | `prompt_combo` | 304 | 0.0456 | 0.0693 | 0.6589 | 0.9901 | 0.0080 |
| deepseek7b | `object_holdout` | `global_combo` | 304 | 0.0589 | 0.0880 | 0.5364 | 0.9901 | NA |
| deepseek7b | `object_holdout` | `object_combo` | 304 | 0.0589 | 0.0880 | 0.5364 | 0.9901 | 0.0000 |
| deepseek7b | `object_holdout` | `object_prompt_combo` | 304 | 0.0589 | 0.0880 | 0.5364 | 0.9901 | 0.0000 |
| deepseek7b | `object_holdout` | `prompt_combo` | 304 | 0.0608 | 0.0924 | 0.5217 | 0.9901 | -0.0019 |
| deepseek7b | `prompt_holdout` | `global_combo` | 304 | 0.0727 | 0.1011 | 0.4503 | 0.9901 | NA |
| deepseek7b | `prompt_holdout` | `object_combo` | 304 | 0.0925 | 0.1283 | 0.3378 | 0.9803 | -0.0198 |
| deepseek7b | `prompt_holdout` | `object_prompt_combo` | 304 | 0.0727 | 0.1011 | 0.4503 | 0.9901 | 0.0000 |
| deepseek7b | `prompt_holdout` | `prompt_combo` | 304 | 0.0727 | 0.1011 | 0.4503 | 0.9901 | 0.0000 |

## Context Strength

| model | residual rows | prompt shift | object shift | object+prompt shift | terms |
|---|---:|---:|---:|---:|---|
| qwen3 | 304 | 0.0410 | 0.1651 | 0.1799 | `{"global": 38, "prompt": 76, "object": 152, "object_prompt": 304}` |
| glm4 | 304 | 0.0094 | 0.0315 | 0.0381 | `{"global": 38, "prompt": 76, "object": 152, "object_prompt": 304}` |
| deepseek7b | 304 | 0.0310 | 0.0300 | 0.0536 | `{"global": 38, "prompt": 76, "object": 152, "object_prompt": 304}` |
