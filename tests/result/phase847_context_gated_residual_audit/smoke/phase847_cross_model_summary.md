# Phase 847 Context Gated Residual Audit (smoke)

- Source: Phase 845 pair/triplet interaction residual rows.
- Method: compare global residual means with prompt/object conditioned residual means.
- Boundary: residual transfer diagnostics; not token closure.

## Residual Prediction

| model | split | predictor | n | MAE | RMSE | sign acc | class acc | mean MAE gain vs global |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | `in_sample` | `global_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| qwen3 | `in_sample` | `object_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | `in_sample` | `object_prompt_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| qwen3 | `in_sample` | `prompt_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| glm4 | `in_sample` | `global_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| glm4 | `in_sample` | `object_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| glm4 | `in_sample` | `object_prompt_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| glm4 | `in_sample` | `prompt_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| deepseek7b | `in_sample` | `global_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| deepseek7b | `in_sample` | `object_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| deepseek7b | `in_sample` | `object_prompt_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| deepseek7b | `in_sample` | `prompt_combo` | 12 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |

## Context Strength

| model | residual rows | prompt shift | object shift | object+prompt shift | terms |
|---|---:|---:|---:|---:|---|
| qwen3 | 12 | 0.0000 | 0.0000 | 0.0000 | `{"global": 12, "prompt": 12, "object": 12, "object_prompt": 12}` |
| glm4 | 12 | 0.0000 | 0.0000 | 0.0000 | `{"global": 12, "prompt": 12, "object": 12, "object_prompt": 12}` |
| deepseek7b | 12 | 0.0000 | 0.0000 | 0.0000 | `{"global": 12, "prompt": 12, "object": 12, "object_prompt": 12}` |
