# Phase 848 Internal Route Gate Discovery (confirm)

- Source: Phase 845 residual rows plus fresh natural activation captures.
- Method: compare external and internal activation-gate residual predictors.
- Boundary: internal gate discovery probe; not geometry closure.

## Model Summary

| model | feature rows | prompts | split | predictor | n | MAE | sign acc | strong F1 | MAE gain vs global |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| qwen3 | 570 | 15 | `in_sample` | `global_combo` | 570 | 0.1612 | 0.5221 | 0.4688 | NA |
| qwen3 | 570 | 15 | `in_sample` | `internal_count_combo` | 570 | 0.1506 | 0.6088 | 0.4667 | 0.0106 |
| qwen3 | 570 | 15 | `in_sample` | `internal_sign_combo` | 570 | 0.1495 | 0.6122 | 0.4667 | 0.0117 |
| qwen3 | 570 | 15 | `in_sample` | `internal_strength_combo` | 570 | 0.1113 | 0.7243 | 0.5926 | 0.0500 |
| qwen3 | 570 | 15 | `in_sample` | `object_combo` | 570 | 0.1124 | 0.7065 | 0.5574 | 0.0488 |
| qwen3 | 570 | 15 | `in_sample` | `prompt_combo` | 570 | 0.1384 | 0.6131 | 0.5556 | 0.0229 |
| qwen3 | 570 | 15 | `object_holdout` | `global_combo` | 570 | 0.1813 | 0.4491 | 0.4688 | NA |
| qwen3 | 570 | 15 | `object_holdout` | `internal_count_combo` | 570 | 0.1757 | 0.5179 | 0.4667 | 0.0056 |
| qwen3 | 570 | 15 | `object_holdout` | `internal_sign_combo` | 570 | 0.1763 | 0.5161 | 0.4667 | 0.0050 |
| qwen3 | 570 | 15 | `object_holdout` | `internal_strength_combo` | 570 | 0.1706 | 0.5421 | 0.4667 | 0.0107 |
| qwen3 | 570 | 15 | `object_holdout` | `object_combo` | 570 | 0.1813 | 0.4491 | 0.4688 | 0.0000 |
| qwen3 | 570 | 15 | `object_holdout` | `prompt_combo` | 570 | 0.1730 | 0.4753 | 0.5556 | 0.0083 |
| qwen3 | 570 | 15 | `prompt_holdout` | `global_combo` | 570 | 0.1802 | 0.4886 | 0.4688 | NA |
| qwen3 | 570 | 15 | `prompt_holdout` | `internal_count_combo` | 570 | 0.1814 | 0.5381 | 0.4242 | -0.0011 |
| qwen3 | 570 | 15 | `prompt_holdout` | `internal_sign_combo` | 570 | 0.1815 | 0.5349 | 0.4242 | -0.0013 |
| qwen3 | 570 | 15 | `prompt_holdout` | `internal_strength_combo` | 570 | 0.1797 | 0.5118 | 0.4267 | 0.0005 |
| qwen3 | 570 | 15 | `prompt_holdout` | `object_combo` | 570 | 0.1686 | 0.5364 | 0.4118 | 0.0116 |
| qwen3 | 570 | 15 | `prompt_holdout` | `prompt_combo` | 570 | 0.1802 | 0.4886 | 0.4688 | 0.0000 |
| glm4 | 570 | 15 | `in_sample` | `global_combo` | 570 | 0.0406 | 0.5702 | 0.0000 | NA |
| glm4 | 570 | 15 | `in_sample` | `internal_count_combo` | 570 | 0.0380 | 0.6102 | 0.0000 | 0.0025 |
| glm4 | 570 | 15 | `in_sample` | `internal_sign_combo` | 570 | 0.0369 | 0.6190 | 0.0000 | 0.0037 |
| glm4 | 570 | 15 | `in_sample` | `internal_strength_combo` | 570 | 0.0315 | 0.6890 | 0.0000 | 0.0091 |
| glm4 | 570 | 15 | `in_sample` | `object_combo` | 570 | 0.0324 | 0.6912 | 0.0000 | 0.0081 |
| glm4 | 570 | 15 | `in_sample` | `prompt_combo` | 570 | 0.0388 | 0.6263 | 0.0000 | 0.0017 |
| glm4 | 570 | 15 | `object_holdout` | `global_combo` | 570 | 0.0447 | 0.4474 | 0.0000 | NA |
| glm4 | 570 | 15 | `object_holdout` | `internal_count_combo` | 570 | 0.0456 | 0.5237 | 0.0000 | -0.0009 |
| glm4 | 570 | 15 | `object_holdout` | `internal_sign_combo` | 570 | 0.0458 | 0.5070 | 0.0000 | -0.0011 |
| glm4 | 570 | 15 | `object_holdout` | `internal_strength_combo` | 570 | 0.0481 | 0.4974 | 0.0000 | -0.0034 |
| glm4 | 570 | 15 | `object_holdout` | `object_combo` | 570 | 0.0447 | 0.4474 | 0.0000 | 0.0000 |
| glm4 | 570 | 15 | `object_holdout` | `prompt_combo` | 570 | 0.0486 | 0.4947 | 0.0000 | -0.0039 |
| glm4 | 570 | 15 | `prompt_holdout` | `global_combo` | 570 | 0.0438 | 0.5175 | 0.0000 | NA |
| glm4 | 570 | 15 | `prompt_holdout` | `internal_count_combo` | 570 | 0.0445 | 0.5360 | 0.0000 | -0.0008 |
| glm4 | 570 | 15 | `prompt_holdout` | `internal_sign_combo` | 570 | 0.0450 | 0.5351 | 0.0000 | -0.0012 |
| glm4 | 570 | 15 | `prompt_holdout` | `internal_strength_combo` | 570 | 0.0462 | 0.5325 | 0.0000 | -0.0024 |
| glm4 | 570 | 15 | `prompt_holdout` | `object_combo` | 570 | 0.0487 | 0.5140 | 0.0000 | -0.0049 |
| glm4 | 570 | 15 | `prompt_holdout` | `prompt_combo` | 570 | 0.0438 | 0.5175 | 0.0000 | 0.0000 |
| deepseek7b | 570 | 15 | `in_sample` | `global_combo` | 570 | 0.0403 | 0.4509 | 0.0000 | NA |
| deepseek7b | 570 | 15 | `in_sample` | `internal_count_combo` | 570 | 0.0399 | 0.4807 | 0.0000 | 0.0005 |
| deepseek7b | 570 | 15 | `in_sample` | `internal_sign_combo` | 570 | 0.0399 | 0.4807 | 0.0000 | 0.0005 |
| deepseek7b | 570 | 15 | `in_sample` | `internal_strength_combo` | 570 | 0.0394 | 0.5009 | 0.0000 | 0.0010 |
| deepseek7b | 570 | 15 | `in_sample` | `object_combo` | 570 | 0.0341 | 0.6443 | 0.0000 | 0.0062 |
| deepseek7b | 570 | 15 | `in_sample` | `prompt_combo` | 570 | 0.0369 | 0.5644 | 0.0000 | 0.0034 |
| deepseek7b | 570 | 15 | `object_holdout` | `global_combo` | 570 | 0.0432 | 0.4482 | 0.0000 | NA |
| deepseek7b | 570 | 15 | `object_holdout` | `internal_count_combo` | 570 | 0.0441 | 0.4605 | 0.0000 | -0.0009 |
| deepseek7b | 570 | 15 | `object_holdout` | `internal_sign_combo` | 570 | 0.0441 | 0.4605 | 0.0000 | -0.0009 |
| deepseek7b | 570 | 15 | `object_holdout` | `internal_strength_combo` | 570 | 0.0474 | 0.4583 | 0.0000 | -0.0042 |
| deepseek7b | 570 | 15 | `object_holdout` | `object_combo` | 570 | 0.0432 | 0.4482 | 0.0000 | 0.0000 |
| deepseek7b | 570 | 15 | `object_holdout` | `prompt_combo` | 570 | 0.0462 | 0.4446 | 0.0000 | -0.0030 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `global_combo` | 570 | 0.0448 | 0.4208 | 0.0000 | NA |
| deepseek7b | 570 | 15 | `prompt_holdout` | `internal_count_combo` | 570 | 0.0444 | 0.4549 | 0.0000 | 0.0004 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `internal_sign_combo` | 570 | 0.0444 | 0.4549 | 0.0000 | 0.0004 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `internal_strength_combo` | 570 | 0.0455 | 0.4654 | 0.0000 | -0.0007 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `object_combo` | 570 | 0.0512 | 0.4430 | 0.0000 | -0.0064 |
| deepseek7b | 570 | 15 | `prompt_holdout` | `prompt_combo` | 570 | 0.0448 | 0.4208 | 0.0000 | 0.0000 |

## Feature Summary

| model | sign patterns | neg count | residual classes | mean abs sum |
|---|---|---|---|---:|
| qwen3 | `{"+-": 200, "++": 58, "-+": 44, "--": 148, "+-+": 6, "+--": 26, "++-": 74, "+++": 14}` | `{"1": 324, "0": 72, "2": 174}` | `{"additive": 536, "synergy": 19, "antagonistic": 15}` | 17.5632 |
| glm4 | `{"++": 278, "+-": 94, "-+": 70, "++-": 32, "+++": 80, "--": 8, "-+-": 2, "-++": 6}` | `{"0": 358, "1": 202, "2": 10}` | `{"additive": 570}` | 5.2971 |
| deepseek7b | `{"--": 212, "-+": 164, "++": 22, "+-": 52, "---": 68, "--+": 52}` | `{"2": 264, "1": 216, "0": 22, "3": 68}` | `{"additive": 567, "synergy": 3}` | 157.7471 |
