# Phase 848 Internal Route Gate Discovery (main)

- Source: Phase 845 residual rows plus fresh natural activation captures.
- Method: compare external and internal activation-gate residual predictors.
- Boundary: internal gate discovery probe; not geometry closure.

## Model Summary

| model | feature rows | prompts | split | predictor | n | MAE | sign acc | strong F1 | MAE gain vs global |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| qwen3 | 304 | 8 | `in_sample` | `global_combo` | 304 | 0.1799 | 0.5921 | 0.5417 | NA |
| qwen3 | 304 | 8 | `in_sample` | `internal_count_combo` | 304 | 0.1393 | 0.6894 | 0.6154 | 0.0406 |
| qwen3 | 304 | 8 | `in_sample` | `internal_sign_combo` | 304 | 0.1377 | 0.6979 | 0.6154 | 0.0422 |
| qwen3 | 304 | 8 | `in_sample` | `internal_strength_combo` | 304 | 0.1144 | 0.7621 | 0.7119 | 0.0655 |
| qwen3 | 304 | 8 | `in_sample` | `object_combo` | 304 | 0.0800 | 0.7801 | 0.8438 | 0.1000 |
| qwen3 | 304 | 8 | `in_sample` | `prompt_combo` | 304 | 0.1763 | 0.6054 | 0.5417 | 0.0037 |
| qwen3 | 304 | 8 | `object_holdout` | `global_combo` | 304 | 0.2302 | 0.5050 | 0.5417 | NA |
| qwen3 | 304 | 8 | `object_holdout` | `internal_count_combo` | 304 | 0.2168 | 0.5492 | 0.5098 | 0.0135 |
| qwen3 | 304 | 8 | `object_holdout` | `internal_sign_combo` | 304 | 0.2177 | 0.5608 | 0.5098 | 0.0126 |
| qwen3 | 304 | 8 | `object_holdout` | `internal_strength_combo` | 304 | 0.2217 | 0.5700 | 0.5357 | 0.0085 |
| qwen3 | 304 | 8 | `object_holdout` | `object_combo` | 304 | 0.2302 | 0.5050 | 0.5417 | 0.0000 |
| qwen3 | 304 | 8 | `object_holdout` | `prompt_combo` | 304 | 0.2350 | 0.4796 | 0.5417 | -0.0048 |
| qwen3 | 304 | 8 | `prompt_holdout` | `global_combo` | 304 | 0.1960 | 0.5452 | 0.5417 | NA |
| qwen3 | 304 | 8 | `prompt_holdout` | `internal_count_combo` | 304 | 0.1823 | 0.5685 | 0.4667 | 0.0137 |
| qwen3 | 304 | 8 | `prompt_holdout` | `internal_sign_combo` | 304 | 0.1817 | 0.5782 | 0.4667 | 0.0143 |
| qwen3 | 304 | 8 | `prompt_holdout` | `internal_strength_combo` | 304 | 0.1794 | 0.5764 | 0.5714 | 0.0166 |
| qwen3 | 304 | 8 | `prompt_holdout` | `object_combo` | 304 | 0.1600 | 0.5603 | 0.6875 | 0.0361 |
| qwen3 | 304 | 8 | `prompt_holdout` | `prompt_combo` | 304 | 0.1960 | 0.5452 | 0.5417 | 0.0000 |
| glm4 | 304 | 8 | `in_sample` | `global_combo` | 304 | 0.0381 | 0.5888 | 0.0000 | NA |
| glm4 | 304 | 8 | `in_sample` | `internal_count_combo` | 304 | 0.0365 | 0.5954 | 0.0000 | 0.0016 |
| glm4 | 304 | 8 | `in_sample` | `internal_sign_combo` | 304 | 0.0366 | 0.5954 | 0.0000 | 0.0015 |
| glm4 | 304 | 8 | `in_sample` | `internal_strength_combo` | 304 | 0.0297 | 0.7114 | 0.0000 | 0.0084 |
| glm4 | 304 | 8 | `in_sample` | `object_combo` | 304 | 0.0240 | 0.7796 | 0.0000 | 0.0141 |
| glm4 | 304 | 8 | `in_sample` | `prompt_combo` | 304 | 0.0368 | 0.6020 | 0.0000 | 0.0014 |
| glm4 | 304 | 8 | `object_holdout` | `global_combo` | 304 | 0.0469 | 0.4112 | 0.0000 | NA |
| glm4 | 304 | 8 | `object_holdout` | `internal_count_combo` | 304 | 0.0494 | 0.4079 | 0.0000 | -0.0025 |
| glm4 | 304 | 8 | `object_holdout` | `internal_sign_combo` | 304 | 0.0497 | 0.4026 | 0.0000 | -0.0028 |
| glm4 | 304 | 8 | `object_holdout` | `internal_strength_combo` | 304 | 0.0533 | 0.4106 | 0.0000 | -0.0064 |
| glm4 | 304 | 8 | `object_holdout` | `object_combo` | 304 | 0.0469 | 0.4112 | 0.0000 | 0.0000 |
| glm4 | 304 | 8 | `object_holdout` | `prompt_combo` | 304 | 0.0490 | 0.4243 | 0.0000 | -0.0021 |
| glm4 | 304 | 8 | `prompt_holdout` | `global_combo` | 304 | 0.0416 | 0.5296 | 0.0000 | NA |
| glm4 | 304 | 8 | `prompt_holdout` | `internal_count_combo` | 304 | 0.0432 | 0.4967 | 0.0000 | -0.0016 |
| glm4 | 304 | 8 | `prompt_holdout` | `internal_sign_combo` | 304 | 0.0433 | 0.4950 | 0.0000 | -0.0017 |
| glm4 | 304 | 8 | `prompt_holdout` | `internal_strength_combo` | 304 | 0.0462 | 0.4818 | 0.0000 | -0.0045 |
| glm4 | 304 | 8 | `prompt_holdout` | `object_combo` | 304 | 0.0480 | 0.5592 | 0.0000 | -0.0064 |
| glm4 | 304 | 8 | `prompt_holdout` | `prompt_combo` | 304 | 0.0416 | 0.5296 | 0.0000 | 0.0000 |
| deepseek7b | 304 | 8 | `in_sample` | `global_combo` | 304 | 0.0536 | 0.5526 | 0.0000 | NA |
| deepseek7b | 304 | 8 | `in_sample` | `internal_count_combo` | 304 | 0.0515 | 0.5880 | 0.0000 | 0.0021 |
| deepseek7b | 304 | 8 | `in_sample` | `internal_sign_combo` | 304 | 0.0515 | 0.5880 | 0.0000 | 0.0021 |
| deepseek7b | 304 | 8 | `in_sample` | `internal_strength_combo` | 304 | 0.0485 | 0.6033 | 0.0000 | 0.0051 |
| deepseek7b | 304 | 8 | `in_sample` | `object_combo` | 304 | 0.0463 | 0.6689 | 0.0000 | 0.0073 |
| deepseek7b | 304 | 8 | `in_sample` | `prompt_combo` | 304 | 0.0456 | 0.6589 | 0.0000 | 0.0080 |
| deepseek7b | 304 | 8 | `object_holdout` | `global_combo` | 304 | 0.0589 | 0.5364 | 0.0000 | NA |
| deepseek7b | 304 | 8 | `object_holdout` | `internal_count_combo` | 304 | 0.0596 | 0.5364 | 0.0000 | -0.0007 |
| deepseek7b | 304 | 8 | `object_holdout` | `internal_sign_combo` | 304 | 0.0596 | 0.5364 | 0.0000 | -0.0007 |
| deepseek7b | 304 | 8 | `object_holdout` | `internal_strength_combo` | 304 | 0.0622 | 0.5033 | 0.0000 | -0.0033 |
| deepseek7b | 304 | 8 | `object_holdout` | `object_combo` | 304 | 0.0589 | 0.5364 | 0.0000 | 0.0000 |
| deepseek7b | 304 | 8 | `object_holdout` | `prompt_combo` | 304 | 0.0608 | 0.5217 | 0.0000 | -0.0019 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `global_combo` | 304 | 0.0727 | 0.4503 | 0.0000 | NA |
| deepseek7b | 304 | 8 | `prompt_holdout` | `internal_count_combo` | 304 | 0.0719 | 0.4618 | 0.0000 | 0.0008 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `internal_sign_combo` | 304 | 0.0719 | 0.4618 | 0.0000 | 0.0008 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `internal_strength_combo` | 304 | 0.0722 | 0.4633 | 0.0000 | 0.0006 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `object_combo` | 304 | 0.0925 | 0.3378 | 0.0000 | -0.0198 |
| deepseek7b | 304 | 8 | `prompt_holdout` | `prompt_combo` | 304 | 0.0727 | 0.4503 | 0.0000 | 0.0000 |

## Feature Summary

| model | sign patterns | neg count | residual classes | mean abs sum |
|---|---|---|---|---:|
| qwen3 | `{"+-": 106, "++": 30, "-+": 24, "--": 80, "+-+": 2, "+--": 14, "++-": 40, "+++": 8}` | `{"1": 172, "0": 38, "2": 94}` | `{"additive": 272, "synergy": 17, "antagonistic": 15}` | 23.1012 |
| glm4 | `{"++": 152, "+-": 58, "-+": 28, "++-": 18, "+++": 46, "--": 2}` | `{"0": 198, "1": 104, "2": 2}` | `{"additive": 304}` | 5.1481 |
| deepseek7b | `{"--": 104, "-+": 92, "++": 14, "+-": 30, "---": 34, "--+": 30}` | `{"2": 134, "1": 122, "0": 14, "3": 34}` | `{"additive": 301, "synergy": 3}` | 151.3390 |
