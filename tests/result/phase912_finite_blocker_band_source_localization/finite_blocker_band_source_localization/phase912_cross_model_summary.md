# Phase 912 finite blocker band source localization

## Overall

- models: qwen3, glm4, deepseek7b
- band16_source_candidate: 800
- band16_strong_source_candidate: 383
- band32_source_candidate: 730
- band32_strong_source_candidate: 337
- route_eos_top10: 4
- route_eos_top50: 15
- route_rows: 68
- rows: 9076
- source_eos_top1: 0
- source_eos_top10: 592
- source_eos_top5: 3
- source_eos_top50: 2326
- source_margin_nonnegative: 0
- source_rows: 9008
- source_strict_clean_candidate: 0
- strict_clean_candidate: 0

## Model Summaries

| model | rows | source rows | route top10 | route top50 | source top1 | source top5 | source top10 | margin>=0 | band16 candidates | strong band16 | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 2610 | 2592 | 0 | 0 | 0 | 0 | 0 | 0 | 173 | 90 | strong_blocker_band_source_candidates_found |
| glm4 | 2737 | 2720 | 4 | 15 | 0 | 3 | 592 | 0 | 179 | 98 | component_source_scan_reaches_eos_top5 |
| deepseek7b | 3729 | 3696 | 0 | 0 | 0 | 0 | 0 | 0 | 448 | 195 | strong_blocker_band_source_candidates_found |

## Top Sources

| model | layer | bucket | component | factor | rows | top1 | top5 | top10 | margin>=0 | band16 cand | strong16 | median band16 mean delta | median band16 max delta | route blockers |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| glm4 | 0 | early | attention | 0.0 | 17 | 0 | 2 | 13 | 0 | 17 | 6 | -0.88671875 | -1.125 | {'a': 15, ' Fish': 2} |
| glm4 | 4 | early | mlp | 0.0 | 17 | 0 | 1 | 12 | 0 | 3 | 2 | -0.326171875 | -0.625 | {'a': 15, ' Fish': 2} |
| deepseek7b | 27 | late | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 33 | 33 | -7.970703125 | -5.5625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 27 | late | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 33 | 33 | -4.0 | -1.75 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 27 | late | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 32 | 32 | -3.0859375 | -1.9375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 19 | late | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 32 | 27 | -1.04296875 | -1.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 35 | late | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 18 | 18 | -11.92578125 | -12.625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 35 | late | mlp | 0.5 | 18 | 0 | 0 | 0 | 0 | 18 | 18 | -6.193359375 | -6.75 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 25 | late | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 32 | 17 | -1.00390625 | -1.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 27 | late | attention | 0.5 | 33 | 0 | 0 | 0 | 0 | 23 | 17 | -1.11328125 | 0.375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 38 | late | mlp | 0.0 | 17 | 0 | 0 | 0 | 0 | 17 | 17 | -3.460693359375 | -2.90625 | {'a': 15, ' Fish': 2} |
| glm4 | 39 | late | mlp | 0.0 | 17 | 0 | 0 | 0 | 0 | 17 | 17 | -2.5810546875 | -1.5625 | {'a': 15, ' Fish': 2} |
| glm4 | 35 | late | mlp | 0.0 | 17 | 0 | 0 | 0 | 0 | 15 | 15 | -5.32635498046875 | 0.125 | {'a': 15, ' Fish': 2} |
| glm4 | 0 | early | mlp | 0.0 | 17 | 0 | 0 | 0 | 0 | 15 | 15 | -2.10552978515625 | 3.8125 | {'a': 15, ' Fish': 2} |
| glm4 | 37 | late | mlp | 0.0 | 17 | 0 | 0 | 4 | 0 | 15 | 15 | -1.615234375 | -1.8125 | {'a': 15, ' Fish': 2} |
| qwen3 | 34 | late | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 16 | 12 | -1.216796875 | -0.875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 0 | early | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 15 | 12 | -1.146484375 | -1.5 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 25 | late | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 26 | 9 | -0.7265625 | -0.625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 8 | early | mlp | 0.0 | 17 | 0 | 0 | 7 | 0 | 15 | 7 | -0.9140625 | -1.1875 | {'a': 15, ' Fish': 2} |
| qwen3 | 28 | late | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 12 | 7 | -0.58984375 | -0.3125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 35 | late | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 10 | 7 | -0.67578125 | -1.0625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 23 | late | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 28 | 6 | -0.7109375 | -0.6875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 0 | early | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 13 | 6 | -0.31640625 | 2.5 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 35 | late | attention | 0.5 | 18 | 0 | 0 | 0 | 0 | 6 | 6 | -0.12109375 | -0.3125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 11 | middle | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 18 | 5 | -0.56640625 | -0.75 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 1 | early | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 5 | 5 | -0.2421875 | -0.375 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 14 | middle | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 14 | 4 | -0.2890625 | -0.875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 20 | late | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 22 | 3 | -0.58203125 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 23 | late | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 8 | 3 | -0.3828125 | -0.625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 22 | late | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 21 | 2 | -0.625 | -0.1875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 0 | early | attention | 0.5 | 17 | 0 | 0 | 5 | 0 | 2 | 2 | -0.134765625 | -0.3125 | {'a': 15, ' Fish': 2} |
| glm4 | 14 | middle | mlp | 0.0 | 17 | 0 | 0 | 4 | 0 | 2 | 2 | 0.056640625 | 0.0625 | {'a': 15, ' Fish': 2} |
| qwen3 | 27 | late | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 5 | 1 | -0.34765625 | -0.5 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 5 | early | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 2 | 1 | 0.08203125 | 0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 10 | early | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 1 | -0.1171875 | 0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 19 | late | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 18 | 0 | -0.5234375 | -0.5 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 38 | late | mlp | 0.5 | 17 | 0 | 0 | 9 | 0 | 17 | 0 | -0.728515625 | -1.1875 | {'a': 15, ' Fish': 2} |
| deepseek7b | 24 | late | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 16 | 0 | -0.4765625 | -0.625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 37 | late | mlp | 0.5 | 17 | 0 | 0 | 14 | 0 | 15 | 0 | -0.75390625 | -0.9375 | {'a': 15, ' Fish': 2} |
| glm4 | 7 | early | mlp | 0.0 | 17 | 0 | 0 | 1 | 0 | 15 | 0 | -0.619140625 | -1.0 | {'a': 15, ' Fish': 2} |
| deepseek7b | 9 | middle | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 11 | 0 | -0.296875 | 0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 11 | middle | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 10 | 0 | -0.2265625 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 21 | middle | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 9 | 0 | -0.439453125 | -0.5 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 25 | late | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 9 | 0 | -0.30078125 | -0.375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 13 | middle | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 8 | 0 | -0.30859375 | 0.5 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 12 | middle | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 7 | 0 | -0.40234375 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 18 | late | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 7 | 0 | -0.20703125 | 0.375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 37 | late | attention | 0.0 | 17 | 0 | 0 | 0 | 0 | 6 | 0 | -0.419921875 | -0.0625 | {'a': 15, ' Fish': 2} |
| deepseek7b | 25 | late | attention | 0.5 | 33 | 0 | 0 | 0 | 0 | 6 | 0 | -0.359375 | -0.375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 34 | late | attention | 0.5 | 18 | 0 | 0 | 0 | 0 | 6 | 0 | -0.322265625 | -0.0625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 31 | late | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 5 | 0 | -0.40625 | -0.6875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 7 | early | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 5 | 0 | 0.0703125 | 0.0625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 28 | late | mlp | 0.5 | 18 | 0 | 0 | 0 | 0 | 4 | 0 | -0.359375 | -0.375 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 17 | middle | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 4 | 0 | -0.16796875 | 0.875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 21 | middle | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 4 | 0 | -0.0078125 | 0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 25 | late | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 3 | 0 | -0.3046875 | 0.9375 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 13 | middle | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 3 | 0 | -0.234375 | 0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 25 | late | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 3 | 0 | -0.1328125 | -0.375 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 29 | late | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 3 | 0 | -0.1171875 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 14 | middle | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 3 | 0 | -0.02734375 | 0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 19 | middle | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 3 | 0 | 0.015625 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 16 | middle | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 3 | 0 | 0.1328125 | 0.5 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 5 | early | mlp | 0.0 | 17 | 0 | 0 | 8 | 0 | 2 | 0 | -0.3046875 | -0.5 | {'a': 15, ' Fish': 2} |
| qwen3 | 15 | middle | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 2 | 0 | -0.23046875 | -0.1875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 6 | early | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 2 | 0 | -0.19921875 | -0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 13 | middle | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 2 | 0 | -0.19140625 | 0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 6 | early | mlp | 0.0 | 17 | 0 | 0 | 7 | 0 | 2 | 0 | -0.123046875 | -0.3125 | {'a': 15, ' Fish': 2} |
| glm4 | 0 | early | mlp | 0.5 | 17 | 0 | 0 | 0 | 0 | 2 | 0 | -0.109375 | -0.1875 | {'a': 15, ' Fish': 2} |
| deepseek7b | 20 | late | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 2 | 0 | -0.078125 | -0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 15 | middle | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 2 | 0 | -0.04296875 | 0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 17 | middle | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 2 | 0 | -0.0078125 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | 11 | early | mlp | 0.0 | 17 | 0 | 0 | 5 | 0 | 2 | 0 | 0.0703125 | 0.0625 | {'a': 15, ' Fish': 2} |
| deepseek7b | 17 | middle | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 2 | 0 | 0.23828125 | -0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 16 | middle | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | -0.3984375 | -0.5625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 12 | middle | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 1 | 0 | -0.33984375 | -0.75 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 20 | middle | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | -0.33203125 | -0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 12 | middle | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 1 | 0 | -0.23828125 | 0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 24 | late | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | -0.171875 | 0.1875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 30 | late | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | -0.15625 | -0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 8 | early | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | -0.12890625 | -0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 10 | early | mlp | 0.5 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | -0.11328125 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 26 | late | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | -0.095703125 | 0.3125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 18 | late | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 1 | 0 | -0.0390625 | -0.3125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 23 | late | attention | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | -0.0234375 | 0.3125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 24 | late | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | -0.001953125 | -0.5625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 14 | middle | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 1 | 0 | 0.015625 | -0.1875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 15 | middle | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 1 | 0 | 0.18359375 | 1.375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 16 | middle | attention | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.228515625 | -0.375 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 21 | middle | attention | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.21875 | -0.25 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 0 | early | attention | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.20703125 | -0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 23 | late | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.1953125 | -0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 8 | early | mlp | 0.5 | 17 | 0 | 0 | 5 | 0 | 0 | 0 | -0.1796875 | -0.375 | {'a': 15, ' Fish': 2} |
| deepseek7b | 7 | early | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.17578125 | -0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 33 | late | mlp | 0.0 | 17 | 0 | 0 | 5 | 0 | 0 | 0 | -0.171875 | -0.3125 | {'a': 15, ' Fish': 2} |
| deepseek7b | 3 | early | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.16796875 | -0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 23 | late | mlp | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.1640625 | -0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 4 | early | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.1640625 | -0.25 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 16 | middle | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.15234375 | 0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | 5 | early | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.1484375 | -0.1875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 11 | middle | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.1484375 | -0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 4 | early | mlp | 0.5 | 17 | 0 | 0 | 6 | 0 | 0 | 0 | -0.140625 | -0.3125 | {'a': 15, ' Fish': 2} |
| glm4 | 3 | early | mlp | 0.0 | 17 | 0 | 0 | 5 | 0 | 0 | 0 | -0.13671875 | -0.25 | {'a': 15, ' Fish': 2} |
| glm4 | 1 | early | mlp | 0.0 | 17 | 0 | 0 | 4 | 0 | 0 | 0 | -0.134765625 | -0.25 | {'a': 15, ' Fish': 2} |
| deepseek7b | 8 | early | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.1328125 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 7 | early | mlp | 0.5 | 17 | 0 | 0 | 5 | 0 | 0 | 0 | -0.126953125 | -0.1875 | {'a': 15, ' Fish': 2} |
| qwen3 | 0 | early | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.126953125 | -0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 9 | middle | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.125 | 0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 13 | middle | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.12109375 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 12 | middle | attention | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.12109375 | -0.375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 22 | late | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.109375 | -0.1875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 6 | early | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.10546875 | 0.0625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 22 | late | attention | 0.0 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.10546875 | 0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | 37 | late | attention | 0.5 | 17 | 0 | 0 | 8 | 0 | 0 | 0 | -0.1015625 | -0.3125 | {'a': 15, ' Fish': 2} |
| qwen3 | 31 | late | attention | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.1015625 | -0.25 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 3 | early | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.09765625 | 0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | 6 | early | attention | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.09765625 | -0.25 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 29 | late | mlp | 0.0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.095703125 | -0.25 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 0 | early | mlp | 0.0 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.09375 | -0.375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | 26 | late | mlp | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.09375 | 0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | 8 | early | mlp | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.08984375 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |

## Top Buckets

| model | bucket | component | factor | rows | top1 | top5 | top10 | band16 cand | median band16 mean delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | early | attention | 0.0 | 221 | 0 | 2 | 54 | 17 | 0.0 |
| glm4 | early | mlp | 0.0 | 221 | 0 | 1 | 62 | 54 | -0.140625 |
| deepseek7b | late | mlp | 0.0 | 330 | 0 | 0 | 0 | 185 | -0.59375 |
| glm4 | late | mlp | 0.0 | 238 | 0 | 0 | 33 | 64 | -0.0185546875 |
| deepseek7b | late | attention | 0.0 | 330 | 0 | 0 | 0 | 68 | -0.009765625 |
| deepseek7b | late | mlp | 0.5 | 330 | 0 | 0 | 0 | 59 | -0.13671875 |
| qwen3 | late | mlp | 0.0 | 234 | 0 | 0 | 0 | 49 | -0.109375 |
| qwen3 | late | attention | 0.0 | 234 | 0 | 0 | 0 | 39 | -0.087890625 |
| qwen3 | late | mlp | 0.5 | 234 | 0 | 0 | 0 | 22 | -0.03125 |
| qwen3 | early | attention | 0.0 | 198 | 0 | 0 | 0 | 20 | -0.009765625 |
| deepseek7b | late | attention | 0.5 | 330 | 0 | 0 | 0 | 29 | 0.0625 |
| qwen3 | late | attention | 0.5 | 234 | 0 | 0 | 0 | 12 | 0.015625 |
| deepseek7b | early | attention | 0.0 | 297 | 0 | 0 | 0 | 15 | 0.0 |
| deepseek7b | middle | mlp | 0.0 | 297 | 0 | 0 | 0 | 53 | -0.09375 |
| deepseek7b | middle | attention | 0.0 | 297 | 0 | 0 | 0 | 31 | -0.046875 |
| glm4 | early | attention | 0.5 | 221 | 0 | 0 | 55 | 2 | 0.0078125 |
| glm4 | middle | mlp | 0.0 | 221 | 0 | 0 | 55 | 2 | 0.017578125 |
| qwen3 | early | mlp | 0.0 | 198 | 0 | 0 | 0 | 7 | -0.07421875 |
| deepseek7b | early | mlp | 0.0 | 297 | 0 | 0 | 0 | 2 | 0.01953125 |
| qwen3 | middle | mlp | 0.0 | 216 | 0 | 0 | 0 | 13 | -0.078125 |
| deepseek7b | middle | mlp | 0.5 | 297 | 0 | 0 | 0 | 6 | -0.0546875 |
| glm4 | early | mlp | 0.5 | 221 | 0 | 0 | 56 | 2 | -0.048828125 |
| qwen3 | early | mlp | 0.5 | 198 | 0 | 0 | 0 | 1 | -0.03125 |
| qwen3 | middle | mlp | 0.5 | 216 | 0 | 0 | 0 | 0 | -0.025390625 |
| deepseek7b | early | attention | 0.5 | 297 | 0 | 0 | 0 | 0 | -0.015625 |
| deepseek7b | middle | attention | 0.5 | 297 | 0 | 0 | 0 | 0 | -0.00390625 |
| glm4 | middle | attention | 0.0 | 221 | 0 | 0 | 37 | 0 | -0.001953125 |
| glm4 | middle | attention | 0.5 | 221 | 0 | 0 | 48 | 0 | 0.00390625 |
| deepseek7b | early | mlp | 0.5 | 297 | 0 | 0 | 0 | 0 | 0.00390625 |
| qwen3 | middle | attention | 0.5 | 216 | 0 | 0 | 0 | 0 | 0.0078125 |
| qwen3 | early | attention | 0.5 | 198 | 0 | 0 | 0 | 0 | 0.013671875 |
| glm4 | late | mlp | 0.5 | 238 | 0 | 0 | 60 | 32 | 0.015625 |
| glm4 | middle | mlp | 0.5 | 221 | 0 | 0 | 55 | 0 | 0.017578125 |
| glm4 | late | attention | 0.5 | 238 | 0 | 0 | 51 | 0 | 0.03125 |
| glm4 | late | attention | 0.0 | 238 | 0 | 0 | 26 | 6 | 0.037109375 |
| qwen3 | middle | attention | 0.0 | 216 | 0 | 0 | 0 | 10 | 0.0 |
