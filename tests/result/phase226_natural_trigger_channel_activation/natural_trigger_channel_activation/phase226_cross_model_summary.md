# Phase 226 natural trigger to channel activation

activation_rows: 1728
hidden_rows: 2304
readout_rows: 576
channel_score_rows: 1728

## Activation summary

| spec | group | variant | step | layer | K | axis | delta | success closer |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4_repeat_l30_natural_trigger | success | no_answer_anchor | 1 | 30 | 4 | -0.6807 | -1.6599 | 0 |
| glm4_repeat_l30_natural_trigger | success | no_answer_anchor | 1 | 30 | 16 | -0.3347 | -1.3872 | 0 |
| glm4_repeat_l30_natural_trigger | success | no_answer_anchor | 1 | 30 | 64 | 0.0092 | -1.0138 | 0 |
| glm4_repeat_l30_natural_trigger | drift | no_answer_anchor | 1 | 30 | 4 | -0.6748 | -0.8954 | 0 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 29 | 64 | 0.7352 | 0.7388 | 6 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 24 | 64 | 0.7240 | 0.7240 | 2 |
| qwen3_explain_l29_natural_trigger | success | no_answer_anchor | 1 | 29 | 4 | 0.3200 | -0.6877 | 6 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 29 | 16 | 0.6337 | 0.6378 | 6 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 24 | 16 | 0.6035 | 0.6035 | 2 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 24 | 4 | 0.5988 | 0.5988 | 2 |
| glm4_repeat_l30_natural_trigger | drift | no_answer_anchor | 1 | 30 | 16 | -0.3417 | -0.5685 | 0 |
| glm4_repeat_l30_natural_trigger | success | no_instruction | 1 | 30 | 16 | 0.4885 | -0.5641 | 5 |
| glm4_repeat_l30_natural_trigger | success | explain_instruction | 2 | 30 | 4 | 0.4309 | -0.5601 | 6 |
| glm4_repeat_l30_natural_trigger | success | short_answer_instruction | 1 | 30 | 64 | 0.4731 | -0.5499 | 5 |
| glm4_repeat_l30_natural_trigger | success | short_answer_instruction | 2 | 30 | 4 | 0.4503 | -0.5407 | 6 |
| glm4_repeat_l30_natural_trigger | drift | short_answer_instruction | 1 | 30 | 16 | -0.3065 | -0.5333 | 0 |
| glm4_repeat_l30_natural_trigger | success | explain_instruction | 2 | 30 | 16 | 0.5131 | -0.5167 | 6 |
| glm4_repeat_l30_natural_trigger | success | short_answer_instruction | 1 | 30 | 16 | 0.5412 | -0.5114 | 5 |
| glm4_repeat_l30_natural_trigger | success | no_instruction | 2 | 30 | 4 | 0.5022 | -0.4889 | 6 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 29 | 4 | 0.3203 | 0.4749 | 6 |
| glm4_repeat_l30_natural_trigger | success | no_instruction | 1 | 30 | 64 | 0.5605 | -0.4625 | 5 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 2 | 29 | 64 | 0.3673 | 0.4150 | 0 |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 3 | 24 | 4 | 0.5900 | -0.4100 | 6 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 1 | 24 | 16 | 0.4027 | 0.4027 | 0 |
| glm4_repeat_l30_natural_trigger | success | explain_instruction | 2 | 30 | 64 | 0.6021 | -0.4022 | 6 |
| qwen3_explain_l29_natural_trigger | drift | short_answer_instruction | 1 | 29 | 4 | -0.5491 | -0.3944 | 2 |
| glm4_repeat_l30_natural_trigger | success | short_answer_instruction | 2 | 30 | 16 | 0.6396 | -0.3901 | 6 |
| qwen3_explain_l29_natural_trigger | success | short_answer_instruction | 1 | 29 | 4 | 0.6192 | -0.3885 | 4 |
| glm4_repeat_l30_natural_trigger | drift | no_instruction | 1 | 30 | 16 | -0.1561 | -0.3829 | 0 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 24 | 16 | 0.3459 | 0.3459 | 0 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 24 | 64 | 0.3451 | 0.3451 | 0 |
| glm4_repeat_l30_natural_trigger | success | short_answer_instruction | 2 | 30 | 64 | 0.6676 | -0.3367 | 6 |
| qwen3_explain_l29_natural_trigger | success | no_instruction | 1 | 29 | 4 | 0.6727 | -0.3350 | 6 |
| glm4_repeat_l30_natural_trigger | success | no_instruction | 2 | 30 | 64 | 0.6761 | -0.3282 | 6 |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 1 | 29 | 16 | 0.3230 | 0.3271 | 2 |
| qwen3_explain_l29_natural_trigger | success | no_answer_anchor | 1 | 29 | 16 | 0.6458 | -0.3263 | 6 |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 1 | 24 | 16 | 0.6768 | -0.3232 | 6 |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 1 | 29 | 64 | 0.3157 | 0.3193 | 2 |
| glm4_repeat_l30_natural_trigger | success | explain_instruction | 1 | 30 | 64 | 0.7105 | -0.3125 | 6 |
| glm4_repeat_l30_natural_trigger | success | comma_removed | 1 | 30 | 4 | 0.6770 | -0.3021 | 6 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 2 | 29 | 4 | 0.1380 | 0.3007 | 0 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 1 | 24 | 64 | 0.2941 | 0.2941 | 0 |
| qwen3_explain_l29_natural_trigger | success | no_instruction | 3 | 29 | 4 | 1.3012 | 0.2865 | 6 |
| deepseek7b_explain_l24_natural_trigger | drift | short_answer_instruction | 1 | 24 | 64 | 0.2775 | 0.2775 | 0 |
| glm4_repeat_l30_natural_trigger | drift | short_answer_instruction | 1 | 30 | 64 | -0.0581 | -0.2760 | 0 |
| glm4_repeat_l30_natural_trigger | success | short_answer_instruction | 1 | 30 | 4 | 0.7180 | -0.2611 | 6 |
| glm4_repeat_l30_natural_trigger | success | no_instruction | 2 | 30 | 16 | 0.7687 | -0.2611 | 6 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 2 | 29 | 4 | -0.4228 | -0.2601 | 0 |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 1 | 24 | 4 | 0.7466 | -0.2534 | 6 |
| glm4_repeat_l30_natural_trigger | success | short_answer_instruction | 3 | 30 | 4 | 1.2267 | 0.2528 | 4 |
| glm4_repeat_l30_natural_trigger | drift | short_answer_instruction | 1 | 30 | 4 | -0.0251 | -0.2458 | 0 |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 2 | 24 | 16 | 0.7575 | -0.2425 | 6 |
| glm4_repeat_l30_natural_trigger | drift | short_answer_instruction | 2 | 30 | 16 | -0.1308 | -0.2377 | 0 |
| qwen3_explain_l29_natural_trigger | success | repeat_instruction | 2 | 29 | 16 | 0.6837 | -0.2357 | 4 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 24 | 4 | 0.2355 | 0.2355 | 0 |
| glm4_repeat_l30_natural_trigger | drift | no_answer_anchor | 1 | 30 | 64 | -0.0147 | -0.2326 | 0 |
| qwen3_explain_l29_natural_trigger | success | no_instruction | 3 | 29 | 16 | 1.2333 | 0.2307 | 6 |
| qwen3_explain_l29_natural_trigger | success | short_answer_instruction | 3 | 29 | 4 | 1.2409 | 0.2261 | 6 |
| glm4_repeat_l30_natural_trigger | drift | short_answer_instruction | 2 | 30 | 64 | -0.0374 | -0.2239 | 0 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 1 | 29 | 16 | -0.2274 | -0.2233 | 0 |

## Hidden summary

| spec | group | variant | step | layer | projection delta |
| --- | --- | --- | ---: | ---: | ---: |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 26 | 219.5280 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 26 | 146.7012 |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 1 | 24 | -102.4149 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 24 | 97.1485 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 33 | 89.1919 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 2 | 26 | 80.3140 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 1 | 26 | 77.5213 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 31 | 73.0376 |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 2 | 26 | -72.5512 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 3 | 33 | 63.5865 |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 2 | 24 | -56.1423 |
| deepseek7b_explain_l24_natural_trigger | drift | short_answer_instruction | 1 | 24 | -55.2653 |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 2 | 26 | 54.9299 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 2 | 24 | 54.7182 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 27 | 50.0152 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 3 | 26 | 49.9923 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 20 | 49.2188 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 3 | 26 | 48.3859 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 29 | 46.3266 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 24 | 45.7630 |
| qwen3_explain_l29_natural_trigger | success | repeat_instruction | 3 | 33 | -43.1403 |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 3 | 33 | 42.6386 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 3 | 31 | 42.4876 |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 2 | 26 | -41.1865 |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 1 | 26 | -40.9305 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 2 | 20 | 38.8744 |
| qwen3_explain_l29_natural_trigger | drift | short_answer_instruction | 3 | 33 | 38.5367 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 3 | 24 | 38.2713 |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 1 | 24 | -38.1265 |
| qwen3_explain_l29_natural_trigger | success | repeat_instruction | 3 | 31 | -37.9783 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 3 | 29 | 36.0306 |
| deepseek7b_explain_l24_natural_trigger | drift | short_answer_instruction | 2 | 26 | 35.9809 |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 1 | 24 | 34.7322 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 2 | 33 | 33.5070 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 20 | 33.3249 |
| deepseek7b_explain_l24_natural_trigger | success | no_instruction | 1 | 20 | -32.8403 |
| glm4_repeat_l30_natural_trigger | success | no_answer_anchor | 1 | 32 | -32.7680 |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 3 | 24 | 32.3437 |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 2 | 24 | -31.6751 |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 1 | 33 | 29.8790 |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 3 | 26 | -29.3991 |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 3 | 33 | 29.1102 |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 2 | 24 | 28.5128 |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 2 | 20 | 28.2887 |
| qwen3_explain_l29_natural_trigger | success | no_answer_anchor | 1 | 33 | -28.2540 |
| qwen3_explain_l29_natural_trigger | success | repeat_instruction | 2 | 33 | -27.8990 |
| glm4_repeat_l30_natural_trigger | success | no_answer_anchor | 1 | 30 | -27.8257 |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 1 | 31 | 27.3107 |
| qwen3_explain_l29_natural_trigger | success | repeat_instruction | 3 | 29 | -27.2162 |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 2 | 31 | 26.0935 |

## Readout summary

| spec | group | variant | step | top changed | rank delta | prose delta | echo delta | top tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b_explain_l24_natural_trigger | success | because_removed | 1 | 2 | -20206.0000 | 0.0312 | 1.3229 | {' Cup': 2, ' Glass': 2, ' The': 2} |
| deepseek7b_explain_l24_natural_trigger | success | short_answer_instruction | 1 | 4 | 18558.6667 | 0.0104 | 1.3438 | {' A': 2, ' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 2 | 0 | 16015.0000 | 2.0625 | -1.9062 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 1 | 0 | 14899.6667 | -0.9583 | 0.1875 | {' Glass': 2, ' The': 4} |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 2 | 0 | 14899.0000 | 0.7500 | 0.5312 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 1 | 2 | -14631.0000 | -0.4688 | -0.4688 | {' The': 2} |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 1 | 2 | 13643.6667 | -0.1562 | -1.5938 | {' ': 2, ' The': 4} |
| deepseek7b_explain_l24_natural_trigger | success | no_instruction | 1 | 4 | 12494.0000 | 0.4271 | 1.9062 | {' Cup': 2, ' Dog': 2, ' Glass': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | because_removed | 1 | 0 | -7688.0000 | 0.3750 | 1.0625 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | short_answer_instruction | 1 | 0 | -6857.0000 | -0.7500 | 0.8750 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | because_removed | 2 | 0 | 6678.0000 | -1.1562 | -1.7188 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | success | because_removed | 2 | 0 | -5501.6667 | -1.5208 | -0.7083 | {' cup': 2, ' dog': 2, ' is': 2} |
| glm4_repeat_l30_natural_trigger | success | no_answer_anchor | 1 | 6 | -5285.6667 | 1.7344 | -2.2500 | {' For': 6} |
| glm4_repeat_l30_natural_trigger | drift | no_answer_anchor | 1 | 6 | -5263.6667 | 0.9948 | -3.1771 | {' For': 6} |
| deepseek7b_explain_l24_natural_trigger | drift | short_answer_instruction | 2 | 0 | 4927.0000 | -1.2188 | -1.8750 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 2 | 0 | -4855.0000 | 0.3125 | 0.5000 | {'orses': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | repeat_instruction | 1 | 0 | 4361.0000 | -0.7812 | 0.4062 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 3 | 0 | 4255.3333 | -1.6875 | 0.3958 | {' is': 4, ' used': 2} |
| deepseek7b_explain_l24_natural_trigger | success | no_answer_anchor | 2 | 4 | 3072.0000 | 0.6458 | -3.7396 | {' answer': 2, ' is': 2, ' reason': 2} |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 1 | 0 | -2909.0000 | 0.5312 | 1.8438 | {' H': 2} |
| deepseek7b_explain_l24_natural_trigger | success | no_instruction | 3 | 0 | 2031.3333 | 0.3333 | 0.7083 | {' is': 4, ' used': 2} |
| glm4_repeat_l30_natural_trigger | drift | short_answer_instruction | 1 | 6 | -1390.0000 | 2.5052 | 2.3646 | {' Dogs': 1, ' H': 1, ' The': 1, ' They': 1, ' Wood': 2} |
| deepseek7b_explain_l24_natural_trigger | success | short_answer_instruction | 2 | 0 | -1260.0000 | -1.3333 | -0.2292 | {' cup': 2, ' dog': 2, ' is': 2} |
| deepseek7b_explain_l24_natural_trigger | success | short_answer_instruction | 3 | 0 | 1119.0000 | 0.9375 | 0.5625 | {' is': 4, ' used': 2} |
| deepseek7b_explain_l24_natural_trigger | success | because_removed | 3 | 0 | -1004.6667 | -0.3333 | -0.1667 | {' is': 4, ' used': 2} |
| glm4_repeat_l30_natural_trigger | drift | no_instruction | 1 | 4 | -862.6667 | 2.1172 | 1.2526 | {' Horse': 1, ' The': 3, ' White': 1, ' white': 1} |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 2 | 0 | -770.3333 | 1.2917 | 1.5625 | {' cup': 2, ' dog': 2, ' is': 2} |
| qwen3_explain_l29_natural_trigger | success | no_answer_anchor | 1 | 6 | -740.6667 | -5.3750 | -10.7708 | {' Then': 6} |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 1 | 6 | -691.6667 | -5.6667 | -9.0000 | {' The': 2, ' Then': 4} |
| deepseek7b_explain_l24_natural_trigger | drift | no_instruction | 3 | 0 | 649.0000 | 1.8750 | -2.1875 | {' are': 2} |
| glm4_repeat_l30_natural_trigger | drift | no_instruction | 2 | 5 | -541.1667 | -1.2917 | -1.9167 | {'\n': 4, ' is': 1, ' used': 1} |
| glm4_repeat_l30_natural_trigger | drift | explain_instruction | 1 | 2 | -487.8333 | 1.8802 | 1.4896 | {' Dog': 1, ' Horse': 1, ' The': 2, ' White': 2} |
| deepseek7b_explain_l24_natural_trigger | success | repeat_instruction | 3 | 0 | 442.0000 | 0.3542 | 2.7292 | {' is': 4, ' used': 2} |
| qwen3_explain_l29_natural_trigger | drift | repeat_instruction | 3 | 4 | 402.3333 | -15.3542 | -0.5391 | {' be': 2, 'Answer': 4} |
| deepseek7b_explain_l24_natural_trigger | drift | short_answer_instruction | 3 | 0 | 343.0000 | 0.3125 | -1.5000 | {' are': 2} |
| qwen3_explain_l29_natural_trigger | drift | short_answer_instruction | 2 | 4 | -324.0000 | -0.2708 | -1.4062 | {' can': 2, '.': 4} |
| deepseek7b_explain_l24_natural_trigger | drift | no_answer_anchor | 3 | 0 | 321.0000 | -0.1250 | -1.0625 | {' are': 2} |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 3 | 0 | 266.3333 | -9.5208 | 0.8359 | {' be': 2, 'Because': 4} |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 2 | 6 | -237.0000 | 1.7708 | 1.8333 | {' is': 2, '.': 2, '.\n\n': 2} |
| qwen3_explain_l29_natural_trigger | drift | no_instruction | 3 | 6 | -226.6667 | -10.9375 | -0.2422 | {' come': 2, 'But': 2, 'The': 2} |
| glm4_repeat_l30_natural_trigger | drift | comma_removed | 1 | 3 | 220.5000 | 1.5026 | 1.2630 | {' A': 1, ' Horse': 1, ' The': 2, ' White': 2} |
| deepseek7b_explain_l24_natural_trigger | success | no_instruction | 2 | 0 | 212.0000 | -1.3438 | -0.4896 | {' cup': 2, ' dog': 2, ' is': 2} |
| qwen3_explain_l29_natural_trigger | drift | because_removed | 2 | 2 | -209.3333 | -0.9375 | -1.7292 | {' can': 2, '.': 2, '.\n': 2} |
| qwen3_explain_l29_natural_trigger | success | repeat_instruction | 2 | 6 | 193.6667 | -6.2083 | 4.8854 | {',': 6} |
| qwen3_explain_l29_natural_trigger | drift | no_answer_anchor | 2 | 4 | 149.3333 | 1.8542 | 2.2083 | {' can': 2, '.': 4} |
| glm4_repeat_l30_natural_trigger | drift | explain_instruction | 2 | 5 | -141.5000 | -1.5417 | -2.6875 | {'\n': 4, ' is': 1, ' used': 1} |
| glm4_repeat_l30_natural_trigger | success | no_instruction | 2 | 6 | -132.1667 | 0.5729 | -2.4844 | {'\n': 6} |
| qwen3_explain_l29_natural_trigger | success | short_answer_instruction | 2 | 0 | -138.0000 | -2.9375 | 0.4583 | {' is': 6} |
| glm4_repeat_l30_natural_trigger | success | short_answer_instruction | 2 | 6 | -92.1667 | 1.3750 | -1.3021 | {'\n': 4, ' or': 2} |
| qwen3_explain_l29_natural_trigger | success | no_instruction | 2 | 0 | -97.8333 | -3.7500 | 1.6146 | {' is': 6} |
