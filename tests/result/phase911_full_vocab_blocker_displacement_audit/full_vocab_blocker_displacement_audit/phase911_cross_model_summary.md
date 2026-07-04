# Phase 911 full-vocabulary blocker displacement audit

## Overall

- models: qwen3, glm4, deepseek7b
- diagnostic_eos_margin_nonnegative: 25
- diagnostic_eos_top1: 25
- diagnostic_eos_top10: 51
- diagnostic_eos_top5: 36
- diagnostic_eos_top50: 75
- diagnostic_route_blocker_displaced: 340
- diagnostic_rows: 340
- diagnostic_strict_clean_candidate: 25
- internal_eos_margin_nonnegative: 0
- internal_eos_top1: 0
- internal_eos_top10: 57
- internal_eos_top5: 0
- internal_eos_top50: 210
- internal_route_blocker_displaced: 30
- internal_rows: 952
- internal_strict_clean_candidate: 0
- patched_eos_margin_nonnegative: 25
- patched_eos_top1: 25
- patched_eos_top10: 108
- patched_eos_top5: 36
- patched_eos_top50: 285
- route_blocker_displaced: 370
- route_eos_top10: 76
- route_eos_top50: 285
- rows: 1292
- strict_clean_candidate: 25

## Model Summaries

| model | rows | route top10 | route top50 | internal top1 | internal top5 | internal top10 | diagnostic top1 | diagnostic top10 | internal margin>=0 | strict clean | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 342 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_route_near_and_no_blocker_displacement |
| glm4 | 323 | 76 | 285 | 0 | 0 | 57 | 25 | 51 | 0 | 25 | logit_mask_diagnostic_shows_narrow_blocker_bottleneck |
| deepseek7b | 627 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_route_near_and_no_blocker_displacement |

## Top Controls

| model | control | family | neural | diagnostic | rows | internal top1 | internal top5 | internal top10 | diagnostic top1 | diagnostic top10 | margin>=0 | median margin delta | route blockers | patched blockers |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| glm4 | route_minus_unembed_blocker_top1_beta_0.1 | internal_readout_blocker_suppression | True | False | 17 | 0 | 0 | 5 | 0 | 0 | 0 | 0.0 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_minus_unembed_blocker_top3_beta_0.1 | internal_readout_blocker_suppression | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | -0.03125 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_minus_unembed_blocker_top1_beta_0.25 | internal_readout_blocker_suppression | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | -0.03125 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_minus_unembed_blocker_top3_beta_0.25 | internal_readout_blocker_suppression | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | -0.03125 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_minus_unembed_blocker_top1_beta_0.5 | internal_readout_blocker_suppression | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | -0.03125 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_minus_unembed_blocker_top3_beta_0.5 | internal_readout_blocker_suppression | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | -0.03125 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_plus_unembed_eos_beta_0.5 | internal_readout_eos_boost | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | -0.03125 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_plus_unembed_eos_beta_0.1 | internal_readout_eos_boost | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | -0.0625 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_only_alpha_1 | prompt_preserving_route_control | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | 0.0 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_plus_unembed_margin_top1_beta_0.05 | internal_readout_margin_direction | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | 0.0 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_plus_unembed_margin_top1_beta_0.1 | internal_readout_margin_direction | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | 0.0 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_plus_unembed_margin_top1_beta_0.25 | internal_readout_margin_direction | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | 0.0 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_plus_unembed_margin_top1_beta_0.5 | internal_readout_margin_direction | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | 0.0 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_plus_unembed_eos_beta_0.25 | internal_readout_eos_boost | True | False | 17 | 0 | 0 | 4 | 0 | 0 | 0 | 0.0 | {'a': 15, ' Fish': 2} | {'a': 15, ' Fish': 2} |
| glm4 | route_logit_mask_blocker_top32 | logit_blocker_mask_diagnostic | False | True | 17 | 0 | 0 | 0 | 15 | 15 | 15 | 3.25 | {'a': 15, ' Fish': 2} | {'8': 5, ' was': 4, ' ': 2, 'C': 2, 'L': 2, ' and': 1, 'N': 1} |
| glm4 | route_logit_mask_blocker_top16 | logit_blocker_mask_diagnostic | False | True | 17 | 0 | 0 | 0 | 10 | 15 | 10 | 2.5625 | {'a': 15, ' Fish': 2} | {'B': 4, ' ': 3, '9': 3, '%': 3, ' Specifically': 2, '6': 1, 'F': 1} |
| glm4 | route_logit_mask_blocker_top8 | logit_blocker_mask_diagnostic | False | True | 17 | 0 | 0 | 0 | 0 | 11 | 0 | 1.9375 | {'a': 15, ' Fish': 2} | {' ': 5, ' is': 3, ' \n': 2, 'A': 2, 's': 2, ' for': 1, ' in': 1, ' of': 1} |
| glm4 | route_logit_mask_blocker_top3 | logit_blocker_mask_diagnostic | False | True | 17 | 0 | 0 | 0 | 0 | 6 | 0 | 1.0 | {'a': 15, ' Fish': 2} | {'4': 8, '3': 7, ' However': 2} |
| glm4 | route_logit_mask_blocker_top1 | logit_blocker_mask_diagnostic | False | True | 17 | 0 | 0 | 0 | 0 | 4 | 0 | 0.625 | {'a': 15, ' Fish': 2} | {'1': 15, ' fish': 2} |
| qwen3 | route_logit_mask_blocker_top32 | logit_blocker_mask_diagnostic | False | True | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 5.78125 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {'A': 3, 'Make': 2, ' Yes': 1, ' Alright': 1, ' Please': 1, ' You': 1, ' Just': 1, ' -': 1, '?\n': 1, 'Correct': 1, ' How': 1, ' **': 1} |
| deepseek7b | route_logit_mask_blocker_top32 | logit_blocker_mask_diagnostic | False | True | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 5.3125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'Sub': 4, 'Circle': 4, 'Response': 3, '<think>': 3, 'Can': 2, 'How': 2, 'Oh': 2, 'Sure': 2, 'To': 2, 'Yeah': 2, 'M': 1, ' \n': 1} |
| qwen3 | route_logit_mask_blocker_top16 | logit_blocker_mask_diagnostic | False | True | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 4.75 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {'  \n\n': 3, ' But': 3, 'Can': 3, ' That': 2, 'Is': 2, ' Short': 1, ' Or': 1, 'Sure': 1, ' This': 1, ' Correct': 1} |
| deepseek7b | route_logit_mask_blocker_top16 | logit_blocker_mask_diagnostic | False | True | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 4.1875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'Please': 7, 'Is': 7, 'I': 4, 'Step': 3, 'Which': 3, 'Animal': 3, '-': 2, ' To': 1, 'What': 1, ' \n\n': 1, 'You': 1} |
| qwen3 | route_logit_mask_blocker_top8 | logit_blocker_mask_diagnostic | False | True | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 3.3125 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' Or': 4, 'Classification': 3, 'But': 2, ' Yes': 1, ' This': 1, ' To': 1, 'Question': 1, ' What': 1, ' That': 1, ' Category': 1, ' Explanation': 1, ' \n': 1} |
| deepseek7b | route_logit_mask_blocker_top8 | logit_blocker_mask_diagnostic | False | True | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 2.625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'Wait': 8, 'Hmm': 5, 'Okay': 4, 'Another': 2, 'That': 2, 'But': 2, 'Answer': 2, 'Sh': 2, 'What': 2, 'No': 1, ' Okay': 1, 'Con': 1} |
| qwen3 | route_logit_mask_blocker_top3 | logit_blocker_mask_diagnostic | False | True | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 1.75 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' Answer': 5, 'Class': 3, ' Explanation': 2, ' Okay': 2, 'The': 2, ' (': 1, 'Alright': 1, ' What': 1, ' cotton': 1} |
| deepseek7b | route_logit_mask_blocker_top3 | logit_blocker_mask_diagnostic | False | True | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 1.5 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'Okay': 8, 'Polygon': 5, 'Answer': 4, 'Category': 3, 'I': 2, 'Yes': 2, '</think>': 2, 'So': 2, 'The': 2, ' Correct': 1, 'Wait': 1, ' Yes': 1} |
| qwen3 | route_logit_mask_blocker_top1 | logit_blocker_mask_diagnostic | False | True | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.625 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' The': 5, ' Explanation': 3, 'Wood': 3, 'Item': 2, ' Answer': 1, ' ': 1, 'Class': 1, ' \n\n': 1, ' \n': 1} |
| deepseek7b | route_logit_mask_blocker_top1 | logit_blocker_mask_diagnostic | False | True | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'The': 12, 'Yes': 7, '</think>': 3, 'But': 2, 'I': 2, 'Wait': 2, 'Category': 2, ' The': 1, 'Sh': 1, ' Sub': 1} |
| qwen3 | route_minus_unembed_blocker_top1_beta_0.1 | internal_readout_blocker_suppression | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 9, 'Okay': 6, ' The': 3} |
| qwen3 | route_minus_unembed_blocker_top3_beta_0.25 | internal_readout_blocker_suppression | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.046875 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | route_plus_unembed_eos_beta_0.5 | internal_readout_eos_boost | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.046875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'The': 4, 'Category': 4, ' Sub': 2, 'Wait': 2, 'Sh': 1, 'Yes': 1} |
| qwen3 | route_minus_unembed_blocker_top1_beta_0.25 | internal_readout_blocker_suppression | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 9, 'Okay': 6, ' The': 3} |
| qwen3 | route_plus_unembed_eos_beta_0.25 | internal_readout_eos_boost | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 8, 'Okay': 6, ' The': 4} |
| deepseek7b | route_plus_unembed_margin_top1_beta_0.25 | internal_readout_margin_direction | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'The': 4, 'Category': 4, ' Sub': 2, 'Wait': 2, 'Sh': 1, 'Okay': 1} |
| deepseek7b | route_plus_unembed_eos_beta_0.1 | internal_readout_eos_boost | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'The': 5, 'Category': 4, ' Sub': 2, 'Wait': 2, 'Okay': 1} |
| qwen3 | route_plus_unembed_margin_top1_beta_0.05 | internal_readout_margin_direction | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 9, 'Okay': 6, ' The': 3} |
| qwen3 | route_plus_unembed_margin_top1_beta_0.5 | internal_readout_margin_direction | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | route_minus_unembed_blocker_top1_beta_0.5 | internal_readout_blocker_suppression | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 17, 'The': 4, 'Category': 4, 'But': 2, 'I': 2, ' The': 1, 'Sh': 1, ' In': 1, 'Okay': 1} |
| qwen3 | route_plus_unembed_eos_beta_0.1 | internal_readout_eos_boost | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.0078125 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | route_plus_unembed_margin_top1_beta_0.05 | internal_readout_margin_direction | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 17, 'The': 6, 'Category': 4, 'Wait': 2, ' The': 1, 'Sh': 1, ' Sub': 1, 'Okay': 1} |
| deepseek7b | route_plus_unembed_margin_top1_beta_0.5 | internal_readout_margin_direction | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'Category': 4, ' Sub': 2, 'Yes': 2, 'Wait': 2, 'The': 2, 'Sh': 1, 'Okay': 1} |
| deepseek7b | route_minus_unembed_blocker_top1_beta_0.25 | internal_readout_blocker_suppression | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'Category': 4, ' Sub': 2, 'Yes': 2, 'Wait': 2, 'The': 2, 'Sh': 1, 'Okay': 1} |
| deepseek7b | route_minus_unembed_blocker_top3_beta_0.5 | internal_readout_blocker_suppression | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'The': 4, 'Category': 4, ' Sub': 2, 'Wait': 2, 'Sh': 1, 'Okay': 1} |
| deepseek7b | route_plus_unembed_margin_top1_beta_0.1 | internal_readout_margin_direction | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.03125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'The': 4, 'Category': 4, ' Sub': 2, 'Okay': 2, 'Wait': 2} |
| deepseek7b | route_minus_unembed_blocker_top1_beta_0.1 | internal_readout_blocker_suppression | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | -0.03125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'The': 4, 'Category': 4, ' Sub': 2, 'Okay': 2, 'Wait': 2} |
| qwen3 | route_minus_unembed_blocker_top1_beta_0.5 | internal_readout_blocker_suppression | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.0625 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | route_plus_unembed_eos_beta_0.5 | internal_readout_eos_boost | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | -0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 9, 'Okay': 6, ' The': 3} |
| qwen3 | route_only_alpha_1 | prompt_preserving_route_control | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | route_plus_unembed_margin_top1_beta_0.1 | internal_readout_margin_direction | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 9, 'Okay': 6, ' The': 3} |
| qwen3 | route_plus_unembed_margin_top1_beta_0.25 | internal_readout_margin_direction | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | route_minus_unembed_blocker_top3_beta_0.1 | internal_readout_blocker_suppression | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 9, 'Okay': 6, ' The': 3} |
| qwen3 | route_minus_unembed_blocker_top3_beta_0.5 | internal_readout_blocker_suppression | True | False | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} | {' \n\n': 8, 'Okay': 6, ' The': 4} |
| deepseek7b | route_only_alpha_1 | prompt_preserving_route_control | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | route_minus_unembed_blocker_top3_beta_0.1 | internal_readout_blocker_suppression | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'The': 4, 'Category': 4, ' Sub': 2, 'Wait': 2, 'Sh': 1, 'Okay': 1} |
| deepseek7b | route_minus_unembed_blocker_top3_beta_0.25 | internal_readout_blocker_suppression | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'Category': 4, ' Sub': 2, 'Yes': 2, 'Wait': 2, 'The': 2, 'Sh': 1, 'Okay': 1} |
| deepseek7b | route_plus_unembed_eos_beta_0.25 | internal_readout_eos_boost | True | False | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} | {'</think>': 19, 'The': 6, 'Category': 4, ' Sub': 2, 'Sh': 1, 'Okay': 1} |
