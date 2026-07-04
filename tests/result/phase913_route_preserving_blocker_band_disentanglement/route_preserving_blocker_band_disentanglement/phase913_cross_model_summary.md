# Phase 913 route-preserving blocker band disentanglement

## Overall

- models: qwen3, glm4, deepseek7b
- route_eos_top10: 4
- route_eos_top50: 15
- route_preserving_disentangle_candidate: 241
- route_rows: 68
- rows: 8444
- source_eos_top1: 0
- source_eos_top10: 503
- source_eos_top5: 33
- source_eos_top50: 1982
- source_margin_nonnegative: 0
- source_rows: 8376
- source_strict_clean_candidate: 0
- strict_clean_candidate: 0
- strong_route_preserving_disentangle_candidate: 143

## Model Summaries

| model | rows | source rows | route top10 | route top50 | source top1 | source top5 | margin>=0 | weak disentangle | strong disentangle | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 2340 | 2322 | 0 | 0 | 0 | 0 | 0 | 76 | 22 | strong_route_preserving_disentangle_candidates_found |
| glm4 | 2210 | 2193 | 4 | 15 | 0 | 2 | 0 | 41 | 20 | route_preserving_disentangle_reaches_eos_top5 |
| deepseek7b | 3894 | 3861 | 0 | 0 | 0 | 31 | 0 | 124 | 101 | route_preserving_disentangle_reaches_eos_top5 |

## Top Specs

| model | family | label | factor | rows | top1 | top5 | margin>=0 | weak | strong | median band16 delta | median eos delta | blockers |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b | l0_attention_span | L0_attention_span_prompt_all_scale_0.25 | 0.25 | 33 | 0 | 28 | 0 | 33 | 33 | -12.52294921875 | 5.921875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | l0_attention_span | L0_attention_span_prompt_all_scale_0.25 | 0.25 | 17 | 0 | 2 | 0 | 2 | 2 | -5.532470703125 | -6.703125 | {'a': 15, ' Fish': 2} |
| deepseek7b | l0_attention_span | L0_attention_span_last8_before_period_scale_0.25 | 0.25 | 33 | 0 | 2 | 0 | 2 | 2 | -0.87890625 | -1.9375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_span | L0_attention_span_prompt_last8_scale_0.25 | 0.25 | 33 | 0 | 1 | 0 | 3 | 3 | -1.78125 | -2.484375 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_span | L0_attention_span_prompt_all_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 32 | 31 | -4.09765625 | 8.8125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_span | L0_attention_span_prompt_first8_scale_0.25 | 0.25 | 33 | 0 | 0 | 0 | 24 | 24 | -3.12890625 | 2.875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_span | L0_attention_span_period_token_scale_0.25 | 0.25 | 33 | 0 | 0 | 0 | 15 | 5 | -0.40625 | 0.40625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_span | L0_attention_span_prompt_all_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 10 | 5 | -0.4765625 | 0.25 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_prompt_first8_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 7 | 5 | -0.265625 | 0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_prompt_all_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 9 | 2 | -0.712890625 | 0.171875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_prompt_last8_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 9 | 2 | -0.36328125 | 0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_last8_before_period_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 8 | 2 | -0.46484375 | -0.2265625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_last8_before_period_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 5 | 2 | -0.140625 | -0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_prompt_last8_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 5 | 2 | -0.06640625 | -0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | l0_attention_span | L0_attention_span_prompt_first8_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 4 | 2 | -0.9921875 | -0.32421875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_span | L0_attention_span_prompt_first8_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 4 | 2 | -0.275390625 | 0.15625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l0_attention_span | L0_attention_span_prompt_last8_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 3 | 2 | -0.0234375 | 0.0 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_span | L0_attention_span_prompt_all_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 2 | 2 | -0.564453125 | -1.59375 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_span | L0_attention_span_period_token_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 2 | 2 | -0.27734375 | -0.40625 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_head | L0_attention_head_26_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 2 | 2 | -0.103515625 | -0.1875 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_span | L0_attention_span_prompt_last8_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 2 | 2 | -0.07421875 | -0.03125 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_span | L0_attention_span_last8_before_period_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 2 | 2 | -0.06640625 | -0.03125 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_span | L0_attention_span_period_token_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 2 | 2 | -0.06640625 | 0.0 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_span | L0_attention_span_last8_before_period_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 2 | 2 | -0.033203125 | -0.03125 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_span | L0_attention_span_prompt_first8_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 2 | 2 | 0.033203125 | 0.5 | {'a': 15, ' Fish': 2} |
| deepseek7b | l0_attention_span | L0_attention_span_last8_before_period_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 1 | 1 | 0.10546875 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_span | L0_attention_span_prompt_all_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 8 | 0 | -0.265625 | 0.109375 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_top_abs_64_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 6 | 0 | -0.189453125 | 0.1875 | {'a': 15, ' Fish': 2} |
| qwen3 | l0_attention_span | L0_attention_span_prompt_first8_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 6 | 0 | -0.109375 | 0.125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_last8_before_period_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 5 | 0 | -0.0390625 | 0.078125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_band32_support_64_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 4 | 0 | -0.18359375 | 0.0 | {'a': 15, ' Fish': 2} |
| deepseek7b | l0_attention_head | L0_attention_head_15_scale_0.25 | 0.25 | 33 | 0 | 0 | 0 | 4 | 0 | -0.12109375 | -0.203125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_band16_support_32_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 3 | 0 | -0.224609375 | -0.03125 | {'a': 15, ' Fish': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_band16_support_64_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 3 | 0 | -0.1953125 | -0.03125 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_span | L0_attention_span_prompt_last8_scale_0.75 | 0.75 | 17 | 0 | 0 | 0 | 2 | 0 | -0.05078125 | -0.0625 | {'a': 15, ' Fish': 2} |
| glm4 | l0_attention_span | L0_attention_span_last8_before_period_scale_0.75 | 0.75 | 17 | 0 | 0 | 0 | 2 | 0 | -0.05078125 | -0.09375 | {'a': 15, ' Fish': 2} |
| deepseek7b | l0_attention_head | L0_attention_head_15_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 2 | 0 | 0.01171875 | 0.0625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_span | L0_attention_span_period_token_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 2 | 0 | 0.015625 | 0.125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_span | L0_attention_span_prompt_last8_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 2 | 0 | 0.12109375 | -0.046875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | l0_attention_head | L0_attention_head_22_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 0 | 0 | -1.146484375 | -0.90625 | {'a': 15, ' Fish': 2} |
| deepseek7b | l0_attention_span | L0_attention_span_prompt_first8_scale_0.75 | 0.75 | 33 | 0 | 0 | 0 | 0 | 0 | -0.3125 | -0.21875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_span | L0_attention_span_prompt_all_scale_0.75 | 0.75 | 33 | 0 | 0 | 0 | 0 | 0 | -0.2734375 | -0.3046875 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| glm4 | l0_attention_head | L0_attention_head_22_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 0 | 0 | -0.1328125 | -0.15625 | {'a': 15, ' Fish': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_band32_support_64_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 0 | 0 | -0.107421875 | 0.03125 | {'a': 15, ' Fish': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_top_abs_64_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 0 | 0 | -0.095703125 | 0.125 | {'a': 15, ' Fish': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_band16_support_64_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 0 | 0 | -0.08984375 | 0.03125 | {'a': 15, ' Fish': 2} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_band16_support_64_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.07421875 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l0_attention_head | L0_attention_head_24_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 0 | 0 | -0.0703125 | 0.03125 | {'a': 15, ' Fish': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_band16_support_32_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 0 | 0 | -0.06640625 | 0.0625 | {'a': 15, ' Fish': 2} |
| qwen3 | l0_attention_span | L0_attention_span_answer_prefix_all_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0625 | -0.0625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l0_attention_head | L0_attention_head_26_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 0 | 0 | -0.0625 | -0.125 | {'a': 15, ' Fish': 2} |
| qwen3 | l0_attention_span | L0_attention_span_period_token_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.05859375 | -0.140625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_band16_support_32_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0546875 | -0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_15_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0546875 | -0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l0_attention_span | L0_attention_span_prompt_all_scale_0.75 | 0.75 | 17 | 0 | 0 | 0 | 0 | 0 | -0.0546875 | -0.09375 | {'a': 15, ' Fish': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_band16_support_64_scale_0.75 | 0.75 | 17 | 0 | 0 | 0 | 0 | 0 | -0.052734375 | 0.0625 | {'a': 15, ' Fish': 2} |
| qwen3 | l0_attention_head | L0_attention_head_21_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.05078125 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_24_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.05078125 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_top_abs_64_scale_0.75 | 0.75 | 17 | 0 | 0 | 0 | 0 | 0 | -0.05078125 | 0.03125 | {'a': 15, ' Fish': 2} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_band16_support_32_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.048828125 | 0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_17_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.046875 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_12_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.046875 | -0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_answer_prefix_all_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.046875 | -0.078125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | l0_attention_span | L0_attention_span_answer_prefix_all_scale_0.25 | 0.25 | 33 | 0 | 0 | 0 | 0 | 0 | -0.046875 | -0.5 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_head | L0_attention_head_0_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.046875 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_band32_support_64_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.044921875 | 0.046875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_8_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.04296875 | -0.0078125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | l0_attention_head | L0_attention_head_17_scale_0.25 | 0.25 | 33 | 0 | 0 | 0 | 0 | 0 | -0.04296875 | -0.0625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_head | L0_attention_head_9_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.04296875 | -0.109375 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l4_mlp_channel_group | L4_mlp_channels_band16_support_32_scale_0.75 | 0.75 | 17 | 0 | 0 | 0 | 0 | 0 | -0.041015625 | 0.03125 | {'a': 15, ' Fish': 2} |
| qwen3 | l0_attention_head | L0_attention_head_18_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0390625 | -0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_2_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0390625 | -0.1875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_period_token_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0390625 | -0.265625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_band32_support_64_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.03515625 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_31_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.03515625 | -0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_7_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.03515625 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l0_attention_head | L0_attention_head_20_scale_0.25 | 0.25 | 17 | 0 | 0 | 0 | 0 | 0 | -0.03515625 | 0.0 | {'a': 15, ' Fish': 2} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_top_abs_64_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.03125 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_low_abs_64_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.03125 | 0.0078125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_29_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.03125 | -0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_15_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.03125 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | l0_attention_head | L0_attention_head_7_scale_0.75 | 0.75 | 33 | 0 | 0 | 0 | 0 | 0 | -0.03125 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_head | L0_attention_head_18_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.02734375 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_20_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.02734375 | 0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_5_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.02734375 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_8_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.02734375 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | l0_attention_head | L0_attention_head_6_scale_0.25 | 0.25 | 33 | 0 | 0 | 0 | 0 | 0 | -0.02734375 | 0.0 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_head | L0_attention_head_4_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0234375 | 0.046875 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_30_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0234375 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_27_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0234375 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_22_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0234375 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| glm4 | l0_attention_head | L0_attention_head_16_scale_0.5 | 0.5 | 17 | 0 | 0 | 0 | 0 | 0 | -0.0234375 | 0.03125 | {'a': 15, ' Fish': 2} |
| qwen3 | l0_attention_head | L0_attention_head_17_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0234375 | -0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_26_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0234375 | -0.0234375 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_band16_support_32_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.0234375 | 0.0 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_8_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_20_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | 0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_21_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | 0.0078125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | l0_attention_head | L0_attention_head_11_scale_0.75 | 0.75 | 33 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | -0.00390625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_band16_support_64_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | -0.0078125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | l0_attention_head | L0_attention_head_4_scale_0.75 | 0.75 | 33 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | -0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_head | L0_attention_head_6_scale_0.75 | 0.75 | 33 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | -0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_head | L0_attention_head_5_scale_0.25 | 0.25 | 33 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | -0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_span | L0_attention_span_answer_prefix_all_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | -0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | l0_attention_span | L0_attention_span_answer_prefix_all_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | -0.03125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_head | L0_attention_head_25_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.01953125 | -0.0625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_span | L0_attention_span_prompt_last8_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.0625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_0_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_1_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_16_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_27_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.03125 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_1_scale_0.75 | 0.75 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_7_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l4_mlp_channel_group | L4_mlp_channels_band16_support_64_scale_0.5 | 0.5 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| qwen3 | l0_attention_head | L0_attention_head_26_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.015625 | {' \n\n': 10, 'Okay': 6, ' The': 2} |
| deepseek7b | l0_attention_head | L0_attention_head_6_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_head | L0_attention_head_13_scale_0.25 | 0.25 | 33 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_head | L0_attention_head_22_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | -0.015625 | -0.0078125 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| deepseek7b | l0_attention_head | L0_attention_head_5_scale_0.5 | 0.5 | 33 | 0 | 0 | 0 | 0 | 0 | -0.015625 | -0.015625 | {'</think>': 19, 'The': 5, 'Category': 4, 'Wait': 2, ' Sub': 1, ' In': 1, 'Okay': 1} |
| qwen3 | l0_attention_head | L0_attention_head_28_scale_0.25 | 0.25 | 18 | 0 | 0 | 0 | 0 | 0 | -0.015625 | -0.0234375 | {' \n\n': 10, 'Okay': 6, ' The': 2} |

## Top Families

| model | family | factor | rows | top1 | top5 | margin>=0 | weak | strong | median band16 delta | median eos delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | l0_attention_span | 0.25 | 198 | 0 | 31 | 0 | 77 | 67 | -1.599609375 | -0.15625 |
| glm4 | l0_attention_span | 0.25 | 102 | 0 | 2 | 0 | 11 | 10 | -0.068359375 | -0.03125 |
| deepseek7b | l0_attention_span | 0.5 | 198 | 0 | 0 | 0 | 41 | 34 | -0.083984375 | 0.0078125 |
| qwen3 | l0_attention_span | 0.5 | 108 | 0 | 0 | 0 | 27 | 14 | -0.0859375 | -0.0625 |
| qwen3 | l0_attention_span | 0.25 | 108 | 0 | 0 | 0 | 30 | 8 | -0.197265625 | -0.078125 |
| glm4 | l0_attention_span | 0.5 | 102 | 0 | 0 | 0 | 8 | 8 | -0.060546875 | -0.03125 |
| glm4 | l0_attention_head | 0.25 | 544 | 0 | 0 | 0 | 2 | 2 | 0.01171875 | 0.0 |
| qwen3 | l0_attention_span | 0.75 | 108 | 0 | 0 | 0 | 19 | 0 | -0.046875 | 0.0234375 |
| glm4 | l4_mlp_channel_group | 0.25 | 85 | 0 | 0 | 0 | 16 | 0 | -0.16796875 | 0.0 |
| glm4 | l0_attention_span | 0.75 | 102 | 0 | 0 | 0 | 4 | 0 | 0.0 | -0.03125 |
| deepseek7b | l0_attention_head | 0.25 | 924 | 0 | 0 | 0 | 4 | 0 | 0.0 | 0.0 |
| deepseek7b | l0_attention_head | 0.5 | 924 | 0 | 0 | 0 | 2 | 0 | 0.00390625 | 0.0 |
| qwen3 | l0_attention_head | 0.75 | 576 | 0 | 0 | 0 | 0 | 0 | -0.0078125 | 0.0 |
| qwen3 | l4_mlp_channel_group | 0.75 | 90 | 0 | 0 | 0 | 0 | 0 | -0.0078125 | 0.0 |
| qwen3 | l0_attention_head | 0.5 | 576 | 0 | 0 | 0 | 0 | 0 | -0.015625 | 0.0 |
| qwen3 | l4_mlp_channel_group | 0.5 | 90 | 0 | 0 | 0 | 0 | 0 | -0.0234375 | 0.03125 |
| qwen3 | l0_attention_head | 0.25 | 576 | 0 | 0 | 0 | 0 | 0 | -0.0078125 | 0.0 |
| qwen3 | l4_mlp_channel_group | 0.25 | 90 | 0 | 0 | 0 | 0 | 0 | -0.03515625 | 0.0 |
| glm4 | l0_attention_head | 0.75 | 544 | 0 | 0 | 0 | 0 | 0 | 0.013671875 | 0.0 |
| glm4 | l4_mlp_channel_group | 0.75 | 85 | 0 | 0 | 0 | 0 | 0 | -0.017578125 | 0.03125 |
| glm4 | l0_attention_head | 0.5 | 544 | 0 | 0 | 0 | 0 | 0 | 0.01171875 | 0.0 |
| glm4 | l4_mlp_channel_group | 0.5 | 85 | 0 | 0 | 0 | 0 | 0 | -0.080078125 | 0.03125 |
| deepseek7b | l0_attention_head | 0.75 | 924 | 0 | 0 | 0 | 0 | 0 | 0.00390625 | 0.0 |
| deepseek7b | l0_attention_span | 0.75 | 198 | 0 | 0 | 0 | 0 | 0 | -0.0078125 | -0.0625 |
| deepseek7b | l4_mlp_channel_group | 0.75 | 165 | 0 | 0 | 0 | 0 | 0 | 0.0390625 | 0.015625 |
| deepseek7b | l4_mlp_channel_group | 0.5 | 165 | 0 | 0 | 0 | 0 | 0 | 0.03125 | 0.03125 |
| deepseek7b | l4_mlp_channel_group | 0.25 | 165 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.0390625 |
