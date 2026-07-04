# Phase 910 prompt-preserving termination route reconstruction

## Overall

- models: qwen3, glm4, deepseek7b
- continuation_suppressed: 539
- direct_eos_lift: 427
- eos_rank_improved: 461
- eos_rank_improved_100: 407
- eos_rank_improved_1000: 237
- next_category_changed: 195
- next_top_changed: 295
- patched_eos_top1: 0
- patched_eos_top10: 4
- patched_eos_top5: 0
- patched_eos_top50: 18
- prompt_preserving_eos_top1: 0
- prompt_preserving_eos_top10: 4
- prompt_preserving_eos_top50: 15
- protocol_suppressed: 597
- rows: 1020
- strict_clean_candidate: 0

## Prompt-Intact Overall

- continuation_suppressed: 168
- direct_eos_lift: 191
- eos_rank_improved: 196
- eos_rank_improved_100: 160
- eos_rank_improved_1000: 62
- next_category_changed: 45
- next_top_changed: 65
- patched_eos_top1: 0
- patched_eos_top10: 4
- patched_eos_top5: 0
- patched_eos_top50: 15
- prompt_preserving_eos_top1: 0
- prompt_preserving_eos_top10: 4
- prompt_preserving_eos_top50: 15
- protocol_suppressed: 188
- rows: 476
- strict_clean_candidate: 0

## Model Summaries

| model | rows | prompt intact rows | eos top1 | eos top10 | eos top50 | prompt-intact top10 | prompt-intact top50 | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 270 | 126 | 0 | 0 | 0 | 0 | 0 | prompt_preserving_route_improves_eos_but_not_near |
| glm4 | 255 | 119 | 0 | 4 | 16 | 4 | 15 | prompt_preserving_route_reaches_eos_top10 |
| deepseek7b | 495 | 231 | 0 | 0 | 2 | 0 | 0 | prompt_preserving_route_improves_eos_but_not_near |

## Top Controls

| model | control | family | rows | intact | eos top10 | eos top50 | intact top10 | intact top50 | blocker median margin |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| glm4 | L0_promptzero_delta_alpha_1 | prompt_intact_counterfactual_direction | 17 | True | 4 | 15 | 4 | 15 | -2.4375 |
| deepseek7b | L0_input_period_zero | limited_span_adjustment | 33 | False | 0 | 2 | 0 | 0 | -28.265625 |
| glm4 | L0_input_prompt_last8_zero | limited_span_adjustment | 17 | False | 0 | 1 | 0 | 0 | -16.361328125 |
| deepseek7b | L0_input_prompt_first8_half | limited_span_adjustment | 33 | False | 0 | 0 | 0 | 0 | -14.2421875 |
| qwen3 | L0_input_prompt_first8_zero | limited_span_adjustment | 18 | False | 0 | 0 | 0 | 0 | -12.578125 |
| deepseek7b | L0_input_prompt_last8_half | limited_span_adjustment | 33 | False | 0 | 0 | 0 | 0 | -15.171875 |
| glm4 | L0_input_prompt_first8_zero | limited_span_adjustment | 17 | False | 0 | 0 | 0 | 0 | -12.484375 |
| glm4 | L0_input_prompt_all_half | limited_span_adjustment | 17 | False | 0 | 0 | 0 | 0 | -13.578125 |
| deepseek7b | L0_attn_output_last_scale_0.50 | prompt_intact_output_scale | 33 | True | 0 | 0 | 0 | 0 | -14.5625 |
| deepseek7b | L0_attn_output_last_scale_0.75 | prompt_intact_output_scale | 33 | True | 0 | 0 | 0 | 0 | -14.5 |
| glm4 | L0_input_period_zero | limited_span_adjustment | 17 | False | 0 | 0 | 0 | 0 | -13.54296875 |
| glm4 | L0_input_prompt_first8_half | limited_span_adjustment | 17 | False | 0 | 0 | 0 | 0 | -13.3984375 |
| glm4 | L0_input_period_half | limited_span_adjustment | 17 | False | 0 | 0 | 0 | 0 | -14.2978515625 |
| deepseek7b | L0_input_prompt_all_half | limited_span_adjustment | 33 | False | 0 | 0 | 0 | 0 | -15.140625 |
| deepseek7b | L0_input_answer_prefix_last_half | limited_span_adjustment | 33 | False | 0 | 0 | 0 | 0 | -14.796875 |
| deepseek7b | L0_input_period_half | limited_span_adjustment | 33 | False | 0 | 0 | 0 | 0 | -14.6875 |
| qwen3 | L0_input_period_zero | limited_span_adjustment | 18 | False | 0 | 0 | 0 | 0 | -16.734375 |
| deepseek7b | L0_input_prompt_first8_zero | limited_span_adjustment | 33 | False | 0 | 0 | 0 | 0 | -14.15625 |
| qwen3 | L0_input_prompt_last8_half | limited_span_adjustment | 18 | False | 0 | 0 | 0 | 0 | -13.609375 |
| qwen3 | L0_promptzero_delta_alpha_1 | prompt_intact_counterfactual_direction | 18 | True | 0 | 0 | 0 | 0 | -13.921875 |
| glm4 | L0_promptzero_delta_alpha_0.25 | prompt_intact_counterfactual_direction | 17 | True | 0 | 0 | 0 | 0 | -14.34521484375 |
| deepseek7b | L0_promptzero_delta_alpha_0.1 | prompt_intact_counterfactual_direction | 33 | True | 0 | 0 | 0 | 0 | -14.5625 |
| deepseek7b | L0_promptzero_delta_alpha_1 | prompt_intact_counterfactual_direction | 33 | True | 0 | 0 | 0 | 0 | -15.09375 |
| qwen3 | L0_input_prompt_all_half | limited_span_adjustment | 18 | False | 0 | 0 | 0 | 0 | -13.625 |
| glm4 | L0_input_answer_prefix_last_half | limited_span_adjustment | 17 | False | 0 | 0 | 0 | 0 | -14.40478515625 |
| glm4 | L0_promptzero_delta_alpha_0.1 | prompt_intact_counterfactual_direction | 17 | True | 0 | 0 | 0 | 0 | -14.36578369140625 |
| deepseek7b | L0_promptzero_delta_alpha_0.5 | prompt_intact_counterfactual_direction | 33 | True | 0 | 0 | 0 | 0 | -14.671875 |
| qwen3 | L0_input_prompt_last8_zero | limited_span_adjustment | 18 | False | 0 | 0 | 0 | 0 | -15.390625 |
| deepseek7b | L0_promptzero_delta_alpha_0.05 | prompt_intact_counterfactual_direction | 33 | True | 0 | 0 | 0 | 0 | -14.6875 |
| qwen3 | L0_input_answer_prefix_last_half | limited_span_adjustment | 18 | False | 0 | 0 | 0 | 0 | -13.859375 |
| qwen3 | L0_input_prompt_first8_half | limited_span_adjustment | 18 | False | 0 | 0 | 0 | 0 | -13.71875 |
| qwen3 | L0_promptzero_delta_alpha_0.25 | prompt_intact_counterfactual_direction | 18 | True | 0 | 0 | 0 | 0 | -13.875 |
| qwen3 | L0_promptzero_delta_alpha_0.05 | prompt_intact_counterfactual_direction | 18 | True | 0 | 0 | 0 | 0 | -13.78125 |
| qwen3 | L0_promptzero_delta_alpha_0.1 | prompt_intact_counterfactual_direction | 18 | True | 0 | 0 | 0 | 0 | -13.90625 |
| glm4 | L0_promptzero_delta_alpha_0.05 | prompt_intact_counterfactual_direction | 17 | True | 0 | 0 | 0 | 0 | -14.4554443359375 |
| qwen3 | L0_attn_output_last_scale_0.75 | prompt_intact_output_scale | 18 | True | 0 | 0 | 0 | 0 | -13.890625 |
| qwen3 | L0_promptzero_delta_alpha_0.5 | prompt_intact_counterfactual_direction | 18 | True | 0 | 0 | 0 | 0 | -13.859375 |
| glm4 | L0_attn_output_last_scale_0.50 | prompt_intact_output_scale | 17 | True | 0 | 0 | 0 | 0 | -14.37890625 |
| qwen3 | L0_attn_output_last_scale_0.50 | prompt_intact_output_scale | 18 | True | 0 | 0 | 0 | 0 | -13.875 |
| glm4 | L0_attn_output_last_scale_0.75 | prompt_intact_output_scale | 17 | True | 0 | 0 | 0 | 0 | -14.24609375 |
| qwen3 | L0_input_period_half | limited_span_adjustment | 18 | False | 0 | 0 | 0 | 0 | -13.96875 |
| deepseek7b | L0_promptzero_delta_alpha_0.25 | prompt_intact_counterfactual_direction | 33 | True | 0 | 0 | 0 | 0 | -14.546875 |
| glm4 | L0_promptzero_delta_alpha_0.5 | prompt_intact_counterfactual_direction | 17 | True | 0 | 0 | 0 | 0 | -14.412109375 |
| glm4 | L0_input_prompt_last8_half | limited_span_adjustment | 17 | False | 0 | 0 | 0 | 0 | -14.622314453125 |
| deepseek7b | L0_input_prompt_last8_zero | limited_span_adjustment | 33 | False | 0 | 0 | 0 | 0 | -27.703125 |
