# Phase 909 L0 attention source-span EOS boundary audit

## Overall

- models: qwen3, glm4, deepseek7b
- continuation_suppressed: 436
- direct_eos_lift: 247
- eos_rank_improved: 257
- eos_rank_improved_100: 247
- eos_rank_improved_1000: 193
- next_category_changed: 224
- next_top_changed: 327
- patched_eos_top1: 0
- patched_eos_top10: 47
- patched_eos_top50: 52
- protocol_suppressed: 507
- rows: 612

## Model Summaries

| model | rows | eos top1 | eos top10 | eos top50 | direct eos lift | cont suppressed | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 162 | 0 | 0 | 0 | 65 | 126 | source_span_audit_improves_eos_but_not_near |
| glm4 | 153 | 0 | 15 | 17 | 106 | 104 | source_span_audit_reaches_eos_top10 |
| deepseek7b | 297 | 0 | 32 | 35 | 76 | 206 | source_span_audit_reaches_eos_top10 |

## Top Controls

| model | control | span | factor | rows | eos top10 | eos top50 | lift | suppress | median margin delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | L0_attn_input_prompt_all_zero | prompt_all | 0.0 | 33 | 32 | 33 | 33 | 33 | 11.796875 |
| glm4 | L0_attn_input_prompt_all_zero | prompt_all | 0.0 | 17 | 15 | 15 | 15 | 17 | 12.4833984375 |
| deepseek7b | L0_attn_input_period_zero | period_token | 0.0 | 33 | 0 | 2 | 11 | 25 | -14.296875 |
| glm4 | L0_attn_input_prompt_last8_zero | prompt_last8 | 0.0 | 17 | 0 | 1 | 4 | 14 | -1.8154296875 |
| glm4 | L0_attn_input_last8_before_period_zero | last8_before_period | 0.0 | 17 | 0 | 1 | 4 | 14 | -2.0712890625 |
| qwen3 | L0_attn_input_prompt_first8_zero | prompt_first8 | 0.0 | 18 | 0 | 0 | 16 | 16 | 3.125 |
| glm4 | L0_attn_input_prompt_first8_zero | prompt_first8 | 0.0 | 17 | 0 | 0 | 14 | 13 | 1.11767578125 |
| qwen3 | L0_attn_input_prompt_all_zero | prompt_all | 0.0 | 18 | 0 | 0 | 14 | 18 | 3.65625 |
| glm4 | L0_attn_input_prompt_all_half | prompt_all | 0.5 | 17 | 0 | 0 | 16 | 14 | 0.7763671875 |
| glm4 | L0_attn_input_period_zero | period_token | 0.0 | 17 | 0 | 0 | 15 | 7 | 0.09375 |
| glm4 | L0_attn_input_period_half | period_token | 0.5 | 17 | 0 | 0 | 14 | 9 | 0.0765380859375 |
| deepseek7b | L0_attn_input_prompt_all_half | prompt_all | 0.5 | 33 | 0 | 0 | 3 | 32 | -0.09375 |
| deepseek7b | L0_attn_input_period_half | period_token | 0.5 | 33 | 0 | 0 | 24 | 7 | -0.125 |
| qwen3 | L0_attn_input_period_zero | period_token | 0.0 | 18 | 0 | 0 | 6 | 18 | -1.0234375 |
| deepseek7b | L0_attn_input_prompt_first8_zero | prompt_first8 | 0.0 | 33 | 0 | 0 | 1 | 33 | 0.38671875 |
| glm4 | L0_attn_input_answer_prefix_all_zero | answer_prefix_all | 0.0 | 17 | 0 | 0 | 12 | 8 | 0.001953125 |
| glm4 | L0_attn_input_answer_prefix_last_zero | answer_prefix_last | 0.0 | 17 | 0 | 0 | 12 | 8 | 0.001953125 |
| qwen3 | L0_attn_input_prompt_all_half | prompt_all | 0.5 | 18 | 0 | 0 | 16 | 17 | 0.671875 |
| qwen3 | L0_attn_input_last8_before_period_zero | last8_before_period | 0.0 | 18 | 0 | 0 | 4 | 15 | -0.9921875 |
| qwen3 | L0_attn_input_prompt_last8_zero | prompt_last8 | 0.0 | 18 | 0 | 0 | 2 | 15 | -0.2890625 |
| qwen3 | L0_attn_input_answer_prefix_last_zero | answer_prefix_last | 0.0 | 18 | 0 | 0 | 4 | 9 | -0.9921875 |
| qwen3 | L0_attn_input_period_half | period_token | 0.5 | 18 | 0 | 0 | 0 | 9 | -0.109375 |
| qwen3 | L0_attn_input_answer_prefix_all_zero | answer_prefix_all | 0.0 | 18 | 0 | 0 | 3 | 9 | -1.09375 |
| deepseek7b | L0_attn_input_last8_before_period_zero | last8_before_period | 0.0 | 33 | 0 | 0 | 2 | 17 | -9.810546875 |
| deepseek7b | L0_attn_input_prompt_last8_zero | prompt_last8 | 0.0 | 33 | 0 | 0 | 2 | 17 | -11.515625 |
| deepseek7b | L0_attn_input_answer_prefix_all_zero | answer_prefix_all | 0.0 | 33 | 0 | 0 | 0 | 21 | -11.671875 |
| deepseek7b | L0_attn_input_answer_prefix_last_zero | answer_prefix_last | 0.0 | 33 | 0 | 0 | 0 | 21 | -11.671875 |
