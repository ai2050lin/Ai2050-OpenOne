# Phase 908 L0 attention EOS proximity fine audit

## Overall

- models: qwen3, glm4, deepseek7b
- continuation_suppressed: 924
- direct_eos_lift: 995
- eos_rank_improved: 1108
- eos_rank_improved_100: 897
- eos_rank_improved_1000: 381
- next_category_changed: 326
- next_top_changed: 394
- patched_eos_top1: 0
- patched_eos_top10: 20
- patched_eos_top50: 42
- protocol_suppressed: 1028
- rows: 2452

## Model Summaries

| model | rows | eos top1 | eos top10 | eos top50 | direct eos lift | cont suppressed | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 684 | 0 | 0 | 0 | 265 | 296 | l0_attention_fine_audit_improves_eos_but_not_near |
| glm4 | 646 | 0 | 20 | 42 | 366 | 253 | l0_attention_fine_audit_reaches_eos_top10 |
| deepseek7b | 1122 | 0 | 0 | 0 | 364 | 375 | l0_attention_fine_audit_improves_eos_but_not_near |

## Top Controls

| model | control | family | rows | eos top10 | eos top50 | lift | suppress | median margin delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | L0_attention_last_zero | l0_attention_intensity | 17 | 13 | 15 | 15 | 17 | 12.5980224609375 |
| glm4 | L0_attention_last_negative_half | l0_attention_intensity | 17 | 7 | 12 | 15 | 15 | 12.4105224609375 |
| glm4 | L0_attention_all_zero | position_scope_control | 17 | 0 | 15 | 15 | 17 | 11.4833984375 |
| deepseek7b | L0_attention_all_zero | position_scope_control | 33 | 0 | 0 | 0 | 33 | 5.484375 |
| deepseek7b | L1_attention_last_zero | nearby_layer_control | 33 | 0 | 0 | 12 | 32 | 0.93359375 |
| deepseek7b | L0_attention_last_zero | l0_attention_intensity | 33 | 0 | 0 | 18 | 13 | -2.5 |
| glm4 | L0_mlp_last_zero | component_control | 17 | 0 | 0 | 15 | 11 | 0.083984375 |
| deepseek7b | L0_attention_last_half | l0_attention_intensity | 33 | 0 | 0 | 13 | 29 | 0.04296875 |
| deepseek7b | L0_attention_head16_zero | head_zero | 33 | 0 | 0 | 17 | 13 | 0.0 |
| deepseek7b | L0_attention_head14_zero | head_zero | 33 | 0 | 0 | 19 | 12 | -0.015625 |
| qwen3 | L0_attention_all_zero | position_scope_control | 18 | 0 | 0 | 12 | 18 | 4.09375 |
| glm4 | L1_attention_last_zero | nearby_layer_control | 17 | 0 | 0 | 14 | 11 | 0.125 |
| glm4 | L0_attention_head26_zero | head_zero | 17 | 0 | 0 | 14 | 12 | 0.125 |
| deepseek7b | L0_attention_head6_zero | head_zero | 33 | 0 | 0 | 18 | 6 | 0.0 |
| deepseek7b | L0_attention_head1_zero | head_zero | 33 | 0 | 0 | 14 | 9 | -0.0234375 |
| deepseek7b | L0_attention_head7_zero | head_zero | 33 | 0 | 0 | 15 | 5 | -0.03125 |
| deepseek7b | L0_attention_head18_zero | head_zero | 33 | 0 | 0 | 14 | 14 | -0.03125 |
| deepseek7b | L0_mlp_last_zero | component_control | 33 | 0 | 0 | 13 | 17 | -0.125 |
| glm4 | L0_attention_head9_zero | head_zero | 17 | 0 | 0 | 12 | 2 | 0.0535888671875 |
| glm4 | L0_attention_head24_zero | head_zero | 17 | 0 | 0 | 11 | 3 | -0.0142822265625 |
| deepseek7b | L0_attention_head22_zero | head_zero | 33 | 0 | 0 | 12 | 4 | -0.03125 |
| glm4 | L0_attention_head17_zero | head_zero | 17 | 0 | 0 | 13 | 12 | 0.07568359375 |
| qwen3 | L0_mlp_last_zero | component_control | 18 | 0 | 0 | 8 | 5 | -0.078125 |
| qwen3 | L0_attention_last_negative_half | l0_attention_intensity | 18 | 0 | 0 | 6 | 18 | -1.546875 |
| glm4 | L0_attention_head31_zero | head_zero | 17 | 0 | 0 | 17 | 10 | 0.0863037109375 |
| deepseek7b | L0_attention_head9_zero | head_zero | 33 | 0 | 0 | 17 | 9 | 0.03125 |
| glm4 | L0_attention_head20_zero | head_zero | 17 | 0 | 0 | 5 | 11 | 0.015625 |
| deepseek7b | L0_attention_head8_zero | head_zero | 33 | 0 | 0 | 16 | 6 | 0.0 |
| glm4 | L0_attention_head28_zero | head_zero | 17 | 0 | 0 | 9 | 3 | -0.0225830078125 |
| deepseek7b | L0_attention_head17_zero | head_zero | 33 | 0 | 0 | 7 | 7 | -0.125 |
| glm4 | L0_attention_head27_zero | head_zero | 17 | 0 | 0 | 12 | 4 | 0.046875 |
| qwen3 | L0_attention_head11_zero | head_zero | 18 | 0 | 0 | 10 | 2 | 0.03125 |
| deepseek7b | L0_attention_head20_zero | head_zero | 33 | 0 | 0 | 11 | 12 | 0.0 |
| deepseek7b | L0_attention_head19_zero | head_zero | 33 | 0 | 0 | 8 | 13 | -0.03125 |
| glm4 | L0_attention_head1_zero | head_zero | 17 | 0 | 0 | 13 | 7 | 0.07080078125 |
| glm4 | L0_attention_head13_zero | head_zero | 17 | 0 | 0 | 9 | 2 | 0.002197265625 |
| glm4 | L0_attention_head11_zero | head_zero | 17 | 0 | 0 | 12 | 3 | 0.00048828125 |
| deepseek7b | L0_attention_head11_zero | head_zero | 33 | 0 | 0 | 12 | 9 | 0.0 |
| deepseek7b | L0_attention_head10_zero | head_zero | 33 | 0 | 0 | 11 | 5 | -0.03125 |
| deepseek7b | L0_attention_head12_zero | head_zero | 33 | 0 | 0 | 9 | 3 | -0.0625 |
| deepseek7b | L0_attention_head27_zero | head_zero | 33 | 0 | 0 | 5 | 4 | -0.0625 |
| deepseek7b | L0_attention_head15_zero | head_zero | 33 | 0 | 0 | 0 | 26 | -0.34375 |
| deepseek7b | L0_attention_last_negative_half | l0_attention_intensity | 33 | 0 | 0 | 7 | 30 | -1.65625 |
| glm4 | L0_attention_head19_zero | head_zero | 17 | 0 | 0 | 10 | 5 | 0.025390625 |
| qwen3 | L1_attention_last_zero | nearby_layer_control | 18 | 0 | 0 | 7 | 7 | -0.015625 |
| glm4 | L0_attention_head22_zero | head_zero | 17 | 0 | 0 | 9 | 3 | -0.029296875 |
| deepseek7b | L0_attention_head3_zero | head_zero | 33 | 0 | 0 | 9 | 3 | -0.03125 |
| deepseek7b | L0_attention_head4_zero | head_zero | 33 | 0 | 0 | 6 | 4 | -0.0625 |
| qwen3 | L0_attention_last_zero | l0_attention_intensity | 18 | 0 | 0 | 6 | 17 | -0.515625 |
| glm4 | L0_attention_head0_zero | head_zero | 17 | 0 | 0 | 5 | 10 | 0.04827880859375 |
