# Phase 918 L39 MLP channel a-blocker suppressor localization

## Overall

- models: qwen3, glm4, deepseek7b
- boundary_margin_nonnegative: 0
- boundary_rows: 12
- boundary_top1: 0
- boundary_top5: 8
- channel_blocker_suppressed: 72
- channel_margin_nonnegative: 80
- channel_promoted_margin: 80
- channel_promoted_top1: 80
- channel_promoted_top5: 76
- channel_rank_improved: 407
- channel_rows: 528
- channel_strict_clean_candidate: 56
- channel_top1: 80
- channel_top10: 489
- channel_top5: 407
- rows: 540
- selected_phase915_l39_candidates: 12
- weak_channel_candidate: 370

## Model Summaries

| model | selected | rows | channel rows | channel top1 | channel margin>=0 | promoted margin | promoted top5 | weak channel | blocker suppressed | strict | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |
| glm4 | 12 | 540 | 528 | 80 | 80 | 80 | 76 | 370 | 72 | 56 | l39_channel_strict_clean_candidate_found |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |

## Top Controls

| model | control | family | group | factor | rows | top1 | margin>=0 | promoted margin | promoted top5 | weak | rank improved | blocker suppressed | median margin delta | mean eos delta | median blocker delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | L39_mlp_channels_margin_support_pos_64_scale_2 | l39_channel_amplify | margin_support_pos_64 | 2.0 | 12 | 12 | 12 | 12 | 4 | 12 | 12 | 0 | 3.375 | 4.177083333333333 | 0.75 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0 | l39_channel_suppress | a_blocker_support_64 | 0.0 | 12 | 10 | 10 | 10 | 4 | 12 | 12 | 0 | 2.25 | 4.223958333333333 | 1.84375 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0 | l39_channel_suppress | margin_support_neg_64 | 0.0 | 12 | 10 | 10 | 10 | 4 | 12 | 12 | 0 | 2.25 | 4.223958333333333 | 1.84375 |
| glm4 | L39_mlp_channels_margin_support_pos_64_scale_1.5 | l39_channel_amplify | margin_support_pos_64 | 1.5 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.671875 | 2.140625 | 0.5 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0.25 | l39_channel_suppress | a_blocker_support_64 | 0.25 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.5625 | 3.3854166666666665 | 1.8125 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0.25 | l39_channel_suppress | margin_support_neg_64 | 0.25 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.5625 | 3.3854166666666665 | 1.8125 |
| glm4 | L39_mlp_channels_eos_support_64_scale_2 | l39_channel_amplify | eos_support_64 | 2.0 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.5 | 3.5208333333333335 | 2.09375 |
| glm4 | L39_mlp_channels_margin_support_pos_32_scale_2 | l39_channel_amplify | margin_support_pos_32 | 2.0 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.453125 | 2.2760416666666665 | 0.8125 |
| glm4 | L39_mlp_channels_band_blocker_support_64_scale_0 | l39_channel_suppress | band_blocker_support_64 | 0.0 | 12 | 2 | 2 | 2 | 4 | 12 | 12 | 0 | 0.96875 | 4.520833333333333 | 3.5 |
| glm4 | L39_mlp_channels_band_blocker_support_64_scale_0.25 | l39_channel_suppress | band_blocker_support_64 | 0.25 | 12 | 2 | 2 | 2 | 4 | 12 | 12 | 0 | 0.578125 | 3.6354166666666665 | 3.0 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0.5 | l39_channel_suppress | a_blocker_support_64 | 0.5 | 12 | 2 | 2 | 2 | 2 | 12 | 12 | 0 | 0.8125 | 2.1979166666666665 | 1.34375 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0.5 | l39_channel_suppress | margin_support_neg_64 | 0.5 | 12 | 2 | 2 | 2 | 2 | 12 | 12 | 0 | 0.8125 | 2.1979166666666665 | 1.34375 |
| glm4 | L39_mlp_channels_eos_support_32_scale_2 | l39_channel_amplify | eos_support_32 | 2.0 | 12 | 0 | 0 | 0 | 4 | 12 | 12 | 0 | 1.0 | 2.3697916666666665 | 1.4375 |
| glm4 | L39_mlp_channels_margin_support_pos_64_scale_1.25 | l39_channel_amplify | margin_support_pos_64 | 1.25 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0 | 0.796875 | 1.0885416666666667 | 0.3125 |
| glm4 | L39_mlp_channels_eos_support_64_scale_1.5 | l39_channel_amplify | eos_support_64 | 1.5 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0 | 0.6875 | 1.8072916666666667 | 1.09375 |
| glm4 | L39_mlp_channels_a_blocker_support_32_scale_0 | l39_channel_suppress | a_blocker_support_32 | 0.0 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0 | 0.625 | 2.5989583333333335 | 1.96875 |
| glm4 | L39_mlp_channels_margin_support_neg_32_scale_0 | l39_channel_suppress | margin_support_neg_32 | 0.0 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0 | 0.625 | 2.5989583333333335 | 1.96875 |
| glm4 | L39_mlp_channels_eos_support_32_scale_1.5 | l39_channel_amplify | eos_support_32 | 1.5 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0 | 0.453125 | 1.1822916666666667 | 0.71875 |
| glm4 | L39_mlp_channels_band_blocker_support_64_scale_0.5 | l39_channel_suppress | band_blocker_support_64 | 0.5 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0 | 0.328125 | 2.359375 | 2.03125 |
| glm4 | L39_mlp_channels_a_blocker_support_32_scale_0.25 | l39_channel_suppress | a_blocker_support_32 | 0.25 | 12 | 0 | 0 | 0 | 2 | 10 | 12 | 0 | 0.4375 | 2.0104166666666665 | 1.59375 |
| glm4 | L39_mlp_channels_margin_support_neg_32_scale_0.25 | l39_channel_suppress | margin_support_neg_32 | 0.25 | 12 | 0 | 0 | 0 | 2 | 10 | 12 | 0 | 0.4375 | 2.0104166666666665 | 1.59375 |
| glm4 | L39_mlp_channels_margin_support_pos_32_scale_1.5 | l39_channel_amplify | margin_support_pos_32 | 1.5 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.6875 | 1.1510416666666667 | 0.4375 |
| glm4 | L39_mlp_channels_eos_support_64_scale_1.25 | l39_channel_amplify | eos_support_64 | 1.25 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.375 | 0.9270833333333334 | 0.5625 |
| glm4 | L39_mlp_channels_margin_support_pos_32_scale_1.25 | l39_channel_amplify | margin_support_pos_32 | 1.25 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.359375 | 0.5989583333333334 | 0.25 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0.75 | l39_channel_suppress | a_blocker_support_64 | 0.75 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.328125 | 1.0260416666666667 | 0.6875 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0.75 | l39_channel_suppress | margin_support_neg_64 | 0.75 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.328125 | 1.0260416666666667 | 0.6875 |
| glm4 | L39_mlp_channels_eos_support_32_scale_1.25 | l39_channel_amplify | eos_support_32 | 1.25 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.25 | 0.6197916666666666 | 0.375 |
| glm4 | L39_mlp_channels_band_blocker_support_64_scale_0.75 | l39_channel_suppress | band_blocker_support_64 | 0.75 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.125 | 1.1145833333333333 | 1.0 |
| glm4 | L39_mlp_channels_a_blocker_support_32_scale_0.5 | l39_channel_suppress | a_blocker_support_32 | 0.5 | 12 | 0 | 0 | 0 | 1 | 10 | 12 | 0 | 0.25 | 1.3020833333333333 | 1.0625 |
| glm4 | L39_mlp_channels_margin_support_neg_32_scale_0.5 | l39_channel_suppress | margin_support_neg_32 | 0.5 | 12 | 0 | 0 | 0 | 1 | 10 | 12 | 0 | 0.25 | 1.3020833333333333 | 1.0625 |
| glm4 | L39_mlp_channels_a_blocker_support_32_scale_0.75 | l39_channel_suppress | a_blocker_support_32 | 0.75 | 12 | 0 | 0 | 0 | 1 | 9 | 12 | 0 | 0.125 | 0.640625 | 0.53125 |
| glm4 | L39_mlp_channels_margin_support_neg_32_scale_0.75 | l39_channel_suppress | margin_support_neg_32 | 0.75 | 12 | 0 | 0 | 0 | 1 | 9 | 12 | 0 | 0.125 | 0.640625 | 0.53125 |
| glm4 | L39_mlp_channels_top_abs_64_scale_2 | l39_channel_control | top_abs_64 | 2.0 | 12 | 0 | 0 | 0 | 1 | 0 | 1 | 12 | 0.5 | -0.23697916666666666 | -0.6875 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0.25 | l39_channel_suppress | a_logit_support_64 | 0.25 | 12 | 0 | 0 | 0 | 0 | 0 | 8 | 12 | 0.375 | -1.6614583333333333 | -3.53125 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0.5 | l39_channel_suppress | a_logit_support_64 | 0.5 | 12 | 0 | 0 | 0 | 0 | 0 | 4 | 12 | 0.5625 | -1.0651041666666667 | -2.34375 |
| glm4 | L39_mlp_channels_top_abs_64_scale_0.5 | l39_channel_control | top_abs_64 | 0.5 | 12 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | -0.34375 | 0.028645833333333332 | 0.34375 |
| glm4 | L39_mlp_channels_top_abs_64_scale_0 | l39_channel_control | top_abs_64 | 0.0 | 12 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | -0.8125 | -0.359375 | 0.375 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0.75 | l39_channel_suppress | a_logit_support_64 | 0.75 | 12 | 0 | 0 | 0 | 0 | 0 | 2 | 12 | 0.65625 | -0.5182291666666666 | -1.1875 |
| glm4 | L39_mlp_channels_top_abs_64_scale_1.5 | l39_channel_control | top_abs_64 | 1.5 | 12 | 0 | 0 | 0 | 0 | 0 | 2 | 12 | 0.28125 | -0.09375 | -0.34375 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0 | l39_channel_suppress | a_logit_support_64 | 0.0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0.15625 | -2.2291666666666665 | -4.703125 |
| glm4 | boundary_precondition_only | boundary_precondition | None | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | None | None |
| glm4 | L39_mlp_channels_low_abs_64_scale_0 | l39_channel_control | low_abs_64 | 0.0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| glm4 | L39_mlp_channels_low_abs_64_scale_0.5 | l39_channel_control | low_abs_64 | 0.5 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| glm4 | L39_mlp_channels_low_abs_64_scale_1.5 | l39_channel_control | low_abs_64 | 1.5 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| glm4 | L39_mlp_channels_low_abs_64_scale_2 | l39_channel_control | low_abs_64 | 2.0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |

## Top Groups

| model | group | family | rows | top1 | margin>=0 | promoted top5 | weak | rank improved | blocker suppressed | median margin delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | margin_support_pos_64 | l39_channel_amplify | 36 | 20 | 20 | 10 | 36 | 36 | 0 | 1.671875 |
| glm4 | a_blocker_support_64 | l39_channel_suppress | 48 | 20 | 20 | 11 | 48 | 48 | 0 | 1.03125 |
| glm4 | margin_support_neg_64 | l39_channel_suppress | 48 | 20 | 20 | 11 | 48 | 48 | 0 | 1.03125 |
| glm4 | eos_support_64 | l39_channel_amplify | 36 | 8 | 8 | 7 | 36 | 36 | 0 | 0.6875 |
| glm4 | margin_support_pos_32 | l39_channel_amplify | 36 | 8 | 8 | 6 | 36 | 36 | 0 | 0.6875 |
| glm4 | band_blocker_support_64 | l39_channel_suppress | 48 | 4 | 4 | 11 | 48 | 48 | 0 | 0.5 |
| glm4 | eos_support_32 | l39_channel_amplify | 36 | 0 | 0 | 7 | 36 | 36 | 0 | 0.453125 |
| glm4 | a_blocker_support_32 | l39_channel_suppress | 48 | 0 | 0 | 6 | 41 | 48 | 0 | 0.25 |
| glm4 | margin_support_neg_32 | l39_channel_suppress | 48 | 0 | 0 | 6 | 41 | 48 | 0 | 0.25 |
| glm4 | top_abs_64 | l39_channel_control | 48 | 0 | 0 | 1 | 0 | 9 | 24 | -0.03125 |
| glm4 | a_logit_support_64 | l39_channel_suppress | 48 | 0 | 0 | 0 | 0 | 14 | 48 | 0.5 |
| glm4 | low_abs_64 | l39_channel_control | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
