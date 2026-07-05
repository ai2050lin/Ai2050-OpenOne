# Phase 918 L39 MLP channel a-blocker suppressor localization

## Overall

- models: qwen3, glm4, deepseek7b
- boundary_margin_nonnegative: 0
- boundary_rows: 12
- boundary_top1: 0
- boundary_top5: 8
- channel_blocker_suppressed: 84
- channel_margin_nonnegative: 125
- channel_promoted_margin: 125
- channel_promoted_top1: 125
- channel_promoted_top5: 66
- channel_rank_improved: 339
- channel_rows: 396
- channel_strict_clean_candidate: 93
- channel_top1: 125
- channel_top10: 371
- channel_top5: 322
- rows: 408
- selected_phase915_l39_candidates: 12
- weak_channel_candidate: 306

## Model Summaries

| model | selected | rows | channel rows | channel top1 | channel margin>=0 | promoted margin | promoted top5 | weak channel | blocker suppressed | strict | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |
| glm4 | 12 | 408 | 396 | 125 | 125 | 125 | 66 | 306 | 84 | 93 | l39_channel_strict_clean_candidate_found |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |

## Top Controls

| model | control | family | group | factor | rows | top1 | margin>=0 | promoted margin | promoted top5 | weak | rank improved | blocker suppressed | median margin delta | mean eos delta | median blocker delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | L39_mlp_channels_margin_support_pos_64_scale_2 | l39_channel_amplify | margin_support_pos_64 | 2.0 | 12 | 12 | 12 | 12 | 4 | 12 | 12 | 0 | 3.359375 | 4.166666666666667 | 0.625 |
| glm4 | L39_mlp_channels_margin_support_pos_64_scale_1.75 | l39_channel_amplify | margin_support_pos_64 | 1.75 | 12 | 12 | 12 | 12 | 4 | 12 | 12 | 0 | 2.640625 | 3.171875 | 0.5 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0 | l39_channel_suppress | a_blocker_support_64 | 0.0 | 12 | 10 | 10 | 10 | 4 | 12 | 12 | 0 | 2.25 | 4.208333333333333 | 1.8125 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0 | l39_channel_suppress | margin_support_neg_64 | 0.0 | 12 | 10 | 10 | 10 | 4 | 12 | 12 | 0 | 2.25 | 4.208333333333333 | 1.8125 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0.125 | l39_channel_suppress | a_blocker_support_64 | 0.125 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.9375 | 3.859375 | 1.90625 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0.125 | l39_channel_suppress | margin_support_neg_64 | 0.125 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.9375 | 3.859375 | 1.90625 |
| glm4 | L39_mlp_channels_margin_support_pos_64_scale_1.5 | l39_channel_amplify | margin_support_pos_64 | 1.5 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.734375 | 2.1354166666666665 | 0.375 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0.25 | l39_channel_suppress | a_blocker_support_64 | 0.25 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.5625 | 3.375 | 1.78125 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0.25 | l39_channel_suppress | margin_support_neg_64 | 0.25 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.5625 | 3.375 | 1.78125 |
| glm4 | L39_mlp_channels_eos_support_64_scale_2 | l39_channel_amplify | eos_support_64 | 2.0 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.5 | 3.5208333333333335 | 2.09375 |
| glm4 | L39_mlp_channels_margin_support_pos_64_scale_1.375 | l39_channel_amplify | margin_support_pos_64 | 1.375 | 12 | 8 | 8 | 8 | 4 | 12 | 12 | 0 | 1.296875 | 1.6145833333333333 | 0.375 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0.375 | l39_channel_suppress | a_blocker_support_64 | 0.375 | 12 | 8 | 8 | 8 | 2 | 12 | 12 | 0 | 1.25 | 2.8125 | 1.625 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0.375 | l39_channel_suppress | margin_support_neg_64 | 0.375 | 12 | 8 | 8 | 8 | 2 | 12 | 12 | 0 | 1.25 | 2.8125 | 1.625 |
| glm4 | L39_mlp_channels_eos_support_64_scale_1.75 | l39_channel_amplify | eos_support_64 | 1.75 | 12 | 5 | 5 | 5 | 4 | 12 | 12 | 0 | 1.0625 | 2.6979166666666665 | 1.6875 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0.5 | l39_channel_suppress | a_blocker_support_64 | 0.5 | 12 | 2 | 2 | 2 | 2 | 12 | 12 | 0 | 0.8125 | 2.1979166666666665 | 1.3125 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0.5 | l39_channel_suppress | margin_support_neg_64 | 0.5 | 12 | 2 | 2 | 2 | 2 | 12 | 12 | 0 | 0.8125 | 2.1979166666666665 | 1.3125 |
| glm4 | L39_mlp_channels_margin_support_pos_64_scale_1.25 | l39_channel_amplify | margin_support_pos_64 | 1.25 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0 | 0.859375 | 1.09375 | 0.25 |
| glm4 | L39_mlp_channels_eos_support_64_scale_1.5 | l39_channel_amplify | eos_support_64 | 1.5 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0 | 0.703125 | 1.8333333333333333 | 1.09375 |
| glm4 | L39_mlp_channels_eos_support_64_scale_1.375 | l39_channel_amplify | eos_support_64 | 1.375 | 12 | 0 | 0 | 0 | 2 | 12 | 12 | 0 | 0.484375 | 1.3645833333333333 | 0.875 |
| glm4 | L39_mlp_channels_eos_support_64_scale_1.25 | l39_channel_amplify | eos_support_64 | 1.25 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.375 | 0.9270833333333334 | 0.5625 |
| glm4 | L39_mlp_channels_margin_support_pos_64_scale_1.1 | l39_channel_amplify | margin_support_pos_64 | 1.1 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.359375 | 0.46875 | 0.125 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0.75 | l39_channel_suppress | a_blocker_support_64 | 0.75 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.34375 | 1.0208333333333333 | 0.6875 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0.75 | l39_channel_suppress | margin_support_neg_64 | 0.75 | 12 | 0 | 0 | 0 | 1 | 12 | 12 | 0 | 0.34375 | 1.0208333333333333 | 0.6875 |
| glm4 | L39_mlp_channels_eos_support_64_scale_1.1 | l39_channel_amplify | eos_support_64 | 1.1 | 12 | 0 | 0 | 0 | 0 | 12 | 12 | 0 | 0.171875 | 0.3723958333333333 | 0.25 |
| glm4 | L39_mlp_channels_a_blocker_support_64_scale_0.875 | l39_channel_suppress | a_blocker_support_64 | 0.875 | 12 | 0 | 0 | 0 | 0 | 9 | 9 | 0 | 0.1875 | 0.5 | 0.3125 |
| glm4 | L39_mlp_channels_margin_support_neg_64_scale_0.875 | l39_channel_suppress | margin_support_neg_64 | 0.875 | 12 | 0 | 0 | 0 | 0 | 9 | 9 | 0 | 0.1875 | 0.5 | 0.3125 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0.375 | l39_channel_suppress | a_logit_support_64 | 0.375 | 12 | 0 | 0 | 0 | 0 | 0 | 10 | 12 | 0.5 | -1.3541666666666667 | -2.96875 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0.25 | l39_channel_suppress | a_logit_support_64 | 0.25 | 12 | 0 | 0 | 0 | 0 | 0 | 8 | 12 | 0.375 | -1.6588541666666667 | -3.5625 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0.125 | l39_channel_suppress | a_logit_support_64 | 0.125 | 12 | 0 | 0 | 0 | 0 | 0 | 6 | 12 | 0.28125 | -1.9479166666666667 | -4.140625 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0.5 | l39_channel_suppress | a_logit_support_64 | 0.5 | 12 | 0 | 0 | 0 | 0 | 0 | 4 | 12 | 0.5625 | -1.0625 | -2.359375 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0.875 | l39_channel_suppress | a_logit_support_64 | 0.875 | 12 | 0 | 0 | 0 | 0 | 0 | 3 | 12 | 0.3125 | -0.23177083333333334 | -0.5625 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0.75 | l39_channel_suppress | a_logit_support_64 | 0.75 | 12 | 0 | 0 | 0 | 0 | 0 | 2 | 12 | 0.65625 | -0.515625 | -1.1875 |
| glm4 | L39_mlp_channels_a_logit_support_64_scale_0 | l39_channel_suppress | a_logit_support_64 | 0.0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0.15625 | -2.2265625 | -4.734375 |
| glm4 | boundary_precondition_only | boundary_precondition | None | None | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | None | None | None |

## Top Groups

| model | group | family | rows | top1 | margin>=0 | promoted top5 | weak | rank improved | blocker suppressed | median margin delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | margin_support_pos_64 | l39_channel_amplify | 72 | 40 | 40 | 19 | 72 | 72 | 0 | 1.46875 |
| glm4 | a_blocker_support_64 | l39_channel_suppress | 84 | 36 | 36 | 17 | 81 | 81 | 0 | 1.125 |
| glm4 | margin_support_neg_64 | l39_channel_suppress | 84 | 36 | 36 | 17 | 81 | 81 | 0 | 1.125 |
| glm4 | eos_support_64 | l39_channel_amplify | 72 | 13 | 13 | 13 | 72 | 72 | 0 | 0.5625 |
| glm4 | a_logit_support_64 | l39_channel_suppress | 84 | 0 | 0 | 0 | 0 | 33 | 84 | 0.40625 |
