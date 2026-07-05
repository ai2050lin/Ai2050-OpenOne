# Phase 919 frozen L39 signed margin group transfer validation

## Overall

- models: qwen3, glm4, deepseek7b
- all_margin_nonnegative: 1438
- all_rows: 3024
- all_strict_clean_candidate: 1078
- all_top1: 1438
- all_weak_transfer_candidate: 2304
- cross_domain_margin_nonnegative: 752
- cross_domain_rows: 1722
- cross_domain_strict_clean_candidate: 552
- cross_domain_top1: 752
- cross_domain_weak_transfer_candidate: 1312
- cross_margin_nonnegative: 1313
- cross_rows: 2772
- cross_same_case_margin_nonnegative: 513
- cross_same_case_rows: 882
- cross_same_case_strict_clean_candidate: 417
- cross_same_case_top1: 513
- cross_same_case_weak_transfer_candidate: 672
- cross_same_domain_margin_nonnegative: 48
- cross_same_domain_rows: 168
- cross_same_domain_strict_clean_candidate: 16
- cross_same_domain_top1: 48
- cross_same_domain_weak_transfer_candidate: 128
- cross_strict_clean_candidate: 985
- cross_target_states_with_margin: 12
- cross_target_states_with_strict: 10
- cross_target_states_with_top1: 12
- cross_target_states_with_weak: 12
- cross_top1: 1313
- cross_weak_transfer_candidate: 2112
- selected_phase915_l39_candidates: 12
- self_margin_nonnegative: 125
- self_rows: 252
- self_strict_clean_candidate: 93
- self_top1: 125
- self_weak_transfer_candidate: 192
- target_state_count: 12

## Model Summaries

| model | selected | target states | cross rows | cross top1 | cross margin>=0 | cross weak | cross strict | cross targets top1 | cross targets margin | cross targets weak | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |
| glm4 | 12 | 12 | 2772 | 1313 | 1313 | 2112 | 985 | 12 | 12 | 12 | frozen_cross_strict_clean_transfer_found |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |

## Transfer Kinds

| model | kind | rows | top1 | margin>=0 | weak | strict | median margin delta | mean eos delta | overlap median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | cross_domain | 1722 | 752 | 752 | 1312 | 552 | 1.1875 | 1.8875762195121952 | 54.0 |
| glm4 | cross_same_case | 882 | 513 | 513 | 672 | 417 | 1.25 | 1.9839498299319729 | 64.0 |
| glm4 | self | 252 | 125 | 125 | 192 | 93 | 1.25 | 2.002232142857143 | 64.0 |
| glm4 | cross_same_domain | 168 | 48 | 48 | 128 | 16 | 1.1875 | 1.9579613095238095 | 57.0 |

## Top Controls

| model | control | kind | group | rows | top1 | margin>=0 | weak | strict | target states top1 | target states margin | median margin delta | mean eos delta | overlap median |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | frozen_L39_margin_support_pos_64_scale_2 | cross_domain | margin_support_pos_64 | 82 | 82 | 82 | 82 | 68 | 12 | 12 | 3.125 | 3.9458841463414633 | 51.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.75 | cross_domain | margin_support_pos_64 | 82 | 80 | 80 | 82 | 66 | 12 | 12 | 2.390625 | 3.002286585365854 | 51.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0 | cross_domain | a_blocker_support_64 | 82 | 61 | 61 | 82 | 47 | 9 | 9 | 2.25 | 4.060975609756097 | 55.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0 | cross_domain | margin_support_neg_64 | 82 | 61 | 61 | 82 | 47 | 9 | 9 | 2.25 | 4.060975609756097 | 55.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.125 | cross_domain | a_blocker_support_64 | 82 | 53 | 53 | 82 | 39 | 9 | 9 | 1.875 | 3.692073170731707 | 55.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.125 | cross_domain | margin_support_neg_64 | 82 | 53 | 53 | 82 | 39 | 9 | 9 | 1.875 | 3.692073170731707 | 55.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.5 | cross_domain | margin_support_pos_64 | 82 | 50 | 50 | 82 | 36 | 8 | 8 | 1.5625 | 2.0373475609756095 | 51.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.25 | cross_domain | a_blocker_support_64 | 82 | 50 | 50 | 82 | 36 | 8 | 8 | 1.5 | 3.214176829268293 | 55.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.25 | cross_domain | margin_support_neg_64 | 82 | 50 | 50 | 82 | 36 | 8 | 8 | 1.5 | 3.214176829268293 | 55.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.375 | cross_domain | margin_support_pos_64 | 82 | 50 | 50 | 82 | 36 | 8 | 8 | 1.1875 | 1.5297256097560976 | 51.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_2 | cross_same_case | margin_support_pos_64 | 42 | 42 | 42 | 42 | 36 | 10 | 10 | 3.4375 | 4.169642857142857 | 61.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.75 | cross_same_case | margin_support_pos_64 | 42 | 42 | 42 | 42 | 36 | 10 | 10 | 2.6875 | 3.150297619047619 | 61.0 |
| glm4 | frozen_L39_eos_support_64_scale_2 | cross_domain | eos_support_64 | 82 | 47 | 47 | 82 | 33 | 8 | 8 | 1.28125 | 3.3346036585365852 | 50.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.375 | cross_domain | a_blocker_support_64 | 82 | 44 | 44 | 82 | 30 | 8 | 8 | 1.1875 | 2.65625 | 55.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.375 | cross_domain | margin_support_neg_64 | 82 | 44 | 44 | 82 | 30 | 8 | 8 | 1.1875 | 2.65625 | 55.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0 | cross_same_case | a_blocker_support_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 2.25 | 4.163690476190476 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0 | cross_same_case | margin_support_neg_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 2.25 | 4.163690476190476 | 64.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.125 | cross_same_case | a_blocker_support_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 1.9375 | 3.8199404761904763 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.125 | cross_same_case | margin_support_neg_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 1.9375 | 3.8199404761904763 | 64.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.5 | cross_same_case | margin_support_pos_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 1.75 | 2.136904761904762 | 61.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.25 | cross_same_case | a_blocker_support_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 1.59375 | 3.355654761904762 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.25 | cross_same_case | margin_support_neg_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 1.59375 | 3.355654761904762 | 64.0 |
| glm4 | frozen_L39_eos_support_64_scale_2 | cross_same_case | eos_support_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 1.5 | 3.4970238095238093 | 64.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.375 | cross_same_case | margin_support_pos_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 1.3125 | 1.6294642857142858 | 61.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.375 | cross_same_case | a_blocker_support_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 1.25 | 2.787202380952381 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.375 | cross_same_case | margin_support_neg_64 | 42 | 36 | 36 | 42 | 30 | 8 | 8 | 1.25 | 2.787202380952381 | 64.0 |
| glm4 | frozen_L39_eos_support_64_scale_1.75 | cross_same_case | eos_support_64 | 42 | 21 | 21 | 42 | 15 | 5 | 5 | 1.125 | 2.6711309523809526 | 64.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_2 | self | margin_support_pos_64 | 12 | 12 | 12 | 12 | 10 | 12 | 12 | 3.375 | 4.177083333333333 | 64.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.75 | self | margin_support_pos_64 | 12 | 12 | 12 | 12 | 10 | 12 | 12 | 2.578125 | 3.1614583333333335 | 64.0 |
| glm4 | frozen_L39_eos_support_64_scale_1.75 | cross_domain | eos_support_64 | 82 | 23 | 23 | 82 | 9 | 5 | 5 | 0.9375 | 2.5221036585365852 | 50.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0 | self | a_blocker_support_64 | 12 | 10 | 10 | 12 | 8 | 10 | 10 | 2.25 | 4.223958333333333 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0 | self | margin_support_neg_64 | 12 | 10 | 10 | 12 | 8 | 10 | 10 | 2.25 | 4.223958333333333 | 64.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.125 | self | a_blocker_support_64 | 12 | 8 | 8 | 12 | 6 | 8 | 8 | 1.9375 | 3.8697916666666665 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.125 | self | margin_support_neg_64 | 12 | 8 | 8 | 12 | 6 | 8 | 8 | 1.9375 | 3.8697916666666665 | 64.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.5 | self | margin_support_pos_64 | 12 | 8 | 8 | 12 | 6 | 8 | 8 | 1.671875 | 2.140625 | 64.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.25 | self | a_blocker_support_64 | 12 | 8 | 8 | 12 | 6 | 8 | 8 | 1.5625 | 3.3854166666666665 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.25 | self | margin_support_neg_64 | 12 | 8 | 8 | 12 | 6 | 8 | 8 | 1.5625 | 3.3854166666666665 | 64.0 |
| glm4 | frozen_L39_eos_support_64_scale_2 | self | eos_support_64 | 12 | 8 | 8 | 12 | 6 | 8 | 8 | 1.5 | 3.5208333333333335 | 64.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.375 | self | margin_support_pos_64 | 12 | 8 | 8 | 12 | 6 | 8 | 8 | 1.265625 | 1.625 | 64.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.375 | self | a_blocker_support_64 | 12 | 8 | 8 | 12 | 6 | 8 | 8 | 1.25 | 2.8177083333333335 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.375 | self | margin_support_neg_64 | 12 | 8 | 8 | 12 | 6 | 8 | 8 | 1.25 | 2.8177083333333335 | 64.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_2 | cross_same_domain | margin_support_pos_64 | 8 | 8 | 8 | 8 | 6 | 5 | 5 | 3.109375 | 4.046875 | 55.0 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.75 | cross_same_domain | margin_support_pos_64 | 8 | 8 | 8 | 8 | 6 | 5 | 5 | 2.359375 | 3.0625 | 55.0 |
| glm4 | frozen_L39_eos_support_64_scale_1.75 | self | eos_support_64 | 12 | 5 | 5 | 12 | 3 | 5 | 5 | 1.109375 | 2.6979166666666665 | 64.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0 | cross_same_domain | a_blocker_support_64 | 8 | 4 | 4 | 8 | 2 | 4 | 4 | 2.046875 | 4.203125 | 57.5 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0 | cross_same_domain | margin_support_neg_64 | 8 | 4 | 4 | 8 | 2 | 4 | 4 | 2.046875 | 4.203125 | 57.5 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.5 | cross_same_case | a_blocker_support_64 | 42 | 6 | 6 | 42 | 0 | 2 | 2 | 0.875 | 2.1875 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.5 | cross_same_case | margin_support_neg_64 | 42 | 6 | 6 | 42 | 0 | 2 | 2 | 0.875 | 2.1875 | 64.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.5 | cross_domain | a_blocker_support_64 | 82 | 2 | 2 | 82 | 0 | 2 | 2 | 0.875 | 2.0724085365853657 | 55.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.5 | cross_domain | margin_support_neg_64 | 82 | 2 | 2 | 82 | 0 | 2 | 2 | 0.875 | 2.0724085365853657 | 55.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.5 | self | a_blocker_support_64 | 12 | 2 | 2 | 12 | 0 | 2 | 2 | 0.8125 | 2.1979166666666665 | 64.0 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.5 | self | margin_support_neg_64 | 12 | 2 | 2 | 12 | 0 | 2 | 2 | 0.8125 | 2.1979166666666665 | 64.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.125 | cross_same_domain | a_blocker_support_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 1.609375 | 3.828125 | 57.5 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.125 | cross_same_domain | margin_support_neg_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 1.609375 | 3.828125 | 57.5 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.5 | cross_same_domain | margin_support_pos_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 1.546875 | 2.03125 | 55.0 |
| glm4 | frozen_L39_eos_support_64_scale_2 | cross_same_domain | eos_support_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 1.53125 | 3.359375 | 55.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.25 | cross_same_domain | a_blocker_support_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 1.265625 | 3.296875 | 57.5 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.25 | cross_same_domain | margin_support_neg_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 1.265625 | 3.296875 | 57.5 |
| glm4 | frozen_L39_margin_support_pos_64_scale_1.375 | cross_same_domain | margin_support_pos_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 1.171875 | 1.5625 | 55.0 |
| glm4 | frozen_L39_eos_support_64_scale_1.75 | cross_same_domain | eos_support_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 1.078125 | 2.515625 | 55.0 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.375 | cross_same_domain | a_blocker_support_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 0.921875 | 2.75 | 57.5 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.375 | cross_same_domain | margin_support_neg_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 0.921875 | 2.75 | 57.5 |
| glm4 | frozen_L39_a_blocker_support_64_scale_0.5 | cross_same_domain | a_blocker_support_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 0.671875 | 2.125 | 57.5 |
| glm4 | frozen_L39_margin_support_neg_64_scale_0.5 | cross_same_domain | margin_support_neg_64 | 8 | 2 | 2 | 8 | 0 | 2 | 2 | 0.671875 | 2.125 | 57.5 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.25 | cross_same_domain | a_logit_support_64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0.984375 | -1.5859375 | 57.5 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.375 | cross_same_domain | a_logit_support_64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0.953125 | -1.2890625 | 57.5 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.5 | cross_same_domain | a_logit_support_64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0.84375 | -1.046875 | 57.5 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.125 | cross_same_domain | a_logit_support_64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0.84375 | -1.828125 | 57.5 |
| glm4 | frozen_L39_a_logit_support_64_scale_0 | cross_same_domain | a_logit_support_64 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0.71875 | -2.1171875 | 57.5 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.5 | cross_domain | a_logit_support_64 | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0.625 | -1.0586890243902438 | 54.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.375 | cross_domain | a_logit_support_64 | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0.59375 | -1.3422256097560976 | 54.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.5 | self | a_logit_support_64 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5625 | -1.0651041666666667 | 64.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.5 | cross_same_case | a_logit_support_64 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0.546875 | -1.0461309523809523 | 63.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.25 | cross_domain | a_logit_support_64 | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5 | -1.626905487804878 | 54.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.375 | self | a_logit_support_64 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5 | -1.3567708333333333 | 64.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.375 | cross_same_case | a_logit_support_64 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0.421875 | -1.3549107142857142 | 63.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.125 | cross_domain | a_logit_support_64 | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0.40625 | -1.915015243902439 | 54.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.25 | self | a_logit_support_64 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.375 | -1.6614583333333333 | 64.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0 | cross_domain | a_logit_support_64 | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0.34375 | -2.1817835365853657 | 54.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.25 | cross_same_case | a_logit_support_64 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0.3125 | -1.6480654761904763 | 63.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.125 | self | a_logit_support_64 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.28125 | -1.953125 | 64.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0.125 | cross_same_case | a_logit_support_64 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0.203125 | -1.9486607142857142 | 63.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0 | self | a_logit_support_64 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.15625 | -2.2291666666666665 | 64.0 |
| glm4 | frozen_L39_a_logit_support_64_scale_0 | cross_same_case | a_logit_support_64 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0.09375 | -2.2217261904761907 | 63.0 |
