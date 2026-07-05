# Phase 920 consensus L39 signed margin gear holdout controls

## Overall

- models: qwen3, glm4, deepseek7b
- all_margin_nonnegative: 226
- all_rows: 540
- all_strict_clean_candidate: 174
- all_top1: 226
- all_weak_transfer_candidate: 439
- consensus_positive_margin_nonnegative: 226
- consensus_positive_rows: 432
- consensus_positive_strict_clean_candidate: 174
- consensus_positive_top1: 226
- consensus_positive_weak_transfer_candidate: 432
- leave_one_case_margin_nonnegative: 74
- leave_one_case_rows: 144
- leave_one_case_strict_clean_candidate: 58
- leave_one_case_top1: 74
- leave_one_case_weak_transfer_candidate: 144
- leave_one_domain_margin_nonnegative: 74
- leave_one_domain_rows: 144
- leave_one_domain_strict_clean_candidate: 58
- leave_one_domain_top1: 74
- leave_one_domain_weak_transfer_candidate: 144
- negative_margin_nonnegative: 0
- negative_rows: 108
- negative_strict_clean_candidate: 0
- negative_top1: 0
- negative_weak_transfer_candidate: 7
- positive_margin_nonnegative: 226
- positive_rows: 432
- positive_strict_clean_candidate: 174
- positive_top1: 226
- positive_weak_transfer_candidate: 432
- selected_phase915_l39_candidates: 12
- target_state_count: 12

## Model Summaries

| model | selected | target states | positive top1 | positive margin | positive strict | negative top1 | negative margin | negative strict | loo-case strict | loo-domain strict | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |
| glm4 | 12 | 12 | 226 | 226 | 174 | 0 | 0 | 0 | 58 | 58 | consensus_holdout_positive_beats_negative_controls |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_phase915_l39_candidates |

## Top Controls

| model | control | class | family | fold | group | factor | rows | top1 | margin | strict | weak | targets top1 | targets margin | median delta | overlap |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm4 | consensus_margin_support_pos_64_all_train_scale_2 | positive | consensus_positive_margin | all_train | margin_support_pos_64 | 2.0 | 12 | 12 | 12 | 10 | 12 | 12 | 12 | 3.40625 | 57.5 |
| glm4 | consensus_margin_support_pos_64_leave_one_case_scale_2 | positive | consensus_positive_margin | leave_one_case | margin_support_pos_64 | 2.0 | 12 | 12 | 12 | 10 | 12 | 12 | 12 | 3.1875 | 52.0 |
| glm4 | consensus_margin_support_pos_64_leave_one_domain_scale_2 | positive | consensus_positive_margin | leave_one_domain | margin_support_pos_64 | 2.0 | 12 | 12 | 12 | 10 | 12 | 12 | 12 | 3.1875 | 52.0 |
| glm4 | consensus_margin_support_pos_64_all_train_scale_1.75 | positive | consensus_positive_margin | all_train | margin_support_pos_64 | 1.75 | 12 | 12 | 12 | 10 | 12 | 12 | 12 | 2.625 | 57.5 |
| glm4 | consensus_margin_support_pos_64_leave_one_domain_scale_1.75 | positive | consensus_positive_margin | leave_one_domain | margin_support_pos_64 | 1.75 | 12 | 12 | 12 | 10 | 12 | 12 | 12 | 2.40625 | 52.0 |
| glm4 | consensus_margin_support_pos_64_leave_one_case_scale_1.75 | positive | consensus_positive_margin | leave_one_case | margin_support_pos_64 | 1.75 | 12 | 12 | 12 | 10 | 12 | 12 | 12 | 2.390625 | 52.0 |
| glm4 | consensus_a_blocker_support_64_all_train_scale_0 | positive | consensus_suppress_margin_blocker | all_train | a_blocker_support_64 | 0.0 | 12 | 9 | 9 | 7 | 12 | 9 | 9 | 2.4375 | 60.0 |
| glm4 | consensus_margin_support_neg_64_all_train_scale_0 | positive | consensus_suppress_margin_blocker | all_train | margin_support_neg_64 | 0.0 | 12 | 9 | 9 | 7 | 12 | 9 | 9 | 2.4375 | 60.0 |
| glm4 | consensus_a_blocker_support_64_leave_one_case_scale_0 | positive | consensus_suppress_margin_blocker | leave_one_case | a_blocker_support_64 | 0.0 | 12 | 9 | 9 | 7 | 12 | 9 | 9 | 2.1875 | 54.0 |
| glm4 | consensus_a_blocker_support_64_leave_one_domain_scale_0 | positive | consensus_suppress_margin_blocker | leave_one_domain | a_blocker_support_64 | 0.0 | 12 | 9 | 9 | 7 | 12 | 9 | 9 | 2.1875 | 54.0 |
| glm4 | consensus_margin_support_neg_64_leave_one_case_scale_0 | positive | consensus_suppress_margin_blocker | leave_one_case | margin_support_neg_64 | 0.0 | 12 | 9 | 9 | 7 | 12 | 9 | 9 | 2.1875 | 54.0 |
| glm4 | consensus_margin_support_neg_64_leave_one_domain_scale_0 | positive | consensus_suppress_margin_blocker | leave_one_domain | margin_support_neg_64 | 0.0 | 12 | 9 | 9 | 7 | 12 | 9 | 9 | 2.1875 | 54.0 |
| glm4 | consensus_margin_support_pos_64_all_train_scale_1.5 | positive | consensus_positive_margin | all_train | margin_support_pos_64 | 1.5 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.6875 | 57.5 |
| glm4 | consensus_a_blocker_support_64_all_train_scale_0.25 | positive | consensus_suppress_margin_blocker | all_train | a_blocker_support_64 | 0.25 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.6875 | 60.0 |
| glm4 | consensus_margin_support_neg_64_all_train_scale_0.25 | positive | consensus_suppress_margin_blocker | all_train | margin_support_neg_64 | 0.25 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.6875 | 60.0 |
| glm4 | consensus_margin_support_pos_64_leave_one_case_scale_1.5 | positive | consensus_positive_margin | leave_one_case | margin_support_pos_64 | 1.5 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.625 | 52.0 |
| glm4 | consensus_margin_support_pos_64_leave_one_domain_scale_1.5 | positive | consensus_positive_margin | leave_one_domain | margin_support_pos_64 | 1.5 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.625 | 52.0 |
| glm4 | consensus_a_blocker_support_64_leave_one_case_scale_0.25 | positive | consensus_suppress_margin_blocker | leave_one_case | a_blocker_support_64 | 0.25 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.4375 | 54.0 |
| glm4 | consensus_a_blocker_support_64_leave_one_domain_scale_0.25 | positive | consensus_suppress_margin_blocker | leave_one_domain | a_blocker_support_64 | 0.25 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.4375 | 54.0 |
| glm4 | consensus_margin_support_neg_64_leave_one_case_scale_0.25 | positive | consensus_suppress_margin_blocker | leave_one_case | margin_support_neg_64 | 0.25 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.4375 | 54.0 |
| glm4 | consensus_margin_support_neg_64_leave_one_domain_scale_0.25 | positive | consensus_suppress_margin_blocker | leave_one_domain | margin_support_neg_64 | 0.25 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.4375 | 54.0 |
| glm4 | consensus_margin_support_pos_64_all_train_scale_1.375 | positive | consensus_positive_margin | all_train | margin_support_pos_64 | 1.375 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.28125 | 57.5 |
| glm4 | consensus_margin_support_pos_64_leave_one_case_scale_1.375 | positive | consensus_positive_margin | leave_one_case | margin_support_pos_64 | 1.375 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.1875 | 52.0 |
| glm4 | consensus_margin_support_pos_64_leave_one_domain_scale_1.375 | positive | consensus_positive_margin | leave_one_domain | margin_support_pos_64 | 1.375 | 12 | 8 | 8 | 6 | 12 | 8 | 8 | 1.1875 | 52.0 |
| glm4 | consensus_a_blocker_support_64_all_train_scale_0.5 | positive | consensus_suppress_margin_blocker | all_train | a_blocker_support_64 | 0.5 | 12 | 2 | 2 | 0 | 12 | 2 | 2 | 1.0 | 60.0 |
| glm4 | consensus_margin_support_neg_64_all_train_scale_0.5 | positive | consensus_suppress_margin_blocker | all_train | margin_support_neg_64 | 0.5 | 12 | 2 | 2 | 0 | 12 | 2 | 2 | 1.0 | 60.0 |
| glm4 | consensus_margin_support_pos_64_all_train_scale_1.25 | positive | consensus_positive_margin | all_train | margin_support_pos_64 | 1.25 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.84375 | 57.5 |
| glm4 | consensus_a_blocker_support_64_leave_one_case_scale_0.5 | positive | consensus_suppress_margin_blocker | leave_one_case | a_blocker_support_64 | 0.5 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.8125 | 54.0 |
| glm4 | consensus_a_blocker_support_64_leave_one_domain_scale_0.5 | positive | consensus_suppress_margin_blocker | leave_one_domain | a_blocker_support_64 | 0.5 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.8125 | 54.0 |
| glm4 | consensus_margin_support_neg_64_leave_one_case_scale_0.5 | positive | consensus_suppress_margin_blocker | leave_one_case | margin_support_neg_64 | 0.5 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.8125 | 54.0 |
| glm4 | consensus_margin_support_neg_64_leave_one_domain_scale_0.5 | positive | consensus_suppress_margin_blocker | leave_one_domain | margin_support_neg_64 | 0.5 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.8125 | 54.0 |
| glm4 | consensus_margin_support_pos_64_leave_one_case_scale_1.25 | positive | consensus_positive_margin | leave_one_case | margin_support_pos_64 | 1.25 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.765625 | 52.0 |
| glm4 | consensus_margin_support_pos_64_leave_one_domain_scale_1.25 | positive | consensus_positive_margin | leave_one_domain | margin_support_pos_64 | 1.25 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.765625 | 52.0 |
| glm4 | consensus_margin_support_pos_64_all_train_scale_1.125 | positive | consensus_positive_margin | all_train | margin_support_pos_64 | 1.125 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.4375 | 57.5 |
| glm4 | consensus_margin_support_pos_64_leave_one_case_scale_1.125 | positive | consensus_positive_margin | leave_one_case | margin_support_pos_64 | 1.125 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.421875 | 52.0 |
| glm4 | consensus_margin_support_pos_64_leave_one_domain_scale_1.125 | positive | consensus_positive_margin | leave_one_domain | margin_support_pos_64 | 1.125 | 12 | 0 | 0 | 0 | 12 | 0 | 0 | 0.421875 | 52.0 |
| glm4 | rotated_consensus_margin_support_pos_64_scale_1.75 | negative | negative_rotated_consensus | all_train | margin_support_pos_64 | 1.75 | 12 | 0 | 0 | 0 | 2 | 0 | 0 | 0.125 | 0.0 |
| glm4 | rotated_consensus_margin_support_pos_64_scale_2 | negative | negative_rotated_consensus | all_train | margin_support_pos_64 | 2.0 | 12 | 0 | 0 | 0 | 2 | 0 | 0 | 0.125 | 0.0 |
| glm4 | random_all_64_scale_1.75 | negative | negative_random_all | none | random_all_64 | 1.75 | 12 | 0 | 0 | 0 | 2 | 0 | 0 | 0.0625 | 0.0 |
| glm4 | random_all_64_scale_2 | negative | negative_random_all | none | random_all_64 | 2.0 | 12 | 0 | 0 | 0 | 1 | 0 | 0 | -0.015625 | 0.0 |
| glm4 | consensus_a_logit_support_64_scale_0.5 | negative | negative_a_logit_only | all_train | a_logit_support_64 | 0.5 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5 | 59.0 |
| glm4 | consensus_a_logit_support_64_scale_0.25 | negative | negative_a_logit_only | all_train | a_logit_support_64 | 0.25 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.28125 | 59.0 |
| glm4 | rotated_consensus_margin_support_pos_64_scale_1.375 | negative | negative_rotated_consensus | all_train | margin_support_pos_64 | 1.375 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0625 | 0.0 |
| glm4 | consensus_a_logit_support_64_scale_0 | negative | negative_a_logit_only | all_train | a_logit_support_64 | 0.0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03125 | 59.0 |
| glm4 | random_all_64_scale_1.375 | negative | negative_random_all | none | random_all_64 | 1.375 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0.015625 | 0.0 |
