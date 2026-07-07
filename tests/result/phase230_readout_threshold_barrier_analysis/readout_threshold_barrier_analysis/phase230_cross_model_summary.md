# Phase 230 readout threshold barrier analysis

input_rows: 1188
barrier_rows: 321
closure_candidate_rows: 59

## Barrier Summary

| spec | group | type | variant | step | winner | rows | target delta | rank improve | remaining gap | margin delta | efficiency | top tokens |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_all_a0.5 | 3 | because_reason | 2 | 0.1250 | 82.0000 | 29.8125 | 0.3750 | 3.0000 | {'Because': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_all_a0.5 | 3 | because_reason | 2 | 0.1250 | 83.0000 | 29.8125 | 0.3750 | 3.0000 | {'Because': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_top64_a0.5 | 3 | because_reason | 2 | 0.1875 | 85.0000 | 29.7500 | 0.4375 | 2.3333 | {'Because': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_top16_a0.5 | 3 | because_reason | 2 | 0.3125 | 126.0000 | 29.6250 | 0.5625 | 1.8000 | {'Because': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_top16_a0.5 | 3 | because_reason | 4 | 0.3125 | 89.0000 | 29.3750 | 0.8125 | 3.5000 | {'Because': 4} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_top64_a0.5 | 3 | because_reason | 2 | 0.2500 | 65.0000 | 29.1875 | 1.0000 | 4.0000 | {'Because': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_top16_a1 | 3 | because_reason | 2 | 0.2500 | 121.0000 | 28.6875 | 1.5000 | 6.0000 | {'Because': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_top16_a1 | 3 | because_reason | 2 | 0.4375 | 170.0000 | 28.2500 | 1.9375 | 4.4286 | {'Because': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_all_a0.5 | 2 | period_stop | 2 | 0.0625 | 17.0000 | 22.6875 | 0.3125 | 5.0000 | {'.\n': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_top64_a0.5 | 2 | period_stop | 2 | 0.3750 | 58.0000 | 22.6250 | 0.3750 | 1.0000 | {'.\n': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_top16_a0.5 | 2 | period_stop | 2 | 0.4375 | 50.0000 | 22.5625 | 0.4375 | 1.0000 | {'.\n': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_all_a1 | 2 | period_stop | 2 | 0.2500 | 61.0000 | 22.2500 | 0.7500 | 3.0000 | {'.\n': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a0.5 | 2 | be_continuation | 2 | 0.0625 | -45.0000 | 21.9375 | -0.5625 | -9.0000 | {' is': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_top64_a1 | 2 | period_stop | 2 | 0.8750 | 124.0000 | 21.8750 | 1.1250 | 1.2857 | {'.\n': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | short_answer_instruction | 2 | be_continuation | 2 | 1.2812 | -2905.0000 | 21.4375 | 0.2812 | 0.2195 | {' is': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top16_a0.5 | 2 | be_continuation | 2 | 0.0625 | 10.0000 | 21.4375 | -0.0625 | -1.0000 | {' is': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_top64_a0.5 | 2 | period_stop | 4 | 0.3438 | 47.5000 | 20.8438 | 0.2812 | 0.8333 | {'.\n': 4} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_top16_a0.5 | 2 | period_stop | 4 | 0.5000 | 57.5000 | 20.7500 | 0.3750 | 0.8182 | {'.\n': 4} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_top16_a1 | 2 | period_stop | 4 | 0.5000 | 55.5000 | 20.7500 | 0.3750 | 0.6667 | {'.': 2, '.\n': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_top64_a1 | 2 | period_stop | 4 | 0.6562 | 76.0000 | 20.4688 | 0.6562 | 1.3824 | {'.': 2, '.\n': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_top16_a1 | 2 | period_stop | 4 | 0.8438 | 93.5000 | 20.3438 | 0.7812 | 0.9444 | {'.\n': 4} |
| qwen3_explain_l29_readout_regime | drift | natural | repeat_instruction | 2 | comma_repeat | 4 | 3.0312 | 78.5000 | 20.1562 | 0.9688 | -0.4545 | {',': 4} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_all_a0.5 | 2 | period_stop | 2 | 0.1250 | 18.0000 | 19.1250 | 0.1250 | 1.0000 | {'.': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a0.5 | 3 | be_continuation | 2 | 0.1875 | 13.0000 | 17.7500 | 0.0625 | 0.3333 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_all_a1 | 3 | be_continuation | 2 | 0.3750 | -69.0000 | 17.6875 | 0.1250 | 0.3333 | {' are': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | short_answer_instruction | 2 | period_stop | 2 | 1.1250 | -34.0000 | 17.6250 | 1.6250 | 1.4444 | {'.': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a1 | 3 | be_continuation | 2 | 0.1875 | -22.0000 | 17.6250 | 0.1875 | 1.0000 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | no_instruction | 3 | prose | 2 | 0.5039 | -109.0000 | 17.5234 | -2.2461 | -4.4574 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top16_a0.5 | 3 | be_continuation | 2 | 0.4375 | 153.0000 | 17.5000 | 0.3125 | 0.7143 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_top16_a1 | 3 | be_continuation | 2 | 0.6875 | 228.0000 | 17.3750 | 0.4375 | 0.6364 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top64_a0.5 | 3 | be_continuation | 2 | 0.6875 | 229.0000 | 17.3750 | 0.4375 | 0.6364 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_all_a0.5 | 3 | be_continuation | 2 | 0.4375 | 141.0000 | 17.3750 | 0.4375 | 1.0000 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_top16_a0.5 | 3 | be_continuation | 2 | 0.3750 | 186.0000 | 17.3125 | 0.5000 | 1.3333 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_top64_a0.5 | 3 | be_continuation | 2 | 0.8125 | 240.0000 | 17.2500 | 0.5625 | 0.6923 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_top64_a1 | 3 | be_continuation | 2 | 1.2500 | 301.0000 | 17.1875 | 0.6250 | 0.5000 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top16_a1 | 3 | be_continuation | 2 | 0.8750 | 286.0000 | 17.1875 | 0.6250 | 0.7143 | {' are': 2} |
| qwen3_explain_l29_readout_regime | success | natural | because_removed | 2 | be_continuation | 2 | 0.1250 | -2.0000 | 17.1250 | -1.3750 | -11.0000 | {' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | because_removed | 3 | prose | 2 | 0.6758 | -2437.0000 | 17.1016 | -1.8242 | -2.6994 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | short_answer_instruction | 3 | be_continuation | 2 | 0.5625 | -6.0000 | 17.0625 | 0.0625 | 0.1111 | {' is': 2} |
| qwen3_explain_l29_readout_regime | success | natural | no_instruction | 2 | be_continuation | 2 | 0.5000 | -143.0000 | 17.0000 | 4.3750 | 8.7500 | {' is': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top64_a1 | 3 | be_continuation | 2 | 1.3750 | 322.0000 | 16.9375 | 0.8750 | 0.6364 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | short_answer_instruction | 3 | prose | 2 | 0.1992 | -26.0000 | 16.7031 | -1.4258 | -7.1569 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | short_answer_instruction | 3 | be_continuation | 2 | 1.1250 | 343.0000 | 16.6875 | 1.1250 | 1.0000 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_top64_a0.5 | 3 | be_continuation | 2 | 0.6250 | 207.0000 | 16.3750 | 0.7500 | 1.2000 | {' is': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | no_answer_anchor | 3 | because_reason | 4 | 2.4375 | 444.5000 | 16.1875 | 14.0000 | 6.9722 | {'Because': 4} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_top16_a0.5 | 3 | be_continuation | 2 | 0.8750 | 235.0000 | 16.1250 | 1.0000 | 1.1429 | {' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_all_a0.5 | 3 | be_continuation | 2 | 0.9375 | 263.0000 | 15.9375 | 1.1875 | 1.2667 | {' is': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top64_a0.5 | 2 | be_continuation | 2 | 0.2500 | 1.0000 | 15.7500 | 0.0000 | 0.0000 | {' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_top16_a0.5 | 3 | prose | 2 | 0.0039 | 242.0000 | 15.3984 | -0.1211 | -31.0000 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | short_answer_instruction | 1 | echo | 4 | 2.0449 | 27510.5000 | 15.3965 | 0.5137 | 0.0217 | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_all_a0.5 | 3 | prose | 2 | 0.0469 | 1844.0000 | 15.3555 | -0.0781 | -1.6667 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | repeat_instruction | 1 | echo | 2 | 0.4697 | 5125.0000 | 15.3428 | -0.9678 | -2.0603 | {' Glass': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | because_removed | 3 | because_reason | 2 | 0.6250 | 135.0000 | 15.3125 | 14.8750 | 23.8000 | {'Reason': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_top16_a0.5 | 3 | prose | 2 | 0.1445 | 846.0000 | 15.2578 | 0.0195 | 0.1351 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_all_a0.5 | 3 | prose | 2 | 0.2227 | 297.0000 | 15.1797 | 0.0977 | 0.4386 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | repeat_instruction | 3 | prose | 2 | 2.8555 | -1614.0000 | 15.1719 | 0.1055 | 0.0369 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | repeat_instruction | 1 | the_continuation | 2 | 1.9531 | 36274.0000 | 15.1172 | 2.3281 | 1.1920 | {' The': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_top16_a1 | 3 | prose | 2 | 0.2930 | 1848.0000 | 15.1094 | 0.1680 | 0.5733 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_all_a1 | 3 | prose | 2 | 0.6211 | 1149.0000 | 15.0312 | 0.2461 | 0.3962 | {' used': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_top64_a1 | 3 | be_continuation | 2 | 0.8750 | 346.0000 | 14.7500 | 2.3750 | 2.7143 | {' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_top16_a1 | 3 | be_continuation | 2 | 1.3750 | 365.0000 | 14.6250 | 2.5000 | 1.8182 | {' is': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | no_instruction | 3 | the_continuation | 2 | 1.6875 | -95.0000 | 14.5000 | 15.6875 | 9.2963 | {'The': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | short_answer_instruction | 3 | the_continuation | 4 | 1.1875 | -35.5000 | 14.3750 | 15.8125 | 21.0833 | {'The': 4} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_all_a1 | 3 | be_continuation | 2 | 0.8125 | 368.0000 | 14.3125 | 2.8125 | 3.4615 | {' is': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | no_instruction | 3 | be_continuation | 2 | 1.9375 | 649.0000 | 14.2500 | 3.5625 | 1.8387 | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | no_answer_anchor | 1 | the_continuation | 2 | 2.5391 | 40425.0000 | 14.2188 | 3.2266 | 1.2708 | {' The': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | short_answer_instruction | 2 | newline_boundary | 2 | 0.1172 | 4927.0000 | 13.7656 | -1.4453 | -12.3333 | {'orses': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a0.5 | 1 | echo | 4 | 0.1875 | 0.0000 | 13.7500 | 0.1875 | 1.0000 | {' Car': 2, ' Cherry': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | no_instruction | 2 | period_stop | 2 | 1.9375 | 4.0000 | 13.6875 | 5.5625 | 2.8710 | {'.': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | no_answer_anchor | 1 | the_continuation | 2 | 0.1406 | -14631.0000 | 13.6562 | -0.8594 | -6.1111 | {' The': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top64_a0.5 | 1 | echo | 4 | 0.3125 | 3.5000 | 13.6250 | 0.3125 | 1.0000 | {' Car': 2, ' Cherry': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | no_instruction | 3 | be_continuation | 2 | 0.1250 | 271.0000 | 13.6250 | 3.5000 | 28.0000 | {' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | no_instruction | 1 | echo | 2 | 4.6641 | 52820.0000 | 13.5312 | 3.9141 | 0.8392 | {' Cup': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top16_a0.5 | 1 | echo | 4 | 0.4062 | 4.0000 | 13.4688 | 0.4688 | 1.2000 | {' Car': 2, ' Cherry': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top64_a1 | 1 | echo | 4 | 0.5625 | 6.0000 | 13.4375 | 0.5000 | 1.1071 | {' Car': 2, ' Cherry': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a1 | 1 | echo | 4 | 0.3438 | -0.5000 | 13.4062 | 0.5312 | 1.1667 | {' Car': 2, ' Cherry': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | because_removed | 2 | newline_boundary | 2 | 0.4688 | 6678.0000 | 13.2891 | -0.9688 | -2.0667 | {'orses': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_product_top64_a0.5 | 1 | echo | 4 | 0.6250 | 6.0000 | 13.2500 | 0.6875 | 1.2000 | {' Car': 2, ' Cherry': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top16_a1 | 1 | echo | 4 | 0.8750 | 7.0000 | 13.0625 | 0.8750 | 1.0208 | {' Car': 2, ' Cherry': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_product_all_a0.5 | 1 | echo | 4 | 0.7500 | 8.0000 | 13.0000 | 0.9375 | 1.9500 | {' Car': 2, ' Cherry': 2} |

## Closure Candidates

| model | spec | group | type | variant | step | winner | margin | target delta | rank improve | top token |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| qwen3 | qwen3_explain_l29_readout_regime | drift | natural | repeat_instruction | 1 | space_boundary | 6.1250 | 3.5000 | 2.0000 |  Red |
| qwen3 | qwen3_explain_l29_readout_regime | drift | natural | repeat_instruction | 1 | space_boundary | 6.1250 | 3.5000 | 2.0000 |  Red |
| qwen3 | qwen3_explain_l29_readout_regime | drift | natural | repeat_instruction | 1 | the_continuation | 5.8750 | 4.8750 | 6.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | drift | natural | repeat_instruction | 1 | the_continuation | 5.8750 | 4.8750 | 6.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | natural | repeat_instruction | 1 | the_continuation | 5.8750 | 10.1875 | 11.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | natural | repeat_instruction | 1 | the_continuation | 5.8750 | 10.1875 | 11.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top16_a1 | 3 | prose | 5.6250 | 1.2500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a0.5 | 3 | prose | 5.6250 | 1.2500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top16_a1 | 3 | prose | 5.6250 | 1.2500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a0.5 | 3 | prose | 5.6250 | 1.2500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top16_a0.5 | 3 | prose | 5.5000 | 1.0000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top16_a0.5 | 3 | prose | 5.5000 | 1.0000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top64_a0.5 | 3 | prose | 5.1250 | 1.0000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top64_a0.5 | 3 | prose | 5.1250 | 1.0000 | 0.0000 |  red |
| glm4 | glm4_repeat_l30_readout_regime | success | patch | patch_product_all_a1 | 3 | echo | 5.1250 | 0.0625 | 0.0000 |  red |
| glm4 | glm4_repeat_l30_readout_regime | success | patch | patch_product_top16_a0.5 | 3 | echo | 5.0625 | 0.0625 | 0.0000 |  red |
| glm4 | glm4_repeat_l30_readout_regime | success | patch | patch_product_all_a0.5 | 3 | echo | 5.0625 | 0.0625 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_top64_a1 | 3 | prose | 5.0000 | 0.7500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_all_a0.5 | 3 | prose | 5.0000 | 0.7500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_all_a1 | 3 | prose | 5.0000 | 0.7500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_top64_a1 | 3 | prose | 5.0000 | 0.7500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_all_a0.5 | 3 | prose | 5.0000 | 0.7500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_all_a1 | 3 | prose | 5.0000 | 0.7500 | 0.0000 |  red |
| glm4 | glm4_repeat_l30_readout_regime | success | patch | patch_gate_up_pair_top16_a0.5 | 3 | echo | 5.0000 | 0.0625 | 0.0000 |  red |
| glm4 | glm4_repeat_l30_readout_regime | success | patch | patch_product_top64_a0.5 | 3 | echo | 5.0000 | 0.0625 | 0.0000 |  red |
| glm4 | glm4_repeat_l30_readout_regime | success | patch | patch_gate_up_pair_all_a0.5 | 3 | echo | 5.0000 | 0.0625 | 0.0000 |  red |
| glm4 | glm4_repeat_l30_readout_regime | success | patch | patch_gate_up_pair_top64_a0.5 | 3 | echo | 4.9375 | 0.0625 | 0.0000 |  red |
| glm4 | glm4_repeat_l30_readout_regime | success | patch | patch_gate_up_pair_top64_a1 | 3 | echo | 4.9375 | 0.0625 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_top16_a1 | 3 | prose | 4.8750 | 0.7500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a1 | 3 | prose | 4.8750 | 1.2500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_top16_a1 | 3 | prose | 4.8750 | 0.7500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a1 | 3 | prose | 4.8750 | 1.2500 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_top16_a0.5 | 3 | prose | 4.7500 | 0.5000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_top64_a0.5 | 3 | prose | 4.7500 | 0.5000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top64_a1 | 3 | prose | 4.7500 | 1.0000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_top16_a0.5 | 3 | prose | 4.7500 | 0.5000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_top64_a0.5 | 3 | prose | 4.7500 | 0.5000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top64_a1 | 3 | prose | 4.7500 | 1.0000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | natural | because_removed | 3 | prose | 4.5000 | 1.5000 | 0.0000 |  red |
| qwen3 | qwen3_explain_l29_readout_regime | success | natural | because_removed | 3 | prose | 4.5000 | 1.5000 | 0.0000 |  red |
