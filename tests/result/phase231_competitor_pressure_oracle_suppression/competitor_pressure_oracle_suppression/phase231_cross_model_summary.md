# Phase 231 competitor pressure oracle suppression

input_barrier_rows: 321
suppression_rows: 3210

## Gap Distribution

| model | spec | group | winner | rows | mean gap | median gap | p75 gap | p90 gap | max gap |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3 | qwen3_explain_l29_readout_regime | success | echo | 52 | 12.7981 | 12.6250 | 14.0000 | 14.5500 | 15.0000 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | period_stop | 40 | 19.4375 | 20.5000 | 22.5156 | 22.6875 | 22.6875 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | be_continuation | 28 | 17.1071 | 17.3438 | 17.5000 | 17.6875 | 17.7500 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | the_continuation | 28 | 12.4233 | 12.4043 | 12.5273 | 12.5703 | 13.6562 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | because_reason | 24 | 25.9635 | 29.1875 | 29.6562 | 29.8125 | 29.8125 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | be_continuation | 20 | 15.5938 | 15.3438 | 16.3750 | 17.5000 | 21.4375 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | prose | 20 | 15.7832 | 15.3066 | 16.7031 | 17.1438 | 17.5234 |
| qwen3 | qwen3_explain_l29_readout_regime | success | prose | 20 | 7.4500 | 7.5000 | 7.6250 | 7.7500 | 7.7500 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | the_continuation | 20 | 9.0875 | 7.1250 | 12.9375 | 14.6313 | 15.8125 |
| qwen3 | qwen3_explain_l29_readout_regime | success | be_continuation | 12 | 17.5208 | 17.0625 | 21.4375 | 21.8875 | 21.9375 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | newline_boundary | 10 | 12.2602 | 11.7578 | 13.2891 | 13.7656 | 13.7656 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | echo | 8 | 14.9167 | 15.0210 | 15.5305 | 16.0938 | 16.0938 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | echo | 8 | 7.1875 | 7.2500 | 7.6562 | 7.7500 | 7.7500 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | comma_repeat | 4 | 20.1562 | 20.1562 | 23.2500 | 23.2500 | 23.2500 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | the_continuation | 4 | 14.6680 | 14.6680 | 15.1172 | 15.1172 | 15.1172 |
| qwen3 | qwen3_explain_l29_readout_regime | success | comma_repeat | 4 | 9.0312 | 9.0312 | 10.3125 | 10.3125 | 10.3125 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | answer_boundary | 4 | 5.9688 | 5.9688 | 6.1875 | 6.1875 | 6.1875 |
| glm4 | glm4_repeat_l30_readout_regime | drift | the_continuation | 4 | 1.7656 | 1.7656 | 2.0312 | 2.0312 | 2.0312 |
| glm4 | glm4_repeat_l30_readout_regime | drift | because_reason | 4 | 0.7656 | 0.8750 | 0.9375 | 0.9375 | 0.9375 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | space_boundary | 2 | 12.3672 | 12.3672 | 12.3672 | 12.3672 | 12.3672 |
| glm4 | glm4_repeat_l30_readout_regime | drift | comma_repeat | 2 | 4.5625 | 4.5625 | 4.5625 | 4.5625 | 4.5625 |
| glm4 | glm4_repeat_l30_readout_regime | success | the_continuation | 2 | 0.2188 | 0.2188 | 0.2969 | 0.3438 | 0.3750 |
| glm4 | glm4_repeat_l30_readout_regime | drift | prose | 1 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 |

## Budget Summary

| model | budget | rows | closed | closure rate | mean post margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen3 | 1.0 | 188 | 0 | 0.0000 | -13.9215 |
| qwen3 | 2.0 | 188 | 0 | 0.0000 | -12.9215 |
| qwen3 | 4.0 | 188 | 0 | 0.0000 | -10.9215 |
| qwen3 | 8.0 | 188 | 48 | 0.2553 | -6.9215 |
| qwen3 | 12.0 | 188 | 66 | 0.3511 | -2.9215 |
| qwen3 | 16.0 | 188 | 122 | 0.6489 | 1.0785 |
| qwen3 | 20.0 | 188 | 144 | 0.7660 | 5.0785 |
| qwen3 | 24.0 | 188 | 170 | 0.9043 | 9.0785 |
| qwen3 | 28.0 | 188 | 170 | 0.9043 | 13.0785 |
| qwen3 | 32.0 | 188 | 188 | 1.0000 | 17.0785 |
| glm4 | 1.0 | 13 | 7 | 0.5385 | -0.5817 |
| glm4 | 2.0 | 13 | 9 | 0.6923 | 0.4183 |
| glm4 | 4.0 | 13 | 11 | 0.8462 | 2.4183 |
| glm4 | 8.0 | 13 | 13 | 1.0000 | 6.4183 |
| glm4 | 12.0 | 13 | 13 | 1.0000 | 10.4183 |
| glm4 | 16.0 | 13 | 13 | 1.0000 | 14.4183 |
| glm4 | 20.0 | 13 | 13 | 1.0000 | 18.4183 |
| glm4 | 24.0 | 13 | 13 | 1.0000 | 22.4183 |
| glm4 | 28.0 | 13 | 13 | 1.0000 | 26.4183 |
| glm4 | 32.0 | 13 | 13 | 1.0000 | 30.4183 |
| deepseek7b | 1.0 | 120 | 0 | 0.0000 | -13.8311 |
| deepseek7b | 2.0 | 120 | 0 | 0.0000 | -12.8311 |
| deepseek7b | 4.0 | 120 | 0 | 0.0000 | -10.8311 |
| deepseek7b | 8.0 | 120 | 0 | 0.0000 | -6.8311 |
| deepseek7b | 12.0 | 120 | 10 | 0.0833 | -2.8311 |
| deepseek7b | 16.0 | 120 | 78 | 0.6500 | 1.1689 |
| deepseek7b | 20.0 | 120 | 118 | 0.9833 | 5.1689 |
| deepseek7b | 24.0 | 120 | 120 | 1.0000 | 9.1689 |
| deepseek7b | 28.0 | 120 | 120 | 1.0000 | 13.1689 |
| deepseek7b | 32.0 | 120 | 120 | 1.0000 | 17.1689 |

## Winner Budget Summary

| model | spec | group | winner | budget | rows | closed | closure rate | mean gap |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3 | qwen3_explain_l29_readout_regime | drift | answer_boundary | 8.0 | 4 | 4 | 1.0000 | 5.9688 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | answer_boundary | 16.0 | 4 | 4 | 1.0000 | 5.9688 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | answer_boundary | 24.0 | 4 | 4 | 1.0000 | 5.9688 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | answer_boundary | 32.0 | 4 | 4 | 1.0000 | 5.9688 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | because_reason | 8.0 | 24 | 0 | 0.0000 | 25.9635 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | because_reason | 16.0 | 24 | 4 | 0.1667 | 25.9635 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | because_reason | 24.0 | 24 | 6 | 0.2500 | 25.9635 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | because_reason | 32.0 | 24 | 24 | 1.0000 | 25.9635 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | comma_repeat | 8.0 | 4 | 0 | 0.0000 | 20.1562 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | comma_repeat | 16.0 | 4 | 0 | 0.0000 | 20.1562 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | comma_repeat | 24.0 | 4 | 4 | 1.0000 | 20.1562 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | comma_repeat | 32.0 | 4 | 4 | 1.0000 | 20.1562 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | echo | 8.0 | 8 | 8 | 1.0000 | 7.1875 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | echo | 16.0 | 8 | 8 | 1.0000 | 7.1875 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | echo | 24.0 | 8 | 8 | 1.0000 | 7.1875 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | echo | 32.0 | 8 | 8 | 1.0000 | 7.1875 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | period_stop | 8.0 | 40 | 0 | 0.0000 | 19.4375 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | period_stop | 16.0 | 40 | 6 | 0.1500 | 19.4375 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | period_stop | 24.0 | 40 | 40 | 1.0000 | 19.4375 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | period_stop | 32.0 | 40 | 40 | 1.0000 | 19.4375 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | the_continuation | 8.0 | 20 | 14 | 0.7000 | 9.0875 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | the_continuation | 16.0 | 20 | 20 | 1.0000 | 9.0875 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | the_continuation | 24.0 | 20 | 20 | 1.0000 | 9.0875 |
| qwen3 | qwen3_explain_l29_readout_regime | drift | the_continuation | 32.0 | 20 | 20 | 1.0000 | 9.0875 |
| qwen3 | qwen3_explain_l29_readout_regime | success | be_continuation | 8.0 | 12 | 0 | 0.0000 | 17.5208 |
| qwen3 | qwen3_explain_l29_readout_regime | success | be_continuation | 16.0 | 12 | 4 | 0.3333 | 17.5208 |
| qwen3 | qwen3_explain_l29_readout_regime | success | be_continuation | 24.0 | 12 | 12 | 1.0000 | 17.5208 |
| qwen3 | qwen3_explain_l29_readout_regime | success | be_continuation | 32.0 | 12 | 12 | 1.0000 | 17.5208 |
| qwen3 | qwen3_explain_l29_readout_regime | success | comma_repeat | 8.0 | 4 | 2 | 0.5000 | 9.0312 |
| qwen3 | qwen3_explain_l29_readout_regime | success | comma_repeat | 16.0 | 4 | 4 | 1.0000 | 9.0312 |
| qwen3 | qwen3_explain_l29_readout_regime | success | comma_repeat | 24.0 | 4 | 4 | 1.0000 | 9.0312 |
| qwen3 | qwen3_explain_l29_readout_regime | success | comma_repeat | 32.0 | 4 | 4 | 1.0000 | 9.0312 |
| qwen3 | qwen3_explain_l29_readout_regime | success | echo | 8.0 | 52 | 0 | 0.0000 | 12.7981 |
| qwen3 | qwen3_explain_l29_readout_regime | success | echo | 16.0 | 52 | 52 | 1.0000 | 12.7981 |
| qwen3 | qwen3_explain_l29_readout_regime | success | echo | 24.0 | 52 | 52 | 1.0000 | 12.7981 |
| qwen3 | qwen3_explain_l29_readout_regime | success | echo | 32.0 | 52 | 52 | 1.0000 | 12.7981 |
| qwen3 | qwen3_explain_l29_readout_regime | success | prose | 8.0 | 20 | 20 | 1.0000 | 7.4500 |
| qwen3 | qwen3_explain_l29_readout_regime | success | prose | 16.0 | 20 | 20 | 1.0000 | 7.4500 |
| qwen3 | qwen3_explain_l29_readout_regime | success | prose | 24.0 | 20 | 20 | 1.0000 | 7.4500 |
| qwen3 | qwen3_explain_l29_readout_regime | success | prose | 32.0 | 20 | 20 | 1.0000 | 7.4500 |
| glm4 | glm4_repeat_l30_readout_regime | drift | because_reason | 8.0 | 4 | 4 | 1.0000 | 0.7656 |
| glm4 | glm4_repeat_l30_readout_regime | drift | because_reason | 16.0 | 4 | 4 | 1.0000 | 0.7656 |
| glm4 | glm4_repeat_l30_readout_regime | drift | because_reason | 24.0 | 4 | 4 | 1.0000 | 0.7656 |
| glm4 | glm4_repeat_l30_readout_regime | drift | because_reason | 32.0 | 4 | 4 | 1.0000 | 0.7656 |
| glm4 | glm4_repeat_l30_readout_regime | drift | comma_repeat | 8.0 | 2 | 2 | 1.0000 | 4.5625 |
| glm4 | glm4_repeat_l30_readout_regime | drift | comma_repeat | 16.0 | 2 | 2 | 1.0000 | 4.5625 |
| glm4 | glm4_repeat_l30_readout_regime | drift | comma_repeat | 24.0 | 2 | 2 | 1.0000 | 4.5625 |
| glm4 | glm4_repeat_l30_readout_regime | drift | comma_repeat | 32.0 | 2 | 2 | 1.0000 | 4.5625 |
| glm4 | glm4_repeat_l30_readout_regime | drift | prose | 8.0 | 1 | 1 | 1.0000 | 0.8750 |
| glm4 | glm4_repeat_l30_readout_regime | drift | prose | 16.0 | 1 | 1 | 1.0000 | 0.8750 |
| glm4 | glm4_repeat_l30_readout_regime | drift | prose | 24.0 | 1 | 1 | 1.0000 | 0.8750 |
| glm4 | glm4_repeat_l30_readout_regime | drift | prose | 32.0 | 1 | 1 | 1.0000 | 0.8750 |
| glm4 | glm4_repeat_l30_readout_regime | drift | the_continuation | 8.0 | 4 | 4 | 1.0000 | 1.7656 |
| glm4 | glm4_repeat_l30_readout_regime | drift | the_continuation | 16.0 | 4 | 4 | 1.0000 | 1.7656 |
| glm4 | glm4_repeat_l30_readout_regime | drift | the_continuation | 24.0 | 4 | 4 | 1.0000 | 1.7656 |
| glm4 | glm4_repeat_l30_readout_regime | drift | the_continuation | 32.0 | 4 | 4 | 1.0000 | 1.7656 |
| glm4 | glm4_repeat_l30_readout_regime | success | the_continuation | 8.0 | 2 | 2 | 1.0000 | 0.2188 |
| glm4 | glm4_repeat_l30_readout_regime | success | the_continuation | 16.0 | 2 | 2 | 1.0000 | 0.2188 |
| glm4 | glm4_repeat_l30_readout_regime | success | the_continuation | 24.0 | 2 | 2 | 1.0000 | 0.2188 |
| glm4 | glm4_repeat_l30_readout_regime | success | the_continuation | 32.0 | 2 | 2 | 1.0000 | 0.2188 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | be_continuation | 8.0 | 28 | 0 | 0.0000 | 17.1071 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | be_continuation | 16.0 | 28 | 2 | 0.0714 | 17.1071 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | be_continuation | 24.0 | 28 | 28 | 1.0000 | 17.1071 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | be_continuation | 32.0 | 28 | 28 | 1.0000 | 17.1071 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | newline_boundary | 8.0 | 10 | 0 | 0.0000 | 12.2602 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | newline_boundary | 16.0 | 10 | 10 | 1.0000 | 12.2602 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | newline_boundary | 24.0 | 10 | 10 | 1.0000 | 12.2602 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | newline_boundary | 32.0 | 10 | 10 | 1.0000 | 12.2602 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | the_continuation | 8.0 | 28 | 0 | 0.0000 | 12.4233 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | the_continuation | 16.0 | 28 | 28 | 1.0000 | 12.4233 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | the_continuation | 24.0 | 28 | 28 | 1.0000 | 12.4233 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | drift | the_continuation | 32.0 | 28 | 28 | 1.0000 | 12.4233 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | be_continuation | 8.0 | 20 | 0 | 0.0000 | 15.5938 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | be_continuation | 16.0 | 20 | 12 | 0.6000 | 15.5938 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | be_continuation | 24.0 | 20 | 20 | 1.0000 | 15.5938 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | be_continuation | 32.0 | 20 | 20 | 1.0000 | 15.5938 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | echo | 8.0 | 8 | 0 | 0.0000 | 14.9167 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | echo | 16.0 | 8 | 6 | 0.7500 | 14.9167 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | echo | 24.0 | 8 | 8 | 1.0000 | 14.9167 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | echo | 32.0 | 8 | 8 | 1.0000 | 14.9167 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | prose | 8.0 | 20 | 0 | 0.0000 | 15.7832 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | prose | 16.0 | 20 | 14 | 0.7000 | 15.7832 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | prose | 24.0 | 20 | 20 | 1.0000 | 15.7832 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | prose | 32.0 | 20 | 20 | 1.0000 | 15.7832 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | space_boundary | 8.0 | 2 | 0 | 0.0000 | 12.3672 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | space_boundary | 16.0 | 2 | 2 | 1.0000 | 12.3672 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | space_boundary | 24.0 | 2 | 2 | 1.0000 | 12.3672 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | space_boundary | 32.0 | 2 | 2 | 1.0000 | 12.3672 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | the_continuation | 8.0 | 4 | 0 | 0.0000 | 14.6680 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | the_continuation | 16.0 | 4 | 4 | 1.0000 | 14.6680 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | the_continuation | 24.0 | 4 | 4 | 1.0000 | 14.6680 |
| deepseek7b | deepseek7b_explain_l24_readout_regime | success | the_continuation | 32.0 | 4 | 4 | 1.0000 | 14.6680 |
