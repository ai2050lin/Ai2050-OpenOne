# Phase 232 competitor source localization

input_rows: 1188
pressure_source_rows: 4536
switch_source_rows: 51
coupling_rows: 3024

## Priority Rows

| kind | score | model | spec | group | type | variant | step | regime | rows | winner/switch rate | target delta | regime delta | comp-target | top tokens |
| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| pressure_source | 30.0454 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 1 | for_continuation | 4 | 1.0000 | -8.4224 | 9.0820 | 17.5044 | {' For': 4} |
| pressure_source | 28.5851 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 1 | for_continuation | 4 | 1.0000 | -7.2902 | 8.8633 | 16.1535 | {' For': 4} |
| pressure_source | 25.8203 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 3 | because_reason | 4 | 1.0000 | -4.9531 | 8.5781 | 13.5312 | {' because': 4} |
| pressure_source | 19.5156 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 2 | newline_boundary | 4 | 1.0000 | -6.0078 | 3.6719 | 9.6797 | {'\n': 4} |
| pressure_source | 18.7656 | qwen3 | qwen3_explain_l29_readout_regime | drift | natural | no_answer_anchor | 1 | because_reason | 4 | 0.0000 | -12.0625 | 4.4688 | 16.5312 | {' The': 2, ' Then': 2} |
| pressure_source | 18.5430 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | explain_instruction | 3 | because_reason | 4 | 1.0000 | 0.8125 | 7.5703 | 6.7578 | {' because': 4} |
| pressure_source | 18.4531 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 2 | newline_boundary | 4 | 1.0000 | -6.3516 | 2.7344 | 9.0859 | {'\n': 4} |
| pressure_source | 17.1172 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 2 | because_reason | 4 | 0.0000 | -6.0078 | 7.4062 | 13.4141 | {'\n': 4} |
| switch_source | 16.2121 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 1 | for_continuation | 4 | 4.0000 | -7.2902 | 0.0000 | 0.0000 | {' For': 4} |
| switch_source | 15.7845 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 1 | for_continuation | 3 | 3.0000 | -8.5553 | 0.0000 | 0.0000 | {' For': 3} |
| pressure_source | 15.1797 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_instruction | 2 | newline_boundary | 4 | 1.0000 | -2.8438 | 2.8906 | 5.7344 | {'\n': 4} |
| pressure_source | 14.7651 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 1 | newline_boundary | 4 | 0.0000 | -8.4224 | 4.2285 | 12.6509 | {' For': 4} |
| pressure_source | 14.4531 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | explain_instruction | 2 | newline_boundary | 4 | 1.0000 | -1.4844 | 3.3125 | 4.7969 | {'\n': 4} |
| switch_source | 14.3359 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 1 | for_continuation | 1 | 1.0000 | -8.0234 | 0.0000 | 0.0000 | {' For': 1} |
| pressure_source | 13.8203 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 2 | newline_boundary | 4 | 0.7500 | -6.2266 | 1.0625 | 7.2891 | {'\n': 3, ' or': 1} |
| pressure_source | 13.4924 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 1 | newline_boundary | 4 | 0.0000 | -7.2902 | 4.1348 | 11.4250 | {' For': 4} |
| pressure_source | 13.3091 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 1 | the_continuation | 4 | 0.0000 | -8.4224 | 3.2578 | 11.6802 | {' For': 4} |
| switch_source | 13.0781 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 3 | because_reason | 4 | 4.0000 | -4.9531 | 0.0000 | 0.0000 | {' because': 4} |
| pressure_source | 13.0703 | qwen3 | qwen3_explain_l29_readout_regime | drift | natural | no_answer_anchor | 1 | period_stop | 4 | 0.0000 | -12.0625 | 0.6719 | 12.7344 | {' The': 2, ' Then': 2} |
| pressure_source | 13.0508 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | short_answer_instruction | 3 | because_reason | 4 | 1.0000 | 1.5000 | 4.3672 | 2.8672 | {' brown': 2, ' gray': 1, ' grey': 1} |
| pressure_source | 13.0220 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 1 | because_reason | 4 | 0.0000 | -8.4224 | 3.0664 | 11.4888 | {' For': 4} |
| pressure_source | 11.9777 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 1 | the_continuation | 4 | 0.0000 | -7.2902 | 3.1250 | 10.4152 | {' For': 4} |
| switch_source | 11.8516 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 2 | newline_boundary | 4 | 4.0000 | -6.0078 | 0.0000 | 0.0000 | {'\n': 4} |
| switch_source | 11.2578 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 2 | newline_boundary | 4 | 4.0000 | -6.3516 | 0.0000 | 0.0000 | {'\n': 4} |
| pressure_source | 11.2500 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | natural | no_instruction | 1 | echo | 2 | 1.0000 | -0.0625 | 2.1250 | 2.1875 | {' H': 2} |
| pressure_source | 11.1047 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 1 | because_reason | 4 | 0.0000 | -7.2902 | 2.5430 | 9.8332 | {' For': 4} |
| pressure_source | 11.0117 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_instruction | 3 | because_reason | 4 | 0.5000 | 0.0781 | 4.7266 | 4.6484 | {' but': 4} |
| pressure_source | 10.9375 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | no_answer_anchor | 1 | because_reason | 4 | 0.0000 | -5.0312 | 3.9375 | 8.9688 | {' Then': 4} |
| pressure_source | 10.8477 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 3 | because_reason | 4 | 0.2500 | -3.8906 | 3.3047 | 7.1953 | {' green': 1, ' of': 1, ' white': 1, ' with': 1} |
| pressure_source | 10.7422 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | explain_instruction | 2 | because_reason | 4 | 0.0000 | -1.4844 | 6.1719 | 7.6562 | {'\n': 4} |
| pressure_source | 10.5820 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | natural | short_answer_instruction | 1 | echo | 2 | 1.0000 | -0.0508 | 1.6875 | 1.7383 | {' H': 2} |
| pressure_source | 10.5625 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 3 | because_reason | 4 | 0.2500 | -3.9219 | 3.0938 | 7.0156 | {' black': 1, ' green': 1, ' pink': 1, ' white': 1} |
| pressure_source | 10.3359 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 2 | the_continuation | 4 | 0.0000 | -6.3516 | 2.6562 | 9.0078 | {'\n': 4} |
| pressure_source | 9.3711 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 1 | the_continuation | 4 | 0.7500 | -0.4062 | 1.4766 | 1.8828 | {' Red': 3, ' The': 1} |
| pressure_source | 8.8828 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 2 | for_continuation | 4 | 0.0000 | -6.3516 | 1.6875 | 8.0391 | {'\n': 4} |
| pressure_source | 8.8398 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 2 | because_reason | 4 | 0.0000 | -6.2266 | 1.7422 | 7.9688 | {'\n': 3, ' or': 1} |
| pressure_source | 8.7446 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 1 | comma_repeat | 4 | 0.0000 | -8.4224 | 0.2148 | 8.6372 | {' For': 4} |
| pressure_source | 8.5664 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 2 | because_reason | 4 | 0.0000 | -6.3516 | 1.4766 | 7.8281 | {'\n': 4} |
| pressure_source | 8.5625 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | no_answer_anchor | 3 | period_stop | 4 | 0.0000 | -5.3750 | 2.1250 | 7.5000 | {' a': 4} |
| switch_source | 8.5000 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 2 | newline_boundary | 3 | 3.0000 | -6.2917 | 0.0000 | 0.0000 | {'\n': 3} |
| pressure_source | 8.4062 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 2 | the_continuation | 4 | 0.0000 | -6.2266 | 1.4531 | 7.6797 | {'\n': 3, ' or': 1} |
| pressure_source | 8.0898 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 2 | for_continuation | 4 | 0.0000 | -6.2266 | 1.2422 | 7.4688 | {'\n': 3, ' or': 1} |
| switch_source | 8.0469 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_instruction | 2 | newline_boundary | 4 | 4.0000 | -2.8438 | 0.0000 | 0.0000 | {'\n': 4} |
| pressure_source | 8.0312 | qwen3 | qwen3_explain_l29_readout_regime | drift | natural | because_removed | 2 | period_stop | 4 | 0.0000 | -1.2812 | 2.5000 | 3.7812 | {'.': 2, '.\n': 2} |
| pressure_source | 8.0234 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 3 | the_continuation | 4 | 0.0000 | -4.9531 | 2.0469 | 7.0000 | {' because': 4} |
| pressure_source | 7.8750 | qwen3 | qwen3_explain_l29_readout_regime | drift | natural | no_instruction | 1 | because_reason | 4 | 0.0000 | -1.5000 | 4.2500 | 5.7500 | {' Apple': 2, ' The': 2} |
| pressure_source | 7.7852 | deepseek7b | deepseek7b_explain_l24_readout_regime | success | natural | because_removed | 1 | echo | 4 | 0.5000 | -1.3945 | 0.5938 | 1.9883 | {' Cup': 2, ' Glass': 2} |
| pressure_source | 7.7656 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 2 | for_continuation | 4 | 0.0000 | -6.0078 | 1.1719 | 7.1797 | {'\n': 4} |
| pressure_source | 7.6602 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | comma_removed | 3 | because_reason | 4 | 0.0000 | -2.5156 | 3.4297 | 5.9453 | {' green': 2, ' white': 2} |
| switch_source | 7.6406 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | explain_instruction | 3 | because_reason | 4 | 4.0000 | 0.8125 | 0.0000 | 0.0000 | {' because': 4} |
| pressure_source | 7.5773 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 1 | comma_repeat | 4 | 0.0000 | -7.2902 | 0.1914 | 7.4816 | {' For': 4} |
| switch_source | 7.5625 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 3 | because_reason | 1 | 1.0000 | -4.3750 | 0.0000 | 0.0000 | {' white': 1} |
| pressure_source | 7.5352 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 3 | for_continuation | 4 | 0.0000 | -3.8906 | 2.4297 | 6.3203 | {' green': 1, ' of': 1, ' white': 1, ' with': 1} |
| pressure_source | 7.3477 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 3 | the_continuation | 4 | 0.0000 | -3.8906 | 2.3047 | 6.1953 | {' green': 1, ' of': 1, ' white': 1, ' with': 1} |
| switch_source | 7.1875 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 3 | because_reason | 1 | 1.0000 | -4.3750 | 0.0000 | 0.0000 | {' white': 1} |
| pressure_source | 7.1211 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 2 | the_continuation | 4 | 0.0000 | -6.0078 | 0.7422 | 6.7500 | {'\n': 4} |
| pressure_source | 7.1094 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | no_answer_anchor | 3 | because_reason | 4 | 0.0000 | -5.3750 | 1.1562 | 6.5312 | {' a': 4} |
| switch_source | 7.1094 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | explain_instruction | 2 | newline_boundary | 4 | 4.0000 | -1.4844 | 0.0000 | 0.0000 | {'\n': 4} |
| pressure_source | 7.0859 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 3 | for_continuation | 4 | 0.0000 | -4.9531 | 1.4219 | 6.3750 | {' because': 4} |
| pressure_source | 6.9805 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_instruction | 2 | the_continuation | 4 | 0.0000 | -2.8438 | 2.7578 | 5.6016 | {'\n': 4} |
| pressure_source | 6.5586 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 3 | the_continuation | 4 | 0.0000 | -3.9219 | 1.7578 | 5.6797 | {' black': 1, ' green': 1, ' pink': 1, ' white': 1} |
| pressure_source | 6.5234 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | comma_removed | 2 | newline_boundary | 4 | 0.5000 | -1.9375 | 0.3906 | 2.3281 | {'\n': 2, '.\n': 2} |
| pressure_source | 6.4531 | glm4 | glm4_repeat_l30_readout_regime | success | natural | comma_removed | 2 | newline_boundary | 4 | 0.5000 | -1.5625 | 0.5938 | 2.1562 | {'\n': 3, ',': 1} |
| pressure_source | 6.4531 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_instruction | 2 | because_reason | 4 | 0.0000 | -2.8438 | 2.4062 | 5.2500 | {'\n': 4} |
| pressure_source | 6.3633 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | short_answer_instruction | 2 | because_reason | 4 | 0.0000 | -2.1094 | 2.8359 | 4.9453 | {' or': 2, ',': 2} |
| pressure_source | 6.2656 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | no_instruction | 3 | period_stop | 4 | 0.0000 | -5.9375 | 0.2188 | 6.1562 | {' a': 4} |
| patch_coupling | 6.1875 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a1 | 3 | echo | 2 | 1.0000 | 0.1875 | 1.1875 | 1.0000 |  |
| patch_coupling | 6.0625 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_all_a1 | 3 | prose | 2 | 1.0000 | 0.3750 | 1.0625 | 0.6875 |  |
| pressure_source | 6.0527 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 3 | comma_repeat | 4 | 0.0000 | -3.8906 | 1.4414 | 5.3320 | {' green': 1, ' of': 1, ' white': 1, ' with': 1} |
| pressure_source | 6.0371 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 3 | comma_repeat | 4 | 0.0000 | -4.9531 | 0.7227 | 5.6758 | {' because': 4} |
| pressure_source | 6.0000 | glm4 | glm4_repeat_l30_readout_regime | success | natural | comma_removed | 3 | because_reason | 4 | 0.0000 | -1.0781 | 3.2812 | 4.3594 | {' red': 4} |
| patch_coupling | 6.0000 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_all_a1 | 3 | echo | 2 | 1.0000 | 0.3750 | 1.0000 | 0.6250 |  |
| pressure_source | 5.9922 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 3 | for_continuation | 4 | 0.0000 | -0.7188 | 3.5156 | 4.2344 | {' green': 2, ' white': 2} |
| pressure_source | 5.9844 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top64_a1 | 2 | be_continuation | 2 | 0.0000 | -1.6484 | 2.8906 | 4.5391 | {'orses': 2} |
| pressure_source | 5.9727 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_instruction | 2 | for_continuation | 4 | 0.0000 | -2.8438 | 2.0859 | 4.9297 | {'\n': 4} |
| pressure_source | 5.9434 | deepseek7b | deepseek7b_explain_l24_readout_regime | success | patch | patch_product_top64_a1 | 1 | echo | 4 | 0.5000 | -0.4434 | 0.0000 | 0.4434 | {' Cup': 2, ' Glass': 2} |
| pressure_source | 5.9395 | deepseek7b | deepseek7b_explain_l24_readout_regime | success | natural | short_answer_instruction | 1 | echo | 4 | 0.5000 | 2.0449 | 1.6562 | -0.3887 | {' Cup': 2, ' Glass': 2} |
| pressure_source | 5.9258 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 3 | for_continuation | 4 | 0.0000 | -3.9219 | 1.3359 | 5.2578 | {' black': 1, ' green': 1, ' pink': 1, ' white': 1} |
| switch_source | 5.9141 | deepseek7b | deepseek7b_explain_l24_readout_regime | success | natural | no_instruction | 1 | echo | 2 | 2.0000 | 4.6641 | 0.0000 | 0.0000 | {' Cup': 2} |
| pressure_source | 5.9023 | deepseek7b | deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_top64_a1 | 1 | echo | 4 | 0.5000 | -0.3555 | 0.0312 | 0.3867 | {' Cup': 2, ' Glass': 2} |
| pressure_source | 5.8438 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | repeat_instruction | 3 | period_stop | 4 | 0.0000 | -4.0625 | 1.1875 | 5.2500 | {' a': 4} |
| pressure_source | 5.7500 | qwen3 | qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_all_a0.5 | 1 | echo | 4 | 0.5000 | 0.1250 | 0.2500 | 0.1250 | {' Apple': 2, ' Red': 2} |
| pressure_source | 5.7500 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | no_instruction | 3 | because_reason | 4 | 0.0000 | -5.9375 | -0.1250 | 5.8125 | {' a': 4} |
| pressure_source | 5.7070 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 3 | for_continuation | 4 | 0.0000 | -1.0781 | 3.0859 | 4.1641 | {' red': 4} |
| pressure_source | 5.7031 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | no_answer_anchor | 2 | because_reason | 4 | 0.0000 | -0.4062 | 3.5312 | 3.9375 | {' is': 4} |
| pressure_source | 5.6250 | qwen3 | qwen3_explain_l29_readout_regime | drift | patch | patch_gate_up_pair_all_a1 | 1 | echo | 4 | 0.5000 | 0.4375 | 0.3750 | -0.0625 | {' Apple': 2, ' Red': 2} |
| patch_coupling | 5.6250 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a0.5 | 3 | echo | 2 | 1.0000 | 0.1875 | 0.6250 | 0.4375 |  |
| pressure_source | 5.6172 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | no_answer_anchor | 1 | period_stop | 4 | 0.0000 | -5.0312 | 0.3906 | 5.4219 | {' Then': 4} |
| pressure_source | 5.5859 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | comma_removed | 3 | the_continuation | 4 | 0.0000 | -2.5156 | 2.0469 | 4.5625 | {' green': 2, ' white': 2} |
| pressure_source | 5.5742 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | comma_removed | 3 | for_continuation | 4 | 0.0000 | -2.5156 | 2.0391 | 4.5547 | {' green': 2, ' white': 2} |
| patch_coupling | 5.5625 | qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a1 | 1 | because_reason | 4 | 1.0000 | 0.3438 | 0.5625 | 0.2188 |  |
| patch_coupling | 5.5625 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_all_a0.5 | 3 | prose | 2 | 1.0000 | 0.4375 | 0.5625 | 0.1250 |  |
| pressure_source | 5.5312 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a1 | 2 | be_continuation | 2 | 0.0000 | -1.0078 | 3.0156 | 4.0234 | {'orses': 2} |
| patch_coupling | 5.5000 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_all_a1 | 3 | newline_boundary | 2 | 1.0000 | 0.3750 | 0.5000 | 0.1250 |  |
| pressure_source | 5.4395 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 3 | comma_repeat | 4 | 0.0000 | -3.9219 | 1.0117 | 4.9336 | {' black': 1, ' green': 1, ' pink': 1, ' white': 1} |
| pressure_source | 5.4375 | qwen3 | qwen3_explain_l29_readout_regime | drift | natural | short_answer_instruction | 1 | because_reason | 4 | 0.0000 | -3.0000 | 1.6250 | 4.6250 | {' Apple': 2, ' The': 2} |
| patch_coupling | 5.4375 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a1 | 3 | prose | 2 | 1.0000 | 0.1875 | 0.4375 | 0.2500 |  |
| pressure_source | 5.4219 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_top64_a1 | 2 | be_continuation | 2 | 0.0000 | -1.5078 | 2.6094 | 4.1172 | {'orses': 2} |
| pressure_source | 5.3828 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 3 | the_continuation | 4 | 0.0000 | -0.7188 | 3.1094 | 3.8281 | {' green': 2, ' white': 2} |
| patch_coupling | 5.3750 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a0.5 | 3 | prose | 2 | 1.0000 | 0.1875 | 0.3750 | 0.1875 |  |
| patch_coupling | 5.3750 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a1 | 3 | newline_boundary | 2 | 1.0000 | 0.1875 | 0.3750 | 0.1875 |  |
| pressure_source | 5.3594 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top16_a1 | 2 | be_continuation | 2 | 0.0000 | -1.2109 | 2.7656 | 3.9766 | {'orses': 2} |
| pressure_source | 5.3359 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | no_instruction | 3 | for_continuation | 4 | 0.0000 | 0.0781 | 3.6094 | 3.5312 | {' but': 4} |
| pressure_source | 5.3164 | glm4 | glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 3 | newline_boundary | 4 | 0.0000 | -4.9531 | 0.2422 | 5.1953 | {' because': 4} |
| patch_coupling | 5.3125 | qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a0.5 | 1 | because_reason | 4 | 1.0000 | 0.1875 | 0.3125 | 0.1250 |  |
| patch_coupling | 5.3125 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a0.5 | 3 | the_continuation | 2 | 1.0000 | 0.1875 | 0.3125 | 0.1250 |  |
| patch_coupling | 5.2500 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a0.5 | 3 | newline_boundary | 2 | 1.0000 | 0.1875 | 0.2500 | 0.0625 |  |
| patch_coupling | 5.2500 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a1 | 3 | the_continuation | 2 | 1.0000 | 0.1875 | 0.2500 | 0.0625 |  |
| pressure_source | 5.2344 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | repeat_instruction | 3 | because_reason | 4 | 0.0000 | -4.0625 | 0.7812 | 4.8438 | {' a': 4} |
| pressure_source | 5.2266 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | natural | short_answer_instruction | 2 | newline_boundary | 2 | 0.0000 | 0.1172 | 1.5625 | 1.4453 | {'orses': 2} |
| pressure_source | 5.0938 | qwen3 | qwen3_explain_l29_readout_regime | success | natural | repeat_instruction | 3 | echo | 4 | 0.0000 | -4.0625 | 0.6875 | 4.7500 | {' a': 4} |
| pressure_source | 5.0547 | glm4 | glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 2 | comma_repeat | 4 | 0.0000 | -6.2266 | -1.2812 | 4.9453 | {'\n': 3, ' or': 1} |
| patch_coupling | 5.0156 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_all_a1 | 2 | be_continuation | 2 | 0.0000 | -1.0078 | 3.0156 | 4.0234 |  |
| pressure_source | 4.9844 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_all_a1 | 2 | be_continuation | 2 | 0.0000 | -0.8828 | 2.7344 | 3.6172 | {'orses': 2} |
| patch_coupling | 4.9375 | qwen3 | qwen3_explain_l29_readout_regime | success | patch | patch_product_top64_a1 | 3 | because_reason | 4 | 1.0000 | 0.4375 | 0.9375 | 0.5000 |  |
| patch_coupling | 4.8906 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top64_a1 | 2 | be_continuation | 2 | 0.0000 | -1.6484 | 2.8906 | 4.5391 |  |
| pressure_source | 4.8750 | glm4 | glm4_repeat_l30_readout_regime | success | natural | no_instruction | 2 | comma_repeat | 4 | 0.0000 | -6.3516 | -0.9844 | 5.3672 | {'\n': 4} |
| pressure_source | 4.8750 | deepseek7b | deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_top16_a1 | 2 | be_continuation | 2 | 0.0000 | -1.2891 | 2.3906 | 3.6797 | {'orses': 2} |
| pressure_source | 4.8086 | deepseek7b | deepseek7b_explain_l24_readout_regime | success | natural | no_instruction | 1 | echo | 4 | 0.5000 | 1.2539 | 0.3750 | -0.8789 | {' Cup': 2, ' Glass': 2} |
| pressure_source | 4.7812 | glm4 | glm4_repeat_l30_readout_regime | drift | natural | explain_instruction | 1 | the_continuation | 4 | 0.0000 | 0.3281 | 1.4062 | 1.0781 | {' The': 2, ' White': 2} |
