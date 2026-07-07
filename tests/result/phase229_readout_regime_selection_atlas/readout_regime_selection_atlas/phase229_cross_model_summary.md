# Phase 229 readout regime selection source atlas

regime_rows: 1188

## Summary

| spec | group | type | variant | step | rows | rank improve | target logit delta | margin delta | target margin | winner changed | winners | top tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_l29_readout_regime | drift | natural | repeat_instruction | 3 | 4 | 620.5000 | 5.6562 | 24.2188 | -5.9688 | 4 | {'answer_boundary': 4} | {'Answer': 4} |
| qwen3_explain_l29_readout_regime | success | natural | repeat_instruction | 1 | 4 | 33.0000 | 8.6875 | 18.6250 | 4.6875 | 2 | {'echo': 2, 'the_continuation': 2} | {' red': 4} |
| glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 1 | 4 | -5467.0000 | -8.4224 | -12.9224 | -12.4849 | 4 | {'for_continuation': 4} | {' For': 4} |
| glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 1 | 4 | -6151.2500 | -7.2902 | -12.2121 | -12.6027 | 4 | {'for_continuation': 4} | {' For': 4} |
| qwen3_explain_l29_readout_regime | drift | natural | no_instruction | 3 | 4 | -351.0000 | 0.7500 | 17.1250 | -13.0625 | 4 | {'the_continuation': 4} | {'But': 2, 'The': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | repeat_instruction | 1 | 4 | 4.0000 | 4.1875 | 13.5625 | 6.0000 | 2 | {'space_boundary': 2, 'the_continuation': 2} | {' Red': 2, ' red': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | short_answer_instruction | 3 | 4 | -35.5000 | 1.1875 | 15.8125 | -14.3750 | 4 | {'the_continuation': 4} | {'The': 4} |
| qwen3_explain_l29_readout_regime | drift | natural | no_answer_anchor | 1 | 4 | -646.5000 | -12.0625 | -4.5000 | -12.0625 | 2 | {'the_continuation': 2, 'then_continuation': 2} | {' The': 2, ' Then': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | no_answer_anchor | 3 | 4 | 444.5000 | 2.4375 | 14.0000 | -16.1875 | 0 | {'because_reason': 4} | {'Because': 4} |
| qwen3_explain_l29_readout_regime | drift | natural | no_answer_anchor | 2 | 4 | 243.0000 | 3.8125 | 11.1250 | -10.0000 | 0 | {'period_stop': 4} | {'.': 4} |
| glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 3 | 4 | -10.7500 | -4.9531 | -9.0781 | -4.6875 | 4 | {'because_reason': 4} | {' because': 4} |
| glm4_repeat_l30_readout_regime | success | natural | explain_instruction | 2 | 4 | -84.0000 | -6.0078 | -7.8516 | -9.0234 | 4 | {'newline_boundary': 4} | {'\n': 4} |
| qwen3_explain_l29_readout_regime | drift | natural | because_removed | 3 | 4 | 37.0000 | 0.1250 | 13.6875 | -16.5000 | 0 | {'because_reason': 4} | {'Reason': 4} |
| qwen3_explain_l29_readout_regime | success | natural | repeat_instruction | 2 | 4 | 158.5000 | 4.0938 | 9.5312 | -9.0312 | 4 | {'comma_repeat': 4} | {',': 4} |
| glm4_repeat_l30_readout_regime | success | natural | no_instruction | 2 | 4 | -134.7500 | -6.3516 | -7.2578 | -8.4297 | 4 | {'newline_boundary': 4} | {'\n': 4} |
| glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 2 | 4 | -87.2500 | -6.2266 | -5.6016 | -6.7734 | 3 | {'comma_repeat': 1, 'newline_boundary': 3} | {'\n': 3, ' or': 1} |
| qwen3_explain_l29_readout_regime | success | natural | no_instruction | 3 | 4 | -2.0000 | -5.9375 | -4.1875 | -5.8125 | 0 | {'prose': 4} | {' a': 4} |
| glm4_repeat_l30_readout_regime | success | natural | no_instruction | 3 | 4 | -11.2500 | -3.8906 | -4.6875 | -0.2969 | 4 | {'because_reason': 1, 'prose': 3} | {' green': 1, ' of': 1, ' white': 1, ' with': 1} |
| glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 3 | 4 | -7.5000 | -3.9219 | -3.8438 | 0.5469 | 4 | {'because_reason': 1, 'prose': 3} | {' black': 1, ' green': 1, ' pink': 1, ' white': 1} |
| qwen3_explain_l29_readout_regime | success | natural | no_answer_anchor | 1 | 4 | -622.5000 | -5.0312 | 2.5938 | -11.3438 | 4 | {'then_continuation': 4} | {' Then': 4} |
| qwen3_explain_l29_readout_regime | success | natural | no_answer_anchor | 3 | 4 | -1.0000 | -5.3750 | -1.9375 | -3.5625 | 0 | {'prose': 4} | {' a': 4} |
| qwen3_explain_l29_readout_regime | success | natural | short_answer_instruction | 3 | 4 | -1.0000 | -4.0625 | -3.1250 | -4.7500 | 0 | {'prose': 4} | {' a': 4} |
| glm4_repeat_l30_readout_regime | drift | natural | no_instruction | 2 | 4 | -126.7500 | -2.8438 | -4.0469 | -8.6562 | 4 | {'newline_boundary': 4} | {'\n': 4} |
| qwen3_explain_l29_readout_regime | success | natural | no_answer_anchor | 2 | 4 | 80.5000 | -0.4062 | 6.4688 | -12.0938 | 0 | {'be_continuation': 4} | {' is': 4} |
| deepseek7b_explain_l24_readout_regime | success | natural | no_answer_anchor | 2 | 4 | 3934.5000 | -2.6719 | 3.5156 | -15.2344 | 2 | {'answer_boundary': 2, 'be_continuation': 2} | {' answer': 2, ' is': 2} |
| glm4_repeat_l30_readout_regime | drift | natural | comma_removed | 3 | 4 | -6.0000 | -2.5156 | -3.4844 | -0.6094 | 4 | {'prose': 4} | {' green': 2, ' white': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_all_a1 | 2 | 4 | -46266.5000 | -4.4688 | -1.5312 | -20.2812 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | no_instruction | 2 | 4 | -331.0000 | 0.6562 | 5.2188 | -15.9062 | 0 | {'period_stop': 4} | {'.': 2, '.\n\n': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | no_instruction | 3 | 2 | 649.0000 | 1.9375 | 3.5625 | -14.2500 | 0 | {'be_continuation': 2} | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_all_a1 | 2 | 4 | -31012.0000 | -4.4258 | -1.0508 | -19.8008 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | no_answer_anchor | 1 | 4 | 27064.0000 | 2.5547 | 2.6172 | -13.2930 | 2 | {'space_boundary': 2, 'the_continuation': 2} | {' ': 2, ' The': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | because_removed | 2 | 4 | -309.5000 | -1.2812 | -3.7812 | -24.9062 | 0 | {'period_stop': 4} | {'.': 2, '.\n': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | repeat_instruction | 3 | 2 | 82.0000 | -2.4062 | 2.3438 | -15.4688 | 0 | {'be_continuation': 2} | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_top64_a1 | 2 | 4 | -28304.5000 | -3.6880 | -1.0005 | -19.7505 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| qwen3_explain_l29_readout_regime | success | natural | no_instruction | 2 | 4 | -92.0000 | -1.3125 | 3.3125 | -15.2500 | 0 | {'be_continuation': 4} | {' is': 4} |
| qwen3_explain_l29_readout_regime | success | natural | repeat_instruction | 3 | 4 | 4.0000 | -4.0625 | 0.5625 | -1.0625 | 0 | {'prose': 4} | {' a': 4} |
| glm4_repeat_l30_readout_regime | drift | natural | explain_instruction | 2 | 4 | -98.2500 | -1.4844 | -3.1094 | -7.7188 | 4 | {'newline_boundary': 4} | {'\n': 4} |
| qwen3_explain_l29_readout_regime | success | natural | no_instruction | 1 | 4 | -26.5000 | 0.1562 | 4.3438 | -9.5938 | 0 | {'echo': 4} | {' Car': 2, ' Cherry': 2} |
| glm4_repeat_l30_readout_regime | drift | natural | explain_instruction | 3 | 4 | 0.7500 | 0.8125 | -3.6406 | -0.7656 | 4 | {'because_reason': 4} | {' because': 4} |
| qwen3_explain_l29_readout_regime | drift | natural | short_answer_instruction | 1 | 4 | -3.5000 | -3.0000 | -1.4375 | -9.0000 | 0 | {'echo': 2, 'the_continuation': 2} | {' Apple': 2, ' The': 2} |
| qwen3_explain_l29_readout_regime | success | natural | short_answer_instruction | 2 | 4 | -89.5000 | -1.7188 | 2.4062 | -16.1562 | 0 | {'be_continuation': 4} | {' is': 4} |
| qwen3_explain_l29_readout_regime | drift | natural | repeat_instruction | 2 | 4 | 78.5000 | 3.0312 | 0.9688 | -20.1562 | 4 | {'comma_repeat': 4} | {',': 4} |
| qwen3_explain_l29_readout_regime | success | natural | short_answer_instruction | 1 | 4 | 9.5000 | -0.6562 | 3.2188 | -10.7188 | 0 | {'echo': 4} | {' Car': 2, ' Cherry': 2} |
| glm4_repeat_l30_readout_regime | drift | natural | no_answer_anchor | 3 | 4 | -1.2500 | -0.7188 | -3.0000 | -0.1250 | 4 | {'be_continuation': 4} | {' green': 2, ' white': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | no_answer_anchor | 3 | 2 | 321.0000 | -0.3125 | 3.1875 | -14.6250 | 0 | {'be_continuation': 2} | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | no_answer_anchor | 3 | 4 | 2060.0000 | 0.0996 | 3.2871 | -12.9141 | 0 | {'be_continuation': 2, 'prose': 2} | {' is': 2, ' used': 2} |
| qwen3_explain_l29_readout_regime | success | natural | because_removed | 2 | 4 | -94.0000 | -0.3750 | -3.0000 | -21.5625 | 0 | {'be_continuation': 4} | {' is': 4} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_top16_a1 | 2 | 4 | -23259.5000 | -2.7422 | -0.6172 | -19.3672 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_all_a0.5 | 2 | 4 | -17369.0000 | -2.2266 | -1.0391 | -19.7891 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | because_removed | 1 | 4 | -29119.0000 | -1.3945 | -1.8633 | -17.7734 | 2 | {'echo': 4} | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_top64_a1 | 2 | 4 | -15324.5000 | -2.7734 | -0.3359 | -19.0859 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_product_top16_a1 | 1 | 4 | 13.5000 | 1.4688 | 1.5938 | -12.3438 | 0 | {'echo': 4} | {' Car': 2, ' Cherry': 2} |
| glm4_repeat_l30_readout_regime | drift | natural | short_answer_instruction | 2 | 4 | -65.5000 | -2.1094 | -0.8438 | -5.4531 | 0 | {'comma_repeat': 4} | {' or': 2, ',': 2} |
| glm4_repeat_l30_readout_regime | drift | natural | short_answer_instruction | 1 | 4 | -10.5000 | -0.4766 | -2.3984 | -2.7891 | 3 | {'echo': 3, 'the_continuation': 1} | {' The': 1, ' They': 1, ' Wood': 2} |
| glm4_repeat_l30_readout_regime | success | natural | no_answer_anchor | 3 | 4 | 0.0000 | -1.0781 | -1.7812 | 2.6094 | 3 | {'be_continuation': 3, 'echo': 1} | {' red': 4} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_top16_a1 | 2 | 4 | -15571.0000 | -2.3789 | -0.3164 | -19.0664 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_product_all_a1 | 1 | 4 | 11.5000 | 1.1562 | 1.5312 | -12.4062 | 0 | {'echo': 4} | {' Car': 2, ' Cherry': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | repeat_instruction | 1 | 2 | 4361.0000 | 1.2812 | 1.4062 | -11.3906 | 0 | {'the_continuation': 2} | {' H': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top64_a1 | 2 | 2 | -12029.0000 | -1.6484 | -1.0234 | -13.3438 | 0 | {'newline_boundary': 2} | {'orses': 2} |
| glm4_repeat_l30_readout_regime | drift | natural | comma_removed | 2 | 4 | -31.0000 | -1.9375 | -0.6719 | -5.2812 | 4 | {'newline_boundary': 2, 'period_stop': 2} | {'\n': 2, '.\n': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | short_answer_instruction | 1 | 4 | 27510.5000 | 2.0449 | 0.5137 | -15.3965 | 2 | {'echo': 4} | {' Cup': 2, ' Glass': 2} |
| glm4_repeat_l30_readout_regime | success | natural | short_answer_instruction | 1 | 4 | -4.0000 | -0.4297 | -2.1016 | -1.6641 | 0 | {'echo': 3, 'the_continuation': 1} | {' Cardinal': 1, ' Cherry': 1, ' Red': 1, ' The': 1} |
| qwen3_explain_l29_readout_regime | success | patch | patch_product_top64_a1 | 1 | 4 | 10.5000 | 1.2188 | 1.2812 | -12.6562 | 0 | {'echo': 4} | {' Car': 2, ' Cherry': 2} |
| qwen3_explain_l29_readout_regime | drift | natural | no_instruction | 1 | 4 | -4.5000 | -1.5000 | -0.9375 | -8.5000 | 0 | {'echo': 2, 'the_continuation': 2} | {' Apple': 2, ' The': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_top64_a1 | 2 | 2 | -11471.0000 | -1.5078 | -0.8828 | -13.2031 | 0 | {'newline_boundary': 2} | {'orses': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | no_instruction | 2 | 4 | 385.5000 | -1.2969 | 1.0156 | -17.7344 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | no_instruction | 1 | 4 | 17258.5000 | 1.2539 | 1.0039 | -14.9062 | 2 | {'echo': 4} | {' Cup': 2, ' Glass': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | short_answer_instruction | 3 | 2 | 343.0000 | 1.1250 | 1.1250 | -16.6875 | 0 | {'be_continuation': 2} | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top64_a1 | 3 | 2 | 322.0000 | 1.3750 | 0.8750 | -16.9375 | 0 | {'be_continuation': 2} | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_top64_a0.5 | 2 | 4 | -9091.0000 | -1.6172 | -0.6172 | -19.3672 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_readout_regime | drift | natural | repeat_instruction | 2 | 2 | 14899.0000 | 1.1055 | 1.1055 | -11.2148 | 0 | {'newline_boundary': 2} | {'orses': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_all_a1 | 2 | 4 | -66.5000 | -0.8438 | -1.3438 | -19.9062 | 0 | {'be_continuation': 4} | {' is': 4} |
| glm4_repeat_l30_readout_regime | drift | natural | no_instruction | 3 | 4 | -1.2500 | 0.0781 | -2.0625 | 0.8125 | 4 | {'because_reason': 2, 'prose': 2} | {' but': 4} |
| glm4_repeat_l30_readout_regime | success | natural | comma_removed | 3 | 4 | 0.0000 | -1.0781 | -1.0000 | 3.3906 | 2 | {'echo': 2, 'prose': 2} | {' red': 4} |
| glm4_repeat_l30_readout_regime | success | natural | comma_removed | 2 | 4 | -2.7500 | -1.5625 | -0.4844 | -1.6562 | 3 | {'comma_repeat': 1, 'newline_boundary': 2, 'period_stop': 1} | {'\n': 3, ',': 1} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_all_a0.5 | 2 | 4 | -6491.0000 | -1.4844 | -0.4219 | -19.1719 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| deepseek7b_explain_l24_readout_regime | success | natural | repeat_instruction | 1 | 4 | 20699.5000 | 1.2114 | 0.6802 | -15.2300 | 0 | {'echo': 2, 'the_continuation': 2} | {' Glass': 2, ' The': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_top16_a1 | 2 | 2 | -6077.0000 | -1.2891 | -0.6016 | -12.9219 | 0 | {'newline_boundary': 2} | {'orses': 2} |
| qwen3_explain_l29_readout_regime | success | patch | patch_product_top16_a0.5 | 1 | 4 | 9.0000 | 0.8438 | 1.0312 | -12.9062 | 0 | {'echo': 4} | {' Car': 2, ' Cherry': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_gate_up_pair_top64_a1 | 3 | 2 | 301.0000 | 1.2500 | 0.6250 | -17.1875 | 0 | {'be_continuation': 2} | {' are': 2} |
| deepseek7b_explain_l24_readout_regime | drift | patch | patch_product_top16_a1 | 2 | 2 | -4518.0000 | -1.2109 | -0.5859 | -12.9062 | 0 | {'newline_boundary': 2} | {'orses': 2} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_gate_up_pair_top16_a1 | 3 | 4 | -1323.5000 | 0.5566 | 1.2129 | -14.9883 | 0 | {'be_continuation': 2, 'prose': 2} | {' is': 2, ' used': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_top64_a1 | 3 | 4 | -25.0000 | -0.2812 | 1.4688 | -28.7188 | 0 | {'because_reason': 4} | {'Because': 4} |
| qwen3_explain_l29_readout_regime | success | patch | patch_gate_up_pair_top16_a1 | 1 | 4 | 7.0000 | 0.8750 | 0.8750 | -13.0625 | 0 | {'echo': 4} | {' Car': 2, ' Cherry': 2} |
| qwen3_explain_l29_readout_regime | success | natural | because_removed | 3 | 4 | -2.0000 | 0.5625 | -1.1875 | -2.8125 | 0 | {'prose': 4} | {' a': 2, ' red': 2} |
| glm4_repeat_l30_readout_regime | drift | natural | short_answer_instruction | 3 | 4 | 2.0000 | 1.5000 | 0.2500 | 3.1250 | 4 | {'because_reason': 4} | {' brown': 2, ' gray': 1, ' grey': 1} |
| glm4_repeat_l30_readout_regime | success | natural | no_instruction | 1 | 4 | -8.0000 | -0.4062 | -1.3281 | -0.8906 | 3 | {'the_continuation': 4} | {' Red': 3, ' The': 1} |
| deepseek7b_explain_l24_readout_regime | success | patch | patch_product_top16_a0.5 | 2 | 4 | -8144.0000 | -1.2344 | -0.4844 | -19.2344 | 0 | {'be_continuation': 2, 'echo': 2} | {' cup': 2, ' is': 2} |
| qwen3_explain_l29_readout_regime | drift | patch | patch_product_top16_a1 | 3 | 4 | 85.5000 | 0.0938 | 1.5938 | -28.5938 | 0 | {'because_reason': 4} | {'Because': 4} |
| qwen3_explain_l29_readout_regime | success | patch | patch_product_all_a0.5 | 1 | 4 | 8.0000 | 0.7500 | 0.9375 | -13.0000 | 0 | {'echo': 4} | {' Car': 2, ' Cherry': 2} |
