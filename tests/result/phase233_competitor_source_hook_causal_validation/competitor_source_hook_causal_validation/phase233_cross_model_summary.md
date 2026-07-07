# Phase 233 competitor source hook causal validation

observation_rows: 1368
suppression_rows: 8208

## Prompt Source Summary

| model | group | variant | step | regime | rows | switch rate | target delta | regime delta | comp-target | winners | top tokens |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| glm4 | success | no_answer_anchor | 1 | for_continuation | 10 | 1.0000 | -8.3561 | 9.0469 | 17.4029 | {'for_continuation': 10} | {' For': 10} |
| glm4 | success | explain_instruction | 2 | because_reason | 10 | 1.0000 | -5.5812 | 7.5422 | 13.1234 | {'newline_boundary': 10} | {'\n': 10} |
| glm4 | success | explain_instruction | 3 | because_reason | 10 | 1.0000 | -4.4062 | 8.6594 | 13.0656 | {'because_reason': 10} | {' because': 10} |
| glm4 | success | no_answer_anchor | 1 | newline_boundary | 10 | 1.0000 | -8.3561 | 4.2084 | 12.5645 | {'for_continuation': 10} | {' For': 10} |
| glm4 | success | no_answer_anchor | 1 | because_reason | 10 | 1.0000 | -8.3561 | 3.0484 | 11.4045 | {'for_continuation': 10} | {' For': 10} |
| glm4 | drift | no_answer_anchor | 1 | for_continuation | 10 | 1.0000 | -3.9463 | 7.1797 | 11.1260 | {'for_continuation': 10} | {' For': 10} |
| glm4 | success | explain_instruction | 2 | newline_boundary | 10 | 1.0000 | -5.5812 | 3.6063 | 9.1875 | {'newline_boundary': 10} | {'\n': 10} |
| glm4 | success | no_instruction | 2 | newline_boundary | 10 | 1.0000 | -6.1156 | 2.6812 | 8.7969 | {'newline_boundary': 10} | {'\n': 10} |
| qwen3 | success | no_answer_anchor | 1 | because_reason | 10 | 1.0000 | -5.3719 | 2.7375 | 8.1094 | {'then_continuation': 10} | {' Then': 10} |
| glm4 | drift | no_answer_anchor | 1 | newline_boundary | 10 | 1.0000 | -3.9463 | 4.1263 | 8.0725 | {'for_continuation': 10} | {' For': 10} |
| glm4 | success | no_instruction | 2 | for_continuation | 10 | 1.0000 | -6.1156 | 1.7094 | 7.8250 | {'newline_boundary': 10} | {'\n': 10} |
| glm4 | success | no_instruction | 2 | because_reason | 10 | 1.0000 | -6.1156 | 1.5859 | 7.7016 | {'newline_boundary': 10} | {'\n': 10} |
| qwen3 | success | no_answer_anchor | 1 | answer_boundary | 10 | 1.0000 | -5.3719 | 2.2500 | 7.6219 | {'then_continuation': 10} | {' Then': 10} |
| glm4 | success | no_instruction | 3 | because_reason | 10 | 1.0000 | -3.4813 | 3.4500 | 6.9313 | {'because_reason': 2, 'prose': 8} | {' black': 1, ' brown': 1, ' green': 2, ' of': 2, ' white': 2, ' with': 2} |
| glm4 | success | explain_instruction | 2 | for_continuation | 10 | 1.0000 | -5.5812 | 1.1844 | 6.7656 | {'newline_boundary': 10} | {'\n': 10} |
| glm4 | success | no_instruction | 3 | for_continuation | 10 | 1.0000 | -3.4813 | 2.5125 | 5.9938 | {'because_reason': 2, 'prose': 8} | {' black': 1, ' brown': 1, ' green': 2, ' of': 2, ' white': 2, ' with': 2} |
| glm4 | success | explain_instruction | 3 | for_continuation | 10 | 1.0000 | -4.4062 | 1.5375 | 5.9437 | {'because_reason': 10} | {' because': 10} |
| glm4 | drift | no_answer_anchor | 1 | because_reason | 10 | 1.0000 | -3.9463 | 1.8328 | 5.7791 | {'for_continuation': 10} | {' For': 10} |
| qwen3 | success | no_answer_anchor | 1 | period_stop | 10 | 1.0000 | -5.3719 | 0.2500 | 5.6219 | {'then_continuation': 10} | {' Then': 10} |
| glm4 | success | explain_instruction | 3 | newline_boundary | 10 | 1.0000 | -4.4062 | 0.2609 | 4.6672 | {'because_reason': 10} | {' because': 10} |
| glm4 | success | no_instruction | 3 | newline_boundary | 10 | 1.0000 | -3.4813 | -0.2031 | 3.2781 | {'because_reason': 2, 'prose': 8} | {' black': 1, ' brown': 1, ' green': 2, ' of': 2, ' white': 2, ' with': 2} |
| deepseek7b | drift | no_instruction | 1 | be_continuation | 2 | 1.0000 | -0.0625 | 2.7500 | 2.8125 | {'echo': 2} | {' H': 2} |
| deepseek7b | drift | no_instruction | 1 | echo | 2 | 1.0000 | -0.0625 | 2.1250 | 2.1875 | {'echo': 2} | {' H': 2} |
| deepseek7b | drift | short_answer_instruction | 1 | echo | 2 | 1.0000 | -0.0508 | 1.6875 | 1.7383 | {'echo': 2} | {' H': 2} |
| deepseek7b | drift | no_instruction | 1 | prose | 2 | 1.0000 | -0.0625 | 0.8125 | 0.8750 | {'echo': 2} | {' H': 2} |
| deepseek7b | drift | short_answer_instruction | 1 | be_continuation | 2 | 1.0000 | -0.0508 | 0.7812 | 0.8320 | {'echo': 2} | {' H': 2} |
| deepseek7b | drift | short_answer_instruction | 1 | prose | 2 | 1.0000 | -0.0508 | 0.0625 | 0.1133 | {'echo': 2} | {' H': 2} |
| deepseek7b | drift | short_answer_instruction | 1 | the_continuation | 2 | 1.0000 | -0.0508 | -0.3125 | -0.2617 | {'echo': 2} | {' H': 2} |
| deepseek7b | drift | no_instruction | 1 | the_continuation | 2 | 1.0000 | -0.0625 | -1.5000 | -1.4375 | {'echo': 2} | {' H': 2} |
| qwen3 | success | repeat_instruction | 2 | because_reason | 10 | 1.0000 | 4.9656 | 1.4031 | -3.5625 | {'comma_repeat': 10} | {',': 10} |
| qwen3 | success | repeat_instruction | 2 | answer_boundary | 10 | 1.0000 | 4.9656 | -1.9563 | -6.9219 | {'comma_repeat': 10} | {',': 10} |
| qwen3 | success | repeat_instruction | 2 | period_stop | 10 | 1.0000 | 4.9656 | -2.0125 | -6.9781 | {'comma_repeat': 10} | {',': 10} |
| qwen3 | drift | no_answer_anchor | 1 | because_reason | 10 | 0.8000 | -10.9062 | 3.5000 | 14.4062 | {'the_continuation': 2, 'then_continuation': 8} | {' The': 2, ' Then': 8} |
| qwen3 | drift | no_answer_anchor | 1 | period_stop | 10 | 0.8000 | -10.9062 | -0.0813 | 10.8250 | {'the_continuation': 2, 'then_continuation': 8} | {' The': 2, ' Then': 8} |
| qwen3 | drift | no_answer_anchor | 1 | answer_boundary | 10 | 0.8000 | -10.9062 | -0.1875 | 10.7188 | {'the_continuation': 2, 'then_continuation': 8} | {' The': 2, ' Then': 8} |
| glm4 | success | no_answer_anchor | 3 | for_continuation | 10 | 0.8000 | -1.1187 | 3.3781 | 4.4969 | {'be_continuation': 8, 'echo': 2} | {' black': 1, ' brown': 1, ' red': 8} |
| glm4 | success | no_answer_anchor | 3 | because_reason | 10 | 0.8000 | -1.1187 | 0.7469 | 1.8656 | {'be_continuation': 8, 'echo': 2} | {' black': 1, ' brown': 1, ' red': 8} |
| glm4 | success | no_answer_anchor | 3 | newline_boundary | 10 | 0.8000 | -1.1187 | -0.6172 | 0.5016 | {'be_continuation': 8, 'echo': 2} | {' black': 1, ' brown': 1, ' red': 8} |
| qwen3 | drift | repeat_instruction | 1 | period_stop | 10 | 0.8000 | 5.0000 | 2.2687 | -2.7313 | {'space_boundary': 8, 'the_continuation': 2} | {' Red': 2, ' black': 2, ' brown': 4, ' red': 2} |
| qwen3 | drift | repeat_instruction | 1 | because_reason | 10 | 0.8000 | 5.0000 | 1.8625 | -3.1375 | {'space_boundary': 8, 'the_continuation': 2} | {' Red': 2, ' black': 2, ' brown': 4, ' red': 2} |
| qwen3 | drift | repeat_instruction | 1 | answer_boundary | 10 | 0.8000 | 5.0000 | -2.8625 | -7.8625 | {'space_boundary': 8, 'the_continuation': 2} | {' Red': 2, ' black': 2, ' brown': 4, ' red': 2} |
| deepseek7b | success | short_answer_instruction | 1 | echo | 6 | 0.6667 | 1.5404 | 1.3125 | -0.2279 | {'echo': 4, 'prose': 2} | {' A': 2, ' Cup': 2, ' Glass': 2} |
| deepseek7b | success | no_instruction | 1 | be_continuation | 6 | 0.6667 | 1.0859 | 0.7604 | -0.3255 | {'echo': 6} | {' Cup': 2, ' Dog': 2, ' Glass': 2} |
| deepseek7b | success | no_instruction | 1 | echo | 6 | 0.6667 | 1.0859 | 0.3333 | -0.7526 | {'echo': 6} | {' Cup': 2, ' Dog': 2, ' Glass': 2} |
| deepseek7b | success | short_answer_instruction | 1 | prose | 6 | 0.6667 | 1.5404 | -0.0208 | -1.5612 | {'echo': 4, 'prose': 2} | {' A': 2, ' Cup': 2, ' Glass': 2} |
| deepseek7b | success | short_answer_instruction | 1 | be_continuation | 6 | 0.6667 | 1.5404 | -0.0417 | -1.5820 | {'echo': 4, 'prose': 2} | {' A': 2, ' Cup': 2, ' Glass': 2} |
| deepseek7b | success | no_instruction | 1 | prose | 6 | 0.6667 | 1.0859 | -1.1458 | -2.2318 | {'echo': 6} | {' Cup': 2, ' Dog': 2, ' Glass': 2} |
| deepseek7b | success | short_answer_instruction | 1 | the_continuation | 6 | 0.6667 | 1.5404 | -0.8750 | -2.4154 | {'echo': 4, 'prose': 2} | {' A': 2, ' Cup': 2, ' Glass': 2} |
| deepseek7b | success | no_instruction | 1 | the_continuation | 6 | 0.6667 | 1.0859 | -3.3333 | -4.4193 | {'echo': 6} | {' Cup': 2, ' Dog': 2, ' Glass': 2} |
| glm4 | success | no_instruction | 1 | because_reason | 10 | 0.6000 | -0.4719 | 2.0391 | 2.5109 | {'the_continuation': 10} | {' B': 1, ' Red': 6, ' The': 3} |
| glm4 | success | no_instruction | 1 | for_continuation | 10 | 0.6000 | -0.4719 | 1.3562 | 1.8281 | {'the_continuation': 10} | {' B': 1, ' Red': 6, ' The': 3} |
| glm4 | success | no_instruction | 1 | newline_boundary | 10 | 0.6000 | -0.4719 | 0.2349 | 0.7068 | {'the_continuation': 10} | {' B': 1, ' Red': 6, ' The': 3} |
| qwen3 | drift | repeat_instruction | 2 | because_reason | 10 | 0.6000 | 3.5625 | 1.3125 | -2.2500 | {'be_continuation': 4, 'comma_repeat': 6} | {' is': 4, ',': 6} |
| qwen3 | drift | repeat_instruction | 2 | answer_boundary | 10 | 0.6000 | 3.5625 | -0.2375 | -3.8000 | {'be_continuation': 4, 'comma_repeat': 6} | {' is': 4, ',': 6} |
| qwen3 | success | repeat_instruction | 1 | period_stop | 10 | 0.6000 | 7.5625 | 2.0438 | -5.5187 | {'echo': 4, 'space_boundary': 4, 'the_continuation': 2} | {' brown': 5, ' red': 4, ' white': 1} |
| qwen3 | drift | repeat_instruction | 2 | period_stop | 10 | 0.6000 | 3.5625 | -1.9750 | -5.5375 | {'be_continuation': 4, 'comma_repeat': 6} | {' is': 4, ',': 6} |
| qwen3 | success | repeat_instruction | 1 | because_reason | 10 | 0.6000 | 7.5625 | 0.9375 | -6.6250 | {'echo': 4, 'space_boundary': 4, 'the_continuation': 2} | {' brown': 5, ' red': 4, ' white': 1} |
| qwen3 | success | repeat_instruction | 1 | answer_boundary | 10 | 0.6000 | 7.5625 | -0.9563 | -8.5188 | {'echo': 4, 'space_boundary': 4, 'the_continuation': 2} | {' brown': 5, ' red': 4, ' white': 1} |
| glm4 | drift | explain_instruction | 2 | because_reason | 10 | 0.4000 | -0.2234 | 4.0961 | 4.3195 | {'be_continuation': 5, 'echo': 1, 'newline_boundary': 4} | {'\n': 4, ' bus': 1, ' is': 3, ' oil': 1, ' used': 1} |
| glm4 | drift | explain_instruction | 3 | because_reason | 10 | 0.4000 | 0.2922 | 3.9797 | 3.6875 | {'be_continuation': 1, 'because_reason': 4, 'prose': 5} | {' because': 4, ' is': 1, ' used': 5} |
| qwen3 | drift | repeat_instruction | 3 | answer_boundary | 10 | 0.4000 | 1.1625 | 3.9375 | 2.7750 | {'answer_boundary': 4, 'be_continuation': 4, 'prose': 2} | {' a': 2, ' be': 4, 'Answer': 4} |
| glm4 | drift | explain_instruction | 2 | newline_boundary | 10 | 0.4000 | -0.2234 | 2.1219 | 2.3453 | {'be_continuation': 5, 'echo': 1, 'newline_boundary': 4} | {'\n': 4, ' bus': 1, ' is': 3, ' oil': 1, ' used': 1} |
| glm4 | drift | no_instruction | 3 | because_reason | 10 | 0.4000 | 0.0500 | 2.1016 | 2.0516 | {'be_continuation': 1, 'because_reason': 2, 'prose': 7} | {' a': 1, ' but': 4, ' is': 1, ' used': 4} |
| glm4 | drift | explain_instruction | 2 | for_continuation | 10 | 0.4000 | -0.2234 | 1.6781 | 1.9016 | {'be_continuation': 5, 'echo': 1, 'newline_boundary': 4} | {'\n': 4, ' bus': 1, ' is': 3, ' oil': 1, ' used': 1} |
| glm4 | drift | no_instruction | 2 | for_continuation | 10 | 0.4000 | -0.8187 | 1.0281 | 1.8469 | {'be_continuation': 5, 'echo': 1, 'newline_boundary': 4} | {'\n': 4, ' bus': 1, ' is': 3, ' oil': 1, ' used': 1} |
| glm4 | drift | no_answer_anchor | 3 | for_continuation | 10 | 0.4000 | -0.1406 | 1.6219 | 1.7625 | {'be_continuation': 5, 'prose': 5} | {' green': 2, ' is': 1, ' used': 5, ' white': 2} |
| glm4 | drift | no_instruction | 2 | because_reason | 10 | 0.4000 | -0.8187 | 0.8648 | 1.6836 | {'be_continuation': 5, 'echo': 1, 'newline_boundary': 4} | {'\n': 4, ' bus': 1, ' is': 3, ' oil': 1, ' used': 1} |
| glm4 | drift | no_instruction | 3 | for_continuation | 10 | 0.4000 | 0.0500 | 1.4469 | 1.3969 | {'be_continuation': 1, 'because_reason': 2, 'prose': 7} | {' a': 1, ' but': 4, ' is': 1, ' used': 4} |
| glm4 | drift | no_instruction | 2 | newline_boundary | 10 | 0.4000 | -0.8187 | 0.5750 | 1.3938 | {'be_continuation': 5, 'echo': 1, 'newline_boundary': 4} | {'\n': 4, ' bus': 1, ' is': 3, ' oil': 1, ' used': 1} |
| glm4 | drift | explain_instruction | 3 | for_continuation | 10 | 0.4000 | 0.2922 | 1.5219 | 1.2297 | {'be_continuation': 1, 'because_reason': 4, 'prose': 5} | {' because': 4, ' is': 1, ' used': 5} |
| qwen3 | drift | repeat_instruction | 3 | period_stop | 10 | 0.4000 | 1.1625 | 1.9000 | 0.7375 | {'answer_boundary': 4, 'be_continuation': 4, 'prose': 2} | {' a': 2, ' be': 4, 'Answer': 4} |
| glm4 | drift | no_answer_anchor | 3 | because_reason | 10 | 0.4000 | -0.1406 | 0.3859 | 0.5266 | {'be_continuation': 5, 'prose': 5} | {' green': 2, ' is': 1, ' used': 5, ' white': 2} |
| glm4 | drift | no_answer_anchor | 3 | newline_boundary | 10 | 0.4000 | -0.1406 | -0.2781 | -0.1375 | {'be_continuation': 5, 'prose': 5} | {' green': 2, ' is': 1, ' used': 5, ' white': 2} |
| glm4 | drift | explain_instruction | 3 | newline_boundary | 10 | 0.4000 | 0.2922 | -0.5672 | -0.8594 | {'be_continuation': 1, 'because_reason': 4, 'prose': 5} | {' because': 4, ' is': 1, ' used': 5} |
| glm4 | drift | no_instruction | 3 | newline_boundary | 10 | 0.4000 | 0.0500 | -1.8359 | -1.8859 | {'be_continuation': 1, 'because_reason': 2, 'prose': 7} | {' a': 1, ' but': 4, ' is': 1, ' used': 4} |
| qwen3 | drift | repeat_instruction | 3 | because_reason | 10 | 0.4000 | 1.1625 | -8.5531 | -9.7156 | {'answer_boundary': 4, 'be_continuation': 4, 'prose': 2} | {' a': 2, ' be': 4, 'Answer': 4} |
| deepseek7b | success | because_removed | 1 | echo | 6 | 0.3333 | -1.0990 | 0.4167 | 1.5156 | {'echo': 4, 'the_continuation': 2} | {' Cup': 2, ' Glass': 2, ' The': 2} |
| deepseek7b | success | because_removed | 1 | the_continuation | 6 | 0.3333 | -1.0990 | -0.8750 | 0.2240 | {'echo': 4, 'the_continuation': 2} | {' Cup': 2, ' Glass': 2, ' The': 2} |
| deepseek7b | success | because_removed | 1 | prose | 6 | 0.3333 | -1.0990 | -0.8750 | 0.2240 | {'echo': 4, 'the_continuation': 2} | {' Cup': 2, ' Glass': 2, ' The': 2} |
| deepseek7b | success | because_removed | 1 | be_continuation | 6 | 0.3333 | -1.0990 | -1.6042 | -0.5052 | {'echo': 4, 'the_continuation': 2} | {' Cup': 2, ' Glass': 2, ' The': 2} |

## Suppression Summary

| model | group | variant | step | component | alpha | regime | rows | reduce rate | margin help | winner change | regime delta | target delta | margin delta | winners |
| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | success | no_answer_anchor | 1 | down_out | 1.00 | period_stop | 10 | 1.0000 | 1.0000 | 0.5000 | -1.2125 | -0.0906 | 0.6594 | {'answer_boundary': 2, 'echo': 1, 'the_continuation': 2, 'then_continuation': 5} |
| qwen3 | success | no_answer_anchor | 1 | product | 1.00 | period_stop | 10 | 1.0000 | 1.0000 | 0.5000 | -1.2000 | -0.0906 | 0.6844 | {'answer_boundary': 2, 'echo': 1, 'the_continuation': 2, 'then_continuation': 5} |
| glm4 | success | no_answer_anchor | 1 | down_out | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -1.1250 | 3.3186 | 4.4436 | {'for_continuation': 10} |
| glm4 | success | no_answer_anchor | 1 | gate_up_pair | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -1.1187 | 3.2936 | 4.4123 | {'for_continuation': 10} |
| glm4 | success | no_answer_anchor | 1 | product | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -1.1062 | 3.3467 | 4.4529 | {'for_continuation': 10} |
| glm4 | drift | no_answer_anchor | 1 | down_out | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.9313 | 0.6433 | 1.5746 | {'for_continuation': 10} |
| glm4 | drift | no_answer_anchor | 1 | gate_up_pair | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.9313 | 0.5969 | 1.5281 | {'for_continuation': 10} |
| glm4 | drift | no_answer_anchor | 1 | product | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.9187 | 0.6464 | 1.5651 | {'for_continuation': 10} |
| qwen3 | success | no_answer_anchor | 1 | product | 1.00 | because_reason | 10 | 1.0000 | 1.0000 | 0.5000 | -0.9062 | -0.0906 | 0.6844 | {'answer_boundary': 2, 'echo': 1, 'the_continuation': 2, 'then_continuation': 5} |
| qwen3 | success | no_answer_anchor | 1 | down_out | 1.00 | because_reason | 10 | 1.0000 | 1.0000 | 0.5000 | -0.8812 | -0.0906 | 0.6594 | {'answer_boundary': 2, 'echo': 1, 'the_continuation': 2, 'then_continuation': 5} |
| qwen3 | drift | no_answer_anchor | 1 | product | 0.50 | period_stop | 10 | 1.0000 | 1.0000 | 0.0000 | -0.6250 | 0.0125 | 0.3875 | {'the_continuation': 2, 'then_continuation': 8} |
| deepseek7b | drift | short_answer_instruction | 2 | product | 1.00 | echo | 2 | 1.0000 | 1.0000 | 0.0000 | -0.6250 | 0.0859 | 0.2109 | {'newline_boundary': 2} |
| qwen3 | success | no_answer_anchor | 1 | product | 0.50 | period_stop | 10 | 1.0000 | 1.0000 | 0.5000 | -0.6188 | 0.0094 | 0.4469 | {'answer_boundary': 2, 'echo': 1, 'the_continuation': 2, 'then_continuation': 5} |
| qwen3 | success | no_answer_anchor | 1 | gate_up_pair | 0.50 | period_stop | 10 | 1.0000 | 1.0000 | 0.5000 | -0.6000 | 0.3656 | 0.7406 | {'answer_boundary': 2, 'echo': 1, 'the_continuation': 2, 'then_continuation': 5} |
| deepseek7b | drift | short_answer_instruction | 2 | down_out | 1.00 | echo | 2 | 1.0000 | 1.0000 | 0.0000 | -0.5938 | 0.0781 | 0.2031 | {'newline_boundary': 2} |
| qwen3 | success | no_answer_anchor | 1 | down_out | 0.50 | period_stop | 10 | 1.0000 | 1.0000 | 0.3000 | -0.5813 | -0.0031 | 0.4344 | {'echo': 1, 'the_continuation': 2, 'then_continuation': 7} |
| deepseek7b | drift | short_answer_instruction | 2 | gate_up_pair | 1.00 | echo | 2 | 1.0000 | 1.0000 | 0.0000 | -0.5625 | 0.0938 | 0.2188 | {'newline_boundary': 2} |
| glm4 | success | no_answer_anchor | 1 | product | 0.50 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.5062 | 1.8342 | 2.3404 | {'for_continuation': 10} |
| glm4 | success | no_answer_anchor | 1 | down_out | 0.50 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.4875 | 1.8326 | 2.3201 | {'for_continuation': 10} |
| glm4 | drift | no_answer_anchor | 1 | down_out | 0.50 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.4500 | 0.3269 | 0.7769 | {'for_continuation': 10} |
| glm4 | drift | no_answer_anchor | 1 | product | 0.50 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.4313 | 0.3224 | 0.7537 | {'for_continuation': 10} |
| glm4 | success | no_answer_anchor | 1 | gate_up_pair | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.4234 | 3.2936 | 4.4123 | {'for_continuation': 10} |
| glm4 | success | no_answer_anchor | 1 | down_out | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.4000 | 3.3186 | 4.4436 | {'for_continuation': 10} |
| deepseek7b | success | short_answer_instruction | 3 | product | 1.00 | be_continuation | 6 | 1.0000 | 1.0000 | 0.0000 | -0.3958 | -0.1081 | 0.1836 | {'be_continuation': 4, 'prose': 2} |
| glm4 | success | no_answer_anchor | 1 | product | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.3781 | 3.3467 | 4.4529 | {'for_continuation': 10} |
| glm4 | success | no_answer_anchor | 1 | gate_up_pair | 0.50 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.3750 | 1.4186 | 1.7936 | {'for_continuation': 10} |
| glm4 | drift | no_answer_anchor | 1 | gate_up_pair | 0.50 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.3750 | 0.2334 | 0.6084 | {'for_continuation': 10} |
| qwen3 | success | no_answer_anchor | 1 | product | 0.50 | because_reason | 10 | 1.0000 | 1.0000 | 0.5000 | -0.3750 | 0.0094 | 0.4469 | {'answer_boundary': 2, 'echo': 1, 'the_continuation': 2, 'then_continuation': 5} |
| deepseek7b | drift | short_answer_instruction | 2 | gate_up_pair | 0.50 | echo | 2 | 1.0000 | 1.0000 | 0.0000 | -0.3750 | 0.1250 | 0.1875 | {'newline_boundary': 2} |
| deepseek7b | success | short_answer_instruction | 3 | down_out | 1.00 | be_continuation | 6 | 1.0000 | 1.0000 | 0.0000 | -0.3646 | -0.0781 | 0.1719 | {'be_continuation': 4, 'prose': 2} |
| deepseek7b | success | short_answer_instruction | 3 | gate_up_pair | 1.00 | echo | 6 | 1.0000 | 1.0000 | 0.0000 | -0.3542 | -0.1523 | 0.1810 | {'be_continuation': 4, 'prose': 2} |
| qwen3 | success | no_answer_anchor | 1 | down_out | 0.50 | because_reason | 10 | 1.0000 | 1.0000 | 0.3000 | -0.3438 | -0.0031 | 0.4344 | {'echo': 1, 'the_continuation': 2, 'then_continuation': 7} |
| deepseek7b | success | short_answer_instruction | 3 | gate_up_pair | 1.00 | be_continuation | 6 | 1.0000 | 1.0000 | 0.0000 | -0.3333 | -0.1523 | 0.1810 | {'be_continuation': 4, 'prose': 2} |
| deepseek7b | success | short_answer_instruction | 3 | gate_up_pair | 1.00 | prose | 6 | 1.0000 | 1.0000 | 0.0000 | -0.3333 | -0.1523 | 0.1810 | {'be_continuation': 4, 'prose': 2} |
| glm4 | success | no_instruction | 1 | gate_up_pair | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.3259 | 0.2188 | 0.4188 | {'the_continuation': 10} |
| glm4 | success | no_instruction | 1 | product | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.3126 | 0.2219 | 0.4156 | {'the_continuation': 10} |
| glm4 | success | no_instruction | 1 | down_out | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.3107 | 0.2313 | 0.4062 | {'the_continuation': 10} |
| deepseek7b | success | short_answer_instruction | 3 | product | 1.00 | prose | 6 | 1.0000 | 1.0000 | 0.0000 | -0.2917 | -0.1081 | 0.1836 | {'be_continuation': 4, 'prose': 2} |
| deepseek7b | success | short_answer_instruction | 3 | product | 1.00 | echo | 6 | 1.0000 | 1.0000 | 0.0000 | -0.2917 | -0.1081 | 0.1836 | {'be_continuation': 4, 'prose': 2} |
| deepseek7b | success | short_answer_instruction | 3 | gate_up_pair | 1.00 | the_continuation | 6 | 1.0000 | 1.0000 | 0.0000 | -0.2708 | -0.1523 | 0.1810 | {'be_continuation': 4, 'prose': 2} |
| deepseek7b | success | short_answer_instruction | 3 | down_out | 1.00 | echo | 6 | 1.0000 | 1.0000 | 0.0000 | -0.2708 | -0.0781 | 0.1719 | {'be_continuation': 4, 'prose': 2} |
| qwen3 | drift | no_answer_anchor | 1 | product | 0.50 | because_reason | 10 | 0.8000 | 1.0000 | 0.0000 | -0.2625 | 0.0125 | 0.3875 | {'the_continuation': 2, 'then_continuation': 8} |
| glm4 | drift | no_answer_anchor | 1 | down_out | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.2578 | 0.6433 | 1.5746 | {'for_continuation': 10} |
| glm4 | drift | no_answer_anchor | 1 | product | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.2578 | 0.6464 | 1.5651 | {'for_continuation': 10} |
| qwen3 | success | no_answer_anchor | 1 | gate_up_pair | 0.50 | because_reason | 10 | 1.0000 | 1.0000 | 0.5000 | -0.2562 | 0.3656 | 0.7406 | {'answer_boundary': 2, 'echo': 1, 'the_continuation': 2, 'then_continuation': 5} |
| deepseek7b | drift | no_instruction | 1 | gate_up_pair | 1.00 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.2500 | 0.9531 | 0.2656 | {'echo': 2} |
| deepseek7b | drift | short_answer_instruction | 1 | down_out | 0.50 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.2500 | 0.4414 | 0.1914 | {'echo': 2} |
| deepseek7b | success | short_answer_instruction | 3 | down_out | 1.00 | prose | 6 | 1.0000 | 1.0000 | 0.0000 | -0.2500 | -0.0781 | 0.1719 | {'be_continuation': 4, 'prose': 2} |
| deepseek7b | drift | no_instruction | 1 | gate_up_pair | 0.50 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.2500 | 0.4766 | 0.1641 | {'echo': 2} |
| deepseek7b | drift | short_answer_instruction | 1 | gate_up_pair | 0.50 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.2500 | 0.3359 | 0.1484 | {'echo': 2} |
| deepseek7b | drift | short_answer_instruction | 2 | product | 0.50 | echo | 2 | 1.0000 | 1.0000 | 0.0000 | -0.2500 | 0.0781 | 0.1406 | {'newline_boundary': 2} |
| deepseek7b | drift | short_answer_instruction | 2 | down_out | 0.50 | echo | 2 | 1.0000 | 1.0000 | 0.0000 | -0.2500 | 0.0469 | 0.1094 | {'newline_boundary': 2} |
| glm4 | success | explain_instruction | 2 | down_out | 1.00 | because_reason | 10 | 1.0000 | 1.0000 | 0.0000 | -0.2437 | 1.8500 | 1.9688 | {'newline_boundary': 10} |
| glm4 | drift | no_answer_anchor | 1 | gate_up_pair | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.2359 | 0.5969 | 1.5281 | {'for_continuation': 10} |
| glm4 | success | explain_instruction | 2 | product | 1.00 | because_reason | 10 | 1.0000 | 1.0000 | 0.0000 | -0.2250 | 1.8500 | 1.9812 | {'newline_boundary': 10} |
| glm4 | success | explain_instruction | 2 | gate_up_pair | 1.00 | because_reason | 10 | 1.0000 | 1.0000 | 0.0000 | -0.2188 | 1.8875 | 1.9937 | {'newline_boundary': 10} |
| glm4 | drift | explain_instruction | 1 | gate_up_pair | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.2094 | 0.1868 | 0.3243 | {'echo': 5, 'prose': 1, 'the_continuation': 4} |
| deepseek7b | drift | short_answer_instruction | 1 | down_out | 1.00 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1875 | 0.7930 | 0.4180 | {'echo': 2} |
| deepseek7b | drift | short_answer_instruction | 1 | gate_up_pair | 1.00 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1875 | 0.8086 | 0.3711 | {'echo': 2} |
| deepseek7b | drift | short_answer_instruction | 1 | product | 1.00 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1875 | 0.7852 | 0.3477 | {'echo': 2} |
| deepseek7b | drift | no_instruction | 1 | product | 1.00 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1875 | 1.0000 | 0.3125 | {'echo': 2} |
| deepseek7b | drift | no_instruction | 1 | down_out | 1.00 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1875 | 0.9766 | 0.2891 | {'echo': 2} |
| deepseek7b | drift | short_answer_instruction | 1 | product | 0.50 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1875 | 0.4336 | 0.1836 | {'echo': 2} |
| deepseek7b | drift | because_removed | 3 | gate_up_pair | 1.00 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1875 | 0.0000 | 0.1250 | {'be_continuation': 2} |
| deepseek7b | drift | because_removed | 3 | down_out | 1.00 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1875 | 0.0000 | 0.1250 | {'be_continuation': 2} |
| deepseek7b | drift | because_removed | 2 | gate_up_pair | 1.00 | be_continuation | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1875 | -0.0469 | 0.0156 | {'newline_boundary': 2} |
| deepseek7b | drift | short_answer_instruction | 2 | down_out | 1.00 | be_continuation | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1719 | 0.0781 | 0.2031 | {'newline_boundary': 2} |
| glm4 | success | no_answer_anchor | 1 | down_out | 0.50 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1688 | 1.8326 | 2.3201 | {'for_continuation': 10} |
| glm4 | drift | explain_instruction | 1 | gate_up_pair | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1603 | 0.1868 | 0.3243 | {'echo': 5, 'prose': 1, 'the_continuation': 4} |
| glm4 | success | no_instruction | 3 | down_out | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1594 | 0.2313 | 0.3812 | {'because_reason': 2, 'prose': 8} |
| glm4 | success | no_instruction | 3 | product | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1562 | 0.2062 | 0.3688 | {'because_reason': 2, 'prose': 8} |
| glm4 | success | no_instruction | 1 | product | 0.50 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1542 | 0.1344 | 0.2094 | {'the_continuation': 10} |
| glm4 | success | explain_instruction | 3 | product | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1500 | 0.2812 | 0.3875 | {'because_reason': 10} |
| glm4 | success | explain_instruction | 3 | down_out | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1500 | 0.2812 | 0.3750 | {'because_reason': 10} |
| glm4 | success | no_instruction | 1 | down_out | 1.00 | because_reason | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1437 | 0.2313 | 0.4062 | {'the_continuation': 10} |
| glm4 | success | no_instruction | 1 | product | 1.00 | because_reason | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1406 | 0.2219 | 0.4156 | {'the_continuation': 10} |
| glm4 | success | no_instruction | 3 | gate_up_pair | 1.00 | for_continuation | 10 | 0.9000 | 1.0000 | 0.0000 | -0.1406 | 0.2125 | 0.3937 | {'because_reason': 2, 'prose': 8} |
| deepseek7b | drift | short_answer_instruction | 2 | gate_up_pair | 1.00 | be_continuation | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1406 | 0.0938 | 0.2188 | {'newline_boundary': 2} |
| deepseek7b | drift | because_removed | 2 | down_out | 0.50 | be_continuation | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1406 | -0.0391 | 0.0859 | {'newline_boundary': 2} |
| glm4 | success | no_instruction | 1 | down_out | 0.50 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1396 | 0.1219 | 0.1906 | {'the_continuation': 10} |
| glm4 | success | no_answer_anchor | 1 | product | 0.50 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1391 | 1.8342 | 2.3404 | {'for_continuation': 10} |
| glm4 | success | no_instruction | 1 | gate_up_pair | 0.50 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1380 | 0.1719 | 0.2531 | {'the_continuation': 10} |
| glm4 | success | explain_instruction | 2 | product | 1.00 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1313 | 1.8500 | 1.9812 | {'newline_boundary': 10} |
| glm4 | success | no_instruction | 1 | gate_up_pair | 1.00 | because_reason | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1281 | 0.2188 | 0.4188 | {'the_continuation': 10} |
| glm4 | success | no_instruction | 1 | gate_up_pair | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1250 | 0.2188 | 0.4188 | {'the_continuation': 10} |
| glm4 | success | no_instruction | 1 | product | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1250 | 0.2219 | 0.4156 | {'the_continuation': 10} |
| deepseek7b | drift | because_removed | 3 | product | 1.00 | be_continuation | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1250 | 0.0625 | 0.1875 | {'be_continuation': 2} |
| deepseek7b | drift | because_removed | 3 | product | 1.00 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1250 | 0.0625 | 0.1875 | {'be_continuation': 2} |
| deepseek7b | drift | because_removed | 3 | gate_up_pair | 1.00 | be_continuation | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1250 | 0.0000 | 0.1250 | {'be_continuation': 2} |
| deepseek7b | drift | because_removed | 3 | down_out | 1.00 | be_continuation | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1250 | 0.0000 | 0.1250 | {'be_continuation': 2} |
| deepseek7b | drift | no_instruction | 1 | down_out | 0.50 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1250 | 0.4766 | 0.0391 | {'echo': 2} |
| deepseek7b | drift | no_instruction | 1 | product | 0.50 | prose | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1250 | 0.4688 | 0.0312 | {'echo': 2} |
| glm4 | success | explain_instruction | 2 | down_out | 1.00 | newline_boundary | 10 | 0.8000 | 1.0000 | 0.0000 | -0.1187 | 1.8500 | 1.9688 | {'newline_boundary': 10} |
| glm4 | success | explain_instruction | 2 | product | 0.50 | because_reason | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1187 | 0.9656 | 1.0219 | {'newline_boundary': 10} |
| glm4 | drift | no_answer_anchor | 1 | down_out | 0.50 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1172 | 0.3269 | 0.7769 | {'for_continuation': 10} |
| glm4 | drift | no_answer_anchor | 1 | product | 0.50 | newline_boundary | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1172 | 0.3224 | 0.7537 | {'for_continuation': 10} |
| glm4 | success | no_instruction | 1 | down_out | 1.00 | for_continuation | 10 | 1.0000 | 1.0000 | 0.0000 | -0.1156 | 0.2313 | 0.4062 | {'the_continuation': 10} |
| deepseek7b | drift | because_removed | 1 | down_out | 1.00 | be_continuation | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1094 | 0.5000 | 0.5625 | {'the_continuation': 2} |
| deepseek7b | drift | short_answer_instruction | 2 | product | 1.00 | be_continuation | 2 | 1.0000 | 1.0000 | 0.0000 | -0.1094 | 0.0859 | 0.2109 | {'newline_boundary': 2} |
| glm4 | success | explain_instruction | 2 | gate_up_pair | 1.00 | newline_boundary | 10 | 0.8000 | 1.0000 | 0.0000 | -0.1062 | 1.8875 | 1.9937 | {'newline_boundary': 10} |
