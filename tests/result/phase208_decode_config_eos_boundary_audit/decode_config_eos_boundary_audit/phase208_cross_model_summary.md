# Phase 208 decode config EOS boundary audit

Total EOS positives: 1

| model | mode | protocol | temp | rows | eos | avg steps | first tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| qwen3 | generate_beam | chat_eos | None | 19 | 0 | 32.00 | {'<think>': 19} |
| qwen3 | generate_beam | eos_instruction | None | 19 | 0 | 32.00 | {' The': 15, ' red': 4} |
| qwen3 | generate_beam | final_answer | None | 19 | 0 | 32.00 | {' brown': 12, ' red': 7} |
| qwen3 | generate_beam | short_answer | None | 20 | 0 | 32.00 | {' red': 8, ' black': 8, ' brown': 4} |
| qwen3 | generate_beam | stop_explicit | None | 19 | 0 | 32.00 | {' black': 8, ' red': 7, ' brown': 4} |
| qwen3 | generate_greedy | chat_eos | None | 19 | 0 | 32.00 | {'<think>': 19} |
| qwen3 | generate_greedy | eos_instruction | None | 19 | 0 | 32.00 | {' The': 15, ' red': 4} |
| qwen3 | generate_greedy | final_answer | None | 19 | 0 | 32.00 | {' brown': 12, ' red': 7} |
| qwen3 | generate_greedy | short_answer | None | 20 | 0 | 32.00 | {' red': 8, ' black': 8, ' brown': 4} |
| qwen3 | generate_greedy | stop_explicit | None | 19 | 0 | 32.00 | {' brown': 8, ' red': 7, ' black': 4} |
| qwen3 | generate_sample | chat_eos | 0.7 | 57 | 0 | 32.00 | {'<think>': 57} |
| qwen3 | generate_sample | chat_eos | 1.0 | 57 | 0 | 32.00 | {'<think>': 57} |
| qwen3 | generate_sample | eos_instruction | 0.7 | 57 | 0 | 32.00 | {' The': 34, ' red': 11, ' If': 4, ' You': 2, ' \n\n': 2, ' Please': 2, ' black': 1, ' brown': 1} |
| qwen3 | generate_sample | eos_instruction | 1.0 | 57 | 0 | 32.00 | {' The': 21, ' red': 10, ' If': 7, ' You': 4, ' brown': 3, ' Please': 3, ' Black': 2, ' black': 1} |
| qwen3 | generate_sample | final_answer | 0.7 | 57 | 0 | 32.00 | {' brown': 35, ' red': 21, ' black': 1} |
| qwen3 | generate_sample | final_answer | 1.0 | 57 | 0 | 32.00 | {' brown': 35, ' red': 21, ' black': 1} |
| qwen3 | generate_sample | short_answer | 0.7 | 60 | 0 | 32.00 | {' red': 20, ' black': 12, ' brown': 10, '...': 5, ' a': 3, ' usually': 3, ' determined': 3, ' typically': 1} |
| qwen3 | generate_sample | short_answer | 1.0 | 60 | 0 | 32.00 | {' red': 16, ' black': 9, ' brown': 7, ' typically': 5, ' a': 3, ' the': 3, '...': 3, ' yellow': 2} |
| qwen3 | generate_sample | stop_explicit | 0.7 | 57 | 0 | 32.00 | {' red': 20, ' brown': 20, ' black': 10, ' green': 2, ' usually': 2, ' blue': 1, ' dark': 1, ' yellow': 1} |
| qwen3 | generate_sample | stop_explicit | 1.0 | 57 | 0 | 31.98 | {' red': 19, ' brown': 19, ' black': 9, ' green': 3, ' usually': 2, ' a': 1, '...': 1, ' dark': 1} |
| qwen3 | manual_greedy | chat_eos | None | 19 | 0 | 32.00 | {'<think>': 19} |
| qwen3 | manual_greedy | eos_instruction | None | 19 | 0 | 32.00 | {' The': 15, ' red': 4} |
| qwen3 | manual_greedy | final_answer | None | 19 | 0 | 32.00 | {' brown': 12, ' red': 7} |
| qwen3 | manual_greedy | short_answer | None | 20 | 0 | 32.00 | {' red': 8, ' black': 8, ' brown': 4} |
| qwen3 | manual_greedy | stop_explicit | None | 19 | 0 | 32.00 | {' brown': 8, ' red': 7, ' black': 4} |
| glm4 | generate_beam | chat_eos | None | 19 | 0 | 32.00 | {'\n': 19} |
| glm4 | generate_beam | eos_instruction | None | 19 | 0 | 32.00 | {' Do': 19} |
| glm4 | generate_beam | final_answer | None | 19 | 0 | 32.00 | {' red': 12, ' brown': 6, ' ': 1} |
| glm4 | generate_beam | short_answer | None | 20 | 0 | 32.00 | {' ______': 10, ' __': 4, ':\n': 2, ' brown': 2, ' ride': 1, ' be': 1} |
| glm4 | generate_beam | stop_explicit | None | 19 | 0 | 32.00 | {' ______': 18, ' ride': 1} |
| glm4 | generate_greedy | chat_eos | None | 19 | 0 | 32.00 | {'\n': 19} |
| glm4 | generate_greedy | eos_instruction | None | 19 | 0 | 32.00 | {' Do': 19} |
| glm4 | generate_greedy | final_answer | None | 19 | 0 | 32.00 | {' red': 12, ' brown': 6, ' ': 1} |
| glm4 | generate_greedy | short_answer | None | 20 | 0 | 32.00 | {' ______': 10, ':\n': 6, ' called': 2, ' ride': 1, ' be': 1} |
| glm4 | generate_greedy | stop_explicit | None | 19 | 0 | 32.00 | {':\n': 8, ' ______': 6, ' __': 4, ' ride': 1} |
| glm4 | generate_sample | chat_eos | 0.7 | 57 | 0 | 32.00 | {'\n': 57} |
| glm4 | generate_sample | chat_eos | 1.0 | 57 | 0 | 32.00 | {'\n': 57} |
| glm4 | generate_sample | eos_instruction | 0.7 | 57 | 0 | 32.00 | {' Do': 37, ' No': 14, ' If': 1, ' There': 1, ' Thank': 1, ' The': 1, ' You': 1, ' Don': 1} |
| glm4 | generate_sample | eos_instruction | 1.0 | 57 | 0 | 32.00 | {' Do': 28, ' No': 12, ' If': 4, ' There': 3, ' You': 2, ' The': 2, ' Your': 1, ' Thank': 1} |
| glm4 | generate_sample | final_answer | 0.7 | 57 | 0 | 32.00 | {' red': 33, ' brown': 14, ' white': 2, ' eat': 2, ' black': 2, ' green': 2, ' blue': 1, ' "': 1} |
| glm4 | generate_sample | final_answer | 1.0 | 57 | 0 | 32.00 | {' red': 29, ' brown': 12, ' "': 3, ' eat': 2, ' black': 2, ' green': 2, ' RED': 1, ' bay': 1} |
| glm4 | generate_sample | short_answer | 0.7 | 60 | 0 | 32.00 | {':\n': 11, ' ______': 9, ' __': 8, ':': 4, ' red': 3, ' __________________': 3, ' called': 3, ' brown': 2} |
| glm4 | generate_sample | short_answer | 1.0 | 60 | 0 | 31.48 | {' ______': 6, ':\n': 6, ' __': 5, ':': 4, ' __________________': 4, '...\n': 3, ' called': 3, ' ...\n': 2} |
| glm4 | generate_sample | stop_explicit | 0.7 | 57 | 0 | 32.00 | {':\n': 14, ' ______': 10, ' __': 9, ':': 3, ' green': 2, '?\n': 2, ' called': 2, ' (': 2} |
| glm4 | generate_sample | stop_explicit | 1.0 | 57 | 0 | 31.70 | {':\n': 13, ' __': 11, ' ______': 4, ':': 3, ' [': 3, ' green': 2, ' called': 2, ' (': 2} |
| glm4 | manual_greedy | chat_eos | None | 19 | 0 | 32.00 | {'\n': 19} |
| glm4 | manual_greedy | eos_instruction | None | 19 | 0 | 32.00 | {' Do': 19} |
| glm4 | manual_greedy | final_answer | None | 19 | 0 | 32.00 | {' red': 12, ' brown': 6, ' ': 1} |
| glm4 | manual_greedy | short_answer | None | 20 | 0 | 32.00 | {' ______': 10, ':\n': 6, ' called': 2, ' ride': 1, ' be': 1} |
| glm4 | manual_greedy | stop_explicit | None | 19 | 0 | 32.00 | {':\n': 8, ' ______': 6, ' __': 4, ' ride': 1} |
| deepseek7b | generate_beam | answer_boundary | None | 1 | 0 | 32.00 | {' A': 1} |
| deepseek7b | generate_beam | chat_eos | None | 6 | 0 | 32.00 | {'Okay': 6} |
| deepseek7b | generate_beam | empty_completion | None | 1 | 0 | 32.00 | {' sucht': 1} |
| deepseek7b | generate_beam | end_now | None | 1 | 0 | 12.00 | {' \n\n': 1} |
| deepseek7b | generate_beam | eos_instruction | None | 6 | 0 | 23.83 | {'**\n\n': 5, '**\n': 1} |
| deepseek7b | generate_beam | final_answer | None | 6 | 0 | 26.33 | {' What': 3, ' ?\n\n': 2, ' The': 1} |
| deepseek7b | generate_beam | final_marker | None | 1 | 0 | 32.00 | {' \\': 1} |
| deepseek7b | generate_beam | short_answer | None | 6 | 0 | 32.00 | {' \\': 2, ' hold': 2, ' carry': 1, ' fetch': 1} |
| deepseek7b | generate_beam | single_word_done | None | 1 | 0 | 32.00 | {'\n': 1} |
| deepseek7b | generate_beam | space_completion | None | 1 | 0 | 32.00 | {'2': 1} |
| deepseek7b | generate_beam | stop_explicit | None | 6 | 0 | 31.83 | {' hold': 4, ' pull': 1, ' fetch': 1} |
| deepseek7b | generate_greedy | answer_boundary | None | 1 | 0 | 32.00 | {' ': 1} |
| deepseek7b | generate_greedy | chat_eos | None | 6 | 0 | 32.00 | {'Okay': 6} |
| deepseek7b | generate_greedy | empty_completion | None | 1 | 0 | 32.00 | {' sucht': 1} |
| deepseek7b | generate_greedy | end_now | None | 1 | 0 | 12.00 | {' \n\n': 1} |
| deepseek7b | generate_greedy | eos_instruction | None | 6 | 0 | 32.00 | {'**\n\n': 6} |
| deepseek7b | generate_greedy | final_answer | None | 6 | 0 | 32.00 | {' ?\n\n': 4, ' ': 2} |
| deepseek7b | generate_greedy | final_marker | None | 1 | 0 | 32.00 | {' \\': 1} |
| deepseek7b | generate_greedy | short_answer | None | 6 | 0 | 32.00 | {' hold': 4, ' ride': 1, ' fetch': 1} |
| deepseek7b | generate_greedy | single_word_done | None | 1 | 0 | 32.00 | {'\n': 1} |
| deepseek7b | generate_greedy | space_completion | None | 1 | 0 | 32.00 | {' ': 1} |
| deepseek7b | generate_greedy | stop_explicit | None | 6 | 0 | 32.00 | {' hold': 4, ' carry': 1, ' fetch': 1} |
| deepseek7b | generate_sample | answer_boundary | 0.7 | 3 | 0 | 32.00 | {'\n': 1, ' ': 1, ' A': 1} |
| deepseek7b | generate_sample | answer_boundary | 1.0 | 3 | 0 | 32.00 | {' D': 1, ' ': 1, ' A': 1} |
| deepseek7b | generate_sample | chat_eos | 0.7 | 18 | 0 | 32.00 | {'Okay': 15, 'Alright': 3} |
| deepseek7b | generate_sample | chat_eos | 1.0 | 18 | 0 | 32.00 | {'Okay': 13, 'Alright': 5} |
| deepseek7b | generate_sample | empty_completion | 0.7 | 3 | 0 | 32.00 | {' sucht': 3} |
| deepseek7b | generate_sample | empty_completion | 1.0 | 3 | 0 | 32.00 | {' sucht': 1, ' FIFO': 1, ' implode': 1} |
| deepseek7b | generate_sample | end_now | 0.7 | 3 | 0 | 25.33 | {' \n\n': 2, ' Wait': 1} |
| deepseek7b | generate_sample | end_now | 1.0 | 3 | 0 | 25.33 | {' Wait': 1, '**\n\n': 1, ' \n\n': 1} |
| deepseek7b | generate_sample | eos_instruction | 0.7 | 18 | 0 | 28.50 | {'**\n\n': 11, ' \n\n': 3, ' To': 1, ' Do': 1, ' Use': 1, '**\n': 1} |
| deepseek7b | generate_sample | eos_instruction | 1.0 | 18 | 0 | 28.83 | {'**\n\n': 7, ' Use': 2, ' \n\n': 2, ' To': 1, ' Please': 1, ' Do': 1, '  \n': 1, ' The': 1} |
| deepseek7b | generate_sample | final_answer | 0.7 | 18 | 0 | 30.00 | {' ?\n\n': 5, ' ': 4, ' The': 3, ' ?\n': 3, ' \\': 1, ' What': 1, ' what': 1} |
| deepseek7b | generate_sample | final_answer | 1.0 | 18 | 0 | 30.22 | {' ?\n\n': 5, ' The': 3, ' ': 3, ' ?\n': 3, ' \\': 1, ' What': 1, ' ...\n': 1, ' \n': 1} |
| deepseek7b | generate_sample | final_marker | 0.7 | 3 | 0 | 32.00 | {' \\': 3} |
| deepseek7b | generate_sample | final_marker | 1.0 | 3 | 0 | 32.00 | {' \\': 3} |
| deepseek7b | generate_sample | short_answer | 0.7 | 18 | 0 | 31.61 | {' hold': 6, ' contain': 2, ' ride': 2, ' fetch': 2, ' \\': 2, ' serve': 1, ' store': 1, ' carry': 1} |
| deepseek7b | generate_sample | short_answer | 1.0 | 18 | 0 | 31.22 | {' hold': 4, ' contain': 2, ' serve': 2, ' ride': 2, ' \\': 2, ' measure': 1, ' get': 1, ' store': 1} |
| deepseek7b | generate_sample | single_word_done | 0.7 | 3 | 0 | 32.00 | {'\n': 2, '.\n\n': 1} |
| deepseek7b | generate_sample | single_word_done | 1.0 | 3 | 0 | 32.00 | {'\n': 2, '.\n\n': 1} |
| deepseek7b | generate_sample | space_completion | 0.7 | 3 | 0 | 32.00 | {'4': 2, '2': 1} |
| deepseek7b | generate_sample | space_completion | 1.0 | 3 | 0 | 32.00 | {'4': 2, '5': 1} |
| deepseek7b | generate_sample | stop_explicit | 0.7 | 18 | 0 | 31.56 | {' hold': 9, ' serve': 2, ' carry': 2, ' contain': 1, ' ride': 1, ' fetch': 1, ' pull': 1, ' {': 1} |
| deepseek7b | generate_sample | stop_explicit | 1.0 | 18 | 0 | 31.39 | {' hold': 4, ' contain': 2, ' carry': 2, ' serve': 1, ' ride': 1, ' __': 1, ' make': 1, ' pour': 1} |
| deepseek7b | manual_greedy | answer_boundary | None | 1 | 0 | 32.00 | {' ': 1} |
| deepseek7b | manual_greedy | chat_eos | None | 6 | 0 | 32.00 | {'Okay': 6} |
| deepseek7b | manual_greedy | empty_completion | None | 1 | 0 | 32.00 | {' sucht': 1} |
| deepseek7b | manual_greedy | end_now | None | 1 | 1 | 13.00 | {' \n\n': 1} |
| deepseek7b | manual_greedy | eos_instruction | None | 6 | 0 | 32.00 | {'**\n\n': 6} |
| deepseek7b | manual_greedy | final_answer | None | 6 | 0 | 32.00 | {' ?\n\n': 4, ' ': 2} |
| deepseek7b | manual_greedy | final_marker | None | 1 | 0 | 32.00 | {' \\': 1} |
| deepseek7b | manual_greedy | short_answer | None | 6 | 0 | 32.00 | {' hold': 4, ' ride': 1, ' fetch': 1} |
| deepseek7b | manual_greedy | single_word_done | None | 1 | 0 | 32.00 | {'\n': 1} |
| deepseek7b | manual_greedy | space_completion | None | 1 | 0 | 32.00 | {' ': 1} |
| deepseek7b | manual_greedy | stop_explicit | None | 6 | 0 | 32.00 | {' hold': 4, ' carry': 1, ' fetch': 1} |
