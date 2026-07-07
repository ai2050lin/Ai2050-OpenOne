# Phase 209 pattern running contrast atlas

Total prompts: 833
Pattern match: 140
Pattern drift: 693
Answer present: 415
Ended with EOS: 8

| model | pattern | rows | match | drift | answer | eos | output patterns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | answer_echo_control | 60 | 0 | 60 | 36 | 0 | {'explain_answer': 36, 'list_answer': 24} |
| qwen3 | answer_explain | 60 | 27 | 33 | 26 | 0 | {'explain_answer': 27, 'list_answer': 21, 'repeat_answer': 12} |
| qwen3 | answer_list | 60 | 24 | 36 | 42 | 0 | {'list_answer': 24, 'repeat_answer': 20, 'explain_answer': 8, 'next_task_or_format': 4, 'other_or_wrong': 4} |
| qwen3 | answer_repeat | 60 | 28 | 32 | 30 | 0 | {'list_answer': 32, 'repeat_answer': 28} |
| qwen3 | answer_short | 60 | 0 | 60 | 29 | 0 | {'explain_answer': 50, 'repeat_answer': 6, 'next_task_or_format': 2, 'echo_then_answer': 2} |
| qwen3 | answer_stop | 60 | 0 | 60 | 28 | 0 | {'explain_answer': 43, 'next_task_or_format': 7, 'repeat_answer': 6, 'other_or_wrong': 2, 'list_answer': 2} |
| qwen3 | answer_target_seeded | 60 | 0 | 60 | 57 | 0 | {'repeat_answer': 34, 'explain_answer': 16, 'list_answer': 10} |
| glm4 | answer_echo_control | 49 | 0 | 49 | 34 | 0 | {'explain_answer': 26, 'repeat_answer': 19, 'list_answer': 3, 'next_task_or_format': 1} |
| glm4 | answer_explain | 49 | 24 | 25 | 18 | 0 | {'explain_answer': 24, 'repeat_answer': 16, 'list_answer': 9} |
| glm4 | answer_list | 49 | 4 | 45 | 16 | 0 | {'other_or_wrong': 23, 'repeat_answer': 16, 'explain_answer': 6, 'list_answer': 4} |
| glm4 | answer_repeat | 49 | 10 | 39 | 18 | 0 | {'other_or_wrong': 17, 'repeat_answer': 10, 'explain_answer': 8, 'next_task_or_format': 8, 'short_answer': 4, 'list_answer': 2} |
| glm4 | answer_short | 49 | 0 | 49 | 6 | 0 | {'other_or_wrong': 34, 'explain_answer': 11, 'next_task_or_format': 2, 'echo_then_answer': 2} |
| glm4 | answer_stop | 49 | 0 | 49 | 11 | 0 | {'explain_answer': 43, 'repeat_answer': 6} |
| glm4 | answer_target_seeded | 49 | 4 | 45 | 42 | 0 | {'other_or_wrong': 25, 'next_task_or_format': 11, 'repeat_answer': 7, 'short_answer': 4, 'echo_then_answer': 2} |
| deepseek7b | answer_echo_control | 10 | 0 | 10 | 2 | 0 | {'explain_answer': 4, 'list_answer': 4, 'repeat_answer': 2} |
| deepseek7b | answer_explain | 10 | 10 | 0 | 2 | 0 | {'explain_answer': 10} |
| deepseek7b | answer_list | 10 | 9 | 1 | 0 | 0 | {'list_answer': 9, 'explain_answer': 1} |
| deepseek7b | answer_repeat | 10 | 0 | 10 | 0 | 0 | {'list_answer': 10} |
| deepseek7b | answer_short | 10 | 0 | 10 | 4 | 0 | {'explain_answer': 5, 'list_answer': 3, 'repeat_answer': 2} |
| deepseek7b | answer_stop | 10 | 0 | 10 | 4 | 0 | {'explain_answer': 10} |
| deepseek7b | answer_target_seeded | 10 | 0 | 10 | 10 | 8 | {'repeat_answer': 8, 'echo_then_answer': 2} |
