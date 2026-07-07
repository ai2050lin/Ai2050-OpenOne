# Phase 210 minimal pattern transition atlas

Total prompts: 440
State rows: 36848
Contrast rows: 1008
Pattern match: 119
Pattern drift: 321

| model | pattern | rows | match | drift | answer | eos | output patterns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | answer_explain | 40 | 20 | 20 | 20 | 0 | {'explain_answer': 20, 'repeat_answer': 8, 'list_answer': 8, 'echo_then_answer': 4} |
| qwen3 | answer_list | 40 | 14 | 26 | 34 | 0 | {'repeat_answer': 18, 'list_answer': 14, 'other_or_wrong': 6, 'short_answer': 2} |
| qwen3 | answer_repeat | 40 | 28 | 12 | 28 | 0 | {'repeat_answer': 28, 'list_answer': 8, 'other_or_wrong': 4} |
| qwen3 | answer_short | 40 | 0 | 40 | 26 | 0 | {'other_or_wrong': 26, 'next_task_or_format': 8, 'repeat_answer': 6} |
| qwen3 | answer_target_seeded | 40 | 0 | 40 | 38 | 0 | {'echo_then_answer': 24, 'repeat_answer': 10, 'other_or_wrong': 6} |
| glm4 | answer_explain | 40 | 7 | 33 | 15 | 0 | {'other_or_wrong': 11, 'short_answer': 8, 'next_task_or_format': 8, 'explain_answer': 7, 'repeat_answer': 4, 'list_answer': 2} |
| glm4 | answer_list | 40 | 1 | 39 | 21 | 0 | {'other_or_wrong': 15, 'repeat_answer': 13, 'short_answer': 5, 'next_task_or_format': 4, 'echo_then_answer': 2, 'list_answer': 1} |
| glm4 | answer_repeat | 40 | 20 | 20 | 25 | 0 | {'repeat_answer': 20, 'list_answer': 10, 'echo_then_answer': 5, 'next_task_or_format': 3, 'other_or_wrong': 2} |
| glm4 | answer_short | 40 | 0 | 40 | 6 | 0 | {'other_or_wrong': 32, 'explain_answer': 7, 'echo_then_answer': 1} |
| glm4 | answer_target_seeded | 40 | 17 | 23 | 40 | 0 | {'short_answer': 17, 'repeat_answer': 14, 'echo_then_answer': 8, 'next_task_or_format': 1} |
| deepseek7b | answer_explain | 8 | 6 | 2 | 2 | 0 | {'explain_answer': 6, 'other_or_wrong': 2} |
| deepseek7b | answer_list | 8 | 6 | 2 | 0 | 0 | {'list_answer': 6, 'other_or_wrong': 2} |
| deepseek7b | answer_repeat | 8 | 0 | 8 | 0 | 0 | {'list_answer': 8} |
| deepseek7b | answer_short | 8 | 0 | 8 | 4 | 0 | {'explain_answer': 6, 'repeat_answer': 2} |
| deepseek7b | answer_target_seeded | 8 | 0 | 8 | 8 | 6 | {'repeat_answer': 6, 'echo_then_answer': 2} |

| model | pattern vs short | layer | rows | mean l2 diff | mean cosine | norm delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | answer_target_seeded | 26 | 12 | 708.7709 | 0.7892 | -37.4039 |
| deepseek7b | answer_explain | 26 | 12 | 680.8006 | 0.8052 | 14.4602 |
| deepseek7b | answer_list | 26 | 12 | 675.6195 | 0.8063 | 17.7638 |
| deepseek7b | answer_repeat | 26 | 12 | 655.2843 | 0.8182 | -20.7846 |
| deepseek7b | answer_target_seeded | 24 | 12 | 539.1616 | 0.6770 | -16.5554 |
| deepseek7b | answer_explain | 24 | 12 | 522.3514 | 0.6870 | 18.5740 |
| deepseek7b | answer_list | 24 | 12 | 509.0346 | 0.7032 | 28.8797 |
| deepseek7b | answer_repeat | 24 | 12 | 495.1650 | 0.7166 | -15.3919 |
| deepseek7b | answer_target_seeded | 20 | 12 | 291.9694 | 0.5508 | 8.2138 |
| deepseek7b | answer_explain | 20 | 12 | 269.8932 | 0.6116 | 17.3763 |
| deepseek7b | answer_list | 20 | 12 | 263.8280 | 0.6198 | 8.2989 |
| deepseek7b | answer_repeat | 20 | 12 | 260.2379 | 0.6285 | 3.2511 |
| qwen3 | answer_repeat | 34 | 12 | 260.2043 | 0.8918 | 4.3033 |
| qwen3 | answer_list | 34 | 12 | 248.0878 | 0.8962 | -15.0934 |
| qwen3 | answer_explain | 34 | 12 | 229.8908 | 0.9119 | -1.5453 |
| qwen3 | answer_target_seeded | 34 | 12 | 225.7228 | 0.9221 | 35.3633 |
| qwen3 | answer_repeat | 32 | 12 | 201.4473 | 0.8180 | 1.4032 |
| qwen3 | answer_list | 32 | 12 | 191.1067 | 0.8243 | -5.9250 |
| deepseek7b | answer_target_seeded | 16 | 12 | 181.1718 | 0.5020 | -1.1782 |
| deepseek7b | answer_list | 16 | 12 | 174.8167 | 0.5304 | 9.4103 |
| deepseek7b | answer_explain | 16 | 12 | 173.6381 | 0.5353 | 7.6480 |
| qwen3 | answer_target_seeded | 32 | 12 | 173.1637 | 0.8664 | 22.4169 |
| qwen3 | answer_explain | 32 | 12 | 172.2792 | 0.8594 | 12.3001 |
| deepseek7b | answer_repeat | 16 | 12 | 169.7990 | 0.5552 | 4.6182 |
| glm4 | answer_list | 38 | 12 | 143.0399 | 0.8796 | -30.3744 |
| glm4 | answer_repeat | 38 | 12 | 134.8747 | 0.9001 | 0.5828 |
| glm4 | answer_target_seeded | 38 | 12 | 130.9192 | 0.8997 | -19.2615 |
| deepseek7b | answer_list | 12 | 12 | 127.8675 | 0.4555 | 3.8917 |
| deepseek7b | answer_explain | 12 | 12 | 126.9871 | 0.4650 | 3.1568 |
| deepseek7b | answer_target_seeded | 12 | 12 | 126.6080 | 0.4695 | -9.6410 |
| deepseek7b | answer_repeat | 12 | 12 | 124.2646 | 0.4920 | 1.3325 |
| glm4 | answer_explain | 38 | 12 | 121.6812 | 0.9150 | -9.4925 |
| deepseek7b | answer_target_seeded | 8 | 12 | 100.2232 | 0.4696 | 3.1576 |
| deepseek7b | answer_list | 8 | 12 | 99.5779 | 0.4351 | 5.3566 |
| deepseek7b | answer_explain | 8 | 12 | 99.2155 | 0.4449 | 6.3477 |
| qwen3 | answer_repeat | 26 | 12 | 97.9385 | 0.7186 | 2.8731 |
