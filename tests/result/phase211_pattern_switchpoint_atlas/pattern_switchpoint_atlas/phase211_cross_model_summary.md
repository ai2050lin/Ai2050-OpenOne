# Phase 211 pattern switchpoint atlas

Outcome rows: 50
State summary rows: 4186
Switchpoint rows: 2268

| model | pattern | drift group | best step | best layer | score | norm delta | prose delta | echo delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3 | answer_list | drift:other_or_wrong | 11 | 32 | 181.7317 | -103.9330 | -38.2857 | -16.4472 |
| glm4 | answer_list | drift:short_answer | 8 | 29 | 157.2038 | 17.6202 | 8.6188 | 2.2938 |
| glm4 | answer_list | drift:echo_then_answer | 8 | 35 | 153.6382 | 50.3385 | 5.5312 | 11.6250 |
| glm4 | answer_list | drift:next_task_or_format | 10 | 35 | 149.4235 | -31.0286 | -5.0938 | -3.7812 |
| glm4 | answer_list | drift:repeat_answer | 8 | 29 | 135.6103 | 11.4460 | 8.0096 | 2.0817 |
| deepseek7b | answer_explain | drift:other_or_wrong | 7 | 26 | 133.4597 | 132.9665 | -3.7188 | 10.9974 |
| qwen3 | answer_list | drift:short_answer | 9 | 32 | 126.8714 | -74.9570 | -38.2679 | -30.7946 |
| glm4 | answer_list | drift:other_or_wrong | 8 | 35 | 113.4356 | 30.2071 | 4.9750 | 6.2396 |
| deepseek7b | answer_list | drift:other_or_wrong | 7 | 24 | 111.8690 | 99.6835 | 0.6458 | 10.2083 |
| qwen3 | answer_explain | drift:repeat_answer | 2 | 16 | 93.2204 | 2.7625 | -17.8562 | -1.0938 |
| qwen3 | answer_repeat | drift:other_or_wrong | 12 | 10 | 87.6090 | -6.5679 | 10.3080 | 10.5446 |
| qwen3 | answer_list | drift:repeat_answer | 11 | 16 | 73.6787 | -12.2756 | -9.3343 | 9.2716 |
| qwen3 | answer_explain | drift:echo_then_answer | 12 | 32 | 64.3892 | -60.2693 | -13.2812 | -10.4500 |
| glm4 | answer_target_seeded | drift:repeat_answer | 2 | 6 | 57.2535 | 0.5872 | -8.8553 | -7.2742 |
| glm4 | answer_repeat | drift:other_or_wrong | 12 | 18 | 53.8780 | 2.1194 | -6.1750 | -8.0232 |
| glm4 | answer_explain | drift:list_answer | 7 | 18 | 53.0625 | -1.2017 | -6.4464 | -6.5335 |
| qwen3 | answer_repeat | drift:list_answer | 2 | 10 | 51.3953 | 1.9265 | 5.7277 | 0.7969 |
| qwen3 | answer_explain | drift:list_answer | 3 | 16 | 47.1016 | 5.9327 | -11.3969 | -9.0652 |
| glm4 | answer_explain | drift:next_task_or_format | 3 | 35 | 45.2692 | -12.8647 | -2.0179 | -3.9364 |
| glm4 | answer_explain | drift:repeat_answer | 5 | 38 | 44.1782 | 72.3201 | 7.0915 | 10.0022 |
| glm4 | answer_repeat | drift:echo_then_answer | 10 | 18 | 41.3465 | -0.7401 | 15.6500 | 9.7938 |
| glm4 | answer_explain | drift:short_answer | 12 | 6 | 32.1882 | 0.5037 | -7.3929 | -3.0625 |
| glm4 | answer_explain | drift:other_or_wrong | 3 | 35 | 31.7867 | -11.2476 | -2.2451 | -5.5513 |
| glm4 | answer_repeat | drift:next_task_or_format | 10 | 38 | 21.9288 | -45.4693 | 5.1000 | 7.2792 |
| glm4 | answer_repeat | drift:list_answer | 12 | 35 | 21.7201 | 12.5755 | 2.0875 | 4.6221 |
| glm4 | answer_target_seeded | drift:next_task_or_format | 10 | 12 | 11.5740 | 0.5976 | -3.4099 | -3.0882 |
| glm4 | answer_target_seeded | drift:echo_then_answer | 12 | 35 | 9.2855 | 10.2814 | 1.7555 | 5.6310 |
