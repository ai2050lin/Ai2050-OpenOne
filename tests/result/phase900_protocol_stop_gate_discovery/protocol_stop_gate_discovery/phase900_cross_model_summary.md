# Phase 900 protocol stop gate discovery

## Overall

- models: qwen3, glm4, deepseek7b
- control_rows: 1606
- selected_answer_drift_rows: 68

## Control summaries

| model | control | type | head set | rows | answer | clean | drift | reduced | field | explanation | list | long | labels |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | head_zero_step1_2::L31H19+L31H26+L31H30+L31H12+L31H17 | head_zero_step1_2 | L31H19+L31H26+L31H30+L31H12+L31H17 | 18 | 18 | 0 | 18 | 3 | 2 | 8 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| deepseek7b | semantic_axis_repeated_to_step1_2 | source_repeat_step1_2 | none | 33 | 33 | 0 | 33 | 2 | 11 | 8 | 19 | 28 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | semantic_axis_zero_after_answer | source_after_zero_step1_2 | none | 33 | 33 | 0 | 33 | 2 | 11 | 8 | 19 | 27 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | semantic_axis_flip_after_answer | source_after_flip_step1_2 | none | 33 | 33 | 0 | 33 | 2 | 11 | 8 | 19 | 29 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1::L26H11 | head_zero_step1 | L26H11 | 33 | 33 | 0 | 33 | 2 | 11 | 9 | 20 | 25 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1_2::L26H11 | head_zero_step1_2 | L26H11 | 33 | 33 | 0 | 33 | 2 | 11 | 9 | 20 | 25 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1::L27H1 | head_zero_step1 | L27H1 | 33 | 33 | 0 | 33 | 2 | 10 | 8 | 20 | 25 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1_2::L27H1 | head_zero_step1_2 | L27H1 | 33 | 33 | 0 | 33 | 2 | 10 | 9 | 20 | 25 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1::L27H2 | head_zero_step1 | L27H2 | 33 | 33 | 0 | 33 | 2 | 10 | 8 | 20 | 25 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1_2::L27H2 | head_zero_step1_2 | L27H2 | 33 | 33 | 0 | 33 | 2 | 10 | 8 | 20 | 25 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | baseline | source_step0_only | none | 33 | 33 | 0 | 33 | 0 | 10 | 10 | 20 | 27 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | semantic_axis_repeated_to_step1 | source_repeat_step1 | none | 33 | 33 | 0 | 33 | 0 | 11 | 10 | 19 | 27 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1::L26H3 | head_zero_step1 | L26H3 | 33 | 33 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1_2::L26H3 | head_zero_step1_2 | L26H3 | 33 | 33 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1::L26H14 | head_zero_step1 | L26H14 | 33 | 33 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1_2::L26H14 | head_zero_step1_2 | L26H14 | 33 | 33 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1::L26H3+L26H7+L26H11+L26H14 | head_zero_step1 | L26H3+L26H7+L26H11+L26H14 | 33 | 33 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1_2::L26H3+L26H7+L26H11+L26H14 | head_zero_step1_2 | L26H3+L26H7+L26H11+L26H14 | 33 | 33 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'strict_canonical': 21} |
| deepseek7b | head_zero_step1::L26H7 | head_zero_step1 | L26H7 | 33 | 32 | 0 | 33 | 0 | 12 | 10 | 18 | 27 | {'answer_alias': 12, 'other': 1, 'strict_canonical': 20} |
| deepseek7b | head_zero_step1_2::L26H7 | head_zero_step1_2 | L26H7 | 33 | 32 | 0 | 33 | 0 | 12 | 10 | 18 | 27 | {'answer_alias': 12, 'other': 1, 'strict_canonical': 20} |
| deepseek7b | head_zero_step1::L27H0 | head_zero_step1 | L27H0 | 33 | 32 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'other': 1, 'strict_canonical': 20} |
| deepseek7b | head_zero_step1_2::L27H0 | head_zero_step1_2 | L27H0 | 33 | 32 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'other': 1, 'strict_canonical': 20} |
| deepseek7b | head_zero_step1::L27H3 | head_zero_step1 | L27H3 | 33 | 32 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'other': 1, 'strict_canonical': 20} |
| deepseek7b | head_zero_step1_2::L27H3 | head_zero_step1_2 | L27H3 | 33 | 32 | 0 | 33 | 0 | 11 | 11 | 20 | 27 | {'answer_alias': 12, 'other': 1, 'strict_canonical': 20} |
| deepseek7b | head_zero_step1::L27H0+L27H1+L27H2+L27H3 | head_zero_step1 | L27H0+L27H1+L27H2+L27H3 | 33 | 32 | 0 | 33 | 0 | 10 | 10 | 20 | 27 | {'answer_alias': 12, 'other': 1, 'strict_canonical': 20} |
| deepseek7b | head_zero_step1_2::L27H0+L27H1+L27H2+L27H3 | head_zero_step1_2 | L27H0+L27H1+L27H2+L27H3 | 33 | 32 | 0 | 33 | 0 | 10 | 10 | 20 | 27 | {'answer_alias': 12, 'other': 1, 'strict_canonical': 20} |
| qwen3 | baseline | source_step0_only | none | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | semantic_axis_repeated_to_step1 | source_repeat_step1 | none | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 10, 'strict_canonical': 8} |
| qwen3 | semantic_axis_repeated_to_step1_2 | source_repeat_step1_2 | none | 18 | 18 | 0 | 18 | 0 | 2 | 11 | 0 | 18 | {'answer_alias': 10, 'strict_canonical': 8} |
| qwen3 | semantic_axis_zero_after_answer | source_after_zero_step1_2 | none | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 10, 'strict_canonical': 8} |
| qwen3 | semantic_axis_flip_after_answer | source_after_flip_step1_2 | none | 18 | 18 | 0 | 18 | 0 | 2 | 11 | 0 | 18 | {'answer_alias': 10, 'strict_canonical': 8} |
| qwen3 | head_zero_step1::L31H19 | head_zero_step1 | L31H19 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1_2::L31H19 | head_zero_step1_2 | L31H19 | 18 | 18 | 0 | 18 | 0 | 2 | 10 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1::L31H26 | head_zero_step1 | L31H26 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1_2::L31H26 | head_zero_step1_2 | L31H26 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1::L31H30 | head_zero_step1 | L31H30 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1_2::L31H30 | head_zero_step1_2 | L31H30 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1::L31H12 | head_zero_step1 | L31H12 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1_2::L31H12 | head_zero_step1_2 | L31H12 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1::L31H17 | head_zero_step1 | L31H17 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1_2::L31H17 | head_zero_step1_2 | L31H17 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| qwen3 | head_zero_step1::L31H19+L31H26+L31H30+L31H12+L31H17 | head_zero_step1 | L31H19+L31H26+L31H30+L31H12+L31H17 | 18 | 18 | 0 | 18 | 0 | 1 | 11 | 0 | 18 | {'answer_alias': 8, 'strict_canonical': 10} |
| glm4 | baseline | source_step0_only | none | 17 | 17 | 0 | 17 | 0 | 17 | 0 | 0 | 17 | {'answer_alias': 5, 'strict_canonical': 12} |
| glm4 | semantic_axis_repeated_to_step1 | source_repeat_step1 | none | 17 | 17 | 0 | 17 | 0 | 17 | 0 | 0 | 17 | {'answer_alias': 5, 'strict_canonical': 12} |
| glm4 | semantic_axis_repeated_to_step1_2 | source_repeat_step1_2 | none | 17 | 17 | 0 | 17 | 0 | 17 | 0 | 0 | 17 | {'answer_alias': 5, 'strict_canonical': 12} |
| glm4 | semantic_axis_zero_after_answer | source_after_zero_step1_2 | none | 17 | 17 | 0 | 17 | 0 | 17 | 0 | 0 | 17 | {'answer_alias': 5, 'strict_canonical': 12} |
| glm4 | semantic_axis_flip_after_answer | source_after_flip_step1_2 | none | 17 | 17 | 0 | 17 | 0 | 17 | 0 | 0 | 17 | {'answer_alias': 5, 'strict_canonical': 12} |
| qwen3 | head_zero_step1::L31H0 | head_zero_step1 | L31H0 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| qwen3 | head_zero_step1_2::L31H0 | head_zero_step1_2 | L31H0 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| qwen3 | head_zero_step1::L31H1 | head_zero_step1 | L31H1 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| qwen3 | head_zero_step1_2::L31H1 | head_zero_step1_2 | L31H1 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| qwen3 | head_zero_step1::L31H2 | head_zero_step1 | L31H2 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| qwen3 | head_zero_step1_2::L31H2 | head_zero_step1_2 | L31H2 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| qwen3 | head_zero_step1::L31H3 | head_zero_step1 | L31H3 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| qwen3 | head_zero_step1_2::L31H3 | head_zero_step1_2 | L31H3 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| qwen3 | head_zero_step1::L31H0+L31H1+L31H2+L31H3 | head_zero_step1 | L31H0+L31H1+L31H2+L31H3 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| qwen3 | head_zero_step1_2::L31H0+L31H1+L31H2+L31H3 | head_zero_step1_2 | L31H0+L31H1+L31H2+L31H3 | 16 | 16 | 0 | 16 | 0 | 1 | 10 | 0 | 16 | {'answer_alias': 8, 'strict_canonical': 8} |
| glm4 | head_zero_step1::L35H0 | head_zero_step1 | L35H0 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1_2::L35H0 | head_zero_step1_2 | L35H0 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1::L35H1 | head_zero_step1 | L35H1 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1_2::L35H1 | head_zero_step1_2 | L35H1 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1::L35H2 | head_zero_step1 | L35H2 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1_2::L35H2 | head_zero_step1_2 | L35H2 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1::L35H3 | head_zero_step1 | L35H3 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1_2::L35H3 | head_zero_step1_2 | L35H3 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1::L35H0+L35H1+L35H2+L35H3 | head_zero_step1 | L35H0+L35H1+L35H2+L35H3 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1_2::L35H0+L35H1+L35H2+L35H3 | head_zero_step1_2 | L35H0+L35H1+L35H2+L35H3 | 11 | 11 | 0 | 11 | 0 | 11 | 0 | 0 | 11 | {'strict_canonical': 11} |
| glm4 | head_zero_step1::L39H0 | head_zero_step1 | L39H0 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| glm4 | head_zero_step1_2::L39H0 | head_zero_step1_2 | L39H0 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| glm4 | head_zero_step1::L39H1 | head_zero_step1 | L39H1 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| glm4 | head_zero_step1_2::L39H1 | head_zero_step1_2 | L39H1 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| glm4 | head_zero_step1::L39H2 | head_zero_step1 | L39H2 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| glm4 | head_zero_step1_2::L39H2 | head_zero_step1_2 | L39H2 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| glm4 | head_zero_step1::L39H3 | head_zero_step1 | L39H3 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| glm4 | head_zero_step1_2::L39H3 | head_zero_step1_2 | L39H3 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| glm4 | head_zero_step1::L39H0+L39H1+L39H2+L39H3 | head_zero_step1 | L39H0+L39H1+L39H2+L39H3 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| glm4 | head_zero_step1_2::L39H0+L39H1+L39H2+L39H3 | head_zero_step1_2 | L39H0+L39H1+L39H2+L39H3 | 6 | 6 | 0 | 6 | 0 | 6 | 0 | 0 | 6 | {'answer_alias': 5, 'strict_canonical': 1} |
| qwen3 | head_zero_step1::L35H0 | head_zero_step1 | L35H0 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1_2::L35H0 | head_zero_step1_2 | L35H0 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1::L35H1 | head_zero_step1 | L35H1 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1_2::L35H1 | head_zero_step1_2 | L35H1 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1::L35H2 | head_zero_step1 | L35H2 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1_2::L35H2 | head_zero_step1_2 | L35H2 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1::L35H3 | head_zero_step1 | L35H3 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1_2::L35H3 | head_zero_step1_2 | L35H3 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1::L35H0+L35H1+L35H2+L35H3 | head_zero_step1 | L35H0+L35H1+L35H2+L35H3 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1_2::L35H0+L35H1+L35H2+L35H3 | head_zero_step1_2 | L35H0+L35H1+L35H2+L35H3 | 4 | 4 | 0 | 4 | 0 | 1 | 2 | 0 | 4 | {'answer_alias': 1, 'strict_canonical': 3} |
| qwen3 | head_zero_step1::L32H0 | head_zero_step1 | L32H0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L32H0 | head_zero_step1_2 | L32H0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1::L32H1 | head_zero_step1 | L32H1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L32H1 | head_zero_step1_2 | L32H1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1::L32H2 | head_zero_step1 | L32H2 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L32H2 | head_zero_step1_2 | L32H2 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1::L32H3 | head_zero_step1 | L32H3 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L32H3 | head_zero_step1_2 | L32H3 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1::L32H0+L32H1+L32H2+L32H3 | head_zero_step1 | L32H0+L32H1+L32H2+L32H3 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L32H0+L32H1+L32H2+L32H3 | head_zero_step1_2 | L32H0+L32H1+L32H2+L32H3 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1::L30H0 | head_zero_step1 | L30H0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L30H0 | head_zero_step1_2 | L30H0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1::L30H1 | head_zero_step1 | L30H1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L30H1 | head_zero_step1_2 | L30H1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1::L30H2 | head_zero_step1 | L30H2 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L30H2 | head_zero_step1_2 | L30H2 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1::L30H3 | head_zero_step1 | L30H3 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L30H3 | head_zero_step1_2 | L30H3 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1::L30H0+L30H1+L30H2+L30H3 | head_zero_step1 | L30H0+L30H1+L30H2+L30H3 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
| qwen3 | head_zero_step1_2::L30H0+L30H1+L30H2+L30H3 | head_zero_step1_2 | L30H0+L30H1+L30H2+L30H3 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | {'strict_canonical': 1} |
