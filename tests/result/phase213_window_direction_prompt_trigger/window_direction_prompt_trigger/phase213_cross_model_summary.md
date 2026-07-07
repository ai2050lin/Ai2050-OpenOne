# Phase 213 window direction patch and prompt trigger atlas

Rollout rows: 92
Prompt trigger rows: 920
Total repair match gain: 0
Total damage match loss: 0

| model | window | success rows | drift rows | sites | repair gain | damage loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3 | answer_list->other_or_wrong L[31, 32, 33] S[10, 11] | 8 | 6 | 6 | 0 | 0 |
| qwen3 | answer_list->short_answer L[31, 32, 33] S[8, 9] | 8 | 2 | 6 | 0 | 0 |
| glm4 | answer_list->repeat_answer L[28, 29, 30] S[7, 8] | 1 | 8 | 6 | 0 | 0 |
| glm4 | answer_list->echo_then_answer L[34, 35, 36] S[7, 8] | 1 | 2 | 6 | -2 | 0 |
| deepseek7b | answer_explain->other_or_wrong L[25, 26, 27] S[6, 7] | 6 | 2 | 6 | 2 | 0 |
| deepseek7b | answer_list->other_or_wrong L[23, 24, 25] S[6, 7] | 6 | 2 | 6 | 0 | 0 |
