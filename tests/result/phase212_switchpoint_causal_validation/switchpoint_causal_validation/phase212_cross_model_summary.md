# Phase 212 switchpoint causal validation

Rollout rows: 92
Total repair match gain: 0
Total damage match loss: 2

| model | candidate | success rows | drift rows | repair gain | damage loss |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3 | answer_list->other_or_wrong L32 S11 | 8 | 6 | 0 | 0 |
| qwen3 | answer_list->short_answer L32 S9 | 8 | 2 | 0 | 0 |
| glm4 | answer_list->repeat_answer L29 S8 | 1 | 8 | 0 | 0 |
| glm4 | answer_list->echo_then_answer L35 S8 | 1 | 2 | -2 | 0 |
| deepseek7b | answer_explain->other_or_wrong L26 S7 | 6 | 2 | 2 | 0 |
| deepseek7b | answer_list->other_or_wrong L24 S7 | 6 | 2 | 0 | 2 |
