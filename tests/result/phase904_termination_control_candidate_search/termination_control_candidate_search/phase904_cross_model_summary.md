# Phase 904 termination control candidate search

## Overall

- models: qwen3, glm4, deepseek7b
- candidate_count: 24
- control_rows: 612
- non_base_clean_answer_no_protocol: 6
- non_base_next_top_changed: 102
- non_base_protocol_drift: 535
- non_base_protocol_logit_reduced_strong: 380
- non_base_protocol_rank1_removed: 48
- non_base_stop_rank_improved: 92
- non_base_stop_top1: 55
- non_base_strict_clean_answer_no_protocol: 0
- non_base_strict_protocol_drift: 541
- selected_answer_drift_rows: 68

## Model Summaries

| model | candidates | rows | non-base strict clean | drift | removed | stop improved | stop top1 | top changed | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | 8 | 162 | 0 | 140 | 0 | 35 | 55 | 15 | termination_candidate_reduces_protocol_without_clean_rollout |
| glm4 | 8 | 153 | 0 | 136 | 12 | 26 | 0 | 14 | termination_candidate_changes_competition_without_clean_rollout |
| deepseek7b | 8 | 297 | 0 | 259 | 36 | 31 | 0 | 73 | termination_candidate_changes_competition_without_clean_rollout |

## Best Controls

| model | control | layer | kind | category | rows | strict clean | nominal clean | drift | removed | stop improved | first suffix categories |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b | attention_zero_L27 | 27 | attention | comma | 33 | 0 | 0 | 32 | 17 | 0 | `{"comma": 12, "newline": 4, "other": 17}` |
| glm4 | attention_zero_L38 | 38 | attention | newline | 17 | 0 | 0 | 17 | 6 | 2 | `{"comma": 1, "newline": 9, "other": 7}` |
| qwen3 | attention_zero_L34 | 34 | attention | newline | 18 | 0 | 0 | 18 | 0 | 8 | `{"newline": 11, "period": 7}` |
