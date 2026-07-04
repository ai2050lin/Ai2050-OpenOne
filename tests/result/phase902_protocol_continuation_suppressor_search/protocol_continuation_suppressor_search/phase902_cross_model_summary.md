# Phase 902 protocol continuation suppressor search

## Overall

- models: qwen3, glm4, deepseek7b
- control_rows: 973
- non_base_clean_answer_no_protocol: 0
- non_base_next_top_changed: 28
- non_base_protocol_drift: 905
- non_base_protocol_logit_delta_below_minus_0_5: 57
- non_base_protocol_logit_delta_negative: 281
- non_base_protocol_rank1_removed: 8
- non_base_stop_rank_improved: 5
- non_base_stop_top1: 110
- selected_answer_drift_rows: 68

## Best controls

| model | control | type | head set | rows | clean | class clean | drift | protocol removed | protocol delta < -0.5 | stop improved | stop top1 | top changed | evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | head_zero_after_prefix::L31H0+L31H1+L31H2+L31H3 | head_zero_after_prefix | L31H0+L31H1+L31H2+L31H3 | 16 | 0 | 0 | 16 | 2 | 0 | 0 | 6 | 2 | logit_competition_shift_without_clean_rollout_closure |
| deepseek7b | head_zero_after_prefix::L26H3+L26H7+L26H11+L26H14 | head_zero_after_prefix | L26H3+L26H7+L26H11+L26H14 | 33 | 0 | 0 | 33 | 1 | 5 | 0 | 0 | 1 | weak_protocol_logit_suppression_without_clean_rollout_closure |
| glm4 | source_repeat_after_prefix | source_repeat_after_prefix | none | 17 | 0 | 0 | 17 | 0 | 0 | 2 | 0 | 0 | weak_protocol_logit_suppression_without_clean_rollout_closure |

## Model summaries

| model | selected | control rows | non-base clean | protocol removed | protocol delta negative | stop improved | stop top1 | best control | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| qwen3 | 18 | 308 | 0 | 2 | 72 | 0 | 110 | head_zero_after_prefix::L31H0+L31H1+L31H2+L31H3 | logit_competition_shift_without_clean_rollout_closure |
| glm4 | 17 | 170 | 0 | 0 | 55 | 5 | 0 | source_repeat_after_prefix | weak_protocol_logit_suppression_without_clean_rollout_closure |
| deepseek7b | 33 | 495 | 0 | 6 | 154 | 0 | 0 | head_zero_after_prefix::L26H3+L26H7+L26H11+L26H14 | weak_protocol_logit_suppression_without_clean_rollout_closure |
