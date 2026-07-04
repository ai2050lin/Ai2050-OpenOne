# Phase 905 stop action boundary audit

## Overall

- phase904_round: termination_control_candidate_search
- evidence: stop_top1_is_period_not_termination_action
- rows: 544
- strict_clean_answer_no_protocol: 0
- protocol_drift: 535
- strict_protocol_drift: 541
- stop_top1: 55
- stop_top1_strict_clean: 0
- stop_top1_protocol_drift: 55
- stop_top1_period_best: 55
- stop_top1_eos_best: 0
- stop_top1_period_first_suffix: 55
- stop_top1_eos_first_suffix: 0
- stop_top1_period_then_continuation: 55
- stop_top1_decoded_special_marker: 5

## Model Summaries

| model | stop_top1 | strict clean | drift | period best | eos best | period first | eos first | period then continuation | special marker | evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_stop_top1_boundary_to_audit |
| glm4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no_stop_top1_boundary_to_audit |
| qwen3 | 55 | 0 | 55 | 55 | 0 | 55 | 0 | 55 | 5 | stop_top1_is_period_not_termination_action |

## Stop Top1 Samples

| model | control | case | first | second | suffix |
| --- | --- | --- | --- | --- | --- |
| qwen3 | mlp_zero_L35 | p885_046_animal_cow | `.` | `<|endoftext|>` | `.Humanity. 1.` |
| qwen3 | attention_zero_L35 | p885_046_animal_cow | `.` | ` The` | `. The cow is a domesticated animal` |
| qwen3 | attention_zero_L34 | p885_046_animal_cow | `.` | ` \n\n` | `. \n\nThe answer is "Animal."` |
| qwen3 | attention_zero_L32 | p885_046_animal_cow | `.` | ` The` | `. The cow is a domesticated animal` |
| qwen3 | mlp_zero_L25 | p885_046_animal_cow | `.` | ` \n\n` | `. \n\nThe answer is "Animal."` |
| qwen3 | mlp_zero_L21 | p885_046_animal_cow | `.` | ` The` | `. The cow is a domesticated animal` |
| qwen3 | mlp_zero_L20 | p885_046_animal_cow | `.` | ` The` | `. The cow is a domesticated animal` |
| qwen3 | mlp_zero_L19 | p885_046_animal_cow | `.` | ` The` | `. The cow is a domesticated animal` |
| qwen3 | mlp_zero_L35 | p856_002_geometry_square | `.` | `<|endoftext|>` | `.Human: What is the category` |
| qwen3 | attention_zero_L35 | p856_002_geometry_square | `.` | ` \n\n` | `. \n\nThe category that best describes a` |
| qwen3 | attention_zero_L34 | p856_002_geometry_square | `.` | ` \n\n` | `. \n\nThe category that best describes a` |
| qwen3 | attention_zero_L32 | p856_002_geometry_square | `.` | ` \n\n` | `. \n\nThe category that best describes a` |
| qwen3 | mlp_zero_L25 | p856_002_geometry_square | `.` | ` \n\n` | `. \n\nThe category that best describes a` |
| qwen3 | mlp_zero_L21 | p856_002_geometry_square | `.` | ` \n\n` | `. \n\nThe category that best describes a` |
| qwen3 | mlp_zero_L20 | p856_002_geometry_square | `.` | ` \n\n` | `. \n\nThe category that best describes a` |
| qwen3 | mlp_zero_L19 | p856_002_geometry_square | `.` | ` \n\n` | `. \n\nThe category that best describes a` |
