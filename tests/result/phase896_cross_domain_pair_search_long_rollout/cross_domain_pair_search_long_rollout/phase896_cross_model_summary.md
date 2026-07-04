# Phase 896 cross-domain no-single pair search and long rollout stability

## Overall

- models: qwen3, glm4, deepseek7b
- selected_conditions: 952
- output_search_rows: 3400
- output_condition_rows: 952
- output_long_rollout_rows: 140
- cross_domain_known_axis_minimal_pair_conditions: 0
- focus_closure_from_open: 41
- known_axis_minimal_pair_conditions: 11
- no_single_pair_conditions: 11
- phase895_known_axis_replicated: 11
- rollout_answer_like_no_echo: 36
- rollout_class_hit: 39
- rollout_class_lost_vs_none: 9
- rollout_clear_answer: 36
- rollout_clear_lost_vs_none: 8
- rollout_object_echo: 0
- rollout_other_or_format: 101
- rollout_protocol_drift: 0

## Domain groups

| model | domain | conditions | focus closure | no-single pair | known minimal pair | phase895 replicated | pair keys |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek7b | color | 138 | 36 | 11 | 11 | 11 | {"L26C8587+L27C15369": 11} |
| qwen3 | abstract | 20 | 0 | 0 | 0 | 0 | {} |
| qwen3 | animal | 44 | 0 | 0 | 0 | 0 | {} |
| qwen3 | color | 92 | 0 | 0 | 0 | 0 | {} |
| qwen3 | geometry | 20 | 0 | 0 | 0 | 0 | {} |
| qwen3 | material | 44 | 4 | 0 | 0 | 0 | {} |
| qwen3 | object | 16 | 0 | 0 | 0 | 0 | {} |
| qwen3 | plant | 16 | 0 | 0 | 0 | 0 | {} |
| qwen3 | tool | 20 | 0 | 0 | 0 | 0 | {} |
| glm4 | abstract | 20 | 0 | 0 | 0 | 0 | {} |
| glm4 | animal | 44 | 0 | 0 | 0 | 0 | {} |
| glm4 | color | 92 | 0 | 0 | 0 | 0 | {} |
| glm4 | geometry | 20 | 0 | 0 | 0 | 0 | {} |
| glm4 | material | 44 | 0 | 0 | 0 | 0 | {} |
| glm4 | object | 16 | 0 | 0 | 0 | 0 | {} |
| glm4 | plant | 16 | 0 | 0 | 0 | 0 | {} |
| glm4 | tool | 20 | 0 | 0 | 0 | 0 | {} |
| deepseek7b | abstract | 30 | 0 | 0 | 0 | 0 | {} |
| deepseek7b | animal | 66 | 1 | 0 | 0 | 0 | {} |
| deepseek7b | geometry | 30 | 0 | 0 | 0 | 0 | {} |
| deepseek7b | material | 66 | 0 | 0 | 0 | 0 | {} |
| deepseek7b | object | 24 | 0 | 0 | 0 | 0 | {} |
| deepseek7b | plant | 24 | 0 | 0 | 0 | 0 | {} |
| deepseek7b | tool | 30 | 0 | 0 | 0 | 0 | {} |

## Pair-domain groups

| model | domain | subset | rows | closure | no-single | known minimal | mean lift | mean blocker reduction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | color | L26C8587+L27C15369 | 138 | 36 | 11 | 11 | 2.216 | -0.254 |
| deepseek7b | animal | L27C15369+L27C16651 | 66 | 39 | 0 | 0 | 2.629 | 1.970 |
| deepseek7b | animal | L26C8587+L27C16651 | 66 | 38 | 0 | 0 | 2.630 | 1.955 |
| deepseek7b | color | L27C15369+L27C16651 | 138 | 23 | 0 | 0 | 1.115 | -2.855 |
| deepseek7b | color | L26C8587+L27C16651 | 138 | 8 | 0 | 0 | 0.418 | 1.913 |
| deepseek7b | animal | L26C8587+L27C15369 | 66 | 1 | 0 | 0 | -0.009 | -0.091 |
| deepseek7b | abstract | L26C8587+L27C15369 | 30 | 0 | 0 | 0 | -0.007 | -0.233 |
| deepseek7b | abstract | L26C8587+L27C16651 | 30 | 0 | 0 | 0 | -0.004 | -0.100 |
| deepseek7b | abstract | L27C15369+L27C16651 | 30 | 0 | 0 | 0 | -0.009 | -0.133 |
| deepseek7b | geometry | L26C8587+L27C15369 | 30 | 0 | 0 | 0 | 0.010 | -0.033 |
| deepseek7b | geometry | L26C8587+L27C16651 | 30 | 0 | 0 | 0 | 0.004 | -0.133 |
| deepseek7b | geometry | L27C15369+L27C16651 | 30 | 0 | 0 | 0 | -0.019 | -0.067 |
| deepseek7b | material | L26C8587+L27C15369 | 66 | 0 | 0 | 0 | 0.015 | 0.197 |
| deepseek7b | material | L26C8587+L27C16651 | 66 | 0 | 0 | 0 | 0.010 | -0.530 |
| deepseek7b | material | L27C15369+L27C16651 | 66 | 0 | 0 | 0 | -0.012 | -1.242 |
| deepseek7b | object | L26C8587+L27C15369 | 24 | 0 | 0 | 0 | 0.008 | 0.167 |
| deepseek7b | object | L26C8587+L27C16651 | 24 | 0 | 0 | 0 | 0.005 | 0.583 |
| deepseek7b | object | L27C15369+L27C16651 | 24 | 0 | 0 | 0 | -0.012 | 0.542 |
| deepseek7b | plant | L26C8587+L27C15369 | 24 | 0 | 0 | 0 | 0.001 | -0.125 |
| deepseek7b | plant | L26C8587+L27C16651 | 24 | 0 | 0 | 0 | 0.021 | 0.625 |
| deepseek7b | plant | L27C15369+L27C16651 | 24 | 0 | 0 | 0 | -0.005 | 0.417 |
| deepseek7b | tool | L26C8587+L27C15369 | 30 | 0 | 0 | 0 | 0.004 | 0.000 |
| deepseek7b | tool | L26C8587+L27C16651 | 30 | 0 | 0 | 0 | -0.015 | 0.067 |
| deepseek7b | tool | L27C15369+L27C16651 | 30 | 0 | 0 | 0 | -0.027 | -0.033 |

## Long rollout groups

| model | domain | subset | head set | rows | class hit | clear | answer-like | object echo | other/format | drift | class lost | clear lost |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek7b | color | L26C8587+L27C15369 | L26H3+L26H7+L26H11+L26H14 | 11 | 7 | 7 | 7 | 0 | 4 | 0 | 4 | 3 |
| deepseek7b | color | L26C8587+L27C15369 | L26H7+L26H11 | 11 | 9 | 8 | 8 | 0 | 2 | 0 | 2 | 2 |
| qwen3 | material | L31C2257 | L31H19+L31H26+L31H30+L31H12+L31H17 | 4 | 0 | 0 | 0 | 0 | 4 | 0 | 2 | 2 |
| deepseek7b | color | L26C8587+L27C15369 | L26H7 | 11 | 10 | 9 | 9 | 0 | 1 | 0 | 1 | 1 |
| deepseek7b | color | L26C8587+L27C15369 | none | 11 | 11 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| qwen3 | material | L31C2257 | none | 4 | 2 | 2 | 2 | 0 | 2 | 0 | 0 | 0 |
| deepseek7b | color | L27C15369+L27C16651 | none | 11 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| deepseek7b | color | L27C15369+L27C16651 | L26H7 | 11 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| deepseek7b | color | L27C15369+L27C16651 | L26H7+L26H11 | 11 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| deepseek7b | color | L27C15369+L27C16651 | L26H3+L26H7+L26H11+L26H14 | 11 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| deepseek7b | color | L26C8587+L27C16651 | none | 11 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| deepseek7b | color | L26C8587+L27C16651 | L26H7 | 11 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| deepseek7b | color | L26C8587+L27C16651 | L26H7+L26H11 | 11 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| deepseek7b | color | L26C8587+L27C16651 | L26H3+L26H7+L26H11+L26H14 | 11 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
