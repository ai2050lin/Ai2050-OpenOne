# Phase60 Temporal Order Reader Calibration Summary

## qwen3

cases=384, rows=3072

| rank | reader | acc | min_ctx | min_rel | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | json_first_event | 0.7578 | 0.6562 | 0.4479 | 1.4408 | 2.0475 | no |
| 2 | a_first_yesno | 0.7448 | 0.6771 | 0.0833 | 1.2572 | 1.9466 | no |
| 3 | first_event_event_label | 0.7109 | 0.6146 | 0.2708 | 2.5003 | 3.1494 | no |
| 4 | after_statement_yesno | 0.6536 | 0.5104 | 0.4167 | 0.9668 | 1.5775 | no |
| 5 | before_statement_yesno | 0.6354 | 0.5000 | 0.2708 | 0.9414 | 1.9844 | no |
| 6 | b_first_yesno | 0.6198 | 0.5000 | 0.0208 | 1.1696 | 2.2002 | no |
| 7 | first_event_letter | 0.5677 | 0.5000 | 0.1146 | 1.3211 | 2.8081 | no |
| 8 | order_pair | 0.4974 | 0.4896 | 0.0000 | -0.1269 | 4.4584 | no |

## glm4

cases=384, rows=3072

| rank | reader | acc | min_ctx | min_rel | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | first_event_letter | 0.6068 | 0.5208 | 0.2292 | 0.3229 | 0.7998 | no |
| 2 | first_event_event_label | 0.5625 | 0.4896 | 0.0417 | 0.2630 | 0.7106 | no |
| 3 | b_first_yesno | 0.5495 | 0.5000 | 0.0104 | 0.0628 | 0.7220 | no |
| 4 | order_pair | 0.5469 | 0.5000 | 0.0938 | 0.4397 | 1.5186 | no |
| 5 | json_first_event | 0.5286 | 0.4896 | 0.2812 | 0.2358 | 1.6461 | no |
| 6 | after_statement_yesno | 0.5052 | 0.5000 | 0.0000 | 0.0535 | 1.0288 | no |
| 7 | a_first_yesno | 0.5000 | 0.5000 | 0.0000 | 0.1475 | 1.4001 | no |
| 8 | before_statement_yesno | 0.5000 | 0.5000 | 0.0000 | 0.0568 | 1.3853 | no |

## deepseek7b

cases=384, rows=3072

| rank | reader | acc | min_ctx | min_rel | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | json_first_event | 0.5990 | 0.5729 | 0.1354 | 1.2222 | 3.1520 | no |
| 2 | first_event_event_label | 0.5807 | 0.5000 | 0.0521 | 0.6961 | 2.0492 | no |
| 3 | first_event_letter | 0.5573 | 0.5000 | 0.0625 | 0.4843 | 1.8053 | no |
| 4 | order_pair | 0.5286 | 0.5000 | 0.0104 | 0.6781 | 3.6450 | no |
| 5 | b_first_yesno | 0.5104 | 0.4792 | 0.0000 | 0.0758 | 0.8896 | no |
| 6 | a_first_yesno | 0.5078 | 0.5000 | 0.0000 | 0.2342 | 1.4943 | no |
| 7 | after_statement_yesno | 0.5000 | 0.5000 | 0.0000 | 0.1379 | 1.9816 | no |
| 8 | before_statement_yesno | 0.5000 | 0.5000 | 0.0000 | 0.1115 | 2.4360 | no |

## Cross Model

| rank | reader | mean_acc | min_acc | min_ctx | min_rel | all_pass |
|---:|---|---:|---:|---:|---:|---|
| 1 | first_event_event_label | 0.6181 | 0.5625 | 0.4896 | 0.0417 | no |
| 2 | first_event_letter | 0.5773 | 0.5573 | 0.5000 | 0.0625 | no |
| 3 | json_first_event | 0.6285 | 0.5286 | 0.4896 | 0.1354 | no |
| 4 | b_first_yesno | 0.5599 | 0.5104 | 0.4792 | 0.0000 | no |
| 5 | a_first_yesno | 0.5842 | 0.5000 | 0.5000 | 0.0000 | no |
| 6 | after_statement_yesno | 0.5530 | 0.5000 | 0.5000 | 0.0000 | no |
| 7 | before_statement_yesno | 0.5451 | 0.5000 | 0.5000 | 0.0000 | no |
| 8 | order_pair | 0.5243 | 0.4974 | 0.4896 | 0.0000 | no |
