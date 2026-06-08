# Phase60 Temporal Order Reader Calibration Summary

## qwen3

cases=8, rows=64

| rank | reader | acc | min_ctx | min_rel | margin | abs_margin | pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | after_statement_yesno | 0.8750 | 0.8750 | 0.5000 | 1.5172 | 1.6109 | no |
| 2 | first_event_event_label | 0.8750 | 0.8750 | 0.5000 | 3.0918 | 3.1021 | no |
| 3 | a_first_yesno | 0.7500 | 0.7500 | 0.0000 | 0.9688 | 1.7500 | no |
| 4 | b_first_yesno | 0.7500 | 0.7500 | 0.0000 | 1.7656 | 2.2344 | no |
| 5 | json_first_event | 0.7500 | 0.7500 | 0.0000 | 3.0156 | 3.3281 | no |
| 6 | before_statement_yesno | 0.5000 | 0.5000 | 0.0000 | 0.8594 | 1.5156 | no |
| 7 | first_event_letter | 0.5000 | 0.5000 | 0.0000 | 1.3281 | 2.9844 | no |
| 8 | order_pair | 0.5000 | 0.5000 | 0.0000 | 0.0064 | 4.6409 | no |

## glm4

missing

## deepseek7b

missing

## Cross Model

| rank | reader | mean_acc | min_acc | min_ctx | min_rel | all_pass |
|---:|---|---:|---:|---:|---:|---|
