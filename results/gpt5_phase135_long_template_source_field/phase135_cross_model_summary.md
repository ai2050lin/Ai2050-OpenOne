# Phase 135 Cross-model Long-template Source Field Summary

## qwen3

Peak layer: L35; true last layer: L36; causal heads: [11, 10, 28, 3, 31, 2, 5, 20]

| category | audit | reference | best | prefix | object | relation | bridge | tail | all-pre |
|---|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=28.8 | last_input_pre_answer T-2.80 R+0.57 A+3.62 | prefix T-0.16 R+0.00 A-0.73 | prefix T-0.16 R+0.00 A-0.73 | object_span T+0.00 R+0.00 A-0.00 | relation_phrase T+0.00 R+0.02 A-0.02 | reasoning_bridge T-0.02 R+0.02 A-0.04 | answer_tail T-0.02 R+0.03 A+0.06 | all_pre_answer T-0.12 R+0.00 A-0.64 |
| container | old_mismatch=0, mean_pre=28.9 | last_input_pre_answer T-0.70 R+0.16 A+0.15 | prefix T-0.17 R+0.00 A-0.72 | prefix T-0.17 R+0.00 A-0.72 | object_span T+0.00 R+0.00 A-0.01 | relation_phrase T+0.01 R+0.02 A-0.02 | reasoning_bridge T+0.01 R+0.02 A-0.02 | answer_tail T+0.01 R+0.03 A+0.08 | all_pre_answer T-0.09 R+0.00 A-0.61 |
| plant | old_mismatch=0, mean_pre=29.2 | last_input_pre_answer T-0.41 R+0.08 A+5.56 | prefix T-0.23 R+0.00 A-0.70 | prefix T-0.23 R+0.00 A-0.70 | object_span T+0.00 R+0.00 A-0.01 | relation_phrase T+0.02 R+0.03 A-0.01 | reasoning_bridge T+0.02 R+0.02 A+0.00 | answer_tail T+0.02 R+0.03 A+0.09 | all_pre_answer T-0.15 R+0.00 A-0.59 |
| time | old_mismatch=0, mean_pre=28.8 | last_input_pre_answer T-1.90 R+0.28 A-0.53 | prefix T-0.18 R+0.00 A-0.67 | prefix T-0.18 R+0.00 A-0.67 | object_span T+0.00 R+0.00 A-0.01 | relation_phrase T+0.01 R+0.02 A-0.01 | reasoning_bridge T-0.02 R+0.02 A-0.02 | answer_tail T-0.03 R+0.03 A+0.07 | all_pre_answer T-0.14 R+0.00 A-0.55 |

## glm4

Peak layer: L18; true last layer: L40; causal heads: [1, 28, 0, 18, 11, 27, 23, 4]

| category | audit | reference | best | prefix | object | relation | bridge | tail | all-pre |
|---|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=80, mean_pre=30.8 | last_input_pre_answer T-0.04 R+0.67 A+2.82 | all_pre_answer T-0.08 R+0.02 A-1.75 | prefix T-0.03 R+0.02 A-0.42 | object_span T-0.00 R+0.00 A-0.08 | relation_phrase T-0.01 R+0.01 A-0.22 | reasoning_bridge T-0.01 R+0.02 A-0.45 | answer_tail T-0.01 R+0.01 A-0.40 | all_pre_answer T-0.08 R+0.02 A-1.75 |
| container | old_mismatch=95, mean_pre=30.9 | last_input_pre_answer T+0.09 R+0.60 A+4.74 | all_pre_answer T-0.03 R+0.02 A-1.86 | prefix T-0.03 R+0.01 A-0.53 | object_span T-0.01 R+0.00 A-0.22 | relation_phrase T+0.00 R+0.01 A-0.21 | reasoning_bridge T+0.01 R+0.02 A-0.39 | answer_tail T+0.00 R+0.01 A-0.40 | all_pre_answer T-0.03 R+0.02 A-1.86 |
| plant | old_mismatch=90, mean_pre=31.2 | last_input_pre_answer T-0.26 R+0.58 A+4.09 | prefix T-0.03 R+0.01 A-0.40 | prefix T-0.03 R+0.01 A-0.40 | object_span T-0.01 R+0.00 A-0.43 | relation_phrase T+0.00 R+0.02 A-0.29 | reasoning_bridge T-0.01 R+0.02 A-0.45 | answer_tail T-0.01 R+0.01 A-0.38 | all_pre_answer T-0.00 R+0.03 A-1.70 |
| time | old_mismatch=80, mean_pre=30.8 | last_input_pre_answer T+0.11 R+0.58 A+3.73 | all_pre_answer T-0.10 R+0.02 A-1.72 | prefix T-0.04 R+0.01 A-0.39 | object_span T-0.00 R+0.00 A-0.11 | relation_phrase T-0.01 R+0.01 A-0.22 | reasoning_bridge T-0.01 R+0.01 A-0.41 | answer_tail T-0.01 R+0.01 A-0.44 | all_pre_answer T-0.10 R+0.02 A-1.72 |

## deepseek7b

Peak layer: L27; true last layer: L28; causal heads: [13, 12, 11, 8, 25, 10, 26, 24]

| category | audit | reference | best | prefix | object | relation | bridge | tail | all-pre |
|---|---|---|---|---|---|---|---|---|---|
| number | old_mismatch=0, mean_pre=28.8 | last_input_pre_answer T-2.66 R+0.49 A-28.80 | all_pre_answer T-0.64 R+0.00 A-10.13 | prefix T-0.45 R+0.00 A-12.02 | object_span T+0.00 R+0.00 A+0.12 | relation_phrase T-0.02 R+0.02 A+0.26 | reasoning_bridge T-0.10 R+0.03 A+0.46 | answer_tail T-0.02 R+0.04 A+0.63 | all_pre_answer T-0.64 R+0.00 A-10.13 |
| container | old_mismatch=0, mean_pre=28.9 | last_input_pre_answer T-2.67 R+0.00 A-31.88 | all_pre_answer T-1.05 R+0.00 A-10.52 | prefix T-0.46 R+0.00 A-12.20 | object_span T+0.00 R+0.01 A+0.15 | relation_phrase T-0.11 R+0.02 A+0.24 | reasoning_bridge T-0.21 R+0.05 A+0.55 | answer_tail T-0.10 R+0.04 A+0.49 | all_pre_answer T-1.05 R+0.00 A-10.52 |
| plant | old_mismatch=0, mean_pre=29.3 | last_input_pre_answer T-2.70 R+0.59 A-23.81 | all_pre_answer T-1.08 R+0.00 A-8.96 | prefix T-0.52 R+0.00 A-10.27 | object_span T-0.00 R+0.02 A+0.02 | relation_phrase T-0.12 R+0.02 A+0.36 | reasoning_bridge T-0.27 R+0.02 A+0.42 | answer_tail T-0.09 R+0.02 A+0.65 | all_pre_answer T-1.08 R+0.00 A-8.96 |
| time | old_mismatch=0, mean_pre=28.8 | last_input_pre_answer T-3.13 R+0.84 A-34.96 | all_pre_answer T-0.82 R+0.02 A-9.94 | prefix T-0.49 R+0.00 A-11.20 | object_span T-0.00 R+0.02 A+0.02 | relation_phrase T-0.04 R+0.01 A+0.26 | reasoning_bridge T-0.13 R+0.04 A+0.54 | answer_tail T-0.05 R+0.04 A+0.50 | all_pre_answer T-0.82 R+0.02 A-9.94 |

