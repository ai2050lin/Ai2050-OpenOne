# Phase 225 readout competition threshold

propagation_rows: 1296
threshold_rows: 108
behavior_correlation_rows: 18
total_top_token_changed: 141
total_rank_improved: 219

| spec | group | condition | layer | rows | shift | top changed | rank improved | prose d | echo d | token pairs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K64 | 32 | 12 | -1.6159 | 4 | 1 | 0.1589 | 0.1589 | {' Green-> The': 2, ' White-> Blue': 1, ' White-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K64 | 30 | 12 | -1.4614 | 4 | 1 | 0.1589 | 0.1589 | {' Green-> The': 2, ' White-> Blue': 1, ' White-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K16 | 32 | 12 | -1.2589 | 4 | 1 | 0.1328 | 0.1979 | {' Green-> The': 2, ' White-> Blue': 1, ' White-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K16 | 30 | 12 | -1.2205 | 4 | 1 | 0.1328 | 0.1979 | {' Green-> The': 2, ' White-> Blue': 1, ' White-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K64 | 31 | 12 | -1.2052 | 4 | 1 | 0.1589 | 0.1589 | {' Green-> The': 2, ' White-> Blue': 1, ' White-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K16 | 31 | 12 | -0.9133 | 4 | 1 | 0.1328 | 0.1979 | {' Green-> The': 2, ' White-> Blue': 1, ' White-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K4 | 30 | 12 | -0.9458 | 4 | 0 | 0.1146 | 0.1953 | {' Green-> The': 2, ' White-> Blue': 1, ' White-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K4 | 32 | 12 | -0.9346 | 4 | 0 | 0.1146 | 0.1953 | {' Green-> The': 2, ' White-> Blue': 1, ' White-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_zero_L30_K4 | 31 | 12 | -0.6860 | 4 | 0 | 0.1146 | 0.1953 | {' Green-> The': 2, ' White-> Blue': 1, ' White-> The': 1} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K64 | 33 | 12 | 18.0175 | 3 | 7 | 1.4427 | 0.1126 | {' can-> is': 3} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K64 | 31 | 12 | 15.1672 | 3 | 7 | 1.4427 | 0.1126 | {' can-> is': 3} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K64 | 29 | 12 | 11.4991 | 3 | 7 | 1.4427 | 0.1126 | {' can-> is': 3} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K64 | 33 | 12 | 25.0593 | 3 | 5 | 1.1510 | 0.2197 | {' can-> is': 3} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K64 | 31 | 12 | 22.7522 | 3 | 5 | 1.1510 | 0.2197 | {' can-> is': 3} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K64 | 29 | 12 | 19.0338 | 3 | 5 | 1.1510 | 0.2197 | {' can-> is': 3} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K16 | 33 | 12 | 11.8067 | 3 | 4 | 0.8281 | 0.0052 | {' can-> is': 3} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K16 | 31 | 12 | 9.2156 | 3 | 4 | 0.8281 | 0.0052 | {' can-> is': 3} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_zero_L29_K16 | 29 | 12 | 6.4078 | 3 | 4 | 0.8281 | 0.0052 | {' can-> is': 3} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K64 | 30 | 12 | -1.4884 | 3 | 1 | 0.1745 | 0.2474 | {' Green-> The': 2, ' White-> Blue': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K64 | 32 | 12 | -1.4553 | 3 | 1 | 0.1745 | 0.2474 | {' Green-> The': 2, ' White-> Blue': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K64 | 31 | 12 | -1.2424 | 3 | 1 | 0.1745 | 0.2474 | {' Green-> The': 2, ' White-> Blue': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K16 | 30 | 12 | -1.0077 | 3 | 1 | 0.0703 | 0.1432 | {' Green-> The': 2, ' White-> Blue': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K16 | 32 | 12 | -0.9972 | 3 | 1 | 0.0703 | 0.1432 | {' Green-> The': 2, ' White-> Blue': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K16 | 31 | 12 | -0.8012 | 3 | 1 | 0.0703 | 0.1432 | {' Green-> The': 2, ' White-> Blue': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K4 | 32 | 12 | -1.9992 | 3 | 0 | 0.0990 | 0.1693 | {' Red-> Cardinal': 1, ' Red-> Green': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K4 | 30 | 12 | -1.9028 | 3 | 0 | 0.0990 | 0.1693 | {' Red-> Cardinal': 1, ' Red-> Green': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K4 | 31 | 12 | -1.8937 | 3 | 0 | 0.0990 | 0.1693 | {' Red-> Cardinal': 1, ' Red-> Green': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K64 | 32 | 12 | -3.9675 | 2 | 0 | 0.2578 | 0.2578 | {' Red-> Cardinal': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K64 | 32 | 12 | -3.7960 | 2 | 0 | 0.2292 | 0.2135 | {' Red-> Blue': 1, ' Red-> Cardinal': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K64 | 31 | 12 | -3.6013 | 2 | 0 | 0.2292 | 0.2135 | {' Red-> Blue': 1, ' Red-> Cardinal': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K64 | 31 | 12 | -3.5755 | 2 | 0 | 0.2578 | 0.2578 | {' Red-> Cardinal': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K64 | 30 | 12 | -3.5139 | 2 | 0 | 0.2292 | 0.2135 | {' Red-> Blue': 1, ' Red-> Cardinal': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K64 | 30 | 12 | -3.4881 | 2 | 0 | 0.2578 | 0.2578 | {' Red-> Cardinal': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K16 | 32 | 12 | -3.2894 | 2 | 0 | 0.1953 | 0.2526 | {' Red-> Cardinal': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K16 | 31 | 12 | -2.9701 | 2 | 0 | 0.1953 | 0.2526 | {' Red-> Cardinal': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K16 | 30 | 12 | -2.9606 | 2 | 0 | 0.1953 | 0.2526 | {' Red-> Cardinal': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K4 | 32 | 12 | -2.3403 | 2 | 0 | 0.1484 | 0.2474 | {' Red-> Cardinal': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K4 | 30 | 12 | -2.2296 | 2 | 0 | 0.1484 | 0.2474 | {' Red-> Cardinal': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_zero_L30_K4 | 31 | 12 | -2.1579 | 2 | 0 | 0.1484 | 0.2474 | {' Red-> Cardinal': 1, ' Red-> The': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_success_L30_K64 | 32 | 12 | 2.1308 | 2 | 0 | 0.0208 | -0.0156 | {' Green-> Red': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_success_L30_K64 | 31 | 12 | 2.1234 | 2 | 0 | 0.0208 | -0.0156 | {' Green-> Red': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_success_L30_K16 | 31 | 12 | 1.8263 | 2 | 0 | -0.0104 | -0.0286 | {' Green-> Red': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_success_L30_K16 | 32 | 12 | 1.8200 | 2 | 0 | -0.0104 | -0.0286 | {' Green-> Red': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_success_L30_K64 | 30 | 12 | 1.7797 | 2 | 0 | 0.0208 | -0.0156 | {' Green-> Red': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_success_L30_K16 | 30 | 12 | 1.5057 | 2 | 0 | -0.0104 | -0.0286 | {' Green-> Red': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_success_L30_K4 | 31 | 12 | 1.2768 | 2 | 0 | 0.0234 | 0.0130 | {' Green-> Red': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_success_L30_K4 | 32 | 12 | 1.2329 | 2 | 0 | 0.0234 | 0.0130 | {' Green-> Red': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_success_L30_K4 | 30 | 12 | 1.0755 | 2 | 0 | 0.0234 | 0.0130 | {' Green-> Red': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K4 | 30 | 12 | -0.6207 | 2 | 0 | 0.0625 | 0.1120 | {' Green-> The': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K4 | 32 | 12 | -0.5746 | 2 | 0 | 0.0625 | 0.1120 | {' Green-> The': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | drift_repro | mlpchan_pos_drift_L30_K4 | 31 | 12 | -0.4312 | 2 | 0 | 0.0625 | 0.1120 | {' Green-> The': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K16 | 32 | 12 | -3.0138 | 1 | 0 | 0.1146 | 0.1458 | {' Red-> Cardinal': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K16 | 31 | 12 | -2.8619 | 1 | 0 | 0.1146 | 0.1458 | {' Red-> Cardinal': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | success_repro | mlpchan_pos_drift_L30_K16 | 30 | 12 | -2.7466 | 1 | 0 | 0.1146 | 0.1458 | {' Red-> Cardinal': 1} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K16 | 33 | 12 | 16.5491 | 0 | 7 | 0.6927 | 0.0361 | {} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K16 | 31 | 12 | 14.8315 | 0 | 7 | 0.6927 | 0.0361 | {} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K16 | 29 | 12 | 12.1548 | 0 | 7 | 0.6927 | 0.0361 | {} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K4 | 33 | 12 | 7.9728 | 0 | 6 | 0.3958 | -0.1071 | {} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K4 | 31 | 12 | 7.7317 | 0 | 6 | 0.3958 | -0.1071 | {} |
| qwen3_explain_l29_to_l31_l33_propagation | drift_repro | mlpchan_pos_success_L29_K4 | 29 | 12 | 6.3995 | 0 | 6 | 0.3958 | -0.1071 | {} |

## Behavior correlation

| spec | condition | shift | top changed | rank improved | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| glm4_repeat_l30_to_l31_l32_propagation | mlpchan_pos_zero_L30_K64 | -2.5523 | 18 | 3 | 2 | 0 | {'echo_then_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | mlpchan_pos_zero_L30_K16 | -2.1021 | 18 | 3 | 2 | 0 | {'echo_then_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | mlpchan_pos_zero_L30_K4 | -1.5490 | 18 | 0 | 2 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 1, 'repeat_answer': 3} | {'echo_then_answer': 2, 'next_task_or_format': 1, 'other_or_wrong': 2} |
| glm4_repeat_l30_to_l31_l32_propagation | mlpchan_pos_drift_L30_K4 | -1.2371 | 15 | 0 | 4 | 0 | {'next_task_or_format': 3, 'other_or_wrong': 1, 'repeat_answer': 1} | {'echo_then_answer': 2, 'next_task_or_format': 2, 'other_or_wrong': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | mlpchan_pos_drift_L30_K64 | -2.5162 | 15 | 3 | 2 | 0 | {'next_task_or_format': 1, 'repeat_answer': 3, 'short_answer': 1} | {'echo_then_answer': 2, 'next_task_or_format': 2, 'other_or_wrong': 1} |
| qwen3_explain_l29_to_l31_l33_propagation | mlpchan_pos_zero_L29_K64 | 3.7813 | 9 | 33 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_l29_to_l31_l33_propagation | mlpchan_pos_success_L29_K64 | 10.6432 | 9 | 21 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_l29_to_l31_l33_propagation | mlpchan_pos_zero_L29_K16 | 1.4858 | 9 | 24 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | mlpchan_pos_drift_L30_K16 | -1.9047 | 12 | 3 | 1 | 0 | {'next_task_or_format': 1, 'repeat_answer': 4} | {'echo_then_answer': 2, 'next_task_or_format': 2, 'other_or_wrong': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | mlpchan_pos_success_L30_K64 | 0.8863 | 6 | 3 | 0 | 0 | {'repeat_answer': 5} | {'list_answer': 1, 'next_task_or_format': 4} |
| glm4_repeat_l30_to_l31_l32_propagation | mlpchan_pos_success_L30_K16 | 0.7525 | 6 | 3 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| glm4_repeat_l30_to_l31_l32_propagation | mlpchan_pos_success_L30_K4 | 0.5014 | 6 | 3 | 0 | 0 | {'repeat_answer': 5} | {'next_task_or_format': 4, 'other_or_wrong': 1} |
| qwen3_explain_l29_to_l31_l33_propagation | mlpchan_pos_drift_L29_K64 | -10.3911 | 0 | 21 | 1 | 0 | {'explain_answer': 4, 'other_or_wrong': 1} | {'other_or_wrong': 5} |
| qwen3_explain_l29_to_l31_l33_propagation | mlpchan_pos_drift_L29_K16 | -7.0703 | 0 | 18 | 1 | 0 | {'explain_answer': 4, 'other_or_wrong': 1} | {'other_or_wrong': 5} |
| qwen3_explain_l29_to_l31_l33_propagation | mlpchan_pos_success_L29_K16 | 6.8104 | 0 | 27 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_to_l31_l33_propagation | mlpchan_pos_success_L29_K4 | 3.4294 | 0 | 24 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_to_l31_l33_propagation | mlpchan_pos_drift_L29_K4 | -3.9780 | 0 | 15 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
| qwen3_explain_l29_to_l31_l33_propagation | mlpchan_pos_zero_L29_K4 | 0.1633 | 0 | 15 | 0 | 0 | {'explain_answer': 5} | {'other_or_wrong': 5} |
