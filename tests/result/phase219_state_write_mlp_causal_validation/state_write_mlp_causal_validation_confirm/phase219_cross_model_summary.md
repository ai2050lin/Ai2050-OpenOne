# Phase 219 state write and MLP causal validation

spec_count: 5
filter_rows: 99
reproducible_success_rows: 28
reproducible_drift_rows: 36
rollout_rows: 595
write_score_rows: 792
total_damage_match_loss: 66
total_repair_match_gain: 50

| spec | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_state_write | resid_add_L31 | 5 | 5 | 2 | 5 | {'explain_answer': 3, 'other_or_wrong': 2} | {'explain_answer': 5} |
| qwen3_explain_state_write | mlp_zero_L31 | 5 | 5 | 2 | 4 | {'explain_answer': 3, 'other_or_wrong': 2} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_state_write | resid_add_L29 | 5 | 5 | 2 | 4 | {'explain_answer': 3, 'other_or_wrong': 2} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_repeat_state_write | resid_sub_L31 | 5 | 5 | 5 | 1 | {'other_or_wrong': 3, 'short_answer': 2} | {'next_task_or_format': 2, 'other_or_wrong': 2, 'repeat_answer': 1} |
| qwen3_explain_state_write | resid_sub_L29 | 5 | 5 | 5 | 0 | {'other_or_wrong': 5} | {'other_or_wrong': 5} |
| qwen3_explain_state_write | resid_sub_L31 | 5 | 5 | 5 | 0 | {'other_or_wrong': 3, 'short_answer': 2} | {'other_or_wrong': 5} |
| qwen3_explain_state_write | resid_sub_L33 | 5 | 5 | 5 | 0 | {'other_or_wrong': 3, 'short_answer': 2} | {'other_or_wrong': 5} |
| qwen3_repeat_state_write | resid_sub_L33 | 5 | 5 | 5 | 0 | {'list_answer': 2, 'next_task_or_format': 1, 'short_answer': 2} | {'list_answer': 1, 'next_task_or_format': 2, 'other_or_wrong': 2} |
| glm4_repeat_state_write | resid_sub_L28 | 5 | 5 | 5 | 0 | {'echo_then_answer': 3, 'other_or_wrong': 2} | {'list_answer': 1, 'other_or_wrong': 4} |
| qwen3_explain_state_write | attn_zero_L11 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_state_write | mlp_zero_L11 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_state_write | mlp_zero_L29 | 5 | 5 | 0 | 4 | {'explain_answer': 5} | {'explain_answer': 4, 'other_or_wrong': 1} |
| qwen3_explain_state_write | resid_add_L33 | 5 | 5 | 3 | 1 | {'explain_answer': 2, 'other_or_wrong': 3} | {'explain_answer': 1, 'other_or_wrong': 4} |
| qwen3_explain_state_write | resid_sub_L11 | 5 | 5 | 4 | 0 | {'echo_then_answer': 2, 'explain_answer': 1, 'other_or_wrong': 2} | {'other_or_wrong': 5} |
| qwen3_repeat_state_write | resid_sub_L29 | 5 | 5 | 3 | 1 | {'next_task_or_format': 1, 'other_or_wrong': 2, 'repeat_answer': 2} | {'next_task_or_format': 2, 'other_or_wrong': 2, 'repeat_answer': 1} |
| qwen3_repeat_state_write | resid_sub_L32 | 5 | 5 | 3 | 1 | {'other_or_wrong': 3, 'repeat_answer': 2} | {'next_task_or_format': 4, 'repeat_answer': 1} |
| glm4_explain_competition_state_write | resid_add_L28 | 1 | 4 | 0 | 4 | {'explain_answer': 1} | {'explain_answer': 4} |
| glm4_explain_competition_state_write | resid_add_L29 | 1 | 4 | 0 | 4 | {'explain_answer': 1} | {'explain_answer': 4} |
| glm4_explain_competition_state_write | resid_add_L30 | 1 | 4 | 0 | 4 | {'explain_answer': 1} | {'explain_answer': 4} |
| glm4_repeat_state_write | resid_sub_L29 | 5 | 5 | 4 | 0 | {'echo_then_answer': 3, 'other_or_wrong': 1, 'repeat_answer': 1} | {'next_task_or_format': 3, 'other_or_wrong': 2} |
| glm4_repeat_state_write | resid_sub_L30 | 5 | 5 | 4 | 0 | {'echo_then_answer': 1, 'next_task_or_format': 2, 'other_or_wrong': 1, 'repeat_answer': 1} | {'next_task_or_format': 3, 'other_or_wrong': 2} |
| qwen3_explain_state_write | resid_add_L11 | 5 | 5 | 2 | 1 | {'explain_answer': 3, 'other_or_wrong': 2} | {'echo_then_answer': 4, 'explain_answer': 1} |
| glm4_explain_competition_state_write | mlp_zero_L12 | 1 | 4 | 0 | 2 | {'explain_answer': 1} | {'explain_answer': 2, 'repeat_answer': 2} |
| glm4_explain_competition_state_write | mlp_zero_L28 | 1 | 4 | 0 | 2 | {'explain_answer': 1} | {'explain_answer': 2, 'other_or_wrong': 2} |
| glm4_explain_competition_state_write | resid_sub_L28 | 1 | 4 | 1 | 1 | {'other_or_wrong': 1} | {'explain_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 2} |
| qwen3_explain_state_write | attn_zero_L29 | 5 | 5 | 1 | 0 | {'echo_then_answer': 1, 'explain_answer': 4} | {'other_or_wrong': 5} |
| qwen3_explain_state_write | mlp_zero_L33 | 5 | 5 | 1 | 0 | {'echo_then_answer': 1, 'explain_answer': 4} | {'other_or_wrong': 5} |
| glm4_explain_competition_state_write | attn_zero_L12 | 1 | 4 | 0 | 1 | {'explain_answer': 1} | {'explain_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 2} |
| glm4_explain_competition_state_write | attn_zero_L30 | 1 | 4 | 0 | 1 | {'explain_answer': 1} | {'explain_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 2} |
| glm4_explain_competition_state_write | mlp_zero_L29 | 1 | 4 | 0 | 1 | {'explain_answer': 1} | {'explain_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 2} |

## Top write scores

| spec | group | layer | module | rows | cosine | abs cosine | norm |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| qwen3_explain_state_write | drift_repro | 33 | resid | 9 | -0.3868487477302551 | 0.3868487477302551 | 473.1840582953559 |
| qwen3_explain_state_write | drift_repro | 31 | resid | 9 | -0.36477706167433 | 0.36477706167433 | 335.16221788194446 |
| glm4_repeat_state_write | success_repro | 12 | resid | 9 | 0.3611602667305205 | 0.3611602667305205 | 4.300837066438463 |
| qwen3_explain_state_write | drift_repro | 29 | resid | 9 | -0.33141430301798713 | 0.33141430301798713 | 236.74288092719183 |
| glm4_repeat_state_write | success_repro | 28 | resid | 9 | 0.314813612235917 | 0.314813612235917 | 52.20660654703776 |
| qwen3_repeat_state_write | success_repro | 32 | resid | 9 | 0.29971497278246617 | 0.3022208145509164 | 370.8600429958767 |
| glm4_repeat_state_write | success_repro | 30 | mlp | 9 | 0.30187686284383136 | 0.30187686284383136 | 14.501598676045736 |
| glm4_repeat_state_write | success_repro | 30 | resid | 9 | 0.3008497357368469 | 0.3008497357368469 | 71.66311815049913 |
| glm4_repeat_state_write | success_repro | 29 | resid | 9 | 0.29068902962737614 | 0.29068902962737614 | 61.508076985677086 |
| qwen3_repeat_state_write | success_repro | 29 | resid | 9 | 0.2887906986806128 | 0.2887906986806128 | 228.24241299099393 |
| qwen3_repeat_state_write | success_repro | 31 | resid | 9 | 0.2833828168610732 | 0.2833828168610732 | 321.9454752604167 |
| qwen3_repeat_state_write | success_repro | 32 | mlp | 9 | 0.2644081951843368 | 0.2644081951843368 | 115.52143436008029 |
| glm4_repeat_state_write | success_repro | 28 | mlp | 9 | 0.26049769918123883 | 0.26049769918123883 | 10.896239651574028 |
| qwen3_repeat_state_write | success_repro | 33 | resid | 9 | 0.24964028803838623 | 0.2545557729899883 | 433.8326416015625 |
| glm4_explain_competition_state_write | success_repro | 30 | attn | 3 | 0.24944415191809335 | 0.24944415191809335 | 8.421830495198568 |
| qwen3_explain_state_write | drift_repro | 11 | resid | 9 | -0.20223557866281933 | 0.24678823931349647 | 40.99767515394423 |
| qwen3_repeat_state_write | drift_repro | 29 | attn | 9 | -0.23989821018444168 | 0.23989821018444168 | 53.3101938035753 |
| glm4_explain_competition_state_write | success_repro | 29 | attn | 3 | 0.18789608279863992 | 0.23599893848101297 | 7.313747723897298 |
| qwen3_explain_state_write | drift_repro | 31 | mlp | 9 | -0.21912367145220438 | 0.21912367145220438 | 87.94123840332031 |
| qwen3_explain_state_write | drift_repro | 33 | mlp | 9 | -0.21560735503832498 | 0.21560735503832498 | 134.99171702067056 |
| qwen3_repeat_state_write | drift_repro | 31 | resid | 9 | -0.14670364227559832 | 0.21018781099054548 | 334.6019219292535 |
| qwen3_repeat_state_write | success_repro | 29 | attn | 9 | 0.1949134882953432 | 0.1949134882953432 | 39.041568756103516 |
| qwen3_explain_state_write | drift_repro | 29 | mlp | 9 | -0.19177965654267204 | 0.19177965654267204 | 69.66951073540582 |
| qwen3_repeat_state_write | drift_repro | 32 | resid | 9 | -0.14594699318210283 | 0.19136550650000572 | 375.387200249566 |
| glm4_explain_competition_state_write | success_repro | 30 | resid | 3 | 0.14536846180756888 | 0.1908620943625768 | 73.33503977457683 |
| glm4_repeat_state_write | drift_repro | 12 | resid | 9 | 0.19035377436214024 | 0.19035377436214024 | 4.325107124116686 |
| qwen3_repeat_state_write | drift_repro | 29 | resid | 9 | -0.14925881971915564 | 0.18936033298571905 | 242.98592291937933 |
| qwen3_repeat_state_write | drift_repro | 29 | mlp | 9 | -0.1806060050924619 | 0.18410580025778878 | 82.0936902364095 |
| qwen3_repeat_state_write | drift_repro | 33 | resid | 9 | -0.1487081605527136 | 0.18251830753352907 | 445.4742160373264 |
| glm4_explain_competition_state_write | success_repro | 29 | resid | 3 | 0.1121943990389506 | 0.18033483624458313 | 62.99005381266276 |
