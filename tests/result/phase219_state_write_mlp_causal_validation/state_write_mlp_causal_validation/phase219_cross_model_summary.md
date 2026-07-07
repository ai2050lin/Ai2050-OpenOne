# Phase 219 state write and MLP causal validation

spec_count: 5
filter_rows: 71
reproducible_success_rows: 19
reproducible_drift_rows: 24
rollout_rows: 357
write_score_rows: 540
total_damage_match_loss: 32
total_repair_match_gain: 29

| spec | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_state_write | resid_add_L29 | 3 | 3 | 1 | 3 | {'explain_answer': 2, 'other_or_wrong': 1} | {'explain_answer': 3} |
| qwen3_explain_state_write | mlp_zero_L31 | 3 | 3 | 1 | 2 | {'explain_answer': 2, 'other_or_wrong': 1} | {'explain_answer': 2, 'other_or_wrong': 1} |
| qwen3_explain_state_write | resid_add_L11 | 3 | 3 | 1 | 2 | {'explain_answer': 2, 'other_or_wrong': 1} | {'explain_answer': 2, 'other_or_wrong': 1} |
| qwen3_explain_state_write | resid_add_L31 | 3 | 3 | 0 | 3 | {'explain_answer': 3} | {'explain_answer': 3} |
| qwen3_explain_state_write | resid_add_L33 | 3 | 3 | 0 | 3 | {'explain_answer': 3} | {'explain_answer': 3} |
| qwen3_explain_state_write | resid_sub_L11 | 3 | 3 | 3 | 0 | {'echo_then_answer': 2, 'other_or_wrong': 1} | {'other_or_wrong': 3} |
| qwen3_explain_state_write | resid_sub_L29 | 3 | 3 | 3 | 0 | {'other_or_wrong': 3} | {'other_or_wrong': 3} |
| qwen3_explain_state_write | resid_sub_L31 | 3 | 3 | 3 | 0 | {'other_or_wrong': 3} | {'other_or_wrong': 3} |
| qwen3_repeat_state_write | resid_sub_L29 | 3 | 3 | 3 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 2} | {'next_task_or_format': 2, 'other_or_wrong': 1} |
| qwen3_repeat_state_write | resid_sub_L31 | 3 | 3 | 3 | 0 | {'other_or_wrong': 3} | {'echo_then_answer': 1, 'next_task_or_format': 2} |
| qwen3_repeat_state_write | resid_sub_L32 | 3 | 3 | 3 | 0 | {'next_task_or_format': 1, 'other_or_wrong': 1, 'short_answer': 1} | {'next_task_or_format': 3} |
| glm4_repeat_state_write | resid_sub_L28 | 3 | 3 | 3 | 0 | {'echo_then_answer': 1, 'next_task_or_format': 1, 'other_or_wrong': 1} | {'next_task_or_format': 1, 'other_or_wrong': 2} |
| qwen3_explain_state_write | attn_zero_L11 | 3 | 3 | 0 | 2 | {'explain_answer': 3} | {'explain_answer': 2, 'other_or_wrong': 1} |
| qwen3_explain_state_write | mlp_zero_L11 | 3 | 3 | 0 | 2 | {'explain_answer': 3} | {'explain_answer': 2, 'other_or_wrong': 1} |
| qwen3_explain_state_write | mlp_zero_L29 | 3 | 3 | 0 | 2 | {'explain_answer': 3} | {'explain_answer': 2, 'other_or_wrong': 1} |
| qwen3_explain_state_write | resid_sub_L33 | 3 | 3 | 2 | 0 | {'explain_answer': 1, 'other_or_wrong': 2} | {'other_or_wrong': 3} |
| glm4_explain_competition_state_write | resid_add_L28 | 1 | 2 | 0 | 2 | {'explain_answer': 1} | {'explain_answer': 2} |
| glm4_explain_competition_state_write | resid_add_L29 | 1 | 2 | 0 | 2 | {'explain_answer': 1} | {'explain_answer': 2} |
| glm4_explain_competition_state_write | resid_add_L30 | 1 | 2 | 0 | 2 | {'explain_answer': 1} | {'explain_answer': 2} |
| glm4_explain_competition_state_write | resid_sub_L28 | 1 | 2 | 0 | 2 | {'explain_answer': 1} | {'explain_answer': 2} |
| glm4_explain_competition_state_write | resid_sub_L30 | 1 | 2 | 0 | 2 | {'explain_answer': 1} | {'explain_answer': 2} |
| glm4_repeat_state_write | resid_sub_L29 | 3 | 3 | 2 | 0 | {'echo_then_answer': 1, 'next_task_or_format': 1, 'repeat_answer': 1} | {'next_task_or_format': 3} |
| glm4_repeat_state_write | resid_sub_L30 | 3 | 3 | 2 | 0 | {'echo_then_answer': 1, 'next_task_or_format': 1, 'repeat_answer': 1} | {'next_task_or_format': 3} |
| qwen3_repeat_state_write | resid_sub_L33 | 3 | 3 | 1 | 0 | {'next_task_or_format': 1, 'repeat_answer': 2} | {'next_task_or_format': 2, 'other_or_wrong': 1} |
| glm4_repeat_state_write | mlp_zero_L30 | 3 | 3 | 1 | 0 | {'next_task_or_format': 1, 'repeat_answer': 2} | {'next_task_or_format': 1, 'other_or_wrong': 2} |
| qwen3_explain_state_write | attn_zero_L29 | 3 | 3 | 0 | 0 | {'explain_answer': 3} | {'other_or_wrong': 3} |
| qwen3_explain_state_write | attn_zero_L31 | 3 | 3 | 0 | 0 | {'explain_answer': 3} | {'other_or_wrong': 3} |
| qwen3_explain_state_write | attn_zero_L33 | 3 | 3 | 0 | 0 | {'explain_answer': 3} | {'other_or_wrong': 3} |
| qwen3_explain_state_write | mlp_zero_L33 | 3 | 3 | 0 | 0 | {'explain_answer': 3} | {'other_or_wrong': 3} |
| qwen3_repeat_state_write | attn_zero_L29 | 3 | 3 | 0 | 0 | {'repeat_answer': 3} | {'next_task_or_format': 3} |

## Top write scores

| spec | group | layer | module | rows | cosine | abs cosine | norm |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| qwen3_explain_state_write | drift_repro | 33 | resid | 6 | -0.462424099445343 | 0.462424099445343 | 478.59991455078125 |
| qwen3_explain_state_write | drift_repro | 31 | resid | 6 | -0.45162245631217957 | 0.45162245631217957 | 337.2796325683594 |
| qwen3_explain_state_write | drift_repro | 29 | resid | 6 | -0.4272415141264598 | 0.4272415141264598 | 237.70547485351562 |
| qwen3_repeat_state_write | success_repro | 32 | resid | 6 | 0.3774862736463547 | 0.3774862736463547 | 374.6107126871745 |
| qwen3_repeat_state_write | success_repro | 31 | resid | 6 | 0.3369416395823161 | 0.3369416395823161 | 322.76935323079425 |
| glm4_repeat_state_write | success_repro | 12 | resid | 6 | 0.33632151285807294 | 0.33632151285807294 | 4.294267892837524 |
| qwen3_repeat_state_write | drift_repro | 29 | attn | 6 | -0.32714980964859325 | 0.32714980964859325 | 53.241759618123375 |
| qwen3_repeat_state_write | success_repro | 32 | mlp | 6 | 0.3208695600430171 | 0.3208695600430171 | 132.64029820760092 |
| qwen3_repeat_state_write | success_repro | 29 | resid | 6 | 0.31787242243687314 | 0.31787242243687314 | 231.39857737223306 |
| qwen3_repeat_state_write | success_repro | 33 | resid | 6 | 0.3174182524283727 | 0.3174182524283727 | 435.13275146484375 |
| qwen3_repeat_state_write | drift_repro | 29 | resid | 6 | -0.29671114434798557 | 0.29671114434798557 | 246.3101018269857 |
| glm4_repeat_state_write | success_repro | 30 | mlp | 6 | 0.29614779849847156 | 0.29614779849847156 | 14.83812157313029 |
| glm4_repeat_state_write | success_repro | 28 | resid | 6 | 0.2954012056191762 | 0.2954012056191762 | 52.8080202738444 |
| qwen3_explain_state_write | drift_repro | 11 | resid | 6 | -0.2934342871109645 | 0.2934342871109645 | 40.72657140096029 |
| glm4_repeat_state_write | success_repro | 30 | resid | 6 | 0.28705688814322156 | 0.28705688814322156 | 72.39910634358723 |
| qwen3_repeat_state_write | drift_repro | 29 | mlp | 6 | -0.2838299746314685 | 0.2838299746314685 | 88.65630149841309 |
| qwen3_repeat_state_write | drift_repro | 31 | resid | 6 | -0.28016989678144455 | 0.28016989678144455 | 335.50099182128906 |
| glm4_repeat_state_write | success_repro | 29 | resid | 6 | 0.27725162853797275 | 0.27725162853797275 | 62.21690813700358 |
| qwen3_explain_state_write | drift_repro | 31 | mlp | 6 | -0.2734910249710083 | 0.2734910249710083 | 85.24729919433594 |
| qwen3_repeat_state_write | drift_repro | 32 | resid | 6 | -0.2702585893372695 | 0.2702585893372695 | 376.7112731933594 |
| qwen3_repeat_state_write | drift_repro | 33 | resid | 6 | -0.2650384120643139 | 0.2650384120643139 | 445.0270589192708 |
| qwen3_explain_state_write | drift_repro | 33 | mlp | 6 | -0.2638918956120809 | 0.2638918956120809 | 129.054931640625 |
| qwen3_repeat_state_write | success_repro | 29 | attn | 6 | 0.24922161300977072 | 0.24922161300977072 | 44.04215176900228 |
| qwen3_explain_state_write | success_repro | 29 | resid | 6 | 0.24815724790096283 | 0.24815724790096283 | 225.2453358968099 |
| glm4_repeat_state_write | success_repro | 28 | mlp | 6 | 0.23895880579948425 | 0.23895880579948425 | 11.25683879852295 |
| qwen3_explain_state_write | drift_repro | 29 | mlp | 6 | -0.23243054995934168 | 0.23243054995934168 | 65.60297139485677 |
| qwen3_repeat_state_write | success_repro | 31 | mlp | 6 | 0.22905457516511282 | 0.22905457516511282 | 109.71664047241211 |
| qwen3_repeat_state_write | drift_repro | 31 | mlp | 6 | -0.21927257099499306 | 0.21927257099499306 | 92.3239860534668 |
| qwen3_explain_state_write | success_repro | 29 | mlp | 6 | 0.21646000444889069 | 0.21646000444889069 | 67.50181325276692 |
| glm4_explain_competition_state_write | drift_repro | 28 | resid | 6 | -0.21402175724506378 | 0.21402175724506378 | 54.931811014811196 |
