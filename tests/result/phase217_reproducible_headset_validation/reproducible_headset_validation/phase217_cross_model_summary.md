# Phase 217 reproducible baseline and headset validation

Headset count: 7
Filter rows: 95
Reproducible success rows: 38
Reproducible drift rows: 40
Rollout rows: 198
Norm rows: 2754
Total damage match loss: 4
Total repair match gain: 0

| headset | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| deepseek7b_explain_l24_route_set | headset_all_steps | 6 | 2 | 4 | 0 | {'echo_then_answer': 2, 'explain_answer': 2, 'other_or_wrong': 2} | {'other_or_wrong': 2} |
| qwen3_explain_route_set | headset_anchor_step | 6 | 6 | 0 | 0 | {'explain_answer': 6} | {'list_answer': 3, 'other_or_wrong': 1, 'repeat_answer': 2} |
| qwen3_explain_route_set | headset_all_steps | 6 | 6 | 0 | 0 | {'explain_answer': 6} | {'list_answer': 3, 'other_or_wrong': 1, 'repeat_answer': 2} |
| qwen3_repeat_route_set | headset_anchor_step | 6 | 6 | 0 | 0 | {'repeat_answer': 6} | {'list_answer': 5, 'other_or_wrong': 1} |
| qwen3_repeat_route_set | headset_all_steps | 6 | 6 | 0 | 0 | {'repeat_answer': 6} | {'list_answer': 5, 'other_or_wrong': 1} |
| glm4_explain_competition_route_set | headset_anchor_step | 4 | 6 | 0 | 0 | {'explain_answer': 4} | {'repeat_answer': 6} |
| glm4_explain_competition_route_set | headset_all_steps | 4 | 6 | 0 | 0 | {'explain_answer': 4} | {'repeat_answer': 6} |
| glm4_repeat_route_set | headset_anchor_step | 6 | 6 | 0 | 0 | {'repeat_answer': 6} | {'list_answer': 6} |
| glm4_repeat_route_set | headset_all_steps | 6 | 6 | 0 | 0 | {'repeat_answer': 6} | {'list_answer': 6} |
| glm4_target_seeded_l29_route_set | headset_anchor_step | 0 | 6 | 0 | 0 | {} | {'repeat_answer': 6} |
| glm4_target_seeded_l29_route_set | headset_all_steps | 0 | 6 | 0 | 0 | {} | {'repeat_answer': 6} |
| deepseek7b_explain_l24_route_set | headset_anchor_step | 6 | 2 | 0 | 0 | {'explain_answer': 6} | {'other_or_wrong': 2} |
| deepseek7b_list_l24_route_set | headset_anchor_step | 4 | 2 | 0 | 0 | {'list_answer': 4} | {'other_or_wrong': 2} |
| deepseek7b_list_l24_route_set | headset_all_steps | 4 | 2 | 0 | 0 | {'list_answer': 4} | {'other_or_wrong': 2} |
