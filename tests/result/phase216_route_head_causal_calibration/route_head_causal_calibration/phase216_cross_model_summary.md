# Phase 216 route head causal calibration

Candidate count: 15
Rollout rows: 342
Total damage match loss: 1
Total repair match gain: 0

| candidate | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| deepseek7b|answer_explain|gen_after_step_1|L24H16 | ablate_all_steps | 4 | 2 | 4 | 0 | {'echo_then_answer': 2, 'other_or_wrong': 2} | {'other_or_wrong': 2} |
| glm4|answer_explain|gen_after_step_3|L12H18 | ablate_all_steps | 4 | 4 | -3 | 0 | {'explain_answer': 4} | {'explain_answer': 1, 'repeat_answer': 3} |
| qwen3|answer_explain|gen_after_step_1|L11H3 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'explain_answer': 2, 'repeat_answer': 2} |
| qwen3|answer_explain|gen_after_step_1|L11H3 | ablate_all_steps | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'explain_answer': 2, 'repeat_answer': 2} |
| qwen3|answer_explain|gen_after_step_3|L3H15 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'explain_answer': 2, 'repeat_answer': 2} |
| qwen3|answer_explain|gen_after_step_3|L3H15 | ablate_all_steps | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'explain_answer': 2, 'repeat_answer': 2} |
| qwen3|answer_explain|gen_after_step_6|L29H11 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'explain_answer': 2, 'repeat_answer': 2} |
| qwen3|answer_explain|gen_after_step_6|L29H11 | ablate_all_steps | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'explain_answer': 2, 'repeat_answer': 2} |
| qwen3|answer_repeat|gen_after_step_1|L29H11 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| qwen3|answer_repeat|gen_after_step_1|L29H11 | ablate_all_steps | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| qwen3|answer_repeat|prompt_last|L31H26 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| qwen3|answer_repeat|prompt_last|L31H26 | ablate_all_steps | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| glm4|answer_explain|gen_after_step_3|L12H18 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'explain_answer': 1, 'list_answer': 1, 'other_or_wrong': 1, 'repeat_answer': 1} | {'explain_answer': 1, 'repeat_answer': 3} |
| glm4|answer_repeat|gen_after_step_3|L12H21 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| glm4|answer_repeat|gen_after_step_3|L12H21 | ablate_all_steps | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H10 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H10 | ablate_all_steps | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H11 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H11 | ablate_all_steps | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H18 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H18 | ablate_all_steps | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H25 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H25 | ablate_all_steps | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H28 | ablate_anchor_step | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| glm4|answer_target_seeded|gen_after_step_6|L29H28 | ablate_all_steps | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'repeat_answer': 4} |
| deepseek7b|answer_explain|gen_after_step_1|L24H16 | ablate_anchor_step | 4 | 2 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 2} |
| deepseek7b|answer_explain|gen_after_step_1|L24H20 | ablate_anchor_step | 4 | 2 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 2} |
| deepseek7b|answer_explain|gen_after_step_1|L24H20 | ablate_all_steps | 4 | 2 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 2} |
| deepseek7b|answer_list|gen_after_step_1|L24H20 | ablate_anchor_step | 4 | 2 | 0 | 0 | {'echo_then_answer': 2, 'list_answer': 2} | {'other_or_wrong': 2} |
| deepseek7b|answer_list|gen_after_step_1|L24H20 | ablate_all_steps | 4 | 2 | 0 | 0 | {'echo_then_answer': 2, 'list_answer': 2} | {'other_or_wrong': 2} |
