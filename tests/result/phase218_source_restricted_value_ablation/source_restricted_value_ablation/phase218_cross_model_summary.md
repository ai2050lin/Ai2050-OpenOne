# Phase 218 source restricted value ablation

Headset count: 6
Filter rows: 79
Rollout rows: 180
Source value rows: 1920
Total damage match loss: 0
Total repair match gain: 0

| headset | condition | success | drift | damage | repair | success outputs | drift outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| qwen3_explain_route_set | block_answer_slot | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_explain_route_set | block_instruction_to_answer | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'echo_then_answer': 3, 'other_or_wrong': 1} |
| qwen3_explain_route_set | block_trigger:any | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'echo_then_answer': 3, 'other_or_wrong': 1} |
| qwen3_explain_route_set | block_question_prefix | 4 | 4 | 0 | 0 | {'explain_answer': 4} | {'other_or_wrong': 4} |
| qwen3_repeat_route_set | block_answer_slot | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| qwen3_repeat_route_set | block_instruction_to_answer | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| qwen3_repeat_route_set | block_trigger:any | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| qwen3_repeat_route_set | block_question_prefix | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| glm4_explain_competition_route_set | block_answer_slot | 4 | 2 | 0 | 0 | {'explain_answer': 4} | {'repeat_answer': 2} |
| glm4_explain_competition_route_set | block_instruction_to_answer | 4 | 2 | 0 | 0 | {'explain_answer': 4} | {'repeat_answer': 2} |
| glm4_explain_competition_route_set | block_trigger:any | 4 | 2 | 0 | 0 | {'explain_answer': 4} | {'repeat_answer': 2} |
| glm4_explain_competition_route_set | block_question_prefix | 4 | 2 | 0 | 0 | {'explain_answer': 4} | {'repeat_answer': 2} |
| glm4_repeat_route_set | block_answer_slot | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| glm4_repeat_route_set | block_instruction_to_answer | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| glm4_repeat_route_set | block_trigger:any | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| glm4_repeat_route_set | block_question_prefix | 4 | 4 | 0 | 0 | {'repeat_answer': 4} | {'list_answer': 4} |
| deepseek7b_explain_l24_route_set | block_answer_slot | 2 | 2 | 0 | 0 | {'explain_answer': 2} | {'other_or_wrong': 2} |
| deepseek7b_explain_l24_route_set | block_instruction_to_answer | 2 | 2 | 0 | 0 | {'explain_answer': 2} | {'other_or_wrong': 2} |
| deepseek7b_explain_l24_route_set | block_trigger:any | 2 | 2 | 0 | 0 | {'explain_answer': 2} | {'other_or_wrong': 2} |
| deepseek7b_explain_l24_route_set | block_question_prefix | 2 | 2 | 0 | 0 | {'explain_answer': 2} | {'other_or_wrong': 2} |
| deepseek7b_list_l24_route_set | block_answer_slot | 0 | 2 | 0 | 0 | {} | {'other_or_wrong': 2} |
| deepseek7b_list_l24_route_set | block_instruction_to_answer | 0 | 2 | 0 | 0 | {} | {'other_or_wrong': 2} |
| deepseek7b_list_l24_route_set | block_trigger:any | 0 | 2 | 0 | 0 | {} | {'other_or_wrong': 2} |
| deepseek7b_list_l24_route_set | block_question_prefix | 0 | 2 | 0 | 0 | {} | {'other_or_wrong': 2} |
