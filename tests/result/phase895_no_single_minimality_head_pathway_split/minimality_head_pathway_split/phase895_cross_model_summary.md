# Phase 895 no-single closure minimality and multi-head pathway split

## Overall

- models: qwen3, glm4, deepseek7b
- selected_conditions: 46
- output_minimality_rows: 112
- output_condition_rows: 46
- output_head_split_rows: 200
- output_rollout_rows: 176
- any_alternative_pair_closure: 0
- any_single_axis_closure: 11
- focus_closure_from_open: 22
- head_blocker_damage_gt_0: 40
- head_closure_lost: 40
- head_target_damage_gt_0_25: 38
- known_axis_minimal_candidate: 11
- model_u_not_required_for_focus_closure: 22
- rollout_answer_like_no_echo: 121
- rollout_class_hit: 150
- rollout_class_lost_vs_none: 26
- rollout_object_echo_added_vs_none: 2
- rollout_protocol_drift_added_vs_none: 0

## Subset groups

| model | subset | rows | closure | mean lift | mean blocker reduction | top roles |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| deepseek7b | L26C8587+L27C15369 | 11 | 11 | 1.830 | 2.364 | {"none": 11} |
| deepseek7b | L26C8587+L27C15369+L27C16651 | 11 | 11 | 1.818 | 2.364 | {"none": 11} |
| qwen3 | L31C2257 | 11 | 11 | 0.739 | 2.364 | {"none": 11} |
| deepseek7b | L27C15369+L27C16651 | 11 | 0 | 0.801 | 1.000 | {"format_punct": 5, "other_blocker": 6} |
| deepseek7b | L27C15369 | 11 | 0 | 0.795 | 1.091 | {"format_punct": 5, "other_blocker": 6} |
| deepseek7b | L26C8587+L27C16651 | 11 | 0 | 0.472 | 0.455 | {"format_punct": 5, "other_blocker": 6} |
| deepseek7b | L26C8587 | 11 | 0 | 0.466 | 0.545 | {"format_punct": 5, "other_blocker": 6} |
| deepseek7b | L27C16651 | 11 | 0 | 0.017 | 0.000 | {"format_punct": 5, "other_blocker": 6} |
| glm4 | L31C6437 | 24 | 0 | 0.003 | 0.000 | {"none": 22, "object_echo": 2} |

## Head groups

| model | head set | rows | closure lost | target damage | blocker damage | mean target damage | mean blocker damage | labels |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3 | L31H19+L31H26+L31H30+L31H12+L31H17 | 11 | 10 | 7 | 10 | 0.420 | 1.455 | {"blocker_boundary_candidate": 4, "target_and_blocker_boundary_candidate": 6, "target_lift_damage_candidate": 1} |
| qwen3 | L31H19 | 11 | 6 | 0 | 6 | 0.159 | 0.636 | {"blocker_boundary_candidate": 6, "weak_or_no_damage": 5} |
| qwen3 | L31H26 | 11 | 6 | 1 | 6 | 0.136 | 0.636 | {"blocker_boundary_candidate": 5, "target_and_blocker_boundary_candidate": 1, "weak_or_no_damage": 5} |
| deepseek7b | L26H3+L26H7+L26H11+L26H14 | 11 | 4 | 8 | 4 | 0.511 | 0.364 | {"target_and_blocker_boundary_candidate": 4, "target_lift_damage_candidate": 4, "weak_or_no_damage": 3} |
| qwen3 | L31H17 | 11 | 4 | 2 | 4 | 0.170 | 0.455 | {"blocker_boundary_candidate": 4, "target_lift_damage_candidate": 2, "weak_or_no_damage": 5} |
| qwen3 | L31H30 | 11 | 3 | 0 | 3 | 0.023 | 0.364 | {"blocker_boundary_candidate": 3, "weak_or_no_damage": 8} |
| deepseek7b | L26H7+L26H11 | 11 | 2 | 8 | 2 | 0.347 | 0.182 | {"target_and_blocker_boundary_candidate": 2, "target_lift_damage_candidate": 6, "weak_or_no_damage": 3} |
| qwen3 | L31H12 | 11 | 2 | 1 | 2 | 0.034 | 0.182 | {"blocker_boundary_candidate": 1, "target_and_blocker_boundary_candidate": 1, "weak_or_no_damage": 9} |
| deepseek7b | L26H3+L26H7 | 11 | 1 | 4 | 1 | 0.233 | 0.091 | {"target_and_blocker_boundary_candidate": 1, "target_lift_damage_candidate": 3, "weak_or_no_damage": 7} |
| deepseek7b | L26H3+L26H11 | 11 | 1 | 3 | 1 | 0.222 | 0.091 | {"target_and_blocker_boundary_candidate": 1, "target_lift_damage_candidate": 2, "weak_or_no_damage": 8} |
| deepseek7b | L26H7 | 11 | 1 | 2 | 1 | 0.170 | 0.091 | {"blocker_boundary_candidate": 1, "target_lift_damage_candidate": 2, "weak_or_no_damage": 8} |
| deepseek7b | L26H11 | 11 | 0 | 0 | 0 | 0.176 | 0.000 | {"weak_or_no_damage": 11} |
| deepseek7b | L26H14 | 11 | 0 | 1 | 0 | 0.119 | 0.000 | {"target_lift_damage_candidate": 1, "weak_or_no_damage": 10} |
| deepseek7b | L26H3 | 11 | 0 | 1 | 0 | 0.062 | 0.000 | {"target_lift_damage_candidate": 1, "weak_or_no_damage": 10} |
| qwen3 | none | 11 | 0 | 0 | 0 | 0.000 | 0.000 | {"none_control": 11} |
| glm4 | none | 24 | 0 | 0 | 0 | 0.000 | 0.000 | {"none_control": 24} |
| deepseek7b | none | 11 | 0 | 0 | 0 | 0.000 | 0.000 | {"none_control": 11} |

## Rollout groups

| model | head set | rows | class hit | answer-like | object echo | protocol drift | class lost | echo added | drift added |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3 | L31H19+L31H26+L31H30+L31H12+L31H17 | 11 | 6 | 3 | 4 | 0 | 5 | 1 | 0 |
| deepseek7b | L26H3+L26H7+L26H11+L26H14 | 11 | 7 | 7 | 0 | 0 | 4 | 0 | 0 |
| qwen3 | L31H30 | 11 | 8 | 5 | 3 | 0 | 3 | 0 | 0 |
| qwen3 | L31H12 | 11 | 8 | 5 | 4 | 0 | 3 | 1 | 0 |
| qwen3 | L31H19 | 11 | 9 | 6 | 3 | 0 | 2 | 0 | 0 |
| qwen3 | L31H26 | 11 | 9 | 6 | 3 | 0 | 2 | 0 | 0 |
| qwen3 | L31H17 | 11 | 9 | 6 | 3 | 0 | 2 | 0 | 0 |
| deepseek7b | L26H7+L26H11 | 11 | 9 | 8 | 1 | 0 | 2 | 0 | 0 |
| deepseek7b | L26H7 | 11 | 10 | 9 | 1 | 0 | 1 | 0 | 0 |
| deepseek7b | L26H3+L26H7 | 11 | 10 | 9 | 1 | 0 | 1 | 0 | 0 |
| deepseek7b | L26H3+L26H11 | 11 | 10 | 9 | 1 | 0 | 1 | 0 | 0 |
| qwen3 | none | 11 | 11 | 8 | 3 | 0 | 0 | 0 | 0 |
| deepseek7b | none | 11 | 11 | 10 | 1 | 0 | 0 | 0 | 0 |
| deepseek7b | L26H3 | 11 | 11 | 10 | 1 | 0 | 0 | 0 | 0 |
| deepseek7b | L26H11 | 11 | 11 | 10 | 1 | 0 | 0 | 0 | 0 |
| deepseek7b | L26H14 | 11 | 11 | 10 | 1 | 0 | 0 | 0 | 0 |
