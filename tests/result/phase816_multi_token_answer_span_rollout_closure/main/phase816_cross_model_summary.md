# Phase 816 Multi Token Answer Span Rollout Closure (main)

- Boundary: target phrase must be multi-token; closure is tested by teacher-forced span score and greedy rollout.

## Model Summary

| model | rows | cases | multi-token rows | span-score | rollout | full | contrast cleared | generic cleared | strict step top1 | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 24 | 12 | 24 | 21 | 15 | 15 | 24 | 21 | 14 | `{"contrast_cleared_but_other_span_wins": 3, "span_score_and_rollout_closed": 15, "span_score_closed_rollout_not_closed": 6}` |
| glm4 | 24 | 12 | 24 | 21 | 17 | 17 | 24 | 21 | 14 | `{"contrast_cleared_but_other_span_wins": 3, "span_score_and_rollout_closed": 17, "span_score_closed_rollout_not_closed": 4}` |
| deepseek7b | 24 | 12 | 24 | 12 | 10 | 10 | 23 | 13 | 10 | `{"contrast_cleared_but_other_span_wins": 11, "generic_blocker_span_wins": 1, "span_score_and_rollout_closed": 10, "span_score_closed_rollout_not_closed": 2}` |

## Best Rows

| model | variant | case | target | best target | best non-target | span-score | rollout | full | margin | generated | label |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| qwen3 | exact_choices | p816_gold_precious_metal | `precious metal` | ` precious metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.945 | `precious metal` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_salmon_aquatic_animal | `aquatic animal` | ` aquatic animal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.769 | `aquatic animal` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_heart_body_organ | `body organ` | ` body organ` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.191 | `body organ` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_guitar_musical_instrument | `musical instrument` | ` musical instrument` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.118 | `musical instrument` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_laptop_electronic_device | `electronic device` | ` electronic device` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.063 | `electronic device` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.901 | `living thing` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_oak_tall_tree | `tall tree` | ` tall tree` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.715 | `tall tree` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_chair_household_furniture | `household furniture` | ` household furniture` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.706 | `household furniture` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.697 | `medical worker` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_bus_public_transport | `public transport` | ` public transport` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.667 | `public transport` | `span_score_and_rollout_closed` |
| qwen3 | no_choices | p816_guitar_musical_instrument | `musical instrument` | ` Musical Instrument` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.590 | `Musical Instrument` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_oxygen_chemical_element | `chemical element` | ` chemical element` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.370 | `chemical element` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_triangle_geometric_shape | `geometric shape` | ` geometric shape` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.285 | `geometric shape` | `span_score_and_rollout_closed` |
| qwen3 | no_choices | p816_gold_precious_metal | `precious metal` | ` Precious Metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.537 | `Precious Metal` | `span_score_and_rollout_closed` |
| qwen3 | no_choices | p816_bus_public_transport | `public transport` | ` Public Transport` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 0.373 | `Public Transportation` | `span_score_and_rollout_closed` |
| qwen3 | no_choices | p816_heart_body_organ | `body organ` | ` Body Organ` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 2.954 | `Circulatory System` | `span_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_laptop_electronic_device | `electronic device` | ` Electronic Device` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 1.839 | `Electronics` | `span_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_oxygen_chemical_element | `chemical element` | ` Chemical Element` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 1.688 | `Gas` | `span_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_triangle_geometric_shape | `geometric shape` | ` Geometric Shape` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 1.570 | `Geometry` | `span_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_chair_household_furniture | `household furniture` | ` Household Furniture` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 0.545 | `Furniture` | `span_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | ` Aquatic Animal` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 0.219 | `Fish` | `span_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_oak_tall_tree | `tall tree` | ` Tall Tree` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -3.811 | `Tree` | `contrast_cleared_but_other_span_wins` |
| qwen3 | no_choices | p816_doctor_medical_worker | `medical worker` | ` Medical Worker` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -6.255 | `Medical Professional` | `contrast_cleared_but_other_span_wins` |
| qwen3 | no_choices | p816_cat_living_thing | `living thing` | ` Living Thing` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -6.431 | `Domestic animal` | `contrast_cleared_but_other_span_wins` |
| glm4 | exact_choices | p816_gold_precious_metal | `precious metal` | ` precious metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.692 | `precious metal` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_guitar_musical_instrument | `musical instrument` | ` Musical Instrument` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.591 | `Musical Instrument` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_chair_household_furniture | `household furniture` | ` household furniture` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.263 | `household furniture` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_oxygen_chemical_element | `chemical element` | ` chemical element` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.207 | `chemical element` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_gold_precious_metal | `precious metal` | ` Precious Metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.133 | `Precious Metal` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_salmon_aquatic_animal | `aquatic animal` | ` aquatic animal` | ` public transport`/distractor | 1 | 1 | 1 | 2.813 | `aquatic animal` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` warm color`/distractor | 1 | 1 | 1 | 2.750 | `medical worker` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_laptop_electronic_device | `electronic device` | ` electronic device` | ` hand tool`/distractor | 1 | 1 | 1 | 2.469 | `electronic device` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_triangle_geometric_shape | `geometric shape` | ` Geometric Shape` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.389 | `Geometric shape` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_guitar_musical_instrument | `musical instrument` | ` musical instrument` | ` cold season`/distractor | 1 | 1 | 1 | 2.250 | `musical instrument` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_heart_body_organ | `body organ` | ` body organ` | ` public transport`/contrast | 1 | 1 | 1 | 2.126 | `body organ` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` public transport`/distractor | 1 | 1 | 1 | 2.125 | `living thing` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_oxygen_chemical_element | `chemical element` | ` Chemical Element` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.956 | `Chemical element` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_bus_public_transport | `public transport` | ` public transport` | ` household furniture`/distractor | 1 | 1 | 1 | 1.939 | `public transport` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_triangle_geometric_shape | `geometric shape` | ` geometric shape` | ` warm color`/distractor | 1 | 1 | 1 | 1.813 | `geometric shape` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_bus_public_transport | `public transport` | ` Public Transport` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.595 | `Public transportation` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_oak_tall_tree | `tall tree` | ` tall tree` | ` warm color`/distractor | 1 | 1 | 1 | 0.156 | `tall tree` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | ` Aquatic Animal` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 2.499 | `Freshwater fish` | `span_score_closed_rollout_not_closed` |
| glm4 | no_choices | p816_laptop_electronic_device | `electronic device` | ` Electronic Device` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 2.300 | `Personal Computing Device` | `span_score_closed_rollout_not_closed` |
| glm4 | no_choices | p816_chair_household_furniture | `household furniture` | ` Household Furniture` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 1.530 | `Furniture category` | `span_score_closed_rollout_not_closed` |
| glm4 | no_choices | p816_heart_body_organ | `body organ` | ` Body Organ` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 0.822 | `Human body part` | `span_score_closed_rollout_not_closed` |
| glm4 | no_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -1.042 | `Health Professional` | `contrast_cleared_but_other_span_wins` |
| glm4 | no_choices | p816_oak_tall_tree | `tall tree` | `tall tree` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -1.425 | `Hardwood Tree` | `contrast_cleared_but_other_span_wins` |
| glm4 | no_choices | p816_cat_living_thing | `living thing` | ` Living Thing` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -1.597 | `Pet category` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | exact_choices | p816_chair_household_furniture | `household furniture` | ` household furniture` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.623 | `household furniture` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_oxygen_chemical_element | `chemical element` | ` chemical element` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.597 | `chemical element` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_triangle_geometric_shape | `geometric shape` | ` geometric shape` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.506 | `geometric shape` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_guitar_musical_instrument | `musical instrument` | ` musical instrument` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.135 | `musical instrument` | `span_score_and_rollout_closed` |
| deepseek7b | no_choices | p816_gold_precious_metal | `precious metal` | ` Precious Metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.974 | `Precious Metal` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_gold_precious_metal | `precious metal` | ` precious metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.971 | `precious metal` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.869 | `medical worker` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_laptop_electronic_device | `electronic device` | ` electronic device` | ` body organ`/distractor | 1 | 1 | 1 | 1.781 | `electronic device` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_bus_public_transport | `public transport` | ` public transport` | ` household furniture`/distractor | 1 | 1 | 1 | 1.781 | `public transport` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` public transport`/distractor | 1 | 1 | 1 | 0.907 | `living thing` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_oak_tall_tree | `tall tree` | ` tall tree` | ` warm color`/distractor | 1 | 0 | 0 | 1.188 | `?` | `span_score_closed_rollout_not_closed` |
| deepseek7b | exact_choices | p816_heart_body_organ | `body organ` | ` body organ` | ` public transport`/contrast | 1 | 0 | 0 | 0.719 | `?` | `span_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_triangle_geometric_shape | `geometric shape` | ` Geometric Shape` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -0.003 | `Triangle` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | exact_choices | p816_salmon_aquatic_animal | `aquatic animal` | `aquatic animal` | ` public transport`/distractor | 0 | 0 | 0 | -0.111 | `The correct phrase is "aquatic animal` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_laptop_electronic_device | `electronic device` | `electronic device` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -0.600 | `Personal computer` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_guitar_musical_instrument | `musical instrument` | `Musical Instrument` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -0.744 | `Answer must be one of the following` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | ` Aquatic Animal` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -1.003 | `Salmon is best described as a freshwater fish` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_bus_public_transport | `public transport` | ` public transport` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -1.064 | `__________` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_chair_household_furniture | `household furniture` | ` household furniture` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -1.094 | `furniture` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_oxygen_chemical_element | `chemical element` | `Chemical Element` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -1.765 | `O2` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -3.314 | `___________` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_oak_tall_tree | `tall tree` | ` tall tree` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -4.419 | `Tree` | `generic_blocker_span_wins` |
| deepseek7b | no_choices | p816_heart_body_organ | `body organ` | ` Body Organ` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -4.551 | `___________` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_doctor_medical_worker | `medical worker` | ` Medical Worker` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -6.330 | `</think>` | `contrast_cleared_but_other_span_wins` |
