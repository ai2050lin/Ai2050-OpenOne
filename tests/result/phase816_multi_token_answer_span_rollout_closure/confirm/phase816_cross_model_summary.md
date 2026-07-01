# Phase 816 Multi Token Answer Span Rollout Closure (confirm)

- Boundary: target phrase must be multi-token; closure is tested by teacher-forced span score and greedy rollout.

## Model Summary

| model | rows | cases | multi-token rows | span-score | rollout | full | contrast cleared | generic cleared | strict step top1 | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 40 | 20 | 40 | 36 | 23 | 23 | 40 | 36 | 22 | `{"contrast_cleared_but_other_span_wins": 4, "span_score_and_rollout_closed": 23, "span_score_closed_rollout_not_closed": 13}` |
| glm4 | 40 | 20 | 40 | 36 | 27 | 27 | 40 | 36 | 22 | `{"contrast_cleared_but_other_span_wins": 4, "span_score_and_rollout_closed": 27, "span_score_closed_rollout_not_closed": 9}` |
| deepseek7b | 40 | 20 | 40 | 18 | 14 | 14 | 38 | 22 | 15 | `{"contrast_cleared_but_other_span_wins": 20, "distractor_or_other_span_wins": 1, "generic_blocker_span_wins": 1, "span_score_and_rollout_closed": 14, "span_score_closed_rollout_not_closed": 4}` |

## Best Rows

| model | variant | case | target | best target | best non-target | span-score | rollout | full | margin | generated | label |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| qwen3 | exact_choices | p816_gold_precious_metal | `precious metal` | ` precious metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.934 | `precious metal` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_salmon_aquatic_animal | `aquatic animal` | ` aquatic animal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.827 | `aquatic animal` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_apple_edible_fruit | `edible fruit` | ` edible fruit` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.574 | `edible fruit` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_cactus_desert_plant | `desert plant` | ` desert plant` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.331 | `desert plant` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_heart_body_organ | `body organ` | ` body organ` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.228 | `body organ` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_guitar_musical_instrument | `musical instrument` | ` musical instrument` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.152 | `musical instrument` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_laptop_electronic_device | `electronic device` | ` electronic device` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 5.088 | `electronic device` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.940 | `living thing` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_winter_cold_season | `cold season` | ` cold season` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.920 | `cold season` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_spoon_eating_utensil | `eating utensil` | ` eating utensil` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.917 | `eating utensil` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_oak_tall_tree | `tall tree` | ` tall tree` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.792 | `tall tree` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_chair_household_furniture | `household furniture` | ` household furniture` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.748 | `household furniture` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_bus_public_transport | `public transport` | ` public transport` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.732 | `public transport` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.729 | `medical worker` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_hammer_hand_tool | `hand tool` | ` hand tool` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.697 | `hand tool` | `span_score_and_rollout_closed` |
| qwen3 | no_choices | p816_guitar_musical_instrument | `musical instrument` | ` Musical Instrument` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.578 | `Musical Instrument` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_carrot_root_vegetable | `root vegetable` | ` root vegetable` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.461 | `root vegetable` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_rose_flowering_plant | `flowering plant` | ` flowering plant` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.375 | `flowering plant` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_oxygen_chemical_element | `chemical element` | ` chemical element` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.351 | `chemical element` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_red_warm_color | `warm color` | ` warm color` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.280 | `warm color` | `span_score_and_rollout_closed` |
| qwen3 | exact_choices | p816_triangle_geometric_shape | `geometric shape` | ` geometric shape` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 4.265 | `geometric shape` | `span_score_and_rollout_closed` |
| qwen3 | no_choices | p816_gold_precious_metal | `precious metal` | ` Precious Metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.533 | `Precious Metal` | `span_score_and_rollout_closed` |
| qwen3 | no_choices | p816_bus_public_transport | `public transport` | ` Public Transport` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 0.405 | `Public Transportation` | `span_score_and_rollout_closed` |
| qwen3 | no_choices | p816_winter_cold_season | `cold season` | ` Cold Season` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 3.002 | `Cold Weather` | `span_score_closed_rollout_not_closed` |
| glm4 | exact_choices | p816_gold_precious_metal | `precious metal` | ` precious metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.683 | `precious metal` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_guitar_musical_instrument | `musical instrument` | ` Musical Instrument` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.592 | `Musical Instrument` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_chair_household_furniture | `household furniture` | ` household furniture` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.285 | `household furniture` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_apple_edible_fruit | `edible fruit` | ` edible fruit` | ` warm color`/distractor | 1 | 1 | 1 | 3.251 | `edible fruit` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_oxygen_chemical_element | `chemical element` | ` chemical element` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.219 | `chemical element` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_gold_precious_metal | `precious metal` | ` Precious Metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 3.119 | `Precious Metal` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_hammer_hand_tool | `hand tool` | ` hand tool` | ` body organ`/distractor | 1 | 1 | 1 | 3.097 | `hand tool` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_salmon_aquatic_animal | `aquatic animal` | ` aquatic animal` | ` public transport`/distractor | 1 | 1 | 1 | 2.813 | `aquatic animal` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_cactus_desert_plant | `desert plant` | ` desert plant` | ` hand tool`/contrast | 1 | 1 | 1 | 2.813 | `desert plant` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` warm color`/distractor | 1 | 1 | 1 | 2.719 | `medical worker` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_spoon_eating_utensil | `eating utensil` | ` Eating Utensil` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.692 | `Eating utensil` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_spoon_eating_utensil | `eating utensil` | ` eating utensil` | ` body organ`/distractor | 1 | 1 | 1 | 2.615 | `eating utensil` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_rose_flowering_plant | `flowering plant` | ` flowering plant` | ` warm color`/distractor | 1 | 1 | 1 | 2.500 | `flowering plant` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_laptop_electronic_device | `electronic device` | ` electronic device` | ` hand tool`/distractor | 1 | 1 | 1 | 2.438 | `electronic device` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_triangle_geometric_shape | `geometric shape` | ` Geometric Shape` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.376 | `Geometric shape` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_guitar_musical_instrument | `musical instrument` | ` musical instrument` | ` cold season`/distractor | 1 | 1 | 1 | 2.281 | `musical instrument` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_rose_flowering_plant | `flowering plant` | ` Flowering Plant` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.278 | `Flowering plant` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` public transport`/distractor | 1 | 1 | 1 | 2.125 | `living thing` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_heart_body_organ | `body organ` | ` body organ` | ` public transport`/contrast | 1 | 1 | 1 | 2.095 | `body organ` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_winter_cold_season | `cold season` | ` cold season` | ` hand tool`/distractor | 1 | 1 | 1 | 2.031 | `cold season` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_bus_public_transport | `public transport` | ` public transport` | ` household furniture`/distractor | 1 | 1 | 1 | 2.001 | `public transport` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_oxygen_chemical_element | `chemical element` | ` Chemical Element` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.926 | `Chemical element` | `span_score_and_rollout_closed` |
| glm4 | exact_choices | p816_triangle_geometric_shape | `geometric shape` | ` geometric shape` | ` warm color`/distractor | 1 | 1 | 1 | 1.782 | `geometric shape` | `span_score_and_rollout_closed` |
| glm4 | no_choices | p816_bus_public_transport | `public transport` | ` Public Transport` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.556 | `Public transportation` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_chair_household_furniture | `household furniture` | ` household furniture` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.667 | `household furniture` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_apple_edible_fruit | `edible fruit` | ` edible fruit` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.641 | `edible fruit` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_oxygen_chemical_element | `chemical element` | ` chemical element` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.563 | `chemical element` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_triangle_geometric_shape | `geometric shape` | ` geometric shape` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.503 | `geometric shape` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_guitar_musical_instrument | `musical instrument` | ` musical instrument` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.159 | `musical instrument` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_gold_precious_metal | `precious metal` | ` precious metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 2.040 | `precious metal` | `span_score_and_rollout_closed` |
| deepseek7b | no_choices | p816_gold_precious_metal | `precious metal` | ` Precious Metal` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.988 | `Precious Metal` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_rose_flowering_plant | `flowering plant` | ` flowering plant` | ` warm color`/distractor | 1 | 1 | 1 | 1.969 | `flowering plant` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_laptop_electronic_device | `electronic device` | ` electronic device` | ` body organ`/distractor | 1 | 1 | 1 | 1.906 | `electronic device` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_doctor_medical_worker | `medical worker` | ` medical worker` | ` I don't know`/generic_blocker | 1 | 1 | 1 | 1.846 | `medical worker` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_bus_public_transport | `public transport` | ` public transport` | ` household furniture`/distractor | 1 | 1 | 1 | 1.687 | `public transport` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_winter_cold_season | `cold season` | ` cold season` | ` hand tool`/distractor | 1 | 1 | 1 | 1.627 | `cold season` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_spoon_eating_utensil | `eating utensil` | ` eating utensil` | ` body organ`/distractor | 1 | 1 | 1 | 1.442 | `eating utensil` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_cat_living_thing | `living thing` | ` living thing` | ` public transport`/distractor | 1 | 1 | 1 | 0.844 | `living thing` | `span_score_and_rollout_closed` |
| deepseek7b | exact_choices | p816_oak_tall_tree | `tall tree` | ` tall tree` | ` warm color`/distractor | 1 | 0 | 0 | 1.125 | `?` | `span_score_closed_rollout_not_closed` |
| deepseek7b | exact_choices | p816_heart_body_organ | `body organ` | ` body organ` | ` public transport`/contrast | 1 | 0 | 0 | 0.688 | `?` | `span_score_closed_rollout_not_closed` |
| deepseek7b | exact_choices | p816_carrot_root_vegetable | `root vegetable` | ` root vegetable` | ` body organ`/distractor | 1 | 0 | 0 | 0.655 | `?` | `span_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_cactus_desert_plant | `desert plant` | ` Desert Plant` | ` I don't know`/generic_blocker | 1 | 0 | 0 | 0.364 | `[Answer]` | `span_score_closed_rollout_not_closed` |
| deepseek7b | exact_choices | p816_salmon_aquatic_animal | `aquatic animal` | `aquatic animal` | ` public transport`/distractor | 0 | 0 | 0 | -0.032 | `The correct phrase is "aquatic animal` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | exact_choices | p816_hammer_hand_tool | `hand tool` | ` hand tool` | ` body organ`/distractor | 0 | 0 | 0 | -0.056 | `hammer is a hand tool` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_triangle_geometric_shape | `geometric shape` | ` Geometric Shape` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -0.098 | `Triangle` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_spoon_eating_utensil | `eating utensil` | ` eating utensil` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -0.110 | `___________` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | exact_choices | p816_cactus_desert_plant | `desert plant` | ` desert plant` | ` body organ`/distractor | 0 | 0 | 0 | -0.250 | `body organ` | `contrast_cleared_but_other_span_wins` |
| deepseek7b | no_choices | p816_apple_edible_fruit | `edible fruit` | ` Edible Fruit` | ` I don't know`/generic_blocker | 0 | 0 | 0 | -0.532 | `__________` | `contrast_cleared_but_other_span_wins` |
