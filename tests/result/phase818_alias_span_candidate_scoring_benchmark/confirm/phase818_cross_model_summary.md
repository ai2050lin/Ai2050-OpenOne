# Phase 818 Alias Span Candidate Scoring Benchmark (confirm)

- Boundary: target answer is evaluated as an alias class, while near-miss, wrong, and generic spans remain explicit competitors.

## Model Summary

| model | rows | exact score | alias score | exact rollout | alias rollout | exact full | alias full | near cleared | wrong cleared | generic cleared | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 40 | 24 | 34 | 22 | 32 | 22 | 31 | 34 | 40 | 39 | `{"alias_class_closes_exact_phrase_fails": 9, "alias_rollout_closed_score_not_closed": 1, "alias_score_and_rollout_closed": 22, "alias_score_closed_rollout_not_closed": 3, "near_miss_span_wins": 5}` |
| glm4 | 40 | 28 | 36 | 26 | 33 | 26 | 33 | 36 | 40 | 39 | `{"alias_class_closes_exact_phrase_fails": 7, "alias_score_and_rollout_closed": 26, "alias_score_closed_rollout_not_closed": 3, "near_miss_span_wins": 4}` |
| deepseek7b | 40 | 18 | 26 | 14 | 17 | 14 | 17 | 38 | 37 | 29 | `{"alias_class_closes_exact_phrase_fails": 3, "alias_score_and_rollout_closed": 14, "alias_score_closed_rollout_not_closed": 9, "generic_blocker_span_wins": 11, "wrong_span_wins": 3}` |

## Prompt Variant Summary

| model | prompt | n | alias score | alias rollout | alias full | generation classes | labels |
|---|---|---:|---:|---:|---:|---|---|
| qwen3 | exact_choices | 20 | 20 | 20 | 20 | `{"target_alias": 20}` | `{"alias_score_and_rollout_closed": 20}` |
| qwen3 | no_choices | 20 | 14 | 12 | 11 | `{"near_miss": 5, "other": 3, "target_alias": 12}` | `{"alias_class_closes_exact_phrase_fails": 9, "alias_rollout_closed_score_not_closed": 1, "alias_score_and_rollout_closed": 2, "alias_score_closed_rollout_not_closed": 3, "near_miss_span_wins": 5}` |
| glm4 | exact_choices | 20 | 20 | 20 | 20 | `{"target_alias": 20}` | `{"alias_score_and_rollout_closed": 20}` |
| glm4 | no_choices | 20 | 16 | 13 | 13 | `{"near_miss": 4, "other": 3, "target_alias": 13}` | `{"alias_class_closes_exact_phrase_fails": 7, "alias_score_and_rollout_closed": 6, "alias_score_closed_rollout_not_closed": 3, "near_miss_span_wins": 4}` |
| deepseek7b | exact_choices | 20 | 17 | 13 | 13 | `{"other": 6, "target_alias": 13, "wrong": 1}` | `{"alias_score_and_rollout_closed": 13, "alias_score_closed_rollout_not_closed": 4, "wrong_span_wins": 3}` |
| deepseek7b | no_choices | 20 | 9 | 4 | 4 | `{"other": 16, "target_alias": 4}` | `{"alias_class_closes_exact_phrase_fails": 3, "alias_score_and_rollout_closed": 1, "alias_score_closed_rollout_not_closed": 5, "generic_blocker_span_wins": 11}` |

## First Failure Rows

| model | prompt | case | target | generated | gen class | best alias | best non-alias | margin | label |
|---|---|---|---|---|---|---|---|---:|---|
| qwen3 | no_choices | p816_hammer_hand_tool | `hand tool` | `Tools` | `other` | ` Hand Tool` | ` Hardware`/near_miss | 1.076 | `alias_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_heart_body_organ | `body organ` | `Circulatory System` | `other` | ` Body Organ` | ` Body Part`/near_miss | 1.375 | `alias_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_red_warm_color | `warm color` | `Color` | `near_miss` | ` Warm Color` | ` Color`/near_miss | -5.858 | `near_miss_span_wins` |
| qwen3 | no_choices | p816_carrot_root_vegetable | `root vegetable` | `Vegetables` | `other` | `Vegetable` | ` I don't know`/generic_blocker | 3.382 | `alias_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_laptop_electronic_device | `electronic device` | `Electronics` | `near_miss` | ` Electronic Device` | ` Electronics`/near_miss | -1.237 | `near_miss_span_wins` |
| qwen3 | no_choices | p816_triangle_geometric_shape | `geometric shape` | `Geometry` | `near_miss` | ` Polygon` | ` Geometry`/near_miss | -1.250 | `near_miss_span_wins` |
| qwen3 | no_choices | p816_winter_cold_season | `cold season` | `Cold Weather` | `near_miss` | ` Cold Season` | ` Cold Weather`/near_miss | -1.375 | `near_miss_span_wins` |
| qwen3 | no_choices | p816_oxygen_chemical_element | `chemical element` | `Gas` | `near_miss` | ` Element` | ` Gas`/near_miss | -0.750 | `near_miss_span_wins` |
| qwen3 | no_choices | p816_cactus_desert_plant | `desert plant` | `Cactus plants` | `target_alias` | ` Cactus` | ` Succulent`/near_miss | -0.197 | `alias_rollout_closed_score_not_closed` |
| glm4 | no_choices | p816_cat_living_thing | `living thing` | `Pet category` | `near_miss` | ` Mammal` | ` Pet`/near_miss | -0.053 | `near_miss_span_wins` |
| glm4 | no_choices | p816_heart_body_organ | `body organ` | `Human body part` | `near_miss` | ` Body Organ` | ` Body Part`/near_miss | -1.063 | `near_miss_span_wins` |
| glm4 | no_choices | p816_red_warm_color | `warm color` | `Color category` | `near_miss` | ` Red Color` | ` Color`/near_miss | -2.123 | `near_miss_span_wins` |
| glm4 | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | `Freshwater fish` | `other` | ` Aquatic Animal` | ` Seafood`/near_miss | 0.798 | `alias_score_closed_rollout_not_closed` |
| glm4 | no_choices | p816_laptop_electronic_device | `electronic device` | `Personal Computing Device` | `other` | ` Personal Computer` | `Electronics`/near_miss | 1.181 | `alias_score_closed_rollout_not_closed` |
| glm4 | no_choices | p816_winter_cold_season | `cold season` | `Seasonal weather patterns` | `other` | ` cold season` | ` cold weather`/near_miss | 0.656 | `alias_score_closed_rollout_not_closed` |
| glm4 | no_choices | p816_cactus_desert_plant | `desert plant` | `Plant Life Cycle` | `near_miss` | ` Succulent Plant` | ` Plant`/near_miss | -0.016 | `near_miss_span_wins` |
| deepseek7b | no_choices | p816_cat_living_thing | `living thing` | `___________` | `other` | ` Mammal` | ` I don't know`/generic_blocker | 1.745 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | exact_choices | p816_hammer_hand_tool | `hand tool` | `hammer is a hand tool` | `other` | ` hand tool` | ` body organ`/wrong | 0.022 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_hammer_hand_tool | `hand tool` | `_____` | `other` | ` tool` | ` I don't know`/generic_blocker | -0.583 | `generic_blocker_span_wins` |
| deepseek7b | no_choices | p816_bus_public_transport | `public transport` | `__________` | `other` | ` Public Transportation` | ` I don't know`/generic_blocker | -0.412 | `generic_blocker_span_wins` |
| deepseek7b | no_choices | p816_guitar_musical_instrument | `musical instrument` | `Answer must be one of the following` | `other` | `Musical Instrument` | ` I don't know`/generic_blocker | -0.744 | `generic_blocker_span_wins` |
| deepseek7b | no_choices | p816_rose_flowering_plant | `flowering plant` | `___________` | `other` | ` flowering plant` | ` I don't know`/generic_blocker | -1.134 | `generic_blocker_span_wins` |
| deepseek7b | no_choices | p816_apple_edible_fruit | `edible fruit` | `__________` | `other` | ` fruit` | ` I don't know`/generic_blocker | -0.230 | `generic_blocker_span_wins` |
| deepseek7b | exact_choices | p816_heart_body_organ | `body organ` | `?` | `other` | ` body organ` | ` public transport`/wrong | 0.688 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_heart_body_organ | `body organ` | `___________` | `other` | ` Organ` | ` I don't know`/generic_blocker | -1.670 | `generic_blocker_span_wins` |
| deepseek7b | exact_choices | p816_red_warm_color | `warm color` | `?` | `other` | ` warm color` | ` body organ`/wrong | -0.625 | `wrong_span_wins` |
| deepseek7b | no_choices | p816_red_warm_color | `warm color` | `[Answer]` | `other` | ` Warm Color` | ` I don't know`/generic_blocker | -3.046 | `generic_blocker_span_wins` |
| deepseek7b | exact_choices | p816_salmon_aquatic_animal | `aquatic animal` | `The correct phrase is "aquatic animal` | `other` | `aquatic animal` | ` public transport`/wrong | -0.032 | `wrong_span_wins` |
| deepseek7b | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | `Salmon is best described as a freshwater fish` | `other` | ` Aquatic Animal` | ` I don't know`/generic_blocker | -0.920 | `generic_blocker_span_wins` |
| deepseek7b | exact_choices | p816_oak_tall_tree | `tall tree` | `?` | `other` | ` tall tree` | ` warm color`/wrong | 1.166 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | exact_choices | p816_carrot_root_vegetable | `root vegetable` | `?` | `other` | ` root vegetable` | ` body organ`/wrong | 0.614 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_carrot_root_vegetable | `root vegetable` | `___________` | `other` | `vegetable` | ` I don't know`/generic_blocker | -1.130 | `generic_blocker_span_wins` |
| deepseek7b | no_choices | p816_spoon_eating_utensil | `eating utensil` | `___________` | `other` | ` kitchen utensil` | ` I don't know`/generic_blocker | 0.088 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_triangle_geometric_shape | `geometric shape` | `Triangle` | `other` | ` polygon` | ` I don't know`/generic_blocker | 0.802 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_winter_cold_season | `cold season` | `winter` | `other` | ` Season` | ` I don't know`/generic_blocker | 0.737 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_oxygen_chemical_element | `chemical element` | `O2` | `other` | ` Element` | ` I don't know`/generic_blocker | -0.805 | `generic_blocker_span_wins` |
| deepseek7b | exact_choices | p816_cactus_desert_plant | `desert plant` | `body organ` | `wrong` | ` desert plant` | ` body organ`/wrong | -0.250 | `wrong_span_wins` |
| deepseek7b | no_choices | p816_cactus_desert_plant | `desert plant` | `[Answer]` | `other` | ` Cactus` | ` Plant`/near_miss | 0.155 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_doctor_medical_worker | `medical worker` | `</think>` | `other` | ` Health Professional` | ` I don't know`/generic_blocker | -0.397 | `generic_blocker_span_wins` |
