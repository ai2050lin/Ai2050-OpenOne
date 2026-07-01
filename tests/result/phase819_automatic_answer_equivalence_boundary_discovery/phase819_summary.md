# Phase 819 Automatic Answer Equivalence Boundary Discovery

- Source: offline reanalysis of Phase 816-818 generated phrases and Phase 818 candidate-score rows.
- Boundary: no model loading; this phase fixes strict / medium / loose answer-boundary standards before returning to causal localization.

## Primary Confirm Summary

| model/prompt | n | strict rollout | medium rollout | loose rollout | strict full | medium full | loose full | boundary classes | score closures |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| deepseek7b/exact_choices | 20 | 13 | 13 | 13 | 13 | 13 | 13 | `{"target_equivalent": 13, "unknown_other": 1, "format_echo": 5, "wrong": 1}` | `{"strict_score": 17, "medium_score": 17, "loose_score": 17}` |
| deepseek7b/no_choices | 20 | 4 | 4 | 4 | 4 | 4 | 4 | `{"format_echo": 12, "target_equivalent": 4, "unknown_other": 4}` | `{"strict_score": 9, "medium_score": 9, "loose_score": 9}` |
| glm4/exact_choices | 20 | 20 | 20 | 20 | 20 | 20 | 20 | `{"target_equivalent": 20}` | `{"strict_score": 20, "medium_score": 20, "loose_score": 20}` |
| glm4/no_choices | 20 | 13 | 13 | 17 | 13 | 13 | 16 | `{"broad_near_miss": 4, "target_equivalent": 13, "unknown_other": 3}` | `{"strict_score": 16, "medium_score": 18, "loose_score": 20}` |
| qwen3/exact_choices | 20 | 20 | 20 | 20 | 20 | 20 | 20 | `{"target_equivalent": 20}` | `{"strict_score": 20, "medium_score": 20, "loose_score": 20}` |
| qwen3/no_choices | 20 | 14 | 16 | 19 | 13 | 15 | 18 | `{"target_equivalent": 14, "unknown_other": 1, "close_near_miss": 2, "broad_near_miss": 3}` | `{"strict_score": 14, "medium_score": 17, "loose_score": 20}` |

## Phrase Boundary Distribution

- Total observations: 612
- Unique generated phrase aggregates: 76
- Boundary classes: `{"target_equivalent": 447, "unknown_other": 45, "broad_near_miss": 33, "close_near_miss": 9, "format_echo": 75, "wrong": 3}`

## Most Frequent Phrase Aggregates

| case | target | phrase | n | class | models | prompts |
|---|---|---|---:|---|---|---|
| p816_gold_precious_metal | `precious metal` | `precious metal` | 36 | `target_equivalent` | `{"qwen3": 12, "glm4": 12, "deepseek7b": 12}` | `{"exact_choices": 18, "no_choices": 18}` |
| p816_guitar_musical_instrument | `musical instrument` | `musical instrument` | 30 | `target_equivalent` | `{"qwen3": 12, "glm4": 12, "deepseek7b": 6}` | `{"exact_choices": 18, "no_choices": 12}` |
| p816_cat_living_thing | `living thing` | `living thing` | 27 | `target_equivalent` | `{"qwen3": 9, "glm4": 9, "deepseek7b": 9}` | `{"exact_choices": 27}` |
| p816_doctor_medical_worker | `medical worker` | `medical worker` | 27 | `target_equivalent` | `{"qwen3": 9, "glm4": 9, "deepseek7b": 9}` | `{"exact_choices": 27}` |
| p816_oxygen_chemical_element | `chemical element` | `chemical element` | 24 | `target_equivalent` | `{"qwen3": 6, "glm4": 12, "deepseek7b": 6}` | `{"exact_choices": 18, "no_choices": 6}` |
| p816_triangle_geometric_shape | `geometric shape` | `geometric shape` | 24 | `target_equivalent` | `{"qwen3": 6, "glm4": 12, "deepseek7b": 6}` | `{"exact_choices": 18, "no_choices": 6}` |
| p816_spoon_eating_utensil | `eating utensil` | `eating utensil` | 21 | `target_equivalent` | `{"qwen3": 6, "glm4": 9, "deepseek7b": 6}` | `{"exact_choices": 18, "no_choices": 3}` |
| p816_apple_edible_fruit | `edible fruit` | `edible fruit` | 18 | `target_equivalent` | `{"qwen3": 6, "glm4": 6, "deepseek7b": 6}` | `{"exact_choices": 18}` |
| p816_bus_public_transport | `public transport` | `public transport` | 18 | `target_equivalent` | `{"qwen3": 6, "glm4": 6, "deepseek7b": 6}` | `{"exact_choices": 18}` |
| p816_chair_household_furniture | `household furniture` | `household furniture` | 18 | `target_equivalent` | `{"qwen3": 6, "glm4": 6, "deepseek7b": 6}` | `{"exact_choices": 18}` |
| p816_laptop_electronic_device | `electronic device` | `electronic device` | 18 | `target_equivalent` | `{"qwen3": 6, "glm4": 6, "deepseek7b": 6}` | `{"exact_choices": 18}` |
| p816_bus_public_transport | `public transport` | `public transportation` | 12 | `target_equivalent` | `{"qwen3": 6, "glm4": 6}` | `{"no_choices": 12}` |
| p816_chair_household_furniture | `household furniture` | `furniture` | 12 | `target_equivalent` | `{"qwen3": 6, "deepseek7b": 6}` | `{"no_choices": 12}` |
| p816_heart_body_organ | `body organ` | `body organ` | 12 | `target_equivalent` | `{"qwen3": 6, "glm4": 6}` | `{"exact_choices": 12}` |
| p816_oak_tall_tree | `tall tree` | `tall tree` | 12 | `target_equivalent` | `{"qwen3": 6, "glm4": 6}` | `{"exact_choices": 12}` |
| p816_rose_flowering_plant | `flowering plant` | `flowering plant` | 12 | `target_equivalent` | `{"qwen3": 3, "glm4": 6, "deepseek7b": 3}` | `{"exact_choices": 9, "no_choices": 3}` |
| p816_salmon_aquatic_animal | `aquatic animal` | `aquatic animal` | 12 | `target_equivalent` | `{"qwen3": 6, "glm4": 6}` | `{"exact_choices": 12}` |
| p816_winter_cold_season | `cold season` | `cold season` | 9 | `target_equivalent` | `{"qwen3": 3, "glm4": 3, "deepseek7b": 3}` | `{"exact_choices": 9}` |
| p816_oak_tall_tree | `tall tree` | `tree` | 8 | `target_equivalent` | `{"qwen3": 2, "deepseek7b": 6}` | `{"no_choices": 8}` |
| p816_bus_public_transport | `public transport` | `__________` | 6 | `format_echo` | `{"deepseek7b": 6}` | `{"no_choices": 6}` |
| p816_cactus_desert_plant | `desert plant` | `desert plant` | 6 | `target_equivalent` | `{"qwen3": 3, "glm4": 3}` | `{"exact_choices": 6}` |
| p816_carrot_root_vegetable | `root vegetable` | `root vegetable` | 6 | `target_equivalent` | `{"qwen3": 3, "glm4": 3}` | `{"exact_choices": 6}` |
| p816_cat_living_thing | `living thing` | `___________` | 6 | `format_echo` | `{"deepseek7b": 6}` | `{"no_choices": 6}` |
| p816_cat_living_thing | `living thing` | `domestic animal` | 6 | `target_equivalent` | `{"qwen3": 6}` | `{"no_choices": 6}` |
| p816_cat_living_thing | `living thing` | `pet category` | 6 | `broad_near_miss` | `{"glm4": 6}` | `{"no_choices": 6}` |
| p816_chair_household_furniture | `household furniture` | `furniture category` | 6 | `target_equivalent` | `{"glm4": 6}` | `{"no_choices": 6}` |
| p816_doctor_medical_worker | `medical worker` | `</think>` | 6 | `format_echo` | `{"deepseek7b": 6}` | `{"no_choices": 6}` |
| p816_doctor_medical_worker | `medical worker` | `health professional` | 6 | `target_equivalent` | `{"glm4": 6}` | `{"no_choices": 6}` |
| p816_doctor_medical_worker | `medical worker` | `medical professional` | 6 | `target_equivalent` | `{"qwen3": 6}` | `{"no_choices": 6}` |
| p816_guitar_musical_instrument | `musical instrument` | `answer must be one of the following` | 6 | `format_echo` | `{"deepseek7b": 6}` | `{"no_choices": 6}` |
| p816_hammer_hand_tool | `hand tool` | `hand tool` | 6 | `target_equivalent` | `{"qwen3": 3, "glm4": 3}` | `{"exact_choices": 6}` |
| p816_heart_body_organ | `body organ` | `?` | 6 | `format_echo` | `{"deepseek7b": 6}` | `{"exact_choices": 6}` |
| p816_heart_body_organ | `body organ` | `___________` | 6 | `format_echo` | `{"deepseek7b": 6}` | `{"no_choices": 6}` |
| p816_heart_body_organ | `body organ` | `circulatory system` | 6 | `unknown_other` | `{"qwen3": 6}` | `{"no_choices": 6}` |
| p816_heart_body_organ | `body organ` | `human body part` | 6 | `broad_near_miss` | `{"glm4": 6}` | `{"no_choices": 6}` |
| p816_laptop_electronic_device | `electronic device` | `electronics` | 6 | `broad_near_miss` | `{"qwen3": 6}` | `{"no_choices": 6}` |
| p816_laptop_electronic_device | `electronic device` | `personal computer` | 6 | `target_equivalent` | `{"deepseek7b": 6}` | `{"no_choices": 6}` |
| p816_laptop_electronic_device | `electronic device` | `personal computing device` | 6 | `unknown_other` | `{"glm4": 6}` | `{"no_choices": 6}` |
| p816_oak_tall_tree | `tall tree` | `?` | 6 | `format_echo` | `{"deepseek7b": 6}` | `{"exact_choices": 6}` |
| p816_oak_tall_tree | `tall tree` | `hardwood tree` | 6 | `target_equivalent` | `{"glm4": 6}` | `{"no_choices": 6}` |
