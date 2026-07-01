# Phase 817 Alias Aware Answer Span Audit (confirm)

- Source: Phase 816 saved rollout rows; no new model forward pass.
- Boundary: tests whether exact phrase rollout undercounts semantically acceptable multi-token answers.

## Model Summary

| model | rows | exact rollout | alias rollout | exact full | alias full | rollout gain | full gain | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 40 | 23 | 38 | 23 | 34 | 15 | 11 | `{"exact_span_and_rollout_closed": 23, "alias_rollout_without_span_score": 4, "alias_rollout_rescues_span_score": 11, "span_score_closed_alias_rollout_not_closed": 2}` |
| glm4 | 40 | 27 | 40 | 27 | 36 | 13 | 9 | `{"exact_span_and_rollout_closed": 27, "alias_rollout_without_span_score": 4, "alias_rollout_rescues_span_score": 9}` |
| deepseek7b | 40 | 14 | 17 | 14 | 14 | 3 | 0 | `{"exact_span_and_rollout_closed": 14, "alias_unclosed": 19, "alias_rollout_without_span_score": 3, "span_score_closed_alias_rollout_not_closed": 4}` |

## Prompt Variant Summary

| model | prompt | n | exact rollout | alias rollout | exact full | alias full | labels |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | exact_choices | 20 | 20 | 20 | 20 | 20 | `{"exact_span_and_rollout_closed": 20}` |
| qwen3 | no_choices | 20 | 3 | 18 | 3 | 14 | `{"alias_rollout_without_span_score": 4, "alias_rollout_rescues_span_score": 11, "exact_span_and_rollout_closed": 3, "span_score_closed_alias_rollout_not_closed": 2}` |
| glm4 | exact_choices | 20 | 20 | 20 | 20 | 20 | `{"exact_span_and_rollout_closed": 20}` |
| glm4 | no_choices | 20 | 7 | 20 | 7 | 16 | `{"alias_rollout_without_span_score": 4, "alias_rollout_rescues_span_score": 9, "exact_span_and_rollout_closed": 7}` |
| deepseek7b | exact_choices | 20 | 13 | 13 | 13 | 13 | `{"exact_span_and_rollout_closed": 13, "alias_unclosed": 4, "span_score_closed_alias_rollout_not_closed": 3}` |
| deepseek7b | no_choices | 20 | 1 | 4 | 1 | 1 | `{"alias_unclosed": 15, "alias_rollout_without_span_score": 3, "exact_span_and_rollout_closed": 1, "span_score_closed_alias_rollout_not_closed": 1}` |

## Rescued Rows

| model | prompt | case | target | generated | alias | label |
|---|---|---|---|---|---|---|
| qwen3 | no_choices | p816_cat_living_thing | `living thing` | `Domestic animal` | `domestic animal` | `alias_rollout_without_span_score` |
| qwen3 | no_choices | p816_hammer_hand_tool | `hand tool` | `Tools` | `tools` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_rose_flowering_plant | `flowering plant` | `Flowers` | `flowers` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_chair_household_furniture | `household furniture` | `Furniture` | `furniture` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_apple_edible_fruit | `edible fruit` | `Fruit` | `fruit` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_red_warm_color | `warm color` | `Color` | `color` | `alias_rollout_without_span_score` |
| qwen3 | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | `Fish` | `fish` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_oak_tall_tree | `tall tree` | `Tree Species` | `tree species` | `alias_rollout_without_span_score` |
| qwen3 | no_choices | p816_carrot_root_vegetable | `root vegetable` | `Vegetables` | `vegetables` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_laptop_electronic_device | `electronic device` | `Electronics` | `electronics` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_spoon_eating_utensil | `eating utensil` | `Kitchen Utensil` | `kitchen utensil` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_triangle_geometric_shape | `geometric shape` | `Geometry` | `geometry shape` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_winter_cold_season | `cold season` | `Cold Weather` | `cold weather` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_cactus_desert_plant | `desert plant` | `Cactus plants` | `cactus plants` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_doctor_medical_worker | `medical worker` | `Medical Professional` | `medical professional` | `alias_rollout_without_span_score` |
| glm4 | no_choices | p816_cat_living_thing | `living thing` | `Pet category` | `pet category` | `alias_rollout_without_span_score` |
| glm4 | no_choices | p816_hammer_hand_tool | `hand tool` | `Tool category` | `tool category` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_chair_household_furniture | `household furniture` | `Furniture category` | `furniture category` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_apple_edible_fruit | `edible fruit` | `Fruit category` | `fruit category` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_heart_body_organ | `body organ` | `Human body part` | `human body part` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_red_warm_color | `warm color` | `Color category` | `color category` | `alias_rollout_without_span_score` |
| glm4 | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | `Freshwater fish` | `freshwater fish` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_oak_tall_tree | `tall tree` | `Hardwood Tree` | `hardwood tree` | `alias_rollout_without_span_score` |
| glm4 | no_choices | p816_carrot_root_vegetable | `root vegetable` | `Vegetable` | `vegetable` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_laptop_electronic_device | `electronic device` | `Personal Computing Device` | `personal computing device` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_winter_cold_season | `cold season` | `Seasonal weather patterns` | `season` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_cactus_desert_plant | `desert plant` | `Plant Life Cycle` | `plant` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_doctor_medical_worker | `medical worker` | `Health Professional` | `health professional` | `alias_rollout_without_span_score` |
| deepseek7b | no_choices | p816_chair_household_furniture | `household furniture` | `furniture` | `furniture` | `alias_rollout_without_span_score` |
| deepseek7b | no_choices | p816_oak_tall_tree | `tall tree` | `Tree` | `tree` | `alias_rollout_without_span_score` |
| deepseek7b | no_choices | p816_laptop_electronic_device | `electronic device` | `Personal computer` | `personal computer` | `alias_rollout_without_span_score` |
