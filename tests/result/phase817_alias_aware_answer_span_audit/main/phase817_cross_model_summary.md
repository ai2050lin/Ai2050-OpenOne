# Phase 817 Alias Aware Answer Span Audit (main)

- Source: Phase 816 saved rollout rows; no new model forward pass.
- Boundary: tests whether exact phrase rollout undercounts semantically acceptable multi-token answers.

## Model Summary

| model | rows | exact rollout | alias rollout | exact full | alias full | rollout gain | full gain | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 24 | 15 | 22 | 15 | 19 | 7 | 4 | `{"exact_span_and_rollout_closed": 15, "alias_rollout_without_span_score": 3, "alias_rollout_rescues_span_score": 4, "span_score_closed_alias_rollout_not_closed": 2}` |
| glm4 | 24 | 17 | 24 | 17 | 21 | 7 | 4 | `{"exact_span_and_rollout_closed": 17, "alias_rollout_without_span_score": 3, "alias_rollout_rescues_span_score": 4}` |
| deepseek7b | 24 | 10 | 13 | 10 | 10 | 3 | 0 | `{"exact_span_and_rollout_closed": 10, "alias_unclosed": 9, "alias_rollout_without_span_score": 3, "span_score_closed_alias_rollout_not_closed": 2}` |

## Prompt Variant Summary

| model | prompt | n | exact rollout | alias rollout | exact full | alias full | labels |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | exact_choices | 12 | 12 | 12 | 12 | 12 | `{"exact_span_and_rollout_closed": 12}` |
| qwen3 | no_choices | 12 | 3 | 10 | 3 | 7 | `{"alias_rollout_without_span_score": 3, "exact_span_and_rollout_closed": 3, "alias_rollout_rescues_span_score": 4, "span_score_closed_alias_rollout_not_closed": 2}` |
| glm4 | exact_choices | 12 | 12 | 12 | 12 | 12 | `{"exact_span_and_rollout_closed": 12}` |
| glm4 | no_choices | 12 | 5 | 12 | 5 | 9 | `{"alias_rollout_without_span_score": 3, "exact_span_and_rollout_closed": 5, "alias_rollout_rescues_span_score": 4}` |
| deepseek7b | exact_choices | 12 | 9 | 9 | 9 | 9 | `{"exact_span_and_rollout_closed": 9, "span_score_closed_alias_rollout_not_closed": 2, "alias_unclosed": 1}` |
| deepseek7b | no_choices | 12 | 1 | 4 | 1 | 1 | `{"alias_unclosed": 8, "alias_rollout_without_span_score": 3, "exact_span_and_rollout_closed": 1}` |

## Rescued Rows

| model | prompt | case | target | generated | alias | label |
|---|---|---|---|---|---|---|
| qwen3 | no_choices | p816_cat_living_thing | `living thing` | `Domestic animal` | `domestic animal` | `alias_rollout_without_span_score` |
| qwen3 | no_choices | p816_chair_household_furniture | `household furniture` | `Furniture` | `furniture` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | `Fish` | `fish` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_oak_tall_tree | `tall tree` | `Tree` | `tree` | `alias_rollout_without_span_score` |
| qwen3 | no_choices | p816_laptop_electronic_device | `electronic device` | `Electronics` | `electronics` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_triangle_geometric_shape | `geometric shape` | `Geometry` | `geometry shape` | `alias_rollout_rescues_span_score` |
| qwen3 | no_choices | p816_doctor_medical_worker | `medical worker` | `Medical Professional` | `medical professional` | `alias_rollout_without_span_score` |
| glm4 | no_choices | p816_cat_living_thing | `living thing` | `Pet category` | `pet category` | `alias_rollout_without_span_score` |
| glm4 | no_choices | p816_chair_household_furniture | `household furniture` | `Furniture category` | `furniture category` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_heart_body_organ | `body organ` | `Human body part` | `human body part` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | `Freshwater fish` | `freshwater fish` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_oak_tall_tree | `tall tree` | `Hardwood Tree` | `hardwood tree` | `alias_rollout_without_span_score` |
| glm4 | no_choices | p816_laptop_electronic_device | `electronic device` | `Personal Computing Device` | `personal computing device` | `alias_rollout_rescues_span_score` |
| glm4 | no_choices | p816_doctor_medical_worker | `medical worker` | `Health Professional` | `health professional` | `alias_rollout_without_span_score` |
| deepseek7b | no_choices | p816_chair_household_furniture | `household furniture` | `furniture` | `furniture` | `alias_rollout_without_span_score` |
| deepseek7b | no_choices | p816_oak_tall_tree | `tall tree` | `Tree` | `tree` | `alias_rollout_without_span_score` |
| deepseek7b | no_choices | p816_laptop_electronic_device | `electronic device` | `Personal computer` | `personal computer` | `alias_rollout_without_span_score` |
