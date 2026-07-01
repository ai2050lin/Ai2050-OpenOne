# Phase 818 Alias Span Candidate Scoring Benchmark (main)

- Boundary: target answer is evaluated as an alias class, while near-miss, wrong, and generic spans remain explicit competitors.

## Model Summary

| model | rows | exact score | alias score | exact rollout | alias rollout | exact full | alias full | near cleared | wrong cleared | generic cleared | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 24 | 15 | 21 | 14 | 20 | 14 | 20 | 21 | 24 | 24 | `{"alias_class_closes_exact_phrase_fails": 6, "alias_score_and_rollout_closed": 14, "alias_score_closed_rollout_not_closed": 1, "near_miss_span_wins": 3}` |
| glm4 | 24 | 17 | 22 | 16 | 20 | 16 | 20 | 22 | 24 | 24 | `{"alias_class_closes_exact_phrase_fails": 4, "alias_score_and_rollout_closed": 16, "alias_score_closed_rollout_not_closed": 2, "near_miss_span_wins": 2}` |
| deepseek7b | 24 | 12 | 17 | 10 | 13 | 10 | 13 | 24 | 23 | 18 | `{"alias_class_closes_exact_phrase_fails": 3, "alias_score_and_rollout_closed": 10, "alias_score_closed_rollout_not_closed": 4, "generic_blocker_span_wins": 6, "wrong_span_wins": 1}` |

## Prompt Variant Summary

| model | prompt | n | alias score | alias rollout | alias full | generation classes | labels |
|---|---|---:|---:|---:|---:|---|---|
| qwen3 | exact_choices | 12 | 12 | 12 | 12 | `{"target_alias": 12}` | `{"alias_score_and_rollout_closed": 12}` |
| qwen3 | no_choices | 12 | 9 | 8 | 8 | `{"near_miss": 3, "other": 1, "target_alias": 8}` | `{"alias_class_closes_exact_phrase_fails": 6, "alias_score_and_rollout_closed": 2, "alias_score_closed_rollout_not_closed": 1, "near_miss_span_wins": 3}` |
| glm4 | exact_choices | 12 | 12 | 12 | 12 | `{"target_alias": 12}` | `{"alias_score_and_rollout_closed": 12}` |
| glm4 | no_choices | 12 | 10 | 8 | 8 | `{"near_miss": 2, "other": 2, "target_alias": 8}` | `{"alias_class_closes_exact_phrase_fails": 4, "alias_score_and_rollout_closed": 4, "alias_score_closed_rollout_not_closed": 2, "near_miss_span_wins": 2}` |
| deepseek7b | exact_choices | 12 | 11 | 9 | 9 | `{"other": 3, "target_alias": 9}` | `{"alias_score_and_rollout_closed": 9, "alias_score_closed_rollout_not_closed": 2, "wrong_span_wins": 1}` |
| deepseek7b | no_choices | 12 | 6 | 4 | 4 | `{"other": 8, "target_alias": 4}` | `{"alias_class_closes_exact_phrase_fails": 3, "alias_score_and_rollout_closed": 1, "alias_score_closed_rollout_not_closed": 2, "generic_blocker_span_wins": 6}` |

## First Failure Rows

| model | prompt | case | target | generated | gen class | best alias | best non-alias | margin | label |
|---|---|---|---|---|---|---|---|---:|---|
| qwen3 | no_choices | p816_heart_body_organ | `body organ` | `Circulatory System` | `other` | ` Body Organ` | ` Body Part`/near_miss | 1.375 | `alias_score_closed_rollout_not_closed` |
| qwen3 | no_choices | p816_laptop_electronic_device | `electronic device` | `Electronics` | `near_miss` | ` Electronic Device` | ` Electronics`/near_miss | -1.167 | `near_miss_span_wins` |
| qwen3 | no_choices | p816_triangle_geometric_shape | `geometric shape` | `Geometry` | `near_miss` | ` Polygon` | ` Geometry`/near_miss | -0.875 | `near_miss_span_wins` |
| qwen3 | no_choices | p816_oxygen_chemical_element | `chemical element` | `Gas` | `near_miss` | ` Element` | ` Gas`/near_miss | -0.625 | `near_miss_span_wins` |
| glm4 | no_choices | p816_cat_living_thing | `living thing` | `Pet category` | `near_miss` | ` Mammal` | ` Pet`/near_miss | -0.006 | `near_miss_span_wins` |
| glm4 | no_choices | p816_heart_body_organ | `body organ` | `Human body part` | `near_miss` | ` Body Organ` | ` Body Part`/near_miss | -1.094 | `near_miss_span_wins` |
| glm4 | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | `Freshwater fish` | `other` | ` Aquatic Animal` | ` Seafood`/near_miss | 0.779 | `alias_score_closed_rollout_not_closed` |
| glm4 | no_choices | p816_laptop_electronic_device | `electronic device` | `Personal Computing Device` | `other` | ` Personal Computer` | `Electronics`/near_miss | 1.103 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_cat_living_thing | `living thing` | `___________` | `other` | ` Mammal` | ` I don't know`/generic_blocker | 1.730 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_bus_public_transport | `public transport` | `__________` | `other` | ` Public Transportation` | ` I don't know`/generic_blocker | -0.293 | `generic_blocker_span_wins` |
| deepseek7b | no_choices | p816_guitar_musical_instrument | `musical instrument` | `Answer must be one of the following` | `other` | `Musical Instrument` | ` I don't know`/generic_blocker | -0.681 | `generic_blocker_span_wins` |
| deepseek7b | exact_choices | p816_heart_body_organ | `body organ` | `?` | `other` | ` body organ` | ` public transport`/wrong | 0.750 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_heart_body_organ | `body organ` | `___________` | `other` | ` Organ` | ` I don't know`/generic_blocker | -1.735 | `generic_blocker_span_wins` |
| deepseek7b | exact_choices | p816_salmon_aquatic_animal | `aquatic animal` | `The correct phrase is "aquatic animal` | `other` | `aquatic animal` | ` public transport`/wrong | -0.085 | `wrong_span_wins` |
| deepseek7b | no_choices | p816_salmon_aquatic_animal | `aquatic animal` | `Salmon is best described as a freshwater fish` | `other` | ` Aquatic Animal` | ` I don't know`/generic_blocker | -0.924 | `generic_blocker_span_wins` |
| deepseek7b | exact_choices | p816_oak_tall_tree | `tall tree` | `?` | `other` | ` tall tree` | ` warm color`/wrong | 1.219 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_triangle_geometric_shape | `geometric shape` | `Triangle` | `other` | ` polygon` | ` I don't know`/generic_blocker | 0.882 | `alias_score_closed_rollout_not_closed` |
| deepseek7b | no_choices | p816_oxygen_chemical_element | `chemical element` | `O2` | `other` | ` Element` | ` I don't know`/generic_blocker | -0.754 | `generic_blocker_span_wins` |
| deepseek7b | no_choices | p816_doctor_medical_worker | `medical worker` | `</think>` | `other` | ` Health Professional` | ` I don't know`/generic_blocker | -0.521 | `generic_blocker_span_wins` |
