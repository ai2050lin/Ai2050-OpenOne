# Phase 821 Boundary-Standard-Guided Causal Localization (confirm)

- Boundary: Phase 820 answer-boundary standard v1.
- Intervention: exact_choices donor residual state patched into no_choices recipient at the first generated token.

## Model Summary

| model | cases | rows | improved cases | target transitions | protocol repairs | improved rows | degraded rows | mean delta rank | baseline classes | patched classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| qwen3 | 8 | 48 | 5 | 5 | 0 | 6 | 20 | -1.208 | `{"broad_near_miss": 30, "close_near_miss": 18}` | `{"broad_near_miss": 14, "close_near_miss": 8, "object_echo": 1, "target_equivalent": 6, "unknown_other": 19}` |
| glm4 | 6 | 36 | 4 | 4 | 2 | 12 | 9 | 0.639 | `{"broad_near_miss": 12, "close_near_miss": 12, "unknown_other": 12}` | `{"broad_near_miss": 9, "close_near_miss": 5, "target_equivalent": 12, "unknown_other": 10}` |
| deepseek7b | 8 | 48 | 7 | 0 | 0 | 10 | 2 | 0.167 | `{"format_echo": 36, "format_with_target": 6, "object_echo": 6}` | `{"format_echo": 24, "format_with_target": 4, "generic_blocker": 12, "object_echo": 6, "unknown_other": 2}` |

## Best Case Transitions

| model | case | baseline | best patched | best layer | delta | generated |
|---|---|---|---|---:|---:|---|
| qwen3 | p816_cactus_desert_plant | `close_near_miss` | `target_equivalent` | 35 | 1 | `desert plants` |
| qwen3 | p816_carrot_root_vegetable | `close_near_miss` | `target_equivalent` | 35 | 1 | `root vegetable` |
| qwen3 | p816_heart_body_organ | `broad_near_miss` | `target_equivalent` | 14 | 2 | `Body Organ` |
| qwen3 | p816_laptop_electronic_device | `close_near_miss` | `target_equivalent` | 7 | 1 | `Electronic Devices` |
| qwen3 | p816_oxygen_chemical_element | `broad_near_miss` | `broad_near_miss` | 0 | 0 | `Gas` |
| qwen3 | p816_red_warm_color | `broad_near_miss` | `broad_near_miss` | 0 | 0 | `Color` |
| qwen3 | p816_triangle_geometric_shape | `broad_near_miss` | `broad_near_miss` | 0 | 0 | `Geometry` |
| qwen3 | p816_winter_cold_season | `broad_near_miss` | `target_equivalent` | 21 | 2 | `Winter Season` |
| glm4 | p816_cactus_desert_plant | `unknown_other` | `target_equivalent` | 8 | 5 | `Desert plant` |
| glm4 | p816_carrot_root_vegetable | `close_near_miss` | `target_equivalent` | 39 | 1 | `root vegetable` |
| glm4 | p816_cat_living_thing | `broad_near_miss` | `broad_near_miss` | 0 | 0 | `Pet category` |
| glm4 | p816_heart_body_organ | `close_near_miss` | `close_near_miss` | 0 | 0 | `Human body part` |
| glm4 | p816_red_warm_color | `broad_near_miss` | `target_equivalent` | 23 | 2 | `red color` |
| glm4 | p816_winter_cold_season | `unknown_other` | `target_equivalent` | 8 | 5 | `cold season` |
| deepseek7b | p816_apple_edible_fruit | `format_echo` | `generic_blocker` | 27 | 1 | `The answer should be a single word` |
| deepseek7b | p816_bus_public_transport | `format_echo` | `generic_blocker` | 27 | 1 | `The answer should be a single word or` |
| deepseek7b | p816_cactus_desert_plant | `format_echo` | `generic_blocker` | 27 | 1 | `The answer is a plant` |
| deepseek7b | p816_carrot_root_vegetable | `format_echo` | `generic_blocker` | 22 | 1 | `The answer should be a single word or` |
| deepseek7b | p816_cat_living_thing | `format_echo` | `generic_blocker` | 27 | 1 | `The answer is a single word or phrase` |
| deepseek7b | p816_doctor_medical_worker | `format_echo` | `generic_blocker` | 5 | 1 | `The answer should be a single word or` |
| deepseek7b | p816_salmon_aquatic_animal | `format_with_target` | `format_with_target` | 0 | 0 | `Salmon is best described as a freshwater fish` |
| deepseek7b | p816_triangle_geometric_shape | `object_echo` | `generic_blocker` | 27 | 1 | `The answer is a single word or phrase` |
