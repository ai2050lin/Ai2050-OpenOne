# Phase 821 Boundary-Standard-Guided Causal Localization (main)

- Boundary: Phase 820 answer-boundary standard v1.
- Intervention: exact_choices donor residual state patched into no_choices recipient at the first generated token.

## Model Summary

| model | cases | rows | improved cases | target transitions | protocol repairs | improved rows | degraded rows | mean delta rank | baseline classes | patched classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| qwen3 | 5 | 20 | 5 | 5 | 0 | 17 | 3 | 1.050 | `{"broad_near_miss": 16, "close_near_miss": 4}` | `{"target_equivalent": 17, "unknown_other": 3}` |
| glm4 | 5 | 20 | 3 | 3 | 2 | 12 | 8 | 1.600 | `{"broad_near_miss": 8, "close_near_miss": 4, "unknown_other": 8}` | `{"broad_near_miss": 4, "target_equivalent": 12, "unknown_other": 4}` |
| deepseek7b | 5 | 20 | 4 | 3 | 3 | 9 | 4 | 0.750 | `{"format_echo": 12, "format_with_target": 4, "object_echo": 4}` | `{"format_echo": 7, "generic_blocker": 8, "target_equivalent": 3, "unknown_other": 2}` |

## Best Case Transitions

| model | case | baseline | best patched | best layer | delta | generated |
|---|---|---|---|---:|---:|---|
| qwen3 | p816_carrot_root_vegetable | `close_near_miss` | `target_equivalent` | 32 | 1 | `root vegetable` |
| qwen3 | p816_heart_body_organ | `broad_near_miss` | `target_equivalent` | 32 | 2 | `body organ` |
| qwen3 | p816_oxygen_chemical_element | `broad_near_miss` | `target_equivalent` | 32 | 2 | `chemical element` |
| qwen3 | p816_red_warm_color | `broad_near_miss` | `target_equivalent` | 32 | 2 | `warm color` |
| qwen3 | p816_triangle_geometric_shape | `broad_near_miss` | `target_equivalent` | 32 | 2 | `geometric shape` |
| glm4 | p816_cactus_desert_plant | `unknown_other` | `target_equivalent` | 36 | 5 | `desert plant` |
| glm4 | p816_cat_living_thing | `broad_near_miss` | `unknown_other` | 36 | -3 | `living creature` |
| glm4 | p816_heart_body_organ | `close_near_miss` | `broad_near_miss` | 36 | -1 | `body part` |
| glm4 | p816_red_warm_color | `broad_near_miss` | `target_equivalent` | 36 | 2 | `warm color` |
| glm4 | p816_winter_cold_season | `unknown_other` | `target_equivalent` | 36 | 5 | `cold season` |
| deepseek7b | p816_apple_edible_fruit | `format_echo` | `target_equivalent` | 26 | 5 | `edible fruit` |
| deepseek7b | p816_bus_public_transport | `format_echo` | `target_equivalent` | 26 | 5 | `public transportation` |
| deepseek7b | p816_cat_living_thing | `format_echo` | `generic_blocker` | 27 | 1 | `The answer is a single word or phrase` |
| deepseek7b | p816_salmon_aquatic_animal | `format_with_target` | `generic_blocker` | 26 | -1 | `The answer should be a single word or` |
| deepseek7b | p816_triangle_geometric_shape | `object_echo` | `target_equivalent` | 26 | 5 | `geometric shape` |
