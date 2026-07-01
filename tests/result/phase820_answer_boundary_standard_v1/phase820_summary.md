# Phase 820 Answer Boundary Standard v1

- Source: Phase 819 phrase aggregates.
- Boundary: external review standard for strict / medium / loose closure; no model loading.

## Standard Distribution

- Standard rows: 76
- Changed from Phase 819 heuristic class: 17
- Classes: `{"format_echo": 16, "target_equivalent": 39, "wrong": 1, "close_near_miss": 5, "unknown_other": 2, "broad_near_miss": 7, "format_with_target": 3, "object_echo": 3}`

## Phase 818 Confirm Reanalysis

| model/prompt | n | strict rollout | medium rollout | loose rollout | semantic target | protocol valid | strict full | medium full | loose full | classes | scores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| deepseek7b/exact_choices | 20 | 13 | 13 | 13 | 15 | 13 | 13 | 13 | 13 | `{"target_equivalent": 13, "format_with_target": 2, "format_echo": 4, "wrong": 1}` | `{"strict": 17, "medium": 17, "loose": 17}` |
| deepseek7b/no_choices | 20 | 4 | 4 | 4 | 5 | 4 | 4 | 4 | 4 | `{"format_echo": 12, "target_equivalent": 4, "format_with_target": 1, "object_echo": 3}` | `{"strict": 9, "medium": 9, "loose": 9}` |
| glm4/exact_choices | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | `{"target_equivalent": 20}` | `{"strict": 20, "medium": 20, "loose": 20}` |
| glm4/no_choices | 20 | 14 | 16 | 18 | 14 | 18 | 14 | 15 | 17 | `{"broad_near_miss": 2, "target_equivalent": 14, "close_near_miss": 2, "unknown_other": 2}` | `{"strict": 16, "medium": 17, "loose": 20}` |
| qwen3/exact_choices | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | `{"target_equivalent": 20}` | `{"strict": 20, "medium": 20, "loose": 20}` |
| qwen3/no_choices | 20 | 12 | 15 | 20 | 12 | 20 | 12 | 15 | 19 | `{"target_equivalent": 12, "broad_near_miss": 5, "close_near_miss": 3}` | `{"strict": 14, "medium": 16, "loose": 20}` |

## Changed Rows

| case | target | phrase | phase819 | final | reason |
|---|---|---|---|---|---|
| p816_cactus_desert_plant | `desert plant` | `cactus plants` | `target_equivalent` | `close_near_miss` | Plural object-specific plant phrase; related but not strict desert-plant category. |
| p816_cactus_desert_plant | `desert plant` | `plant life cycle` | `broad_near_miss` | `unknown_other` | Related plant process, but not a category phrase for cactus. |
| p816_carrot_root_vegetable | `root vegetable` | `vegetable` | `target_equivalent` | `close_near_miss` | Correct broad category, but misses root-vegetable granularity. |
| p816_carrot_root_vegetable | `root vegetable` | `vegetables` | `target_equivalent` | `close_near_miss` | Plural broad category; acceptable only as medium/loose, not strict target. |
| p816_hammer_hand_tool | `hand tool` | `hammer is a hand tool` | `unknown_other` | `format_with_target` | Contains the target phrase but violates write-only-the-phrase protocol. |
| p816_heart_body_organ | `body organ` | `circulatory system` | `unknown_other` | `broad_near_miss` | Related biological system, not the requested organ category. |
| p816_heart_body_organ | `body organ` | `human body part` | `broad_near_miss` | `close_near_miss` | Close parent category; broader than body organ. |
| p816_laptop_electronic_device | `electronic device` | `electronics` | `broad_near_miss` | `close_near_miss` | Related electronics category; slightly broader than electronic device. |
| p816_laptop_electronic_device | `electronic device` | `personal computing device` | `unknown_other` | `target_equivalent` | Acceptable specific category for laptop under electronic-device task. |
| p816_oxygen_chemical_element | `chemical element` | `o2` | `unknown_other` | `object_echo` | Chemical formula/object identity, not category phrase. |
| p816_red_warm_color | `warm color` | `color` | `close_near_miss` | `broad_near_miss` | Correct parent category, but misses warm-color granularity. |
| p816_salmon_aquatic_animal | `aquatic animal` | `freshwater fish` | `unknown_other` | `target_equivalent` | Specific correct category for salmon under aquatic-animal task. |
| p816_salmon_aquatic_animal | `aquatic animal` | `salmon is best described as a freshwater fish` | `unknown_other` | `format_with_target` | Semantically answers the question but violates phrase-only protocol. |
| p816_salmon_aquatic_animal | `aquatic animal` | `the correct phrase is aquatic animal` | `format_echo` | `format_with_target` | Contains target phrase but adds explanation/prefix. |
| p816_triangle_geometric_shape | `geometric shape` | `geometry` | `close_near_miss` | `broad_near_miss` | Related mathematical domain, not the requested shape category. |
| p816_triangle_geometric_shape | `geometric shape` | `triangle` | `unknown_other` | `object_echo` | Echoes the object rather than giving its category. |
| p816_winter_cold_season | `cold season` | `winter` | `unknown_other` | `object_echo` | Echoes the object rather than giving its category. |
