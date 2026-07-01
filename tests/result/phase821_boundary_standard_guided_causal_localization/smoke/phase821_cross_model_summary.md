# Phase 821 Boundary-Standard-Guided Causal Localization (smoke)

- Boundary: Phase 820 answer-boundary standard v1.
- Intervention: exact_choices donor residual state patched into no_choices recipient at the first generated token.

## Model Summary

| model | cases | rows | improved cases | target transitions | protocol repairs | improved rows | degraded rows | mean delta rank | baseline classes | patched classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| qwen3 | 2 | 4 | 2 | 2 | 0 | 4 | 0 | 1.500 | `{"broad_near_miss": 2, "close_near_miss": 2}` | `{"target_equivalent": 4}` |
| glm4 | 2 | 4 | 1 | 1 | 1 | 2 | 2 | 1.000 | `{"broad_near_miss": 2, "unknown_other": 2}` | `{"target_equivalent": 2, "unknown_other": 2}` |
| deepseek7b | 2 | 4 | 2 | 1 | 1 | 3 | 0 | 1.750 | `{"format_echo": 2, "object_echo": 2}` | `{"format_echo": 1, "generic_blocker": 2, "target_equivalent": 1}` |

## Best Case Transitions

| model | case | baseline | best patched | best layer | delta | generated |
|---|---|---|---|---:|---:|---|
| qwen3 | p816_carrot_root_vegetable | `close_near_miss` | `target_equivalent` | 34 | 1 | `root vegetable` |
| qwen3 | p816_heart_body_organ | `broad_near_miss` | `target_equivalent` | 34 | 2 | `body organ` |
| glm4 | p816_cat_living_thing | `broad_near_miss` | `unknown_other` | 38 | -3 | `living creature` |
| glm4 | p816_winter_cold_season | `unknown_other` | `target_equivalent` | 38 | 5 | `cold season` |
| deepseek7b | p816_cat_living_thing | `format_echo` | `generic_blocker` | 27 | 1 | `The answer is a single word` |
| deepseek7b | p816_triangle_geometric_shape | `object_echo` | `target_equivalent` | 26 | 5 | `geometric shape` |
