# Phase85 Readout Open Generation Audit Summary

## qwen3

items=224, basis_items=224, rows=7392, audit_layers=[4, 8, 12]
module=resid_out, component_rank=24, max_new_tokens=6, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7262 | 0.0000 |
| 3 | restore_frame_suffix_final | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 |
| 4 | erase_frame_suffix_all | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5268 | 0.0000 |
| 5 | restore_frame_suffix_all | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1027 | 0.0000 |
| 6 | erase_frame_suffix_function | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5580 | 0.0000 |
| 7 | restore_frame_suffix_function | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1086 | 0.0000 |
| 8 | erase_frame_suffix_lexical | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3676 | 0.0000 |
| 9 | restore_frame_suffix_lexical | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1071 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9643 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1146 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8170 | 0.0000 |
| 3 | restore_frame_suffix_final:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0804 | 0.0000 |
| 4 | erase_frame_suffix_all:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6473 | 0.0000 |
| 5 | restore_frame_suffix_all:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1161 | 0.0000 |
| 6 | erase_frame_suffix_function:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6384 | 0.0000 |
| 7 | restore_frame_suffix_function:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1339 | 0.0000 |
| 8 | erase_frame_suffix_lexical:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4018 | 0.0000 |
| 9 | restore_frame_suffix_lexical:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1071 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9554 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1116 | 0.0000 |
| 12 | base:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | erase_frame_suffix_final:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7143 | 0.0000 |
| 14 | restore_frame_suffix_final:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0893 | 0.0000 |
| 15 | erase_frame_suffix_all:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5402 | 0.0000 |
| 16 | restore_frame_suffix_all:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1205 | 0.0000 |
| 17 | erase_frame_suffix_function:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5759 | 0.0000 |
| 18 | restore_frame_suffix_function:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1071 | 0.0000 |
| 19 | erase_frame_suffix_lexical:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3616 | 0.0000 |
| 20 | restore_frame_suffix_lexical:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1116 | 0.0000 |
| 21 | erase_frame_all_suffix_tokens:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9732 | 0.0000 |
| 22 | restore_frame_all_suffix_tokens:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0982 | 0.0000 |
| 23 | base:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | erase_frame_suffix_final:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6473 | 0.0000 |
| 25 | restore_frame_suffix_final:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0804 | 0.0000 |
| 26 | erase_frame_suffix_all:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3929 | 0.0000 |
| 27 | restore_frame_suffix_all:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0714 | 0.0000 |
| 28 | erase_frame_suffix_function:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4598 | 0.0000 |
| 29 | restore_frame_suffix_function:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0848 | 0.0000 |
| 30 | erase_frame_suffix_lexical:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3393 | 0.0000 |
| 31 | restore_frame_suffix_lexical:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1027 | 0.0000 |
| 32 | erase_frame_all_suffix_tokens:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9643 | 0.0000 |
| 33 | restore_frame_all_suffix_tokens:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1339 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8125 | 0.0000 |
| 3 | restore_frame_suffix_final:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1042 | 0.0000 |
| 4 | erase_frame_suffix_all:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4896 | 0.0000 |
| 5 | restore_frame_suffix_all:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0729 | 0.0000 |
| 6 | erase_frame_suffix_function:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7396 | 0.0000 |
| 7 | restore_frame_suffix_function:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 |
| 8 | erase_frame_suffix_lexical:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3438 | 0.0000 |
| 9 | restore_frame_suffix_lexical:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9583 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 12 | base:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | erase_frame_suffix_final:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7500 | 0.0000 |
| 14 | restore_frame_suffix_final:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 15 | erase_frame_suffix_all:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4271 | 0.0000 |
| 16 | restore_frame_suffix_all:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1146 | 0.0000 |
| 17 | erase_frame_suffix_function:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5417 | 0.0000 |
| 18 | restore_frame_suffix_function:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1042 | 0.0000 |
| 19 | erase_frame_suffix_lexical:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2396 | 0.0000 |
| 20 | restore_frame_suffix_lexical:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0729 | 0.0000 |
| 21 | erase_frame_all_suffix_tokens:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9583 | 0.0000 |
| 22 | restore_frame_all_suffix_tokens:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 23 | base:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | erase_frame_suffix_final:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8021 | 0.0000 |
| 25 | restore_frame_suffix_final:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0729 | 0.0000 |
| 26 | erase_frame_suffix_all:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7292 | 0.0000 |
| 27 | restore_frame_suffix_all:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0521 | 0.0000 |
| 28 | erase_frame_suffix_function:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7812 | 0.0000 |
| 29 | restore_frame_suffix_function:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0729 | 0.0000 |
| 30 | erase_frame_suffix_lexical:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2812 | 0.0000 |
| 31 | restore_frame_suffix_lexical:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0729 | 0.0000 |
| 32 | erase_frame_all_suffix_tokens:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 33 | restore_frame_all_suffix_tokens:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1042 | 0.0000 |
| 34 | base:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 35 | erase_frame_suffix_final:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6979 | 0.0000 |
| 36 | restore_frame_suffix_final:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 37 | erase_frame_suffix_all:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4896 | 0.0000 |
| 38 | restore_frame_suffix_all:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1146 | 0.0000 |
| 39 | erase_frame_suffix_function:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4583 | 0.0000 |
| 40 | restore_frame_suffix_function:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1458 | 0.0000 |
| 41 | erase_frame_suffix_lexical:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 0.0000 |
| 42 | restore_frame_suffix_lexical:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 43 | erase_frame_all_suffix_tokens:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9896 | 0.0000 |
| 44 | restore_frame_all_suffix_tokens:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 45 | base:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 46 | erase_frame_suffix_final:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7396 | 0.0000 |
| 47 | restore_frame_suffix_final:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 48 | erase_frame_suffix_all:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5312 | 0.0000 |
| 49 | restore_frame_suffix_all:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1458 | 0.0000 |
| 50 | erase_frame_suffix_function:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4583 | 0.0000 |
| 51 | restore_frame_suffix_function:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1042 | 0.0000 |
| 52 | erase_frame_suffix_lexical:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4167 | 0.0000 |
| 53 | restore_frame_suffix_lexical:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1146 | 0.0000 |
| 54 | erase_frame_all_suffix_tokens:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9375 | 0.0000 |
| 55 | restore_frame_all_suffix_tokens:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1354 | 0.0000 |
| 56 | base:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 57 | erase_frame_suffix_final:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7604 | 0.0000 |
| 58 | restore_frame_suffix_final:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 |
| 59 | erase_frame_suffix_all:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5729 | 0.0000 |
| 60 | restore_frame_suffix_all:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 61 | erase_frame_suffix_function:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5417 | 0.0000 |
| 62 | restore_frame_suffix_function:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 63 | erase_frame_suffix_lexical:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5625 | 0.0000 |
| 64 | restore_frame_suffix_lexical:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1146 | 0.0000 |
| 65 | erase_frame_all_suffix_tokens:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9896 | 0.0000 |
| 66 | restore_frame_all_suffix_tokens:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1146 | 0.0000 |
| 67 | base:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 68 | erase_frame_suffix_final:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5208 | 0.0000 |
| 69 | restore_frame_suffix_final:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0729 | 0.0000 |
| 70 | erase_frame_suffix_all:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4479 | 0.0000 |
| 71 | restore_frame_suffix_all:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1562 | 0.0000 |
| 72 | erase_frame_suffix_function:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3854 | 0.0000 |
| 73 | restore_frame_suffix_function:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1562 | 0.0000 |
| 74 | erase_frame_suffix_lexical:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3542 | 0.0000 |
| 75 | restore_frame_suffix_lexical:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.0000 |
| 76 | erase_frame_all_suffix_tokens:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9167 | 0.0000 |
| 77 | restore_frame_all_suffix_tokens:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.0000 |

## glm4

items=224, basis_items=224, rows=7392, audit_layers=[4, 10, 20]
module=resid_out, component_rank=24, max_new_tokens=6, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5104 | 0.0000 |
| 3 | restore_frame_suffix_final | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0878 | 0.0000 |
| 4 | erase_frame_suffix_all | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3512 | 0.0000 |
| 5 | restore_frame_suffix_all | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0863 | 0.0000 |
| 6 | erase_frame_suffix_function | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3720 | 0.0000 |
| 7 | restore_frame_suffix_function | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0789 | 0.0000 |
| 8 | erase_frame_suffix_lexical | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2530 | 0.0000 |
| 9 | restore_frame_suffix_lexical | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0655 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9062 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens | 672 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0744 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4018 | 0.0000 |
| 3 | restore_frame_suffix_final:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0893 | 0.0000 |
| 4 | erase_frame_suffix_all:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2857 | 0.0000 |
| 5 | restore_frame_suffix_all:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0893 | 0.0000 |
| 6 | erase_frame_suffix_function:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2991 | 0.0000 |
| 7 | restore_frame_suffix_function:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 8 | erase_frame_suffix_lexical:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2188 | 0.0000 |
| 9 | restore_frame_suffix_lexical:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0580 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8482 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens:L4 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 12 | base:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | erase_frame_suffix_final:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4196 | 0.0000 |
| 14 | restore_frame_suffix_final:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0848 | 0.0000 |
| 15 | erase_frame_suffix_all:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2232 | 0.0000 |
| 16 | restore_frame_suffix_all:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0670 | 0.0000 |
| 17 | erase_frame_suffix_function:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2188 | 0.0000 |
| 18 | restore_frame_suffix_function:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0759 | 0.0000 |
| 19 | erase_frame_suffix_lexical:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2500 | 0.0000 |
| 20 | restore_frame_suffix_lexical:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0759 | 0.0000 |
| 21 | erase_frame_all_suffix_tokens:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9107 | 0.0000 |
| 22 | restore_frame_all_suffix_tokens:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0804 | 0.0000 |
| 23 | base:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | erase_frame_suffix_final:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7098 | 0.0000 |
| 25 | restore_frame_suffix_final:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0893 | 0.0000 |
| 26 | erase_frame_suffix_all:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5446 | 0.0000 |
| 27 | restore_frame_suffix_all:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1027 | 0.0000 |
| 28 | erase_frame_suffix_function:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5982 | 0.0000 |
| 29 | restore_frame_suffix_function:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0670 | 0.0000 |
| 30 | erase_frame_suffix_lexical:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2902 | 0.0000 |
| 31 | restore_frame_suffix_lexical:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 32 | erase_frame_all_suffix_tokens:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9598 | 0.0000 |
| 33 | restore_frame_all_suffix_tokens:L20 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0804 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5208 | 0.0000 |
| 3 | restore_frame_suffix_final:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 |
| 4 | erase_frame_suffix_all:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4062 | 0.0000 |
| 5 | restore_frame_suffix_all:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 6 | erase_frame_suffix_function:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4375 | 0.0000 |
| 7 | restore_frame_suffix_function:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 8 | erase_frame_suffix_lexical:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2083 | 0.0000 |
| 9 | restore_frame_suffix_lexical:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9792 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens:is_a | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 12 | base:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | erase_frame_suffix_final:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 0.0000 |
| 14 | restore_frame_suffix_final:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0729 | 0.0000 |
| 15 | erase_frame_suffix_all:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2500 | 0.0000 |
| 16 | restore_frame_suffix_all:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 17 | erase_frame_suffix_function:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2604 | 0.0000 |
| 18 | restore_frame_suffix_function:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0521 | 0.0000 |
| 19 | erase_frame_suffix_lexical:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1146 | 0.0000 |
| 20 | restore_frame_suffix_lexical:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0521 | 0.0000 |
| 21 | erase_frame_all_suffix_tokens:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8333 | 0.0000 |
| 22 | restore_frame_all_suffix_tokens:used_for | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0521 | 0.0000 |
| 23 | base:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | erase_frame_suffix_final:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6771 | 0.0000 |
| 25 | restore_frame_suffix_final:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 26 | erase_frame_suffix_all:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3958 | 0.0000 |
| 27 | restore_frame_suffix_all:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 28 | erase_frame_suffix_function:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4167 | 0.0000 |
| 29 | restore_frame_suffix_function:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0729 | 0.0000 |
| 30 | erase_frame_suffix_lexical:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2604 | 0.0000 |
| 31 | restore_frame_suffix_lexical:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 32 | erase_frame_all_suffix_tokens:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9479 | 0.0000 |
| 33 | restore_frame_all_suffix_tokens:can_do | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 |
| 34 | base:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 35 | erase_frame_suffix_final:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3438 | 0.0000 |
| 36 | restore_frame_suffix_final:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 37 | erase_frame_suffix_all:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2396 | 0.0000 |
| 38 | restore_frame_suffix_all:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 |
| 39 | erase_frame_suffix_function:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2812 | 0.0000 |
| 40 | restore_frame_suffix_function:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 |
| 41 | erase_frame_suffix_lexical:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2083 | 0.0000 |
| 42 | restore_frame_suffix_lexical:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 0.0000 |
| 43 | erase_frame_all_suffix_tokens:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8438 | 0.0000 |
| 44 | restore_frame_all_suffix_tokens:location | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0625 | 0.0000 |
| 45 | base:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 46 | erase_frame_suffix_final:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6354 | 0.0000 |
| 47 | restore_frame_suffix_final:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 0.0000 |
| 48 | erase_frame_suffix_all:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3958 | 0.0000 |
| 49 | restore_frame_suffix_all:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1042 | 0.0000 |
| 50 | erase_frame_suffix_function:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4479 | 0.0000 |
| 51 | restore_frame_suffix_function:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 52 | erase_frame_suffix_lexical:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.0000 |
| 53 | restore_frame_suffix_lexical:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 54 | erase_frame_all_suffix_tokens:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9062 | 0.0000 |
| 55 | restore_frame_all_suffix_tokens:material | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0729 | 0.0000 |
| 56 | base:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 57 | erase_frame_suffix_final:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6042 | 0.0000 |
| 58 | restore_frame_suffix_final:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1562 | 0.0000 |
| 59 | erase_frame_suffix_all:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5729 | 0.0000 |
| 60 | restore_frame_suffix_all:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1354 | 0.0000 |
| 61 | erase_frame_suffix_function:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5521 | 0.0000 |
| 62 | restore_frame_suffix_function:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1146 | 0.0000 |
| 63 | erase_frame_suffix_lexical:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6250 | 0.0000 |
| 64 | restore_frame_suffix_lexical:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1042 | 0.0000 |
| 65 | erase_frame_all_suffix_tokens:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9688 | 0.0000 |
| 66 | restore_frame_all_suffix_tokens:property | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1458 | 0.0000 |
| 67 | base:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 68 | erase_frame_suffix_final:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4167 | 0.0000 |
| 69 | restore_frame_suffix_final:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0312 | 0.0000 |
| 70 | erase_frame_suffix_all:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1979 | 0.0000 |
| 71 | restore_frame_suffix_all:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0312 | 0.0000 |
| 72 | erase_frame_suffix_function:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2083 | 0.0000 |
| 73 | restore_frame_suffix_function:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 0.0000 |
| 74 | erase_frame_suffix_lexical:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1875 | 0.0000 |
| 75 | restore_frame_suffix_lexical:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0312 | 0.0000 |
| 76 | erase_frame_all_suffix_tokens:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8646 | 0.0000 |
| 77 | restore_frame_all_suffix_tokens:part_of | 96 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0104 | 0.0000 |

## deepseek7b

items=224, basis_items=224, rows=9856, audit_layers=[8, 10, 12, 14]
module=resid_out, component_rank=24, max_new_tokens=6, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5324 | 0.0000 |
| 3 | restore_frame_suffix_final | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1339 | 0.0000 |
| 4 | erase_frame_suffix_all | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4174 | 0.0000 |
| 5 | restore_frame_suffix_all | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1429 | 0.0000 |
| 6 | erase_frame_suffix_function | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4275 | 0.0000 |
| 7 | restore_frame_suffix_function | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1440 | 0.0000 |
| 8 | erase_frame_suffix_lexical | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4252 | 0.0000 |
| 9 | restore_frame_suffix_lexical | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1384 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8984 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens | 896 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1317 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5714 | 0.0000 |
| 3 | restore_frame_suffix_final:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1562 | 0.0000 |
| 4 | erase_frame_suffix_all:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4509 | 0.0000 |
| 5 | restore_frame_suffix_all:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1607 | 0.0000 |
| 6 | erase_frame_suffix_function:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4598 | 0.0000 |
| 7 | restore_frame_suffix_function:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1786 | 0.0000 |
| 8 | erase_frame_suffix_lexical:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4821 | 0.0000 |
| 9 | restore_frame_suffix_lexical:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1295 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9286 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens:L8 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1429 | 0.0000 |
| 12 | base:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | erase_frame_suffix_final:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5625 | 0.0000 |
| 14 | restore_frame_suffix_final:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1473 | 0.0000 |
| 15 | erase_frame_suffix_all:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4107 | 0.0000 |
| 16 | restore_frame_suffix_all:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1384 | 0.0000 |
| 17 | erase_frame_suffix_function:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4107 | 0.0000 |
| 18 | restore_frame_suffix_function:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1607 | 0.0000 |
| 19 | erase_frame_suffix_lexical:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4330 | 0.0000 |
| 20 | restore_frame_suffix_lexical:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1518 | 0.0000 |
| 21 | erase_frame_all_suffix_tokens:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9018 | 0.0000 |
| 22 | restore_frame_all_suffix_tokens:L10 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 23 | base:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | erase_frame_suffix_final:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4643 | 0.0000 |
| 25 | restore_frame_suffix_final:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1161 | 0.0000 |
| 26 | erase_frame_suffix_all:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4107 | 0.0000 |
| 27 | restore_frame_suffix_all:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1473 | 0.0000 |
| 28 | erase_frame_suffix_function:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4375 | 0.0000 |
| 29 | restore_frame_suffix_function:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1027 | 0.0000 |
| 30 | erase_frame_suffix_lexical:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4062 | 0.0000 |
| 31 | restore_frame_suffix_lexical:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1205 | 0.0000 |
| 32 | erase_frame_all_suffix_tokens:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9062 | 0.0000 |
| 33 | restore_frame_all_suffix_tokens:L12 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1339 | 0.0000 |
| 34 | base:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 35 | erase_frame_suffix_final:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5312 | 0.0000 |
| 36 | restore_frame_suffix_final:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1161 | 0.0000 |
| 37 | erase_frame_suffix_all:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3973 | 0.0000 |
| 38 | restore_frame_suffix_all:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 39 | erase_frame_suffix_function:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4018 | 0.0000 |
| 40 | restore_frame_suffix_function:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1339 | 0.0000 |
| 41 | erase_frame_suffix_lexical:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3795 | 0.0000 |
| 42 | restore_frame_suffix_lexical:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1518 | 0.0000 |
| 43 | erase_frame_all_suffix_tokens:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8571 | 0.0000 |
| 44 | restore_frame_all_suffix_tokens:L14 | 224 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5156 | 0.0000 |
| 3 | restore_frame_suffix_final:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 4 | erase_frame_suffix_all:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5703 | 0.0000 |
| 5 | restore_frame_suffix_all:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 6 | erase_frame_suffix_function:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5234 | 0.0000 |
| 7 | restore_frame_suffix_function:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1562 | 0.0000 |
| 8 | erase_frame_suffix_lexical:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4766 | 0.0000 |
| 9 | restore_frame_suffix_lexical:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1328 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9609 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens:is_a | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1172 | 0.0000 |
| 12 | base:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | erase_frame_suffix_final:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5703 | 0.0000 |
| 14 | restore_frame_suffix_final:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1328 | 0.0000 |
| 15 | erase_frame_suffix_all:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3516 | 0.0000 |
| 16 | restore_frame_suffix_all:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0859 | 0.0000 |
| 17 | erase_frame_suffix_function:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3984 | 0.0000 |
| 18 | restore_frame_suffix_function:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1328 | 0.0000 |
| 19 | erase_frame_suffix_lexical:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3906 | 0.0000 |
| 20 | restore_frame_suffix_lexical:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1328 | 0.0000 |
| 21 | erase_frame_all_suffix_tokens:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9609 | 0.0000 |
| 22 | restore_frame_all_suffix_tokens:used_for | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1484 | 0.0000 |
| 23 | base:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | erase_frame_suffix_final:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6875 | 0.0000 |
| 25 | restore_frame_suffix_final:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2031 | 0.0000 |
| 26 | erase_frame_suffix_all:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3438 | 0.0000 |
| 27 | restore_frame_suffix_all:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2188 | 0.0000 |
| 28 | erase_frame_suffix_function:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4531 | 0.0000 |
| 29 | restore_frame_suffix_function:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2109 | 0.0000 |
| 30 | erase_frame_suffix_lexical:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2578 | 0.0000 |
| 31 | restore_frame_suffix_lexical:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1797 | 0.0000 |
| 32 | erase_frame_all_suffix_tokens:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9922 | 0.0000 |
| 33 | restore_frame_all_suffix_tokens:can_do | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1172 | 0.0000 |
| 34 | base:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 35 | erase_frame_suffix_final:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 36 | restore_frame_suffix_final:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 37 | erase_frame_suffix_all:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4297 | 0.0000 |
| 38 | restore_frame_suffix_all:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1875 | 0.0000 |
| 39 | erase_frame_suffix_function:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4219 | 0.0000 |
| 40 | restore_frame_suffix_function:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1562 | 0.0000 |
| 41 | erase_frame_suffix_lexical:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4453 | 0.0000 |
| 42 | restore_frame_suffix_lexical:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 43 | erase_frame_all_suffix_tokens:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8594 | 0.0000 |
| 44 | restore_frame_all_suffix_tokens:location | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1172 | 0.0000 |
| 45 | base:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 46 | erase_frame_suffix_final:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4922 | 0.0000 |
| 47 | restore_frame_suffix_final:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1016 | 0.0000 |
| 48 | erase_frame_suffix_all:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4453 | 0.0000 |
| 49 | restore_frame_suffix_all:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1016 | 0.0000 |
| 50 | erase_frame_suffix_function:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4062 | 0.0000 |
| 51 | restore_frame_suffix_function:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 52 | erase_frame_suffix_lexical:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4219 | 0.0000 |
| 53 | restore_frame_suffix_lexical:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1016 | 0.0000 |
| 54 | erase_frame_all_suffix_tokens:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.8906 | 0.0000 |
| 55 | restore_frame_all_suffix_tokens:material | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1016 | 0.0000 |
| 56 | base:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 57 | erase_frame_suffix_final:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5391 | 0.0000 |
| 58 | restore_frame_suffix_final:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1875 | 0.0000 |
| 59 | erase_frame_suffix_all:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4453 | 0.0000 |
| 60 | restore_frame_suffix_all:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1562 | 0.0000 |
| 61 | erase_frame_suffix_function:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4453 | 0.0000 |
| 62 | restore_frame_suffix_function:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1719 | 0.0000 |
| 63 | erase_frame_suffix_lexical:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6094 | 0.0000 |
| 64 | restore_frame_suffix_lexical:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1797 | 0.0000 |
| 65 | erase_frame_all_suffix_tokens:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.9297 | 0.0000 |
| 66 | restore_frame_all_suffix_tokens:property | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1875 | 0.0000 |
| 67 | base:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 68 | erase_frame_suffix_final:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.4219 | 0.0000 |
| 69 | restore_frame_suffix_final:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0938 | 0.0000 |
| 70 | erase_frame_suffix_all:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3359 | 0.0000 |
| 71 | restore_frame_suffix_all:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| 72 | erase_frame_suffix_function:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3438 | 0.0000 |
| 73 | restore_frame_suffix_function:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0859 | 0.0000 |
| 74 | erase_frame_suffix_lexical:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.3750 | 0.0000 |
| 75 | restore_frame_suffix_lexical:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1172 | 0.0000 |
| 76 | erase_frame_all_suffix_tokens:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.6953 | 0.0000 |
| 77 | restore_frame_all_suffix_tokens:part_of | 128 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.1328 | 0.0000 |
