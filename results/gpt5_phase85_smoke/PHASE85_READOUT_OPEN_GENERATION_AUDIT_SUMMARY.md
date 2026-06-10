# Phase85 Readout Open Generation Audit Summary

## qwen3

items=2, basis_items=224, rows=44, audit_layers=[4, 8]
module=resid_out, component_rank=24, max_new_tokens=6, relations=['is_a', 'part_of']

### By condition

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2500 | 0.0000 |
| 3 | restore_frame_suffix_final | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | erase_frame_suffix_all | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 5 | restore_frame_suffix_all | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | erase_frame_suffix_function | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.2500 | 0.0000 |
| 7 | restore_frame_suffix_function | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | erase_frame_suffix_lexical | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | restore_frame_suffix_lexical | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7500 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 3 | restore_frame_suffix_final:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | erase_frame_suffix_all:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 5 | restore_frame_suffix_all:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | erase_frame_suffix_function:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 7 | restore_frame_suffix_function:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | erase_frame_suffix_lexical:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | restore_frame_suffix_lexical:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens:L4 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | base:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | erase_frame_suffix_final:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 14 | restore_frame_suffix_final:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 15 | erase_frame_suffix_all:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 16 | restore_frame_suffix_all:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 17 | erase_frame_suffix_function:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 18 | restore_frame_suffix_function:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 19 | erase_frame_suffix_lexical:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 20 | restore_frame_suffix_lexical:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 21 | erase_frame_all_suffix_tokens:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 22 | restore_frame_all_suffix_tokens:L8 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | erase_frame_suffix_final:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 3 | restore_frame_suffix_final:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | erase_frame_suffix_all:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 5 | restore_frame_suffix_all:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | erase_frame_suffix_function:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 7 | restore_frame_suffix_function:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | erase_frame_suffix_lexical:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | restore_frame_suffix_lexical:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10 | erase_frame_all_suffix_tokens:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 11 | restore_frame_all_suffix_tokens:is_a | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | base:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | erase_frame_suffix_final:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 14 | restore_frame_suffix_final:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 15 | erase_frame_suffix_all:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 16 | restore_frame_suffix_all:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 17 | erase_frame_suffix_function:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 18 | restore_frame_suffix_function:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 19 | erase_frame_suffix_lexical:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 20 | restore_frame_suffix_lexical:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 21 | erase_frame_all_suffix_tokens:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 22 | restore_frame_all_suffix_tokens:part_of | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
