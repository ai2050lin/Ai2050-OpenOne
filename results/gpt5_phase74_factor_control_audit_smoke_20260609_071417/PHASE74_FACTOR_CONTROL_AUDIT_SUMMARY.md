# Phase74 Factor Control Audit Summary

## qwen3

items=6, rows=6, layer_pairs=[[4, 8]]
control_types=['wrong_target_same_relation_frame', 'same_target_same_relation_frame', 'same_object_same_relation_other_frame', 'same_object_different_relation']

### By control type

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame | 6 | 4 | 7.5205 | 6.0075 | 1.5130 | 0.2500 | 1.0000 |

### Top control paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:L4->L8:object_last | 6 | 4 | 7.5205 | 6.0075 | 1.5130 | 0.2500 | 1.0000 |

### Top control relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:is_a | 1 | 1 | 11.8688 | 10.0815 | 1.7873 | 0.0000 | 1.0000 |
| 2 | wrong_target_same_relation_frame:used_for | 1 | 1 | 8.7703 | 6.8840 | 1.8863 | 0.0000 | 1.0000 |
| 3 | wrong_target_same_relation_frame:material | 1 | 1 | 6.5678 | 4.3146 | 2.2532 | 1.0000 | 1.0000 |
| 4 | wrong_target_same_relation_frame:property | 1 | 1 | 2.8750 | 2.7500 | 0.1250 | 0.0000 | 1.0000 |
| 5 | wrong_target_same_relation_frame:part_of | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | wrong_target_same_relation_frame:location | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Top control relation paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:is_a:L4->L8:object_last | 1 | 1 | 11.8688 | 10.0815 | 1.7873 | 0.0000 | 1.0000 |
| 2 | wrong_target_same_relation_frame:used_for:L4->L8:object_last | 1 | 1 | 8.7703 | 6.8840 | 1.8863 | 0.0000 | 1.0000 |
| 3 | wrong_target_same_relation_frame:material:L4->L8:object_last | 1 | 1 | 6.5678 | 4.3146 | 2.2532 | 1.0000 | 1.0000 |
| 4 | wrong_target_same_relation_frame:property:L4->L8:object_last | 1 | 1 | 2.8750 | 2.7500 | 0.1250 | 0.0000 | 1.0000 |
| 5 | wrong_target_same_relation_frame:part_of:L4->L8:object_last | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | wrong_target_same_relation_frame:location:L4->L8:object_last | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## glm4

missing

## deepseek7b

missing

