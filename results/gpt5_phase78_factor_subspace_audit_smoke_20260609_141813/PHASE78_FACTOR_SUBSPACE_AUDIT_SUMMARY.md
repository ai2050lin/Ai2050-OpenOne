# Phase78 Factor Subspace Audit Summary

## qwen3

items=28, basis_items=None, rows=196, layer_pairs=[[4, 8]]
module=resid_out, basis_rank=16, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched | 28 | 19 | 9.6297 | 10.7596 | -4.5994 | -2.6868 | 0.1579 | 0.3158 |
| 2 | joint_subspace_mismatched_frame | 28 | 19 | 9.2104 | 11.0917 | -4.1801 | -2.3546 | 0.2105 | 0.2632 |
| 3 | joint_subspace_restore_object_only | 28 | 19 | 4.4681 | 6.8239 | 0.5622 | -6.6224 | 0.5263 | 0.1053 |
| 4 | frame_subspace_matched | 28 | 19 | 3.9284 | 5.9073 | 1.1019 | -7.5390 | 0.6842 | 0.1053 |
| 5 | joint_subspace_restore_frame_only | 28 | 19 | 8.1300 | 5.7798 | -3.0997 | -7.6665 | 0.3158 | 0.0000 |
| 6 | object_subspace_matched | 28 | 19 | 8.1259 | 5.4379 | -3.0956 | -8.0084 | 0.3158 | 0.0000 |
| 7 | joint_subspace_restore_both | 28 | 19 | 1.3928 | 1.4616 | 3.6375 | -11.9847 | 0.8947 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched:L4->L8 | 28 | 19 | 9.6297 | 10.7596 | -4.5994 | -2.6868 | 0.1579 | 0.3158 |
| 2 | joint_subspace_mismatched_frame:L4->L8 | 28 | 19 | 9.2104 | 11.0917 | -4.1801 | -2.3546 | 0.2105 | 0.2632 |
| 3 | joint_subspace_restore_object_only:L4->L8 | 28 | 19 | 4.4681 | 6.8239 | 0.5622 | -6.6224 | 0.5263 | 0.1053 |
| 4 | frame_subspace_matched:L4->L8 | 28 | 19 | 3.9284 | 5.9073 | 1.1019 | -7.5390 | 0.6842 | 0.1053 |
| 5 | joint_subspace_restore_frame_only:L4->L8 | 28 | 19 | 8.1300 | 5.7798 | -3.0997 | -7.6665 | 0.3158 | 0.0000 |
| 6 | object_subspace_matched:L4->L8 | 28 | 19 | 8.1259 | 5.4379 | -3.0956 | -8.0084 | 0.3158 | 0.0000 |
| 7 | joint_subspace_restore_both:L4->L8 | 28 | 19 | 1.3928 | 1.4616 | 3.6375 | -11.9847 | 0.8947 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_mismatched_frame:part_of | 5 | 5 | 10.8417 | 15.7623 | -7.2512 | 2.2794 | 0.0000 | 0.6000 |
| 2 | joint_subspace_matched:can_do | 3 | 2 | 16.4159 | 17.2608 | -7.9252 | 1.7619 | 0.0000 | 0.5000 |
| 3 | joint_subspace_mismatched_frame:can_do | 3 | 2 | 11.5130 | 14.7229 | -3.0223 | -0.7760 | 0.0000 | 0.5000 |
| 4 | joint_subspace_restore_object_only:can_do | 3 | 2 | 12.0763 | 13.4622 | -3.5856 | -2.0366 | 0.0000 | 0.5000 |
| 5 | joint_subspace_matched:is_a | 5 | 2 | 14.4988 | 11.0975 | -5.7963 | 0.2568 | 0.0000 | 0.5000 |
| 6 | joint_subspace_matched:location | 6 | 6 | 8.5406 | 11.0277 | -4.3967 | -2.6854 | 0.1667 | 0.5000 |
| 7 | frame_subspace_matched:location | 6 | 6 | 4.3251 | 7.4576 | -0.1812 | -6.2555 | 0.5000 | 0.3333 |
| 8 | joint_subspace_matched:part_of | 5 | 5 | 9.3986 | 9.9237 | -5.8082 | -3.5592 | 0.0000 | 0.2000 |
| 9 | joint_subspace_mismatched_frame:location | 6 | 6 | 6.8061 | 10.1971 | -2.6622 | -3.5160 | 0.3333 | 0.1667 |
| 10 | joint_subspace_restore_object_only:location | 6 | 6 | 4.2213 | 7.0959 | -0.0774 | -6.6172 | 0.5000 | 0.1667 |
| 11 | frame_subspace_matched:can_do | 3 | 2 | 10.3040 | 12.5117 | -1.8133 | -2.9871 | 0.5000 | 0.0000 |
| 12 | joint_subspace_mismatched_frame:used_for | 3 | 2 | 10.7263 | 10.8640 | -6.1089 | -5.2941 | 0.0000 | 0.0000 |
| 13 | joint_subspace_matched:used_for | 3 | 2 | 8.5230 | 8.9819 | -3.9056 | -7.1761 | 0.0000 | 0.0000 |
| 14 | object_subspace_matched:used_for | 3 | 2 | 9.1881 | 8.0844 | -4.5707 | -8.0737 | 0.0000 | 0.0000 |
| 15 | object_subspace_matched:part_of | 5 | 5 | 8.4961 | 7.5239 | -4.9057 | -5.9589 | 0.2000 | 0.0000 |
| 16 | joint_subspace_restore_object_only:is_a | 5 | 2 | 8.7805 | 7.4468 | -0.0780 | -3.3939 | 0.5000 | 0.0000 |
| 17 | joint_subspace_restore_frame_only:used_for | 3 | 2 | 8.6114 | 7.3377 | -3.9941 | -8.8203 | 0.0000 | 0.0000 |
| 18 | joint_subspace_restore_object_only:part_of | 5 | 5 | 2.3012 | 7.2030 | 1.2892 | -6.2799 | 0.6000 | 0.0000 |
| 19 | joint_subspace_matched:property | 3 | 2 | 2.9264 | 6.9833 | 1.6430 | -3.4129 | 1.0000 | 0.0000 |
| 20 | joint_subspace_restore_frame_only:part_of | 5 | 5 | 8.2769 | 6.7320 | -4.6865 | -6.7509 | 0.2000 | 0.0000 |
| 21 | joint_subspace_restore_frame_only:property | 3 | 2 | 4.5681 | 6.3138 | 0.0013 | -4.0823 | 0.5000 | 0.0000 |
| 22 | frame_subspace_matched:is_a | 5 | 2 | 6.9044 | 6.2886 | 1.7981 | -4.5520 | 1.0000 | 0.0000 |
| 23 | joint_subspace_mismatched_frame:is_a | 5 | 2 | 13.0814 | 5.8639 | -4.3789 | -4.9767 | 0.5000 | 0.0000 |
| 24 | object_subspace_matched:location | 6 | 6 | 6.9780 | 5.6508 | -2.8341 | -8.0623 | 0.3333 | 0.0000 |
| 25 | joint_subspace_restore_object_only:used_for | 3 | 2 | 3.3457 | 5.6371 | 1.2717 | -10.5209 | 0.5000 | 0.0000 |
| 26 | joint_subspace_restore_frame_only:location | 6 | 6 | 6.5632 | 5.5696 | -2.4193 | -8.1435 | 0.3333 | 0.0000 |
| 27 | frame_subspace_matched:used_for | 3 | 2 | 2.6508 | 5.5128 | 1.9666 | -10.6452 | 0.5000 | 0.0000 |
| 28 | frame_subspace_matched:part_of | 5 | 5 | 2.1957 | 4.7767 | 1.3948 | -8.7062 | 0.8000 | 0.0000 |
| 29 | joint_subspace_restore_frame_only:is_a | 5 | 2 | 16.0128 | 4.3622 | -7.3103 | -6.4784 | 0.5000 | 0.0000 |
| 30 | joint_subspace_mismatched_frame:property | 3 | 2 | 4.6552 | 3.9234 | -0.0859 | -6.4727 | 0.5000 | 0.0000 |
| 31 | object_subspace_matched:property | 3 | 2 | 3.6857 | 3.7690 | 0.8836 | -6.6272 | 0.5000 | 0.0000 |
| 32 | joint_subspace_restore_frame_only:can_do | 3 | 2 | 7.6606 | 3.3559 | 0.8301 | -12.1430 | 0.5000 | 0.0000 |
| 33 | object_subspace_matched:is_a | 5 | 2 | 15.0755 | 3.0883 | -6.3730 | -7.7524 | 0.5000 | 0.0000 |
| 34 | joint_subspace_restore_both:used_for | 3 | 2 | 1.5086 | 2.6773 | 3.1088 | -13.4808 | 0.5000 | 0.0000 |
| 35 | joint_subspace_restore_both:property | 3 | 2 | 2.3235 | 2.3918 | 2.2458 | -8.0044 | 1.0000 | 0.0000 |
| 36 | joint_subspace_restore_both:can_do | 3 | 2 | 6.3580 | 2.1664 | 2.1327 | -13.3325 | 1.0000 | 0.0000 |
| 37 | joint_subspace_restore_both:location | 6 | 6 | 0.4264 | 1.1315 | 3.7175 | -12.5816 | 0.8333 | 0.0000 |
| 38 | object_subspace_matched:can_do | 3 | 2 | 7.0721 | 0.9564 | 1.4186 | -14.5425 | 0.5000 | 0.0000 |
| 39 | joint_subspace_restore_both:part_of | 5 | 5 | 0.3173 | 0.9444 | 3.2731 | -12.5385 | 1.0000 | 0.0000 |
| 40 | joint_subspace_restore_both:is_a | 5 | 2 | 0.9691 | 0.8945 | 7.7334 | -9.9462 | 1.0000 | 0.0000 |
| 41 | object_subspace_matched:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 42 | frame_subspace_matched:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 43 | joint_subspace_matched:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 44 | joint_subspace_mismatched_frame:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 45 | joint_subspace_restore_object_only:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 46 | joint_subspace_restore_frame_only:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 47 | joint_subspace_restore_both:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 48 | joint_subspace_restore_object_only:property | 3 | 2 | -0.1724 | -1.0143 | 4.7417 | -11.4105 | 1.0000 | 0.0000 |
| 49 | frame_subspace_matched:property | 3 | 2 | -1.0043 | -2.5082 | 5.5736 | -12.9043 | 1.0000 | 0.0000 |

## glm4

missing

## deepseek7b

missing

