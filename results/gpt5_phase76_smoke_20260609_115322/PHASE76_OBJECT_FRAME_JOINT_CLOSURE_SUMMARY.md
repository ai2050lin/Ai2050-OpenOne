# Phase76 Object-Frame Joint Closure Summary

## qwen3

items=12, rows=84, layer_pairs=[[4, 8]]

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | object_only_matched | 12 | 9 | 11.7832 | 3.7540 | -8.4594 | -5.1122 | 0.0000 | 0.1111 |
| 2 | joint_restore_frame_only | 12 | 9 | 10.1625 | 4.7499 | -6.8387 | -4.1163 | 0.1111 | 0.1111 |
| 3 | joint_matched | 12 | 9 | 10.1294 | 10.1984 | -6.8055 | 1.3322 | 0.0000 | 0.5556 |
| 4 | joint_mismatched_frame | 12 | 9 | 9.6053 | 7.8300 | -6.2815 | -1.0363 | 0.0000 | 0.3333 |
| 5 | joint_restore_object_only | 12 | 9 | 4.7480 | 5.7889 | -1.4242 | -3.0774 | 0.3333 | 0.2222 |
| 6 | frame_only_matched | 12 | 9 | 3.8703 | 5.4902 | -0.5465 | -3.3761 | 0.5556 | 0.2222 |
| 7 | joint_restore_both | 12 | 9 | 1.2836 | 1.1596 | 2.0402 | -7.7066 | 0.6667 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | object_only_matched:L4->L8 | 12 | 9 | 11.7832 | 3.7540 | -8.4594 | -5.1122 | 0.0000 | 0.1111 |
| 2 | joint_restore_frame_only:L4->L8 | 12 | 9 | 10.1625 | 4.7499 | -6.8387 | -4.1163 | 0.1111 | 0.1111 |
| 3 | joint_matched:L4->L8 | 12 | 9 | 10.1294 | 10.1984 | -6.8055 | 1.3322 | 0.0000 | 0.5556 |
| 4 | joint_mismatched_frame:L4->L8 | 12 | 9 | 9.6053 | 7.8300 | -6.2815 | -1.0363 | 0.0000 | 0.3333 |
| 5 | joint_restore_object_only:L4->L8 | 12 | 9 | 4.7480 | 5.7889 | -1.4242 | -3.0774 | 0.3333 | 0.2222 |
| 6 | frame_only_matched:L4->L8 | 12 | 9 | 3.8703 | 5.4902 | -0.5465 | -3.3761 | 0.5556 | 0.2222 |
| 7 | joint_restore_both:L4->L8 | 12 | 9 | 1.2836 | 1.1596 | 2.0402 | -7.7066 | 0.6667 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | object_only_matched:is_a | 2 | 2 | 17.9167 | 10.9916 | -10.2610 | 0.4810 | 0.0000 | 0.5000 |
| 2 | joint_restore_frame_only:is_a | 2 | 2 | 16.2236 | 10.8968 | -8.5679 | 0.3862 | 0.0000 | 0.5000 |
| 3 | joint_matched:property | 2 | 1 | 15.9126 | 22.2222 | -14.6990 | 12.8150 | 0.0000 | 1.0000 |
| 4 | joint_mismatched_frame:can_do | 2 | 2 | 15.7910 | 14.0005 | -10.5964 | 3.7196 | 0.0000 | 1.0000 |
| 5 | object_only_matched:property | 2 | 1 | 15.6955 | 2.9120 | -14.4819 | -6.4951 | 0.0000 | 0.0000 |
| 6 | joint_restore_frame_only:property | 2 | 1 | 13.1826 | 5.0888 | -11.9690 | -4.3183 | 0.0000 | 0.0000 |
| 7 | frame_only_matched:property | 2 | 1 | 13.0764 | 16.8895 | -11.8628 | 7.4824 | 0.0000 | 1.0000 |
| 8 | joint_matched:can_do | 2 | 2 | 12.3197 | 9.5711 | -7.1251 | -0.7099 | 0.0000 | 0.5000 |
| 9 | object_only_matched:can_do | 2 | 2 | 12.3046 | 0.7607 | -7.1100 | -9.5202 | 0.0000 | 0.0000 |
| 10 | joint_matched:is_a | 2 | 2 | 12.2893 | 13.2385 | -4.6336 | 2.7279 | 0.0000 | 1.0000 |
| 11 | joint_mismatched_frame:is_a | 2 | 2 | 11.3763 | 8.3930 | -3.7206 | -2.1176 | 0.0000 | 0.0000 |
| 12 | joint_mismatched_frame:used_for | 2 | 1 | 10.5926 | 1.7027 | -9.5853 | -4.4338 | 0.0000 | 0.0000 |
| 13 | joint_restore_frame_only:can_do | 2 | 2 | 9.7289 | 1.6097 | -4.5343 | -8.6712 | 0.0000 | 0.0000 |
| 14 | joint_restore_object_only:property | 2 | 1 | 8.7668 | 16.9603 | -7.5532 | 7.5532 | 0.0000 | 1.0000 |
| 15 | object_only_matched:location | 2 | 1 | 8.1441 | 1.3336 | -7.5501 | -5.3874 | 0.0000 | 0.0000 |
| 16 | joint_restore_frame_only:used_for | 2 | 1 | 8.0864 | 0.5229 | -7.0791 | -5.6135 | 0.0000 | 0.0000 |
| 17 | object_only_matched:used_for | 2 | 1 | 7.5760 | -0.6737 | -6.5688 | -6.8102 | 0.0000 | 0.0000 |
| 18 | joint_matched:used_for | 2 | 1 | 7.4896 | 2.2776 | -6.4824 | -3.8589 | 0.0000 | 0.0000 |
| 19 | joint_matched:material | 2 | 2 | 7.2750 | 7.5481 | -6.5755 | -0.4262 | 0.0000 | 0.5000 |
| 20 | object_only_matched:material | 2 | 2 | 7.0955 | 3.3549 | -6.3960 | -4.6194 | 0.0000 | 0.0000 |
| 21 | joint_restore_frame_only:material | 2 | 2 | 6.6987 | 3.7263 | -5.9992 | -4.2480 | 0.5000 | 0.0000 |
| 22 | joint_mismatched_frame:material | 2 | 2 | 6.4686 | 6.8503 | -5.7691 | -1.1240 | 0.0000 | 0.5000 |
| 23 | joint_restore_object_only:can_do | 2 | 2 | 5.8053 | 9.1190 | -0.6107 | -1.1619 | 0.5000 | 0.5000 |
| 24 | joint_mismatched_frame:property | 2 | 1 | 5.7216 | 4.3597 | -4.5080 | -5.0474 | 0.0000 | 0.0000 |
| 25 | frame_only_matched:can_do | 2 | 2 | 5.2205 | 10.3069 | -0.0260 | 0.0260 | 0.5000 | 0.5000 |
| 26 | joint_restore_frame_only:location | 2 | 1 | 4.8912 | 4.6721 | -4.2972 | -2.0489 | 0.0000 | 0.0000 |
| 27 | joint_restore_object_only:is_a | 2 | 2 | 4.7906 | 6.4155 | 2.8651 | -4.0951 | 1.0000 | 0.0000 |
| 28 | joint_matched:location | 2 | 1 | 3.9941 | 6.5708 | -3.4001 | -0.1502 | 0.0000 | 0.0000 |
| 29 | joint_restore_object_only:material | 2 | 2 | 3.7398 | 1.4662 | -3.0404 | -6.5081 | 0.0000 | 0.0000 |
| 30 | frame_only_matched:material | 2 | 2 | 3.6709 | 2.4791 | -2.9714 | -5.4952 | 0.5000 | 0.0000 |
| 31 | joint_restore_both:used_for | 2 | 1 | 3.3771 | 0.2316 | -2.3698 | -5.9049 | 0.0000 | 0.0000 |
| 32 | joint_restore_object_only:used_for | 2 | 1 | 2.9265 | -1.1553 | -1.9193 | -7.2918 | 0.0000 | 0.0000 |
| 33 | joint_mismatched_frame:location | 2 | 1 | 2.8615 | 5.9197 | -2.2675 | -0.8013 | 0.0000 | 0.0000 |
| 34 | joint_restore_object_only:location | 2 | 1 | 2.3676 | 2.2937 | -1.7737 | -4.4273 | 0.0000 | 0.0000 |
| 35 | joint_restore_both:can_do | 2 | 2 | 1.8749 | -0.4025 | 3.3197 | -10.6834 | 1.0000 | 0.0000 |
| 36 | frame_only_matched:location | 2 | 1 | 1.6447 | -0.0330 | -1.0508 | -6.7540 | 0.0000 | 0.0000 |
| 37 | frame_only_matched:is_a | 2 | 2 | 1.5365 | 3.7818 | 6.1193 | -6.7288 | 1.0000 | 0.0000 |
| 38 | joint_restore_both:location | 2 | 1 | 1.5286 | 1.6760 | -0.9346 | -5.0450 | 0.0000 | 0.0000 |
| 39 | joint_restore_both:material | 2 | 2 | 0.8418 | 0.1819 | -0.1423 | -7.7924 | 0.5000 | 0.0000 |
| 40 | joint_restore_both:property | 2 | 1 | 0.5595 | 8.2752 | 0.6541 | -1.1320 | 1.0000 | 0.0000 |
| 41 | joint_restore_both:is_a | 2 | 2 | 0.3272 | 0.3475 | 7.3285 | -10.1631 | 1.0000 | 0.0000 |
| 42 | frame_only_matched:used_for | 2 | 1 | -0.7439 | -0.5807 | 1.7511 | -6.7172 | 1.0000 | 0.0000 |

## glm4

missing

## deepseek7b

missing

