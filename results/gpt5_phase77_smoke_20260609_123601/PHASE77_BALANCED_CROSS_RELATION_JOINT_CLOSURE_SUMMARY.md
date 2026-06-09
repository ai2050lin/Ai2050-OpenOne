# Phase77 Balanced Cross-Relation Joint Closure Summary

## qwen3

objects=24, items=28, rows=196, layer_pairs=[[4, 8]]
relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched | 28 | 19 | 9.5482 | 10.8594 | -4.5179 | -2.5870 | 0.1579 | 0.3158 |
| 2 | joint_mismatched_frame | 28 | 19 | 9.3549 | 11.1936 | -4.3247 | -2.2527 | 0.2105 | 0.2632 |
| 3 | joint_restore_object_only | 28 | 19 | 4.8141 | 7.1685 | 0.2162 | -6.2779 | 0.4737 | 0.2632 |
| 4 | frame_only_matched | 28 | 19 | 3.9657 | 5.8811 | 1.0645 | -7.5652 | 0.6842 | 0.1053 |
| 5 | joint_restore_frame_only | 28 | 19 | 8.1698 | 5.9788 | -3.1395 | -7.4675 | 0.3158 | 0.0000 |
| 6 | object_only_matched | 28 | 19 | 8.4190 | 5.5782 | -3.3887 | -7.8681 | 0.3158 | 0.0000 |
| 7 | joint_restore_both | 28 | 19 | 0.8024 | 0.6376 | 4.2278 | -12.8087 | 1.0000 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:L4->L8 | 28 | 19 | 9.5482 | 10.8594 | -4.5179 | -2.5870 | 0.1579 | 0.3158 |
| 2 | joint_mismatched_frame:L4->L8 | 28 | 19 | 9.3549 | 11.1936 | -4.3247 | -2.2527 | 0.2105 | 0.2632 |
| 3 | joint_restore_object_only:L4->L8 | 28 | 19 | 4.8141 | 7.1685 | 0.2162 | -6.2779 | 0.4737 | 0.2632 |
| 4 | frame_only_matched:L4->L8 | 28 | 19 | 3.9657 | 5.8811 | 1.0645 | -7.5652 | 0.6842 | 0.1053 |
| 5 | joint_restore_frame_only:L4->L8 | 28 | 19 | 8.1698 | 5.9788 | -3.1395 | -7.4675 | 0.3158 | 0.0000 |
| 6 | object_only_matched:L4->L8 | 28 | 19 | 8.4190 | 5.5782 | -3.3887 | -7.8681 | 0.3158 | 0.0000 |
| 7 | joint_restore_both:L4->L8 | 28 | 19 | 0.8024 | 0.6376 | 4.2278 | -12.8087 | 1.0000 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_mismatched_frame:part_of | 5 | 5 | 10.7475 | 15.4061 | -7.1571 | 1.9232 | 0.0000 | 0.6000 |
| 2 | joint_matched:can_do | 3 | 2 | 16.4365 | 17.7815 | -7.9458 | 2.2827 | 0.0000 | 0.5000 |
| 3 | joint_mismatched_frame:can_do | 3 | 2 | 12.2963 | 15.7638 | -3.8056 | 0.2649 | 0.0000 | 0.5000 |
| 4 | joint_restore_object_only:can_do | 3 | 2 | 11.5533 | 13.7595 | -3.0626 | -1.7393 | 0.0000 | 0.5000 |
| 5 | joint_matched:is_a | 5 | 2 | 14.1639 | 11.2921 | -5.4614 | 0.4515 | 0.0000 | 0.5000 |
| 6 | joint_matched:location | 6 | 6 | 9.0166 | 10.8690 | -4.8727 | -2.8441 | 0.1667 | 0.5000 |
| 7 | joint_restore_object_only:location | 6 | 6 | 5.4716 | 7.9742 | -1.3277 | -5.7389 | 0.3333 | 0.5000 |
| 8 | frame_only_matched:location | 6 | 6 | 4.4275 | 7.4572 | -0.2836 | -6.2559 | 0.5000 | 0.3333 |
| 9 | joint_matched:part_of | 5 | 5 | 8.6285 | 10.1100 | -5.0381 | -3.3729 | 0.0000 | 0.2000 |
| 10 | joint_restore_object_only:part_of | 5 | 5 | 2.6620 | 7.5345 | 0.9284 | -5.9484 | 0.6000 | 0.2000 |
| 11 | joint_mismatched_frame:location | 6 | 6 | 7.2467 | 10.2763 | -3.1028 | -3.4368 | 0.3333 | 0.1667 |
| 12 | frame_only_matched:can_do | 3 | 2 | 10.4391 | 12.5101 | -1.9485 | -2.9888 | 0.5000 | 0.0000 |
| 13 | joint_mismatched_frame:used_for | 3 | 2 | 10.5965 | 10.1246 | -5.9791 | -6.0334 | 0.0000 | 0.0000 |
| 14 | joint_matched:used_for | 3 | 2 | 8.4063 | 8.5842 | -3.7890 | -7.5739 | 0.0000 | 0.0000 |
| 15 | object_only_matched:used_for | 3 | 2 | 9.1786 | 8.0291 | -4.5612 | -8.1289 | 0.0000 | 0.0000 |
| 16 | joint_restore_frame_only:used_for | 3 | 2 | 9.0955 | 7.8117 | -4.4781 | -8.3464 | 0.0000 | 0.0000 |
| 17 | joint_matched:property | 3 | 2 | 3.0799 | 7.6244 | 1.4895 | -2.7717 | 1.0000 | 0.0000 |
| 18 | joint_restore_frame_only:part_of | 5 | 5 | 8.1047 | 7.2834 | -4.5143 | -6.1995 | 0.2000 | 0.0000 |
| 19 | object_only_matched:part_of | 5 | 5 | 8.9508 | 7.2617 | -5.3604 | -6.2212 | 0.2000 | 0.0000 |
| 20 | joint_restore_object_only:is_a | 5 | 2 | 8.7265 | 7.1036 | -0.0240 | -3.7370 | 0.5000 | 0.0000 |
| 21 | frame_only_matched:is_a | 5 | 2 | 7.0197 | 6.3979 | 1.6828 | -4.4427 | 1.0000 | 0.0000 |
| 22 | joint_mismatched_frame:is_a | 5 | 2 | 12.6945 | 6.2059 | -3.9920 | -4.6348 | 0.5000 | 0.0000 |
| 23 | joint_restore_frame_only:location | 6 | 6 | 6.0864 | 6.1378 | -1.9425 | -7.5753 | 0.3333 | 0.0000 |
| 24 | object_only_matched:location | 6 | 6 | 6.9752 | 6.0985 | -2.8313 | -7.6146 | 0.3333 | 0.0000 |
| 25 | joint_restore_frame_only:property | 3 | 2 | 5.2425 | 5.5800 | -0.6732 | -4.8161 | 0.5000 | 0.0000 |
| 26 | frame_only_matched:used_for | 3 | 2 | 2.6744 | 5.3907 | 1.9430 | -10.7674 | 0.5000 | 0.0000 |
| 27 | joint_restore_object_only:used_for | 3 | 2 | 2.8272 | 5.1241 | 1.7902 | -11.0340 | 0.5000 | 0.0000 |
| 28 | joint_mismatched_frame:property | 3 | 2 | 4.6756 | 4.9009 | -0.1063 | -5.4952 | 0.5000 | 0.0000 |
| 29 | frame_only_matched:part_of | 5 | 5 | 2.2407 | 4.7442 | 1.3497 | -8.7387 | 0.8000 | 0.0000 |
| 30 | object_only_matched:property | 3 | 2 | 3.8960 | 4.3689 | 0.6733 | -6.0272 | 0.5000 | 0.0000 |
| 31 | joint_restore_frame_only:is_a | 5 | 2 | 15.1644 | 4.1749 | -6.4618 | -6.6658 | 0.5000 | 0.0000 |
| 32 | object_only_matched:is_a | 5 | 2 | 14.8764 | 3.3243 | -6.1739 | -7.5164 | 0.5000 | 0.0000 |
| 33 | joint_restore_frame_only:can_do | 3 | 2 | 9.5897 | 2.6108 | -1.0990 | -12.8880 | 0.5000 | 0.0000 |
| 34 | joint_restore_both:used_for | 3 | 2 | -0.1220 | 1.9162 | 4.7394 | -14.2419 | 1.0000 | 0.0000 |
| 35 | joint_restore_both:part_of | 5 | 5 | 0.3156 | 1.0880 | 3.2748 | -12.3948 | 1.0000 | 0.0000 |
| 36 | joint_restore_both:is_a | 5 | 2 | 1.7767 | 0.8803 | 6.9258 | -9.9604 | 1.0000 | 0.0000 |
| 37 | object_only_matched:can_do | 3 | 2 | 8.7270 | 0.8212 | -0.2363 | -14.6777 | 0.5000 | 0.0000 |
| 38 | joint_restore_both:property | 3 | 2 | 0.2135 | 0.4015 | 4.3558 | -9.9946 | 1.0000 | 0.0000 |
| 39 | joint_restore_both:can_do | 3 | 2 | 5.7774 | 0.2205 | 2.7133 | -15.2783 | 1.0000 | 0.0000 |
| 40 | object_only_matched:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 41 | frame_only_matched:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 42 | joint_matched:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 43 | joint_mismatched_frame:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 44 | joint_restore_object_only:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 45 | joint_restore_frame_only:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 46 | joint_restore_both:material | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 47 | joint_restore_both:location | 6 | 6 | -0.2705 | -0.0271 | 4.4144 | -13.7402 | 1.0000 | 0.0000 |
| 48 | joint_restore_object_only:property | 3 | 2 | -0.4431 | -0.6456 | 5.0124 | -11.0417 | 1.0000 | 0.0000 |
| 49 | frame_only_matched:property | 3 | 2 | -1.3430 | -2.6599 | 5.9123 | -13.0560 | 1.0000 | 0.0000 |

## glm4

missing

## deepseek7b

missing

