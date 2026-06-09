# Phase79 Rank Sweep Remainder Audit Summary

## qwen3

items=28, basis_items=28, rows=224, layer_pairs=[[4, 8]]
module=resid_out, ranks=[4, 8], relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By rank and condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | rank8:joint_subspace_mismatched_frame | 28 | 23 | 6.6698 | 8.0630 | -1.9624 | -4.7912 | 0.3478 | 0.2174 |
| 2 | rank8:joint_subspace_matched | 28 | 23 | 6.5142 | 8.2556 | -1.8068 | -4.5985 | 0.3913 | 0.1739 |
| 3 | rank4:joint_remainder_matched | 28 | 23 | 6.2354 | 6.4771 | -1.5279 | -6.3770 | 0.4348 | 0.0870 |
| 4 | rank4:joint_subspace_matched | 28 | 23 | 2.9303 | 5.2889 | 1.7772 | -7.5653 | 0.6522 | 0.0870 |
| 5 | rank4:joint_subspace_mismatched_frame | 28 | 23 | 3.3559 | 3.8438 | 1.3516 | -9.0104 | 0.6957 | 0.0435 |
| 6 | rank8:joint_remainder_matched | 28 | 23 | 2.8380 | 3.2416 | 1.8695 | -9.6125 | 0.6957 | 0.0435 |
| 7 | rank4:joint_subspace_restore_both | 28 | 23 | 1.0259 | 1.5592 | 3.6816 | -11.2950 | 0.9130 | 0.0000 |
| 8 | rank8:joint_subspace_restore_both | 28 | 23 | 1.2938 | 1.5532 | 3.4137 | -11.3010 | 0.8696 | 0.0000 |

### Top rank-condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | rank8:joint_subspace_mismatched_frame:L4->L8 | 28 | 23 | 6.6698 | 8.0630 | -1.9624 | -4.7912 | 0.3478 | 0.2174 |
| 2 | rank8:joint_subspace_matched:L4->L8 | 28 | 23 | 6.5142 | 8.2556 | -1.8068 | -4.5985 | 0.3913 | 0.1739 |
| 3 | rank4:joint_remainder_matched:L4->L8 | 28 | 23 | 6.2354 | 6.4771 | -1.5279 | -6.3770 | 0.4348 | 0.0870 |
| 4 | rank4:joint_subspace_matched:L4->L8 | 28 | 23 | 2.9303 | 5.2889 | 1.7772 | -7.5653 | 0.6522 | 0.0870 |
| 5 | rank4:joint_subspace_mismatched_frame:L4->L8 | 28 | 23 | 3.3559 | 3.8438 | 1.3516 | -9.0104 | 0.6957 | 0.0435 |
| 6 | rank8:joint_remainder_matched:L4->L8 | 28 | 23 | 2.8380 | 3.2416 | 1.8695 | -9.6125 | 0.6957 | 0.0435 |
| 7 | rank4:joint_subspace_restore_both:L4->L8 | 28 | 23 | 1.0259 | 1.5592 | 3.6816 | -11.2950 | 0.9130 | 0.0000 |
| 8 | rank8:joint_subspace_restore_both:L4->L8 | 28 | 23 | 1.2938 | 1.5532 | 3.4137 | -11.3010 | 0.8696 | 0.0000 |

### Top rank-condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | rank8:joint_subspace_mismatched_frame:part_of | 5 | 5 | 8.3773 | 12.5572 | -4.4312 | -0.9257 | 0.2000 | 0.6000 |
| 2 | rank8:joint_subspace_matched:material | 3 | 2 | 8.0508 | 6.1576 | -3.7774 | -0.7368 | 0.0000 | 0.5000 |
| 3 | rank8:joint_subspace_mismatched_frame:material | 3 | 2 | 10.1807 | 5.9051 | -5.9073 | -0.9894 | 0.0000 | 0.5000 |
| 4 | rank4:joint_remainder_matched:material | 3 | 2 | 4.0238 | 5.1311 | 0.2496 | -1.7634 | 0.5000 | 0.5000 |
| 5 | rank4:joint_subspace_mismatched_frame:material | 3 | 2 | 4.2718 | 3.6957 | 0.0016 | -3.1988 | 0.5000 | 0.5000 |
| 6 | rank8:joint_subspace_matched:location | 6 | 6 | 6.5944 | 9.6636 | -2.9640 | -4.0495 | 0.1667 | 0.3333 |
| 7 | rank4:joint_remainder_matched:is_a | 5 | 3 | 8.6865 | 8.1627 | -2.8040 | -3.5886 | 0.3333 | 0.3333 |
| 8 | rank4:joint_subspace_matched:location | 6 | 6 | 3.8997 | 7.4554 | -0.2693 | -6.2577 | 0.3333 | 0.3333 |
| 9 | rank8:joint_remainder_matched:is_a | 5 | 3 | 4.9471 | 6.1677 | 0.9355 | -5.5835 | 0.6667 | 0.3333 |
| 10 | rank8:joint_subspace_matched:part_of | 5 | 5 | 6.7761 | 10.2266 | -2.8299 | -3.2563 | 0.4000 | 0.2000 |
| 11 | rank8:joint_subspace_mismatched_frame:location | 6 | 6 | 5.4331 | 9.4808 | -1.8027 | -4.2322 | 0.1667 | 0.1667 |
| 12 | rank4:joint_remainder_matched:can_do | 3 | 2 | 9.2827 | 14.6801 | -1.2986 | -0.8187 | 0.5000 | 0.0000 |
| 13 | rank8:joint_subspace_matched:can_do | 3 | 2 | 11.0132 | 12.0328 | -3.0291 | -3.4661 | 0.5000 | 0.0000 |
| 14 | rank8:joint_subspace_mismatched_frame:can_do | 3 | 2 | 7.2528 | 11.9682 | 0.7313 | -3.5306 | 0.5000 | 0.0000 |
| 15 | rank4:joint_subspace_matched:part_of | 5 | 5 | 4.0321 | 6.9609 | -0.0859 | -6.5220 | 0.6000 | 0.0000 |
| 16 | rank8:joint_subspace_matched:used_for | 3 | 3 | 4.8452 | 6.5395 | -1.6447 | -8.5005 | 0.3333 | 0.0000 |
| 17 | rank4:joint_remainder_matched:location | 6 | 6 | 6.0632 | 6.3243 | -2.4327 | -7.3888 | 0.1667 | 0.0000 |
| 18 | rank4:joint_remainder_matched:part_of | 5 | 5 | 6.7240 | 5.6967 | -2.7778 | -7.7861 | 0.4000 | 0.0000 |
| 19 | rank8:joint_subspace_matched:is_a | 5 | 3 | 7.2409 | 5.6582 | -1.3584 | -6.0931 | 0.6667 | 0.0000 |
| 20 | rank8:joint_remainder_matched:used_for | 3 | 3 | 3.2472 | 5.5400 | -0.0467 | -9.5000 | 0.6667 | 0.0000 |
| 21 | rank4:joint_subspace_mismatched_frame:part_of | 5 | 5 | 3.7648 | 5.4348 | 0.1814 | -8.0481 | 0.8000 | 0.0000 |
| 22 | rank4:joint_remainder_matched:used_for | 3 | 3 | 4.9904 | 5.4187 | -1.7899 | -9.6212 | 0.6667 | 0.0000 |
| 23 | rank4:joint_subspace_mismatched_frame:location | 6 | 6 | 4.2305 | 5.1706 | -0.6001 | -8.5425 | 0.3333 | 0.0000 |
| 24 | rank8:joint_subspace_mismatched_frame:used_for | 3 | 3 | 5.7694 | 5.1177 | -2.5689 | -9.9222 | 0.3333 | 0.0000 |
| 25 | rank4:joint_subspace_matched:can_do | 3 | 2 | 2.1818 | 4.8738 | 5.8023 | -10.6250 | 1.0000 | 0.0000 |
| 26 | rank4:joint_subspace_matched:used_for | 3 | 3 | 2.2906 | 4.8122 | 0.9099 | -10.2278 | 0.3333 | 0.0000 |
| 27 | rank8:joint_remainder_matched:can_do | 3 | 2 | 3.6045 | 4.2976 | 4.3796 | -11.2013 | 1.0000 | 0.0000 |
| 28 | rank4:joint_subspace_matched:is_a | 5 | 3 | 3.1055 | 4.0873 | 2.7770 | -7.6640 | 1.0000 | 0.0000 |
| 29 | rank8:joint_subspace_matched:property | 3 | 2 | 0.9970 | 3.8958 | 6.5004 | -6.5004 | 1.0000 | 0.0000 |
| 30 | rank4:joint_subspace_mismatched_frame:is_a | 5 | 3 | 2.8526 | 3.4167 | 3.0299 | -8.3346 | 1.0000 | 0.0000 |
| 31 | rank8:joint_remainder_matched:location | 6 | 6 | 4.8866 | 3.3093 | -1.2562 | -10.4038 | 0.5000 | 0.0000 |
| 32 | rank8:joint_subspace_restore_both:can_do | 3 | 2 | 3.1452 | 3.1243 | 4.8389 | -12.3745 | 1.0000 | 0.0000 |
| 33 | rank4:joint_subspace_mismatched_frame:used_for | 3 | 3 | 2.7083 | 3.0565 | 0.4922 | -11.9835 | 0.6667 | 0.0000 |
| 34 | rank8:joint_subspace_mismatched_frame:is_a | 5 | 3 | 5.2714 | 3.0208 | 0.6111 | -8.7304 | 0.6667 | 0.0000 |
| 35 | rank8:joint_subspace_mismatched_frame:property | 3 | 2 | 5.4657 | 2.8074 | 2.0316 | -7.5887 | 1.0000 | 0.0000 |
| 36 | rank8:joint_remainder_matched:property | 3 | 2 | 2.3675 | 2.7924 | 5.1298 | -7.6038 | 1.0000 | 0.0000 |
| 37 | rank4:joint_subspace_matched:material | 3 | 2 | 3.1986 | 2.6845 | 1.0748 | -4.2100 | 1.0000 | 0.0000 |
| 38 | rank8:joint_subspace_restore_both:property | 3 | 2 | 2.2169 | 2.5640 | 5.2805 | -7.8322 | 1.0000 | 0.0000 |
| 39 | rank4:joint_subspace_restore_both:can_do | 3 | 2 | 1.8958 | 2.1565 | 6.0882 | -13.3424 | 1.0000 | 0.0000 |
| 40 | rank8:joint_subspace_restore_both:location | 6 | 6 | 1.9312 | 1.9459 | 1.6992 | -11.7672 | 0.6667 | 0.0000 |
| 41 | rank4:joint_subspace_restore_both:used_for | 3 | 3 | 0.8045 | 1.9436 | 2.3960 | -13.0963 | 1.0000 | 0.0000 |
| 42 | rank8:joint_subspace_restore_both:used_for | 3 | 3 | 1.6252 | 1.8733 | 1.5753 | -13.1667 | 0.6667 | 0.0000 |
| 43 | rank4:joint_subspace_restore_both:property | 3 | 2 | 1.9061 | 1.7873 | 5.5913 | -8.6088 | 1.0000 | 0.0000 |
| 44 | rank4:joint_subspace_restore_both:part_of | 5 | 5 | -0.3834 | 1.7785 | 4.3296 | -11.7044 | 1.0000 | 0.0000 |
| 45 | rank4:joint_subspace_restore_both:location | 6 | 6 | 2.2414 | 1.5278 | 1.3890 | -12.1853 | 0.6667 | 0.0000 |
| 46 | rank8:joint_remainder_matched:part_of | 5 | 5 | 0.1466 | 1.2333 | 3.7996 | -12.2496 | 0.6000 | 0.0000 |
| 47 | rank4:joint_remainder_matched:property | 3 | 2 | 2.8857 | 1.0890 | 4.6117 | -9.3071 | 1.0000 | 0.0000 |
| 48 | rank4:joint_subspace_mismatched_frame:property | 3 | 2 | 2.1193 | 0.9837 | 5.3780 | -9.4124 | 1.0000 | 0.0000 |
| 49 | rank4:joint_subspace_restore_both:is_a | 5 | 3 | 0.6063 | 0.9239 | 5.2763 | -10.8274 | 1.0000 | 0.0000 |
| 50 | rank8:joint_subspace_restore_both:is_a | 5 | 3 | 1.4556 | 0.8332 | 4.4269 | -10.9181 | 1.0000 | 0.0000 |
| 51 | rank8:joint_subspace_restore_both:material | 3 | 2 | 0.8213 | 0.7494 | 3.4521 | -6.1450 | 1.0000 | 0.0000 |
| 52 | rank4:joint_subspace_mismatched_frame:can_do | 3 | 2 | 1.7565 | 0.7156 | 6.2275 | -14.7833 | 1.0000 | 0.0000 |
| 53 | rank4:joint_subspace_restore_both:material | 3 | 2 | 0.1142 | 0.6558 | 4.1592 | -6.2387 | 1.0000 | 0.0000 |
| 54 | rank8:joint_subspace_restore_both:part_of | 5 | 5 | -0.6879 | 0.6106 | 4.6341 | -12.8722 | 1.0000 | 0.0000 |
| 55 | rank4:joint_subspace_matched:property | 3 | 2 | -1.5558 | 0.1459 | 9.0532 | -10.2502 | 1.0000 | 0.0000 |
| 56 | rank8:joint_remainder_matched:material | 3 | 2 | -0.6529 | -0.3837 | 4.9263 | -7.2782 | 1.0000 | 0.0000 |

## glm4

missing

## deepseek7b

missing

