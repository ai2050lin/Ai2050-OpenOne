# Phase78 Factor Subspace Audit Summary

## qwen3

items=672, basis_items=168, rows=9408, layer_pairs=[[4, 8], [8, 12]]
module=resid_out, basis_rank=16, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched | 1344 | 780 | 8.8485 | 12.0854 | -4.6308 | -1.0326 | 0.1641 | 0.4051 |
| 2 | joint_subspace_restore_object_only | 1344 | 780 | 4.6753 | 8.7413 | -0.4577 | -4.3768 | 0.4654 | 0.1769 |
| 3 | frame_subspace_matched | 1344 | 780 | 4.0292 | 7.9465 | 0.1884 | -5.1716 | 0.5397 | 0.1385 |
| 4 | joint_subspace_mismatched_frame | 1344 | 780 | 7.6611 | 7.7940 | -3.4435 | -5.3241 | 0.2141 | 0.1205 |
| 5 | joint_subspace_restore_frame_only | 1344 | 780 | 6.2710 | 4.8659 | -2.0534 | -8.2522 | 0.3705 | 0.0436 |
| 6 | object_subspace_matched | 1344 | 780 | 6.1465 | 4.1399 | -1.9288 | -8.9782 | 0.3692 | 0.0256 |
| 7 | joint_subspace_restore_both | 1344 | 780 | 1.0651 | 1.2686 | 3.1526 | -11.8495 | 0.8423 | 0.0077 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched:L8->L12 | 672 | 390 | 8.2416 | 12.3113 | -4.0240 | -0.8068 | 0.1846 | 0.4436 |
| 2 | joint_subspace_matched:L4->L8 | 672 | 390 | 9.4554 | 11.8595 | -5.2377 | -1.2585 | 0.1436 | 0.3667 |
| 3 | joint_subspace_restore_object_only:L4->L8 | 672 | 390 | 4.9528 | 8.4962 | -0.7352 | -4.6219 | 0.4256 | 0.1795 |
| 4 | joint_subspace_restore_object_only:L8->L12 | 672 | 390 | 4.3978 | 8.9863 | -0.1802 | -4.1317 | 0.5051 | 0.1744 |
| 5 | frame_subspace_matched:L8->L12 | 672 | 390 | 3.8563 | 8.2088 | 0.3613 | -4.9092 | 0.5538 | 0.1487 |
| 6 | joint_subspace_mismatched_frame:L4->L8 | 672 | 390 | 8.2767 | 8.0866 | -4.0591 | -5.0315 | 0.1795 | 0.1333 |
| 7 | frame_subspace_matched:L4->L8 | 672 | 390 | 4.2022 | 7.6841 | 0.0155 | -5.4339 | 0.5256 | 0.1282 |
| 8 | joint_subspace_mismatched_frame:L8->L12 | 672 | 390 | 7.0455 | 7.5014 | -2.8278 | -5.6167 | 0.2487 | 0.1077 |
| 9 | joint_subspace_restore_frame_only:L4->L8 | 672 | 390 | 7.1737 | 5.4057 | -2.9560 | -7.7123 | 0.3000 | 0.0538 |
| 10 | joint_subspace_restore_frame_only:L8->L12 | 672 | 390 | 5.3684 | 4.3261 | -1.1507 | -8.7920 | 0.4410 | 0.0333 |
| 11 | object_subspace_matched:L8->L12 | 672 | 390 | 5.3234 | 3.9370 | -1.1057 | -9.1810 | 0.4308 | 0.0308 |
| 12 | object_subspace_matched:L4->L8 | 672 | 390 | 6.9696 | 4.3428 | -2.7519 | -8.7753 | 0.3077 | 0.0205 |
| 13 | joint_subspace_restore_both:L4->L8 | 672 | 390 | 1.2900 | 1.6178 | 2.9277 | -11.5003 | 0.8154 | 0.0128 |
| 14 | joint_subspace_restore_both:L8->L12 | 672 | 390 | 0.8402 | 0.9194 | 3.3774 | -12.1987 | 0.8692 | 0.0026 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched:can_do | 192 | 144 | 9.4519 | 15.8256 | -4.2996 | 0.7488 | 0.1875 | 0.5625 |
| 2 | joint_subspace_matched:location | 192 | 74 | 7.8314 | 10.6939 | -5.1260 | -0.1807 | 0.1216 | 0.5405 |
| 3 | joint_subspace_matched:property | 192 | 76 | 8.5340 | 8.7689 | -5.4249 | 0.3055 | 0.1184 | 0.4868 |
| 4 | joint_subspace_matched:used_for | 192 | 110 | 11.6082 | 14.4755 | -6.6879 | -0.4444 | 0.0727 | 0.3909 |
| 5 | joint_subspace_matched:is_a | 192 | 160 | 9.9058 | 12.3707 | -4.6318 | -1.2252 | 0.1625 | 0.3750 |
| 6 | joint_subspace_matched:part_of | 192 | 106 | 7.8696 | 10.5701 | -4.2751 | -3.2893 | 0.1887 | 0.3019 |
| 7 | joint_subspace_restore_object_only:can_do | 192 | 144 | 5.7917 | 12.2263 | -0.6394 | -2.8505 | 0.4792 | 0.2569 |
| 8 | joint_subspace_mismatched_frame:can_do | 192 | 144 | 8.4916 | 12.1967 | -3.3392 | -2.8800 | 0.2500 | 0.2431 |
| 9 | joint_subspace_restore_object_only:property | 192 | 76 | 4.3291 | 5.4393 | -1.2200 | -3.0241 | 0.3421 | 0.2368 |
| 10 | frame_subspace_matched:property | 192 | 76 | 3.7345 | 4.3738 | -0.6254 | -4.0896 | 0.3553 | 0.2368 |
| 11 | joint_subspace_restore_object_only:location | 192 | 74 | 3.0269 | 6.0857 | -0.3215 | -4.7888 | 0.4595 | 0.2162 |
| 12 | joint_subspace_matched:material | 192 | 110 | 5.6059 | 9.0720 | -2.4671 | -2.9959 | 0.2636 | 0.2091 |
| 13 | joint_subspace_restore_object_only:is_a | 192 | 160 | 5.3533 | 9.2380 | -0.0794 | -4.3579 | 0.5625 | 0.1875 |
| 14 | frame_subspace_matched:is_a | 192 | 160 | 4.8421 | 8.5932 | 0.4319 | -5.0027 | 0.5938 | 0.1688 |
| 15 | joint_subspace_restore_frame_only:property | 192 | 76 | 4.8787 | 4.7459 | -1.7696 | -3.7175 | 0.3684 | 0.1579 |
| 16 | joint_subspace_restore_object_only:used_for | 192 | 110 | 6.4216 | 10.4056 | -1.5014 | -4.5143 | 0.3545 | 0.1545 |
| 17 | frame_subspace_matched:can_do | 192 | 144 | 4.6998 | 10.8792 | 0.4525 | -4.1976 | 0.5764 | 0.1528 |
| 18 | joint_subspace_restore_frame_only:location | 192 | 74 | 5.7549 | 5.6033 | -3.0495 | -5.2712 | 0.2027 | 0.1486 |
| 19 | frame_subspace_matched:location | 192 | 74 | 2.4252 | 5.4991 | 0.2802 | -5.3755 | 0.5811 | 0.1486 |
| 20 | frame_subspace_matched:used_for | 192 | 110 | 5.7184 | 9.4675 | -0.7982 | -5.4524 | 0.4364 | 0.1455 |
| 21 | joint_subspace_restore_object_only:part_of | 192 | 106 | 2.8102 | 8.1754 | 0.7844 | -5.6839 | 0.6132 | 0.1415 |
| 22 | joint_subspace_mismatched_frame:location | 192 | 74 | 5.9374 | 5.5377 | -3.2320 | -5.3369 | 0.1351 | 0.1216 |
| 23 | object_subspace_matched:location | 192 | 74 | 5.6237 | 4.6086 | -2.9183 | -6.2660 | 0.1892 | 0.1216 |
| 24 | joint_subspace_mismatched_frame:is_a | 192 | 160 | 8.6545 | 7.8457 | -3.3805 | -5.7502 | 0.2250 | 0.1062 |
| 25 | frame_subspace_matched:part_of | 192 | 106 | 2.3308 | 7.6512 | 1.2638 | -6.2082 | 0.6887 | 0.1038 |
| 26 | joint_subspace_mismatched_frame:used_for | 192 | 110 | 10.7168 | 9.2434 | -5.7966 | -5.6765 | 0.1000 | 0.1000 |
| 27 | joint_subspace_mismatched_frame:part_of | 192 | 106 | 7.0886 | 7.1735 | -3.4940 | -6.6859 | 0.2170 | 0.0943 |
| 28 | joint_subspace_mismatched_frame:property | 192 | 76 | 6.1942 | 4.4269 | -3.0851 | -4.0365 | 0.2105 | 0.0921 |
| 29 | object_subspace_matched:property | 192 | 76 | 4.4329 | 3.7088 | -1.3238 | -4.7546 | 0.4474 | 0.0789 |
| 30 | joint_subspace_restore_object_only:material | 192 | 110 | 3.6270 | 6.4052 | -0.4883 | -5.6626 | 0.3636 | 0.0455 |
| 31 | joint_subspace_mismatched_frame:material | 192 | 110 | 4.7982 | 4.9479 | -1.6594 | -7.1200 | 0.3182 | 0.0455 |
| 32 | joint_subspace_restore_both:property | 192 | 76 | 0.9291 | 1.7022 | 2.1800 | -6.7612 | 0.8421 | 0.0395 |
| 33 | joint_subspace_restore_frame_only:part_of | 192 | 106 | 6.9369 | 3.3359 | -3.3424 | -10.5234 | 0.2736 | 0.0283 |
| 34 | joint_subspace_restore_frame_only:used_for | 192 | 110 | 8.3013 | 7.6474 | -3.3811 | -7.2725 | 0.2909 | 0.0273 |
| 35 | frame_subspace_matched:material | 192 | 110 | 3.1994 | 6.0448 | -0.0607 | -6.0230 | 0.4727 | 0.0273 |
| 36 | joint_subspace_restore_frame_only:material | 192 | 110 | 3.1656 | 3.4189 | -0.0269 | -8.6489 | 0.5455 | 0.0273 |
| 37 | joint_subspace_restore_both:part_of | 192 | 106 | 1.3099 | 1.0297 | 2.2847 | -12.8296 | 0.7642 | 0.0189 |
| 38 | object_subspace_matched:material | 192 | 110 | 3.1047 | 2.6268 | 0.0341 | -9.4411 | 0.5182 | 0.0182 |
| 39 | joint_subspace_restore_both:location | 192 | 74 | 1.0314 | 1.3516 | 1.6740 | -9.5230 | 0.7297 | 0.0135 |
| 40 | joint_subspace_restore_frame_only:is_a | 192 | 160 | 7.3843 | 4.5405 | -2.1103 | -9.0554 | 0.4313 | 0.0125 |
| 41 | object_subspace_matched:is_a | 192 | 160 | 7.4887 | 3.8124 | -2.2148 | -9.7835 | 0.4062 | 0.0125 |
| 42 | object_subspace_matched:part_of | 192 | 106 | 6.6202 | 2.9793 | -3.0257 | -10.8800 | 0.2736 | 0.0094 |
| 43 | object_subspace_matched:used_for | 192 | 110 | 8.2677 | 7.1872 | -3.3474 | -7.7327 | 0.2818 | 0.0000 |
| 44 | joint_subspace_restore_frame_only:can_do | 192 | 144 | 6.3651 | 5.0187 | -1.2128 | -10.0581 | 0.3889 | 0.0000 |
| 45 | object_subspace_matched:can_do | 192 | 144 | 6.1827 | 4.1727 | -1.0304 | -10.9041 | 0.4028 | 0.0000 |
| 46 | joint_subspace_restore_both:can_do | 192 | 144 | 1.5943 | 1.6085 | 3.5580 | -13.4683 | 0.8750 | 0.0000 |
| 47 | joint_subspace_restore_both:used_for | 192 | 110 | 0.7489 | 1.5738 | 4.1713 | -13.3461 | 0.8545 | 0.0000 |
| 48 | joint_subspace_restore_both:is_a | 192 | 160 | 0.9633 | 1.1616 | 4.3107 | -12.4343 | 0.9250 | 0.0000 |
| 49 | joint_subspace_restore_both:material | 192 | 110 | 0.7172 | 0.5489 | 2.4216 | -11.5190 | 0.8182 | 0.0000 |

## glm4

items=672, basis_items=168, rows=9408, layer_pairs=[[4, 10], [10, 20]]
module=resid_out, basis_rank=16, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched | 1344 | 888 | 7.1730 | 11.4769 | -2.8415 | -0.7403 | 0.3052 | 0.4403 |
| 2 | joint_subspace_restore_object_only | 1344 | 888 | 3.7260 | 8.4447 | 0.6055 | -3.7725 | 0.5957 | 0.2128 |
| 3 | joint_subspace_mismatched_frame | 1344 | 888 | 6.1109 | 7.1422 | -1.7794 | -5.0750 | 0.3637 | 0.1273 |
| 4 | frame_subspace_matched | 1344 | 888 | 2.6983 | 7.0353 | 1.6332 | -5.1818 | 0.6892 | 0.1261 |
| 5 | joint_subspace_restore_frame_only | 1344 | 888 | 4.7197 | 4.7553 | -0.3881 | -7.4619 | 0.5068 | 0.0743 |
| 6 | object_subspace_matched | 1344 | 888 | 4.4448 | 3.8216 | -0.1133 | -8.3956 | 0.5372 | 0.0507 |
| 7 | joint_subspace_restore_both | 1344 | 888 | 1.0388 | 1.5692 | 3.2928 | -10.6480 | 0.8806 | 0.0135 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched:L4->L10 | 672 | 444 | 8.1984 | 11.8243 | -3.8669 | -0.3929 | 0.2410 | 0.4730 |
| 2 | joint_subspace_matched:L10->L20 | 672 | 444 | 6.1475 | 11.1294 | -1.8160 | -1.0877 | 0.3694 | 0.4077 |
| 3 | joint_subspace_restore_object_only:L4->L10 | 672 | 444 | 4.0741 | 8.8756 | 0.2575 | -3.3416 | 0.5586 | 0.2387 |
| 4 | joint_subspace_restore_object_only:L10->L20 | 672 | 444 | 3.3779 | 8.0138 | 0.9536 | -4.2034 | 0.6329 | 0.1869 |
| 5 | joint_subspace_mismatched_frame:L4->L10 | 672 | 444 | 7.2348 | 7.5943 | -2.9033 | -4.6229 | 0.2545 | 0.1419 |
| 6 | frame_subspace_matched:L10->L20 | 672 | 444 | 2.7232 | 7.1057 | 1.6083 | -5.1115 | 0.6892 | 0.1329 |
| 7 | frame_subspace_matched:L4->L10 | 672 | 444 | 2.6735 | 6.9650 | 1.6581 | -5.2522 | 0.6892 | 0.1194 |
| 8 | joint_subspace_mismatched_frame:L10->L20 | 672 | 444 | 4.9870 | 6.6901 | -0.6555 | -5.5270 | 0.4730 | 0.1126 |
| 9 | joint_subspace_restore_frame_only:L4->L10 | 672 | 444 | 6.0486 | 5.3985 | -1.7171 | -6.8187 | 0.3851 | 0.0968 |
| 10 | object_subspace_matched:L4->L10 | 672 | 444 | 6.0175 | 4.4124 | -1.6860 | -7.8048 | 0.3896 | 0.0586 |
| 11 | joint_subspace_restore_frame_only:L10->L20 | 672 | 444 | 3.3907 | 4.1121 | 0.9408 | -8.1051 | 0.6284 | 0.0518 |
| 12 | object_subspace_matched:L10->L20 | 672 | 444 | 2.8720 | 3.2309 | 1.4595 | -8.9863 | 0.6847 | 0.0428 |
| 13 | joint_subspace_restore_both:L4->L10 | 672 | 444 | 1.3296 | 2.0076 | 3.0020 | -10.2096 | 0.8514 | 0.0135 |
| 14 | joint_subspace_restore_both:L10->L20 | 672 | 444 | 0.7480 | 1.1308 | 3.5836 | -11.0864 | 0.9099 | 0.0135 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched:property | 192 | 98 | 7.1639 | 9.7980 | -4.5629 | 1.6380 | 0.1633 | 0.6837 |
| 2 | joint_subspace_matched:can_do | 192 | 144 | 6.8664 | 14.7039 | -3.1771 | 0.5725 | 0.2431 | 0.5278 |
| 3 | joint_subspace_matched:used_for | 192 | 146 | 9.6510 | 14.7515 | -3.6592 | -0.0149 | 0.2740 | 0.5000 |
| 4 | joint_subspace_matched:location | 192 | 102 | 6.6450 | 8.7722 | -3.9063 | -0.0000 | 0.2451 | 0.4902 |
| 5 | joint_subspace_matched:is_a | 192 | 148 | 7.9419 | 12.0766 | -2.5219 | -1.3555 | 0.3649 | 0.3919 |
| 6 | joint_subspace_restore_object_only:property | 192 | 98 | 3.3373 | 6.6346 | -0.7362 | -1.5254 | 0.3878 | 0.3776 |
| 7 | joint_subspace_restore_object_only:can_do | 192 | 144 | 3.7964 | 11.7471 | -0.1071 | -2.3843 | 0.5208 | 0.2917 |
| 8 | joint_subspace_matched:part_of | 192 | 142 | 6.6247 | 9.7700 | -1.6582 | -2.6996 | 0.3944 | 0.2746 |
| 9 | joint_subspace_matched:material | 192 | 108 | 4.4060 | 8.2477 | -0.7145 | -2.9094 | 0.4167 | 0.2593 |
| 10 | joint_subspace_restore_object_only:location | 192 | 102 | 3.2406 | 5.7174 | -0.5020 | -3.0548 | 0.5098 | 0.2549 |
| 11 | joint_subspace_mismatched_frame:can_do | 192 | 144 | 6.3260 | 11.0252 | -2.6367 | -3.1062 | 0.2500 | 0.2431 |
| 12 | joint_subspace_restore_object_only:used_for | 192 | 146 | 5.2346 | 10.8174 | 0.7571 | -3.9490 | 0.6438 | 0.2260 |
| 13 | joint_subspace_restore_frame_only:property | 192 | 98 | 4.9251 | 4.9962 | -2.3240 | -3.1639 | 0.3163 | 0.2041 |
| 14 | frame_subspace_matched:can_do | 192 | 144 | 2.8901 | 10.4626 | 0.7992 | -3.6688 | 0.6250 | 0.1944 |
| 15 | frame_subspace_matched:property | 192 | 98 | 2.1550 | 4.3800 | 0.4461 | -3.7800 | 0.4694 | 0.1939 |
| 16 | joint_subspace_restore_frame_only:location | 192 | 102 | 3.7581 | 4.1560 | -1.0194 | -4.6162 | 0.4608 | 0.1765 |
| 17 | joint_subspace_restore_object_only:is_a | 192 | 148 | 4.3464 | 9.2609 | 1.0736 | -4.1713 | 0.6622 | 0.1757 |
| 18 | object_subspace_matched:location | 192 | 102 | 3.8576 | 3.5013 | -1.1189 | -5.2710 | 0.4216 | 0.1569 |
| 19 | joint_subspace_mismatched_frame:property | 192 | 98 | 5.6721 | 4.4961 | -3.0710 | -3.6639 | 0.2245 | 0.1531 |
| 20 | frame_subspace_matched:location | 192 | 102 | 2.3072 | 4.3834 | 0.4315 | -4.3888 | 0.5882 | 0.1471 |
| 21 | joint_subspace_mismatched_frame:used_for | 192 | 146 | 8.6829 | 9.6545 | -2.6911 | -5.1119 | 0.2671 | 0.1438 |
| 22 | object_subspace_matched:property | 192 | 98 | 4.3381 | 4.4084 | -1.7371 | -3.7516 | 0.4286 | 0.1429 |
| 23 | frame_subspace_matched:used_for | 192 | 146 | 4.1541 | 8.8261 | 1.8376 | -5.9402 | 0.7192 | 0.1233 |
| 24 | joint_subspace_mismatched_frame:location | 192 | 102 | 5.0979 | 3.8726 | -2.3592 | -4.8997 | 0.3333 | 0.1176 |
| 25 | joint_subspace_restore_object_only:part_of | 192 | 142 | 3.0240 | 6.9430 | 1.9425 | -5.5265 | 0.7324 | 0.1127 |
| 26 | frame_subspace_matched:is_a | 192 | 148 | 3.0485 | 8.0363 | 2.3715 | -5.3958 | 0.7568 | 0.1081 |
| 27 | joint_subspace_restore_object_only:material | 192 | 108 | 2.4765 | 5.9081 | 1.2150 | -5.2490 | 0.6296 | 0.0833 |
| 28 | joint_subspace_mismatched_frame:part_of | 192 | 142 | 5.8991 | 5.9726 | -0.9326 | -6.4970 | 0.4718 | 0.0775 |
| 29 | joint_subspace_restore_frame_only:part_of | 192 | 142 | 4.7396 | 4.7226 | 0.2269 | -7.7470 | 0.5423 | 0.0775 |
| 30 | joint_subspace_mismatched_frame:is_a | 192 | 148 | 6.8069 | 7.4348 | -1.3869 | -5.9973 | 0.4122 | 0.0743 |
| 31 | joint_subspace_mismatched_frame:material | 192 | 108 | 3.0271 | 5.1946 | 0.6644 | -5.9625 | 0.5926 | 0.0741 |
| 32 | joint_subspace_restore_both:property | 192 | 98 | 0.7371 | 1.5703 | 1.8640 | -6.5897 | 0.7857 | 0.0714 |
| 33 | frame_subspace_matched:part_of | 192 | 142 | 2.0775 | 5.9899 | 2.8890 | -6.4796 | 0.8310 | 0.0704 |
| 34 | frame_subspace_matched:material | 192 | 108 | 1.6735 | 4.9617 | 2.0180 | -6.1954 | 0.7500 | 0.0556 |
| 35 | joint_subspace_restore_frame_only:material | 192 | 108 | 2.2113 | 3.3152 | 1.4802 | -7.8419 | 0.6574 | 0.0463 |
| 36 | joint_subspace_restore_frame_only:is_a | 192 | 148 | 5.7658 | 4.1424 | -0.3458 | -9.2898 | 0.5405 | 0.0405 |
| 37 | object_subspace_matched:part_of | 192 | 142 | 4.3999 | 3.6653 | 0.5666 | -8.8042 | 0.5634 | 0.0352 |
| 38 | joint_subspace_restore_frame_only:used_for | 192 | 146 | 6.5968 | 7.3855 | -0.6050 | -7.3809 | 0.5000 | 0.0342 |
| 39 | object_subspace_matched:is_a | 192 | 148 | 5.6672 | 3.4237 | -0.2472 | -10.0085 | 0.5541 | 0.0338 |
| 40 | object_subspace_matched:material | 192 | 108 | 1.9354 | 2.1591 | 1.7561 | -8.9979 | 0.7130 | 0.0278 |
| 41 | joint_subspace_restore_both:location | 192 | 102 | 0.7129 | 1.5925 | 2.0258 | -7.1798 | 0.8529 | 0.0196 |
| 42 | joint_subspace_restore_both:part_of | 192 | 142 | 1.3082 | 1.7423 | 3.6583 | -10.7273 | 0.9014 | 0.0141 |
| 43 | object_subspace_matched:used_for | 192 | 146 | 6.1554 | 6.5724 | -0.1637 | -8.1940 | 0.5342 | 0.0137 |
| 44 | joint_subspace_restore_both:material | 192 | 108 | 1.0153 | 1.2627 | 2.6762 | -9.8944 | 0.8704 | 0.0093 |
| 45 | joint_subspace_restore_frame_only:can_do | 192 | 144 | 4.1443 | 4.0914 | -0.4549 | -10.0400 | 0.4931 | 0.0069 |
| 46 | object_subspace_matched:can_do | 192 | 144 | 3.8687 | 2.6703 | -0.1794 | -11.4611 | 0.5208 | 0.0000 |
| 47 | joint_subspace_restore_both:used_for | 192 | 146 | 1.2140 | 2.1018 | 4.7777 | -12.6646 | 0.9452 | 0.0000 |
| 48 | joint_subspace_restore_both:can_do | 192 | 144 | 0.8822 | 1.6593 | 2.8071 | -12.4721 | 0.8681 | 0.0000 |
| 49 | joint_subspace_restore_both:is_a | 192 | 148 | 1.2012 | 0.9970 | 4.2188 | -12.4352 | 0.8986 | 0.0000 |

## deepseek7b

items=672, basis_items=168, rows=9408, layer_pairs=[[8, 10], [12, 14]]
module=resid_out, basis_rank=16, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched | 1344 | 572 | 5.0480 | 9.6949 | -1.1080 | -3.7951 | 0.4108 | 0.2640 |
| 2 | joint_subspace_restore_object_only | 1344 | 572 | 3.2095 | 7.9452 | 0.7305 | -5.5448 | 0.5892 | 0.1661 |
| 3 | frame_subspace_matched | 1344 | 572 | 2.8841 | 7.6324 | 1.0560 | -5.8577 | 0.6101 | 0.1486 |
| 4 | joint_subspace_mismatched_frame | 1344 | 572 | 4.8122 | 6.0316 | -0.8721 | -7.4584 | 0.4231 | 0.0769 |
| 5 | joint_subspace_restore_frame_only | 1344 | 572 | 2.8638 | 2.6017 | 1.0763 | -10.8883 | 0.6154 | 0.0192 |
| 6 | object_subspace_matched | 1344 | 572 | 2.7596 | 2.4039 | 1.1805 | -11.0861 | 0.6399 | 0.0140 |
| 7 | joint_subspace_restore_both | 1344 | 572 | 0.6379 | 0.5466 | 3.3022 | -12.9434 | 0.8392 | 0.0035 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched:L12->L14 | 672 | 286 | 4.5778 | 9.5341 | -0.6377 | -3.9559 | 0.4371 | 0.2657 |
| 2 | joint_subspace_matched:L8->L10 | 672 | 286 | 5.5183 | 9.8557 | -1.5782 | -3.6343 | 0.3846 | 0.2622 |
| 3 | joint_subspace_restore_object_only:L8->L10 | 672 | 286 | 3.5548 | 8.2609 | 0.3853 | -5.2292 | 0.5559 | 0.1818 |
| 4 | frame_subspace_matched:L8->L10 | 672 | 286 | 3.0434 | 7.7451 | 0.8967 | -5.7449 | 0.5839 | 0.1538 |
| 5 | joint_subspace_restore_object_only:L12->L14 | 672 | 286 | 2.8643 | 7.6295 | 1.0758 | -5.8605 | 0.6224 | 0.1503 |
| 6 | frame_subspace_matched:L12->L14 | 672 | 286 | 2.7248 | 7.5196 | 1.2153 | -5.9704 | 0.6364 | 0.1434 |
| 7 | joint_subspace_mismatched_frame:L8->L10 | 672 | 286 | 5.1359 | 6.2464 | -1.1958 | -7.2437 | 0.4021 | 0.0804 |
| 8 | joint_subspace_mismatched_frame:L12->L14 | 672 | 286 | 4.4885 | 5.8169 | -0.5485 | -7.6731 | 0.4441 | 0.0734 |
| 9 | joint_subspace_restore_frame_only:L8->L10 | 672 | 286 | 3.4902 | 2.9470 | 0.4498 | -10.5431 | 0.5594 | 0.0280 |
| 10 | object_subspace_matched:L8->L10 | 672 | 286 | 3.2329 | 2.6878 | 0.7072 | -10.8022 | 0.6014 | 0.0210 |
| 11 | joint_subspace_restore_frame_only:L12->L14 | 672 | 286 | 2.2374 | 2.2564 | 1.7027 | -11.2336 | 0.6713 | 0.0105 |
| 12 | object_subspace_matched:L12->L14 | 672 | 286 | 2.2863 | 2.1201 | 1.6537 | -11.3700 | 0.6783 | 0.0070 |
| 13 | joint_subspace_restore_both:L8->L10 | 672 | 286 | 1.1277 | 0.8211 | 2.8124 | -12.6689 | 0.7867 | 0.0070 |
| 14 | joint_subspace_restore_both:L12->L14 | 672 | 286 | 0.1481 | 0.2721 | 3.7920 | -13.2179 | 0.8916 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_subspace_matched:can_do | 192 | 122 | 5.7798 | 13.8316 | -1.3724 | -1.6609 | 0.3689 | 0.4016 |
| 2 | joint_subspace_matched:used_for | 192 | 74 | 7.1615 | 12.9967 | -2.1954 | -2.9896 | 0.2703 | 0.3243 |
| 3 | joint_subspace_matched:property | 192 | 70 | 4.0154 | 6.4051 | -0.7194 | -2.5540 | 0.4857 | 0.3143 |
| 4 | joint_subspace_restore_object_only:property | 192 | 70 | 3.0705 | 5.2058 | 0.2255 | -3.7533 | 0.5000 | 0.2857 |
| 5 | joint_subspace_matched:location | 192 | 26 | 5.3592 | 5.5857 | -2.3806 | -4.5328 | 0.3462 | 0.2692 |
| 6 | joint_subspace_matched:part_of | 192 | 74 | 4.9776 | 10.6791 | -1.1809 | -4.9319 | 0.4189 | 0.2568 |
| 7 | joint_subspace_restore_object_only:part_of | 192 | 74 | 2.4697 | 9.0877 | 1.3269 | -6.5232 | 0.5946 | 0.2432 |
| 8 | joint_subspace_restore_object_only:can_do | 192 | 122 | 4.4360 | 11.9726 | -0.0286 | -3.5198 | 0.5492 | 0.2295 |
| 9 | frame_subspace_matched:can_do | 192 | 122 | 4.2253 | 11.6362 | 0.1821 | -3.8562 | 0.5492 | 0.2295 |
| 10 | frame_subspace_matched:property | 192 | 70 | 2.9284 | 4.7275 | 0.3677 | -4.2316 | 0.5286 | 0.2143 |
| 11 | frame_subspace_matched:part_of | 192 | 74 | 2.1916 | 9.1283 | 1.6050 | -6.4826 | 0.6486 | 0.1892 |
| 12 | joint_subspace_mismatched_frame:can_do | 192 | 122 | 6.1151 | 10.6451 | -1.7077 | -4.8474 | 0.3525 | 0.1803 |
| 13 | joint_subspace_matched:is_a | 192 | 128 | 4.6367 | 7.9725 | -1.1234 | -5.1322 | 0.3906 | 0.1797 |
| 14 | joint_subspace_restore_object_only:used_for | 192 | 74 | 4.0832 | 10.3482 | 0.8829 | -5.6381 | 0.6622 | 0.1757 |
| 15 | frame_subspace_matched:used_for | 192 | 74 | 3.3268 | 9.7253 | 1.6393 | -6.2610 | 0.6892 | 0.1622 |
| 16 | joint_subspace_restore_object_only:location | 192 | 26 | 2.1782 | 2.8388 | 0.8003 | -7.2797 | 0.5000 | 0.1538 |
| 17 | joint_subspace_mismatched_frame:part_of | 192 | 74 | 4.9150 | 6.8943 | -1.1184 | -8.7166 | 0.3649 | 0.1351 |
| 18 | frame_subspace_matched:location | 192 | 26 | 2.0018 | 2.5701 | 0.9767 | -7.5484 | 0.4615 | 0.1154 |
| 19 | joint_subspace_matched:material | 192 | 78 | 3.4633 | 6.3071 | 0.5072 | -5.4927 | 0.5897 | 0.0897 |
| 20 | joint_subspace_mismatched_frame:property | 192 | 70 | 3.6893 | 2.1942 | -0.3933 | -6.7649 | 0.4571 | 0.0857 |
| 21 | object_subspace_matched:location | 192 | 26 | 3.1079 | 1.7233 | -0.1293 | -8.3953 | 0.4231 | 0.0769 |
| 22 | joint_subspace_restore_frame_only:location | 192 | 26 | 2.9612 | 1.5318 | 0.0174 | -8.5867 | 0.4231 | 0.0769 |
| 23 | joint_subspace_restore_frame_only:property | 192 | 70 | 1.8821 | 2.0259 | 1.4140 | -6.9332 | 0.7286 | 0.0714 |
| 24 | joint_subspace_restore_object_only:is_a | 192 | 128 | 2.6472 | 6.3579 | 0.8662 | -6.7468 | 0.5703 | 0.0703 |
| 25 | frame_subspace_matched:is_a | 192 | 128 | 2.3512 | 6.2303 | 1.1622 | -6.8744 | 0.6094 | 0.0703 |
| 26 | frame_subspace_matched:material | 192 | 78 | 2.1520 | 4.5601 | 1.8185 | -7.2398 | 0.7179 | 0.0513 |
| 27 | joint_subspace_restore_object_only:material | 192 | 78 | 2.5557 | 5.0474 | 1.4148 | -6.7525 | 0.7179 | 0.0385 |
| 28 | joint_subspace_restore_both:location | 192 | 26 | 0.6414 | 0.1468 | 2.3371 | -9.9718 | 0.6538 | 0.0385 |
| 29 | object_subspace_matched:property | 192 | 70 | 1.3780 | 1.2293 | 1.9180 | -7.7298 | 0.8143 | 0.0286 |
| 30 | joint_subspace_mismatched_frame:used_for | 192 | 74 | 6.4733 | 8.1953 | -1.5073 | -7.7910 | 0.4595 | 0.0270 |
| 31 | joint_subspace_restore_frame_only:part_of | 192 | 74 | 3.3883 | 2.6451 | 0.4083 | -12.9658 | 0.6081 | 0.0270 |
| 32 | object_subspace_matched:part_of | 192 | 74 | 3.2433 | 2.4771 | 0.5533 | -13.1338 | 0.5811 | 0.0270 |
| 33 | joint_subspace_mismatched_frame:is_a | 192 | 128 | 4.4589 | 4.7579 | -0.9456 | -8.3467 | 0.4062 | 0.0234 |
| 34 | joint_subspace_mismatched_frame:material | 192 | 78 | 3.1571 | 3.3887 | 0.8134 | -8.4112 | 0.5641 | 0.0128 |
| 35 | object_subspace_matched:material | 192 | 78 | 1.0331 | 1.5891 | 2.9374 | -10.2107 | 0.7949 | 0.0128 |
| 36 | joint_subspace_restore_frame_only:material | 192 | 78 | 0.9778 | 1.5768 | 2.9927 | -10.2231 | 0.7179 | 0.0128 |
| 37 | joint_subspace_restore_frame_only:is_a | 192 | 128 | 3.1001 | 2.5230 | 0.4132 | -10.5817 | 0.5547 | 0.0078 |
| 38 | object_subspace_matched:is_a | 192 | 128 | 3.1118 | 2.2978 | 0.4016 | -10.8068 | 0.5703 | 0.0078 |
| 39 | joint_subspace_restore_both:is_a | 192 | 128 | 0.4784 | 0.4459 | 3.0349 | -12.6588 | 0.7891 | 0.0078 |
| 40 | joint_subspace_restore_frame_only:used_for | 192 | 74 | 5.1066 | 4.7990 | -0.1405 | -11.1873 | 0.4865 | 0.0000 |
| 41 | object_subspace_matched:used_for | 192 | 74 | 4.7305 | 4.5666 | 0.2355 | -11.4197 | 0.5270 | 0.0000 |
| 42 | joint_subspace_restore_frame_only:can_do | 192 | 122 | 2.6858 | 2.5388 | 1.7216 | -12.9537 | 0.6721 | 0.0000 |
| 43 | object_subspace_matched:can_do | 192 | 122 | 2.7236 | 2.4990 | 1.6838 | -12.9935 | 0.6639 | 0.0000 |
| 44 | joint_subspace_restore_both:used_for | 192 | 74 | 0.8905 | 1.0904 | 4.0755 | -14.8959 | 0.8919 | 0.0000 |
| 45 | joint_subspace_restore_both:part_of | 192 | 74 | 0.8886 | 0.8125 | 2.9081 | -14.7984 | 0.8378 | 0.0000 |
| 46 | joint_subspace_restore_both:can_do | 192 | 122 | 0.5608 | 0.4623 | 3.8466 | -15.0302 | 0.8852 | 0.0000 |
| 47 | joint_subspace_restore_both:material | 192 | 78 | 0.4963 | 0.3742 | 3.4742 | -11.4257 | 0.8590 | 0.0000 |
| 48 | joint_subspace_restore_both:property | 192 | 70 | 0.6880 | 0.3624 | 2.6080 | -8.5967 | 0.8429 | 0.0000 |
| 49 | joint_subspace_mismatched_frame:location | 192 | 26 | 3.4060 | 0.3015 | -0.4274 | -9.8170 | 0.3846 | 0.0000 |

