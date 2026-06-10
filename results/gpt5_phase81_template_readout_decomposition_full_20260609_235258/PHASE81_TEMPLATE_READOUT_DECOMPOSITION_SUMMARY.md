# Phase81 Template Readout Decomposition Summary

## qwen3

items=1344, basis_items=448, rows=34944, layer_pairs=[[4, 8], [8, 12]]
module=resid_out, contrast_rank=64, nuisance_rank=24
relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for'], phrase_ids=[0, 1, 2, 3], slot_ids=['answer', 'arrow', 'equals', 'value']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_slot_frame | 2688 | 1228 | 5.8709 | 11.0800 | -0.9336 | -0.9336 | 0.5375 | 0.5375 |
| 2 | joint_same_relation_other_phrase_frame | 2688 | 1228 | 6.0390 | 10.9118 | -1.1017 | -1.1017 | 0.5375 | 0.5375 |
| 3 | joint_orth_relation | 2688 | 1228 | 6.0525 | 5.4461 | -1.1152 | -6.5674 | 0.5391 | 0.1865 |
| 4 | joint_orth_all | 2688 | 1228 | 6.0247 | 5.4375 | -1.0874 | -6.5760 | 0.5415 | 0.1865 |
| 5 | joint_raw | 2688 | 1228 | 6.0488 | 5.4541 | -1.1115 | -6.5594 | 0.5399 | 0.1857 |
| 6 | joint_orth_phrase | 2688 | 1228 | 6.0505 | 5.4413 | -1.1132 | -6.5723 | 0.5407 | 0.1857 |
| 7 | joint_orth_slot | 2688 | 1228 | 6.0451 | 5.4502 | -1.1078 | -6.5633 | 0.5391 | 0.1849 |
| 8 | joint_orth_phrase_slot | 2688 | 1228 | 6.0241 | 5.4315 | -1.0868 | -6.5820 | 0.5391 | 0.1849 |
| 9 | joint_same_object_other_relation_frame | 2688 | 1228 | 6.0440 | 2.9643 | -1.1067 | -9.0492 | 0.5407 | 0.0244 |
| 10 | joint_raw_restore_both | 2688 | 1228 | 0.2158 | 0.2164 | 4.7215 | -11.7972 | 0.9251 | 0.0033 |
| 11 | joint_relation_basis_only | 2688 | 1228 | 0.0326 | 0.0046 | 4.9047 | -12.0089 | 0.9691 | 0.0008 |
| 12 | joint_slot_basis_only | 2688 | 1228 | 0.0229 | 0.0062 | 4.9144 | -12.0074 | 0.9723 | 0.0000 |
| 13 | joint_phrase_basis_only | 2688 | 1228 | 0.0294 | 0.0007 | 4.9080 | -12.0128 | 0.9731 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_phrase_frame:L8->L12 | 1344 | 614 | 5.9096 | 11.0412 | -0.9723 | -0.9723 | 0.5407 | 0.5407 |
| 2 | joint_same_relation_other_slot_frame:L8->L12 | 1344 | 614 | 5.8585 | 11.0924 | -0.9212 | -0.9212 | 0.5375 | 0.5375 |
| 3 | joint_same_relation_other_slot_frame:L4->L8 | 1344 | 614 | 5.8833 | 11.0676 | -0.9460 | -0.9460 | 0.5375 | 0.5375 |
| 4 | joint_same_relation_other_phrase_frame:L4->L8 | 1344 | 614 | 6.1684 | 10.7824 | -1.2311 | -1.2311 | 0.5342 | 0.5342 |
| 5 | joint_orth_relation:L4->L8 | 1344 | 614 | 6.1861 | 5.4814 | -1.2488 | -6.5322 | 0.5309 | 0.1906 |
| 6 | joint_orth_phrase:L4->L8 | 1344 | 614 | 6.1786 | 5.4756 | -1.2413 | -6.5380 | 0.5326 | 0.1889 |
| 7 | joint_orth_slot:L4->L8 | 1344 | 614 | 6.1809 | 5.4599 | -1.2436 | -6.5537 | 0.5326 | 0.1889 |
| 8 | joint_raw:L4->L8 | 1344 | 614 | 6.1728 | 5.4606 | -1.2355 | -6.5529 | 0.5326 | 0.1873 |
| 9 | joint_orth_all:L4->L8 | 1344 | 614 | 6.1633 | 5.4501 | -1.2260 | -6.5635 | 0.5375 | 0.1873 |
| 10 | joint_orth_phrase_slot:L4->L8 | 1344 | 614 | 6.1684 | 5.4496 | -1.2311 | -6.5640 | 0.5326 | 0.1857 |
| 11 | joint_orth_all:L8->L12 | 1344 | 614 | 5.8861 | 5.4250 | -0.9488 | -6.5886 | 0.5456 | 0.1857 |
| 12 | joint_raw:L8->L12 | 1344 | 614 | 5.9247 | 5.4476 | -0.9874 | -6.5659 | 0.5472 | 0.1840 |
| 13 | joint_orth_phrase_slot:L8->L12 | 1344 | 614 | 5.8798 | 5.4135 | -0.9425 | -6.6001 | 0.5456 | 0.1840 |
| 14 | joint_orth_relation:L8->L12 | 1344 | 614 | 5.9190 | 5.4108 | -0.9817 | -6.6027 | 0.5472 | 0.1824 |
| 15 | joint_orth_phrase:L8->L12 | 1344 | 614 | 5.9224 | 5.4070 | -0.9851 | -6.6066 | 0.5489 | 0.1824 |
| 16 | joint_orth_slot:L8->L12 | 1344 | 614 | 5.9093 | 5.4406 | -0.9720 | -6.5730 | 0.5456 | 0.1808 |
| 17 | joint_same_object_other_relation_frame:L8->L12 | 1344 | 614 | 5.9145 | 3.0598 | -0.9772 | -8.9537 | 0.5440 | 0.0261 |
| 18 | joint_same_object_other_relation_frame:L4->L8 | 1344 | 614 | 6.1735 | 2.8688 | -1.2361 | -9.1448 | 0.5375 | 0.0228 |
| 19 | joint_raw_restore_both:L8->L12 | 1344 | 614 | 0.2050 | 0.2217 | 4.7323 | -11.7918 | 0.9300 | 0.0033 |
| 20 | joint_raw_restore_both:L4->L8 | 1344 | 614 | 0.2267 | 0.2110 | 4.7106 | -11.8025 | 0.9202 | 0.0033 |
| 21 | joint_relation_basis_only:L4->L8 | 1344 | 614 | 0.0210 | -0.0103 | 4.9163 | -12.0238 | 0.9691 | 0.0016 |
| 22 | joint_relation_basis_only:L8->L12 | 1344 | 614 | 0.0441 | 0.0195 | 4.8932 | -11.9941 | 0.9691 | 0.0000 |
| 23 | joint_slot_basis_only:L8->L12 | 1344 | 614 | 0.0326 | 0.0160 | 4.9047 | -11.9976 | 0.9772 | 0.0000 |
| 24 | joint_phrase_basis_only:L8->L12 | 1344 | 614 | 0.0355 | 0.0127 | 4.9018 | -12.0009 | 0.9739 | 0.0000 |
| 25 | joint_slot_basis_only:L4->L8 | 1344 | 614 | 0.0131 | -0.0037 | 4.9242 | -12.0172 | 0.9674 | 0.0000 |
| 26 | joint_phrase_basis_only:L4->L8 | 1344 | 614 | 0.0232 | -0.0112 | 4.9141 | -12.0247 | 0.9723 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_slot_frame:is_a | 384 | 292 | 5.8098 | 14.9029 | 1.5794 | 1.5794 | 0.6473 | 0.6473 |
| 2 | joint_same_relation_other_phrase_frame:is_a | 384 | 292 | 6.1645 | 14.5482 | 1.2247 | 1.2247 | 0.6370 | 0.6370 |
| 3 | joint_same_relation_other_phrase_frame:location | 384 | 194 | 4.4626 | 12.5321 | 0.5170 | 0.5170 | 0.6340 | 0.6340 |
| 4 | joint_same_relation_other_slot_frame:location | 384 | 194 | 4.3421 | 12.6526 | 0.6375 | 0.6375 | 0.6134 | 0.6134 |
| 5 | joint_same_relation_other_phrase_frame:used_for | 384 | 124 | 5.3016 | 12.9582 | -0.7736 | -0.7736 | 0.6129 | 0.6129 |
| 6 | joint_same_relation_other_slot_frame:used_for | 384 | 124 | 5.2409 | 13.0188 | -0.7130 | -0.7130 | 0.5968 | 0.5968 |
| 7 | joint_same_relation_other_slot_frame:can_do | 384 | 134 | 4.8112 | 9.3807 | -0.2772 | -0.2772 | 0.5896 | 0.5896 |
| 8 | joint_same_relation_other_phrase_frame:can_do | 384 | 134 | 4.9124 | 9.2796 | -0.3784 | -0.3784 | 0.5746 | 0.5746 |
| 9 | joint_same_relation_other_slot_frame:material | 384 | 204 | 5.1994 | 10.5967 | -1.6678 | -1.6678 | 0.4706 | 0.4706 |
| 10 | joint_same_relation_other_phrase_frame:material | 384 | 204 | 5.5295 | 10.2665 | -1.9979 | -1.9979 | 0.4657 | 0.4657 |
| 11 | joint_same_relation_other_phrase_frame:property | 384 | 146 | 8.8671 | 6.0585 | -5.3494 | -5.3494 | 0.3904 | 0.3904 |
| 12 | joint_same_relation_other_slot_frame:property | 384 | 146 | 8.7018 | 6.2239 | -5.1840 | -5.1840 | 0.3767 | 0.3767 |
| 13 | joint_orth_slot:part_of | 384 | 134 | 7.5823 | 7.0674 | -3.5803 | -3.1328 | 0.3358 | 0.3582 |
| 14 | joint_orth_phrase:part_of | 384 | 134 | 7.6077 | 7.0590 | -3.6058 | -3.1412 | 0.3582 | 0.3582 |
| 15 | joint_orth_relation:part_of | 384 | 134 | 7.6313 | 7.0524 | -3.6293 | -3.1477 | 0.3507 | 0.3582 |
| 16 | joint_same_relation_other_slot_frame:part_of | 384 | 134 | 7.7979 | 6.4042 | -3.7959 | -3.7959 | 0.3582 | 0.3582 |
| 17 | joint_orth_phrase_slot:part_of | 384 | 134 | 7.5758 | 7.0150 | -3.5738 | -3.1852 | 0.3433 | 0.3507 |
| 18 | joint_raw:part_of | 384 | 134 | 7.5588 | 7.0307 | -3.5568 | -3.1695 | 0.3433 | 0.3433 |
| 19 | joint_same_relation_other_phrase_frame:part_of | 384 | 134 | 7.5511 | 6.6511 | -3.5491 | -3.5491 | 0.3433 | 0.3433 |
| 20 | joint_orth_all:part_of | 384 | 134 | 7.5080 | 6.9626 | -3.5060 | -3.2375 | 0.3657 | 0.3358 |
| 21 | joint_orth_phrase:property | 384 | 146 | 8.8236 | 5.1446 | -5.3059 | -6.2633 | 0.3973 | 0.2260 |
| 22 | joint_orth_phrase_slot:property | 384 | 146 | 8.7946 | 5.1297 | -5.2769 | -6.2782 | 0.3904 | 0.2260 |
| 23 | joint_orth_all:is_a | 384 | 292 | 6.1559 | 6.6624 | 1.2333 | -6.6612 | 0.6370 | 0.2192 |
| 24 | joint_orth_all:property | 384 | 146 | 8.7720 | 5.1610 | -5.2542 | -6.2468 | 0.3904 | 0.2192 |
| 25 | joint_orth_relation:property | 384 | 146 | 8.8616 | 5.1597 | -5.3439 | -6.2482 | 0.3973 | 0.2192 |
| 26 | joint_orth_relation:is_a | 384 | 292 | 6.1580 | 6.6738 | 1.2312 | -6.6498 | 0.6404 | 0.2123 |
| 27 | joint_raw:property | 384 | 146 | 8.8194 | 5.1490 | -5.3017 | -6.2589 | 0.3904 | 0.2123 |
| 28 | joint_raw:is_a | 384 | 292 | 6.1898 | 6.6883 | 1.1994 | -6.6353 | 0.6404 | 0.2089 |
| 29 | joint_orth_phrase_slot:is_a | 384 | 292 | 6.1289 | 6.6699 | 1.2603 | -6.6536 | 0.6370 | 0.2089 |
| 30 | joint_orth_slot:is_a | 384 | 292 | 6.1571 | 6.6626 | 1.2321 | -6.6609 | 0.6404 | 0.2089 |
| 31 | joint_orth_slot:property | 384 | 146 | 8.8072 | 5.1378 | -5.2894 | -6.2701 | 0.3904 | 0.2055 |
| 32 | joint_orth_phrase:is_a | 384 | 292 | 6.1581 | 6.6359 | 1.2311 | -6.6876 | 0.6404 | 0.2021 |
| 33 | joint_orth_relation:can_do | 384 | 134 | 4.9971 | 4.6192 | -0.4631 | -5.0388 | 0.5896 | 0.2015 |
| 34 | joint_orth_slot:can_do | 384 | 134 | 5.0021 | 4.5969 | -0.4681 | -5.0611 | 0.5896 | 0.2015 |
| 35 | joint_raw:can_do | 384 | 134 | 4.9248 | 4.5757 | -0.3908 | -5.0822 | 0.5821 | 0.2015 |
| 36 | joint_orth_phrase:can_do | 384 | 134 | 4.9664 | 4.5563 | -0.4324 | -5.1017 | 0.5821 | 0.2015 |
| 37 | joint_orth_all:can_do | 384 | 134 | 5.0113 | 4.5838 | -0.4773 | -5.0741 | 0.5896 | 0.1940 |
| 38 | joint_orth_phrase_slot:can_do | 384 | 134 | 4.9521 | 4.5390 | -0.4181 | -5.1190 | 0.5896 | 0.1940 |
| 39 | joint_raw:location | 384 | 194 | 4.5173 | 5.0755 | 0.4624 | -6.9396 | 0.6340 | 0.1392 |
| 40 | joint_orth_slot:location | 384 | 194 | 4.5178 | 5.0570 | 0.4619 | -6.9580 | 0.6340 | 0.1392 |
| 41 | joint_orth_phrase_slot:location | 384 | 194 | 4.4958 | 5.0270 | 0.4839 | -6.9881 | 0.6340 | 0.1392 |
| 42 | joint_orth_phrase:location | 384 | 194 | 4.4994 | 5.0248 | 0.4803 | -6.9902 | 0.6340 | 0.1392 |
| 43 | joint_orth_relation:location | 384 | 194 | 4.4660 | 5.0097 | 0.5136 | -7.0053 | 0.6289 | 0.1392 |
| 44 | joint_orth_all:location | 384 | 194 | 4.4396 | 4.9994 | 0.5400 | -7.0156 | 0.6340 | 0.1392 |
| 45 | joint_raw:material | 384 | 204 | 5.5253 | 4.4390 | -1.9937 | -7.8255 | 0.4706 | 0.1275 |
| 46 | joint_orth_all:material | 384 | 204 | 5.4922 | 4.4524 | -1.9606 | -7.8121 | 0.4608 | 0.1225 |
| 47 | joint_orth_slot:material | 384 | 204 | 5.4855 | 4.4228 | -1.9539 | -7.8417 | 0.4657 | 0.1225 |
| 48 | joint_orth_phrase:material | 384 | 204 | 5.5340 | 4.4387 | -2.0024 | -7.8257 | 0.4706 | 0.1176 |
| 49 | joint_orth_phrase_slot:material | 384 | 204 | 5.4848 | 4.4259 | -1.9532 | -7.8386 | 0.4657 | 0.1176 |
| 50 | joint_orth_relation:material | 384 | 204 | 5.5180 | 4.3939 | -1.9864 | -7.8706 | 0.4706 | 0.1176 |
| 51 | joint_orth_phrase:used_for | 384 | 124 | 5.2974 | 4.4865 | -0.7695 | -9.2453 | 0.5968 | 0.0806 |
| 52 | joint_orth_all:used_for | 384 | 124 | 5.3296 | 4.4594 | -0.8016 | -9.2724 | 0.6210 | 0.0806 |
| 53 | joint_raw:used_for | 384 | 124 | 5.2944 | 4.4151 | -0.7664 | -9.3167 | 0.6129 | 0.0806 |
| 54 | joint_orth_relation:used_for | 384 | 124 | 5.2929 | 4.4638 | -0.7649 | -9.2680 | 0.5887 | 0.0726 |
| 55 | joint_orth_slot:used_for | 384 | 124 | 5.3052 | 4.4431 | -0.7772 | -9.2887 | 0.6129 | 0.0726 |
| 56 | joint_orth_phrase_slot:used_for | 384 | 124 | 5.2751 | 4.4115 | -0.7471 | -9.3203 | 0.6129 | 0.0726 |
| 57 | joint_same_object_other_relation_frame:part_of | 384 | 134 | 7.6101 | 2.5684 | -3.6082 | -7.6318 | 0.3507 | 0.0597 |
| 58 | joint_same_object_other_relation_frame:can_do | 384 | 134 | 4.9607 | 1.8693 | -0.4267 | -7.7887 | 0.5821 | 0.0373 |
| 59 | joint_same_object_other_relation_frame:is_a | 384 | 292 | 6.1636 | 4.1537 | 1.2256 | -9.1699 | 0.6404 | 0.0308 |
| 60 | joint_same_object_other_relation_frame:used_for | 384 | 124 | 5.2837 | 4.4164 | -0.7557 | -9.3154 | 0.6129 | 0.0242 |
| 61 | joint_raw_restore_both:part_of | 384 | 134 | 0.0290 | 0.0472 | 3.9730 | -10.1529 | 0.9254 | 0.0149 |
| 62 | joint_same_object_other_relation_frame:material | 384 | 204 | 5.5230 | 2.4556 | -1.9914 | -9.8088 | 0.4706 | 0.0147 |
| 63 | joint_same_object_other_relation_frame:property | 384 | 146 | 8.8129 | 2.3025 | -5.2951 | -9.1054 | 0.3904 | 0.0137 |
| 64 | joint_raw_restore_both:property | 384 | 146 | 0.1548 | -0.0131 | 3.3630 | -11.4210 | 0.8356 | 0.0068 |
| 65 | joint_raw_restore_both:is_a | 384 | 292 | 0.2362 | 0.4757 | 7.1530 | -12.8479 | 0.9623 | 0.0034 |
| 66 | joint_relation_basis_only:is_a | 384 | 292 | 0.0234 | 0.0154 | 7.3658 | -13.3082 | 0.9932 | 0.0034 |
| 67 | joint_same_object_other_relation_frame:location | 384 | 194 | 4.4803 | 2.3087 | 0.4993 | -9.7064 | 0.6340 | 0.0000 |
| 68 | joint_raw_restore_both:location | 384 | 194 | 0.3492 | 0.4856 | 4.6304 | -11.5295 | 0.9742 | 0.0000 |
| 69 | joint_raw_restore_both:can_do | 384 | 134 | 0.2907 | 0.1546 | 4.2433 | -9.5034 | 0.9328 | 0.0000 |
| 70 | joint_relation_basis_only:location | 384 | 194 | 0.0138 | 0.0429 | 4.9659 | -11.9722 | 0.9897 | 0.0000 |
| 71 | joint_slot_basis_only:part_of | 384 | 134 | 0.0252 | 0.0423 | 3.9768 | -10.1579 | 0.9552 | 0.0000 |
| 72 | joint_slot_basis_only:location | 384 | 194 | 0.0378 | 0.0411 | 4.9419 | -11.9739 | 0.9845 | 0.0000 |
| 73 | joint_relation_basis_only:material | 384 | 204 | 0.0541 | 0.0373 | 3.4775 | -12.2272 | 0.9657 | 0.0000 |
| 74 | joint_phrase_basis_only:location | 384 | 194 | 0.0202 | 0.0340 | 4.9595 | -11.9811 | 0.9897 | 0.0000 |
| 75 | joint_phrase_basis_only:is_a | 384 | 292 | -0.0128 | 0.0266 | 7.4019 | -13.2969 | 0.9966 | 0.0000 |
| 76 | joint_raw_restore_both:material | 384 | 204 | 0.3337 | 0.0264 | 3.1979 | -12.2381 | 0.8873 | 0.0000 |
| 77 | joint_slot_basis_only:can_do | 384 | 134 | 0.0488 | 0.0243 | 4.4851 | -9.6337 | 0.9776 | 0.0000 |
| 78 | joint_raw_restore_both:used_for | 384 | 124 | -0.0419 | 0.0171 | 4.5698 | -13.7147 | 0.9194 | 0.0000 |
| 79 | joint_slot_basis_only:property | 384 | 146 | 0.0766 | 0.0021 | 3.4411 | -11.4058 | 0.9315 | 0.0000 |
| 80 | joint_phrase_basis_only:can_do | 384 | 134 | 0.0529 | -0.0008 | 4.4811 | -9.6588 | 0.9701 | 0.0000 |
| 81 | joint_relation_basis_only:part_of | 384 | 134 | -0.0019 | -0.0046 | 4.0038 | -10.2048 | 0.9403 | 0.0000 |
| 82 | joint_relation_basis_only:property | 384 | 146 | 0.0581 | -0.0083 | 3.4596 | -11.4162 | 0.9315 | 0.0000 |
| 83 | joint_slot_basis_only:is_a | 384 | 292 | -0.0119 | -0.0105 | 7.4011 | -13.3340 | 0.9897 | 0.0000 |
| 84 | joint_phrase_basis_only:part_of | 384 | 134 | 0.0003 | -0.0116 | 4.0017 | -10.2117 | 0.9552 | 0.0000 |
| 85 | joint_slot_basis_only:material | 384 | 204 | -0.0227 | -0.0166 | 3.5543 | -12.2811 | 0.9902 | 0.0000 |
| 86 | joint_phrase_basis_only:material | 384 | 204 | 0.0387 | -0.0166 | 3.4928 | -12.2811 | 0.9755 | 0.0000 |
| 87 | joint_phrase_basis_only:property | 384 | 146 | 0.0593 | -0.0244 | 3.4584 | -11.4323 | 0.9247 | 0.0000 |
| 88 | joint_slot_basis_only:used_for | 384 | 124 | 0.0626 | -0.0258 | 4.4654 | -13.7576 | 0.9435 | 0.0000 |
| 89 | joint_phrase_basis_only:used_for | 384 | 124 | 0.0981 | -0.0390 | 4.4298 | -13.7708 | 0.9677 | 0.0000 |
| 90 | joint_relation_basis_only:can_do | 384 | 134 | 0.0171 | -0.0404 | 4.5169 | -9.6983 | 0.9851 | 0.0000 |
| 91 | joint_relation_basis_only:used_for | 384 | 124 | 0.0722 | -0.0607 | 4.4558 | -13.7925 | 0.9435 | 0.0000 |

### Top condition slots

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_phrase_frame:equals | 672 | 210 | 0.0944 | 10.9563 | 3.4424 | 3.4424 | 0.9714 | 0.9714 |
| 2 | joint_same_relation_other_slot_frame:equals | 672 | 210 | -0.0437 | 11.0944 | 3.5806 | 3.5806 | 0.9524 | 0.9524 |
| 3 | joint_same_relation_other_phrase_frame:value | 672 | 330 | 0.1100 | 19.3529 | 5.2247 | 5.2247 | 0.9394 | 0.9394 |
| 4 | joint_same_relation_other_slot_frame:value | 672 | 330 | -0.0447 | 19.5076 | 5.3794 | 5.3794 | 0.9364 | 0.9364 |
| 5 | joint_orth_relation:answer | 672 | 410 | 13.4493 | 12.1139 | -6.9083 | -2.7681 | 0.2000 | 0.3659 |
| 6 | joint_orth_phrase:answer | 672 | 410 | 13.4577 | 12.0968 | -6.9166 | -2.7851 | 0.1976 | 0.3659 |
| 7 | joint_orth_slot:answer | 672 | 410 | 13.4366 | 12.1635 | -6.8955 | -2.7184 | 0.1951 | 0.3634 |
| 8 | joint_orth_phrase_slot:answer | 672 | 410 | 13.4126 | 12.0908 | -6.8715 | -2.7911 | 0.1951 | 0.3634 |
| 9 | joint_orth_all:answer | 672 | 410 | 13.3847 | 12.0873 | -6.8436 | -2.7947 | 0.1927 | 0.3634 |
| 10 | joint_raw:answer | 672 | 410 | 13.4687 | 12.1982 | -6.9277 | -2.6838 | 0.2000 | 0.3561 |
| 11 | joint_raw:arrow | 672 | 278 | 6.6260 | 5.8934 | -3.4678 | -2.7786 | 0.2374 | 0.2950 |
| 12 | joint_orth_all:arrow | 672 | 278 | 6.5907 | 5.9348 | -3.4324 | -2.7372 | 0.2446 | 0.2878 |
| 13 | joint_orth_relation:arrow | 672 | 278 | 6.6428 | 5.9226 | -3.4845 | -2.7494 | 0.2446 | 0.2842 |
| 14 | joint_orth_phrase:arrow | 672 | 278 | 6.6411 | 5.9340 | -3.4828 | -2.7380 | 0.2482 | 0.2806 |
| 15 | joint_orth_phrase_slot:arrow | 672 | 278 | 6.6166 | 5.9336 | -3.4583 | -2.7384 | 0.2446 | 0.2806 |
| 16 | joint_orth_slot:arrow | 672 | 278 | 6.6327 | 5.9183 | -3.4745 | -2.7537 | 0.2446 | 0.2770 |
| 17 | joint_same_relation_other_phrase_frame:arrow | 672 | 278 | 6.6624 | 5.1678 | -3.5041 | -3.5041 | 0.2410 | 0.2410 |
| 18 | joint_same_relation_other_slot_frame:arrow | 672 | 278 | 6.6261 | 5.2042 | -3.4678 | -3.4678 | 0.2338 | 0.2338 |
| 19 | joint_same_relation_other_slot_frame:answer | 672 | 410 | 13.1496 | 8.2734 | -6.6085 | -6.6085 | 0.2098 | 0.2098 |
| 20 | joint_same_relation_other_phrase_frame:answer | 672 | 410 | 13.4333 | 7.9898 | -6.8922 | -6.8922 | 0.1927 | 0.1927 |
| 21 | joint_same_object_other_relation_frame:answer | 672 | 410 | 13.4378 | 3.9928 | -6.8967 | -10.8891 | 0.1976 | 0.0366 |
| 22 | joint_same_object_other_relation_frame:arrow | 672 | 278 | 6.6540 | 1.5070 | -3.4957 | -7.1650 | 0.2410 | 0.0360 |
| 23 | joint_same_object_other_relation_frame:equals | 672 | 210 | 0.1199 | 1.4400 | 3.4169 | -6.0739 | 0.9810 | 0.0095 |
| 24 | joint_same_object_other_relation_frame:value | 672 | 330 | 0.1137 | 3.8841 | 5.2210 | -10.2441 | 0.9394 | 0.0091 |
| 25 | joint_raw_restore_both:arrow | 672 | 278 | 0.4568 | 0.4049 | 2.7014 | -8.2671 | 0.8705 | 0.0072 |
| 26 | joint_raw_restore_both:answer | 672 | 410 | 0.2449 | 0.3454 | 6.2962 | -14.5366 | 0.9220 | 0.0049 |
| 27 | joint_orth_slot:equals | 672 | 210 | 0.1595 | 0.1084 | 3.3773 | -7.4055 | 0.9667 | 0.0048 |
| 28 | joint_relation_basis_only:value | 672 | 330 | -0.0020 | -0.0292 | 5.3366 | -14.1575 | 0.9697 | 0.0030 |
| 29 | joint_orth_relation:value | 672 | 330 | 0.1369 | 0.1503 | 5.1978 | -13.9779 | 0.9394 | 0.0000 |
| 30 | joint_orth_phrase:value | 672 | 330 | 0.1171 | 0.1421 | 5.2175 | -13.9861 | 0.9455 | 0.0000 |
| 31 | joint_orth_all:value | 672 | 330 | 0.1520 | 0.1399 | 5.1827 | -13.9883 | 0.9455 | 0.0000 |
| 32 | joint_orth_phrase:equals | 672 | 210 | 0.1310 | 0.1221 | 3.4058 | -7.3918 | 0.9619 | 0.0000 |
| 33 | joint_orth_all:equals | 672 | 210 | 0.1346 | 0.1212 | 3.4022 | -7.3926 | 0.9810 | 0.0000 |
| 34 | joint_orth_phrase_slot:equals | 672 | 210 | 0.1428 | 0.1204 | 3.3940 | -7.3934 | 0.9667 | 0.0000 |
| 35 | joint_orth_relation:equals | 672 | 210 | 0.1258 | 0.1193 | 3.4110 | -7.3946 | 0.9619 | 0.0000 |
| 36 | joint_raw:equals | 672 | 210 | 0.1490 | 0.1182 | 3.3878 | -7.3956 | 0.9762 | 0.0000 |
| 37 | joint_orth_phrase_slot:value | 672 | 330 | 0.0879 | 0.1147 | 5.2467 | -14.0135 | 0.9424 | 0.0000 |
| 38 | joint_orth_slot:value | 672 | 330 | 0.1120 | 0.1145 | 5.2227 | -14.0137 | 0.9424 | 0.0000 |
| 39 | joint_raw:value | 672 | 330 | 0.0981 | 0.1007 | 5.2366 | -14.0275 | 0.9394 | 0.0000 |
| 40 | joint_relation_basis_only:answer | 672 | 410 | 0.0589 | 0.0542 | 6.4822 | -14.8278 | 0.9683 | 0.0000 |
| 41 | joint_slot_basis_only:answer | 672 | 410 | 0.0529 | 0.0453 | 6.4882 | -14.8366 | 0.9732 | 0.0000 |
| 42 | joint_raw_restore_both:equals | 672 | 210 | 0.1051 | 0.0423 | 3.4317 | -7.4715 | 0.9714 | 0.0000 |
| 43 | joint_phrase_basis_only:answer | 672 | 410 | 0.0535 | 0.0367 | 6.4876 | -14.8452 | 0.9732 | 0.0000 |
| 44 | joint_raw_restore_both:value | 672 | 330 | 0.0471 | 0.0081 | 5.2875 | -14.1202 | 0.9455 | 0.0000 |
| 45 | joint_slot_basis_only:equals | 672 | 210 | -0.0176 | -0.0078 | 3.5544 | -7.5216 | 0.9810 | 0.0000 |
| 46 | joint_relation_basis_only:arrow | 672 | 278 | 0.0627 | -0.0117 | 3.0956 | -8.6837 | 0.9568 | 0.0000 |
| 47 | joint_phrase_basis_only:arrow | 672 | 278 | 0.0687 | -0.0118 | 3.0896 | -8.6837 | 0.9640 | 0.0000 |
| 48 | joint_slot_basis_only:value | 672 | 330 | -0.0156 | -0.0125 | 5.3503 | -14.1407 | 0.9727 | 0.0000 |
| 49 | joint_phrase_basis_only:equals | 672 | 210 | -0.0020 | -0.0145 | 3.5389 | -7.5283 | 0.9857 | 0.0000 |
| 50 | joint_relation_basis_only:equals | 672 | 210 | -0.0043 | -0.0174 | 3.5411 | -7.5313 | 0.9857 | 0.0000 |
| 51 | joint_slot_basis_only:arrow | 672 | 278 | 0.0548 | -0.0189 | 3.1034 | -8.6909 | 0.9640 | 0.0000 |
| 52 | joint_phrase_basis_only:value | 672 | 330 | -0.0138 | -0.0238 | 5.3484 | -14.1520 | 0.9727 | 0.0000 |

## glm4

items=1344, basis_items=448, rows=34944, layer_pairs=[[4, 10], [10, 20]]
module=resid_out, contrast_rank=64, nuisance_rank=24
relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for'], phrase_ids=[0, 1, 2, 3], slot_ids=['answer', 'arrow', 'equals', 'value']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_slot_frame | 2688 | 1500 | 3.9990 | 10.1912 | 0.1757 | 0.1757 | 0.5987 | 0.5987 |
| 2 | joint_same_relation_other_phrase_frame | 2688 | 1500 | 4.0497 | 10.1406 | 0.1251 | 0.1251 | 0.5967 | 0.5967 |
| 3 | joint_raw | 2688 | 1500 | 4.0473 | 4.2601 | 0.1275 | -5.7554 | 0.5933 | 0.1507 |
| 4 | joint_orth_relation | 2688 | 1500 | 4.0435 | 4.2803 | 0.1313 | -5.7352 | 0.5960 | 0.1500 |
| 5 | joint_orth_slot | 2688 | 1500 | 4.0357 | 4.2757 | 0.1391 | -5.7398 | 0.5973 | 0.1487 |
| 6 | joint_orth_phrase_slot | 2688 | 1500 | 4.0323 | 4.2691 | 0.1425 | -5.7464 | 0.6007 | 0.1467 |
| 7 | joint_orth_all | 2688 | 1500 | 4.0239 | 4.2692 | 0.1508 | -5.7462 | 0.5980 | 0.1460 |
| 8 | joint_orth_phrase | 2688 | 1500 | 4.0425 | 4.2795 | 0.1323 | -5.7360 | 0.6000 | 0.1447 |
| 9 | joint_same_object_other_relation_frame | 2688 | 1500 | 4.0399 | 2.4902 | 0.1349 | -7.5253 | 0.5947 | 0.0220 |
| 10 | joint_raw_restore_both | 2688 | 1500 | 0.1564 | 0.2349 | 4.0184 | -9.7806 | 0.9500 | 0.0073 |
| 11 | joint_relation_basis_only | 2688 | 1500 | -0.0044 | -0.0288 | 4.1792 | -10.0442 | 0.9860 | 0.0027 |
| 12 | joint_phrase_basis_only | 2688 | 1500 | -0.0068 | -0.0328 | 4.1816 | -10.0483 | 0.9880 | 0.0013 |
| 13 | joint_slot_basis_only | 2688 | 1500 | -0.0009 | -0.0239 | 4.1757 | -10.0394 | 0.9920 | 0.0007 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_phrase_frame:L10->L20 | 1344 | 750 | 3.8227 | 10.3676 | 0.3521 | 0.3521 | 0.6160 | 0.6160 |
| 2 | joint_same_relation_other_slot_frame:L10->L20 | 1344 | 750 | 3.7400 | 10.4503 | 0.4348 | 0.4348 | 0.6147 | 0.6147 |
| 3 | joint_same_relation_other_slot_frame:L4->L10 | 1344 | 750 | 4.2581 | 9.9322 | -0.0833 | -0.0833 | 0.5827 | 0.5827 |
| 4 | joint_same_relation_other_phrase_frame:L4->L10 | 1344 | 750 | 4.2767 | 9.9136 | -0.1019 | -0.1019 | 0.5773 | 0.5773 |
| 5 | joint_orth_relation:L4->L10 | 1344 | 750 | 4.2724 | 4.4120 | -0.0976 | -5.6035 | 0.5800 | 0.1667 |
| 6 | joint_raw:L4->L10 | 1344 | 750 | 4.2628 | 4.3882 | -0.0880 | -5.6273 | 0.5747 | 0.1667 |
| 7 | joint_orth_slot:L4->L10 | 1344 | 750 | 4.2652 | 4.4031 | -0.0904 | -5.6124 | 0.5773 | 0.1653 |
| 8 | joint_orth_all:L4->L10 | 1344 | 750 | 4.2617 | 4.4008 | -0.0869 | -5.6146 | 0.5787 | 0.1640 |
| 9 | joint_orth_phrase_slot:L4->L10 | 1344 | 750 | 4.2629 | 4.3999 | -0.0881 | -5.6156 | 0.5827 | 0.1640 |
| 10 | joint_orth_phrase:L4->L10 | 1344 | 750 | 4.2687 | 4.4085 | -0.0939 | -5.6070 | 0.5867 | 0.1600 |
| 11 | joint_raw:L10->L20 | 1344 | 750 | 3.8318 | 4.1319 | 0.3430 | -5.8836 | 0.6120 | 0.1347 |
| 12 | joint_orth_relation:L10->L20 | 1344 | 750 | 3.8146 | 4.1487 | 0.3602 | -5.8668 | 0.6120 | 0.1333 |
| 13 | joint_orth_slot:L10->L20 | 1344 | 750 | 3.8061 | 4.1482 | 0.3686 | -5.8673 | 0.6173 | 0.1320 |
| 14 | joint_orth_phrase:L10->L20 | 1344 | 750 | 3.8163 | 4.1505 | 0.3585 | -5.8649 | 0.6133 | 0.1293 |
| 15 | joint_orth_phrase_slot:L10->L20 | 1344 | 750 | 3.8017 | 4.1384 | 0.3731 | -5.8771 | 0.6187 | 0.1293 |
| 16 | joint_orth_all:L10->L20 | 1344 | 750 | 3.7862 | 4.1376 | 0.3886 | -5.8778 | 0.6173 | 0.1280 |
| 17 | joint_same_object_other_relation_frame:L4->L10 | 1344 | 750 | 4.2636 | 2.4526 | -0.0888 | -7.5629 | 0.5773 | 0.0240 |
| 18 | joint_same_object_other_relation_frame:L10->L20 | 1344 | 750 | 3.8163 | 2.5279 | 0.3585 | -7.4876 | 0.6120 | 0.0200 |
| 19 | joint_raw_restore_both:L4->L10 | 1344 | 750 | 0.1449 | 0.1864 | 4.0299 | -9.8291 | 0.9480 | 0.0080 |
| 20 | joint_raw_restore_both:L10->L20 | 1344 | 750 | 0.1679 | 0.2834 | 4.0069 | -9.7321 | 0.9520 | 0.0067 |
| 21 | joint_relation_basis_only:L4->L10 | 1344 | 750 | 0.0011 | -0.0261 | 4.1736 | -10.0416 | 0.9920 | 0.0027 |
| 22 | joint_relation_basis_only:L10->L20 | 1344 | 750 | -0.0100 | -0.0314 | 4.1848 | -10.0469 | 0.9800 | 0.0027 |
| 23 | joint_phrase_basis_only:L4->L10 | 1344 | 750 | 0.0042 | -0.0208 | 4.1706 | -10.0363 | 0.9920 | 0.0013 |
| 24 | joint_slot_basis_only:L10->L20 | 1344 | 750 | -0.0076 | -0.0327 | 4.1824 | -10.0482 | 0.9867 | 0.0013 |
| 25 | joint_phrase_basis_only:L10->L20 | 1344 | 750 | -0.0178 | -0.0448 | 4.1926 | -10.0602 | 0.9840 | 0.0013 |
| 26 | joint_slot_basis_only:L4->L10 | 1344 | 750 | 0.0058 | -0.0150 | 4.1690 | -10.0305 | 0.9973 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_phrase_frame:location | 384 | 252 | 2.9239 | 11.1994 | 1.7013 | 1.7013 | 0.7222 | 0.7222 |
| 2 | joint_same_relation_other_slot_frame:location | 384 | 252 | 2.8658 | 11.2574 | 1.7594 | 1.7594 | 0.7183 | 0.7183 |
| 3 | joint_same_relation_other_slot_frame:material | 384 | 176 | 2.3022 | 10.4484 | 0.8013 | 0.8013 | 0.6761 | 0.6761 |
| 4 | joint_same_relation_other_phrase_frame:is_a | 384 | 346 | 4.3394 | 12.7814 | 1.5421 | 1.5421 | 0.6734 | 0.6734 |
| 5 | joint_same_relation_other_slot_frame:is_a | 384 | 346 | 4.2607 | 12.8601 | 1.6208 | 1.6208 | 0.6647 | 0.6647 |
| 6 | joint_same_relation_other_phrase_frame:material | 384 | 176 | 2.2853 | 10.4654 | 0.8182 | 0.8182 | 0.6420 | 0.6420 |
| 7 | joint_same_relation_other_slot_frame:can_do | 384 | 196 | 3.6505 | 9.8073 | 0.5098 | 0.5098 | 0.5714 | 0.5714 |
| 8 | joint_same_relation_other_phrase_frame:can_do | 384 | 196 | 3.7726 | 9.6852 | 0.3877 | 0.3877 | 0.5714 | 0.5714 |
| 9 | joint_same_relation_other_phrase_frame:used_for | 384 | 268 | 4.2072 | 10.2092 | -0.4829 | -0.4829 | 0.5672 | 0.5672 |
| 10 | joint_same_relation_other_slot_frame:used_for | 384 | 268 | 4.1432 | 10.2733 | -0.4189 | -0.4189 | 0.5597 | 0.5597 |
| 11 | joint_same_relation_other_slot_frame:part_of | 384 | 100 | 5.5443 | 2.9207 | -2.9767 | -2.9767 | 0.4300 | 0.4300 |
| 12 | joint_same_relation_other_phrase_frame:part_of | 384 | 100 | 5.5755 | 2.8895 | -3.0079 | -3.0079 | 0.4100 | 0.4100 |
| 13 | joint_raw:part_of | 384 | 100 | 5.5969 | 5.8083 | -3.0293 | -0.0891 | 0.3900 | 0.4000 |
| 14 | joint_orth_slot:part_of | 384 | 100 | 5.5778 | 5.8012 | -3.0103 | -0.0962 | 0.3900 | 0.4000 |
| 15 | joint_same_relation_other_slot_frame:property | 384 | 162 | 6.2758 | 7.3699 | -3.5281 | -3.5281 | 0.3889 | 0.3889 |
| 16 | joint_same_relation_other_phrase_frame:property | 384 | 162 | 6.2320 | 7.4137 | -3.4843 | -3.4843 | 0.3827 | 0.3827 |
| 17 | joint_orth_relation:part_of | 384 | 100 | 5.6154 | 5.8380 | -3.0478 | -0.0594 | 0.4000 | 0.3800 |
| 18 | joint_orth_all:part_of | 384 | 100 | 5.5500 | 5.7958 | -2.9824 | -0.1016 | 0.4100 | 0.3700 |
| 19 | joint_orth_phrase_slot:part_of | 384 | 100 | 5.5557 | 5.7665 | -2.9881 | -0.1309 | 0.4000 | 0.3700 |
| 20 | joint_orth_phrase:part_of | 384 | 100 | 5.5909 | 5.8113 | -3.0233 | -0.0861 | 0.4200 | 0.3600 |
| 21 | joint_orth_relation:location | 384 | 252 | 2.8851 | 4.5042 | 1.7401 | -4.9939 | 0.7222 | 0.1825 |
| 22 | joint_orth_phrase_slot:location | 384 | 252 | 2.9026 | 4.5200 | 1.7226 | -4.9781 | 0.7381 | 0.1786 |
| 23 | joint_orth_slot:location | 384 | 252 | 2.9082 | 4.5119 | 1.7169 | -4.9862 | 0.7262 | 0.1786 |
| 24 | joint_raw:location | 384 | 252 | 2.9095 | 4.4755 | 1.7157 | -5.0226 | 0.7262 | 0.1786 |
| 25 | joint_orth_phrase:location | 384 | 252 | 2.9126 | 4.5320 | 1.7126 | -4.9660 | 0.7302 | 0.1746 |
| 26 | joint_orth_all:location | 384 | 252 | 2.8778 | 4.5094 | 1.7473 | -4.9887 | 0.7262 | 0.1746 |
| 27 | joint_orth_relation:can_do | 384 | 196 | 3.7706 | 4.0630 | 0.3897 | -5.2345 | 0.5816 | 0.1480 |
| 28 | joint_orth_all:can_do | 384 | 196 | 3.7605 | 4.0581 | 0.3998 | -5.2394 | 0.5663 | 0.1480 |
| 29 | joint_orth_phrase:can_do | 384 | 196 | 3.7487 | 4.0434 | 0.4116 | -5.2541 | 0.5816 | 0.1480 |
| 30 | joint_raw:can_do | 384 | 196 | 3.7655 | 3.9759 | 0.3948 | -5.3216 | 0.5765 | 0.1480 |
| 31 | joint_orth_phrase_slot:can_do | 384 | 196 | 3.7568 | 4.0554 | 0.4035 | -5.2421 | 0.5867 | 0.1429 |
| 32 | joint_orth_slot:can_do | 384 | 196 | 3.7311 | 4.0046 | 0.4292 | -5.2929 | 0.5816 | 0.1429 |
| 33 | joint_raw:used_for | 384 | 268 | 4.2135 | 3.9820 | -0.4892 | -6.7101 | 0.5634 | 0.1306 |
| 34 | joint_orth_relation:is_a | 384 | 346 | 4.3212 | 4.4716 | 1.5603 | -6.7677 | 0.6763 | 0.1301 |
| 35 | joint_orth_all:is_a | 384 | 346 | 4.3111 | 4.4686 | 1.5704 | -6.7707 | 0.6763 | 0.1301 |
| 36 | joint_orth_slot:is_a | 384 | 346 | 4.3294 | 4.5033 | 1.5521 | -6.7360 | 0.6792 | 0.1272 |
| 37 | joint_orth_phrase_slot:is_a | 384 | 346 | 4.3046 | 4.4580 | 1.5769 | -6.7813 | 0.6792 | 0.1272 |
| 38 | joint_orth_phrase:is_a | 384 | 346 | 4.3105 | 4.4479 | 1.5710 | -6.7914 | 0.6821 | 0.1243 |
| 39 | joint_raw:is_a | 384 | 346 | 4.3365 | 4.4815 | 1.5451 | -6.7578 | 0.6763 | 0.1214 |
| 40 | joint_orth_relation:used_for | 384 | 268 | 4.2100 | 3.9559 | -0.4857 | -6.7362 | 0.5634 | 0.1194 |
| 41 | joint_orth_slot:used_for | 384 | 268 | 4.1907 | 3.9669 | -0.4665 | -6.7252 | 0.5672 | 0.1157 |
| 42 | joint_orth_phrase:used_for | 384 | 268 | 4.2082 | 3.9665 | -0.4839 | -6.7256 | 0.5634 | 0.1157 |
| 43 | joint_orth_phrase_slot:used_for | 384 | 268 | 4.2018 | 3.9508 | -0.4775 | -6.7413 | 0.5634 | 0.1157 |
| 44 | joint_orth_all:used_for | 384 | 268 | 4.1878 | 3.9217 | -0.4635 | -6.7705 | 0.5672 | 0.1157 |
| 45 | joint_orth_phrase:property | 384 | 162 | 6.2289 | 4.9402 | -3.4812 | -5.9578 | 0.3827 | 0.1111 |
| 46 | joint_orth_relation:property | 384 | 162 | 6.2225 | 4.9356 | -3.4747 | -5.9624 | 0.3827 | 0.1111 |
| 47 | joint_orth_slot:property | 384 | 162 | 6.2179 | 4.9312 | -3.4702 | -5.9667 | 0.3827 | 0.1111 |
| 48 | joint_orth_phrase_slot:property | 384 | 162 | 6.2186 | 4.9236 | -3.4708 | -5.9744 | 0.3827 | 0.1111 |
| 49 | joint_raw:property | 384 | 162 | 6.2107 | 4.8758 | -3.4630 | -6.0222 | 0.3765 | 0.1111 |
| 50 | joint_orth_all:property | 384 | 162 | 6.2017 | 4.9268 | -3.4540 | -5.9712 | 0.3889 | 0.0988 |
| 51 | joint_orth_relation:material | 384 | 176 | 2.3075 | 2.8317 | 0.7960 | -6.8155 | 0.6307 | 0.0966 |
| 52 | joint_orth_all:material | 384 | 176 | 2.2727 | 2.8252 | 0.8308 | -6.8220 | 0.6420 | 0.0966 |
| 53 | joint_raw:material | 384 | 176 | 2.2968 | 2.8097 | 0.8067 | -6.8375 | 0.6193 | 0.0966 |
| 54 | joint_orth_phrase_slot:material | 384 | 176 | 2.2850 | 2.8081 | 0.8185 | -6.8390 | 0.6364 | 0.0966 |
| 55 | joint_orth_slot:material | 384 | 176 | 2.2908 | 2.7916 | 0.8127 | -6.8556 | 0.6307 | 0.0966 |
| 56 | joint_orth_phrase:material | 384 | 176 | 2.3163 | 2.8482 | 0.7872 | -6.7989 | 0.6307 | 0.0909 |
| 57 | joint_same_object_other_relation_frame:part_of | 384 | 100 | 5.5569 | 0.0169 | -2.9893 | -5.8805 | 0.4100 | 0.0800 |
| 58 | joint_same_object_other_relation_frame:is_a | 384 | 346 | 4.3241 | 3.5572 | 1.5574 | -7.6821 | 0.6792 | 0.0405 |
| 59 | joint_raw_restore_both:part_of | 384 | 100 | -0.0742 | -0.1461 | 2.6417 | -6.0435 | 0.8700 | 0.0300 |
| 60 | joint_same_object_other_relation_frame:property | 384 | 162 | 6.2214 | 2.4530 | -3.4736 | -8.4450 | 0.3765 | 0.0247 |
| 61 | joint_relation_basis_only:part_of | 384 | 100 | -0.0165 | -0.0344 | 2.5841 | -5.9318 | 0.9700 | 0.0200 |
| 62 | joint_phrase_basis_only:part_of | 384 | 100 | -0.0193 | -0.0404 | 2.5869 | -5.9378 | 0.9700 | 0.0200 |
| 63 | joint_same_object_other_relation_frame:can_do | 384 | 196 | 3.7411 | 2.6181 | 0.4192 | -6.6795 | 0.5714 | 0.0153 |
| 64 | joint_raw_restore_both:property | 384 | 162 | 0.1901 | 0.1754 | 2.5577 | -10.7226 | 0.8951 | 0.0123 |
| 65 | joint_same_object_other_relation_frame:material | 384 | 176 | 2.3103 | 1.6844 | 0.7932 | -7.9628 | 0.6193 | 0.0114 |
| 66 | joint_raw_restore_both:used_for | 384 | 268 | -0.0386 | 0.2139 | 3.7629 | -10.4783 | 0.9590 | 0.0112 |
| 67 | joint_slot_basis_only:part_of | 384 | 100 | -0.0231 | -0.0258 | 2.5907 | -5.9232 | 0.9900 | 0.0100 |
| 68 | joint_same_object_other_relation_frame:used_for | 384 | 268 | 4.1818 | 2.6742 | -0.4575 | -8.0180 | 0.5634 | 0.0075 |
| 69 | joint_relation_basis_only:used_for | 384 | 268 | 0.0272 | 0.0436 | 3.6971 | -10.6485 | 0.9776 | 0.0075 |
| 70 | joint_raw_restore_both:material | 384 | 176 | 0.2010 | 0.0794 | 2.9025 | -9.5678 | 0.9432 | 0.0057 |
| 71 | joint_raw_restore_both:can_do | 384 | 196 | 0.2615 | 0.2829 | 3.8988 | -9.0146 | 0.9490 | 0.0051 |
| 72 | joint_raw_restore_both:is_a | 384 | 346 | 0.2388 | 0.2833 | 5.6428 | -10.9560 | 0.9827 | 0.0029 |
| 73 | joint_same_object_other_relation_frame:location | 384 | 252 | 2.9349 | 2.2985 | 1.6903 | -7.1996 | 0.7262 | 0.0000 |
| 74 | joint_raw_restore_both:location | 384 | 252 | 0.2076 | 0.4516 | 4.4175 | -9.0464 | 0.9683 | 0.0000 |
| 75 | joint_phrase_basis_only:is_a | 384 | 346 | 0.0241 | 0.0428 | 5.8574 | -11.1965 | 0.9971 | 0.0000 |
| 76 | joint_slot_basis_only:used_for | 384 | 268 | 0.0189 | 0.0268 | 3.7054 | -10.6654 | 0.9888 | 0.0000 |
| 77 | joint_phrase_basis_only:used_for | 384 | 268 | 0.0108 | 0.0225 | 3.7135 | -10.6696 | 0.9813 | 0.0000 |
| 78 | joint_relation_basis_only:is_a | 384 | 346 | -0.0046 | 0.0102 | 5.8861 | -11.2291 | 0.9971 | 0.0000 |
| 79 | joint_slot_basis_only:is_a | 384 | 346 | -0.0008 | -0.0149 | 5.8823 | -11.2542 | 1.0000 | 0.0000 |
| 80 | joint_slot_basis_only:material | 384 | 176 | -0.0157 | -0.0219 | 3.1192 | -9.6691 | 1.0000 | 0.0000 |
| 81 | joint_relation_basis_only:material | 384 | 176 | 0.0035 | -0.0298 | 3.1000 | -9.6770 | 0.9943 | 0.0000 |
| 82 | joint_slot_basis_only:property | 384 | 162 | 0.0120 | -0.0365 | 2.7357 | -10.9345 | 0.9877 | 0.0000 |
| 83 | joint_slot_basis_only:can_do | 384 | 196 | -0.0207 | -0.0423 | 4.1810 | -9.3398 | 0.9796 | 0.0000 |
| 84 | joint_relation_basis_only:property | 384 | 162 | -0.0099 | -0.0575 | 2.7576 | -10.9554 | 0.9815 | 0.0000 |
| 85 | joint_phrase_basis_only:material | 384 | 176 | -0.0111 | -0.0592 | 3.1146 | -9.7064 | 1.0000 | 0.0000 |
| 86 | joint_phrase_basis_only:property | 384 | 162 | -0.0186 | -0.0674 | 2.7663 | -10.9654 | 0.9815 | 0.0000 |
| 87 | joint_slot_basis_only:location | 384 | 252 | 0.0042 | -0.0682 | 4.6210 | -9.5663 | 0.9921 | 0.0000 |
| 88 | joint_phrase_basis_only:can_do | 384 | 196 | -0.0507 | -0.0783 | 4.2110 | -9.3758 | 0.9796 | 0.0000 |
| 89 | joint_relation_basis_only:location | 384 | 252 | -0.0009 | -0.0868 | 4.6260 | -9.5849 | 0.9881 | 0.0000 |
| 90 | joint_relation_basis_only:can_do | 384 | 196 | -0.0485 | -0.0943 | 4.2088 | -9.3918 | 0.9796 | 0.0000 |
| 91 | joint_phrase_basis_only:location | 384 | 252 | -0.0183 | -0.1161 | 4.6434 | -9.6142 | 0.9921 | 0.0000 |

### Top condition slots

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_phrase_frame:value | 672 | 358 | 0.0501 | 14.9712 | 4.3813 | 4.3813 | 0.9888 | 0.9888 |
| 2 | joint_same_relation_other_slot_frame:value | 672 | 358 | 0.0690 | 14.9523 | 4.3623 | 4.3623 | 0.9860 | 0.9860 |
| 3 | joint_same_relation_other_slot_frame:equals | 672 | 330 | -0.2995 | 11.7207 | 3.9218 | 3.9218 | 0.9758 | 0.9758 |
| 4 | joint_same_relation_other_phrase_frame:equals | 672 | 330 | 0.0802 | 11.3410 | 3.5421 | 3.5421 | 0.9758 | 0.9758 |
| 5 | joint_orth_relation:arrow | 672 | 344 | 6.0392 | 6.5421 | -2.7442 | -1.6333 | 0.2849 | 0.3488 |
| 6 | joint_raw:arrow | 672 | 344 | 6.0559 | 6.4640 | -2.7609 | -1.7114 | 0.2791 | 0.3430 |
| 7 | joint_orth_slot:arrow | 672 | 344 | 6.0340 | 6.5093 | -2.7390 | -1.6661 | 0.2849 | 0.3401 |
| 8 | joint_orth_phrase:arrow | 672 | 344 | 6.0340 | 6.5344 | -2.7391 | -1.6410 | 0.2936 | 0.3372 |
| 9 | joint_orth_phrase_slot:arrow | 672 | 344 | 6.0308 | 6.5150 | -2.7359 | -1.6604 | 0.2936 | 0.3372 |
| 10 | joint_orth_all:arrow | 672 | 344 | 6.0277 | 6.5388 | -2.7327 | -1.6367 | 0.2849 | 0.3343 |
| 11 | joint_same_relation_other_slot_frame:arrow | 672 | 344 | 6.0487 | 5.4217 | -2.7537 | -2.7537 | 0.3023 | 0.3023 |
| 12 | joint_same_relation_other_phrase_frame:arrow | 672 | 344 | 6.0636 | 5.4068 | -2.7687 | -2.7687 | 0.2849 | 0.2849 |
| 13 | joint_same_relation_other_phrase_frame:answer | 672 | 468 | 8.4279 | 9.0784 | -3.4131 | -3.4131 | 0.2585 | 0.2585 |
| 14 | joint_same_relation_other_slot_frame:answer | 672 | 468 | 8.5298 | 8.9766 | -3.5150 | -3.5150 | 0.2543 | 0.2543 |
| 15 | joint_orth_slot:answer | 672 | 468 | 8.4280 | 8.7290 | -3.4132 | -3.7625 | 0.2564 | 0.2222 |
| 16 | joint_raw:answer | 672 | 468 | 8.4456 | 8.7163 | -3.4308 | -3.7753 | 0.2585 | 0.2222 |
| 17 | joint_orth_relation:answer | 672 | 468 | 8.4340 | 8.7207 | -3.4192 | -3.7709 | 0.2585 | 0.2179 |
| 18 | joint_orth_phrase_slot:answer | 672 | 468 | 8.4121 | 8.7126 | -3.3973 | -3.7790 | 0.2607 | 0.2179 |
| 19 | joint_orth_all:answer | 672 | 468 | 8.3969 | 8.6962 | -3.3821 | -3.7953 | 0.2607 | 0.2158 |
| 20 | joint_orth_phrase:answer | 672 | 468 | 8.4363 | 8.7174 | -3.4215 | -3.7741 | 0.2607 | 0.2115 |
| 21 | joint_same_object_other_relation_frame:answer | 672 | 468 | 8.4373 | 3.0033 | -3.4225 | -9.4882 | 0.2585 | 0.0321 |
| 22 | joint_same_object_other_relation_frame:arrow | 672 | 344 | 6.0521 | 1.3065 | -2.7571 | -6.8690 | 0.2849 | 0.0262 |
| 23 | joint_same_object_other_relation_frame:equals | 672 | 330 | 0.0356 | 2.1705 | 3.5867 | -5.6284 | 0.9667 | 0.0212 |
| 24 | joint_raw_restore_both:arrow | 672 | 344 | 0.3397 | 0.7080 | 2.9553 | -7.4674 | 0.9012 | 0.0203 |
| 25 | joint_raw:equals | 672 | 330 | 0.0546 | 0.1341 | 3.5677 | -7.6648 | 0.9667 | 0.0121 |
| 26 | joint_relation_basis_only:equals | 672 | 330 | 0.0192 | 0.0120 | 3.6032 | -7.7869 | 0.9758 | 0.0121 |
| 27 | joint_orth_relation:equals | 672 | 330 | 0.0650 | 0.1392 | 3.5573 | -7.6597 | 0.9727 | 0.0091 |
| 28 | joint_orth_all:equals | 672 | 330 | 0.0565 | 0.1349 | 3.5658 | -7.6639 | 0.9758 | 0.0091 |
| 29 | joint_orth_phrase:equals | 672 | 330 | 0.0666 | 0.1474 | 3.5557 | -7.6515 | 0.9758 | 0.0061 |
| 30 | joint_orth_slot:equals | 672 | 330 | 0.0488 | 0.1337 | 3.5736 | -7.6652 | 0.9788 | 0.0061 |
| 31 | joint_orth_phrase_slot:equals | 672 | 330 | 0.0633 | 0.1289 | 3.5591 | -7.6700 | 0.9788 | 0.0061 |
| 32 | joint_raw_restore_both:equals | 672 | 330 | 0.0373 | 0.0337 | 3.5850 | -7.7651 | 0.9758 | 0.0061 |
| 33 | joint_phrase_basis_only:equals | 672 | 330 | 0.0127 | 0.0031 | 3.6097 | -7.7958 | 0.9788 | 0.0061 |
| 34 | joint_same_object_other_relation_frame:value | 672 | 358 | 0.0489 | 3.2517 | 4.3824 | -7.3383 | 0.9888 | 0.0056 |
| 35 | joint_raw_restore_both:answer | 672 | 468 | 0.2329 | 0.2071 | 4.7819 | -12.2845 | 0.9316 | 0.0043 |
| 36 | joint_slot_basis_only:equals | 672 | 330 | 0.0334 | 0.0149 | 3.5889 | -7.7840 | 0.9879 | 0.0030 |
| 37 | joint_orth_slot:value | 672 | 358 | 0.0487 | 0.1255 | 4.3826 | -10.4644 | 0.9916 | 0.0000 |
| 38 | joint_orth_phrase:value | 672 | 358 | 0.0500 | 0.1203 | 4.3813 | -10.4697 | 0.9916 | 0.0000 |
| 39 | joint_raw:value | 672 | 358 | 0.0478 | 0.1202 | 4.3835 | -10.4698 | 0.9888 | 0.0000 |
| 40 | joint_orth_relation:value | 672 | 358 | 0.0535 | 0.1196 | 4.3778 | -10.4704 | 0.9888 | 0.0000 |
| 41 | joint_orth_phrase_slot:value | 672 | 358 | 0.0448 | 0.1188 | 4.3865 | -10.4712 | 0.9916 | 0.0000 |
| 42 | joint_orth_all:value | 672 | 358 | 0.0392 | 0.1122 | 4.3921 | -10.4778 | 0.9916 | 0.0000 |
| 43 | joint_phrase_basis_only:value | 672 | 358 | 0.0072 | 0.0094 | 4.4242 | -10.5805 | 1.0000 | 0.0000 |
| 44 | joint_raw_restore_both:value | 672 | 358 | -0.0100 | 0.0021 | 4.4413 | -10.5879 | 0.9972 | 0.0000 |
| 45 | joint_slot_basis_only:value | 672 | 358 | 0.0061 | 0.0017 | 4.4252 | -10.5883 | 1.0000 | 0.0000 |
| 46 | joint_relation_basis_only:value | 672 | 358 | -0.0060 | -0.0009 | 4.4373 | -10.5909 | 0.9972 | 0.0000 |
| 47 | joint_relation_basis_only:answer | 672 | 468 | -0.0031 | -0.0050 | 5.0179 | -12.4965 | 0.9957 | 0.0000 |
| 48 | joint_phrase_basis_only:answer | 672 | 468 | -0.0106 | -0.0118 | 5.0254 | -12.5034 | 0.9979 | 0.0000 |
| 49 | joint_slot_basis_only:answer | 672 | 468 | -0.0059 | -0.0228 | 5.0207 | -12.5144 | 0.9936 | 0.0000 |
| 50 | joint_slot_basis_only:arrow | 672 | 344 | -0.0343 | -0.0890 | 3.3293 | -8.2644 | 0.9855 | 0.0000 |
| 51 | joint_relation_basis_only:arrow | 672 | 344 | -0.0274 | -0.1291 | 3.3223 | -8.3045 | 0.9709 | 0.0000 |
| 52 | joint_phrase_basis_only:arrow | 672 | 344 | -0.0349 | -0.1395 | 3.3298 | -8.3150 | 0.9709 | 0.0000 |

## deepseek7b

items=1344, basis_items=448, rows=34944, layer_pairs=[[8, 10], [12, 14]]
module=resid_out, contrast_rank=64, nuisance_rank=24
relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for'], phrase_ids=[0, 1, 2, 3], slot_ids=['answer', 'arrow', 'equals', 'value']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_phrase_frame | 2688 | 1190 | 3.6715 | 9.9851 | 0.2742 | 0.2742 | 0.6143 | 0.6143 |
| 2 | joint_same_relation_other_slot_frame | 2688 | 1190 | 3.6672 | 9.9894 | 0.2785 | 0.2785 | 0.6050 | 0.6050 |
| 3 | joint_orth_phrase_slot | 2688 | 1190 | 3.6302 | 3.6461 | 0.3155 | -6.0648 | 0.6168 | 0.1555 |
| 4 | joint_orth_all | 2688 | 1190 | 3.5996 | 3.6015 | 0.3461 | -6.1094 | 0.6193 | 0.1538 |
| 5 | joint_orth_slot | 2688 | 1190 | 3.6826 | 3.6926 | 0.2631 | -6.0183 | 0.6202 | 0.1529 |
| 6 | joint_raw | 2688 | 1190 | 3.7176 | 3.7061 | 0.2281 | -6.0048 | 0.6160 | 0.1521 |
| 7 | joint_orth_phrase | 2688 | 1190 | 3.6352 | 3.6745 | 0.3105 | -6.0364 | 0.6218 | 0.1513 |
| 8 | joint_orth_relation | 2688 | 1190 | 3.6099 | 3.6442 | 0.3358 | -6.0667 | 0.6168 | 0.1504 |
| 9 | joint_same_object_other_relation_frame | 2688 | 1190 | 3.7010 | 2.1512 | 0.2446 | -7.5597 | 0.6218 | 0.0462 |
| 10 | joint_raw_restore_both | 2688 | 1190 | 0.2191 | 0.1983 | 3.7266 | -9.5127 | 0.9311 | 0.0109 |
| 11 | joint_phrase_basis_only | 2688 | 1190 | 0.0566 | 0.0227 | 3.8891 | -9.6882 | 0.9723 | 0.0050 |
| 12 | joint_slot_basis_only | 2688 | 1190 | 0.0542 | -0.0028 | 3.8915 | -9.7137 | 0.9723 | 0.0050 |
| 13 | joint_relation_basis_only | 2688 | 1190 | 0.0893 | 0.0577 | 3.8563 | -9.6532 | 0.9706 | 0.0034 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_phrase_frame:L12->L14 | 1344 | 595 | 3.5014 | 10.1551 | 0.4442 | 0.4442 | 0.6185 | 0.6185 |
| 2 | joint_same_relation_other_slot_frame:L12->L14 | 1344 | 595 | 3.4431 | 10.2135 | 0.5026 | 0.5026 | 0.6118 | 0.6118 |
| 3 | joint_same_relation_other_phrase_frame:L8->L10 | 1344 | 595 | 3.8416 | 9.8150 | 0.1041 | 0.1041 | 0.6101 | 0.6101 |
| 4 | joint_same_relation_other_slot_frame:L8->L10 | 1344 | 595 | 3.8913 | 9.7653 | 0.0543 | 0.0543 | 0.5983 | 0.5983 |
| 5 | joint_raw:L8->L10 | 1344 | 595 | 3.8653 | 3.6894 | 0.0804 | -6.0215 | 0.6084 | 0.1613 |
| 6 | joint_orth_slot:L8->L10 | 1344 | 595 | 3.8226 | 3.6651 | 0.1230 | -6.0458 | 0.6218 | 0.1597 |
| 7 | joint_orth_all:L8->L10 | 1344 | 595 | 3.7788 | 3.5904 | 0.1669 | -6.1206 | 0.6202 | 0.1597 |
| 8 | joint_orth_phrase_slot:L8->L10 | 1344 | 595 | 3.7985 | 3.6215 | 0.1472 | -6.0894 | 0.6168 | 0.1580 |
| 9 | joint_orth_relation:L8->L10 | 1344 | 595 | 3.7927 | 3.6468 | 0.1529 | -6.0642 | 0.6168 | 0.1563 |
| 10 | joint_orth_phrase_slot:L12->L14 | 1344 | 595 | 3.4618 | 3.6707 | 0.4839 | -6.0403 | 0.6168 | 0.1529 |
| 11 | joint_orth_phrase:L8->L10 | 1344 | 595 | 3.7904 | 3.6500 | 0.1553 | -6.0610 | 0.6252 | 0.1529 |
| 12 | joint_orth_phrase:L12->L14 | 1344 | 595 | 3.4800 | 3.6990 | 0.4656 | -6.0119 | 0.6185 | 0.1496 |
| 13 | joint_orth_all:L12->L14 | 1344 | 595 | 3.4204 | 3.6126 | 0.5252 | -6.0983 | 0.6185 | 0.1479 |
| 14 | joint_orth_slot:L12->L14 | 1344 | 595 | 3.5425 | 3.7202 | 0.4032 | -5.9908 | 0.6185 | 0.1462 |
| 15 | joint_orth_relation:L12->L14 | 1344 | 595 | 3.4270 | 3.6417 | 0.5187 | -6.0692 | 0.6168 | 0.1445 |
| 16 | joint_raw:L12->L14 | 1344 | 595 | 3.5699 | 3.7228 | 0.3757 | -5.9882 | 0.6235 | 0.1429 |
| 17 | joint_same_object_other_relation_frame:L12->L14 | 1344 | 595 | 3.5309 | 2.2408 | 0.4147 | -7.4701 | 0.6303 | 0.0471 |
| 18 | joint_same_object_other_relation_frame:L8->L10 | 1344 | 595 | 3.8712 | 2.0615 | 0.0745 | -7.6494 | 0.6134 | 0.0454 |
| 19 | joint_raw_restore_both:L8->L10 | 1344 | 595 | 0.2627 | 0.2042 | 3.6829 | -9.5067 | 0.9160 | 0.0134 |
| 20 | joint_raw_restore_both:L12->L14 | 1344 | 595 | 0.1754 | 0.1923 | 3.7703 | -9.5186 | 0.9462 | 0.0084 |
| 21 | joint_slot_basis_only:L8->L10 | 1344 | 595 | 0.0490 | 0.0103 | 3.8966 | -9.7006 | 0.9681 | 0.0084 |
| 22 | joint_phrase_basis_only:L12->L14 | 1344 | 595 | 0.0726 | 0.0319 | 3.8731 | -9.6790 | 0.9748 | 0.0050 |
| 23 | joint_phrase_basis_only:L8->L10 | 1344 | 595 | 0.0407 | 0.0135 | 3.9050 | -9.6975 | 0.9697 | 0.0050 |
| 24 | joint_relation_basis_only:L12->L14 | 1344 | 595 | 0.1057 | 0.0845 | 3.8400 | -9.6264 | 0.9731 | 0.0034 |
| 25 | joint_relation_basis_only:L8->L10 | 1344 | 595 | 0.0730 | 0.0309 | 3.8727 | -9.6800 | 0.9681 | 0.0034 |
| 26 | joint_slot_basis_only:L12->L14 | 1344 | 595 | 0.0593 | -0.0158 | 3.8864 | -9.7268 | 0.9765 | 0.0017 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_slot_frame:material | 384 | 194 | 2.8867 | 10.2808 | 0.8338 | 0.8338 | 0.6701 | 0.6701 |
| 2 | joint_same_relation_other_phrase_frame:material | 384 | 194 | 2.5948 | 10.5727 | 1.1257 | 1.1257 | 0.6649 | 0.6649 |
| 3 | joint_same_relation_other_phrase_frame:can_do | 384 | 128 | 3.3832 | 8.0767 | -0.0705 | -0.0705 | 0.6484 | 0.6484 |
| 4 | joint_same_relation_other_phrase_frame:is_a | 384 | 260 | 3.6398 | 11.5727 | 1.1210 | 1.1210 | 0.6462 | 0.6462 |
| 5 | joint_same_relation_other_slot_frame:location | 384 | 238 | 3.1924 | 11.5555 | 1.0876 | 1.0876 | 0.6429 | 0.6429 |
| 6 | joint_same_relation_other_slot_frame:is_a | 384 | 260 | 3.4539 | 11.7585 | 1.3069 | 1.3069 | 0.6385 | 0.6385 |
| 7 | joint_same_relation_other_phrase_frame:location | 384 | 238 | 3.1885 | 11.5594 | 1.0915 | 1.0915 | 0.6345 | 0.6345 |
| 8 | joint_same_relation_other_slot_frame:can_do | 384 | 128 | 3.4219 | 8.0380 | -0.1091 | -0.1091 | 0.6172 | 0.6172 |
| 9 | joint_same_relation_other_phrase_frame:used_for | 384 | 146 | 3.6726 | 10.7564 | -0.0563 | -0.0563 | 0.6027 | 0.6027 |
| 10 | joint_same_relation_other_slot_frame:used_for | 384 | 146 | 3.6952 | 10.7338 | -0.0789 | -0.0789 | 0.5685 | 0.5685 |
| 11 | joint_same_relation_other_slot_frame:part_of | 384 | 136 | 5.0710 | 6.1440 | -1.2232 | -1.2232 | 0.5368 | 0.5368 |
| 12 | joint_same_relation_other_phrase_frame:part_of | 384 | 136 | 5.3534 | 5.8615 | -1.5057 | -1.5057 | 0.5221 | 0.5221 |
| 13 | joint_same_relation_other_phrase_frame:property | 384 | 88 | 5.2632 | 7.6102 | -2.5156 | -2.5156 | 0.4659 | 0.4659 |
| 14 | joint_same_relation_other_slot_frame:property | 384 | 88 | 5.4427 | 7.4307 | -2.6951 | -2.6951 | 0.4091 | 0.4091 |
| 15 | joint_orth_phrase:part_of | 384 | 136 | 5.3555 | 5.2789 | -1.5078 | -2.0883 | 0.4926 | 0.3162 |
| 16 | joint_raw:part_of | 384 | 136 | 5.4866 | 5.3125 | -1.6389 | -2.0548 | 0.5000 | 0.3088 |
| 17 | joint_orth_phrase_slot:part_of | 384 | 136 | 5.3623 | 5.2627 | -1.5146 | -2.1045 | 0.4779 | 0.3088 |
| 18 | joint_orth_all:part_of | 384 | 136 | 5.3943 | 5.1945 | -1.5466 | -2.1727 | 0.4926 | 0.3088 |
| 19 | joint_orth_relation:part_of | 384 | 136 | 5.3491 | 5.2196 | -1.5014 | -2.1477 | 0.4926 | 0.3015 |
| 20 | joint_orth_slot:part_of | 384 | 136 | 5.4414 | 5.2410 | -1.5937 | -2.1263 | 0.5074 | 0.2941 |
| 21 | joint_raw:material | 384 | 194 | 2.6747 | 2.8832 | 1.0458 | -6.5638 | 0.6649 | 0.1907 |
| 22 | joint_orth_slot:material | 384 | 194 | 2.6281 | 2.8243 | 1.0924 | -6.6227 | 0.6701 | 0.1907 |
| 23 | joint_orth_phrase_slot:material | 384 | 194 | 2.5555 | 2.7715 | 1.1650 | -6.6755 | 0.6701 | 0.1856 |
| 24 | joint_orth_all:material | 384 | 194 | 2.5693 | 2.8018 | 1.1512 | -6.6452 | 0.6701 | 0.1804 |
| 25 | joint_orth_relation:material | 384 | 194 | 2.6107 | 2.8055 | 1.1099 | -6.6415 | 0.6753 | 0.1753 |
| 26 | joint_orth_phrase:material | 384 | 194 | 2.5765 | 2.7728 | 1.1440 | -6.6742 | 0.6856 | 0.1753 |
| 27 | joint_raw:can_do | 384 | 128 | 3.2412 | 2.6569 | 0.0715 | -5.4903 | 0.6406 | 0.1484 |
| 28 | joint_orth_slot:is_a | 384 | 260 | 3.6257 | 4.0761 | 1.1352 | -6.3755 | 0.6538 | 0.1423 |
| 29 | joint_orth_relation:is_a | 384 | 260 | 3.5625 | 4.0543 | 1.1983 | -6.3974 | 0.6500 | 0.1423 |
| 30 | joint_orth_phrase_slot:is_a | 384 | 260 | 3.6158 | 4.0037 | 1.1451 | -6.4479 | 0.6577 | 0.1423 |
| 31 | joint_orth_all:is_a | 384 | 260 | 3.5347 | 3.9487 | 1.2261 | -6.5030 | 0.6577 | 0.1423 |
| 32 | joint_orth_phrase_slot:can_do | 384 | 128 | 3.2902 | 2.6756 | 0.0225 | -5.4715 | 0.6328 | 0.1406 |
| 33 | joint_orth_slot:can_do | 384 | 128 | 3.2994 | 2.6717 | 0.0133 | -5.4754 | 0.6562 | 0.1406 |
| 34 | joint_orth_phrase:can_do | 384 | 128 | 3.2102 | 2.6481 | 0.1025 | -5.4990 | 0.6328 | 0.1406 |
| 35 | joint_orth_all:can_do | 384 | 128 | 3.2666 | 2.6310 | 0.0461 | -5.5161 | 0.6406 | 0.1406 |
| 36 | joint_orth_relation:can_do | 384 | 128 | 3.2511 | 2.6172 | 0.0616 | -5.5300 | 0.6250 | 0.1406 |
| 37 | joint_orth_slot:location | 384 | 238 | 3.1418 | 3.7913 | 1.1382 | -6.6766 | 0.6471 | 0.1345 |
| 38 | joint_orth_phrase_slot:location | 384 | 238 | 3.1308 | 3.7145 | 1.1492 | -6.7534 | 0.6513 | 0.1345 |
| 39 | joint_orth_phrase:is_a | 384 | 260 | 3.6314 | 4.0991 | 1.1295 | -6.3525 | 0.6654 | 0.1308 |
| 40 | joint_raw:is_a | 384 | 260 | 3.6854 | 4.0868 | 1.0754 | -6.3648 | 0.6654 | 0.1308 |
| 41 | joint_raw:location | 384 | 238 | 3.1697 | 3.7849 | 1.1103 | -6.6830 | 0.6471 | 0.1303 |
| 42 | joint_orth_phrase:location | 384 | 238 | 3.1512 | 3.7652 | 1.1288 | -6.7027 | 0.6555 | 0.1303 |
| 43 | joint_orth_relation:location | 384 | 238 | 3.1194 | 3.7174 | 1.1606 | -6.7505 | 0.6513 | 0.1303 |
| 44 | joint_orth_all:location | 384 | 238 | 3.0908 | 3.6638 | 1.1893 | -6.8040 | 0.6597 | 0.1303 |
| 45 | joint_same_object_other_relation_frame:part_of | 384 | 136 | 5.4268 | 2.2313 | -1.5791 | -5.1359 | 0.5147 | 0.1250 |
| 46 | joint_orth_all:property | 384 | 88 | 5.1881 | 4.4329 | -2.4405 | -5.6930 | 0.4659 | 0.1136 |
| 47 | joint_orth_phrase_slot:property | 384 | 88 | 5.2718 | 4.4928 | -2.5242 | -5.6331 | 0.4773 | 0.1023 |
| 48 | joint_orth_phrase:property | 384 | 88 | 5.3195 | 4.4354 | -2.5719 | -5.6905 | 0.4545 | 0.1023 |
| 49 | joint_orth_relation:property | 384 | 88 | 5.2283 | 4.5296 | -2.4806 | -5.5963 | 0.4545 | 0.0909 |
| 50 | joint_orth_slot:property | 384 | 88 | 5.4265 | 4.4383 | -2.6789 | -5.6876 | 0.4545 | 0.0909 |
| 51 | joint_raw:used_for | 384 | 146 | 3.7215 | 3.0170 | -0.1052 | -7.7957 | 0.6027 | 0.0822 |
| 52 | joint_orth_phrase:used_for | 384 | 146 | 3.5927 | 2.9153 | 0.0236 | -7.8973 | 0.6164 | 0.0753 |
| 53 | joint_orth_phrase_slot:used_for | 384 | 146 | 3.5928 | 2.8945 | 0.0235 | -7.9182 | 0.6164 | 0.0753 |
| 54 | joint_orth_slot:used_for | 384 | 146 | 3.7128 | 3.0061 | -0.0965 | -7.8065 | 0.6233 | 0.0685 |
| 55 | joint_orth_all:used_for | 384 | 146 | 3.5762 | 2.8098 | 0.0401 | -8.0028 | 0.6096 | 0.0685 |
| 56 | joint_orth_relation:used_for | 384 | 146 | 3.5403 | 2.8086 | 0.0760 | -8.0041 | 0.6301 | 0.0685 |
| 57 | joint_raw:property | 384 | 88 | 5.5461 | 4.3694 | -2.7984 | -5.7565 | 0.4432 | 0.0682 |
| 58 | joint_same_object_other_relation_frame:property | 384 | 88 | 5.3573 | 3.0585 | -2.6097 | -7.0674 | 0.4659 | 0.0455 |
| 59 | joint_same_object_other_relation_frame:is_a | 384 | 260 | 3.6161 | 2.8647 | 1.1447 | -7.5869 | 0.6692 | 0.0423 |
| 60 | joint_same_object_other_relation_frame:location | 384 | 238 | 3.2683 | 2.0925 | 1.0118 | -8.3754 | 0.6597 | 0.0378 |
| 61 | joint_same_object_other_relation_frame:can_do | 384 | 128 | 3.3956 | 1.1251 | -0.0829 | -7.0221 | 0.6562 | 0.0312 |
| 62 | joint_same_object_other_relation_frame:material | 384 | 194 | 2.5451 | 1.3394 | 1.1755 | -8.1076 | 0.6649 | 0.0309 |
| 63 | joint_raw_restore_both:part_of | 384 | 136 | 0.2381 | 0.1640 | 3.6096 | -7.2033 | 0.9191 | 0.0294 |
| 64 | joint_same_object_other_relation_frame:used_for | 384 | 146 | 3.7558 | 2.3330 | -0.1395 | -8.4797 | 0.5822 | 0.0274 |
| 65 | joint_raw_restore_both:material | 384 | 194 | 0.3825 | 0.3064 | 3.3380 | -9.1406 | 0.9175 | 0.0258 |
| 66 | joint_phrase_basis_only:material | 384 | 194 | 0.0525 | 0.0467 | 3.6680 | -9.4003 | 0.9742 | 0.0206 |
| 67 | joint_slot_basis_only:material | 384 | 194 | 0.1160 | 0.0890 | 3.6046 | -9.3580 | 0.9639 | 0.0155 |
| 68 | joint_relation_basis_only:material | 384 | 194 | 0.0170 | 0.0186 | 3.7036 | -9.4284 | 0.9639 | 0.0155 |
| 69 | joint_raw_restore_both:location | 384 | 238 | 0.2978 | 0.3510 | 3.9823 | -10.1169 | 0.9076 | 0.0084 |
| 70 | joint_slot_basis_only:location | 384 | 238 | 0.0168 | 0.0370 | 4.2632 | -10.4309 | 0.9664 | 0.0084 |
| 71 | joint_raw_restore_both:can_do | 384 | 128 | 0.2530 | 0.1142 | 3.0597 | -8.0330 | 0.9297 | 0.0078 |
| 72 | joint_phrase_basis_only:part_of | 384 | 136 | 0.0544 | 0.0282 | 3.7934 | -7.3391 | 0.9338 | 0.0074 |
| 73 | joint_slot_basis_only:part_of | 384 | 136 | 0.0296 | 0.0061 | 3.8181 | -7.3611 | 0.9485 | 0.0074 |
| 74 | joint_phrase_basis_only:location | 384 | 238 | 0.0140 | 0.0760 | 4.2660 | -10.3919 | 0.9622 | 0.0042 |
| 75 | joint_relation_basis_only:location | 384 | 238 | 0.0344 | 0.0670 | 4.2456 | -10.4009 | 0.9748 | 0.0042 |
| 76 | joint_raw_restore_both:is_a | 384 | 260 | 0.1420 | 0.1781 | 4.6188 | -10.2735 | 0.9615 | 0.0038 |
| 77 | joint_relation_basis_only:property | 384 | 88 | 0.2578 | 0.1376 | 2.4898 | -9.9883 | 0.9205 | 0.0000 |
| 78 | joint_phrase_basis_only:property | 384 | 88 | 0.2362 | 0.1302 | 2.5114 | -9.9957 | 0.9318 | 0.0000 |
| 79 | joint_raw_restore_both:used_for | 384 | 146 | 0.0313 | 0.1299 | 3.5850 | -10.6827 | 0.9521 | 0.0000 |
| 80 | joint_relation_basis_only:part_of | 384 | 136 | 0.1156 | 0.0956 | 3.7321 | -7.2716 | 0.9412 | 0.0000 |
| 81 | joint_relation_basis_only:used_for | 384 | 146 | 0.1308 | 0.0730 | 3.4855 | -10.7396 | 0.9658 | 0.0000 |
| 82 | joint_relation_basis_only:can_do | 384 | 128 | 0.0606 | 0.0559 | 3.2521 | -8.0913 | 0.9922 | 0.0000 |
| 83 | joint_relation_basis_only:is_a | 384 | 260 | 0.1137 | 0.0239 | 4.6472 | -10.4277 | 0.9962 | 0.0000 |
| 84 | joint_slot_basis_only:is_a | 384 | 260 | 0.0502 | 0.0153 | 4.7106 | -10.4364 | 1.0000 | 0.0000 |
| 85 | joint_phrase_basis_only:is_a | 384 | 260 | 0.0738 | -0.0066 | 4.6871 | -10.4582 | 0.9962 | 0.0000 |
| 86 | joint_slot_basis_only:used_for | 384 | 146 | 0.0388 | -0.0315 | 3.5775 | -10.8441 | 0.9726 | 0.0000 |
| 87 | joint_phrase_basis_only:can_do | 384 | 128 | -0.0122 | -0.0341 | 3.3249 | -8.1812 | 1.0000 | 0.0000 |
| 88 | joint_phrase_basis_only:used_for | 384 | 146 | 0.0551 | -0.0640 | 3.5612 | -10.8767 | 0.9795 | 0.0000 |
| 89 | joint_slot_basis_only:can_do | 384 | 128 | -0.0274 | -0.0731 | 3.3401 | -8.2202 | 0.9844 | 0.0000 |
| 90 | joint_raw_restore_both:property | 384 | 88 | 0.1063 | -0.1051 | 2.6413 | -10.2309 | 0.9205 | 0.0000 |
| 91 | joint_slot_basis_only:property | 384 | 88 | 0.2127 | -0.2297 | 2.5349 | -10.3556 | 0.9432 | 0.0000 |

### Top condition slots

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_same_relation_other_phrase_frame:value | 672 | 334 | 0.2238 | 14.0657 | 3.6814 | 3.6814 | 0.9431 | 0.9431 |
| 2 | joint_same_relation_other_slot_frame:value | 672 | 334 | 0.1653 | 14.1241 | 3.7399 | 3.7399 | 0.9311 | 0.9311 |
| 3 | joint_same_relation_other_phrase_frame:equals | 672 | 252 | 0.3319 | 10.4366 | 2.9446 | 2.9446 | 0.9048 | 0.9048 |
| 4 | joint_same_relation_other_slot_frame:equals | 672 | 252 | 0.3415 | 10.4269 | 2.9350 | 2.9350 | 0.8889 | 0.8889 |
| 5 | joint_same_relation_other_slot_frame:arrow | 672 | 252 | 4.6590 | 6.8581 | -1.0766 | -1.0766 | 0.4008 | 0.4008 |
| 6 | joint_same_relation_other_phrase_frame:arrow | 672 | 252 | 4.7169 | 6.8001 | -1.1346 | -1.1346 | 0.3849 | 0.3849 |
| 7 | joint_orth_phrase_slot:answer | 672 | 352 | 8.5391 | 8.4694 | -3.8159 | -3.4627 | 0.2557 | 0.3097 |
| 8 | joint_orth_phrase:answer | 672 | 352 | 8.5639 | 8.5541 | -3.8407 | -3.3781 | 0.2472 | 0.3068 |
| 9 | joint_orth_slot:answer | 672 | 352 | 8.6475 | 8.5322 | -3.9242 | -3.4000 | 0.2585 | 0.3068 |
| 10 | joint_raw:answer | 672 | 352 | 8.7568 | 8.6637 | -4.0335 | -3.2685 | 0.2500 | 0.3040 |
| 11 | joint_orth_relation:answer | 672 | 352 | 8.5011 | 8.5306 | -3.7779 | -3.4016 | 0.2500 | 0.3040 |
| 12 | joint_orth_all:answer | 672 | 352 | 8.4996 | 8.4288 | -3.7764 | -3.5034 | 0.2585 | 0.3040 |
| 13 | joint_orth_phrase_slot:arrow | 672 | 252 | 4.5868 | 4.5566 | -1.0045 | -3.3781 | 0.4167 | 0.2738 |
| 14 | joint_orth_all:arrow | 672 | 252 | 4.5536 | 4.5264 | -0.9712 | -3.4084 | 0.4087 | 0.2738 |
| 15 | joint_orth_slot:arrow | 672 | 252 | 4.5814 | 4.5867 | -0.9990 | -3.3481 | 0.4087 | 0.2659 |
| 16 | joint_orth_phrase:arrow | 672 | 252 | 4.5488 | 4.5842 | -0.9664 | -3.3506 | 0.4365 | 0.2619 |
| 17 | joint_orth_relation:arrow | 672 | 252 | 4.5393 | 4.5340 | -0.9569 | -3.4008 | 0.4127 | 0.2619 |
| 18 | joint_same_relation_other_phrase_frame:answer | 672 | 352 | 8.5854 | 8.0701 | -3.8621 | -3.8621 | 0.2585 | 0.2585 |
| 19 | joint_raw:arrow | 672 | 252 | 4.5432 | 4.4983 | -0.9609 | -3.4365 | 0.4286 | 0.2579 |
| 20 | joint_same_relation_other_slot_frame:answer | 672 | 352 | 8.6609 | 7.9946 | -3.9376 | -3.9376 | 0.2386 | 0.2386 |
| 21 | joint_same_object_other_relation_frame:arrow | 672 | 252 | 4.6990 | 1.2602 | -1.1167 | -6.6745 | 0.4206 | 0.1230 |
| 22 | joint_same_object_other_relation_frame:equals | 672 | 252 | 0.3541 | 1.5238 | 2.9224 | -5.9682 | 0.8810 | 0.0317 |
| 23 | joint_raw:equals | 672 | 252 | 0.4109 | 0.4092 | 2.8656 | -7.0828 | 0.8849 | 0.0317 |
| 24 | joint_same_object_other_relation_frame:answer | 672 | 352 | 8.6186 | 2.0376 | -3.8954 | -9.8945 | 0.2699 | 0.0312 |
| 25 | joint_orth_slot:equals | 672 | 252 | 0.3532 | 0.3989 | 2.9233 | -7.0931 | 0.9048 | 0.0238 |
| 26 | joint_raw_restore_both:arrow | 672 | 252 | 0.3784 | 0.3878 | 3.2039 | -7.5469 | 0.9206 | 0.0238 |
| 27 | joint_orth_phrase_slot:equals | 672 | 252 | 0.2662 | 0.3505 | 3.0102 | -7.1414 | 0.9048 | 0.0238 |
| 28 | joint_orth_all:equals | 672 | 252 | 0.2578 | 0.2986 | 3.0187 | -7.1934 | 0.9087 | 0.0238 |
| 29 | joint_orth_relation:equals | 672 | 252 | 0.3228 | 0.3202 | 2.9536 | -7.1718 | 0.9127 | 0.0198 |
| 30 | joint_phrase_basis_only:equals | 672 | 252 | 0.1486 | -0.0140 | 3.1278 | -7.5060 | 0.9484 | 0.0198 |
| 31 | joint_orth_phrase:equals | 672 | 252 | 0.3001 | 0.3645 | 2.9763 | -7.1275 | 0.9127 | 0.0159 |
| 32 | joint_same_object_other_relation_frame:value | 672 | 334 | 0.2907 | 3.4165 | 3.6145 | -6.9678 | 0.9491 | 0.0150 |
| 33 | joint_relation_basis_only:equals | 672 | 252 | 0.1630 | 0.0942 | 3.1135 | -7.3978 | 0.9563 | 0.0119 |
| 34 | joint_raw_restore_both:equals | 672 | 252 | 0.1265 | 0.0868 | 3.1500 | -7.4051 | 0.9405 | 0.0119 |
| 35 | joint_slot_basis_only:equals | 672 | 252 | 0.0297 | -0.0265 | 3.2467 | -7.5185 | 0.9643 | 0.0119 |
| 36 | joint_raw_restore_both:answer | 672 | 352 | 0.2815 | 0.2707 | 4.4417 | -11.6615 | 0.9034 | 0.0114 |
| 37 | joint_slot_basis_only:arrow | 672 | 252 | 0.1439 | 0.0482 | 3.4384 | -7.8865 | 0.9603 | 0.0079 |
| 38 | joint_orth_phrase:value | 672 | 334 | 0.2678 | 0.3429 | 3.6374 | -10.0413 | 0.9371 | 0.0060 |
| 39 | joint_orth_slot:value | 672 | 334 | 0.2839 | 0.4029 | 3.6213 | -9.9814 | 0.9461 | 0.0030 |
| 40 | joint_raw:value | 672 | 334 | 0.2788 | 0.3711 | 3.6264 | -10.0131 | 0.9401 | 0.0030 |
| 41 | joint_orth_phrase_slot:value | 672 | 334 | 0.2729 | 0.3623 | 3.6323 | -10.0219 | 0.9311 | 0.0030 |
| 42 | joint_orth_relation:value | 672 | 334 | 0.2338 | 0.3312 | 3.6714 | -10.0531 | 0.9341 | 0.0030 |
| 43 | joint_orth_all:value | 672 | 334 | 0.2371 | 0.3082 | 3.6681 | -10.0760 | 0.9401 | 0.0030 |
| 44 | joint_phrase_basis_only:answer | 672 | 352 | 0.0077 | 0.0328 | 4.7156 | -11.8994 | 0.9801 | 0.0028 |
| 45 | joint_relation_basis_only:answer | 672 | 352 | 0.0491 | 0.0295 | 4.6742 | -11.9026 | 0.9659 | 0.0028 |
| 46 | joint_slot_basis_only:answer | 672 | 352 | 0.0356 | 0.0061 | 4.6877 | -11.9261 | 0.9801 | 0.0028 |
| 47 | joint_relation_basis_only:value | 672 | 334 | 0.0514 | 0.0885 | 3.8538 | -10.2957 | 0.9850 | 0.0000 |
| 48 | joint_raw_restore_both:value | 672 | 334 | 0.1028 | 0.0630 | 3.8024 | -10.3213 | 0.9611 | 0.0000 |
| 49 | joint_phrase_basis_only:arrow | 672 | 252 | 0.1216 | 0.0386 | 3.4607 | -7.8962 | 0.9683 | 0.0000 |
| 50 | joint_phrase_basis_only:value | 672 | 334 | -0.0103 | 0.0278 | 3.9155 | -10.3564 | 0.9850 | 0.0000 |
| 51 | joint_relation_basis_only:arrow | 672 | 252 | 0.1222 | 0.0198 | 3.4601 | -7.9149 | 0.9722 | 0.0000 |
| 52 | joint_slot_basis_only:value | 672 | 334 | 0.0245 | -0.0326 | 3.8807 | -10.4169 | 0.9790 | 0.0000 |

