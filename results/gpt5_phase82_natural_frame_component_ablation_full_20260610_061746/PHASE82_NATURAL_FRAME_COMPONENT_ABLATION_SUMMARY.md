# Phase82 Natural Frame Component Ablation Summary

## qwen3

items=672, basis_items=224, rows=18816, layer_pairs=[[4, 8], [8, 12]]
module=resid_out, contrast_rank=64, component_rank=24, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_raw | 1344 | 802 | 10.6943 | 13.8309 | -6.2332 | 0.8283 | 0.0723 | 0.5399 |
| 2 | joint_orth_pre_object | 1344 | 802 | 10.3507 | 13.4426 | -5.8896 | 0.4401 | 0.0923 | 0.5162 |
| 3 | joint_orth_full_frame | 1344 | 802 | 8.6386 | 9.7438 | -4.1774 | -3.2587 | 0.1721 | 0.2693 |
| 4 | joint_orth_relation_label | 1344 | 802 | 8.6822 | 9.7924 | -4.2211 | -3.2101 | 0.1646 | 0.2544 |
| 5 | joint_mismatched_frame_raw | 1344 | 802 | 9.0796 | 9.8269 | -4.6185 | -3.1756 | 0.1297 | 0.2369 |
| 6 | joint_suffix_basis_only | 1344 | 802 | 4.3540 | 8.1683 | 0.1072 | -4.8342 | 0.5449 | 0.1534 |
| 7 | joint_boundary_basis_only | 1344 | 802 | 4.0683 | 7.9751 | 0.3928 | -5.0275 | 0.5599 | 0.1446 |
| 8 | joint_orth_boundary | 1344 | 802 | 7.3376 | 5.9752 | -2.8765 | -7.0273 | 0.2656 | 0.0711 |
| 9 | joint_orth_suffix | 1344 | 802 | 7.2924 | 5.7915 | -2.8312 | -7.2110 | 0.2743 | 0.0711 |
| 10 | joint_orth_all_components | 1344 | 802 | 6.0748 | 5.1338 | -1.6137 | -7.8687 | 0.3616 | 0.0623 |
| 11 | joint_full_frame_basis_only | 1344 | 802 | 2.1489 | 3.3784 | 2.3122 | -9.6241 | 0.7419 | 0.0474 |
| 12 | joint_relation_label_basis_only | 1344 | 802 | 1.9985 | 3.1553 | 2.4626 | -9.8473 | 0.7631 | 0.0436 |
| 13 | joint_raw_restore_both | 1344 | 802 | 0.9450 | 1.0900 | 3.5161 | -11.9125 | 0.8653 | 0.0100 |
| 14 | joint_pre_object_basis_only | 1344 | 802 | 0.2870 | 0.2816 | 4.1742 | -12.7210 | 0.9264 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_raw:L8->L12 | 672 | 401 | 10.3902 | 14.4086 | -5.9291 | 1.4061 | 0.0823 | 0.5885 |
| 2 | joint_orth_pre_object:L8->L12 | 672 | 401 | 9.9737 | 13.8972 | -5.5125 | 0.8947 | 0.0973 | 0.5586 |
| 3 | joint_raw:L4->L8 | 672 | 401 | 10.9985 | 13.2531 | -6.5373 | 0.2505 | 0.0623 | 0.4913 |
| 4 | joint_orth_pre_object:L4->L8 | 672 | 401 | 10.7278 | 12.9880 | -6.2666 | -0.0145 | 0.0873 | 0.4738 |
| 5 | joint_orth_relation_label:L8->L12 | 672 | 401 | 8.7009 | 11.6052 | -4.2398 | -1.3973 | 0.1671 | 0.3716 |
| 6 | joint_orth_full_frame:L8->L12 | 672 | 401 | 8.3873 | 11.1582 | -3.9262 | -1.8443 | 0.1820 | 0.3566 |
| 7 | joint_mismatched_frame_raw:L8->L12 | 672 | 401 | 8.6620 | 10.5450 | -4.2009 | -2.4575 | 0.1471 | 0.2693 |
| 8 | joint_mismatched_frame_raw:L4->L8 | 672 | 401 | 9.4972 | 9.1088 | -5.0360 | -3.8937 | 0.1122 | 0.2045 |
| 9 | joint_orth_full_frame:L4->L8 | 672 | 401 | 8.8898 | 8.3295 | -4.4287 | -4.6731 | 0.1621 | 0.1820 |
| 10 | joint_boundary_basis_only:L8->L12 | 672 | 401 | 4.0298 | 8.3729 | 0.4314 | -4.6297 | 0.5711 | 0.1546 |
| 11 | joint_suffix_basis_only:L4->L8 | 672 | 401 | 4.6783 | 8.1638 | -0.2172 | -4.8388 | 0.5187 | 0.1546 |
| 12 | joint_suffix_basis_only:L8->L12 | 672 | 401 | 4.0297 | 8.1728 | 0.4315 | -4.8297 | 0.5711 | 0.1521 |
| 13 | joint_orth_relation_label:L4->L8 | 672 | 401 | 8.6635 | 7.9796 | -4.2024 | -5.0229 | 0.1621 | 0.1372 |
| 14 | joint_boundary_basis_only:L4->L8 | 672 | 401 | 4.1068 | 7.5773 | 0.3543 | -5.4253 | 0.5486 | 0.1347 |
| 15 | joint_orth_boundary:L4->L8 | 672 | 401 | 8.2743 | 6.2139 | -3.8132 | -6.7887 | 0.1895 | 0.0948 |
| 16 | joint_orth_suffix:L4->L8 | 672 | 401 | 7.5200 | 5.7199 | -3.0588 | -7.2827 | 0.2519 | 0.0823 |
| 17 | joint_relation_label_basis_only:L4->L8 | 672 | 401 | 2.9499 | 4.5619 | 1.5112 | -8.4406 | 0.6808 | 0.0698 |
| 18 | joint_orth_all_components:L4->L8 | 672 | 401 | 6.8207 | 5.2722 | -2.3596 | -7.7303 | 0.3117 | 0.0673 |
| 19 | joint_full_frame_basis_only:L4->L8 | 672 | 401 | 2.9559 | 4.5926 | 1.5052 | -8.4100 | 0.6534 | 0.0673 |
| 20 | joint_orth_suffix:L8->L12 | 672 | 401 | 7.0648 | 5.8632 | -2.6037 | -7.1393 | 0.2968 | 0.0599 |
| 21 | joint_orth_all_components:L8->L12 | 672 | 401 | 5.3290 | 4.9954 | -0.8679 | -8.0071 | 0.4115 | 0.0574 |
| 22 | joint_orth_boundary:L8->L12 | 672 | 401 | 6.4009 | 5.7366 | -1.9398 | -7.2660 | 0.3416 | 0.0474 |
| 23 | joint_full_frame_basis_only:L8->L12 | 672 | 401 | 1.3420 | 2.1643 | 3.1192 | -10.8382 | 0.8304 | 0.0274 |
| 24 | joint_relation_label_basis_only:L8->L12 | 672 | 401 | 1.0471 | 1.7486 | 3.4140 | -11.2540 | 0.8454 | 0.0175 |
| 25 | joint_raw_restore_both:L4->L8 | 672 | 401 | 1.1304 | 1.3319 | 3.3308 | -11.6706 | 0.8404 | 0.0150 |
| 26 | joint_raw_restore_both:L8->L12 | 672 | 401 | 0.7597 | 0.8481 | 3.7014 | -12.1544 | 0.8903 | 0.0050 |
| 27 | joint_pre_object_basis_only:L8->L12 | 672 | 401 | 0.3044 | 0.3627 | 4.1567 | -12.6399 | 0.9401 | 0.0000 |
| 28 | joint_pre_object_basis_only:L4->L8 | 672 | 401 | 0.2695 | 0.2004 | 4.1916 | -12.8021 | 0.9127 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_raw:can_do | 192 | 144 | 11.5155 | 16.9884 | -6.2345 | 1.9116 | 0.0625 | 0.6944 |
| 2 | joint_raw:location | 192 | 84 | 9.0817 | 11.8156 | -6.2764 | 1.4673 | 0.0714 | 0.6548 |
| 3 | joint_orth_pre_object:can_do | 192 | 144 | 11.0761 | 16.3585 | -5.7951 | 1.2817 | 0.1042 | 0.6319 |
| 4 | joint_orth_pre_object:property | 192 | 76 | 9.9649 | 9.8790 | -6.8263 | 1.4156 | 0.0263 | 0.6053 |
| 5 | joint_orth_pre_object:location | 192 | 84 | 8.7370 | 11.3570 | -5.9318 | 1.0087 | 0.1071 | 0.5952 |
| 6 | joint_raw:property | 192 | 76 | 9.9954 | 10.0639 | -6.8568 | 1.6005 | 0.0132 | 0.5921 |
| 7 | joint_orth_pre_object:is_a | 192 | 172 | 11.0871 | 14.5456 | -5.1864 | 1.1342 | 0.1105 | 0.5698 |
| 8 | joint_raw:is_a | 192 | 172 | 11.4908 | 14.9449 | -5.5901 | 1.5335 | 0.1105 | 0.5640 |
| 9 | joint_raw:used_for | 192 | 110 | 13.4656 | 15.6641 | -8.2786 | 0.7442 | 0.0273 | 0.4545 |
| 10 | joint_orth_pre_object:used_for | 192 | 110 | 13.1715 | 15.2273 | -7.9846 | 0.3074 | 0.0364 | 0.4545 |
| 11 | joint_orth_full_frame:can_do | 192 | 144 | 9.7542 | 13.7741 | -4.4732 | -1.3027 | 0.2014 | 0.4306 |
| 12 | joint_raw:part_of | 192 | 106 | 9.7092 | 12.7729 | -6.0846 | -1.0865 | 0.1038 | 0.4057 |
| 13 | joint_raw:material | 192 | 110 | 8.2666 | 11.2833 | -4.8711 | -0.7846 | 0.0818 | 0.3909 |
| 14 | joint_orth_pre_object:part_of | 192 | 106 | 9.2556 | 12.6360 | -5.6311 | -1.2233 | 0.1226 | 0.3868 |
| 15 | joint_orth_relation_label:can_do | 192 | 144 | 9.5167 | 13.6567 | -4.2357 | -1.4201 | 0.1806 | 0.3750 |
| 16 | joint_orth_full_frame:location | 192 | 84 | 7.4256 | 8.1207 | -4.6203 | -2.2276 | 0.1429 | 0.3571 |
| 17 | joint_orth_pre_object:material | 192 | 110 | 7.9830 | 10.9482 | -4.5874 | -1.1197 | 0.1091 | 0.3455 |
| 18 | joint_mismatched_frame_raw:can_do | 192 | 144 | 10.2411 | 13.7226 | -4.9601 | -1.3542 | 0.0972 | 0.3403 |
| 19 | joint_orth_relation_label:location | 192 | 84 | 7.5669 | 8.1352 | -4.7617 | -2.2132 | 0.1786 | 0.3214 |
| 20 | joint_orth_relation_label:property | 192 | 76 | 6.8880 | 6.7823 | -3.7494 | -1.6811 | 0.1316 | 0.3158 |
| 21 | joint_mismatched_frame_raw:property | 192 | 76 | 7.5172 | 5.6616 | -4.3786 | -2.8018 | 0.1711 | 0.3158 |
| 22 | joint_orth_full_frame:property | 192 | 76 | 6.8155 | 6.7803 | -3.6769 | -1.6831 | 0.1842 | 0.2895 |
| 23 | joint_orth_full_frame:is_a | 192 | 172 | 9.2388 | 10.0006 | -3.3382 | -3.4108 | 0.2209 | 0.2674 |
| 24 | joint_mismatched_frame_raw:is_a | 192 | 172 | 9.5139 | 10.9170 | -3.6133 | -2.4944 | 0.2151 | 0.2616 |
| 25 | joint_orth_relation_label:is_a | 192 | 172 | 9.3615 | 10.0972 | -3.4609 | -3.3141 | 0.2267 | 0.2500 |
| 26 | joint_orth_relation_label:material | 192 | 110 | 6.8414 | 8.9559 | -3.4458 | -3.1120 | 0.1636 | 0.2455 |
| 27 | joint_orth_full_frame:material | 192 | 110 | 6.6733 | 8.9254 | -3.2778 | -3.1425 | 0.2182 | 0.2455 |
| 28 | joint_mismatched_frame_raw:location | 192 | 84 | 7.0871 | 7.0878 | -4.2819 | -3.2606 | 0.1310 | 0.2262 |
| 29 | joint_suffix_basis_only:property | 192 | 76 | 4.0250 | 4.4126 | -0.8864 | -4.0508 | 0.3947 | 0.2237 |
| 30 | joint_boundary_basis_only:property | 192 | 76 | 3.8794 | 4.4103 | -0.7408 | -4.0531 | 0.3684 | 0.2237 |
| 31 | joint_mismatched_frame_raw:used_for | 192 | 110 | 12.1838 | 11.0316 | -6.9969 | -3.8883 | 0.0545 | 0.2091 |
| 32 | joint_suffix_basis_only:is_a | 192 | 172 | 5.1921 | 9.0988 | 0.7085 | -4.3126 | 0.5930 | 0.2035 |
| 33 | joint_boundary_basis_only:is_a | 192 | 172 | 4.7659 | 8.7737 | 1.1348 | -4.6377 | 0.6337 | 0.1860 |
| 34 | joint_suffix_basis_only:can_do | 192 | 144 | 5.3021 | 11.2178 | -0.0211 | -3.8590 | 0.5764 | 0.1806 |
| 35 | joint_orth_boundary:location | 192 | 84 | 6.2723 | 5.6238 | -3.4671 | -4.7245 | 0.2143 | 0.1786 |
| 36 | joint_orth_suffix:location | 192 | 84 | 6.3084 | 5.4342 | -3.5032 | -4.9142 | 0.2024 | 0.1786 |
| 37 | joint_boundary_basis_only:location | 192 | 84 | 2.3301 | 5.3255 | 0.4751 | -5.0228 | 0.5833 | 0.1786 |
| 38 | joint_orth_all_components:location | 192 | 84 | 5.5109 | 4.8417 | -2.7057 | -5.5066 | 0.2619 | 0.1786 |
| 39 | joint_suffix_basis_only:used_for | 192 | 110 | 6.1041 | 9.9775 | -0.9172 | -4.9424 | 0.4364 | 0.1727 |
| 40 | joint_suffix_basis_only:location | 192 | 84 | 2.4856 | 5.4884 | 0.3197 | -4.8599 | 0.6071 | 0.1667 |
| 41 | joint_boundary_basis_only:can_do | 192 | 144 | 4.8879 | 10.8435 | 0.3931 | -4.2333 | 0.5903 | 0.1528 |
| 42 | joint_orth_full_frame:used_for | 192 | 110 | 10.6042 | 10.9007 | -5.4173 | -4.0192 | 0.0727 | 0.1455 |
| 43 | joint_mismatched_frame_raw:part_of | 192 | 106 | 8.3409 | 9.2902 | -4.7164 | -4.5691 | 0.0849 | 0.1415 |
| 44 | joint_orth_relation_label:used_for | 192 | 110 | 10.5087 | 10.7068 | -5.3218 | -4.2131 | 0.0909 | 0.1364 |
| 45 | joint_mismatched_frame_raw:material | 192 | 110 | 7.0886 | 7.3047 | -3.6931 | -4.7631 | 0.1273 | 0.1364 |
| 46 | joint_orth_relation_label:part_of | 192 | 106 | 8.6314 | 7.4392 | -5.0069 | -6.4202 | 0.1321 | 0.1321 |
| 47 | joint_boundary_basis_only:used_for | 192 | 110 | 5.7577 | 9.7734 | -0.5707 | -5.1465 | 0.5000 | 0.1273 |
| 48 | joint_orth_full_frame:part_of | 192 | 106 | 8.4168 | 6.9120 | -4.7922 | -6.9473 | 0.1226 | 0.1226 |
| 49 | joint_orth_boundary:property | 192 | 76 | 5.7724 | 4.6771 | -2.6337 | -3.7863 | 0.2368 | 0.1184 |
| 50 | joint_orth_suffix:property | 192 | 76 | 5.8990 | 4.5377 | -2.7604 | -3.9257 | 0.2500 | 0.1184 |
| 51 | joint_orth_all_components:property | 192 | 76 | 5.3229 | 4.4116 | -2.1843 | -4.0518 | 0.2500 | 0.1053 |
| 52 | joint_boundary_basis_only:part_of | 192 | 106 | 2.4459 | 7.3954 | 1.1786 | -6.4640 | 0.6792 | 0.0943 |
| 53 | joint_orth_boundary:is_a | 192 | 172 | 8.1227 | 7.0536 | -2.2220 | -6.3578 | 0.3372 | 0.0930 |
| 54 | joint_full_frame_basis_only:is_a | 192 | 172 | 3.0986 | 4.7986 | 2.8020 | -8.6128 | 0.7616 | 0.0930 |
| 55 | joint_relation_label_basis_only:property | 192 | 76 | 2.9870 | 3.0531 | 0.1517 | -5.4103 | 0.5526 | 0.0921 |
| 56 | joint_suffix_basis_only:part_of | 192 | 106 | 2.5027 | 7.6650 | 1.1218 | -6.1944 | 0.6698 | 0.0849 |
| 57 | joint_orth_suffix:is_a | 192 | 172 | 8.0850 | 6.8684 | -2.1843 | -6.5430 | 0.3314 | 0.0814 |
| 58 | joint_full_frame_basis_only:property | 192 | 76 | 2.9913 | 2.9213 | 0.1473 | -5.5421 | 0.5132 | 0.0789 |
| 59 | joint_relation_label_basis_only:is_a | 192 | 172 | 2.9661 | 4.2586 | 2.9345 | -9.1528 | 0.7849 | 0.0756 |
| 60 | joint_orth_boundary:material | 192 | 110 | 5.7073 | 4.4991 | -2.3117 | -7.5688 | 0.3091 | 0.0727 |
| 61 | joint_orth_suffix:material | 192 | 110 | 5.6683 | 4.3372 | -2.2728 | -7.7307 | 0.3091 | 0.0727 |
| 62 | joint_full_frame_basis_only:location | 192 | 84 | 1.3341 | 2.3573 | 1.4711 | -7.9910 | 0.6905 | 0.0714 |
| 63 | joint_orth_all_components:is_a | 192 | 172 | 6.2817 | 6.1602 | -0.3811 | -7.2512 | 0.4651 | 0.0698 |
| 64 | joint_orth_all_components:material | 192 | 110 | 5.0764 | 3.5319 | -1.6808 | -8.5360 | 0.3727 | 0.0636 |
| 65 | joint_orth_boundary:part_of | 192 | 106 | 7.8871 | 4.9150 | -4.2625 | -8.9443 | 0.1415 | 0.0566 |
| 66 | joint_orth_suffix:part_of | 192 | 106 | 7.9822 | 4.7724 | -4.3576 | -9.0870 | 0.1698 | 0.0566 |
| 67 | joint_boundary_basis_only:material | 192 | 110 | 3.2364 | 6.2178 | 0.1592 | -5.8501 | 0.4636 | 0.0545 |
| 68 | joint_full_frame_basis_only:used_for | 192 | 110 | 3.2326 | 4.8157 | 1.9544 | -10.1042 | 0.6818 | 0.0545 |
| 69 | joint_raw_restore_both:property | 192 | 76 | 0.8941 | 1.4597 | 2.2445 | -7.0037 | 0.7895 | 0.0526 |
| 70 | joint_relation_label_basis_only:location | 192 | 84 | 1.3767 | 2.1113 | 1.4285 | -8.2370 | 0.7500 | 0.0476 |
| 71 | joint_relation_label_basis_only:used_for | 192 | 110 | 2.8117 | 4.2566 | 2.3752 | -10.6633 | 0.7364 | 0.0455 |
| 72 | joint_orth_all_components:part_of | 192 | 106 | 7.0691 | 4.3947 | -3.4445 | -9.4647 | 0.2264 | 0.0377 |
| 73 | joint_raw_restore_both:location | 192 | 84 | 0.3880 | 0.6731 | 2.4173 | -9.6752 | 0.7857 | 0.0357 |
| 74 | joint_relation_label_basis_only:part_of | 192 | 106 | 1.2827 | 2.7243 | 2.3418 | -11.1351 | 0.7925 | 0.0283 |
| 75 | joint_orth_suffix:used_for | 192 | 110 | 9.5484 | 8.5563 | -4.3615 | -6.3635 | 0.1636 | 0.0273 |
| 76 | joint_suffix_basis_only:material | 192 | 110 | 3.4901 | 6.0383 | -0.0946 | -6.0295 | 0.4727 | 0.0273 |
| 77 | joint_full_frame_basis_only:part_of | 192 | 106 | 1.6862 | 3.1970 | 1.9384 | -10.6624 | 0.7830 | 0.0189 |
| 78 | joint_orth_all_components:used_for | 192 | 110 | 8.4328 | 7.3014 | -3.2459 | -7.6185 | 0.2818 | 0.0182 |
| 79 | joint_full_frame_basis_only:material | 192 | 110 | 1.8245 | 2.4998 | 1.5711 | -9.5681 | 0.6909 | 0.0182 |
| 80 | joint_relation_label_basis_only:material | 192 | 110 | 1.7875 | 2.3746 | 1.6081 | -9.6932 | 0.6909 | 0.0182 |
| 81 | joint_orth_boundary:can_do | 192 | 144 | 6.7665 | 5.6985 | -1.4855 | -9.3783 | 0.3750 | 0.0139 |
| 82 | joint_orth_suffix:can_do | 192 | 144 | 6.6645 | 5.1246 | -1.3835 | -9.9522 | 0.3958 | 0.0139 |
| 83 | joint_orth_all_components:can_do | 192 | 144 | 4.7832 | 4.5715 | 0.4978 | -10.5053 | 0.5069 | 0.0139 |
| 84 | joint_raw_restore_both:part_of | 192 | 106 | 1.0839 | 1.2377 | 2.5407 | -12.6217 | 0.7925 | 0.0094 |
| 85 | joint_orth_boundary:used_for | 192 | 110 | 9.8535 | 8.3143 | -4.6666 | -6.6056 | 0.1455 | 0.0091 |
| 86 | joint_relation_label_basis_only:can_do | 192 | 144 | 0.7507 | 2.5725 | 4.5303 | -12.5043 | 0.9097 | 0.0069 |
| 87 | joint_full_frame_basis_only:can_do | 192 | 144 | 0.8060 | 2.2259 | 4.4750 | -12.8509 | 0.9236 | 0.0000 |
| 88 | joint_raw_restore_both:can_do | 192 | 144 | 1.9665 | 1.4962 | 3.3145 | -13.5806 | 0.8819 | 0.0000 |
| 89 | joint_raw_restore_both:used_for | 192 | 110 | 0.8907 | 1.2608 | 4.2962 | -13.6591 | 0.8455 | 0.0000 |
| 90 | joint_raw_restore_both:is_a | 192 | 172 | 0.5921 | 0.7862 | 5.3086 | -12.6251 | 0.9651 | 0.0000 |
| 91 | joint_raw_restore_both:material | 192 | 110 | 0.5408 | 0.7831 | 2.8547 | -11.2848 | 0.8909 | 0.0000 |
| 92 | joint_pre_object_basis_only:used_for | 192 | 110 | 0.1339 | 0.4479 | 5.0530 | -14.4720 | 0.9455 | 0.0000 |
| 93 | joint_pre_object_basis_only:property | 192 | 76 | 0.5101 | 0.3918 | 2.6286 | -8.0716 | 0.9342 | 0.0000 |
| 94 | joint_pre_object_basis_only:material | 192 | 110 | 0.1778 | 0.3052 | 3.2177 | -11.7626 | 0.9091 | 0.0000 |
| 95 | joint_pre_object_basis_only:is_a | 192 | 172 | 0.2080 | 0.2798 | 5.6926 | -13.1316 | 0.9767 | 0.0000 |
| 96 | joint_pre_object_basis_only:location | 192 | 84 | 0.3727 | 0.2411 | 2.4325 | -10.1072 | 0.8690 | 0.0000 |
| 97 | joint_pre_object_basis_only:can_do | 192 | 144 | 0.4304 | 0.1905 | 4.8506 | -14.8863 | 0.9306 | 0.0000 |
| 98 | joint_pre_object_basis_only:part_of | 192 | 106 | 0.2645 | 0.1640 | 3.3601 | -13.6954 | 0.8774 | 0.0000 |

## glm4

items=672, basis_items=224, rows=18816, layer_pairs=[[4, 10], [10, 20]]
module=resid_out, contrast_rank=64, component_rank=24, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_raw | 1344 | 938 | 9.7988 | 13.8849 | -5.3973 | 1.9582 | 0.1247 | 0.6546 |
| 2 | joint_orth_pre_object | 1344 | 938 | 9.4681 | 13.5154 | -5.0667 | 1.5887 | 0.1429 | 0.6365 |
| 3 | joint_orth_relation_label | 1344 | 938 | 8.2273 | 10.9777 | -3.8258 | -0.9490 | 0.2239 | 0.4392 |
| 4 | joint_orth_full_frame | 1344 | 938 | 7.5246 | 9.6759 | -3.1231 | -2.2508 | 0.2687 | 0.3337 |
| 5 | joint_mismatched_frame_raw | 1344 | 938 | 8.0050 | 9.2854 | -3.6036 | -2.6413 | 0.2026 | 0.2751 |
| 6 | joint_suffix_basis_only | 1344 | 938 | 2.7950 | 6.9180 | 1.6065 | -5.0087 | 0.6834 | 0.1461 |
| 7 | joint_boundary_basis_only | 1344 | 938 | 2.3943 | 6.1727 | 2.0072 | -5.7540 | 0.7335 | 0.1194 |
| 8 | joint_orth_boundary | 1344 | 938 | 6.1153 | 6.1659 | -1.7139 | -5.7608 | 0.3753 | 0.1162 |
| 9 | joint_orth_suffix | 1344 | 938 | 6.4254 | 5.9859 | -2.0240 | -5.9408 | 0.3529 | 0.1013 |
| 10 | joint_orth_all_components | 1344 | 938 | 5.9183 | 5.4180 | -1.5169 | -6.5087 | 0.3987 | 0.0981 |
| 11 | joint_full_frame_basis_only | 1344 | 938 | 0.7778 | 1.9469 | 3.6237 | -9.9798 | 0.8977 | 0.0245 |
| 12 | joint_raw_restore_both | 1344 | 938 | 1.0068 | 1.5649 | 3.3946 | -10.3618 | 0.8838 | 0.0171 |
| 13 | joint_relation_label_basis_only | 1344 | 938 | 0.4502 | 1.1602 | 3.9512 | -10.7665 | 0.9190 | 0.0149 |
| 14 | joint_pre_object_basis_only | 1344 | 938 | -0.0207 | 0.0480 | 4.4222 | -11.8787 | 0.9808 | 0.0011 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_raw:L4->L10 | 672 | 469 | 10.9780 | 14.1923 | -6.5766 | 2.2656 | 0.0576 | 0.6823 |
| 2 | joint_orth_pre_object:L4->L10 | 672 | 469 | 10.7595 | 13.9242 | -6.3581 | 1.9975 | 0.0768 | 0.6588 |
| 3 | joint_raw:L10->L20 | 672 | 469 | 8.6195 | 13.5774 | -4.2181 | 1.6507 | 0.1919 | 0.6269 |
| 4 | joint_orth_pre_object:L10->L20 | 672 | 469 | 8.1767 | 13.1066 | -3.7752 | 1.1799 | 0.2090 | 0.6141 |
| 5 | joint_orth_relation_label:L10->L20 | 672 | 469 | 6.7642 | 10.9388 | -2.3628 | -0.9879 | 0.3134 | 0.4456 |
| 6 | joint_orth_relation_label:L4->L10 | 672 | 469 | 9.6903 | 11.0165 | -5.2889 | -0.9102 | 0.1343 | 0.4328 |
| 7 | joint_orth_full_frame:L10->L20 | 672 | 469 | 6.1868 | 9.8413 | -1.7854 | -2.0854 | 0.3710 | 0.3625 |
| 8 | joint_orth_full_frame:L4->L10 | 672 | 469 | 8.8623 | 9.5105 | -4.4609 | -2.4162 | 0.1663 | 0.3049 |
| 9 | joint_mismatched_frame_raw:L10->L20 | 672 | 469 | 6.7634 | 9.6479 | -2.3620 | -2.2788 | 0.3028 | 0.3028 |
| 10 | joint_mismatched_frame_raw:L4->L10 | 672 | 469 | 9.2466 | 8.9229 | -4.8452 | -3.0037 | 0.1023 | 0.2473 |
| 11 | joint_suffix_basis_only:L10->L20 | 672 | 469 | 2.7809 | 6.9478 | 1.6205 | -4.9789 | 0.6866 | 0.1599 |
| 12 | joint_suffix_basis_only:L4->L10 | 672 | 469 | 2.8090 | 6.8882 | 1.5924 | -5.0385 | 0.6802 | 0.1322 |
| 13 | joint_orth_boundary:L4->L10 | 672 | 469 | 7.9446 | 6.7730 | -3.5431 | -5.1536 | 0.2175 | 0.1258 |
| 14 | joint_orth_all_components:L4->L10 | 672 | 469 | 7.9153 | 6.4783 | -3.5139 | -5.4484 | 0.2388 | 0.1258 |
| 15 | joint_boundary_basis_only:L10->L20 | 672 | 469 | 2.2494 | 5.8228 | 2.1520 | -6.1039 | 0.7527 | 0.1215 |
| 16 | joint_boundary_basis_only:L4->L10 | 672 | 469 | 2.5391 | 6.5227 | 1.8623 | -5.4040 | 0.7143 | 0.1173 |
| 17 | joint_orth_suffix:L4->L10 | 672 | 469 | 7.9995 | 6.6961 | -3.5980 | -5.2306 | 0.2239 | 0.1109 |
| 18 | joint_orth_boundary:L10->L20 | 672 | 469 | 4.2860 | 5.5588 | 0.1154 | -6.3679 | 0.5330 | 0.1066 |
| 19 | joint_orth_suffix:L10->L20 | 672 | 469 | 4.8514 | 5.2756 | -0.4499 | -6.6511 | 0.4819 | 0.0917 |
| 20 | joint_orth_all_components:L10->L20 | 672 | 469 | 3.9213 | 4.3576 | 0.4801 | -7.5691 | 0.5586 | 0.0704 |
| 21 | joint_full_frame_basis_only:L4->L10 | 672 | 469 | 0.9236 | 2.5115 | 3.4779 | -9.4152 | 0.8934 | 0.0277 |
| 22 | joint_raw_restore_both:L4->L10 | 672 | 469 | 1.5130 | 2.2943 | 2.8884 | -9.6324 | 0.8422 | 0.0213 |
| 23 | joint_full_frame_basis_only:L10->L20 | 672 | 469 | 0.6320 | 1.3824 | 3.7695 | -10.5443 | 0.9019 | 0.0213 |
| 24 | joint_relation_label_basis_only:L4->L10 | 672 | 469 | 0.4939 | 1.4350 | 3.9075 | -10.4917 | 0.9190 | 0.0149 |
| 25 | joint_relation_label_basis_only:L10->L20 | 672 | 469 | 0.4065 | 0.8854 | 3.9949 | -11.0413 | 0.9190 | 0.0149 |
| 26 | joint_raw_restore_both:L10->L20 | 672 | 469 | 0.5006 | 0.8355 | 3.9008 | -11.0912 | 0.9254 | 0.0128 |
| 27 | joint_pre_object_basis_only:L4->L10 | 672 | 469 | -0.0261 | 0.0108 | 4.4275 | -11.9158 | 0.9787 | 0.0021 |
| 28 | joint_pre_object_basis_only:L10->L20 | 672 | 469 | -0.0154 | 0.0852 | 4.4168 | -11.8415 | 0.9829 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_orth_pre_object:property | 192 | 100 | 9.0826 | 11.2458 | -6.3618 | 3.1676 | 0.0600 | 0.8600 |
| 2 | joint_raw:property | 192 | 100 | 9.7141 | 11.7023 | -6.9934 | 3.6240 | 0.0500 | 0.8500 |
| 3 | joint_raw:can_do | 192 | 148 | 9.3935 | 17.1183 | -5.6141 | 2.9947 | 0.0878 | 0.7500 |
| 4 | joint_orth_pre_object:location | 192 | 122 | 8.4018 | 10.8735 | -5.6308 | 2.5367 | 0.0820 | 0.7377 |
| 5 | joint_orth_pre_object:can_do | 192 | 148 | 9.0215 | 16.6287 | -5.2421 | 2.5051 | 0.1351 | 0.7162 |
| 6 | joint_raw:location | 192 | 122 | 8.8412 | 11.2289 | -6.0703 | 2.8921 | 0.0656 | 0.7131 |
| 7 | joint_raw:used_for | 192 | 148 | 12.2113 | 16.6428 | -6.2205 | 2.0518 | 0.0811 | 0.7027 |
| 8 | joint_orth_relation_label:can_do | 192 | 148 | 8.7876 | 15.9774 | -5.0082 | 1.8538 | 0.0811 | 0.6757 |
| 9 | joint_orth_pre_object:used_for | 192 | 148 | 11.8100 | 16.2476 | -5.8193 | 1.6565 | 0.0811 | 0.6622 |
| 10 | joint_raw:is_a | 192 | 164 | 11.1656 | 14.9113 | -5.5461 | 2.1777 | 0.1402 | 0.6402 |
| 11 | joint_orth_pre_object:is_a | 192 | 164 | 10.9438 | 14.6782 | -5.3242 | 1.9445 | 0.1280 | 0.6220 |
| 12 | joint_orth_full_frame:can_do | 192 | 148 | 8.0407 | 14.2221 | -4.2613 | 0.0985 | 0.1622 | 0.5473 |
| 13 | joint_orth_relation_label:property | 192 | 100 | 8.1978 | 7.6924 | -5.4770 | -0.3859 | 0.1000 | 0.4900 |
| 14 | joint_raw:material | 192 | 114 | 7.3410 | 11.1405 | -3.4775 | 0.1444 | 0.2105 | 0.4825 |
| 15 | joint_raw:part_of | 192 | 142 | 8.9833 | 12.4771 | -3.9807 | 0.0075 | 0.2254 | 0.4718 |
| 16 | joint_orth_pre_object:material | 192 | 114 | 7.1956 | 10.7803 | -3.3321 | -0.2158 | 0.2368 | 0.4649 |
| 17 | joint_orth_relation_label:used_for | 192 | 148 | 10.2133 | 13.8572 | -4.2225 | -0.7338 | 0.1892 | 0.4595 |
| 18 | joint_orth_relation_label:location | 192 | 122 | 6.5855 | 8.0739 | -3.8145 | -0.2630 | 0.1803 | 0.4590 |
| 19 | joint_orth_pre_object:part_of | 192 | 142 | 8.8003 | 12.1440 | -3.7977 | -0.3256 | 0.2676 | 0.4366 |
| 20 | joint_orth_full_frame:location | 192 | 122 | 6.0611 | 7.7067 | -3.2902 | -0.6301 | 0.2049 | 0.4180 |
| 21 | joint_orth_relation_label:is_a | 192 | 164 | 9.3103 | 11.0502 | -3.6907 | -1.6835 | 0.2622 | 0.4024 |
| 22 | joint_orth_full_frame:property | 192 | 100 | 7.5198 | 7.2551 | -4.7991 | -0.8232 | 0.1300 | 0.4000 |
| 23 | joint_orth_relation_label:part_of | 192 | 142 | 8.4073 | 10.3979 | -3.4047 | -2.0716 | 0.3169 | 0.3662 |
| 24 | joint_orth_full_frame:is_a | 192 | 164 | 8.8316 | 9.5026 | -3.2120 | -3.2310 | 0.2683 | 0.3476 |
| 25 | joint_mismatched_frame_raw:used_for | 192 | 148 | 10.7297 | 12.4379 | -4.7389 | -2.1531 | 0.1486 | 0.3378 |
| 26 | joint_mismatched_frame_raw:property | 192 | 100 | 7.7762 | 5.5178 | -5.0554 | -2.5605 | 0.1000 | 0.3200 |
| 27 | joint_mismatched_frame_raw:can_do | 192 | 148 | 8.1437 | 12.7022 | -4.3642 | -1.4214 | 0.1014 | 0.3176 |
| 28 | joint_mismatched_frame_raw:is_a | 192 | 164 | 9.1065 | 10.4020 | -3.4869 | -2.3317 | 0.2500 | 0.2988 |
| 29 | joint_mismatched_frame_raw:location | 192 | 122 | 5.9942 | 5.9950 | -3.2233 | -2.3418 | 0.1967 | 0.2869 |
| 30 | joint_orth_full_frame:part_of | 192 | 142 | 7.7185 | 9.2123 | -2.7159 | -3.2572 | 0.3169 | 0.2676 |
| 31 | joint_suffix_basis_only:can_do | 192 | 148 | 3.1375 | 10.7041 | 0.6419 | -3.4195 | 0.6081 | 0.2432 |
| 32 | joint_orth_all_components:property | 192 | 100 | 5.9114 | 5.3139 | -3.1906 | -2.7644 | 0.2400 | 0.2300 |
| 33 | joint_orth_boundary:location | 192 | 122 | 4.6103 | 5.2458 | -1.8393 | -3.0910 | 0.3197 | 0.2213 |
| 34 | joint_orth_suffix:location | 192 | 122 | 4.9085 | 5.1196 | -2.1375 | -3.2172 | 0.2787 | 0.2131 |
| 35 | joint_orth_boundary:property | 192 | 100 | 6.5347 | 5.8952 | -3.8140 | -2.1831 | 0.2000 | 0.2100 |
| 36 | joint_suffix_basis_only:property | 192 | 100 | 2.1639 | 4.3749 | 0.5568 | -3.7033 | 0.5200 | 0.2100 |
| 37 | joint_mismatched_frame_raw:part_of | 192 | 142 | 7.6573 | 9.3992 | -2.6547 | -3.0704 | 0.2887 | 0.2042 |
| 38 | joint_orth_suffix:property | 192 | 100 | 6.9509 | 5.6219 | -4.2301 | -2.4563 | 0.1800 | 0.2000 |
| 39 | joint_boundary_basis_only:property | 192 | 100 | 1.8968 | 3.9343 | 0.8239 | -4.1439 | 0.5500 | 0.2000 |
| 40 | joint_orth_relation_label:material | 192 | 114 | 4.9220 | 7.3555 | -1.0585 | -3.6406 | 0.4386 | 0.1842 |
| 41 | joint_orth_full_frame:material | 192 | 114 | 4.6627 | 6.9546 | -0.7992 | -4.0415 | 0.4825 | 0.1842 |
| 42 | joint_orth_boundary:is_a | 192 | 164 | 7.6126 | 6.8089 | -1.9930 | -5.9247 | 0.3354 | 0.1829 |
| 43 | joint_boundary_basis_only:can_do | 192 | 148 | 2.7342 | 9.6757 | 1.0452 | -4.4479 | 0.6689 | 0.1824 |
| 44 | joint_suffix_basis_only:location | 192 | 122 | 2.2227 | 4.0582 | 0.5482 | -4.2786 | 0.5902 | 0.1803 |
| 45 | joint_orth_all_components:location | 192 | 122 | 4.6285 | 4.5906 | -1.8575 | -3.7463 | 0.3279 | 0.1721 |
| 46 | joint_orth_suffix:is_a | 192 | 164 | 8.0450 | 7.0109 | -2.4254 | -5.7228 | 0.3354 | 0.1707 |
| 47 | joint_orth_full_frame:used_for | 192 | 148 | 8.7880 | 11.1215 | -2.7973 | -3.4695 | 0.3108 | 0.1689 |
| 48 | joint_boundary_basis_only:used_for | 192 | 148 | 3.7346 | 7.9058 | 2.2561 | -6.6852 | 0.7500 | 0.1554 |
| 49 | joint_orth_all_components:is_a | 192 | 164 | 7.4689 | 6.2368 | -1.8493 | -6.4969 | 0.3720 | 0.1524 |
| 50 | joint_mismatched_frame_raw:material | 192 | 114 | 5.4888 | 5.8352 | -1.6253 | -5.1609 | 0.3246 | 0.1404 |
| 51 | joint_suffix_basis_only:used_for | 192 | 148 | 4.2715 | 8.8553 | 1.7193 | -5.7357 | 0.6959 | 0.1351 |
| 52 | joint_suffix_basis_only:is_a | 192 | 164 | 3.2714 | 7.6567 | 2.3482 | -5.0769 | 0.7378 | 0.1280 |
| 53 | joint_boundary_basis_only:is_a | 192 | 164 | 3.0389 | 6.9421 | 2.5807 | -5.7916 | 0.7622 | 0.1098 |
| 54 | joint_orth_boundary:part_of | 192 | 142 | 6.2103 | 6.1244 | -1.2077 | -6.3452 | 0.3944 | 0.1056 |
| 55 | joint_orth_all_components:part_of | 192 | 142 | 6.1014 | 5.7147 | -1.0988 | -6.7549 | 0.4225 | 0.1056 |
| 56 | joint_full_frame_basis_only:property | 192 | 100 | 1.3126 | 2.3468 | 1.4081 | -5.7315 | 0.6400 | 0.1000 |
| 57 | joint_orth_suffix:part_of | 192 | 142 | 6.6510 | 6.0972 | -1.6484 | -6.3724 | 0.3803 | 0.0986 |
| 58 | joint_boundary_basis_only:location | 192 | 122 | 1.6764 | 3.4519 | 1.0946 | -4.8849 | 0.6885 | 0.0984 |
| 59 | joint_suffix_basis_only:part_of | 192 | 142 | 2.0922 | 6.0683 | 2.9104 | -6.4012 | 0.8380 | 0.0845 |
| 60 | joint_relation_label_basis_only:property | 192 | 100 | 1.1082 | 2.2524 | 1.6126 | -5.8259 | 0.7200 | 0.0800 |
| 61 | joint_orth_boundary:material | 192 | 114 | 3.8239 | 4.3883 | 0.0397 | -6.6078 | 0.5614 | 0.0702 |
| 62 | joint_raw_restore_both:property | 192 | 100 | 0.6118 | 1.1668 | 2.1090 | -6.9115 | 0.8400 | 0.0700 |
| 63 | joint_boundary_basis_only:part_of | 192 | 142 | 1.6182 | 5.2472 | 3.3844 | -7.2224 | 0.8944 | 0.0563 |
| 64 | joint_orth_boundary:used_for | 192 | 148 | 8.2564 | 8.5452 | -2.2657 | -6.0458 | 0.3716 | 0.0473 |
| 65 | joint_suffix_basis_only:material | 192 | 114 | 1.7894 | 4.7744 | 2.0741 | -6.2218 | 0.7368 | 0.0439 |
| 66 | joint_full_frame_basis_only:used_for | 192 | 148 | 1.6176 | 3.2626 | 4.3732 | -11.3284 | 0.9054 | 0.0405 |
| 67 | joint_boundary_basis_only:material | 192 | 114 | 1.4567 | 4.2962 | 2.4068 | -6.6999 | 0.7632 | 0.0351 |
| 68 | joint_orth_all_components:material | 192 | 114 | 3.9778 | 3.5567 | -0.1143 | -7.4394 | 0.5877 | 0.0351 |
| 69 | joint_orth_suffix:used_for | 192 | 148 | 8.4471 | 8.5624 | -2.4564 | -6.0286 | 0.3446 | 0.0338 |
| 70 | joint_orth_all_components:used_for | 192 | 148 | 8.0261 | 7.8853 | -2.0353 | -6.7057 | 0.3649 | 0.0270 |
| 71 | joint_raw_restore_both:location | 192 | 122 | 0.5538 | 1.6288 | 2.2172 | -6.7080 | 0.8770 | 0.0246 |
| 72 | joint_full_frame_basis_only:location | 192 | 122 | 0.2241 | 0.7150 | 2.5469 | -7.6218 | 0.8770 | 0.0246 |
| 73 | joint_relation_label_basis_only:part_of | 192 | 142 | 0.4002 | 1.4246 | 4.6024 | -11.0450 | 0.9577 | 0.0211 |
| 74 | joint_raw_restore_both:is_a | 192 | 164 | 1.1678 | 1.1873 | 4.4518 | -11.5464 | 0.8841 | 0.0183 |
| 75 | joint_orth_suffix:material | 192 | 114 | 3.9983 | 4.1123 | -0.1348 | -6.8838 | 0.5614 | 0.0175 |
| 76 | joint_full_frame_basis_only:part_of | 192 | 142 | 0.7149 | 2.3582 | 4.2877 | -10.1114 | 0.9507 | 0.0141 |
| 77 | joint_raw_restore_both:part_of | 192 | 142 | 1.5126 | 2.1271 | 3.4900 | -10.3424 | 0.8732 | 0.0141 |
| 78 | joint_relation_label_basis_only:used_for | 192 | 148 | 0.9127 | 1.4830 | 5.0780 | -13.1080 | 0.9527 | 0.0135 |
| 79 | joint_full_frame_basis_only:is_a | 192 | 164 | 1.1402 | 1.9857 | 4.4794 | -10.7479 | 0.9329 | 0.0122 |
| 80 | joint_pre_object_basis_only:property | 192 | 100 | 0.0445 | 0.1788 | 2.6762 | -7.8995 | 0.9600 | 0.0100 |
| 81 | joint_raw_restore_both:material | 192 | 114 | 1.2541 | 1.6971 | 2.6094 | -9.2990 | 0.8684 | 0.0088 |
| 82 | joint_orth_boundary:can_do | 192 | 148 | 4.9461 | 5.4247 | -1.1667 | -8.6989 | 0.4257 | 0.0068 |
| 83 | joint_relation_label_basis_only:is_a | 192 | 164 | 0.7872 | 0.9568 | 4.8323 | -11.7768 | 0.9390 | 0.0061 |
| 84 | joint_orth_suffix:can_do | 192 | 148 | 5.1575 | 4.5697 | -1.3781 | -9.5539 | 0.3716 | 0.0000 |
| 85 | joint_orth_all_components:can_do | 192 | 148 | 4.4794 | 3.9445 | -0.7000 | -10.1791 | 0.4595 | 0.0000 |
| 86 | joint_full_frame_basis_only:can_do | 192 | 148 | 0.0038 | 1.8205 | 3.7756 | -12.3031 | 0.9595 | 0.0000 |
| 87 | joint_raw_restore_both:can_do | 192 | 148 | 0.9494 | 1.5717 | 2.8300 | -12.5519 | 0.8784 | 0.0000 |
| 88 | joint_raw_restore_both:used_for | 192 | 148 | 0.8504 | 1.5515 | 5.1404 | -13.0395 | 0.9459 | 0.0000 |
| 89 | joint_relation_label_basis_only:can_do | 192 | 148 | -0.1859 | 1.1351 | 3.9653 | -12.9885 | 0.9797 | 0.0000 |
| 90 | joint_full_frame_basis_only:material | 192 | 114 | 0.3727 | 0.8026 | 3.4909 | -10.1935 | 0.9386 | 0.0000 |
| 91 | joint_relation_label_basis_only:material | 192 | 114 | 0.1500 | 0.5555 | 3.7136 | -10.4406 | 0.9123 | 0.0000 |
| 92 | joint_relation_label_basis_only:location | 192 | 122 | 0.0073 | 0.4345 | 2.7637 | -7.9023 | 0.9016 | 0.0000 |
| 93 | joint_pre_object_basis_only:used_for | 192 | 148 | 0.1027 | 0.2876 | 5.8881 | -14.3034 | 1.0000 | 0.0000 |
| 94 | joint_pre_object_basis_only:part_of | 192 | 142 | 0.0011 | 0.1389 | 5.0015 | -12.3307 | 0.9718 | 0.0000 |
| 95 | joint_pre_object_basis_only:material | 192 | 114 | 0.0609 | 0.0775 | 3.8026 | -10.9187 | 0.9649 | 0.0000 |
| 96 | joint_pre_object_basis_only:can_do | 192 | 148 | -0.0888 | -0.0300 | 3.8682 | -14.1536 | 0.9730 | 0.0000 |
| 97 | joint_pre_object_basis_only:location | 192 | 122 | -0.1599 | -0.0591 | 2.9309 | -8.3959 | 1.0000 | 0.0000 |
| 98 | joint_pre_object_basis_only:is_a | 192 | 164 | -0.0826 | -0.1969 | 5.7022 | -12.9306 | 0.9878 | 0.0000 |

## deepseek7b

items=672, basis_items=224, rows=18816, layer_pairs=[[8, 10], [12, 14]]
module=resid_out, contrast_rank=64, component_rank=24, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_raw | 1344 | 632 | 6.9095 | 11.6139 | -2.8868 | -1.4475 | 0.2769 | 0.4193 |
| 2 | joint_orth_pre_object | 1344 | 632 | 6.1414 | 10.6894 | -2.1187 | -2.3721 | 0.3354 | 0.3655 |
| 3 | joint_orth_relation_label | 1344 | 632 | 5.7042 | 9.5748 | -1.6816 | -3.4866 | 0.3829 | 0.2801 |
| 4 | joint_orth_full_frame | 1344 | 632 | 5.7248 | 9.4406 | -1.7021 | -3.6208 | 0.3703 | 0.2706 |
| 5 | joint_mismatched_frame_raw | 1344 | 632 | 6.0082 | 9.0987 | -1.9855 | -3.9627 | 0.3402 | 0.2674 |
| 6 | joint_suffix_basis_only | 1344 | 632 | 2.9121 | 7.3274 | 1.1106 | -5.7340 | 0.6139 | 0.1566 |
| 7 | joint_boundary_basis_only | 1344 | 632 | 2.3613 | 6.0443 | 1.6613 | -7.0171 | 0.6535 | 0.1171 |
| 8 | joint_orth_boundary | 1344 | 632 | 3.7819 | 4.6512 | 0.2408 | -8.4103 | 0.5269 | 0.0617 |
| 9 | joint_orth_suffix | 1344 | 632 | 4.1141 | 3.9577 | -0.0914 | -9.1037 | 0.4984 | 0.0396 |
| 10 | joint_orth_all_components | 1344 | 632 | 3.0392 | 3.1118 | 0.9835 | -9.9497 | 0.5870 | 0.0316 |
| 11 | joint_relation_label_basis_only | 1344 | 632 | 0.7817 | 1.7022 | 3.2410 | -11.3592 | 0.8370 | 0.0285 |
| 12 | joint_full_frame_basis_only | 1344 | 632 | 0.7139 | 1.7966 | 3.3088 | -11.2648 | 0.8244 | 0.0206 |
| 13 | joint_pre_object_basis_only | 1344 | 632 | 0.4528 | 0.8354 | 3.5699 | -12.2260 | 0.8877 | 0.0079 |
| 14 | joint_raw_restore_both | 1344 | 632 | 0.6336 | 0.6857 | 3.3891 | -12.3757 | 0.8655 | 0.0047 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_raw:L8->L10 | 672 | 316 | 7.3423 | 11.8910 | -3.3196 | -1.1704 | 0.2500 | 0.4335 |
| 2 | joint_raw:L12->L14 | 672 | 316 | 6.4767 | 11.3368 | -2.4540 | -1.7246 | 0.3038 | 0.4051 |
| 3 | joint_orth_pre_object:L8->L10 | 672 | 316 | 6.5690 | 10.9873 | -2.5463 | -2.0741 | 0.3070 | 0.3924 |
| 4 | joint_orth_pre_object:L12->L14 | 672 | 316 | 5.7137 | 10.3914 | -1.6910 | -2.6700 | 0.3639 | 0.3386 |
| 5 | joint_orth_relation_label:L8->L10 | 672 | 316 | 6.1881 | 9.7526 | -2.1654 | -3.3088 | 0.3576 | 0.2816 |
| 6 | joint_orth_full_frame:L8->L10 | 672 | 316 | 6.1015 | 9.6713 | -2.0788 | -3.3901 | 0.3481 | 0.2785 |
| 7 | joint_orth_relation_label:L12->L14 | 672 | 316 | 5.2204 | 9.3970 | -1.1977 | -3.6644 | 0.4082 | 0.2785 |
| 8 | joint_mismatched_frame_raw:L8->L10 | 672 | 316 | 6.3113 | 9.0703 | -2.2887 | -3.9911 | 0.3070 | 0.2722 |
| 9 | joint_orth_full_frame:L12->L14 | 672 | 316 | 5.3480 | 9.2098 | -1.3253 | -3.8516 | 0.3924 | 0.2627 |
| 10 | joint_mismatched_frame_raw:L12->L14 | 672 | 316 | 5.7051 | 9.1271 | -1.6824 | -3.9343 | 0.3734 | 0.2627 |
| 11 | joint_suffix_basis_only:L12->L14 | 672 | 316 | 2.8161 | 7.2323 | 1.2066 | -5.8292 | 0.6076 | 0.1677 |
| 12 | joint_suffix_basis_only:L8->L10 | 672 | 316 | 3.0081 | 7.4226 | 1.0145 | -5.6388 | 0.6203 | 0.1456 |
| 13 | joint_boundary_basis_only:L12->L14 | 672 | 316 | 2.2226 | 5.8490 | 1.8001 | -7.2124 | 0.6677 | 0.1203 |
| 14 | joint_boundary_basis_only:L8->L10 | 672 | 316 | 2.5001 | 6.2396 | 1.5226 | -6.8218 | 0.6392 | 0.1139 |
| 15 | joint_orth_boundary:L8->L10 | 672 | 316 | 4.1310 | 4.7675 | -0.1083 | -8.2939 | 0.4778 | 0.0665 |
| 16 | joint_orth_boundary:L12->L14 | 672 | 316 | 3.4327 | 4.5348 | 0.5899 | -8.5266 | 0.5759 | 0.0570 |
| 17 | joint_orth_suffix:L8->L10 | 672 | 316 | 4.4840 | 4.1443 | -0.4613 | -8.9171 | 0.4747 | 0.0411 |
| 18 | joint_orth_suffix:L12->L14 | 672 | 316 | 3.7441 | 3.7711 | 0.2786 | -9.2903 | 0.5222 | 0.0380 |
| 19 | joint_orth_all_components:L8->L10 | 672 | 316 | 3.4798 | 3.4996 | 0.5428 | -9.5619 | 0.5380 | 0.0316 |
| 20 | joint_orth_all_components:L12->L14 | 672 | 316 | 2.5986 | 2.7240 | 1.4241 | -10.3375 | 0.6361 | 0.0316 |
| 21 | joint_relation_label_basis_only:L8->L10 | 672 | 316 | 0.8237 | 1.6714 | 3.1989 | -11.3900 | 0.8228 | 0.0316 |
| 22 | joint_relation_label_basis_only:L12->L14 | 672 | 316 | 0.7396 | 1.7330 | 3.2831 | -11.3285 | 0.8513 | 0.0253 |
| 23 | joint_full_frame_basis_only:L8->L10 | 672 | 316 | 0.7636 | 1.8114 | 3.2591 | -11.2500 | 0.8133 | 0.0222 |
| 24 | joint_full_frame_basis_only:L12->L14 | 672 | 316 | 0.6641 | 1.7818 | 3.3585 | -11.2796 | 0.8354 | 0.0190 |
| 25 | joint_pre_object_basis_only:L12->L14 | 672 | 316 | 0.4410 | 0.8480 | 3.5817 | -12.2134 | 0.8892 | 0.0095 |
| 26 | joint_raw_restore_both:L8->L10 | 672 | 316 | 1.0436 | 0.9858 | 2.9791 | -12.0756 | 0.8101 | 0.0063 |
| 27 | joint_pre_object_basis_only:L8->L10 | 672 | 316 | 0.4646 | 0.8229 | 3.5581 | -12.2386 | 0.8861 | 0.0063 |
| 28 | joint_raw_restore_both:L12->L14 | 672 | 316 | 0.2236 | 0.3855 | 3.7991 | -12.6759 | 0.9209 | 0.0032 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_raw:can_do | 192 | 124 | 8.5702 | 16.6386 | -4.1872 | 1.3188 | 0.2016 | 0.6129 |
| 2 | joint_orth_relation_label:can_do | 192 | 124 | 7.7401 | 15.6415 | -3.3571 | 0.3216 | 0.2661 | 0.5484 |
| 3 | joint_orth_full_frame:can_do | 192 | 124 | 7.8989 | 15.4521 | -3.5159 | 0.1322 | 0.2419 | 0.5484 |
| 4 | joint_orth_pre_object:can_do | 192 | 124 | 7.3611 | 15.0831 | -2.9781 | -0.2368 | 0.2903 | 0.5081 |
| 5 | joint_raw:used_for | 192 | 74 | 9.1601 | 15.1862 | -4.1940 | -0.8001 | 0.2162 | 0.5000 |
| 6 | joint_orth_pre_object:location | 192 | 56 | 6.5857 | 7.3400 | -3.2708 | -1.5584 | 0.2679 | 0.4821 |
| 7 | joint_raw:location | 192 | 56 | 7.0842 | 8.0804 | -3.7694 | -0.8180 | 0.2679 | 0.4643 |
| 8 | joint_orth_pre_object:used_for | 192 | 74 | 8.2468 | 14.2976 | -3.2807 | -1.6887 | 0.2162 | 0.4595 |
| 9 | joint_orth_relation_label:location | 192 | 56 | 6.5241 | 7.3402 | -3.2092 | -1.5582 | 0.3214 | 0.4464 |
| 10 | joint_raw:property | 192 | 70 | 5.4606 | 8.7253 | -2.1646 | -0.2338 | 0.3143 | 0.4286 |
| 11 | joint_orth_full_frame:location | 192 | 56 | 6.0893 | 6.8467 | -2.7745 | -2.0517 | 0.3214 | 0.3929 |
| 12 | joint_raw:part_of | 192 | 74 | 6.3398 | 12.6851 | -2.5432 | -2.9258 | 0.3243 | 0.3919 |
| 13 | joint_mismatched_frame_raw:can_do | 192 | 124 | 7.4286 | 13.6447 | -3.0456 | -1.6752 | 0.2097 | 0.3871 |
| 14 | joint_orth_pre_object:part_of | 192 | 74 | 5.6699 | 11.9361 | -1.8733 | -3.6748 | 0.3649 | 0.3243 |
| 15 | joint_mismatched_frame_raw:is_a | 192 | 138 | 5.9994 | 9.1053 | -1.8669 | -3.9564 | 0.3551 | 0.3116 |
| 16 | joint_mismatched_frame_raw:part_of | 192 | 74 | 5.9465 | 11.6583 | -2.1498 | -3.9526 | 0.3243 | 0.3108 |
| 17 | joint_mismatched_frame_raw:location | 192 | 56 | 5.1367 | 5.2153 | -1.8219 | -3.6831 | 0.3393 | 0.3036 |
| 18 | joint_orth_pre_object:property | 192 | 70 | 4.4653 | 7.4338 | -1.1692 | -1.5253 | 0.4143 | 0.3000 |
| 19 | joint_raw:is_a | 192 | 138 | 6.3403 | 10.2123 | -2.2078 | -2.8494 | 0.3043 | 0.2899 |
| 20 | joint_orth_pre_object:is_a | 192 | 138 | 5.8583 | 9.6761 | -1.7257 | -3.3856 | 0.3768 | 0.2826 |
| 21 | joint_raw:material | 192 | 96 | 5.2415 | 7.7264 | -1.4522 | -3.6173 | 0.3229 | 0.2812 |
| 22 | joint_suffix_basis_only:property | 192 | 70 | 2.9757 | 4.9705 | 0.3204 | -3.9886 | 0.5000 | 0.2714 |
| 23 | joint_orth_boundary:location | 192 | 56 | 3.1528 | 3.3079 | 0.1621 | -5.5905 | 0.4643 | 0.2679 |
| 24 | joint_orth_full_frame:property | 192 | 70 | 3.9407 | 6.2390 | -0.6447 | -2.7201 | 0.4714 | 0.2571 |
| 25 | joint_orth_relation_label:property | 192 | 70 | 3.8125 | 6.1802 | -0.5164 | -2.7789 | 0.4571 | 0.2571 |
| 26 | joint_mismatched_frame_raw:property | 192 | 70 | 4.7307 | 5.4981 | -1.4347 | -3.4610 | 0.4429 | 0.2571 |
| 27 | joint_orth_pre_object:material | 192 | 96 | 4.6762 | 7.0559 | -0.8870 | -4.2877 | 0.3854 | 0.2396 |
| 28 | joint_boundary_basis_only:property | 192 | 70 | 2.7594 | 4.9527 | 0.5367 | -4.0064 | 0.5143 | 0.2286 |
| 29 | joint_orth_relation_label:used_for | 192 | 74 | 7.2710 | 11.2026 | -2.3049 | -4.7837 | 0.2838 | 0.2162 |
| 30 | joint_suffix_basis_only:location | 192 | 56 | 2.6686 | 3.8836 | 0.6463 | -5.0148 | 0.5000 | 0.1964 |
| 31 | joint_boundary_basis_only:location | 192 | 56 | 2.2165 | 3.4234 | 1.0984 | -5.4749 | 0.5000 | 0.1964 |
| 32 | joint_orth_suffix:location | 192 | 56 | 3.2274 | 3.0633 | 0.0875 | -5.8351 | 0.5179 | 0.1964 |
| 33 | joint_suffix_basis_only:used_for | 192 | 74 | 3.3825 | 9.4581 | 1.5836 | -6.5281 | 0.6622 | 0.1892 |
| 34 | joint_suffix_basis_only:can_do | 192 | 124 | 3.8907 | 11.4075 | 0.4923 | -3.9124 | 0.6129 | 0.1855 |
| 35 | joint_orth_all_components:location | 192 | 56 | 2.7280 | 2.6983 | 0.5869 | -6.2001 | 0.5714 | 0.1786 |
| 36 | joint_boundary_basis_only:can_do | 192 | 124 | 2.7104 | 8.8563 | 1.6726 | -6.4636 | 0.6694 | 0.1774 |
| 37 | joint_orth_full_frame:material | 192 | 96 | 4.4248 | 7.0378 | -0.6356 | -4.3059 | 0.4688 | 0.1771 |
| 38 | joint_orth_full_frame:used_for | 192 | 74 | 6.9090 | 11.2461 | -1.9429 | -4.7401 | 0.2838 | 0.1757 |
| 39 | joint_orth_relation_label:is_a | 192 | 138 | 4.9993 | 8.2364 | -0.8668 | -4.8253 | 0.4638 | 0.1667 |
| 40 | joint_orth_relation_label:part_of | 192 | 74 | 5.1425 | 9.1679 | -1.3458 | -6.4430 | 0.3514 | 0.1622 |
| 41 | joint_orth_full_frame:is_a | 192 | 138 | 5.2551 | 7.6762 | -1.1226 | -5.3855 | 0.4348 | 0.1594 |
| 42 | joint_orth_relation_label:material | 192 | 96 | 4.2143 | 6.5001 | -0.4251 | -4.8435 | 0.5000 | 0.1562 |
| 43 | joint_mismatched_frame_raw:used_for | 192 | 74 | 7.9433 | 10.8781 | -2.9772 | -5.1082 | 0.2838 | 0.1486 |
| 44 | joint_orth_full_frame:part_of | 192 | 74 | 4.8716 | 8.9607 | -1.0750 | -6.6503 | 0.3649 | 0.1486 |
| 45 | joint_suffix_basis_only:part_of | 192 | 74 | 2.1238 | 8.8206 | 1.6728 | -6.7903 | 0.6486 | 0.1486 |
| 46 | joint_relation_label_basis_only:location | 192 | 56 | 1.2075 | 1.5412 | 2.1074 | -7.3572 | 0.7679 | 0.1250 |
| 47 | joint_orth_boundary:property | 192 | 70 | 2.5754 | 3.4306 | 0.7207 | -5.5285 | 0.5714 | 0.1143 |
| 48 | joint_orth_suffix:property | 192 | 70 | 2.8939 | 3.1023 | 0.4021 | -5.8569 | 0.5714 | 0.1143 |
| 49 | joint_suffix_basis_only:is_a | 192 | 138 | 2.6372 | 6.3429 | 1.4954 | -6.7188 | 0.6449 | 0.1087 |
| 50 | joint_full_frame_basis_only:location | 192 | 56 | 0.9223 | 0.9623 | 2.3926 | -7.9360 | 0.6964 | 0.1071 |
| 51 | joint_boundary_basis_only:is_a | 192 | 138 | 2.1799 | 5.5151 | 1.9526 | -7.5466 | 0.6812 | 0.0942 |
| 52 | joint_mismatched_frame_raw:material | 192 | 96 | 4.1821 | 4.7633 | -0.3928 | -6.5803 | 0.4688 | 0.0938 |
| 53 | joint_full_frame_basis_only:property | 192 | 70 | 2.0885 | 2.9706 | 1.2075 | -5.9885 | 0.6571 | 0.0857 |
| 54 | joint_relation_label_basis_only:property | 192 | 70 | 1.8864 | 2.6090 | 1.4097 | -6.3502 | 0.7143 | 0.0857 |
| 55 | joint_boundary_basis_only:used_for | 192 | 74 | 2.6720 | 7.6649 | 2.2941 | -8.3214 | 0.7568 | 0.0811 |
| 56 | joint_boundary_basis_only:part_of | 192 | 74 | 1.8369 | 7.0535 | 1.9597 | -8.5574 | 0.7703 | 0.0811 |
| 57 | joint_suffix_basis_only:material | 192 | 96 | 2.3840 | 4.4067 | 1.4052 | -6.9370 | 0.6562 | 0.0625 |
| 58 | joint_orth_boundary:used_for | 192 | 74 | 6.3556 | 6.9417 | -1.3895 | -9.0446 | 0.3649 | 0.0541 |
| 59 | joint_orth_all_components:property | 192 | 70 | 1.9582 | 1.9483 | 1.3379 | -7.0108 | 0.6714 | 0.0429 |
| 60 | joint_orth_boundary:can_do | 192 | 124 | 3.5790 | 5.7612 | 0.8040 | -9.5586 | 0.5806 | 0.0323 |
| 61 | joint_relation_label_basis_only:can_do | 192 | 124 | 0.1686 | 2.3042 | 4.2144 | -13.0157 | 0.8952 | 0.0323 |
| 62 | joint_orth_suffix:material | 192 | 96 | 2.6782 | 3.2273 | 1.1111 | -8.1163 | 0.6562 | 0.0312 |
| 63 | joint_orth_boundary:is_a | 192 | 138 | 4.3844 | 4.9403 | -0.2519 | -8.1215 | 0.5072 | 0.0290 |
| 64 | joint_orth_all_components:is_a | 192 | 138 | 3.4953 | 3.5226 | 0.6372 | -9.5391 | 0.5145 | 0.0290 |
| 65 | joint_raw_restore_both:property | 192 | 70 | 0.5274 | 0.5766 | 2.7686 | -8.3825 | 0.8857 | 0.0286 |
| 66 | joint_orth_boundary:part_of | 192 | 74 | 3.5796 | 3.5058 | 0.2170 | -12.1051 | 0.5000 | 0.0270 |
| 67 | joint_orth_suffix:part_of | 192 | 74 | 4.4178 | 3.3472 | -0.6212 | -12.2637 | 0.4459 | 0.0270 |
| 68 | joint_orth_boundary:material | 192 | 96 | 2.5964 | 3.5926 | 1.1929 | -7.7511 | 0.6354 | 0.0208 |
| 69 | joint_orth_all_components:material | 192 | 96 | 1.9583 | 2.2009 | 1.8309 | -9.1427 | 0.6875 | 0.0208 |
| 70 | joint_pre_object_basis_only:location | 192 | 56 | 0.2809 | 0.3584 | 3.0339 | -8.5400 | 0.8750 | 0.0179 |
| 71 | joint_pre_object_basis_only:is_a | 192 | 138 | 0.4159 | 0.5473 | 3.7167 | -12.5145 | 0.8696 | 0.0145 |
| 72 | joint_pre_object_basis_only:property | 192 | 70 | 0.2889 | 0.4311 | 3.0071 | -8.5280 | 0.9000 | 0.0143 |
| 73 | joint_orth_suffix:used_for | 192 | 74 | 6.3365 | 6.4815 | -1.3705 | -9.5048 | 0.3108 | 0.0135 |
| 74 | joint_orth_all_components:part_of | 192 | 74 | 2.9909 | 2.6031 | 0.8057 | -13.0078 | 0.6486 | 0.0135 |
| 75 | joint_pre_object_basis_only:can_do | 192 | 124 | 0.5200 | 0.9660 | 3.8630 | -14.3538 | 0.8952 | 0.0081 |
| 76 | joint_full_frame_basis_only:is_a | 192 | 138 | 0.7080 | 1.5049 | 3.4246 | -11.5569 | 0.8478 | 0.0072 |
| 77 | joint_relation_label_basis_only:is_a | 192 | 138 | 0.7318 | 1.2781 | 3.4007 | -11.7837 | 0.8406 | 0.0072 |
| 78 | joint_raw_restore_both:is_a | 192 | 138 | 0.5614 | 0.6610 | 3.5711 | -12.4007 | 0.8551 | 0.0072 |
| 79 | joint_orth_all_components:used_for | 192 | 74 | 5.3129 | 5.5171 | -0.3468 | -10.4692 | 0.4189 | 0.0000 |
| 80 | joint_orth_suffix:is_a | 192 | 138 | 4.8112 | 4.2890 | -0.6787 | -8.7727 | 0.4710 | 0.0000 |
| 81 | joint_orth_suffix:can_do | 192 | 124 | 4.0315 | 3.8995 | 0.3515 | -11.4203 | 0.5000 | 0.0000 |
| 82 | joint_boundary_basis_only:material | 192 | 96 | 2.1304 | 3.4707 | 1.6588 | -7.8730 | 0.6146 | 0.0000 |
| 83 | joint_orth_all_components:can_do | 192 | 124 | 2.7913 | 3.0714 | 1.5917 | -12.2485 | 0.6129 | 0.0000 |
| 84 | joint_full_frame_basis_only:part_of | 192 | 74 | 1.0315 | 2.9878 | 2.7652 | -12.6231 | 0.8514 | 0.0000 |
| 85 | joint_relation_label_basis_only:part_of | 192 | 74 | 0.8192 | 2.4237 | 2.9774 | -13.1872 | 0.8919 | 0.0000 |
| 86 | joint_pre_object_basis_only:used_for | 192 | 74 | 0.9273 | 2.3864 | 4.0387 | -13.5999 | 0.9324 | 0.0000 |
| 87 | joint_full_frame_basis_only:used_for | 192 | 74 | 0.5648 | 2.1369 | 4.4012 | -13.8494 | 0.9189 | 0.0000 |
| 88 | joint_full_frame_basis_only:can_do | 192 | 124 | -0.2260 | 1.6812 | 4.6090 | -13.6387 | 0.9194 | 0.0000 |
| 89 | joint_relation_label_basis_only:used_for | 192 | 74 | 0.7226 | 1.2290 | 4.2434 | -14.7573 | 0.8919 | 0.0000 |
| 90 | joint_raw_restore_both:part_of | 192 | 74 | 0.9491 | 1.0303 | 2.8476 | -14.5807 | 0.8378 | 0.0000 |
| 91 | joint_raw_restore_both:used_for | 192 | 74 | 0.5158 | 1.0162 | 4.4503 | -14.9701 | 0.9054 | 0.0000 |
| 92 | joint_full_frame_basis_only:material | 192 | 96 | 0.6825 | 0.8151 | 3.1067 | -10.5286 | 0.7708 | 0.0000 |
| 93 | joint_pre_object_basis_only:part_of | 192 | 74 | 0.4428 | 0.7948 | 3.3538 | -14.8162 | 0.8784 | 0.0000 |
| 94 | joint_relation_label_basis_only:material | 192 | 96 | 0.6081 | 0.7755 | 3.1812 | -10.5682 | 0.8021 | 0.0000 |
| 95 | joint_raw_restore_both:can_do | 192 | 124 | 0.8602 | 0.6300 | 3.5228 | -14.6898 | 0.8710 | 0.0000 |
| 96 | joint_raw_restore_both:material | 192 | 96 | 0.4222 | 0.5493 | 3.3671 | -10.7944 | 0.8646 | 0.0000 |
| 97 | joint_pre_object_basis_only:material | 192 | 96 | 0.2808 | 0.4897 | 3.5084 | -10.8539 | 0.8750 | 0.0000 |
| 98 | joint_raw_restore_both:location | 192 | 56 | 0.5432 | 0.3478 | 2.7716 | -8.5506 | 0.8393 | 0.0000 |

