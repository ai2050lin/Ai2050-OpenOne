# Phase76 Object-Frame Joint Closure Summary

## qwen3

items=216, rows=3024, layer_pairs=[[4, 8], [8, 12]]

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched | 432 | 282 | 11.3796 | 14.3994 | -6.6304 | 1.5088 | 0.0851 | 0.6028 |
| 2 | joint_mismatched_frame | 432 | 282 | 10.0603 | 8.9988 | -5.3112 | -3.8918 | 0.1099 | 0.2092 |
| 3 | object_only_matched | 432 | 282 | 7.5133 | 5.7724 | -2.7641 | -7.1182 | 0.3014 | 0.0745 |
| 4 | joint_restore_frame_only | 432 | 282 | 6.8484 | 6.0463 | -2.0992 | -6.8443 | 0.3227 | 0.0957 |
| 5 | joint_restore_object_only | 432 | 282 | 5.4833 | 10.1781 | -0.7341 | -2.7125 | 0.4504 | 0.2943 |
| 6 | frame_only_matched | 432 | 282 | 4.0672 | 8.7956 | 0.6820 | -4.0950 | 0.5922 | 0.1702 |
| 7 | joint_restore_both | 432 | 282 | 0.7328 | 0.9057 | 4.0164 | -11.9849 | 0.8546 | 0.0035 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:L4->L8 | 216 | 141 | 12.0060 | 14.2684 | -7.2568 | 1.3778 | 0.0567 | 0.5957 |
| 2 | joint_matched:L8->L12 | 216 | 141 | 10.7532 | 14.5303 | -6.0040 | 1.6398 | 0.1135 | 0.6099 |
| 3 | joint_mismatched_frame:L4->L8 | 216 | 141 | 10.4611 | 9.5774 | -5.7119 | -3.3132 | 0.0851 | 0.2482 |
| 4 | joint_mismatched_frame:L8->L12 | 216 | 141 | 9.6595 | 8.4201 | -4.9104 | -4.4705 | 0.1348 | 0.1702 |
| 5 | object_only_matched:L4->L8 | 216 | 141 | 8.7284 | 6.1882 | -3.9793 | -6.7024 | 0.1915 | 0.0851 |
| 6 | joint_restore_frame_only:L4->L8 | 216 | 141 | 7.9032 | 6.6725 | -3.1540 | -6.2181 | 0.2482 | 0.1135 |
| 7 | object_only_matched:L8->L12 | 216 | 141 | 6.2982 | 5.3566 | -1.5490 | -7.5340 | 0.4113 | 0.0638 |
| 8 | joint_restore_frame_only:L8->L12 | 216 | 141 | 5.7936 | 5.4201 | -1.0444 | -7.4705 | 0.3972 | 0.0780 |
| 9 | joint_restore_object_only:L4->L8 | 216 | 141 | 5.5677 | 9.7795 | -0.8185 | -3.1110 | 0.4539 | 0.2837 |
| 10 | joint_restore_object_only:L8->L12 | 216 | 141 | 5.3989 | 10.5766 | -0.6497 | -2.3140 | 0.4468 | 0.3050 |
| 11 | frame_only_matched:L8->L12 | 216 | 141 | 4.1615 | 9.1390 | 0.5877 | -3.7516 | 0.5816 | 0.1915 |
| 12 | frame_only_matched:L4->L8 | 216 | 141 | 3.9729 | 8.4522 | 0.7763 | -4.4384 | 0.6028 | 0.1489 |
| 13 | joint_restore_both:L4->L8 | 216 | 141 | 0.7417 | 1.0479 | 4.0074 | -11.8427 | 0.8723 | 0.0000 |
| 14 | joint_restore_both:L8->L12 | 216 | 141 | 0.7238 | 0.7635 | 4.0254 | -12.1271 | 0.8369 | 0.0071 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:can_do | 72 | 54 | 13.5527 | 17.3343 | -8.4967 | 2.4839 | 0.0185 | 0.6667 |
| 2 | joint_mismatched_frame:used_for | 72 | 30 | 13.3083 | 10.5167 | -7.0125 | -4.1805 | 0.0667 | 0.2000 |
| 3 | joint_matched:is_a | 72 | 70 | 13.0722 | 17.5949 | -6.6828 | 2.6456 | 0.0857 | 0.7000 |
| 4 | joint_matched:used_for | 72 | 30 | 12.2789 | 11.9462 | -5.9831 | -2.7511 | 0.2333 | 0.2333 |
| 5 | joint_matched:property | 72 | 36 | 11.7313 | 11.9787 | -8.5789 | 4.0655 | 0.0833 | 0.7778 |
| 6 | joint_mismatched_frame:can_do | 72 | 54 | 11.7192 | 12.1088 | -6.6632 | -2.7417 | 0.0370 | 0.2037 |
| 7 | joint_mismatched_frame:is_a | 72 | 70 | 11.2866 | 10.9636 | -4.8971 | -3.9856 | 0.1286 | 0.2286 |
| 8 | joint_matched:location | 72 | 40 | 10.4110 | 13.6215 | -6.2068 | 2.4799 | 0.0250 | 0.7500 |
| 9 | object_only_matched:can_do | 72 | 54 | 8.8103 | 5.5881 | -3.7543 | -9.2624 | 0.2037 | 0.0000 |
| 10 | joint_mismatched_frame:property | 72 | 36 | 8.8012 | 1.7240 | -5.6489 | -6.1892 | 0.1667 | 0.1389 |
| 11 | object_only_matched:used_for | 72 | 30 | 8.7604 | 6.7691 | -2.4646 | -7.9282 | 0.3667 | 0.0000 |
| 12 | joint_restore_frame_only:can_do | 72 | 54 | 8.3941 | 5.7737 | -3.3380 | -9.0768 | 0.2407 | 0.0000 |
| 13 | joint_mismatched_frame:location | 72 | 40 | 8.2164 | 8.9908 | -4.0123 | -2.1508 | 0.1500 | 0.3000 |
| 14 | object_only_matched:is_a | 72 | 70 | 8.1221 | 6.7080 | -1.7327 | -8.2412 | 0.3714 | 0.0286 |
| 15 | joint_restore_frame_only:used_for | 72 | 30 | 8.0287 | 6.4136 | -1.7329 | -8.2836 | 0.3667 | 0.0333 |
| 16 | object_only_matched:property | 72 | 36 | 7.8756 | 5.3471 | -4.7232 | -2.5661 | 0.0833 | 0.2500 |
| 17 | joint_restore_frame_only:property | 72 | 36 | 7.5478 | 6.0709 | -4.3955 | -1.8424 | 0.1667 | 0.3611 |
| 18 | joint_restore_object_only:can_do | 72 | 54 | 7.1875 | 13.8224 | -2.1315 | -1.0281 | 0.2778 | 0.4259 |
| 19 | joint_mismatched_frame:material | 72 | 52 | 7.1032 | 7.2908 | -4.2482 | -4.5421 | 0.1154 | 0.1731 |
| 20 | joint_restore_frame_only:is_a | 72 | 70 | 7.0211 | 6.9955 | -0.6317 | -7.9538 | 0.4000 | 0.0571 |
| 21 | joint_restore_object_only:is_a | 72 | 70 | 6.8897 | 12.6091 | -0.5003 | -2.3401 | 0.4429 | 0.3857 |
| 22 | object_only_matched:location | 72 | 40 | 6.8395 | 5.7411 | -2.6353 | -5.4004 | 0.4000 | 0.1250 |
| 23 | joint_matched:material | 72 | 52 | 6.8272 | 10.7394 | -3.9722 | -1.0935 | 0.1154 | 0.3846 |
| 24 | joint_restore_frame_only:location | 72 | 40 | 5.9200 | 6.4716 | -1.7159 | -4.6700 | 0.4000 | 0.1250 |
| 25 | frame_only_matched:is_a | 72 | 70 | 5.8001 | 11.5300 | 0.5893 | -3.4193 | 0.5571 | 0.2143 |
| 26 | joint_restore_object_only:used_for | 72 | 30 | 5.2821 | 8.8285 | 1.0137 | -5.8687 | 0.5667 | 0.1667 |
| 27 | frame_only_matched:can_do | 72 | 54 | 5.0670 | 12.2803 | -0.0110 | -2.5702 | 0.5000 | 0.2222 |
| 28 | object_only_matched:material | 72 | 52 | 4.8950 | 4.4479 | -2.0400 | -7.3850 | 0.3462 | 0.0962 |
| 29 | joint_restore_frame_only:material | 72 | 52 | 4.5597 | 4.4957 | -1.7048 | -7.3373 | 0.3269 | 0.0769 |
| 30 | joint_restore_object_only:property | 72 | 36 | 4.4345 | 6.5069 | -1.2821 | -1.4063 | 0.5278 | 0.3889 |
| 31 | joint_restore_object_only:location | 72 | 40 | 4.3919 | 8.5973 | -0.1878 | -2.5443 | 0.6000 | 0.2250 |
| 32 | frame_only_matched:used_for | 72 | 30 | 4.1552 | 7.7329 | 2.1407 | -6.9643 | 0.7667 | 0.1000 |
| 33 | joint_restore_object_only:material | 72 | 52 | 3.5020 | 7.6572 | -0.6470 | -4.1758 | 0.4038 | 0.0962 |
| 34 | frame_only_matched:location | 72 | 40 | 3.3136 | 6.9490 | 0.8905 | -4.1926 | 0.7000 | 0.1500 |
| 35 | frame_only_matched:material | 72 | 52 | 2.6798 | 6.8564 | 0.1751 | -4.9765 | 0.5000 | 0.0769 |
| 36 | frame_only_matched:property | 72 | 36 | 1.9659 | 3.9900 | 1.1865 | -3.9232 | 0.6667 | 0.2222 |
| 37 | joint_restore_both:can_do | 72 | 54 | 1.3740 | 1.0270 | 3.6820 | -13.8234 | 0.8148 | 0.0000 |
| 38 | joint_restore_both:property | 72 | 36 | 1.0037 | 0.8002 | 2.1486 | -7.1131 | 0.7222 | 0.0000 |
| 39 | joint_restore_both:material | 72 | 52 | 0.6170 | 1.0223 | 2.2380 | -10.8107 | 0.7692 | 0.0000 |
| 40 | joint_restore_both:location | 72 | 40 | 0.5914 | 0.9839 | 3.6127 | -10.1577 | 0.8750 | 0.0250 |
| 41 | joint_restore_both:is_a | 72 | 70 | 0.4903 | 0.8989 | 5.8991 | -14.0504 | 1.0000 | 0.0000 |
| 42 | joint_restore_both:used_for | 72 | 30 | 0.2081 | 0.5234 | 6.0877 | -14.1738 | 0.8667 | 0.0000 |

## glm4

items=216, rows=3024, layer_pairs=[[4, 10], [10, 20]]

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched | 432 | 300 | 10.3216 | 13.4800 | -5.5684 | 1.8823 | 0.1567 | 0.6533 |
| 2 | joint_mismatched_frame | 432 | 300 | 8.7697 | 7.4227 | -4.0165 | -4.1750 | 0.2133 | 0.1800 |
| 3 | joint_restore_object_only | 432 | 300 | 6.2730 | 10.8527 | -1.5198 | -0.7450 | 0.4100 | 0.4533 |
| 4 | object_only_matched | 432 | 300 | 5.3553 | 4.5411 | -0.6022 | -7.0566 | 0.4867 | 0.0733 |
| 5 | joint_restore_frame_only | 432 | 300 | 4.3353 | 4.9553 | 0.4179 | -6.6425 | 0.5600 | 0.0733 |
| 6 | frame_only_matched | 432 | 300 | 3.9600 | 8.3804 | 0.7932 | -3.2173 | 0.6267 | 0.2633 |
| 7 | joint_restore_both | 432 | 300 | 0.8120 | 1.6062 | 3.9412 | -9.9915 | 0.9067 | 0.0067 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:L4->L10 | 216 | 150 | 11.8084 | 14.1173 | -7.0552 | 2.5196 | 0.0733 | 0.6800 |
| 2 | joint_mismatched_frame:L4->L10 | 216 | 150 | 10.2705 | 8.5797 | -5.5174 | -3.0180 | 0.1000 | 0.2733 |
| 3 | joint_matched:L10->L20 | 216 | 150 | 8.8349 | 12.8428 | -4.0817 | 1.2451 | 0.2400 | 0.6267 |
| 4 | object_only_matched:L4->L10 | 216 | 150 | 8.2509 | 6.1635 | -3.4977 | -5.4342 | 0.2200 | 0.1067 |
| 5 | joint_restore_object_only:L4->L10 | 216 | 150 | 7.3840 | 11.8257 | -2.6308 | 0.2280 | 0.3000 | 0.5467 |
| 6 | joint_mismatched_frame:L10->L20 | 216 | 150 | 7.2689 | 6.2658 | -2.5157 | -5.3319 | 0.3267 | 0.0867 |
| 7 | joint_restore_frame_only:L4->L10 | 216 | 150 | 6.3873 | 6.4474 | -1.6341 | -5.1504 | 0.3600 | 0.1067 |
| 8 | joint_restore_object_only:L10->L20 | 216 | 150 | 5.1620 | 9.8797 | -0.4088 | -1.7180 | 0.5200 | 0.3600 |
| 9 | frame_only_matched:L10->L20 | 216 | 150 | 4.6016 | 9.1656 | 0.1516 | -2.4321 | 0.5667 | 0.3333 |
| 10 | frame_only_matched:L4->L10 | 216 | 150 | 3.3184 | 7.5952 | 1.4348 | -4.0025 | 0.6867 | 0.1933 |
| 11 | object_only_matched:L10->L20 | 216 | 150 | 2.4598 | 2.9186 | 2.2934 | -8.6791 | 0.7533 | 0.0400 |
| 12 | joint_restore_frame_only:L10->L20 | 216 | 150 | 2.2833 | 3.4632 | 2.4699 | -8.1346 | 0.7600 | 0.0400 |
| 13 | joint_restore_both:L4->L10 | 216 | 150 | 1.6132 | 2.7706 | 3.1400 | -8.8271 | 0.8533 | 0.0133 |
| 14 | joint_restore_both:L10->L20 | 216 | 150 | 0.0108 | 0.4418 | 4.7423 | -11.1559 | 0.9600 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:can_do | 72 | 60 | 12.7828 | 16.8680 | -7.8414 | 4.0757 | 0.0000 | 0.8500 |
| 2 | joint_matched:used_for | 72 | 44 | 12.1617 | 13.8461 | -6.0747 | 1.4511 | 0.1591 | 0.6136 |
| 3 | joint_mismatched_frame:used_for | 72 | 44 | 11.3653 | 8.0831 | -5.2783 | -4.3120 | 0.1818 | 0.1364 |
| 4 | joint_matched:is_a | 72 | 72 | 10.8251 | 13.8826 | -4.3774 | 0.6749 | 0.2500 | 0.5278 |
| 5 | joint_mismatched_frame:can_do | 72 | 60 | 10.7153 | 10.4778 | -5.7739 | -2.3144 | 0.0167 | 0.3167 |
| 6 | joint_matched:property | 72 | 46 | 10.2595 | 10.7197 | -7.8395 | 4.0491 | 0.0652 | 0.8478 |
| 7 | joint_mismatched_frame:is_a | 72 | 72 | 9.9112 | 7.6276 | -3.4635 | -5.5801 | 0.3333 | 0.0972 |
| 8 | joint_matched:location | 72 | 42 | 8.8560 | 12.4386 | -4.9084 | 1.7846 | 0.0476 | 0.6667 |
| 9 | joint_restore_object_only:can_do | 72 | 60 | 7.7448 | 14.0172 | -2.8035 | 1.2249 | 0.2667 | 0.5833 |
| 10 | joint_restore_object_only:is_a | 72 | 72 | 7.1023 | 11.5676 | -0.6546 | -1.6400 | 0.4722 | 0.3750 |
| 11 | joint_mismatched_frame:location | 72 | 42 | 7.0690 | 7.8223 | -3.1214 | -2.8317 | 0.2619 | 0.2381 |
| 12 | joint_restore_object_only:used_for | 72 | 44 | 7.0486 | 10.5009 | -0.9617 | -1.8942 | 0.5455 | 0.3864 |
| 13 | object_only_matched:used_for | 72 | 44 | 6.8397 | 6.1183 | -0.7527 | -6.2767 | 0.4091 | 0.1591 |
| 14 | joint_mismatched_frame:property | 72 | 46 | 6.8158 | 2.0104 | -4.3959 | -4.6602 | 0.1522 | 0.1304 |
| 15 | object_only_matched:is_a | 72 | 72 | 6.6421 | 4.5843 | -0.1944 | -8.6234 | 0.5694 | 0.0417 |
| 16 | joint_restore_object_only:location | 72 | 42 | 5.8164 | 10.0367 | -1.8688 | -0.6173 | 0.3333 | 0.4286 |
| 17 | object_only_matched:can_do | 72 | 60 | 5.7620 | 4.2519 | -0.8206 | -8.5404 | 0.5000 | 0.0167 |
| 18 | joint_restore_frame_only:used_for | 72 | 44 | 5.6801 | 5.6856 | 0.4069 | -6.7095 | 0.5227 | 0.0909 |
| 19 | joint_restore_frame_only:is_a | 72 | 72 | 5.6021 | 4.8311 | 0.8456 | -8.3766 | 0.5972 | 0.0417 |
| 20 | frame_only_matched:can_do | 72 | 60 | 5.4273 | 11.8210 | -0.4859 | -0.9712 | 0.5000 | 0.3833 |
| 21 | frame_only_matched:used_for | 72 | 44 | 5.2868 | 8.5753 | 0.8001 | -3.8198 | 0.6591 | 0.2045 |
| 22 | joint_restore_object_only:property | 72 | 46 | 4.9672 | 8.0285 | -2.5472 | 1.3579 | 0.3261 | 0.6522 |
| 23 | object_only_matched:property | 72 | 46 | 4.9118 | 4.0009 | -2.4918 | -2.6697 | 0.1957 | 0.2174 |
| 24 | joint_matched:material | 72 | 36 | 4.7530 | 11.3231 | -1.4115 | -1.4861 | 0.4722 | 0.3611 |
| 25 | joint_mismatched_frame:material | 72 | 36 | 4.5524 | 7.5638 | -1.2109 | -5.2454 | 0.3611 | 0.1667 |
| 26 | joint_restore_frame_only:can_do | 72 | 60 | 4.5061 | 5.0241 | 0.4353 | -7.7681 | 0.5667 | 0.0333 |
| 27 | frame_only_matched:is_a | 72 | 72 | 4.3733 | 9.1016 | 2.0744 | -4.1060 | 0.7778 | 0.1806 |
| 28 | object_only_matched:location | 72 | 42 | 4.3328 | 4.6373 | -0.3852 | -6.0167 | 0.5476 | 0.0000 |
| 29 | joint_restore_frame_only:property | 72 | 46 | 4.3137 | 4.9564 | -1.8937 | -1.7143 | 0.2609 | 0.2174 |
| 30 | frame_only_matched:location | 72 | 42 | 3.6043 | 7.2735 | 0.3433 | -3.3805 | 0.5714 | 0.2619 |
| 31 | joint_restore_object_only:material | 72 | 36 | 3.4144 | 9.1394 | -0.0729 | -3.6697 | 0.5556 | 0.2500 |
| 32 | joint_restore_frame_only:location | 72 | 42 | 2.8100 | 5.0700 | 1.1376 | -5.5840 | 0.6905 | 0.0238 |
| 33 | frame_only_matched:material | 72 | 36 | 2.0849 | 7.2384 | 1.2566 | -5.5708 | 0.6389 | 0.1944 |
| 34 | object_only_matched:material | 72 | 36 | 2.0497 | 3.5870 | 1.2918 | -9.2222 | 0.6944 | 0.0278 |
| 35 | frame_only_matched:property | 72 | 46 | 1.9222 | 4.4819 | 0.4977 | -2.1887 | 0.5652 | 0.3478 |
| 36 | joint_restore_frame_only:material | 72 | 36 | 1.6806 | 4.0610 | 1.6609 | -8.7481 | 0.7500 | 0.0556 |
| 37 | joint_restore_both:is_a | 72 | 72 | 1.5324 | 1.8813 | 4.9153 | -11.3263 | 0.9167 | 0.0000 |
| 38 | joint_restore_both:used_for | 72 | 44 | 0.7529 | 0.8611 | 5.3341 | -11.5339 | 0.9318 | 0.0000 |
| 39 | joint_restore_both:can_do | 72 | 60 | 0.7147 | 1.7125 | 4.2266 | -11.0797 | 0.9500 | 0.0000 |
| 40 | joint_restore_both:material | 72 | 36 | 0.6494 | 1.8814 | 2.6921 | -10.9278 | 0.8056 | 0.0000 |
| 41 | joint_restore_both:location | 72 | 42 | 0.5682 | 1.9556 | 3.3794 | -8.6984 | 0.9048 | 0.0000 |
| 42 | joint_restore_both:property | 72 | 46 | 0.2178 | 1.2152 | 2.2022 | -5.4554 | 0.8913 | 0.0435 |

## deepseek7b

items=216, rows=3024, layer_pairs=[[8, 10], [12, 14]]

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched | 432 | 184 | 9.3725 | 12.5306 | -3.8368 | -1.2280 | 0.2717 | 0.4457 |
| 2 | joint_mismatched_frame | 432 | 184 | 8.4139 | 7.1188 | -2.8783 | -6.6399 | 0.2989 | 0.0707 |
| 3 | joint_restore_object_only | 432 | 184 | 6.3496 | 10.3380 | -0.8139 | -3.4206 | 0.4402 | 0.3043 |
| 4 | frame_only_matched | 432 | 184 | 5.8079 | 9.8481 | -0.2722 | -3.9106 | 0.4891 | 0.2663 |
| 5 | object_only_matched | 432 | 184 | 4.0315 | 3.8427 | 1.5042 | -9.9160 | 0.6196 | 0.0163 |
| 6 | joint_restore_frame_only | 432 | 184 | 3.6703 | 3.7976 | 1.8654 | -9.9610 | 0.6359 | 0.0217 |
| 7 | joint_restore_both | 432 | 184 | 0.4271 | 0.5590 | 5.1086 | -13.1996 | 0.9130 | 0.0054 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:L8->L10 | 216 | 92 | 9.8944 | 12.6858 | -4.3587 | -1.0729 | 0.2174 | 0.4565 |
| 2 | joint_matched:L12->L14 | 216 | 92 | 8.8505 | 12.3755 | -3.3149 | -1.3832 | 0.3261 | 0.4348 |
| 3 | joint_mismatched_frame:L8->L10 | 216 | 92 | 8.8009 | 7.4025 | -3.2653 | -6.3562 | 0.2826 | 0.0543 |
| 4 | joint_mismatched_frame:L12->L14 | 216 | 92 | 8.0270 | 6.8351 | -2.4913 | -6.9236 | 0.3152 | 0.0870 |
| 5 | joint_restore_object_only:L8->L10 | 216 | 92 | 6.7558 | 10.6379 | -1.2201 | -3.1207 | 0.4130 | 0.3261 |
| 6 | joint_restore_object_only:L12->L14 | 216 | 92 | 5.9434 | 10.0382 | -0.4077 | -3.7205 | 0.4674 | 0.2826 |
| 7 | frame_only_matched:L12->L14 | 216 | 92 | 5.8448 | 9.9086 | -0.3091 | -3.8500 | 0.4891 | 0.2717 |
| 8 | frame_only_matched:L8->L10 | 216 | 92 | 5.7710 | 9.7875 | -0.2353 | -3.9712 | 0.4891 | 0.2609 |
| 9 | object_only_matched:L8->L10 | 216 | 92 | 4.9640 | 4.2813 | 0.5717 | -9.4773 | 0.5326 | 0.0217 |
| 10 | joint_restore_frame_only:L8->L10 | 216 | 92 | 4.1775 | 4.0898 | 1.3582 | -9.6688 | 0.5652 | 0.0217 |
| 11 | joint_restore_frame_only:L12->L14 | 216 | 92 | 3.1631 | 3.5055 | 2.3726 | -10.2532 | 0.7065 | 0.0217 |
| 12 | object_only_matched:L12->L14 | 216 | 92 | 3.0989 | 3.4040 | 2.4367 | -10.3546 | 0.7065 | 0.0109 |
| 13 | joint_restore_both:L8->L10 | 216 | 92 | 0.8739 | 0.8418 | 4.6618 | -12.9169 | 0.8696 | 0.0109 |
| 14 | joint_restore_both:L12->L14 | 216 | 92 | -0.0197 | 0.2763 | 5.5553 | -13.4823 | 0.9565 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:can_do | 72 | 42 | 13.8635 | 17.6439 | -7.8250 | 3.4676 | 0.0476 | 0.7857 |
| 2 | joint_mismatched_frame:can_do | 72 | 42 | 11.6858 | 11.0001 | -5.6473 | -3.1763 | 0.0952 | 0.1667 |
| 3 | joint_mismatched_frame:used_for | 72 | 26 | 10.7668 | 9.9648 | -2.8523 | -4.9038 | 0.1923 | 0.0000 |
| 4 | joint_restore_object_only:can_do | 72 | 42 | 10.1122 | 15.6614 | -4.0737 | 1.4851 | 0.1190 | 0.6429 |
| 5 | joint_matched:used_for | 72 | 26 | 9.9869 | 13.9578 | -2.0725 | -0.9108 | 0.3077 | 0.4615 |
| 6 | joint_matched:property | 72 | 20 | 9.3158 | 10.2682 | -5.6210 | 0.8664 | 0.2000 | 0.6500 |
| 7 | frame_only_matched:can_do | 72 | 42 | 9.2662 | 15.2948 | -3.2277 | 1.1185 | 0.2381 | 0.5238 |
| 8 | joint_matched:is_a | 72 | 60 | 8.4558 | 11.6532 | -2.6866 | -3.2344 | 0.3333 | 0.2833 |
| 9 | joint_mismatched_frame:property | 72 | 20 | 8.1618 | 2.7544 | -4.4670 | -6.6474 | 0.2000 | 0.1000 |
| 10 | joint_mismatched_frame:is_a | 72 | 60 | 7.3615 | 6.4835 | -1.5923 | -8.4041 | 0.4500 | 0.0333 |
| 11 | object_only_matched:used_for | 72 | 26 | 5.9924 | 6.8384 | 1.9220 | -8.0302 | 0.6154 | 0.0000 |
| 12 | joint_matched:location | 72 | 14 | 5.9367 | 11.4770 | -0.4033 | -4.1686 | 0.5000 | 0.2857 |
| 13 | joint_restore_object_only:property | 72 | 20 | 5.9358 | 7.9350 | -2.2411 | -1.4667 | 0.4000 | 0.5000 |
| 14 | joint_restore_object_only:used_for | 72 | 26 | 5.8373 | 10.4243 | 2.0772 | -4.4444 | 0.6923 | 0.1538 |
| 15 | frame_only_matched:property | 72 | 20 | 5.7532 | 7.5660 | -2.0584 | -1.8358 | 0.3500 | 0.5500 |
| 16 | joint_restore_frame_only:used_for | 72 | 26 | 5.4413 | 6.6107 | 2.4731 | -8.2579 | 0.6923 | 0.0000 |
| 17 | joint_restore_object_only:is_a | 72 | 60 | 5.3851 | 9.5924 | 0.3841 | -5.2952 | 0.5167 | 0.1667 |
| 18 | frame_only_matched:is_a | 72 | 60 | 5.2359 | 9.1770 | 0.5333 | -5.7106 | 0.5667 | 0.1333 |
| 19 | joint_mismatched_frame:location | 72 | 14 | 5.0061 | 5.9009 | 0.5273 | -9.7447 | 0.4286 | 0.1429 |
| 20 | joint_matched:material | 72 | 22 | 4.8105 | 6.2024 | -2.0078 | -5.1281 | 0.4091 | 0.1364 |
| 21 | object_only_matched:can_do | 72 | 42 | 4.7619 | 4.2761 | 1.2766 | -9.9003 | 0.5952 | 0.0000 |
| 22 | frame_only_matched:used_for | 72 | 26 | 4.6881 | 9.0989 | 3.2263 | -5.7698 | 0.7308 | 0.1538 |
| 23 | joint_mismatched_frame:material | 72 | 22 | 4.6553 | 2.8209 | -1.8527 | -8.5096 | 0.4091 | 0.0000 |
| 24 | joint_restore_object_only:location | 72 | 14 | 4.6070 | 9.1381 | 0.9264 | -6.5075 | 0.6429 | 0.2143 |
| 25 | joint_restore_frame_only:can_do | 72 | 42 | 4.3756 | 4.1580 | 1.6629 | -10.0184 | 0.6190 | 0.0000 |
| 26 | object_only_matched:property | 72 | 20 | 4.1968 | 2.7904 | -0.5020 | -6.6114 | 0.3500 | 0.1500 |
| 27 | frame_only_matched:location | 72 | 14 | 4.0388 | 8.8888 | 1.4946 | -6.7568 | 0.7143 | 0.1429 |
| 28 | joint_restore_object_only:material | 72 | 22 | 3.8875 | 5.0550 | -1.0848 | -6.2755 | 0.4545 | 0.0909 |
| 29 | joint_restore_frame_only:property | 72 | 20 | 3.8238 | 3.0898 | -0.1290 | -6.3120 | 0.3500 | 0.2000 |
| 30 | object_only_matched:is_a | 72 | 60 | 3.8143 | 3.7817 | 1.9548 | -11.1059 | 0.7500 | 0.0000 |
| 31 | joint_restore_frame_only:is_a | 72 | 60 | 3.4536 | 3.6313 | 2.3156 | -11.2563 | 0.7333 | 0.0000 |
| 32 | frame_only_matched:material | 72 | 22 | 3.2645 | 4.8505 | -0.4618 | -6.4800 | 0.4545 | 0.0909 |
| 33 | object_only_matched:location | 72 | 14 | 2.5306 | 3.5400 | 3.0028 | -12.1056 | 0.5714 | 0.0000 |
| 34 | joint_restore_frame_only:location | 72 | 14 | 2.1308 | 4.0009 | 3.4027 | -11.6447 | 0.6429 | 0.0000 |
| 35 | object_only_matched:material | 72 | 22 | 1.7166 | 0.7905 | 1.0860 | -10.5400 | 0.5909 | 0.0000 |
| 36 | joint_restore_frame_only:material | 72 | 22 | 1.6620 | 0.7531 | 1.1406 | -10.5774 | 0.5909 | 0.0000 |
| 37 | joint_restore_both:material | 72 | 22 | 0.6656 | 0.4731 | 2.1371 | -10.8574 | 0.6818 | 0.0000 |
| 38 | joint_restore_both:can_do | 72 | 42 | 0.6463 | 0.7170 | 5.3922 | -13.4593 | 0.9286 | 0.0000 |
| 39 | joint_restore_both:used_for | 72 | 26 | 0.5420 | 1.0438 | 7.3724 | -13.8249 | 0.9615 | 0.0000 |
| 40 | joint_restore_both:property | 72 | 20 | 0.4994 | 0.2682 | 3.1954 | -9.1335 | 0.8500 | 0.0500 |
| 41 | joint_restore_both:is_a | 72 | 60 | 0.2392 | 0.3189 | 5.5299 | -14.5687 | 0.9667 | 0.0000 |
| 42 | joint_restore_both:location | 72 | 14 | -0.1164 | 0.7647 | 5.6498 | -14.8810 | 1.0000 | 0.0000 |

