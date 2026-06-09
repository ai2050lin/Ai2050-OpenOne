# Phase77 Balanced Cross-Relation Joint Closure Summary

## qwen3

objects=24, items=672, rows=9408, layer_pairs=[[4, 8], [8, 12]]
relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched | 1344 | 780 | 10.6469 | 13.6120 | -6.4293 | 0.4939 | 0.0577 | 0.5295 |
| 2 | joint_restore_object_only | 1344 | 780 | 5.2564 | 9.6212 | -1.0387 | -3.4969 | 0.4205 | 0.2244 |
| 3 | joint_mismatched_frame | 1344 | 780 | 9.1362 | 8.7118 | -4.9185 | -4.4062 | 0.1064 | 0.1744 |
| 4 | frame_only_matched | 1344 | 780 | 4.2557 | 8.4183 | -0.0380 | -4.6997 | 0.5333 | 0.1564 |
| 5 | object_only_matched | 1344 | 780 | 7.6197 | 5.4163 | -3.4020 | -7.7018 | 0.2295 | 0.0667 |
| 6 | joint_restore_frame_only | 1344 | 780 | 7.0098 | 5.5489 | -2.7922 | -7.5692 | 0.2846 | 0.0577 |
| 7 | joint_restore_both | 1344 | 780 | 0.7039 | 0.7830 | 3.5138 | -12.3350 | 0.8744 | 0.0077 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:L8->L12 | 672 | 390 | 10.3321 | 14.2122 | -6.1145 | 1.0941 | 0.0641 | 0.5718 |
| 2 | joint_matched:L4->L8 | 672 | 390 | 10.9617 | 13.0117 | -6.7440 | -0.1063 | 0.0513 | 0.4872 |
| 3 | joint_restore_object_only:L8->L12 | 672 | 390 | 5.2555 | 10.3299 | -1.0378 | -2.7881 | 0.4333 | 0.2487 |
| 4 | joint_restore_object_only:L4->L8 | 672 | 390 | 5.2573 | 8.9124 | -1.0396 | -4.2057 | 0.4077 | 0.2000 |
| 5 | joint_mismatched_frame:L4->L8 | 672 | 390 | 9.5129 | 9.0220 | -5.2952 | -4.0960 | 0.0949 | 0.1949 |
| 6 | frame_only_matched:L8->L12 | 672 | 390 | 4.2968 | 9.0375 | -0.0792 | -4.0805 | 0.5359 | 0.1897 |
| 7 | joint_mismatched_frame:L8->L12 | 672 | 390 | 8.7595 | 8.4016 | -4.5418 | -4.7164 | 0.1179 | 0.1538 |
| 8 | frame_only_matched:L4->L8 | 672 | 390 | 4.2145 | 7.7991 | 0.0031 | -5.3190 | 0.5308 | 0.1231 |
| 9 | object_only_matched:L4->L8 | 672 | 390 | 8.5763 | 5.6831 | -4.3587 | -7.4350 | 0.1564 | 0.0821 |
| 10 | joint_restore_frame_only:L4->L8 | 672 | 390 | 8.1279 | 5.9188 | -3.9102 | -7.1993 | 0.1974 | 0.0667 |
| 11 | object_only_matched:L8->L12 | 672 | 390 | 6.6631 | 5.1495 | -2.4454 | -7.9685 | 0.3026 | 0.0513 |
| 12 | joint_restore_frame_only:L8->L12 | 672 | 390 | 5.8918 | 5.1789 | -1.6741 | -7.9391 | 0.3718 | 0.0487 |
| 13 | joint_restore_both:L4->L8 | 672 | 390 | 0.8421 | 0.9132 | 3.3755 | -12.2049 | 0.8641 | 0.0128 |
| 14 | joint_restore_both:L8->L12 | 672 | 390 | 0.5657 | 0.6529 | 3.6520 | -12.4652 | 0.8846 | 0.0026 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:can_do | 192 | 144 | 11.4236 | 16.9673 | -6.2713 | 1.8905 | 0.0625 | 0.7153 |
| 2 | joint_matched:location | 192 | 74 | 9.2001 | 11.9923 | -6.4946 | 1.1177 | 0.0541 | 0.6351 |
| 3 | joint_matched:property | 192 | 76 | 9.8111 | 10.0038 | -6.7019 | 1.5404 | 0.0395 | 0.6053 |
| 4 | joint_matched:is_a | 192 | 160 | 11.5471 | 13.8640 | -6.2731 | 0.2681 | 0.0625 | 0.4938 |
| 5 | joint_matched:used_for | 192 | 110 | 13.1611 | 15.5665 | -8.2409 | 0.6466 | 0.0273 | 0.4545 |
| 6 | joint_matched:part_of | 192 | 106 | 9.7537 | 12.9344 | -6.1591 | -0.9250 | 0.0943 | 0.4151 |
| 7 | joint_matched:material | 192 | 110 | 8.2182 | 11.1337 | -5.0794 | -0.9341 | 0.0545 | 0.4000 |
| 8 | joint_restore_object_only:can_do | 192 | 144 | 6.6679 | 13.1194 | -1.5156 | -1.9574 | 0.3958 | 0.3542 |
| 9 | joint_mismatched_frame:can_do | 192 | 144 | 10.2397 | 12.7335 | -5.0874 | -2.3433 | 0.0903 | 0.3333 |
| 10 | joint_restore_object_only:property | 192 | 76 | 4.6174 | 6.1859 | -1.5083 | -2.2775 | 0.3289 | 0.3289 |
| 11 | joint_restore_object_only:location | 192 | 74 | 3.3819 | 6.8728 | -0.6765 | -4.0018 | 0.4595 | 0.2432 |
| 12 | joint_restore_object_only:is_a | 192 | 160 | 6.0830 | 10.1416 | -0.8091 | -3.4543 | 0.4562 | 0.2375 |
| 13 | frame_only_matched:property | 192 | 76 | 3.8527 | 4.7220 | -0.7436 | -3.7414 | 0.3816 | 0.2368 |
| 14 | frame_only_matched:is_a | 192 | 160 | 5.0199 | 9.0704 | 0.2541 | -4.5255 | 0.5938 | 0.1938 |
| 15 | frame_only_matched:can_do | 192 | 144 | 5.0446 | 11.4608 | 0.1077 | -3.6160 | 0.5972 | 0.1806 |
| 16 | joint_restore_object_only:part_of | 192 | 106 | 3.3631 | 9.3019 | 0.2314 | -4.5575 | 0.5660 | 0.1792 |
| 17 | frame_only_matched:location | 192 | 74 | 2.7559 | 6.0166 | -0.0504 | -4.8579 | 0.5676 | 0.1757 |
| 18 | joint_restore_frame_only:location | 192 | 74 | 6.0652 | 5.9800 | -3.3598 | -4.8946 | 0.2162 | 0.1757 |
| 19 | joint_mismatched_frame:is_a | 192 | 160 | 9.6992 | 9.1897 | -4.4253 | -4.4062 | 0.1375 | 0.1750 |
| 20 | joint_restore_frame_only:property | 192 | 76 | 5.6308 | 5.1396 | -2.5217 | -3.3238 | 0.2368 | 0.1579 |
| 21 | frame_only_matched:used_for | 192 | 110 | 5.8749 | 9.8692 | -0.9546 | -5.0507 | 0.4273 | 0.1545 |
| 22 | object_only_matched:location | 192 | 74 | 6.5929 | 5.4445 | -3.8875 | -5.4300 | 0.1757 | 0.1486 |
| 23 | joint_restore_object_only:used_for | 192 | 110 | 6.6989 | 10.9659 | -1.7787 | -3.9540 | 0.3545 | 0.1455 |
| 24 | object_only_matched:property | 192 | 76 | 5.9527 | 4.4067 | -2.8436 | -4.0567 | 0.2105 | 0.1447 |
| 25 | joint_mismatched_frame:material | 192 | 110 | 7.0453 | 6.6966 | -3.9065 | -5.3712 | 0.1182 | 0.1364 |
| 26 | joint_mismatched_frame:location | 192 | 74 | 7.0961 | 5.8022 | -4.3907 | -5.0724 | 0.0946 | 0.1351 |
| 27 | joint_mismatched_frame:part_of | 192 | 106 | 8.5067 | 8.4280 | -4.9121 | -5.4313 | 0.0943 | 0.1321 |
| 28 | frame_only_matched:part_of | 192 | 106 | 2.5838 | 8.0501 | 1.0108 | -5.8093 | 0.6698 | 0.1321 |
| 29 | joint_mismatched_frame:used_for | 192 | 110 | 12.0008 | 9.7972 | -7.0806 | -5.1227 | 0.0636 | 0.1182 |
| 30 | joint_mismatched_frame:property | 192 | 76 | 7.6042 | 4.6604 | -4.4951 | -3.8030 | 0.1447 | 0.1053 |
| 31 | object_only_matched:is_a | 192 | 160 | 8.7189 | 5.7111 | -3.4449 | -7.8848 | 0.2500 | 0.0813 |
| 32 | joint_restore_object_only:material | 192 | 110 | 4.2903 | 7.4700 | -1.1516 | -4.5979 | 0.3636 | 0.0727 |
| 33 | object_only_matched:material | 192 | 110 | 5.6890 | 4.1057 | -2.5502 | -7.9622 | 0.2727 | 0.0727 |
| 34 | object_only_matched:part_of | 192 | 106 | 8.0608 | 4.7175 | -4.4662 | -9.1419 | 0.1698 | 0.0660 |
| 35 | joint_restore_frame_only:is_a | 192 | 160 | 7.7703 | 5.3202 | -2.4964 | -8.2757 | 0.3375 | 0.0563 |
| 36 | joint_restore_frame_only:material | 192 | 110 | 4.6984 | 4.4315 | -1.5596 | -7.6363 | 0.3909 | 0.0455 |
| 37 | joint_restore_both:property | 192 | 76 | 0.3979 | 1.1247 | 2.7112 | -7.3387 | 0.8553 | 0.0395 |
| 38 | joint_restore_frame_only:part_of | 192 | 106 | 7.8558 | 5.0195 | -4.2612 | -8.8399 | 0.1887 | 0.0283 |
| 39 | frame_only_matched:material | 192 | 110 | 3.3905 | 6.5604 | -0.2518 | -5.5075 | 0.4182 | 0.0273 |
| 40 | joint_restore_both:part_of | 192 | 106 | 0.8978 | 1.1607 | 2.6967 | -12.6986 | 0.8302 | 0.0189 |
| 41 | joint_restore_frame_only:used_for | 192 | 110 | 9.1513 | 8.1769 | -4.2311 | -6.7430 | 0.2091 | 0.0182 |
| 42 | object_only_matched:can_do | 192 | 144 | 7.3321 | 4.9442 | -2.1798 | -10.1326 | 0.3194 | 0.0139 |
| 43 | joint_restore_both:location | 192 | 74 | 0.2329 | 0.6514 | 2.4725 | -10.2232 | 0.8784 | 0.0135 |
| 44 | joint_restore_frame_only:can_do | 192 | 144 | 6.8853 | 5.0330 | -1.7330 | -10.0438 | 0.3333 | 0.0069 |
| 45 | object_only_matched:used_for | 192 | 110 | 9.7455 | 8.2681 | -4.8252 | -6.6518 | 0.1455 | 0.0000 |
| 46 | joint_restore_both:can_do | 192 | 144 | 1.5812 | 1.0358 | 3.5711 | -14.0410 | 0.8611 | 0.0000 |
| 47 | joint_restore_both:used_for | 192 | 110 | 0.5056 | 0.6864 | 4.4146 | -14.2335 | 0.8545 | 0.0000 |
| 48 | joint_restore_both:material | 192 | 110 | 0.3944 | 0.4967 | 2.7443 | -11.5711 | 0.8727 | 0.0000 |
| 49 | joint_restore_both:is_a | 192 | 160 | 0.4981 | 0.4672 | 4.7758 | -13.1287 | 0.9375 | 0.0000 |

## glm4

objects=24, items=672, rows=9408, layer_pairs=[[4, 10], [10, 20]]
relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched | 1344 | 888 | 9.9306 | 14.0403 | -5.5991 | 1.8232 | 0.1126 | 0.6486 |
| 2 | joint_restore_object_only | 1344 | 888 | 4.9531 | 10.2833 | -0.6216 | -1.9339 | 0.4775 | 0.3378 |
| 3 | joint_mismatched_frame | 1344 | 888 | 8.2591 | 8.2753 | -3.9276 | -3.9419 | 0.1847 | 0.2005 |
| 4 | frame_only_matched | 1344 | 888 | 3.3112 | 8.1506 | 1.0203 | -4.0666 | 0.6340 | 0.2005 |
| 5 | joint_restore_frame_only | 1344 | 888 | 5.1869 | 5.6640 | -0.8554 | -6.5532 | 0.4550 | 0.1002 |
| 6 | object_only_matched | 1344 | 888 | 5.8580 | 5.3420 | -1.5265 | -6.8751 | 0.4020 | 0.0923 |
| 7 | joint_restore_both | 1344 | 888 | 0.7635 | 1.2880 | 3.5680 | -10.9291 | 0.8998 | 0.0113 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:L4->L10 | 672 | 444 | 11.0774 | 14.3068 | -6.7459 | 2.0896 | 0.0518 | 0.6712 |
| 2 | joint_matched:L10->L20 | 672 | 444 | 8.7839 | 13.7739 | -4.4524 | 1.5567 | 0.1734 | 0.6261 |
| 3 | joint_restore_object_only:L4->L10 | 672 | 444 | 5.5476 | 10.7125 | -1.2160 | -1.5047 | 0.4234 | 0.3626 |
| 4 | joint_restore_object_only:L10->L20 | 672 | 444 | 4.3587 | 9.8541 | -0.0272 | -2.3631 | 0.5315 | 0.3131 |
| 5 | frame_only_matched:L10->L20 | 672 | 444 | 3.7483 | 8.9200 | 0.5832 | -3.2972 | 0.5788 | 0.2613 |
| 6 | joint_mismatched_frame:L4->L10 | 672 | 444 | 9.3703 | 9.2630 | -5.0387 | -2.9542 | 0.0901 | 0.2500 |
| 7 | joint_mismatched_frame:L10->L20 | 672 | 444 | 7.1479 | 7.2876 | -2.8164 | -4.9296 | 0.2793 | 0.1509 |
| 8 | frame_only_matched:L4->L10 | 672 | 444 | 2.8740 | 7.3812 | 1.4575 | -4.8360 | 0.6892 | 0.1396 |
| 9 | joint_restore_frame_only:L4->L10 | 672 | 444 | 7.0472 | 6.6925 | -2.7156 | -5.5247 | 0.2793 | 0.1284 |
| 10 | object_only_matched:L4->L10 | 672 | 444 | 8.2350 | 6.4682 | -3.9035 | -5.7489 | 0.1959 | 0.1149 |
| 11 | joint_restore_frame_only:L10->L20 | 672 | 444 | 3.3267 | 4.6356 | 1.0048 | -7.5816 | 0.6306 | 0.0721 |
| 12 | object_only_matched:L10->L20 | 672 | 444 | 3.4809 | 4.2158 | 0.8506 | -8.0013 | 0.6081 | 0.0698 |
| 13 | joint_restore_both:L4->L10 | 672 | 444 | 1.2137 | 1.9298 | 3.1178 | -10.2874 | 0.8536 | 0.0135 |
| 14 | joint_restore_both:L10->L20 | 672 | 444 | 0.3134 | 0.6463 | 4.0182 | -11.5709 | 0.9459 | 0.0090 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:property | 192 | 98 | 9.5283 | 11.6585 | -6.9272 | 3.4984 | 0.0510 | 0.8469 |
| 2 | joint_matched:can_do | 192 | 144 | 9.4343 | 17.2041 | -5.7449 | 3.0727 | 0.0903 | 0.7361 |
| 3 | joint_matched:location | 192 | 102 | 9.4483 | 11.6492 | -6.7096 | 2.8769 | 0.0392 | 0.7353 |
| 4 | joint_matched:used_for | 192 | 146 | 12.2855 | 16.7086 | -6.2938 | 1.9423 | 0.0685 | 0.6986 |
| 5 | joint_matched:is_a | 192 | 148 | 11.1705 | 15.2017 | -5.7505 | 1.7695 | 0.1149 | 0.6419 |
| 6 | joint_restore_object_only:property | 192 | 98 | 4.4820 | 8.3886 | -1.8809 | 0.2286 | 0.3163 | 0.5000 |
| 7 | joint_restore_object_only:can_do | 192 | 144 | 5.1537 | 13.7387 | -1.4644 | -0.3927 | 0.3750 | 0.4931 |
| 8 | joint_matched:part_of | 192 | 142 | 9.1177 | 12.6077 | -4.1512 | 0.1382 | 0.2254 | 0.4789 |
| 9 | joint_restore_object_only:location | 192 | 102 | 4.2860 | 7.6346 | -1.5473 | -1.1376 | 0.3922 | 0.4412 |
| 10 | joint_matched:material | 192 | 108 | 7.5995 | 10.9268 | -3.9080 | -0.2303 | 0.1759 | 0.4352 |
| 11 | joint_mismatched_frame:can_do | 192 | 144 | 8.5740 | 11.6968 | -4.8847 | -2.4346 | 0.0903 | 0.3264 |
| 12 | frame_only_matched:can_do | 192 | 144 | 3.3729 | 11.5193 | 0.3165 | -2.6121 | 0.5139 | 0.3194 |
| 13 | joint_restore_object_only:is_a | 192 | 148 | 5.9420 | 11.5238 | -0.5220 | -1.9083 | 0.5000 | 0.2973 |
| 14 | joint_restore_object_only:used_for | 192 | 146 | 6.1192 | 12.2313 | -0.1274 | -2.5351 | 0.5479 | 0.2945 |
| 15 | frame_only_matched:property | 192 | 98 | 2.8050 | 5.5439 | -0.2039 | -2.6162 | 0.4490 | 0.2755 |
| 16 | frame_only_matched:location | 192 | 102 | 2.8519 | 5.5066 | -0.1132 | -3.2657 | 0.5294 | 0.2451 |
| 17 | joint_restore_frame_only:property | 192 | 98 | 5.8062 | 5.4845 | -3.2051 | -2.6755 | 0.1939 | 0.2347 |
| 18 | joint_restore_object_only:part_of | 192 | 142 | 4.2772 | 8.7386 | 0.6893 | -3.7309 | 0.6127 | 0.2324 |
| 19 | joint_mismatched_frame:is_a | 192 | 148 | 9.0190 | 9.5179 | -3.5990 | -3.9143 | 0.2027 | 0.2162 |
| 20 | joint_restore_frame_only:location | 192 | 102 | 3.7942 | 4.8467 | -1.0556 | -3.9255 | 0.4216 | 0.2157 |
| 21 | object_only_matched:property | 192 | 98 | 6.0454 | 5.3342 | -3.4443 | -2.8259 | 0.2245 | 0.2143 |
| 22 | object_only_matched:location | 192 | 102 | 4.8145 | 4.4622 | -2.0758 | -4.3101 | 0.3137 | 0.1961 |
| 23 | joint_mismatched_frame:property | 192 | 98 | 7.7031 | 4.6441 | -5.1020 | -3.5159 | 0.1020 | 0.1939 |
| 24 | frame_only_matched:used_for | 192 | 146 | 5.0154 | 10.3392 | 0.9764 | -4.4272 | 0.6644 | 0.1918 |
| 25 | joint_mismatched_frame:location | 192 | 102 | 6.5440 | 5.1928 | -3.8053 | -3.5794 | 0.1863 | 0.1863 |
| 26 | joint_mismatched_frame:used_for | 192 | 146 | 11.0383 | 10.5710 | -5.0466 | -4.1954 | 0.1301 | 0.1644 |
| 27 | frame_only_matched:is_a | 192 | 148 | 3.9480 | 9.4115 | 1.4720 | -4.0206 | 0.7162 | 0.1622 |
| 28 | joint_mismatched_frame:material | 192 | 108 | 5.6043 | 6.5744 | -1.9128 | -4.5827 | 0.3241 | 0.1481 |
| 29 | joint_mismatched_frame:part_of | 192 | 142 | 7.9250 | 7.1639 | -2.9585 | -5.3057 | 0.2676 | 0.1479 |
| 30 | joint_restore_frame_only:is_a | 192 | 148 | 6.2431 | 5.6777 | -0.8231 | -7.7544 | 0.4257 | 0.1419 |
| 31 | joint_restore_object_only:material | 192 | 108 | 3.7005 | 7.5945 | -0.0090 | -3.5626 | 0.5370 | 0.1389 |
| 32 | frame_only_matched:part_of | 192 | 142 | 2.4041 | 6.6736 | 2.5624 | -5.7959 | 0.8099 | 0.1268 |
| 33 | object_only_matched:is_a | 192 | 148 | 7.2968 | 5.7455 | -1.8768 | -7.6866 | 0.3919 | 0.1216 |
| 34 | frame_only_matched:material | 192 | 108 | 2.1382 | 5.7770 | 1.5533 | -5.3800 | 0.6759 | 0.0926 |
| 35 | object_only_matched:part_of | 192 | 142 | 5.7701 | 5.3863 | -0.8036 | -7.0833 | 0.4507 | 0.0845 |
| 36 | joint_restore_frame_only:part_of | 192 | 142 | 5.7319 | 5.7737 | -0.7654 | -6.6959 | 0.4648 | 0.0775 |
| 37 | joint_restore_frame_only:material | 192 | 108 | 3.4987 | 4.3142 | 0.1928 | -6.8429 | 0.6111 | 0.0648 |
| 38 | object_only_matched:material | 192 | 108 | 3.7406 | 3.6335 | -0.0491 | -7.5235 | 0.5648 | 0.0463 |
| 39 | object_only_matched:used_for | 192 | 146 | 7.8785 | 8.2923 | -1.8868 | -6.4740 | 0.3836 | 0.0411 |
| 40 | joint_restore_both:property | 192 | 98 | 0.2756 | 0.8337 | 2.3255 | -7.3264 | 0.8571 | 0.0408 |
| 41 | joint_restore_frame_only:used_for | 192 | 146 | 7.0361 | 8.3019 | -1.0444 | -6.4645 | 0.4932 | 0.0342 |
| 42 | joint_restore_both:part_of | 192 | 142 | 1.2431 | 1.6444 | 3.7234 | -10.8252 | 0.8803 | 0.0211 |
| 43 | joint_restore_both:location | 192 | 102 | 0.3558 | 1.4069 | 2.3829 | -7.3654 | 0.8824 | 0.0196 |
| 44 | joint_restore_both:material | 192 | 108 | 0.7895 | 1.3497 | 2.9020 | -9.8074 | 0.8796 | 0.0093 |
| 45 | joint_restore_frame_only:can_do | 192 | 144 | 3.5204 | 4.5809 | 0.1689 | -9.5505 | 0.5208 | 0.0000 |
| 46 | object_only_matched:can_do | 192 | 144 | 4.6169 | 3.8025 | -0.9276 | -10.3289 | 0.4444 | 0.0000 |
| 47 | joint_restore_both:can_do | 192 | 144 | 0.7285 | 1.3647 | 2.9608 | -12.7667 | 0.9028 | 0.0000 |
| 48 | joint_restore_both:used_for | 192 | 146 | 0.6015 | 1.2614 | 5.3903 | -13.5050 | 0.9658 | 0.0000 |
| 49 | joint_restore_both:is_a | 192 | 148 | 1.0824 | 1.0718 | 4.3375 | -12.3603 | 0.9054 | 0.0000 |

## deepseek7b

objects=24, items=672, rows=9408, layer_pairs=[[8, 10], [12, 14]]
relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched | 1344 | 572 | 7.1563 | 12.1629 | -3.2163 | -1.3271 | 0.2535 | 0.4161 |
| 2 | joint_restore_object_only | 1344 | 572 | 4.2654 | 9.3509 | -0.3253 | -4.1391 | 0.4948 | 0.2395 |
| 3 | frame_only_matched | 1344 | 572 | 3.7984 | 8.7878 | 0.1417 | -4.7022 | 0.5367 | 0.2133 |
| 4 | joint_mismatched_frame | 1344 | 572 | 6.2317 | 7.2367 | -2.2916 | -6.2533 | 0.3164 | 0.1119 |
| 5 | object_only_matched | 1344 | 572 | 3.6021 | 3.4196 | 0.3380 | -10.0704 | 0.5455 | 0.0262 |
| 6 | joint_restore_frame_only | 1344 | 572 | 3.2990 | 3.4156 | 0.6410 | -10.0744 | 0.5892 | 0.0245 |
| 7 | joint_restore_both | 1344 | 572 | 0.4736 | 0.5091 | 3.4665 | -12.9809 | 0.8794 | 0.0087 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:L8->L10 | 672 | 286 | 7.5145 | 12.3939 | -3.5745 | -1.0962 | 0.2308 | 0.4371 |
| 2 | joint_matched:L12->L14 | 672 | 286 | 6.7981 | 11.9320 | -2.8581 | -1.5581 | 0.2762 | 0.3951 |
| 3 | joint_restore_object_only:L8->L10 | 672 | 286 | 4.5856 | 9.7373 | -0.6455 | -3.7527 | 0.4720 | 0.2552 |
| 4 | joint_restore_object_only:L12->L14 | 672 | 286 | 3.9452 | 8.9646 | -0.0051 | -4.5254 | 0.5175 | 0.2238 |
| 5 | frame_only_matched:L12->L14 | 672 | 286 | 3.8441 | 8.9310 | 0.0960 | -4.5590 | 0.5315 | 0.2168 |
| 6 | frame_only_matched:L8->L10 | 672 | 286 | 3.7526 | 8.6447 | 0.1875 | -4.8453 | 0.5420 | 0.2098 |
| 7 | joint_mismatched_frame:L12->L14 | 672 | 286 | 5.9015 | 7.0251 | -1.9614 | -6.4649 | 0.3322 | 0.1154 |
| 8 | joint_mismatched_frame:L8->L10 | 672 | 286 | 6.5619 | 7.4483 | -2.6219 | -6.0417 | 0.3007 | 0.1084 |
| 9 | object_only_matched:L8->L10 | 672 | 286 | 4.2288 | 3.7607 | -0.2887 | -9.7293 | 0.4720 | 0.0315 |
| 10 | joint_restore_frame_only:L8->L10 | 672 | 286 | 3.6855 | 3.5635 | 0.2545 | -9.9265 | 0.5420 | 0.0315 |
| 11 | object_only_matched:L12->L14 | 672 | 286 | 2.9754 | 3.0785 | 0.9646 | -10.4115 | 0.6189 | 0.0210 |
| 12 | joint_restore_frame_only:L12->L14 | 672 | 286 | 2.9125 | 3.2678 | 1.0275 | -10.2222 | 0.6364 | 0.0175 |
| 13 | joint_restore_both:L8->L10 | 672 | 286 | 0.8265 | 0.8215 | 3.1136 | -12.6685 | 0.8287 | 0.0105 |
| 14 | joint_restore_both:L12->L14 | 672 | 286 | 0.1207 | 0.1966 | 3.8194 | -13.2934 | 0.9301 | 0.0070 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_matched:can_do | 192 | 122 | 8.7604 | 17.0136 | -4.3530 | 1.5211 | 0.1967 | 0.6311 |
| 2 | joint_matched:used_for | 192 | 74 | 9.5306 | 15.3395 | -4.5645 | -0.6467 | 0.1486 | 0.5135 |
| 3 | joint_matched:part_of | 192 | 74 | 6.7018 | 13.4518 | -2.9051 | -2.1591 | 0.2838 | 0.4459 |
| 4 | joint_matched:property | 192 | 70 | 5.5004 | 8.4085 | -2.2044 | -0.5506 | 0.3286 | 0.4143 |
| 5 | joint_restore_object_only:can_do | 192 | 122 | 6.0274 | 13.7362 | -1.6200 | -1.7563 | 0.4344 | 0.3852 |
| 6 | joint_matched:location | 192 | 26 | 7.1816 | 6.8928 | -4.2031 | -3.2257 | 0.3077 | 0.3462 |
| 7 | frame_only_matched:can_do | 192 | 122 | 5.4445 | 13.1448 | -1.0371 | -2.3476 | 0.4590 | 0.3361 |
| 8 | joint_restore_object_only:location | 192 | 26 | 3.5974 | 4.9817 | -0.6189 | -5.1368 | 0.3846 | 0.3077 |
| 9 | joint_restore_object_only:used_for | 192 | 74 | 5.0834 | 11.4137 | -0.1173 | -4.5726 | 0.5405 | 0.2838 |
| 10 | joint_restore_object_only:part_of | 192 | 74 | 3.4742 | 10.7528 | 0.3224 | -4.8581 | 0.5405 | 0.2838 |
| 11 | joint_matched:is_a | 192 | 128 | 6.3044 | 10.2713 | -2.7910 | -2.8333 | 0.2266 | 0.2656 |
| 12 | frame_only_matched:used_for | 192 | 74 | 4.6032 | 11.0205 | 0.3629 | -4.9658 | 0.5541 | 0.2568 |
| 13 | frame_only_matched:part_of | 192 | 74 | 3.0654 | 10.2873 | 0.7312 | -5.3236 | 0.5946 | 0.2568 |
| 14 | joint_mismatched_frame:can_do | 192 | 122 | 7.4703 | 11.8784 | -3.0629 | -3.6141 | 0.1967 | 0.2541 |
| 15 | joint_matched:material | 192 | 78 | 5.7019 | 8.5696 | -1.7314 | -3.2303 | 0.3718 | 0.2308 |
| 16 | joint_restore_object_only:property | 192 | 70 | 3.7137 | 6.5695 | -0.4177 | -2.3896 | 0.5000 | 0.2286 |
| 17 | frame_only_matched:property | 192 | 70 | 3.2338 | 5.5430 | 0.0623 | -3.4161 | 0.5143 | 0.2143 |
| 18 | frame_only_matched:location | 192 | 26 | 3.0951 | 4.4486 | -0.1166 | -5.6699 | 0.4615 | 0.1923 |
| 19 | joint_mismatched_frame:used_for | 192 | 74 | 8.0688 | 9.9995 | -3.1027 | -5.9868 | 0.2703 | 0.1351 |
| 20 | joint_restore_object_only:is_a | 192 | 128 | 3.3492 | 7.4507 | 0.1641 | -5.6540 | 0.5078 | 0.1250 |
| 21 | frame_only_matched:is_a | 192 | 128 | 3.0670 | 7.0100 | 0.4463 | -6.0947 | 0.5469 | 0.1172 |
| 22 | joint_mismatched_frame:part_of | 192 | 74 | 6.5475 | 7.8084 | -2.7509 | -7.8025 | 0.3108 | 0.1081 |
| 23 | joint_restore_object_only:material | 192 | 78 | 3.7051 | 6.2759 | 0.2654 | -5.5240 | 0.5128 | 0.1026 |
| 24 | frame_only_matched:material | 192 | 78 | 3.0966 | 5.7082 | 0.8739 | -6.0917 | 0.6154 | 0.1026 |
| 25 | object_only_matched:property | 192 | 70 | 2.4110 | 2.1850 | 0.8851 | -6.7741 | 0.6143 | 0.0857 |
| 26 | joint_restore_frame_only:location | 192 | 26 | 3.2763 | 1.8610 | -0.2978 | -8.2575 | 0.4231 | 0.0769 |
| 27 | object_only_matched:location | 192 | 26 | 3.6190 | 1.7555 | -0.6404 | -8.3630 | 0.3846 | 0.0769 |
| 28 | joint_mismatched_frame:property | 192 | 70 | 5.0759 | 3.2533 | -1.7798 | -5.7058 | 0.4286 | 0.0714 |
| 29 | joint_mismatched_frame:material | 192 | 78 | 4.4409 | 4.6842 | -0.4704 | -7.1157 | 0.4615 | 0.0641 |
| 30 | joint_restore_frame_only:property | 192 | 70 | 2.0332 | 2.0786 | 1.2629 | -6.8805 | 0.7000 | 0.0571 |
| 31 | joint_restore_both:property | 192 | 70 | 0.4992 | 0.3977 | 2.7968 | -8.5614 | 0.8429 | 0.0571 |
| 32 | joint_mismatched_frame:location | 192 | 26 | 4.4672 | 1.0883 | -1.4886 | -9.0302 | 0.3846 | 0.0385 |
| 33 | joint_mismatched_frame:is_a | 192 | 128 | 5.8882 | 5.8675 | -2.3749 | -7.2372 | 0.2969 | 0.0312 |
| 34 | object_only_matched:used_for | 192 | 74 | 5.9874 | 6.0022 | -1.0213 | -9.9841 | 0.3919 | 0.0270 |
| 35 | joint_restore_frame_only:used_for | 192 | 74 | 5.6819 | 5.9521 | -0.7158 | -10.0342 | 0.4189 | 0.0270 |
| 36 | joint_restore_frame_only:part_of | 192 | 74 | 3.8169 | 3.3491 | -0.0202 | -12.2618 | 0.5676 | 0.0270 |
| 37 | object_only_matched:part_of | 192 | 74 | 3.7562 | 3.2554 | 0.0405 | -12.3556 | 0.5676 | 0.0270 |
| 38 | object_only_matched:material | 192 | 78 | 2.3859 | 2.8902 | 1.5846 | -8.9097 | 0.7051 | 0.0256 |
| 39 | joint_restore_frame_only:material | 192 | 78 | 2.0000 | 2.7716 | 1.9705 | -9.0283 | 0.7436 | 0.0256 |
| 40 | joint_restore_frame_only:can_do | 192 | 122 | 2.8829 | 3.6262 | 1.5245 | -11.8662 | 0.6148 | 0.0082 |
| 41 | object_only_matched:is_a | 192 | 128 | 3.9621 | 3.2672 | -0.4487 | -9.8374 | 0.4531 | 0.0078 |
| 42 | joint_restore_frame_only:is_a | 192 | 128 | 3.5072 | 3.2264 | 0.0062 | -9.8783 | 0.5547 | 0.0078 |
| 43 | joint_restore_both:is_a | 192 | 128 | 0.2864 | 0.3552 | 3.2269 | -12.7495 | 0.8438 | 0.0078 |
| 44 | object_only_matched:can_do | 192 | 122 | 3.1416 | 3.5142 | 1.2658 | -11.9783 | 0.6148 | 0.0000 |
| 45 | joint_restore_both:material | 192 | 78 | 0.5001 | 0.8047 | 3.4704 | -10.9952 | 0.9103 | 0.0000 |
| 46 | joint_restore_both:part_of | 192 | 74 | 0.8223 | 0.6714 | 2.9743 | -14.9395 | 0.8514 | 0.0000 |
| 47 | joint_restore_both:can_do | 192 | 122 | 0.5676 | 0.5610 | 3.8398 | -14.9315 | 0.9098 | 0.0000 |
| 48 | joint_restore_both:used_for | 192 | 74 | 0.3733 | 0.4930 | 4.5928 | -15.4933 | 0.9324 | 0.0000 |
| 49 | joint_restore_both:location | 192 | 26 | 0.0983 | 0.0195 | 2.8802 | -10.0990 | 0.8462 | 0.0000 |

