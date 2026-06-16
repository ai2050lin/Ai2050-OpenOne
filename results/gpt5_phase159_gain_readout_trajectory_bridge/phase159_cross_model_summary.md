# Phase159 Cross-model Summary

## qwen3

cases=300, mean_hit=0.3592, top_traj=correct_surface:117

| metric | mean | corr_with_hit |
|---|---:|---:|
| dcf_mean | 2.7414 | 0.5689 |
| dcf_delta | -0.2212 | 0.4822 |
| dcf_max_delta | -0.4935 | 0.5005 |
| proj_q_over_rms | 0.8376 | 0.1855 |
| cos_v_q | 0.0192 | 0.1865 |
| target_delta | 0.0979 | 0.4283 |
| competitor_delta | 0.3191 | -0.2210 |
| step1_margin | 0.2759 | 0.8341 |
| step2_margin | -7.0461 | -0.2791 |
| step3_margin | -4.9833 | 0.3978 |

### by category

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| action | 30 | 0.1667 | 2.0954 | -0.3307 | 1.1360 | -0.9652 | -6.5519 | -5.4316 | other:10 |
| animal | 30 | 0.4583 | 4.5808 | 0.1510 | 1.1614 | 1.1044 | -7.9744 | -5.0202 | correct_surface:17 |
| clothing | 30 | 0.3667 | 3.3240 | 0.5994 | 1.4530 | 0.2864 | -6.8956 | -5.1411 | correct_surface:14 |
| container | 30 | 0.3417 | 1.2108 | -0.5835 | 0.6184 | -0.0596 | -6.4232 | -4.7019 | correct_surface:11 |
| emotion | 30 | 0.3708 | 3.4029 | -0.6177 | 0.1338 | 0.8191 | -7.6473 | -4.5678 | correct_surface:13 |
| fruit | 30 | 0.4500 | 3.3049 | 0.6679 | 1.5747 | 0.4896 | -8.2944 | -5.6803 | correct_surface:12 |
| furniture | 30 | 0.3708 | 0.7229 | 0.1334 | 1.0702 | 0.2022 | -7.3349 | -5.2191 | correct_surface:13 |
| number | 30 | 0.2167 | 2.2030 | -2.0893 | -0.8451 | -0.5136 | -5.8291 | -5.3126 | other:12 |
| plant | 30 | 0.5208 | 5.4174 | 1.1814 | 1.9946 | 1.5635 | -7.3575 | -4.5447 | correct_surface:15 |
| time | 30 | 0.3292 | 1.1513 | -1.3237 | 0.0787 | -0.1679 | -6.1527 | -4.2140 | correct_surface:12 |

### by format

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| answer_one_word | 60 | 0.1583 | 2.2353 | -0.7202 | 2.3932 | -0.9404 | -6.7738 | -5.4249 | other:27 |
| label_colon | 60 | 0.3438 | 3.1720 | -0.1439 | 0.2247 | 0.2924 | -4.6978 | -4.8698 | correct_surface:21 |
| list_answer | 60 | 0.2167 | 1.5648 | 0.3673 | 1.5025 | -0.7818 | -4.1495 | -5.6289 | object_copy_trap:15 |
| multiple_choice | 60 | 0.9187 | 3.8782 | 0.0497 | 0.5363 | 3.5858 | -10.0103 | -2.3905 | correct_surface:58 |
| quoted_answer | 60 | 0.1583 | 2.8564 | -0.6588 | -0.4688 | -0.7766 | -9.5990 | -6.6025 | fragment_trap:21 |

### by family

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| long | 100 | 0.2475 | 2.1857 | -0.7768 | -0.6419 | 0.3194 | -8.5941 | -3.9484 | fragment_trap:59 |
| neutral | 100 | 0.2475 | 1.6022 | -1.3604 | 0.6291 | -0.8424 | -5.8346 | -6.7611 | other:37 |
| short | 100 | 0.5825 | 4.4361 | 1.4736 | 2.5255 | 1.3506 | -6.7096 | -4.2405 | correct_surface:69 |

### by split

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| back_front | 150 | 0.3783 | 2.7885 | -0.2380 | 0.8665 | 0.3568 | -6.9050 | -4.8962 | correct_surface:62 |
| front_back | 150 | 0.3400 | 2.6942 | -0.2044 | 0.8087 | 0.1950 | -7.1872 | -5.0705 | correct_surface:55 |

### by tc_mode

| mode | n | hit | dcf_delta | competitor_delta | top_traj |
|---|---:|---:|---:|---:|---|
| competitor_dominant | 39 | 0.6763 | 1.2557 | -1.1337 | correct_surface:31 |
| competitor_release | 80 | 0.2578 | -1.0927 | 1.0926 | other:33 |
| target_dominant | 181 | 0.3356 | -0.1542 | 0.2903 | correct_surface:63 |

## glm4

cases=300, mean_hit=0.2608, top_traj=fragment_trap:144

| metric | mean | corr_with_hit |
|---|---:|---:|
| dcf_mean | 2.1193 | 0.4848 |
| dcf_delta | -0.3938 | 0.0584 |
| dcf_max_delta | -0.4849 | 0.0638 |
| proj_q_over_rms | -0.8093 | -0.0622 |
| cos_v_q | -0.0131 | -0.1385 |
| target_delta | -0.5713 | -0.0294 |
| competitor_delta | -0.1775 | -0.1052 |
| step1_margin | 0.1068 | 0.7073 |
| step2_margin | -5.2933 | 0.4544 |
| step3_margin | -4.5850 | 0.4435 |

### by category

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| action | 30 | 0.1250 | 1.6942 | -0.3390 | -0.1645 | -1.1477 | -5.9764 | -5.2368 | fragment_trap:14 |
| animal | 30 | 0.4042 | 2.7736 | -0.3482 | -1.6327 | 1.3701 | -5.3032 | -4.8799 | correct_surface:13 |
| clothing | 30 | 0.2083 | 2.6994 | -0.1122 | -0.7017 | 0.2574 | -5.7291 | -5.1131 | fragment_trap:13 |
| container | 30 | 0.2542 | 1.1086 | -0.7956 | -1.2575 | -0.2805 | -4.5733 | -4.6857 | fragment_trap:13 |
| emotion | 30 | 0.2000 | 3.3928 | -0.5954 | -0.0990 | 0.3509 | -4.5539 | -3.5053 | fragment_trap:19 |
| fruit | 30 | 0.3417 | 2.1207 | -1.1399 | -2.3573 | 1.2395 | -4.8698 | -3.9915 | fragment_trap:16 |
| furniture | 30 | 0.2833 | 0.2061 | -0.7550 | -1.1601 | -0.2271 | -5.6919 | -4.9495 | fragment_trap:13 |
| number | 30 | 0.1625 | 2.2874 | 0.0566 | 0.0537 | -0.6566 | -5.7613 | -4.9495 | fragment_trap:15 |
| plant | 30 | 0.3500 | 3.6944 | -0.3180 | -1.1984 | 0.4698 | -4.5880 | -3.8661 | fragment_trap:18 |
| time | 30 | 0.2792 | 1.2156 | 0.4088 | 0.4248 | -0.3076 | -5.8863 | -4.6726 | fragment_trap:11 |

### by format

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| answer_one_word | 60 | 0.0813 | 1.4798 | -0.5118 | -3.1812 | -0.5858 | -10.0823 | -3.9613 | fragment_trap:44 |
| label_colon | 60 | 0.2458 | 2.7966 | -0.4743 | -1.2278 | 1.0145 | -4.3360 | -4.6494 | fragment_trap:36 |
| list_answer | 60 | 0.1667 | 1.5261 | -0.0515 | -0.0314 | -1.3554 | -4.9676 | -6.0893 | other:19 |
| multiple_choice | 60 | 0.7104 | 3.3677 | -0.9804 | -1.9469 | 2.6548 | -2.4800 | -2.6818 | correct_surface:51 |
| quoted_answer | 60 | 0.1000 | 1.4262 | 0.0490 | 2.3410 | -1.1939 | -4.6006 | -5.5433 | fragment_trap:43 |

### by family

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| long | 100 | 0.1938 | 1.9109 | -0.6022 | -1.1009 | -0.3593 | -5.5577 | -4.6685 | fragment_trap:68 |
| neutral | 100 | 0.1913 | 1.2219 | -1.2912 | -1.9291 | -0.3493 | -6.3102 | -5.3697 | fragment_trap:46 |
| short | 100 | 0.3975 | 3.2250 | 0.7119 | 0.6022 | 1.0291 | -4.0121 | -3.7169 | correct_surface:45 |

### by split

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| back_front | 150 | 0.2650 | 2.1354 | -0.3576 | -0.8319 | 0.0766 | -5.2688 | -4.4735 | fragment_trap:69 |
| front_back | 150 | 0.2567 | 2.1031 | -0.4300 | -0.7867 | 0.1371 | -5.3178 | -4.6966 | fragment_trap:75 |

### by tc_mode

| mode | n | hit | dcf_delta | competitor_delta | top_traj |
|---|---:|---:|---:|---:|---|
| competitor_dominant | 38 | 0.4539 | 0.6371 | -1.7996 | correct_surface:18 |
| competitor_release | 33 | 0.2386 | -0.7506 | 1.5334 | fragment_trap:14 |
| target_dominant | 229 | 0.2320 | -0.5134 | -0.1549 | fragment_trap:122 |

## deepseek7b

cases=300, mean_hit=0.1925, top_traj=fragment_trap:112

| metric | mean | corr_with_hit |
|---|---:|---:|
| dcf_mean | 1.1347 | 0.2059 |
| dcf_delta | -0.1196 | 0.0762 |
| dcf_max_delta | -0.1134 | 0.0296 |
| proj_q_over_rms | 1.4160 | -0.1300 |
| cos_v_q | 0.0202 | -0.0746 |
| target_delta | 0.0583 | 0.0452 |
| competitor_delta | 0.1779 | 0.0000 |
| step1_margin | -1.2589 | 0.6174 |
| step2_margin | -6.7660 | -0.0867 |
| step3_margin | -4.4193 | 0.6238 |

### by category

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| action | 30 | 0.1042 | 1.9803 | -0.1437 | 0.2747 | -2.3328 | -7.0474 | -5.0528 | fragment_trap:10 |
| animal | 30 | 0.2417 | 2.4701 | -0.0642 | 2.2763 | 0.0904 | -6.8211 | -4.5442 | fragment_trap:11 |
| clothing | 30 | 0.2125 | 1.5075 | -0.4560 | 1.2546 | -1.4572 | -6.6927 | -5.6231 | fragment_trap:13 |
| container | 30 | 0.2292 | 0.1284 | 0.4876 | 2.2644 | -1.5208 | -6.4195 | -3.8807 | fragment_trap:12 |
| emotion | 30 | 0.1500 | 0.5989 | -0.4525 | 0.9352 | -1.1951 | -6.7574 | -3.6713 | fragment_trap:11 |
| fruit | 30 | 0.2083 | 0.4443 | 0.3914 | 3.0486 | -0.9539 | -7.5740 | -4.6175 | fragment_trap:10 |
| furniture | 30 | 0.1750 | -2.0985 | -0.7145 | 1.4303 | -1.8354 | -6.7848 | -4.6437 | fragment_trap:13 |
| number | 30 | 0.1625 | 1.7310 | 0.0278 | -0.0277 | -1.1719 | -6.3429 | -4.1934 | fragment_trap:8 |
| plant | 30 | 0.3167 | 3.4907 | 0.1587 | 1.5573 | -0.2190 | -6.3798 | -3.5280 | fragment_trap:13 |
| time | 30 | 0.1250 | 1.0940 | -0.4307 | 1.1465 | -1.9930 | -6.8402 | -4.4384 | fragment_trap:11 |

### by format

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| answer_one_word | 60 | 0.0417 | 0.8061 | -0.0685 | 0.4226 | -1.8490 | -5.4470 | -5.6812 | fragment_trap:30 |
| label_colon | 60 | 0.0437 | 1.3192 | 0.3424 | 2.0491 | -1.1118 | -7.2740 | -5.0801 | fragment_trap:40 |
| list_answer | 60 | 0.0792 | 0.7161 | -0.1213 | 4.9975 | -1.7154 | -5.3643 | -4.8814 | other:21 |
| multiple_choice | 60 | 0.7521 | 1.3953 | -0.2316 | 0.0621 | 0.5052 | -7.3901 | -0.7488 | correct_surface:49 |
| quoted_answer | 60 | 0.0458 | 1.4366 | -0.5192 | -0.4513 | -2.1234 | -8.3545 | -5.7050 | fragment_trap:40 |

### by family

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| long | 100 | 0.1825 | 1.4412 | 0.1869 | 0.6609 | -2.0541 | -7.7007 | -4.3470 | object_copy_trap:31 |
| neutral | 100 | 0.1650 | 0.8084 | -0.4459 | -0.2115 | -1.3530 | -6.6543 | -5.1749 | fragment_trap:44 |
| short | 100 | 0.2300 | 1.1544 | -0.0999 | 3.7986 | -0.3695 | -5.9430 | -3.7360 | fragment_trap:43 |

### by split

| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| back_front | 150 | 0.1942 | 1.1856 | -0.1371 | 1.3941 | -1.2397 | -6.7127 | -4.4160 | fragment_trap:53 |
| front_back | 150 | 0.1908 | 1.0837 | -0.1021 | 1.4380 | -1.2781 | -6.8192 | -4.4227 | fragment_trap:59 |

### by tc_mode

| mode | n | hit | dcf_delta | competitor_delta | top_traj |
|---|---:|---:|---:|---:|---|
| competitor_dominant | 39 | 0.1442 | 1.1385 | -2.0152 | fragment_trap:16 |
| competitor_release | 73 | 0.1199 | -0.7502 | 1.5628 | other:25 |
| target_dominant | 188 | 0.2307 | -0.1358 | 0.0950 | fragment_trap:73 |

## cross_model

cases=900, mean_hit=0.2708

| metric | mean | corr_with_hit |
|---|---:|---:|
| dcf_mean | 1.9984 | 0.4690 |
| dcf_delta | -0.2449 | 0.2553 |
| proj_q_over_rms | 0.4814 | -0.0084 |
| step1_margin | -0.2921 | 0.7369 |
| step2_margin | -6.3685 | 0.0019 |
| step3_margin | -4.6626 | 0.4516 |

