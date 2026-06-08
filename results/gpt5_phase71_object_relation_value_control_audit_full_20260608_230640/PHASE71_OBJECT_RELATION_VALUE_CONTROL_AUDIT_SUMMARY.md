# Phase71 Object-Relation-Value Control Audit Summary

## qwen3

items=342, rows=7560, layer_pairs=[[4, 8], [8, 12], [8, 16]], controls=['mismatch_object', 'same_target_object', 'random_same_norm', 'same_prompt_last']

### By control

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mismatch_object | 2052 | 1794 | 8.4494 | 7.6817 | 0.7676 | 0.2865 | 0.9459 |
| 2 | same_prompt_last | 2052 | 1794 | 4.7106 | 4.2398 | 0.4708 | 0.5886 | 0.9766 |
| 3 | random_same_norm | 2052 | 1794 | 4.3578 | 3.9043 | 0.4535 | 0.6338 | 0.9788 |
| 4 | same_target_object | 1404 | 1248 | 0.1544 | 0.2979 | -0.1435 | 0.9495 | 0.9984 |

### Top control-paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mismatch_object:L4->L8:object_last | 342 | 299 | 9.8621 | 8.7762 | 1.0859 | 0.1773 | 0.8930 |
| 2 | mismatch_object:L4->L8:object_first | 342 | 299 | 9.3580 | 8.2771 | 1.0809 | 0.1973 | 0.8963 |
| 3 | mismatch_object:L8->L12:object_last | 342 | 299 | 8.1869 | 7.7148 | 0.4721 | 0.3110 | 0.9666 |
| 4 | mismatch_object:L8->L16:object_last | 342 | 299 | 8.1869 | 7.4206 | 0.7663 | 0.3110 | 0.9766 |
| 5 | mismatch_object:L8->L12:object_first | 342 | 299 | 7.5512 | 7.0940 | 0.4573 | 0.3612 | 0.9666 |
| 6 | mismatch_object:L8->L16:object_first | 342 | 299 | 7.5512 | 6.8079 | 0.7433 | 0.3612 | 0.9766 |
| 7 | same_prompt_last:L4->L8:object_last | 342 | 299 | 6.7938 | 6.2145 | 0.5793 | 0.2809 | 0.9498 |
| 8 | random_same_norm:L4->L8:object_last | 342 | 299 | 6.6615 | 6.1037 | 0.5578 | 0.3278 | 0.9532 |
| 9 | same_prompt_last:L4->L8:object_first | 342 | 299 | 6.5070 | 5.9586 | 0.5484 | 0.3077 | 0.9498 |
| 10 | random_same_norm:L4->L8:object_first | 342 | 299 | 6.3302 | 5.7920 | 0.5381 | 0.3445 | 0.9532 |
| 11 | same_prompt_last:L8->L12:object_last | 342 | 299 | 3.8661 | 3.6162 | 0.2499 | 0.7324 | 0.9900 |
| 12 | same_prompt_last:L8->L16:object_last | 342 | 299 | 3.8661 | 3.2201 | 0.6460 | 0.7324 | 0.9900 |
| 13 | same_prompt_last:L8->L12:object_first | 342 | 299 | 3.6153 | 3.3911 | 0.2242 | 0.7391 | 0.9900 |
| 14 | same_prompt_last:L8->L16:object_first | 342 | 299 | 3.6153 | 3.0380 | 0.5772 | 0.7391 | 0.9900 |
| 15 | random_same_norm:L8->L12:object_last | 342 | 299 | 3.4550 | 3.2203 | 0.2346 | 0.7759 | 0.9933 |
| 16 | random_same_norm:L8->L16:object_last | 342 | 299 | 3.3752 | 2.7559 | 0.6194 | 0.7759 | 0.9900 |
| 17 | random_same_norm:L8->L12:object_first | 342 | 299 | 3.1935 | 2.9722 | 0.2213 | 0.7893 | 0.9933 |
| 18 | random_same_norm:L8->L16:object_first | 342 | 299 | 3.1313 | 2.5817 | 0.5495 | 0.7893 | 0.9900 |
| 19 | same_target_object:L4->L8:object_first | 234 | 208 | 0.5177 | 0.5688 | -0.0511 | 0.9183 | 0.9952 |
| 20 | same_target_object:L4->L8:object_last | 234 | 208 | 0.2566 | 0.3564 | -0.0998 | 0.9519 | 0.9952 |
| 21 | same_target_object:L8->L12:object_first | 234 | 208 | 0.1130 | 0.2970 | -0.1840 | 0.9519 | 1.0000 |
| 22 | same_target_object:L8->L16:object_first | 234 | 208 | 0.1130 | 0.2677 | -0.1547 | 0.9519 | 1.0000 |
| 23 | same_target_object:L8->L12:object_last | 234 | 208 | -0.0370 | 0.1741 | -0.2111 | 0.9615 | 1.0000 |
| 24 | same_target_object:L8->L16:object_last | 234 | 208 | -0.0370 | 0.1232 | -0.1602 | 0.9615 | 1.0000 |

### Top control-relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mismatch_object:is_a | 324 | 324 | 13.5765 | 12.7790 | 0.7975 | 0.2253 | 1.0000 |
| 2 | mismatch_object:used_for | 288 | 288 | 10.5310 | 9.7953 | 0.7357 | 0.2396 | 0.9514 |
| 3 | mismatch_object:can_do | 288 | 276 | 8.0428 | 7.4382 | 0.6046 | 0.2391 | 0.9203 |
| 4 | mismatch_object:function | 288 | 270 | 7.4355 | 7.0341 | 0.4014 | 0.2370 | 0.9778 |
| 5 | mismatch_object:part_of | 288 | 234 | 6.5652 | 5.7505 | 0.8146 | 0.2906 | 0.9402 |
| 6 | same_prompt_last:is_a | 324 | 324 | 6.2097 | 5.4990 | 0.7106 | 0.7685 | 1.0000 |
| 7 | mismatch_object:material | 288 | 246 | 5.6070 | 4.1918 | 1.4151 | 0.3252 | 0.8537 |
| 8 | same_prompt_last:used_for | 288 | 288 | 5.4264 | 4.9594 | 0.4670 | 0.6250 | 0.9861 |
| 9 | random_same_norm:used_for | 288 | 288 | 5.2923 | 4.8390 | 0.4533 | 0.6528 | 0.9861 |
| 10 | random_same_norm:is_a | 324 | 324 | 5.2417 | 4.6206 | 0.6211 | 0.8241 | 1.0000 |
| 11 | same_prompt_last:can_do | 288 | 276 | 5.0315 | 4.7081 | 0.3234 | 0.4855 | 0.9638 |
| 12 | same_prompt_last:function | 288 | 270 | 4.9049 | 4.5211 | 0.3838 | 0.3889 | 0.9778 |
| 13 | random_same_norm:can_do | 288 | 276 | 4.9038 | 4.6221 | 0.2817 | 0.4783 | 0.9710 |
| 14 | random_same_norm:function | 288 | 270 | 4.2572 | 3.8944 | 0.3627 | 0.5185 | 0.9852 |
| 15 | mismatch_object:location | 288 | 156 | 3.7404 | 3.1450 | 0.5954 | 0.6026 | 0.9679 |
| 16 | same_prompt_last:part_of | 288 | 234 | 3.5673 | 3.0716 | 0.4957 | 0.5726 | 0.9573 |
| 17 | random_same_norm:part_of | 288 | 234 | 3.4904 | 2.9792 | 0.5112 | 0.6154 | 0.9744 |
| 18 | same_prompt_last:material | 288 | 246 | 3.3145 | 2.9070 | 0.4075 | 0.6341 | 0.9675 |
| 19 | same_prompt_last:location | 288 | 156 | 3.2881 | 2.8341 | 0.4539 | 0.6282 | 0.9744 |
| 20 | random_same_norm:material | 288 | 246 | 3.1931 | 2.7282 | 0.4649 | 0.6585 | 0.9512 |
| 21 | random_same_norm:location | 288 | 156 | 3.1422 | 2.6807 | 0.4615 | 0.6667 | 0.9744 |
| 22 | same_target_object:location | 180 | 108 | 1.0920 | 0.8073 | 0.2847 | 0.8333 | 1.0000 |
| 23 | same_target_object:can_do | 144 | 138 | 1.0317 | 0.8967 | 0.1350 | 0.8261 | 1.0000 |
| 24 | same_target_object:used_for | 198 | 198 | 0.1705 | 0.0852 | 0.0852 | 1.0000 | 1.0000 |
| 25 | same_target_object:is_a | 324 | 324 | 0.1698 | 0.2942 | -0.1244 | 0.9907 | 1.0000 |
| 26 | same_target_object:function | 108 | 108 | 0.0411 | 0.2905 | -0.2494 | 0.9815 | 1.0000 |
| 27 | same_target_object:material | 234 | 192 | -0.2852 | 0.1165 | -0.4017 | 0.9896 | 0.9896 |
| 28 | same_target_object:part_of | 216 | 180 | -0.5892 | -0.0285 | -0.5608 | 0.9222 | 1.0000 |

## glm4

items=342, rows=7560, layer_pairs=[[4, 10], [10, 20], [4, 30]], controls=['mismatch_object', 'same_target_object', 'random_same_norm', 'same_prompt_last']

### By control

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mismatch_object | 2052 | 1896 | 7.3136 | 3.4898 | 3.8238 | 0.3534 | 0.6540 |
| 2 | random_same_norm | 2052 | 1896 | 3.1530 | 1.5785 | 1.5745 | 0.7447 | 0.9182 |
| 3 | same_prompt_last | 2052 | 1896 | 2.5322 | 1.3348 | 1.1974 | 0.8223 | 0.9378 |
| 4 | same_target_object | 1404 | 1314 | 0.1218 | 0.1151 | 0.0068 | 0.9581 | 0.9833 |

### Top control-paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mismatch_object:L4->L10:object_last | 342 | 316 | 9.3529 | 6.1331 | 3.2198 | 0.1709 | 0.7057 |
| 2 | mismatch_object:L4->L30:object_last | 342 | 316 | 9.3529 | 0.9985 | 8.3544 | 0.1709 | 0.2437 |
| 3 | mismatch_object:L4->L10:object_first | 342 | 316 | 8.7606 | 5.5938 | 3.1667 | 0.2184 | 0.7120 |
| 4 | mismatch_object:L4->L30:object_first | 342 | 316 | 8.7606 | 0.9556 | 7.8050 | 0.2184 | 0.2880 |
| 5 | random_same_norm:L4->L30:object_last | 342 | 316 | 4.3516 | 0.8970 | 3.4546 | 0.6424 | 0.7975 |
| 6 | random_same_norm:L4->L10:object_last | 342 | 316 | 4.2776 | 3.0883 | 1.1893 | 0.6329 | 0.9525 |
| 7 | mismatch_object:L10->L20:object_last | 342 | 316 | 4.0759 | 3.8696 | 0.2063 | 0.6519 | 0.9873 |
| 8 | random_same_norm:L4->L30:object_first | 342 | 316 | 4.0741 | 0.8404 | 3.2337 | 0.6551 | 0.8070 |
| 9 | random_same_norm:L4->L10:object_first | 342 | 316 | 4.0067 | 2.8875 | 1.1192 | 0.6551 | 0.9525 |
| 10 | mismatch_object:L10->L20:object_first | 342 | 316 | 3.5787 | 3.3883 | 0.1904 | 0.6899 | 0.9873 |
| 11 | same_prompt_last:L4->L10:object_last | 342 | 316 | 3.4650 | 2.6105 | 0.8545 | 0.7468 | 0.9557 |
| 12 | same_prompt_last:L4->L30:object_last | 342 | 316 | 3.4650 | 0.7499 | 2.7151 | 0.7468 | 0.8544 |
| 13 | same_prompt_last:L4->L10:object_first | 342 | 316 | 3.2555 | 2.4519 | 0.8036 | 0.7658 | 0.9557 |
| 14 | same_prompt_last:L4->L30:object_first | 342 | 316 | 3.2555 | 0.6942 | 2.5612 | 0.7658 | 0.8608 |
| 15 | random_same_norm:L10->L20:object_last | 342 | 316 | 1.1646 | 0.9303 | 0.2343 | 0.9399 | 1.0000 |
| 16 | random_same_norm:L10->L20:object_first | 342 | 316 | 1.0436 | 0.8276 | 0.2160 | 0.9430 | 1.0000 |
| 17 | same_prompt_last:L10->L20:object_last | 342 | 316 | 0.9284 | 0.7944 | 0.1340 | 0.9525 | 1.0000 |
| 18 | same_prompt_last:L10->L20:object_first | 342 | 316 | 0.8241 | 0.7080 | 0.1161 | 0.9557 | 1.0000 |
| 19 | same_target_object:L4->L10:object_first | 234 | 219 | 0.2613 | 0.3574 | -0.0961 | 0.9452 | 0.9772 |
| 20 | same_target_object:L4->L30:object_first | 234 | 219 | 0.2613 | 0.0427 | 0.2185 | 0.9452 | 0.9726 |
| 21 | same_target_object:L4->L10:object_last | 234 | 219 | 0.1282 | 0.3129 | -0.1847 | 0.9543 | 0.9772 |
| 22 | same_target_object:L4->L30:object_last | 234 | 219 | 0.1282 | 0.0381 | 0.0901 | 0.9543 | 0.9817 |
| 23 | same_target_object:L10->L20:object_first | 234 | 219 | -0.0062 | -0.0139 | 0.0077 | 0.9726 | 0.9954 |
| 24 | same_target_object:L10->L20:object_last | 234 | 219 | -0.0419 | -0.0469 | 0.0050 | 0.9772 | 0.9954 |

### Top control-relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mismatch_object:is_a | 324 | 324 | 11.5040 | 4.8225 | 6.6815 | 0.3735 | 0.6204 |
| 2 | mismatch_object:used_for | 288 | 288 | 9.7794 | 5.7145 | 4.0650 | 0.2639 | 0.7118 |
| 3 | mismatch_object:function | 288 | 276 | 6.7235 | 3.4296 | 3.2939 | 0.3478 | 0.7355 |
| 4 | mismatch_object:can_do | 288 | 282 | 6.5999 | 3.7027 | 2.8972 | 0.2766 | 0.6738 |
| 5 | mismatch_object:part_of | 288 | 240 | 6.4803 | 2.8368 | 3.6434 | 0.4167 | 0.6583 |
| 6 | random_same_norm:used_for | 288 | 288 | 4.3149 | 2.2699 | 2.0450 | 0.7569 | 0.9514 |
| 7 | mismatch_object:material | 288 | 264 | 4.3085 | 1.4767 | 2.8318 | 0.3864 | 0.5379 |
| 8 | mismatch_object:location | 288 | 222 | 4.1135 | 1.5630 | 2.5505 | 0.4369 | 0.6351 |
| 9 | random_same_norm:is_a | 324 | 324 | 4.0796 | 1.4556 | 2.6240 | 0.8827 | 0.9784 |
| 10 | same_prompt_last:function | 288 | 276 | 3.8439 | 2.0532 | 1.7908 | 0.6594 | 0.9130 |
| 11 | random_same_norm:can_do | 288 | 282 | 3.7515 | 2.5034 | 1.2481 | 0.5390 | 0.8794 |
| 12 | random_same_norm:function | 288 | 276 | 3.7133 | 2.0573 | 1.6560 | 0.6739 | 0.9094 |
| 13 | same_prompt_last:used_for | 288 | 288 | 3.3173 | 1.9481 | 1.3691 | 0.8819 | 0.9792 |
| 14 | same_prompt_last:is_a | 324 | 324 | 3.0780 | 1.2467 | 1.8313 | 0.9753 | 1.0000 |
| 15 | random_same_norm:part_of | 288 | 240 | 2.6713 | 1.1668 | 1.5044 | 0.7833 | 0.9417 |
| 16 | same_prompt_last:can_do | 288 | 282 | 2.3138 | 1.6574 | 0.6565 | 0.7943 | 0.9504 |
| 17 | same_prompt_last:part_of | 288 | 240 | 2.1135 | 1.0237 | 1.0898 | 0.8083 | 0.9500 |
| 18 | random_same_norm:location | 288 | 222 | 1.4346 | 0.6753 | 0.7593 | 0.7568 | 0.8514 |
| 19 | random_same_norm:material | 288 | 264 | 1.4062 | 0.6205 | 0.7857 | 0.8106 | 0.8939 |
| 20 | same_prompt_last:location | 288 | 222 | 1.3274 | 0.7151 | 0.6123 | 0.7973 | 0.8559 |
| 21 | same_prompt_last:material | 288 | 264 | 1.2618 | 0.4822 | 0.7796 | 0.8030 | 0.8864 |
| 22 | same_target_object:location | 180 | 138 | 0.4552 | 0.1891 | 0.2661 | 0.8043 | 0.8986 |
| 23 | same_target_object:can_do | 144 | 144 | 0.4232 | 0.2782 | 0.1450 | 0.9306 | 1.0000 |
| 24 | same_target_object:function | 108 | 108 | 0.4190 | 0.0952 | 0.3238 | 1.0000 | 1.0000 |
| 25 | same_target_object:is_a | 324 | 324 | 0.2165 | 0.1335 | 0.0830 | 1.0000 | 1.0000 |
| 26 | same_target_object:used_for | 198 | 198 | 0.0505 | 0.0200 | 0.0305 | 1.0000 | 1.0000 |
| 27 | same_target_object:material | 234 | 210 | -0.0881 | 0.0527 | -0.1408 | 0.9714 | 0.9714 |
| 28 | same_target_object:part_of | 216 | 192 | -0.3677 | 0.0858 | -0.4535 | 0.9375 | 0.9896 |

## deepseek7b

items=342, rows=10080, layer_pairs=[[8, 10], [8, 12], [12, 14], [12, 16]], controls=['mismatch_object', 'same_target_object', 'random_same_norm', 'same_prompt_last']

### By control

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mismatch_object | 2736 | 1968 | 4.3491 | 3.9909 | 0.3582 | 0.4482 | 0.9355 |
| 2 | random_same_norm | 2736 | 1968 | 1.9750 | 1.8023 | 0.1727 | 0.7368 | 0.9360 |
| 3 | same_prompt_last | 2736 | 1968 | 1.5854 | 1.4619 | 0.1235 | 0.7774 | 0.9543 |
| 4 | same_target_object | 1872 | 1488 | 0.2235 | 0.1759 | 0.0476 | 0.9099 | 0.9745 |

### Top control-paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mismatch_object:L8->L10:object_last | 342 | 246 | 5.0276 | 4.3645 | 0.6631 | 0.3537 | 0.9024 |
| 2 | mismatch_object:L8->L12:object_last | 342 | 246 | 5.0276 | 4.2791 | 0.7485 | 0.3537 | 0.8943 |
| 3 | mismatch_object:L8->L10:object_first | 342 | 246 | 4.7482 | 4.1027 | 0.6455 | 0.3821 | 0.9065 |
| 4 | mismatch_object:L8->L12:object_first | 342 | 246 | 4.7482 | 4.0111 | 0.7370 | 0.3821 | 0.8984 |
| 5 | mismatch_object:L12->L14:object_last | 342 | 246 | 3.9365 | 3.9421 | -0.0056 | 0.5163 | 0.9756 |
| 6 | mismatch_object:L12->L16:object_last | 342 | 246 | 3.9365 | 3.9055 | 0.0310 | 0.5163 | 0.9675 |
| 7 | mismatch_object:L12->L14:object_first | 342 | 246 | 3.6843 | 3.6849 | -0.0006 | 0.5407 | 0.9715 |
| 8 | mismatch_object:L12->L16:object_first | 342 | 246 | 3.6843 | 3.6376 | 0.0466 | 0.5407 | 0.9675 |
| 9 | random_same_norm:L8->L10:object_last | 342 | 246 | 2.2853 | 2.0400 | 0.2453 | 0.6829 | 0.9350 |
| 10 | random_same_norm:L8->L12:object_last | 342 | 246 | 2.1583 | 1.7621 | 0.3962 | 0.7276 | 0.9024 |
| 11 | random_same_norm:L8->L10:object_first | 342 | 246 | 2.1267 | 1.8827 | 0.2440 | 0.6992 | 0.9350 |
| 12 | same_prompt_last:L8->L12:object_last | 342 | 246 | 2.0988 | 1.8418 | 0.2570 | 0.7236 | 0.9268 |
| 13 | same_prompt_last:L8->L10:object_last | 342 | 246 | 2.0988 | 1.8283 | 0.2706 | 0.7236 | 0.9472 |
| 14 | random_same_norm:L8->L12:object_first | 342 | 246 | 2.0044 | 1.6242 | 0.3803 | 0.7398 | 0.9024 |
| 15 | same_prompt_last:L8->L12:object_first | 342 | 246 | 1.9499 | 1.7147 | 0.2353 | 0.7398 | 0.9268 |
| 16 | same_prompt_last:L8->L10:object_first | 342 | 246 | 1.9499 | 1.6907 | 0.2593 | 0.7398 | 0.9472 |
| 17 | random_same_norm:L12->L16:object_last | 342 | 246 | 1.9197 | 1.9273 | -0.0076 | 0.7602 | 0.9512 |
| 18 | random_same_norm:L12->L14:object_last | 342 | 246 | 1.8048 | 1.7347 | 0.0702 | 0.7561 | 0.9553 |
| 19 | random_same_norm:L12->L16:object_first | 342 | 246 | 1.7993 | 1.8116 | -0.0123 | 0.7683 | 0.9512 |
| 20 | random_same_norm:L12->L14:object_first | 342 | 246 | 1.7011 | 1.6356 | 0.0655 | 0.7602 | 0.9553 |
| 21 | same_prompt_last:L12->L14:object_last | 342 | 246 | 1.2074 | 1.2353 | -0.0279 | 0.8171 | 0.9715 |
| 22 | same_prompt_last:L12->L16:object_last | 342 | 246 | 1.2074 | 1.1900 | 0.0174 | 0.8171 | 0.9715 |
| 23 | same_prompt_last:L12->L14:object_first | 342 | 246 | 1.0855 | 1.1186 | -0.0331 | 0.8293 | 0.9715 |
| 24 | same_prompt_last:L12->L16:object_first | 342 | 246 | 1.0855 | 1.0761 | 0.0094 | 0.8293 | 0.9715 |
| 25 | same_target_object:L8->L10:object_first | 234 | 186 | 0.3943 | 0.3157 | 0.0786 | 0.8925 | 0.9785 |
| 26 | same_target_object:L8->L12:object_first | 234 | 186 | 0.3943 | 0.2860 | 0.1083 | 0.8925 | 0.9839 |
| 27 | same_target_object:L8->L10:object_last | 234 | 186 | 0.2769 | 0.2040 | 0.0729 | 0.9140 | 0.9839 |
| 28 | same_target_object:L8->L12:object_last | 234 | 186 | 0.2769 | 0.1584 | 0.1185 | 0.9140 | 0.9839 |
| 29 | same_target_object:L12->L14:object_first | 234 | 186 | 0.1788 | 0.2061 | -0.0273 | 0.9032 | 0.9785 |
| 30 | same_target_object:L12->L16:object_first | 234 | 186 | 0.1788 | 0.1545 | 0.0243 | 0.9032 | 0.9516 |

### Top control-relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mismatch_object:is_a | 432 | 432 | 7.2966 | 6.9274 | 0.3692 | 0.3472 | 1.0000 |
| 2 | mismatch_object:used_for | 384 | 272 | 5.9446 | 5.2757 | 0.6689 | 0.3088 | 0.9044 |
| 3 | mismatch_object:can_do | 384 | 280 | 4.3964 | 4.1404 | 0.2560 | 0.3857 | 0.9143 |
| 4 | mismatch_object:function | 384 | 200 | 4.1909 | 4.1253 | 0.0656 | 0.2800 | 0.8350 |
| 5 | random_same_norm:used_for | 384 | 272 | 3.3868 | 2.7597 | 0.6271 | 0.5074 | 0.8750 |
| 6 | same_prompt_last:used_for | 384 | 272 | 2.7381 | 2.2495 | 0.4885 | 0.5735 | 0.9265 |
| 7 | random_same_norm:can_do | 384 | 280 | 2.6585 | 2.4989 | 0.1596 | 0.6214 | 0.9429 |
| 8 | mismatch_object:location | 384 | 248 | 2.4652 | 2.0694 | 0.3958 | 0.5000 | 0.9113 |
| 9 | random_same_norm:function | 384 | 200 | 2.3156 | 2.5222 | -0.2066 | 0.6000 | 0.8400 |
| 10 | mismatch_object:part_of | 384 | 224 | 2.1451 | 1.7891 | 0.3560 | 0.6429 | 0.9643 |
| 11 | random_same_norm:is_a | 432 | 432 | 2.0488 | 1.9585 | 0.0903 | 0.8889 | 1.0000 |
| 12 | mismatch_object:material | 384 | 312 | 2.0158 | 1.6929 | 0.3229 | 0.6923 | 0.9551 |
| 13 | same_prompt_last:can_do | 384 | 280 | 2.0000 | 1.9442 | 0.0558 | 0.7429 | 0.9857 |
| 14 | same_prompt_last:function | 384 | 200 | 1.8762 | 2.0916 | -0.2153 | 0.6400 | 0.8300 |
| 15 | same_prompt_last:is_a | 432 | 432 | 1.7188 | 1.5864 | 0.1324 | 0.8750 | 0.9907 |
| 16 | random_same_norm:location | 384 | 248 | 1.3884 | 1.2823 | 0.1061 | 0.7419 | 0.8790 |
| 17 | random_same_norm:part_of | 384 | 224 | 1.3650 | 1.1264 | 0.2386 | 0.7589 | 0.9911 |
| 18 | same_prompt_last:location | 384 | 248 | 1.1109 | 1.1400 | -0.0291 | 0.8065 | 0.9516 |
| 19 | same_prompt_last:part_of | 384 | 224 | 0.8504 | 0.7330 | 0.1175 | 0.8214 | 0.9911 |
| 20 | same_prompt_last:material | 384 | 312 | 0.7424 | 0.5459 | 0.1965 | 0.8846 | 0.9551 |
| 21 | random_same_norm:material | 384 | 312 | 0.7143 | 0.5633 | 0.1511 | 0.8974 | 0.9615 |
| 22 | same_target_object:can_do | 192 | 168 | 0.7128 | 0.5253 | 0.1875 | 0.8571 | 0.9762 |
| 23 | same_target_object:used_for | 264 | 216 | 0.5463 | 0.2882 | 0.2581 | 0.9074 | 0.9537 |
| 24 | same_target_object:location | 240 | 168 | 0.4747 | 0.3795 | 0.0952 | 0.7976 | 0.9286 |
| 25 | same_target_object:function | 144 | 80 | 0.1625 | 0.7488 | -0.5863 | 0.8750 | 0.9250 |
| 26 | same_target_object:part_of | 288 | 184 | 0.1325 | 0.0329 | 0.0995 | 0.9348 | 1.0000 |
| 27 | same_target_object:material | 312 | 240 | -0.0068 | 0.0849 | -0.0917 | 0.9167 | 0.9750 |
| 28 | same_target_object:is_a | 432 | 432 | -0.0477 | -0.0898 | 0.0421 | 0.9676 | 1.0000 |

