# Phase74 Factor Control Audit Summary

## qwen3

items=336, rows=3900, layer_pairs=[[4, 8], [8, 12]]
control_types=['wrong_target_same_relation_frame', 'same_target_same_relation_frame', 'same_object_same_relation_other_frame', 'same_object_different_relation']

### By control type

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame | 1344 | 1028 | 8.5678 | 7.4166 | 1.1512 | 0.2451 | 0.8619 |
| 2 | same_target_same_relation_frame | 840 | 644 | 0.5579 | 0.4630 | 0.0949 | 0.8680 | 0.9488 |
| 3 | same_object_same_relation_other_frame | 1344 | 1028 | 0.0611 | 0.0508 | 0.0103 | 0.9708 | 0.9805 |
| 4 | same_object_different_relation | 372 | 268 | 0.0324 | 0.0190 | 0.0133 | 0.9701 | 0.9776 |

### Top control paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:L4->L8:object_last | 336 | 257 | 9.9709 | 8.3849 | 1.5860 | 0.1634 | 0.8171 |
| 2 | wrong_target_same_relation_frame:L4->L8:object_first | 336 | 257 | 9.4119 | 7.8535 | 1.5584 | 0.1984 | 0.8132 |
| 3 | wrong_target_same_relation_frame:L8->L12:object_last | 336 | 257 | 7.7094 | 6.9590 | 0.7504 | 0.2918 | 0.9066 |
| 4 | wrong_target_same_relation_frame:L8->L12:object_first | 336 | 257 | 7.1790 | 6.4688 | 0.7102 | 0.3268 | 0.9105 |
| 5 | same_target_same_relation_frame:L4->L8:object_first | 210 | 161 | 0.8373 | 0.5899 | 0.2474 | 0.8261 | 0.9255 |
| 6 | same_target_same_relation_frame:L4->L8:object_last | 210 | 161 | 0.5586 | 0.3130 | 0.2457 | 0.8509 | 0.9255 |
| 7 | same_target_same_relation_frame:L8->L12:object_first | 210 | 161 | 0.5293 | 0.5694 | -0.0401 | 0.8882 | 0.9752 |
| 8 | same_target_same_relation_frame:L8->L12:object_last | 210 | 161 | 0.3063 | 0.3797 | -0.0733 | 0.9068 | 0.9689 |
| 9 | same_object_same_relation_other_frame:L8->L12:object_last | 336 | 257 | 0.0804 | 0.0381 | 0.0423 | 0.9728 | 0.9883 |
| 10 | same_object_same_relation_other_frame:L8->L12:object_first | 336 | 257 | 0.0618 | 0.0287 | 0.0331 | 0.9767 | 0.9883 |
| 11 | same_object_same_relation_other_frame:L4->L8:object_last | 336 | 257 | 0.0617 | 0.0799 | -0.0183 | 0.9650 | 0.9728 |
| 12 | same_object_same_relation_other_frame:L4->L8:object_first | 336 | 257 | 0.0407 | 0.0564 | -0.0157 | 0.9689 | 0.9728 |
| 13 | same_object_different_relation:L8->L12:object_first | 93 | 67 | 0.0383 | -0.0128 | 0.0511 | 0.9851 | 0.9851 |
| 14 | same_object_different_relation:L8->L12:object_last | 93 | 67 | 0.0383 | -0.0128 | 0.0511 | 0.9851 | 0.9851 |
| 15 | same_object_different_relation:L4->L8:object_first | 93 | 67 | 0.0264 | 0.0508 | -0.0244 | 0.9552 | 0.9701 |
| 16 | same_object_different_relation:L4->L8:object_last | 93 | 67 | 0.0264 | 0.0508 | -0.0244 | 0.9552 | 0.9701 |

### Top control relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:used_for | 192 | 140 | 10.9175 | 10.1865 | 0.7310 | 0.2286 | 0.9000 |
| 2 | wrong_target_same_relation_frame:is_a | 192 | 192 | 10.2747 | 9.4177 | 0.8570 | 0.2344 | 0.9740 |
| 3 | wrong_target_same_relation_frame:can_do | 192 | 140 | 9.0561 | 8.7244 | 0.3318 | 0.1857 | 0.9000 |
| 4 | wrong_target_same_relation_frame:property | 192 | 124 | 8.5958 | 6.8524 | 1.7434 | 0.2742 | 0.8306 |
| 5 | wrong_target_same_relation_frame:material | 192 | 156 | 7.9245 | 5.8116 | 2.1130 | 0.2051 | 0.7821 |
| 6 | wrong_target_same_relation_frame:location | 192 | 120 | 6.8782 | 5.5490 | 1.3292 | 0.3250 | 0.7500 |
| 7 | wrong_target_same_relation_frame:part_of | 192 | 156 | 5.8407 | 4.7842 | 1.0566 | 0.2821 | 0.8462 |
| 8 | same_target_same_relation_frame:location | 132 | 88 | 1.6608 | 1.4317 | 0.2292 | 0.6818 | 0.8977 |
| 9 | same_target_same_relation_frame:can_do | 108 | 84 | 1.2273 | 0.7325 | 0.4948 | 0.8095 | 0.9048 |
| 10 | same_target_same_relation_frame:used_for | 72 | 44 | 0.9715 | 0.4850 | 0.4864 | 0.8182 | 0.8636 |
| 11 | same_object_different_relation:used_for | 36 | 20 | 0.7615 | 0.8406 | -0.0791 | 1.0000 | 1.0000 |
| 12 | same_target_same_relation_frame:material | 156 | 124 | 0.5657 | 0.5619 | 0.0037 | 0.8548 | 0.9839 |
| 13 | same_target_same_relation_frame:property | 120 | 64 | 0.3655 | -0.0480 | 0.4135 | 0.9375 | 0.9375 |
| 14 | same_object_same_relation_other_frame:used_for | 192 | 140 | 0.2396 | 0.2247 | 0.0149 | 0.9286 | 0.9429 |
| 15 | same_object_same_relation_other_frame:material | 192 | 156 | 0.1688 | 0.0361 | 0.1327 | 0.9744 | 0.9744 |
| 16 | same_object_same_relation_other_frame:location | 192 | 120 | 0.1202 | 0.2235 | -0.1033 | 0.9333 | 0.9833 |
| 17 | same_object_same_relation_other_frame:can_do | 192 | 140 | 0.0663 | -0.0401 | 0.1063 | 0.9714 | 0.9571 |
| 18 | same_object_different_relation:can_do | 84 | 76 | 0.0303 | -0.0099 | 0.0402 | 1.0000 | 1.0000 |
| 19 | same_object_different_relation:material | 36 | 32 | 0.0061 | -0.0453 | 0.0514 | 0.8750 | 0.8750 |
| 20 | same_object_same_relation_other_frame:part_of | 192 | 156 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 21 | same_object_same_relation_other_frame:property | 192 | 124 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 22 | same_object_different_relation:property | 60 | 24 | -0.0029 | -0.1219 | 0.1189 | 0.8333 | 0.9167 |
| 23 | same_object_different_relation:location | 60 | 24 | -0.0074 | -0.0002 | -0.0072 | 1.0000 | 1.0000 |
| 24 | same_target_same_relation_frame:is_a | 144 | 144 | -0.0250 | 0.0066 | -0.0316 | 0.9792 | 0.9861 |
| 25 | same_object_different_relation:is_a | 84 | 84 | -0.0394 | -0.0189 | -0.0205 | 1.0000 | 1.0000 |
| 26 | same_object_same_relation_other_frame:is_a | 192 | 192 | -0.1080 | -0.0318 | -0.0762 | 0.9792 | 1.0000 |
| 27 | same_target_same_relation_frame:part_of | 108 | 96 | -0.2357 | 0.2266 | -0.4622 | 0.9167 | 0.9792 |
| 28 | same_object_different_relation:part_of | 12 | 8 | -0.6875 | -0.6250 | -0.0625 | 1.0000 | 1.0000 |

### Top control relation paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:used_for:L4->L8:object_last | 48 | 35 | 12.3930 | 11.4872 | 0.9058 | 0.1143 | 0.9143 |
| 2 | wrong_target_same_relation_frame:is_a:L4->L8:object_last | 48 | 48 | 12.1636 | 10.9576 | 1.2061 | 0.1667 | 0.9583 |
| 3 | wrong_target_same_relation_frame:used_for:L4->L8:object_first | 48 | 35 | 10.9470 | 10.1712 | 0.7758 | 0.2000 | 0.9143 |
| 4 | wrong_target_same_relation_frame:used_for:L8->L12:object_last | 48 | 35 | 10.8038 | 10.1956 | 0.6082 | 0.2571 | 0.8857 |
| 5 | wrong_target_same_relation_frame:is_a:L4->L8:object_first | 48 | 48 | 10.8001 | 9.5548 | 1.2453 | 0.2292 | 0.9375 |
| 6 | wrong_target_same_relation_frame:material:L4->L8:object_first | 48 | 39 | 10.0279 | 6.8425 | 3.1854 | 0.0769 | 0.6667 |
| 7 | wrong_target_same_relation_frame:material:L4->L8:object_last | 48 | 39 | 10.0279 | 6.8425 | 3.1854 | 0.0769 | 0.6667 |
| 8 | wrong_target_same_relation_frame:can_do:L4->L8:object_first | 48 | 35 | 9.7130 | 9.0029 | 0.7101 | 0.1714 | 0.8571 |
| 9 | wrong_target_same_relation_frame:can_do:L4->L8:object_last | 48 | 35 | 9.7130 | 9.0029 | 0.7101 | 0.1714 | 0.8571 |
| 10 | wrong_target_same_relation_frame:is_a:L8->L12:object_last | 48 | 48 | 9.6950 | 9.1914 | 0.5037 | 0.2292 | 1.0000 |
| 11 | wrong_target_same_relation_frame:used_for:L8->L12:object_first | 48 | 35 | 9.5260 | 8.8919 | 0.6340 | 0.3429 | 0.8857 |
| 12 | wrong_target_same_relation_frame:property:L4->L8:object_last | 48 | 31 | 9.4906 | 7.4018 | 2.0888 | 0.2258 | 0.8065 |
| 13 | wrong_target_same_relation_frame:property:L4->L8:object_first | 48 | 31 | 9.3943 | 7.3411 | 2.0532 | 0.2258 | 0.8065 |
| 14 | wrong_target_same_relation_frame:is_a:L8->L12:object_first | 48 | 48 | 8.4399 | 7.9669 | 0.4730 | 0.3125 | 1.0000 |
| 15 | wrong_target_same_relation_frame:can_do:L8->L12:object_first | 48 | 35 | 8.3992 | 8.4458 | -0.0466 | 0.2000 | 0.9429 |
| 16 | wrong_target_same_relation_frame:can_do:L8->L12:object_last | 48 | 35 | 8.3992 | 8.4458 | -0.0466 | 0.2000 | 0.9429 |
| 17 | wrong_target_same_relation_frame:location:L4->L8:object_last | 48 | 30 | 8.2726 | 6.6571 | 1.6155 | 0.2333 | 0.7000 |
| 18 | wrong_target_same_relation_frame:property:L8->L12:object_last | 48 | 31 | 7.7911 | 6.3907 | 1.4004 | 0.3226 | 0.8710 |
| 19 | wrong_target_same_relation_frame:property:L8->L12:object_first | 48 | 31 | 7.7073 | 6.2762 | 1.4312 | 0.3226 | 0.8387 |
| 20 | wrong_target_same_relation_frame:location:L4->L8:object_first | 48 | 30 | 7.4521 | 5.9472 | 1.5049 | 0.3333 | 0.7000 |
| 21 | wrong_target_same_relation_frame:part_of:L4->L8:object_first | 48 | 39 | 6.9608 | 5.5326 | 1.4282 | 0.1795 | 0.7692 |
| 22 | wrong_target_same_relation_frame:part_of:L4->L8:object_last | 48 | 39 | 6.9608 | 5.5326 | 1.4282 | 0.1795 | 0.7692 |
| 23 | wrong_target_same_relation_frame:location:L8->L12:object_last | 48 | 30 | 6.3733 | 5.0966 | 1.2768 | 0.3333 | 0.7667 |
| 24 | wrong_target_same_relation_frame:material:L8->L12:object_first | 48 | 39 | 5.8212 | 4.7806 | 1.0406 | 0.3333 | 0.8974 |
| 25 | wrong_target_same_relation_frame:material:L8->L12:object_last | 48 | 39 | 5.8212 | 4.7806 | 1.0406 | 0.3333 | 0.8974 |
| 26 | wrong_target_same_relation_frame:location:L8->L12:object_first | 48 | 30 | 5.4146 | 4.4951 | 0.9195 | 0.4000 | 0.8333 |
| 27 | wrong_target_same_relation_frame:part_of:L8->L12:object_first | 48 | 39 | 4.7207 | 4.0357 | 0.6850 | 0.3846 | 0.9231 |
| 28 | wrong_target_same_relation_frame:part_of:L8->L12:object_last | 48 | 39 | 4.7207 | 4.0357 | 0.6850 | 0.3846 | 0.9231 |
| 29 | same_target_same_relation_frame:location:L4->L8:object_first | 33 | 22 | 2.5764 | 2.2032 | 0.3732 | 0.5455 | 0.8636 |
| 30 | same_target_same_relation_frame:location:L8->L12:object_first | 33 | 22 | 1.7938 | 1.6969 | 0.0969 | 0.6818 | 0.9545 |
| 31 | same_target_same_relation_frame:location:L4->L8:object_last | 33 | 22 | 1.4566 | 0.9924 | 0.4641 | 0.6818 | 0.8636 |
| 32 | same_target_same_relation_frame:can_do:L8->L12:object_first | 27 | 21 | 1.3076 | 1.0478 | 0.2598 | 0.8571 | 0.9524 |
| 33 | same_target_same_relation_frame:can_do:L8->L12:object_last | 27 | 21 | 1.3076 | 1.0478 | 0.2598 | 0.8571 | 0.9524 |
| 34 | same_object_different_relation:used_for:L8->L12:object_first | 9 | 5 | 1.1704 | 0.8214 | 0.3490 | 1.0000 | 1.0000 |
| 35 | same_object_different_relation:used_for:L8->L12:object_last | 9 | 5 | 1.1704 | 0.8214 | 0.3490 | 1.0000 | 1.0000 |
| 36 | same_target_same_relation_frame:can_do:L4->L8:object_first | 27 | 21 | 1.1469 | 0.4172 | 0.7298 | 0.7619 | 0.8571 |
| 37 | same_target_same_relation_frame:can_do:L4->L8:object_last | 27 | 21 | 1.1469 | 0.4172 | 0.7298 | 0.7619 | 0.8571 |
| 38 | same_target_same_relation_frame:used_for:L8->L12:object_first | 18 | 11 | 1.0897 | 0.7186 | 0.3712 | 0.8182 | 0.8182 |
| 39 | same_target_same_relation_frame:used_for:L8->L12:object_last | 18 | 11 | 1.0897 | 0.7186 | 0.3712 | 0.8182 | 0.8182 |
| 40 | same_target_same_relation_frame:used_for:L4->L8:object_first | 18 | 11 | 0.8532 | 0.2515 | 0.6017 | 0.8182 | 0.9091 |

## glm4

items=336, rows=3900, layer_pairs=[[4, 10], [10, 20]]
control_types=['wrong_target_same_relation_frame', 'same_target_same_relation_frame', 'same_object_same_relation_other_frame', 'same_object_different_relation']

### By control type

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame | 1344 | 1112 | 7.1684 | 5.0584 | 2.1101 | 0.4344 | 0.8076 |
| 2 | same_target_same_relation_frame | 840 | 688 | 0.1032 | 0.1651 | -0.0619 | 0.9462 | 0.9767 |
| 3 | same_object_different_relation | 372 | 280 | 0.0006 | 0.0117 | -0.0111 | 0.9857 | 0.9929 |
| 4 | same_object_same_relation_other_frame | 1344 | 1112 | -0.0287 | 0.0013 | -0.0300 | 0.9910 | 0.9964 |

### Top control paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:L4->L10:object_last | 336 | 278 | 10.5254 | 6.4832 | 4.0421 | 0.1511 | 0.6295 |
| 2 | wrong_target_same_relation_frame:L4->L10:object_first | 336 | 278 | 9.8502 | 5.9064 | 3.9438 | 0.1978 | 0.6331 |
| 3 | wrong_target_same_relation_frame:L10->L20:object_last | 336 | 278 | 4.3480 | 4.1079 | 0.2401 | 0.6799 | 0.9856 |
| 4 | wrong_target_same_relation_frame:L10->L20:object_first | 336 | 278 | 3.9502 | 3.7359 | 0.2143 | 0.7086 | 0.9820 |
| 5 | same_target_same_relation_frame:L4->L10:object_first | 210 | 172 | 0.4452 | 0.4879 | -0.0427 | 0.9012 | 0.9535 |
| 6 | same_target_same_relation_frame:L4->L10:object_last | 210 | 172 | 0.1683 | 0.3233 | -0.1551 | 0.9128 | 0.9651 |
| 7 | same_object_different_relation:L4->L10:object_first | 93 | 70 | 0.0157 | 0.0360 | -0.0203 | 0.9714 | 0.9857 |
| 8 | same_object_different_relation:L4->L10:object_last | 93 | 70 | 0.0157 | 0.0360 | -0.0203 | 0.9714 | 0.9857 |
| 9 | same_object_different_relation:L10->L20:object_first | 93 | 70 | -0.0144 | -0.0126 | -0.0018 | 1.0000 | 1.0000 |
| 10 | same_object_different_relation:L10->L20:object_last | 93 | 70 | -0.0144 | -0.0126 | -0.0018 | 1.0000 | 1.0000 |
| 11 | same_object_same_relation_other_frame:L4->L10:object_first | 336 | 278 | -0.0144 | 0.0140 | -0.0285 | 0.9856 | 0.9964 |
| 12 | same_object_same_relation_other_frame:L4->L10:object_last | 336 | 278 | -0.0163 | 0.0134 | -0.0298 | 0.9856 | 0.9964 |
| 13 | same_object_same_relation_other_frame:L10->L20:object_first | 336 | 278 | -0.0390 | -0.0110 | -0.0280 | 0.9964 | 0.9964 |
| 14 | same_object_same_relation_other_frame:L10->L20:object_last | 336 | 278 | -0.0449 | -0.0112 | -0.0338 | 0.9964 | 0.9964 |
| 15 | same_target_same_relation_frame:L10->L20:object_first | 210 | 172 | -0.0640 | -0.0492 | -0.0148 | 0.9826 | 0.9942 |
| 16 | same_target_same_relation_frame:L10->L20:object_last | 210 | 172 | -0.1366 | -0.1015 | -0.0351 | 0.9884 | 0.9942 |

### Top control relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:used_for | 192 | 188 | 11.2006 | 9.1332 | 2.0674 | 0.3617 | 0.8989 |
| 2 | wrong_target_same_relation_frame:is_a | 192 | 184 | 9.0859 | 6.5937 | 2.4922 | 0.4348 | 0.8261 |
| 3 | wrong_target_same_relation_frame:can_do | 192 | 152 | 7.0156 | 5.0411 | 1.9745 | 0.3947 | 0.8026 |
| 4 | wrong_target_same_relation_frame:part_of | 192 | 160 | 5.6695 | 3.6445 | 2.0250 | 0.4250 | 0.8375 |
| 5 | wrong_target_same_relation_frame:location | 192 | 160 | 5.3087 | 3.3644 | 1.9444 | 0.4500 | 0.8125 |
| 6 | wrong_target_same_relation_frame:property | 192 | 116 | 5.1444 | 3.5571 | 1.5873 | 0.5259 | 0.7845 |
| 7 | wrong_target_same_relation_frame:material | 192 | 152 | 5.0930 | 2.5942 | 2.4988 | 0.4868 | 0.6579 |
| 8 | same_target_same_relation_frame:location | 132 | 108 | 0.7134 | 0.4832 | 0.2301 | 0.8704 | 0.9444 |
| 9 | same_target_same_relation_frame:can_do | 108 | 92 | 0.7095 | 0.4061 | 0.3034 | 0.9130 | 0.9348 |
| 10 | same_object_different_relation:material | 36 | 20 | 0.1788 | 0.0401 | 0.1386 | 0.9000 | 1.0000 |
| 11 | same_object_different_relation:used_for | 36 | 36 | 0.1364 | 0.0643 | 0.0721 | 1.0000 | 1.0000 |
| 12 | same_object_same_relation_other_frame:can_do | 192 | 152 | 0.0456 | 0.0008 | 0.0447 | 0.9868 | 1.0000 |
| 13 | same_object_different_relation:location | 60 | 40 | 0.0343 | 0.0115 | 0.0228 | 1.0000 | 1.0000 |
| 14 | same_target_same_relation_frame:used_for | 72 | 72 | 0.0194 | 0.1123 | -0.0929 | 1.0000 | 1.0000 |
| 15 | same_object_same_relation_other_frame:location | 192 | 160 | 0.0132 | 0.0527 | -0.0395 | 0.9625 | 0.9750 |
| 16 | same_object_same_relation_other_frame:part_of | 192 | 160 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 17 | same_object_same_relation_other_frame:property | 192 | 116 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 18 | same_object_different_relation:can_do | 84 | 76 | -0.0168 | 0.0003 | -0.0172 | 1.0000 | 1.0000 |
| 19 | same_target_same_relation_frame:material | 156 | 116 | -0.0376 | 0.2552 | -0.2928 | 0.9138 | 0.9828 |
| 20 | same_object_different_relation:is_a | 84 | 84 | -0.0437 | 0.0102 | -0.0539 | 1.0000 | 1.0000 |
| 21 | same_object_same_relation_other_frame:used_for | 192 | 188 | -0.0530 | -0.0311 | -0.0218 | 1.0000 | 1.0000 |
| 22 | same_object_same_relation_other_frame:material | 192 | 152 | -0.0854 | -0.0498 | -0.0357 | 0.9868 | 1.0000 |
| 23 | same_object_same_relation_other_frame:is_a | 192 | 184 | -0.0977 | 0.0344 | -0.1321 | 1.0000 | 1.0000 |
| 24 | same_target_same_relation_frame:part_of | 108 | 92 | -0.1675 | 0.1675 | -0.3349 | 0.9565 | 1.0000 |
| 25 | same_object_different_relation:part_of | 12 | 8 | -0.1953 | -0.0078 | -0.1875 | 1.0000 | 1.0000 |
| 26 | same_object_different_relation:property | 60 | 16 | -0.1977 | -0.0696 | -0.1282 | 0.8750 | 0.8750 |
| 27 | same_target_same_relation_frame:property | 120 | 64 | -0.2483 | -0.1310 | -0.1173 | 0.9844 | 0.9688 |
| 28 | same_target_same_relation_frame:is_a | 144 | 144 | -0.2574 | -0.1435 | -0.1139 | 1.0000 | 1.0000 |

### Top control relation paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:used_for:L4->L10:object_last | 48 | 47 | 15.1377 | 11.4758 | 3.6620 | 0.1064 | 0.8085 |
| 2 | wrong_target_same_relation_frame:used_for:L4->L10:object_first | 48 | 47 | 13.7833 | 10.1956 | 3.5878 | 0.1915 | 0.8085 |
| 3 | wrong_target_same_relation_frame:is_a:L4->L10:object_last | 48 | 46 | 13.6337 | 8.6807 | 4.9530 | 0.1087 | 0.6522 |
| 4 | wrong_target_same_relation_frame:is_a:L4->L10:object_first | 48 | 46 | 12.0234 | 7.2791 | 4.7442 | 0.1957 | 0.6522 |
| 5 | wrong_target_same_relation_frame:can_do:L4->L10:object_first | 48 | 38 | 9.4425 | 5.8268 | 3.6157 | 0.2105 | 0.6316 |
| 6 | wrong_target_same_relation_frame:can_do:L4->L10:object_last | 48 | 38 | 9.4425 | 5.8268 | 3.6157 | 0.2105 | 0.6316 |
| 7 | wrong_target_same_relation_frame:part_of:L4->L10:object_first | 48 | 40 | 8.7872 | 4.7997 | 3.9875 | 0.0750 | 0.7000 |
| 8 | wrong_target_same_relation_frame:part_of:L4->L10:object_last | 48 | 40 | 8.7872 | 4.7997 | 3.9875 | 0.0750 | 0.7000 |
| 9 | wrong_target_same_relation_frame:location:L4->L10:object_last | 48 | 40 | 8.6036 | 4.9166 | 3.6870 | 0.1500 | 0.6500 |
| 10 | wrong_target_same_relation_frame:material:L4->L10:object_first | 48 | 38 | 8.4860 | 3.4725 | 5.0136 | 0.1316 | 0.3158 |
| 11 | wrong_target_same_relation_frame:material:L4->L10:object_last | 48 | 38 | 8.4860 | 3.4725 | 5.0136 | 0.1316 | 0.3158 |
| 12 | wrong_target_same_relation_frame:used_for:L10->L20:object_last | 48 | 47 | 8.3652 | 7.8325 | 0.5327 | 0.5532 | 1.0000 |
| 13 | wrong_target_same_relation_frame:used_for:L10->L20:object_first | 48 | 47 | 7.5162 | 7.0289 | 0.4873 | 0.5957 | 0.9787 |
| 14 | wrong_target_same_relation_frame:location:L4->L10:object_first | 48 | 40 | 7.4532 | 4.0972 | 3.3560 | 0.2500 | 0.6500 |
| 15 | wrong_target_same_relation_frame:property:L4->L10:object_last | 48 | 29 | 7.2590 | 4.1947 | 3.0643 | 0.3448 | 0.5862 |
| 16 | wrong_target_same_relation_frame:property:L4->L10:object_first | 48 | 29 | 7.1231 | 4.0935 | 3.0295 | 0.3793 | 0.6207 |
| 17 | wrong_target_same_relation_frame:is_a:L10->L20:object_last | 48 | 46 | 5.8106 | 5.6690 | 0.1416 | 0.6957 | 1.0000 |
| 18 | wrong_target_same_relation_frame:is_a:L10->L20:object_first | 48 | 46 | 4.8759 | 4.7461 | 0.1298 | 0.7391 | 1.0000 |
| 19 | wrong_target_same_relation_frame:can_do:L10->L20:object_first | 48 | 38 | 4.5888 | 4.2555 | 0.3333 | 0.5789 | 0.9737 |
| 20 | wrong_target_same_relation_frame:can_do:L10->L20:object_last | 48 | 38 | 4.5888 | 4.2555 | 0.3333 | 0.5789 | 0.9737 |
| 21 | wrong_target_same_relation_frame:property:L10->L20:object_last | 48 | 29 | 3.0998 | 2.9706 | 0.1292 | 0.6897 | 0.9655 |
| 22 | wrong_target_same_relation_frame:property:L10->L20:object_first | 48 | 29 | 3.0959 | 2.9696 | 0.1263 | 0.6897 | 0.9655 |
| 23 | wrong_target_same_relation_frame:location:L10->L20:object_last | 48 | 40 | 2.9336 | 2.5114 | 0.4222 | 0.6500 | 0.9750 |
| 24 | wrong_target_same_relation_frame:part_of:L10->L20:object_first | 48 | 40 | 2.5518 | 2.4893 | 0.0624 | 0.7750 | 0.9750 |
| 25 | wrong_target_same_relation_frame:part_of:L10->L20:object_last | 48 | 40 | 2.5518 | 2.4893 | 0.0624 | 0.7750 | 0.9750 |
| 26 | wrong_target_same_relation_frame:location:L10->L20:object_first | 48 | 40 | 2.2444 | 1.9322 | 0.3122 | 0.7500 | 0.9750 |
| 27 | same_target_same_relation_frame:location:L4->L10:object_first | 33 | 27 | 1.7773 | 1.1397 | 0.6376 | 0.7407 | 0.8519 |
| 28 | wrong_target_same_relation_frame:material:L10->L20:object_first | 48 | 38 | 1.6999 | 1.7159 | -0.0160 | 0.8421 | 1.0000 |
| 29 | wrong_target_same_relation_frame:material:L10->L20:object_last | 48 | 38 | 1.6999 | 1.7159 | -0.0160 | 0.8421 | 1.0000 |
| 30 | same_target_same_relation_frame:can_do:L4->L10:object_first | 27 | 23 | 1.1744 | 0.6486 | 0.5258 | 0.8261 | 0.9130 |
| 31 | same_target_same_relation_frame:can_do:L4->L10:object_last | 27 | 23 | 1.1744 | 0.6486 | 0.5258 | 0.8261 | 0.9130 |
| 32 | same_target_same_relation_frame:location:L4->L10:object_last | 33 | 27 | 0.7704 | 0.4662 | 0.3043 | 0.8519 | 0.9259 |
| 33 | same_object_different_relation:material:L4->L10:object_first | 9 | 5 | 0.3722 | 0.0784 | 0.2938 | 0.8000 | 1.0000 |
| 34 | same_object_different_relation:material:L4->L10:object_last | 9 | 5 | 0.3722 | 0.0784 | 0.2938 | 0.8000 | 1.0000 |
| 35 | same_target_same_relation_frame:location:L10->L20:object_first | 33 | 27 | 0.3633 | 0.3174 | 0.0459 | 0.9259 | 1.0000 |
| 36 | same_target_same_relation_frame:can_do:L10->L20:object_first | 27 | 23 | 0.2446 | 0.1635 | 0.0811 | 1.0000 | 0.9565 |
| 37 | same_target_same_relation_frame:can_do:L10->L20:object_last | 27 | 23 | 0.2446 | 0.1635 | 0.0811 | 1.0000 | 0.9565 |
| 38 | same_object_different_relation:used_for:L4->L10:object_first | 9 | 9 | 0.1696 | 0.1598 | 0.0097 | 1.0000 | 1.0000 |
| 39 | same_object_different_relation:used_for:L4->L10:object_last | 9 | 9 | 0.1696 | 0.1598 | 0.0097 | 1.0000 | 1.0000 |
| 40 | same_object_same_relation_other_frame:can_do:L4->L10:object_first | 48 | 38 | 0.1616 | 0.0186 | 0.1430 | 0.9737 | 1.0000 |

## deepseek7b

items=336, rows=3900, layer_pairs=[[8, 10], [12, 14]]
control_types=['wrong_target_same_relation_frame', 'same_target_same_relation_frame', 'same_object_same_relation_other_frame', 'same_object_different_relation']

### By control type

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame | 1344 | 716 | 4.5743 | 4.0297 | 0.5446 | 0.5503 | 0.9302 |
| 2 | same_target_same_relation_frame | 840 | 468 | 0.1491 | 0.0311 | 0.1180 | 0.9359 | 0.9444 |
| 3 | same_object_same_relation_other_frame | 1344 | 716 | 0.1462 | 0.1435 | 0.0027 | 0.9860 | 0.9972 |
| 4 | same_object_different_relation | 372 | 184 | 0.1194 | 0.0720 | 0.0474 | 0.9891 | 1.0000 |

### Top control paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:L8->L10:object_last | 336 | 179 | 5.4433 | 4.4255 | 1.0178 | 0.4693 | 0.8827 |
| 2 | wrong_target_same_relation_frame:L8->L10:object_first | 336 | 179 | 5.2762 | 4.2474 | 1.0288 | 0.4804 | 0.8827 |
| 3 | wrong_target_same_relation_frame:L12->L14:object_last | 336 | 179 | 3.8526 | 3.7800 | 0.0726 | 0.6201 | 0.9777 |
| 4 | wrong_target_same_relation_frame:L12->L14:object_first | 336 | 179 | 3.7253 | 3.6661 | 0.0592 | 0.6313 | 0.9777 |
| 5 | same_target_same_relation_frame:L8->L10:object_first | 210 | 117 | 0.2556 | 0.0827 | 0.1729 | 0.9145 | 0.9402 |
| 6 | same_target_same_relation_frame:L8->L10:object_last | 210 | 117 | 0.2082 | 0.0258 | 0.1824 | 0.9145 | 0.9402 |
| 7 | same_object_same_relation_other_frame:L8->L10:object_first | 336 | 179 | 0.1500 | 0.1350 | 0.0150 | 0.9888 | 0.9944 |
| 8 | same_object_same_relation_other_frame:L12->L14:object_first | 336 | 179 | 0.1481 | 0.1450 | 0.0031 | 0.9832 | 1.0000 |
| 9 | same_object_same_relation_other_frame:L8->L10:object_last | 336 | 179 | 0.1442 | 0.1503 | -0.0061 | 0.9888 | 0.9944 |
| 10 | same_object_same_relation_other_frame:L12->L14:object_last | 336 | 179 | 0.1424 | 0.1438 | -0.0014 | 0.9832 | 1.0000 |
| 11 | same_object_different_relation:L8->L10:object_first | 93 | 46 | 0.1201 | 0.0538 | 0.0664 | 0.9783 | 1.0000 |
| 12 | same_object_different_relation:L8->L10:object_last | 93 | 46 | 0.1201 | 0.0538 | 0.0664 | 0.9783 | 1.0000 |
| 13 | same_object_different_relation:L12->L14:object_first | 93 | 46 | 0.1186 | 0.0903 | 0.0283 | 1.0000 | 1.0000 |
| 14 | same_object_different_relation:L12->L14:object_last | 93 | 46 | 0.1186 | 0.0903 | 0.0283 | 1.0000 | 1.0000 |
| 15 | same_target_same_relation_frame:L12->L14:object_first | 210 | 117 | 0.0711 | 0.0096 | 0.0615 | 0.9573 | 0.9487 |
| 16 | same_target_same_relation_frame:L12->L14:object_last | 210 | 117 | 0.0616 | 0.0063 | 0.0553 | 0.9573 | 0.9487 |

### Top control relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:is_a | 192 | 172 | 6.5960 | 6.0607 | 0.5352 | 0.4070 | 0.9535 |
| 2 | wrong_target_same_relation_frame:used_for | 192 | 108 | 6.4740 | 5.5925 | 0.8815 | 0.5741 | 0.9630 |
| 3 | wrong_target_same_relation_frame:can_do | 192 | 68 | 5.9283 | 5.5755 | 0.3528 | 0.5294 | 0.9118 |
| 4 | wrong_target_same_relation_frame:location | 192 | 64 | 3.0529 | 2.7214 | 0.3315 | 0.7500 | 0.9375 |
| 5 | wrong_target_same_relation_frame:property | 192 | 84 | 3.0459 | 2.4770 | 0.5689 | 0.5714 | 0.9048 |
| 6 | wrong_target_same_relation_frame:material | 192 | 108 | 2.9994 | 2.3970 | 0.6024 | 0.6111 | 0.8889 |
| 7 | wrong_target_same_relation_frame:part_of | 192 | 112 | 2.3501 | 1.9518 | 0.3983 | 0.5714 | 0.9286 |
| 8 | same_object_different_relation:used_for | 36 | 12 | 2.0644 | 1.5763 | 0.4880 | 1.0000 | 1.0000 |
| 9 | same_target_same_relation_frame:material | 156 | 84 | 0.4452 | 0.0283 | 0.4169 | 0.9524 | 0.8810 |
| 10 | same_target_same_relation_frame:is_a | 144 | 136 | 0.4041 | 0.2058 | 0.1983 | 0.9706 | 0.9853 |
| 11 | same_object_different_relation:property | 60 | 16 | 0.3571 | 0.2278 | 0.1293 | 0.8750 | 1.0000 |
| 12 | same_object_same_relation_other_frame:can_do | 192 | 68 | 0.3111 | 0.4168 | -0.1057 | 1.0000 | 1.0000 |
| 13 | same_object_same_relation_other_frame:used_for | 192 | 108 | 0.2780 | 0.2914 | -0.0133 | 0.9630 | 1.0000 |
| 14 | same_object_same_relation_other_frame:location | 192 | 64 | 0.2725 | 0.3834 | -0.1109 | 0.9688 | 1.0000 |
| 15 | same_object_same_relation_other_frame:is_a | 192 | 172 | 0.1573 | 0.0939 | 0.0634 | 1.0000 | 1.0000 |
| 16 | same_target_same_relation_frame:part_of | 108 | 72 | 0.1337 | 0.0417 | 0.0920 | 0.9444 | 0.9722 |
| 17 | same_object_same_relation_other_frame:material | 192 | 108 | 0.0832 | 0.0209 | 0.0624 | 0.9630 | 0.9815 |
| 18 | same_target_same_relation_frame:location | 132 | 44 | 0.0774 | 0.0994 | -0.0220 | 0.9545 | 0.9545 |
| 19 | same_object_same_relation_other_frame:part_of | 192 | 112 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 20 | same_object_different_relation:can_do | 84 | 20 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 21 | same_object_different_relation:location | 60 | 24 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 22 | same_object_same_relation_other_frame:property | 192 | 84 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 23 | same_target_same_relation_frame:used_for | 72 | 32 | -0.0142 | 0.3231 | -0.3372 | 0.8125 | 0.8750 |
| 24 | same_object_different_relation:is_a | 84 | 76 | -0.0477 | -0.0452 | -0.0025 | 1.0000 | 1.0000 |
| 25 | same_object_different_relation:part_of | 12 | 12 | -0.0625 | -0.0729 | 0.0104 | 1.0000 | 1.0000 |
| 26 | same_object_different_relation:material | 36 | 24 | -0.1726 | -0.2082 | 0.0356 | 1.0000 | 1.0000 |
| 27 | same_target_same_relation_frame:can_do | 108 | 56 | -0.2083 | -0.1695 | -0.0388 | 0.8571 | 0.9643 |
| 28 | same_target_same_relation_frame:property | 120 | 44 | -0.5335 | -0.5463 | 0.0127 | 0.9545 | 0.9091 |

### Top control relation paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_target_same_relation_frame:used_for:L8->L10:object_last | 48 | 27 | 7.9366 | 6.2927 | 1.6439 | 0.4444 | 0.9259 |
| 2 | wrong_target_same_relation_frame:used_for:L8->L10:object_first | 48 | 27 | 7.8489 | 6.1410 | 1.7079 | 0.4444 | 0.9259 |
| 3 | wrong_target_same_relation_frame:is_a:L8->L10:object_last | 48 | 43 | 7.7731 | 6.8522 | 0.9209 | 0.3256 | 0.9302 |
| 4 | wrong_target_same_relation_frame:is_a:L8->L10:object_first | 48 | 43 | 7.1327 | 6.2058 | 0.9268 | 0.3721 | 0.9302 |
| 5 | wrong_target_same_relation_frame:can_do:L8->L10:object_first | 48 | 17 | 7.0145 | 5.9400 | 1.0745 | 0.4706 | 0.8235 |
| 6 | wrong_target_same_relation_frame:can_do:L8->L10:object_last | 48 | 17 | 7.0145 | 5.9400 | 1.0745 | 0.4706 | 0.8235 |
| 7 | wrong_target_same_relation_frame:is_a:L12->L14:object_last | 48 | 43 | 6.0137 | 5.8386 | 0.1751 | 0.4419 | 0.9767 |
| 8 | wrong_target_same_relation_frame:is_a:L12->L14:object_first | 48 | 43 | 5.4645 | 5.3464 | 0.1181 | 0.4884 | 0.9767 |
| 9 | wrong_target_same_relation_frame:used_for:L12->L14:object_first | 48 | 27 | 5.0706 | 4.9826 | 0.0880 | 0.7037 | 1.0000 |
| 10 | wrong_target_same_relation_frame:used_for:L12->L14:object_last | 48 | 27 | 5.0398 | 4.9536 | 0.0862 | 0.7037 | 1.0000 |
| 11 | wrong_target_same_relation_frame:can_do:L12->L14:object_first | 48 | 17 | 4.8421 | 5.2109 | -0.3688 | 0.5882 | 1.0000 |
| 12 | wrong_target_same_relation_frame:can_do:L12->L14:object_last | 48 | 17 | 4.8421 | 5.2109 | -0.3688 | 0.5882 | 1.0000 |
| 13 | wrong_target_same_relation_frame:material:L8->L10:object_first | 48 | 27 | 3.7479 | 2.5159 | 1.2320 | 0.5185 | 0.7778 |
| 14 | wrong_target_same_relation_frame:material:L8->L10:object_last | 48 | 27 | 3.7479 | 2.5159 | 1.2320 | 0.5185 | 0.7778 |
| 15 | wrong_target_same_relation_frame:location:L8->L10:object_first | 48 | 16 | 3.3367 | 2.5317 | 0.8050 | 0.6875 | 0.8750 |
| 16 | wrong_target_same_relation_frame:location:L8->L10:object_last | 48 | 16 | 3.3367 | 2.5317 | 0.8050 | 0.6875 | 0.8750 |
| 17 | wrong_target_same_relation_frame:property:L8->L10:object_first | 48 | 21 | 3.2188 | 2.6202 | 0.5986 | 0.5238 | 0.9048 |
| 18 | wrong_target_same_relation_frame:property:L8->L10:object_last | 48 | 21 | 3.2188 | 2.6202 | 0.5986 | 0.5238 | 0.9048 |
| 19 | wrong_target_same_relation_frame:part_of:L8->L10:object_first | 48 | 28 | 3.0140 | 2.2564 | 0.7577 | 0.5000 | 0.8929 |
| 20 | wrong_target_same_relation_frame:part_of:L8->L10:object_last | 48 | 28 | 3.0140 | 2.2564 | 0.7577 | 0.5000 | 0.8929 |
| 21 | wrong_target_same_relation_frame:property:L12->L14:object_first | 48 | 21 | 2.8731 | 2.3338 | 0.5393 | 0.6190 | 0.9048 |
| 22 | wrong_target_same_relation_frame:property:L12->L14:object_last | 48 | 21 | 2.8731 | 2.3338 | 0.5393 | 0.6190 | 0.9048 |
| 23 | wrong_target_same_relation_frame:location:L12->L14:object_first | 48 | 16 | 2.7691 | 2.9112 | -0.1421 | 0.8125 | 1.0000 |
| 24 | wrong_target_same_relation_frame:location:L12->L14:object_last | 48 | 16 | 2.7691 | 2.9112 | -0.1421 | 0.8125 | 1.0000 |
| 25 | wrong_target_same_relation_frame:material:L12->L14:object_first | 48 | 27 | 2.2508 | 2.2780 | -0.0272 | 0.7037 | 1.0000 |
| 26 | wrong_target_same_relation_frame:material:L12->L14:object_last | 48 | 27 | 2.2508 | 2.2780 | -0.0272 | 0.7037 | 1.0000 |
| 27 | same_object_different_relation:used_for:L12->L14:object_first | 9 | 3 | 2.2414 | 1.4079 | 0.8335 | 1.0000 | 1.0000 |
| 28 | same_object_different_relation:used_for:L12->L14:object_last | 9 | 3 | 2.2414 | 1.4079 | 0.8335 | 1.0000 | 1.0000 |
| 29 | same_object_different_relation:used_for:L8->L10:object_first | 9 | 3 | 1.8874 | 1.7447 | 0.1426 | 1.0000 | 1.0000 |
| 30 | same_object_different_relation:used_for:L8->L10:object_last | 9 | 3 | 1.8874 | 1.7447 | 0.1426 | 1.0000 | 1.0000 |
| 31 | wrong_target_same_relation_frame:part_of:L12->L14:object_first | 48 | 28 | 1.6863 | 1.6473 | 0.0389 | 0.6429 | 0.9643 |
| 32 | wrong_target_same_relation_frame:part_of:L12->L14:object_last | 48 | 28 | 1.6863 | 1.6473 | 0.0389 | 0.6429 | 0.9643 |
| 33 | same_target_same_relation_frame:is_a:L8->L10:object_first | 36 | 34 | 0.5939 | 0.2488 | 0.3451 | 0.9706 | 1.0000 |
| 34 | same_target_same_relation_frame:material:L8->L10:object_first | 39 | 21 | 0.5168 | -0.1477 | 0.6645 | 0.9048 | 0.8095 |
| 35 | same_target_same_relation_frame:material:L8->L10:object_last | 39 | 21 | 0.5168 | -0.1477 | 0.6645 | 0.9048 | 0.8095 |
| 36 | same_target_same_relation_frame:is_a:L8->L10:object_last | 36 | 34 | 0.4307 | 0.0528 | 0.3779 | 0.9706 | 1.0000 |
| 37 | same_object_different_relation:property:L8->L10:object_first | 15 | 4 | 0.3997 | 0.1738 | 0.2259 | 0.7500 | 1.0000 |
| 38 | same_object_different_relation:property:L8->L10:object_last | 15 | 4 | 0.3997 | 0.1738 | 0.2259 | 0.7500 | 1.0000 |
| 39 | same_target_same_relation_frame:material:L12->L14:object_first | 39 | 21 | 0.3737 | 0.2044 | 0.1693 | 1.0000 | 0.9524 |
| 40 | same_target_same_relation_frame:material:L12->L14:object_last | 39 | 21 | 0.3737 | 0.2044 | 0.1693 | 1.0000 | 0.9524 |

