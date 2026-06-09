# Phase80 Orthogonal Factor Audit Summary

## qwen3

items=28, basis_items=28, rows=280, layer_pairs=[[4, 8]]
module=resid_out, contrast_rank=16, nuisance_rank=8, relations=['can_do', 'is_a', 'location', 'material', 'part_of', 'property', 'used_for']

### By condition

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_mismatched_frame_raw | 28 | 23 | 8.3538 | 11.6050 | -3.6463 | -1.2491 | 0.1304 | 0.4783 |
| 2 | joint_orth_template | 28 | 23 | 8.5016 | 11.7288 | -3.7941 | -1.1254 | 0.2609 | 0.4348 |
| 3 | joint_raw | 28 | 23 | 8.4826 | 11.6815 | -3.7751 | -1.1726 | 0.2609 | 0.4348 |
| 4 | joint_orth_position | 28 | 23 | 8.1007 | 11.3901 | -3.3932 | -1.4640 | 0.2609 | 0.4348 |
| 5 | joint_orth_value | 28 | 23 | 8.1059 | 11.3543 | -3.3984 | -1.4998 | 0.2609 | 0.4348 |
| 6 | joint_orth_all | 28 | 23 | 7.8394 | 11.1169 | -3.1319 | -1.7372 | 0.2609 | 0.4348 |
| 7 | joint_raw_restore_both | 28 | 23 | 0.8930 | 1.3638 | 3.8145 | -11.4904 | 0.9130 | 0.0000 |
| 8 | joint_value_basis_only | 28 | 23 | 0.1064 | 0.0837 | 4.6011 | -12.7705 | 0.9565 | 0.0000 |
| 9 | joint_position_basis_only | 28 | 23 | 0.0869 | 0.0382 | 4.6206 | -12.8159 | 1.0000 | 0.0000 |
| 10 | joint_template_basis_only | 28 | 23 | 0.0000 | 0.0000 | 4.7075 | -12.8542 | 1.0000 | 0.0000 |

### Top condition paths

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_mismatched_frame_raw:L4->L8 | 28 | 23 | 8.3538 | 11.6050 | -3.6463 | -1.2491 | 0.1304 | 0.4783 |
| 2 | joint_orth_template:L4->L8 | 28 | 23 | 8.5016 | 11.7288 | -3.7941 | -1.1254 | 0.2609 | 0.4348 |
| 3 | joint_raw:L4->L8 | 28 | 23 | 8.4826 | 11.6815 | -3.7751 | -1.1726 | 0.2609 | 0.4348 |
| 4 | joint_orth_position:L4->L8 | 28 | 23 | 8.1007 | 11.3901 | -3.3932 | -1.4640 | 0.2609 | 0.4348 |
| 5 | joint_orth_value:L4->L8 | 28 | 23 | 8.1059 | 11.3543 | -3.3984 | -1.4998 | 0.2609 | 0.4348 |
| 6 | joint_orth_all:L4->L8 | 28 | 23 | 7.8394 | 11.1169 | -3.1319 | -1.7372 | 0.2609 | 0.4348 |
| 7 | joint_raw_restore_both:L4->L8 | 28 | 23 | 0.8930 | 1.3638 | 3.8145 | -11.4904 | 0.9130 | 0.0000 |
| 8 | joint_value_basis_only:L4->L8 | 28 | 23 | 0.1064 | 0.0837 | 4.6011 | -12.7705 | 0.9565 | 0.0000 |
| 9 | joint_position_basis_only:L4->L8 | 28 | 23 | 0.0869 | 0.0382 | 4.6206 | -12.8159 | 1.0000 | 0.0000 |
| 10 | joint_template_basis_only:L4->L8 | 28 | 23 | 0.0000 | 0.0000 | 4.7075 | -12.8542 | 1.0000 | 0.0000 |

### Top condition relations

| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | joint_mismatched_frame_raw:part_of | 5 | 5 | 10.9394 | 17.1441 | -6.9932 | 3.6612 | 0.0000 | 1.0000 |
| 2 | joint_orth_template:is_a | 5 | 3 | 14.0062 | 15.6446 | -8.1237 | 3.8933 | 0.0000 | 1.0000 |
| 3 | joint_raw:is_a | 5 | 3 | 13.9409 | 15.5842 | -8.0583 | 3.8329 | 0.0000 | 1.0000 |
| 4 | joint_orth_position:is_a | 5 | 3 | 13.4992 | 15.4828 | -7.6167 | 3.7315 | 0.0000 | 1.0000 |
| 5 | joint_orth_value:is_a | 5 | 3 | 13.4197 | 15.1379 | -7.5371 | 3.3866 | 0.0000 | 1.0000 |
| 6 | joint_orth_all:is_a | 5 | 3 | 12.9807 | 15.1007 | -7.0982 | 3.3495 | 0.0000 | 1.0000 |
| 7 | joint_orth_template:can_do | 3 | 2 | 16.0702 | 17.3515 | -8.0862 | 1.8527 | 0.0000 | 0.5000 |
| 8 | joint_raw:can_do | 3 | 2 | 15.9093 | 17.2608 | -7.9252 | 1.7619 | 0.0000 | 0.5000 |
| 9 | joint_orth_value:can_do | 3 | 2 | 15.6059 | 16.9515 | -7.6219 | 1.4526 | 0.0000 | 0.5000 |
| 10 | joint_orth_all:can_do | 3 | 2 | 15.1399 | 16.8503 | -7.1559 | 1.3515 | 0.0000 | 0.5000 |
| 11 | joint_orth_position:can_do | 3 | 2 | 15.1705 | 16.8263 | -7.1864 | 1.3274 | 0.0000 | 0.5000 |
| 12 | joint_mismatched_frame_raw:can_do | 3 | 2 | 11.0064 | 14.7229 | -3.0223 | -0.7760 | 0.0000 | 0.5000 |
| 13 | joint_orth_template:location | 6 | 6 | 7.2867 | 12.0719 | -3.6563 | -1.6412 | 0.1667 | 0.5000 |
| 14 | joint_raw:location | 6 | 6 | 7.2770 | 12.0129 | -3.6465 | -1.7002 | 0.1667 | 0.5000 |
| 15 | joint_orth_value:location | 6 | 6 | 6.8983 | 11.8546 | -3.2679 | -1.8585 | 0.1667 | 0.5000 |
| 16 | joint_orth_position:location | 6 | 6 | 7.1382 | 11.6552 | -3.5077 | -2.0579 | 0.1667 | 0.5000 |
| 17 | joint_orth_all:location | 6 | 6 | 7.0044 | 11.4976 | -3.3739 | -2.2155 | 0.1667 | 0.5000 |
| 18 | joint_mismatched_frame_raw:material | 3 | 2 | 7.2654 | 8.2665 | -2.9920 | 1.3720 | 0.0000 | 0.5000 |
| 19 | joint_orth_value:material | 3 | 2 | 6.8992 | 7.0890 | -2.6258 | 0.1945 | 0.0000 | 0.5000 |
| 20 | joint_orth_template:material | 3 | 2 | 7.0652 | 7.0355 | -2.7918 | 0.1410 | 0.0000 | 0.5000 |
| 21 | joint_raw:material | 3 | 2 | 6.9970 | 7.0277 | -2.7236 | 0.1332 | 0.0000 | 0.5000 |
| 22 | joint_orth_all:material | 3 | 2 | 6.9994 | 6.6390 | -2.7260 | -0.2555 | 0.0000 | 0.5000 |
| 23 | joint_orth_position:material | 3 | 2 | 6.9800 | 6.5831 | -2.7066 | -0.3114 | 0.0000 | 0.5000 |
| 24 | joint_orth_template:part_of | 5 | 5 | 8.1809 | 11.4788 | -4.2348 | -2.0041 | 0.4000 | 0.4000 |
| 25 | joint_raw:part_of | 5 | 5 | 8.2477 | 11.4153 | -4.3015 | -2.0675 | 0.4000 | 0.4000 |
| 26 | joint_orth_position:part_of | 5 | 5 | 7.1743 | 10.8538 | -3.2281 | -2.6291 | 0.4000 | 0.4000 |
| 27 | joint_orth_value:part_of | 5 | 5 | 7.5633 | 10.8485 | -3.6171 | -2.6344 | 0.4000 | 0.4000 |
| 28 | joint_orth_all:part_of | 5 | 5 | 6.7056 | 10.3622 | -2.7594 | -3.1206 | 0.4000 | 0.4000 |
| 29 | joint_mismatched_frame_raw:location | 6 | 6 | 6.5475 | 11.1104 | -2.9170 | -2.6027 | 0.1667 | 0.3333 |
| 30 | joint_mismatched_frame_raw:used_for | 3 | 3 | 7.1806 | 10.3886 | -3.9801 | -4.6513 | 0.0000 | 0.3333 |
| 31 | joint_mismatched_frame_raw:is_a | 5 | 3 | 8.3011 | 9.8471 | -2.4186 | -1.9042 | 0.3333 | 0.3333 |
| 32 | joint_orth_template:used_for | 3 | 3 | 4.8150 | 10.0840 | -1.6145 | -4.9559 | 0.3333 | 0.0000 |
| 33 | joint_raw:used_for | 3 | 3 | 4.7982 | 10.0749 | -1.5977 | -4.9651 | 0.3333 | 0.0000 |
| 34 | joint_orth_position:used_for | 3 | 3 | 4.6365 | 9.9332 | -1.4360 | -5.1067 | 0.3333 | 0.0000 |
| 35 | joint_orth_all:used_for | 3 | 3 | 4.6011 | 9.6957 | -1.4006 | -5.3443 | 0.3333 | 0.0000 |
| 36 | joint_orth_value:used_for | 3 | 3 | 4.8323 | 9.6746 | -1.6318 | -5.3653 | 0.3333 | 0.0000 |
| 37 | joint_orth_position:property | 3 | 2 | 4.4541 | 7.3529 | 3.0433 | -3.0433 | 1.0000 | 0.0000 |
| 38 | joint_orth_template:property | 3 | 2 | 4.0893 | 6.9880 | 3.4081 | -3.4081 | 1.0000 | 0.0000 |
| 39 | joint_raw:property | 3 | 2 | 4.0845 | 6.9833 | 3.4129 | -3.4129 | 1.0000 | 0.0000 |
| 40 | joint_orth_all:property | 3 | 2 | 3.8637 | 6.7625 | 3.6336 | -3.6336 | 1.0000 | 0.0000 |
| 41 | joint_orth_value:property | 3 | 2 | 3.7318 | 6.6306 | 3.7655 | -3.7655 | 1.0000 | 0.0000 |
| 42 | joint_mismatched_frame_raw:property | 3 | 2 | 7.5832 | 3.9234 | -0.0859 | -6.4727 | 0.5000 | 0.0000 |
| 43 | joint_raw_restore_both:property | 3 | 2 | 2.2373 | 2.3918 | 5.2600 | -8.0044 | 1.0000 | 0.0000 |
| 44 | joint_raw_restore_both:used_for | 3 | 3 | 0.2731 | 2.2696 | 2.9274 | -12.7703 | 1.0000 | 0.0000 |
| 45 | joint_raw_restore_both:is_a | 5 | 3 | 1.8279 | 2.2359 | 4.0547 | -9.5154 | 1.0000 | 0.0000 |
| 46 | joint_raw_restore_both:can_do | 3 | 2 | 2.4467 | 2.1664 | 5.5373 | -13.3325 | 1.0000 | 0.0000 |
| 47 | joint_raw_restore_both:location | 6 | 6 | 0.9043 | 1.0359 | 2.7261 | -12.6772 | 0.6667 | 0.0000 |
| 48 | joint_raw_restore_both:part_of | 5 | 5 | 0.0373 | 0.9444 | 3.9089 | -12.5385 | 1.0000 | 0.0000 |
| 49 | joint_position_basis_only:used_for | 3 | 3 | 0.5188 | 0.5214 | 2.6817 | -14.5186 | 1.0000 | 0.0000 |
| 50 | joint_value_basis_only:material | 3 | 2 | 0.4420 | 0.4071 | 3.8314 | -6.4874 | 1.0000 | 0.0000 |
| 51 | joint_position_basis_only:part_of | 5 | 5 | -0.1281 | 0.2630 | 4.0743 | -13.2199 | 1.0000 | 0.0000 |
| 52 | joint_position_basis_only:property | 3 | 2 | 0.4753 | 0.2170 | 7.0221 | -10.1791 | 1.0000 | 0.0000 |
| 53 | joint_value_basis_only:part_of | 5 | 5 | 0.3306 | 0.1856 | 3.6156 | -13.2973 | 0.8000 | 0.0000 |
| 54 | joint_position_basis_only:location | 6 | 6 | 0.2400 | 0.1393 | 3.3905 | -13.5738 | 1.0000 | 0.0000 |
| 55 | joint_value_basis_only:used_for | 3 | 3 | -0.0056 | 0.1144 | 3.2061 | -14.9256 | 1.0000 | 0.0000 |
| 56 | joint_value_basis_only:location | 6 | 6 | -0.0890 | 0.0918 | 3.7194 | -13.6213 | 1.0000 | 0.0000 |
| 57 | joint_position_basis_only:material | 3 | 2 | -0.1903 | 0.0901 | 4.4637 | -6.8044 | 1.0000 | 0.0000 |
| 58 | joint_value_basis_only:property | 3 | 2 | 0.1739 | 0.0787 | 7.3235 | -10.3175 | 1.0000 | 0.0000 |
| 59 | joint_value_basis_only:is_a | 5 | 3 | 0.3120 | 0.0760 | 5.5705 | -11.6753 | 1.0000 | 0.0000 |
| 60 | joint_template_basis_only:is_a | 5 | 3 | 0.0000 | 0.0000 | 5.8825 | -11.7513 | 1.0000 | 0.0000 |
| 61 | joint_template_basis_only:part_of | 5 | 5 | 0.0000 | 0.0000 | 3.9462 | -13.4829 | 1.0000 | 0.0000 |
| 62 | joint_template_basis_only:property | 3 | 2 | 0.0000 | 0.0000 | 7.4974 | -10.3961 | 1.0000 | 0.0000 |
| 63 | joint_template_basis_only:material | 3 | 2 | 0.0000 | 0.0000 | 4.2734 | -6.8945 | 1.0000 | 0.0000 |
| 64 | joint_template_basis_only:location | 6 | 6 | 0.0000 | 0.0000 | 3.6304 | -13.7131 | 1.0000 | 0.0000 |
| 65 | joint_template_basis_only:can_do | 3 | 2 | 0.0000 | 0.0000 | 7.9841 | -15.4988 | 1.0000 | 0.0000 |
| 66 | joint_template_basis_only:used_for | 3 | 3 | 0.0000 | 0.0000 | 3.2005 | -15.0399 | 1.0000 | 0.0000 |
| 67 | joint_position_basis_only:is_a | 5 | 3 | -0.2431 | -0.4896 | 6.1256 | -12.2409 | 1.0000 | 0.0000 |
| 68 | joint_value_basis_only:can_do | 3 | 2 | -0.4119 | -0.5483 | 8.3959 | -16.0472 | 1.0000 | 0.0000 |
| 69 | joint_position_basis_only:can_do | 3 | 2 | -0.0988 | -0.9905 | 8.0829 | -16.4894 | 1.0000 | 0.0000 |
| 70 | joint_raw_restore_both:material | 3 | 2 | -0.3725 | -1.1018 | 4.6459 | -7.9963 | 1.0000 | 0.0000 |

## glm4

missing

## deepseek7b

missing

