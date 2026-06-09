# Phase75 Relation-Frame Token Intervention Summary

## qwen3

items=216, rows=2592, layer_pairs=[[4, 8], [8, 12]]
control_types=['wrong_relation_same_object', 'same_relation_other_frame', 'same_relation_frame_other_object']

### By control type

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_relation_same_object | 864 | 576 | 1.6677 | 1.6606 | 0.0071 | 0.8160 | 0.9392 |
| 2 | same_relation_other_frame | 864 | 576 | 0.3143 | 0.2620 | 0.0523 | 0.9323 | 0.9670 |
| 3 | same_relation_frame_other_object | 864 | 576 | 0.1397 | 0.0978 | 0.0419 | 0.9566 | 0.9844 |

### Top control paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_relation_same_object:L8->L12:frame_last | 216 | 144 | 2.8263 | 2.8474 | -0.0211 | 0.7431 | 0.9514 |
| 2 | wrong_relation_same_object:L4->L8:frame_last | 216 | 144 | 2.7624 | 2.7346 | 0.0278 | 0.7083 | 0.9375 |
| 3 | wrong_relation_same_object:L4->L8:frame_first | 216 | 144 | 0.6525 | 0.5580 | 0.0945 | 0.8889 | 0.9097 |
| 4 | same_relation_other_frame:L4->L8:frame_first | 216 | 144 | 0.4861 | 0.3544 | 0.1317 | 0.9236 | 0.9514 |
| 5 | same_relation_frame_other_object:L8->L12:frame_last | 216 | 144 | 0.4655 | 0.3356 | 0.1299 | 0.9167 | 0.9722 |
| 6 | wrong_relation_same_object:L8->L12:frame_first | 216 | 144 | 0.4295 | 0.5023 | -0.0728 | 0.9236 | 0.9583 |
| 7 | same_relation_other_frame:L4->L8:frame_last | 216 | 144 | 0.4251 | 0.3426 | 0.0825 | 0.9097 | 0.9653 |
| 8 | same_relation_other_frame:L8->L12:frame_first | 216 | 144 | 0.1854 | 0.2146 | -0.0292 | 0.9722 | 0.9722 |
| 9 | same_relation_other_frame:L8->L12:frame_last | 216 | 144 | 0.1606 | 0.1365 | 0.0241 | 0.9236 | 0.9792 |
| 10 | same_relation_frame_other_object:L8->L12:frame_first | 216 | 144 | 0.1045 | 0.0648 | 0.0397 | 0.9514 | 0.9792 |
| 11 | same_relation_frame_other_object:L4->L8:frame_last | 216 | 144 | 0.0237 | 0.0210 | 0.0026 | 0.9792 | 1.0000 |
| 12 | same_relation_frame_other_object:L4->L8:frame_first | 216 | 144 | -0.0350 | -0.0302 | -0.0048 | 0.9792 | 0.9861 |

### Top control relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_relation_same_object:used_for | 144 | 60 | 3.1260 | 3.0223 | 0.1036 | 0.7667 | 0.9500 |
| 2 | wrong_relation_same_object:can_do | 144 | 108 | 1.7359 | 1.8389 | -0.1029 | 0.8333 | 0.9259 |
| 3 | wrong_relation_same_object:is_a | 144 | 144 | 1.6487 | 1.6950 | -0.0463 | 0.9097 | 0.9722 |
| 4 | wrong_relation_same_object:material | 144 | 104 | 1.5342 | 1.5400 | -0.0058 | 0.7308 | 0.9615 |
| 5 | same_relation_other_frame:used_for | 144 | 60 | 1.4775 | 1.4224 | 0.0551 | 0.8833 | 0.9667 |
| 6 | wrong_relation_same_object:location | 144 | 84 | 1.4671 | 1.2904 | 0.1766 | 0.8095 | 0.8690 |
| 7 | same_relation_other_frame:property | 144 | 76 | 0.8954 | 0.7502 | 0.1452 | 0.8421 | 0.9342 |
| 8 | wrong_relation_same_object:property | 144 | 76 | 0.8598 | 0.8411 | 0.0188 | 0.7763 | 0.9342 |
| 9 | same_relation_other_frame:material | 144 | 104 | 0.3529 | 0.3376 | 0.0152 | 0.9327 | 0.9904 |
| 10 | same_relation_frame_other_object:property | 144 | 76 | 0.2101 | 0.1800 | 0.0301 | 0.9211 | 1.0000 |
| 11 | same_relation_frame_other_object:material | 144 | 104 | 0.1571 | 0.1672 | -0.0100 | 0.9231 | 0.9712 |
| 12 | same_relation_frame_other_object:used_for | 144 | 60 | 0.1559 | 0.1222 | 0.0336 | 0.9500 | 0.9667 |
| 13 | same_relation_frame_other_object:is_a | 144 | 144 | 0.1543 | 0.1242 | 0.0301 | 1.0000 | 1.0000 |
| 14 | same_relation_other_frame:location | 144 | 84 | 0.1162 | -0.0627 | 0.1789 | 0.9286 | 0.9524 |
| 15 | same_relation_frame_other_object:can_do | 144 | 108 | 0.1087 | -0.0218 | 0.1305 | 0.9722 | 0.9907 |
| 16 | same_relation_frame_other_object:location | 144 | 84 | 0.0575 | 0.0286 | 0.0289 | 0.9405 | 0.9643 |
| 17 | same_relation_other_frame:can_do | 144 | 108 | -0.0294 | -0.0937 | 0.0643 | 0.9444 | 0.9352 |
| 18 | same_relation_other_frame:is_a | 144 | 144 | -0.1315 | -0.0775 | -0.0541 | 0.9931 | 1.0000 |

## glm4

items=216, rows=2592, layer_pairs=[[4, 10], [10, 20]]
control_types=['wrong_relation_same_object', 'same_relation_other_frame', 'same_relation_frame_other_object']

### By control type

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_relation_same_object | 864 | 612 | 1.1403 | 1.2104 | -0.0701 | 0.8660 | 0.9641 |
| 2 | same_relation_frame_other_object | 864 | 612 | 0.4396 | 0.4231 | 0.0165 | 0.9232 | 0.9592 |
| 3 | same_relation_other_frame | 864 | 612 | 0.2553 | 0.2783 | -0.0229 | 0.9003 | 0.9641 |

### Top control paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_relation_same_object:L10->L20:frame_last | 216 | 153 | 2.0501 | 2.1373 | -0.0872 | 0.7778 | 0.9673 |
| 2 | wrong_relation_same_object:L4->L10:frame_last | 216 | 153 | 1.9294 | 2.0396 | -0.1103 | 0.7974 | 0.9608 |
| 3 | same_relation_frame_other_object:L10->L20:frame_first | 216 | 153 | 0.9687 | 0.9132 | 0.0555 | 0.8562 | 0.9608 |
| 4 | same_relation_frame_other_object:L10->L20:frame_last | 216 | 153 | 0.8356 | 0.8042 | 0.0314 | 0.8824 | 0.9412 |
| 5 | wrong_relation_same_object:L10->L20:frame_first | 216 | 153 | 0.3269 | 0.3136 | 0.0132 | 0.9542 | 0.9804 |
| 6 | same_relation_other_frame:L4->L10:frame_first | 216 | 153 | 0.3221 | 0.3978 | -0.0757 | 0.8889 | 0.9673 |
| 7 | same_relation_other_frame:L10->L20:frame_first | 216 | 153 | 0.3011 | 0.3005 | 0.0006 | 0.9281 | 0.9673 |
| 8 | wrong_relation_same_object:L4->L10:frame_first | 216 | 153 | 0.2548 | 0.3509 | -0.0961 | 0.9346 | 0.9477 |
| 9 | same_relation_other_frame:L4->L10:frame_last | 216 | 153 | 0.2136 | 0.2138 | -0.0002 | 0.8889 | 0.9542 |
| 10 | same_relation_other_frame:L10->L20:frame_last | 216 | 153 | 0.1845 | 0.2009 | -0.0164 | 0.8954 | 0.9673 |
| 11 | same_relation_frame_other_object:L4->L10:frame_first | 216 | 153 | 0.0050 | 0.0366 | -0.0316 | 0.9739 | 0.9542 |
| 12 | same_relation_frame_other_object:L4->L10:frame_last | 216 | 153 | -0.0509 | -0.0615 | 0.0106 | 0.9804 | 0.9804 |

### Top control relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_relation_same_object:used_for | 144 | 88 | 2.4622 | 2.1074 | 0.3549 | 0.7841 | 0.9545 |
| 2 | wrong_relation_same_object:can_do | 144 | 120 | 1.3125 | 1.5228 | -0.2103 | 0.9000 | 0.9667 |
| 3 | wrong_relation_same_object:is_a | 144 | 144 | 1.2839 | 1.3569 | -0.0730 | 0.9514 | 1.0000 |
| 4 | same_relation_frame_other_object:is_a | 144 | 144 | 1.0175 | 0.9467 | 0.0708 | 0.9097 | 0.9792 |
| 5 | wrong_relation_same_object:location | 144 | 88 | 0.9172 | 1.1956 | -0.2784 | 0.7955 | 0.9659 |
| 6 | same_relation_other_frame:used_for | 144 | 88 | 0.7559 | 0.6208 | 0.1350 | 0.8636 | 0.9659 |
| 7 | same_relation_other_frame:property | 144 | 100 | 0.7404 | 0.7007 | 0.0397 | 0.7100 | 0.8900 |
| 8 | wrong_relation_same_object:material | 144 | 72 | 0.6877 | 0.7018 | -0.0141 | 0.8750 | 0.9722 |
| 9 | same_relation_frame_other_object:can_do | 144 | 120 | 0.4856 | 0.4561 | 0.0295 | 0.9250 | 0.9583 |
| 10 | same_relation_frame_other_object:used_for | 144 | 88 | 0.3833 | 0.3857 | -0.0024 | 0.9205 | 0.9432 |
| 11 | same_relation_other_frame:material | 144 | 72 | 0.2545 | 0.2291 | 0.0254 | 0.9306 | 0.9861 |
| 12 | same_relation_other_frame:can_do | 144 | 120 | 0.1990 | 0.3252 | -0.1261 | 0.9500 | 0.9833 |
| 13 | same_relation_frame_other_object:location | 144 | 88 | 0.1711 | 0.1533 | 0.0178 | 0.9545 | 0.9659 |
| 14 | same_relation_frame_other_object:material | 144 | 72 | 0.1618 | 0.2388 | -0.0770 | 0.9028 | 0.9861 |
| 15 | wrong_relation_same_object:property | 144 | 100 | 0.0859 | 0.2145 | -0.1286 | 0.8300 | 0.9100 |
| 16 | same_relation_frame_other_object:property | 144 | 100 | 0.0382 | 0.0327 | 0.0054 | 0.9300 | 0.9200 |
| 17 | same_relation_other_frame:location | 144 | 88 | -0.0455 | 0.0706 | -0.1160 | 0.9205 | 0.9432 |
| 18 | same_relation_other_frame:is_a | 144 | 144 | -0.1563 | -0.1120 | -0.0443 | 0.9861 | 1.0000 |

## deepseek7b

items=216, rows=2592, layer_pairs=[[8, 10], [12, 14]]
control_types=['wrong_relation_same_object', 'same_relation_other_frame', 'same_relation_frame_other_object']

### By control type

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_relation_same_object | 864 | 372 | 1.6177 | 1.6495 | -0.0318 | 0.7849 | 0.9704 |
| 2 | same_relation_frame_other_object | 864 | 372 | 0.7954 | 0.8009 | -0.0055 | 0.8817 | 0.9677 |
| 3 | same_relation_other_frame | 864 | 372 | 0.5184 | 0.5656 | -0.0472 | 0.8898 | 0.9704 |

### Top control paths

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_relation_same_object:L8->L10:frame_last | 216 | 93 | 2.9246 | 2.7970 | 0.1276 | 0.6989 | 0.9570 |
| 2 | wrong_relation_same_object:L12->L14:frame_last | 216 | 93 | 2.6094 | 2.6047 | 0.0047 | 0.6882 | 0.9570 |
| 3 | same_relation_frame_other_object:L12->L14:frame_last | 216 | 93 | 1.3782 | 1.3211 | 0.0571 | 0.8280 | 0.9677 |
| 4 | same_relation_frame_other_object:L8->L10:frame_last | 216 | 93 | 0.7645 | 0.7483 | 0.0163 | 0.8602 | 0.9677 |
| 5 | same_relation_other_frame:L12->L14:frame_last | 216 | 93 | 0.7148 | 0.6404 | 0.0745 | 0.8280 | 0.9462 |
| 6 | same_relation_other_frame:L8->L10:frame_last | 216 | 93 | 0.6380 | 0.6795 | -0.0414 | 0.8602 | 0.9785 |
| 7 | same_relation_frame_other_object:L12->L14:frame_first | 216 | 93 | 0.5869 | 0.6680 | -0.0812 | 0.8925 | 0.9570 |
| 8 | wrong_relation_same_object:L8->L10:frame_first | 216 | 93 | 0.5063 | 0.6478 | -0.1415 | 0.8710 | 0.9785 |
| 9 | same_relation_other_frame:L8->L10:frame_first | 216 | 93 | 0.4991 | 0.6548 | -0.1557 | 0.9140 | 0.9785 |
| 10 | same_relation_frame_other_object:L8->L10:frame_first | 216 | 93 | 0.4518 | 0.4660 | -0.0142 | 0.9462 | 0.9785 |
| 11 | wrong_relation_same_object:L12->L14:frame_first | 216 | 93 | 0.4303 | 0.5484 | -0.1181 | 0.8817 | 0.9892 |
| 12 | same_relation_other_frame:L12->L14:frame_first | 216 | 93 | 0.2215 | 0.2876 | -0.0661 | 0.9570 | 0.9785 |

### Top control relations

| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | wrong_relation_same_object:can_do | 144 | 84 | 2.6444 | 2.7255 | -0.0812 | 0.7143 | 0.9881 |
| 2 | wrong_relation_same_object:used_for | 144 | 52 | 2.3401 | 2.4907 | -0.1506 | 0.8462 | 0.9808 |
| 3 | same_relation_other_frame:property | 144 | 44 | 1.8287 | 1.6481 | 0.1806 | 0.6818 | 0.9091 |
| 4 | wrong_relation_same_object:material | 144 | 44 | 1.6311 | 1.6731 | -0.0420 | 0.5909 | 0.9091 |
| 5 | same_relation_other_frame:material | 144 | 44 | 1.2916 | 1.2597 | 0.0319 | 0.6364 | 0.9091 |
| 6 | wrong_relation_same_object:is_a | 144 | 120 | 1.1070 | 1.1168 | -0.0098 | 0.9083 | 0.9917 |
| 7 | wrong_relation_same_object:property | 144 | 44 | 1.0765 | 0.9696 | 0.1069 | 0.6818 | 0.9318 |
| 8 | same_relation_frame_other_object:material | 144 | 44 | 1.0575 | 0.8440 | 0.2135 | 0.7500 | 0.9091 |
| 9 | same_relation_frame_other_object:can_do | 144 | 84 | 1.0390 | 1.1730 | -0.1341 | 0.9643 | 1.0000 |
| 10 | same_relation_frame_other_object:property | 144 | 44 | 0.7243 | 0.5855 | 0.1387 | 0.6591 | 0.9091 |
| 11 | same_relation_frame_other_object:used_for | 144 | 52 | 0.7236 | 0.6455 | 0.0781 | 0.8462 | 0.9615 |
| 12 | same_relation_frame_other_object:is_a | 144 | 120 | 0.6924 | 0.7797 | -0.0873 | 0.9750 | 1.0000 |
| 13 | same_relation_frame_other_object:location | 144 | 28 | 0.3387 | 0.3340 | 0.0047 | 0.8571 | 0.9286 |
| 14 | same_relation_other_frame:can_do | 144 | 84 | 0.2969 | 0.4583 | -0.1614 | 1.0000 | 0.9881 |
| 15 | same_relation_other_frame:location | 144 | 28 | 0.2813 | 0.2193 | 0.0620 | 0.9286 | 0.9643 |
| 16 | wrong_relation_same_object:location | 144 | 28 | 0.2137 | 0.1733 | 0.0404 | 0.8214 | 0.9643 |
| 17 | same_relation_other_frame:used_for | 144 | 52 | 0.1758 | 0.3662 | -0.1903 | 0.8846 | 1.0000 |
| 18 | same_relation_other_frame:is_a | 144 | 120 | 0.1132 | 0.1564 | -0.0432 | 0.9750 | 0.9917 |

