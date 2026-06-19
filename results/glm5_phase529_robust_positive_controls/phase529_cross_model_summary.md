# Phase529 Robust Positive Controls Summary

## qwen3

layer=L12, train_n=8, test_n=6, alphas=[2.0, 4.0, 8.0, 12.0], attn=sdpa

### category

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| category_fruit | category_fruit | 12.0 | +2.6927 | 0.0000 | 0.6979 | 3.8582 | yes | 6.17 | 99.81 |

### color

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| color_red_blue | color_red_blue | 12.0 | +0.3854 | 0.9271 | 0.7604 | 0.4157 | no | 1.48 | 99.99 |
| color_black_white | color_black_white | 8.0 | +0.3542 | 0.3750 | 0.1562 | 0.9444 | no | 0.52 | 100.00 |
| color_all_pairs | color_red_blue | 2.0 | -0.0104 | 0.1667 | 0.0677 | 0.0625 | no | 0.74 | 100.00 |
| color_green_yellow | color_green_yellow | 2.0 | -0.0208 | 0.0833 | 0.0521 | 0.2500 | no | 1.54 | 99.99 |

### object

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| object_apple_banana | object_apple_banana | 12.0 | +0.2917 | 0.1250 | 0.4583 | 0.6364 | no | 2.32 | 99.97 |
| object_all_pairs | object_apple_banana | 12.0 | +0.1562 | 0.3333 | 0.7083 | 0.2206 | no | 2.29 | 99.97 |
| object_shirt_jacket | object_shirt_jacket | 2.0 | +0.0000 | 0.0521 | 0.2083 | 0.0000 | no | 1.10 | 99.99 |
| object_car_truck | object_car_truck | 2.0 | -0.0833 | 0.1354 | 0.1250 | 0.6154 | no | 2.98 | 99.96 |

## glm4

layer=L26, train_n=8, test_n=6, alphas=[2.0, 4.0, 8.0, 12.0], attn=sdpa

### category

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| category_fruit | category_fruit | 8.0 | +0.4310 | 0.0000 | 0.6458 | 0.6673 | no | 4.13 | 99.91 |

### color

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| color_red_blue | color_red_blue | 12.0 | +7.5111 | 0.4167 | 0.6979 | 10.7621 | yes | 31.49 | 94.91 |
| color_black_white | color_black_white | 12.0 | +5.3594 | 1.9896 | 0.9062 | 2.6937 | yes | 29.82 | 95.45 |
| color_all_pairs | color_red_blue | 12.0 | +4.0677 | 3.4323 | 1.3125 | 1.1851 | no | 15.68 | 98.76 |
| color_green_yellow | color_green_yellow | 12.0 | +3.5156 | 1.0938 | 1.6719 | 2.1028 | yes | 20.47 | 97.88 |

### object

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| object_car_truck | object_car_truck | 12.0 | +3.0846 | 0.9219 | 0.7083 | 3.3460 | yes | 13.96 | 99.02 |
| object_shirt_jacket | object_shirt_jacket | 12.0 | +2.8516 | 0.5833 | 0.7656 | 3.7245 | yes | 9.80 | 99.52 |
| object_apple_banana | object_apple_banana | 12.0 | +2.3776 | 0.4062 | 1.1953 | 1.9891 | no | 15.33 | 98.82 |
| object_all_pairs | object_apple_banana | 12.0 | +1.3307 | 1.4609 | 1.3385 | 0.9109 | no | 11.17 | 99.37 |

## deepseek7b

layer=L18, train_n=8, test_n=6, alphas=[2.0, 4.0, 8.0, 12.0], attn=sdpa

### category

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| category_fruit | category_fruit | 12.0 | +0.0312 | 0.0000 | 0.1667 | 0.1875 | no | 1.20 | 99.99 |

### color

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| color_black_white | color_black_white | 12.0 | +0.2500 | 0.2708 | 0.1979 | 0.9231 | no | 5.78 | 99.83 |
| color_all_pairs | color_red_blue | 12.0 | +0.1458 | 0.2292 | 0.0781 | 0.6364 | no | 0.52 | 100.00 |
| color_red_blue | color_red_blue | 8.0 | +0.0729 | 0.0833 | 0.0417 | 0.8750 | no | 0.63 | 100.00 |
| color_green_yellow | color_green_yellow | 12.0 | +0.0000 | 0.0208 | 0.1094 | 0.0000 | no | 0.48 | 100.00 |

### object

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| object_car_truck | object_car_truck | 8.0 | +0.0521 | 0.1250 | 0.1042 | 0.4167 | no | 1.69 | 99.99 |
| object_shirt_jacket | object_shirt_jacket | 4.0 | +0.0000 | 0.0312 | 0.1146 | 0.0000 | no | 0.46 | 100.00 |
| object_all_pairs | object_apple_banana | 12.0 | -0.0104 | 0.1302 | 0.1458 | 0.0714 | no | 0.04 | 100.00 |
| object_apple_banana | object_apple_banana | 2.0 | -0.0833 | 0.0469 | 0.0417 | 1.7778 | no | 1.02 | 99.99 |

## Cross-model Compact

| model | passed candidates | best color | color own Δ | color ratio | best object | object own Δ | object ratio |
|---|---|---|---:|---:|---|---:|---:|
| qwen3 | category_fruit | color_red_blue | +0.3854 | 0.4157 | object_apple_banana | +0.2917 | 0.6364 |
| glm4 | color_red_blue,color_green_yellow,color_black_white,object_car_truck,object_shirt_jacket | color_red_blue | +7.5111 | 10.7621 | object_car_truck | +3.0846 | 3.3460 |
| deepseek7b | none | color_black_white | +0.2500 | 0.9231 | object_car_truck | +0.0521 | 0.4167 |

