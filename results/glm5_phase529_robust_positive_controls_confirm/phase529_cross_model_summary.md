# Phase529 Robust Positive Controls Summary

## qwen3

layer=L12, train_n=10, test_n=8, alphas=[4.0, 8.0, 12.0], attn=sdpa

### category

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| category_fruit | category_fruit | 12.0 | +2.5195 | 0.0000 | 0.6771 | 3.7212 | yes | 6.45 | 99.79 |

### color

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| color_black_white | color_black_white | 8.0 | +0.4531 | 0.1719 | 0.0625 | 2.6364 | yes | 0.55 | 100.00 |
| color_red_blue | color_red_blue | 12.0 | +0.4297 | 0.8359 | 0.7812 | 0.5140 | no | 2.61 | 99.97 |
| color_green_yellow | color_green_yellow | 4.0 | -0.0703 | 0.3125 | 0.1562 | 0.2250 | no | 0.97 | 100.00 |
| color_all_pairs | color_red_blue | 4.0 | -0.0703 | 0.1094 | 0.1484 | 0.4737 | no | 1.83 | 99.98 |

### object

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| object_apple_banana | object_apple_banana | 12.0 | +0.2917 | 0.1250 | 0.4180 | 0.6978 | no | 2.32 | 99.97 |
| object_all_pairs | object_apple_banana | 12.0 | +0.1562 | 0.3333 | 0.8438 | 0.1852 | no | 2.29 | 99.97 |
| object_shirt_jacket | object_shirt_jacket | 4.0 | -0.0417 | 0.0833 | 0.3750 | 0.1111 | no | 1.10 | 99.99 |
| object_car_truck | object_car_truck | 4.0 | -0.1458 | 0.2083 | 0.2734 | 0.5333 | no | 2.98 | 99.96 |

## glm4

layer=L26, train_n=10, test_n=8, alphas=[4.0, 8.0, 12.0], attn=sdpa

### category

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| category_fruit | category_fruit | 4.0 | -0.0049 | 0.0000 | 0.3516 | 0.0139 | no | 3.71 | 99.93 |

### color

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| color_red_blue | color_red_blue | 12.0 | +7.3688 | 0.3477 | 0.6615 | 11.1402 | yes | 32.06 | 94.72 |
| color_black_white | color_black_white | 12.0 | +5.3223 | 2.1484 | 0.8594 | 2.4773 | yes | 30.27 | 95.31 |
| color_all_pairs | color_red_blue | 12.0 | +4.0547 | 3.4375 | 0.9583 | 1.1795 | no | 16.77 | 98.58 |
| color_green_yellow | color_green_yellow | 12.0 | +3.6934 | 0.9648 | 2.0938 | 1.7640 | no | 20.87 | 97.80 |

### object

| candidate | own task | best alpha | own Δ | same-family max abs Δ | off-family max abs Δ | ratio | pass | readout % | semantic % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| object_car_truck | object_car_truck | 12.0 | +3.0846 | 0.9219 | 0.6953 | 3.3460 | yes | 13.96 | 99.02 |
| object_shirt_jacket | object_shirt_jacket | 12.0 | +2.8516 | 0.5833 | 0.7295 | 3.9090 | yes | 9.80 | 99.52 |
| object_apple_banana | object_apple_banana | 12.0 | +2.3776 | 0.4062 | 1.1445 | 2.0774 | yes | 15.33 | 98.82 |
| object_all_pairs | object_apple_banana | 12.0 | +1.3307 | 1.4609 | 1.2734 | 0.9109 | no | 11.17 | 99.37 |

## Cross-model Compact

| model | passed candidates | best color | color own Δ | color ratio | best object | object own Δ | object ratio |
|---|---|---|---:|---:|---|---:|---:|
| qwen3 | category_fruit,color_black_white | color_black_white | +0.4531 | 2.6364 | object_apple_banana | +0.2917 | 0.6978 |
| glm4 | color_red_blue,color_black_white,object_apple_banana,object_car_truck,object_shirt_jacket | color_red_blue | +7.3688 | 11.1402 | object_car_truck | +3.0846 | 3.3460 |

