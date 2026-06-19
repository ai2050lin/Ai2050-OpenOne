# Phase530 State-Pair Decomposition Summary

## qwen3

layer=L12, train_n=10, test_n=8, alphas=[4.0, 8.0, 12.0], attn=sdpa

Cell format: best own Δ / selectivity ratio / pass

| candidate | family | own task | parallel norm % | perp norm % | full | parallel | perp |
|---|---|---|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 2.61 | 99.97 | +0.3151/1.34/n | +0.1641/1.66/n | +0.3151/1.26/n |
| color_red_blue_painted | color | color_red_blue | 1.52 | 99.99 | -0.0625/0.67/n | -0.1719/1.83/n | -0.0703/0.64/n |
| color_red_blue_property | color | color_red_blue | 1.60 | 99.99 | -0.2057/2.19/n | -0.1719/1.83/n | -0.2057/2.63/n |
| color_black_white_direct | color | color_black_white | 0.55 | 100.00 | +0.1797/1.25/n | +0.1120/0.36/n | +0.1797/2.76/Y |
| color_black_white_painted | color | color_black_white | 1.60 | 99.99 | -0.0755/0.64/n | +0.1120/0.36/n | -0.0339/0.32/n |
| color_black_white_property | color | color_black_white | 2.69 | 99.96 | -0.0833/1.07/n | +0.1120/0.36/n | -0.1042/1.11/n |
| object_desc_apple_banana | object | object_apple_banana | 0.29 | 100.00 | -0.1562/1.82/n | +0.0312/0.09/n | -0.1094/1.11/n |
| object_desc_car_truck | object | object_car_truck | 1.40 | 99.99 | +0.3281/1.02/n | +0.2969/2.53/Y | +0.3281/1.05/n |

## glm4

layer=L26, train_n=10, test_n=8, alphas=[4.0, 8.0, 12.0], attn=sdpa

Cell format: best own Δ / selectivity ratio / pass

| candidate | family | own task | parallel norm % | perp norm % | full | parallel | perp |
|---|---|---|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 32.06 | 94.72 | +6.7593/3.95/Y | +8.4660/7.86/Y | +4.4932/3.05/Y |
| color_red_blue_painted | color | color_red_blue | 30.93 | 95.10 | +6.7326/3.66/Y | +8.4660/7.86/Y | +4.5342/2.78/Y |
| color_red_blue_property | color | color_red_blue | 30.94 | 95.09 | +7.0353/3.80/Y | +8.4660/7.86/Y | +4.8382/2.99/Y |
| color_black_white_direct | color | color_black_white | 30.27 | 95.31 | +4.4144/12.64/Y | +7.3963/17.07/Y | +2.4186/23.87/Y |
| color_black_white_painted | color | color_black_white | 28.42 | 95.88 | +4.7318/4.61/Y | +7.3963/17.07/Y | +2.8210/2.54/Y |
| color_black_white_property | color | color_black_white | 35.01 | 93.67 | +5.0495/12.65/Y | +7.3963/17.07/Y | +3.0234/8.44/Y |
| object_desc_apple_banana | object | object_apple_banana | 5.24 | 99.86 | +1.5569/3.89/Y | +12.7043/85.31/Y | +0.7777/1.98/n |
| object_desc_car_truck | object | object_car_truck | 5.56 | 99.85 | +1.6980/3.44/Y | +7.7021/23.25/Y | +1.2542/2.50/Y |

## Cross-model Compact

| model | passed components | best BW full | best RB full | best object full | best object perp |
|---|---|---|---|---|---|
| qwen3 | color_black_white_direct:perp,object_desc_car_truck:parallel | color_black_white_direct +0.180/1.25/n | color_red_blue_direct +0.315/1.34/n | object_desc_car_truck +0.328/1.02/n | object_desc_car_truck +0.328/1.05/n |
| glm4 | color_red_blue_direct:full,color_red_blue_direct:parallel,color_red_blue_direct:perp,color_red_blue_painted:full,color_red_blue_painted:parallel,color_red_blue_painted:perp,color_red_blue_property:full,color_red_blue_property:parallel,color_red_blue_property:perp,color_black_white_direct:full,color_black_white_direct:parallel,color_black_white_direct:perp,color_black_white_painted:full,color_black_white_painted:parallel,color_black_white_painted:perp,color_black_white_property:full,color_black_white_property:parallel,color_black_white_property:perp,object_desc_apple_banana:full,object_desc_apple_banana:parallel,object_desc_car_truck:full,object_desc_car_truck:parallel,object_desc_car_truck:perp | color_black_white_property +5.049/12.65/Y | color_red_blue_property +7.035/3.80/Y | object_desc_car_truck +1.698/3.44/Y | object_desc_car_truck +1.254/2.50/Y |

