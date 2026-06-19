# Phase530 State-Pair Decomposition Summary

## qwen3

layer=L12, train_n=8, test_n=6, alphas=[4.0, 8.0, 12.0], attn=sdpa

Cell format: best own Δ / selectivity ratio / pass

| candidate | family | own task | parallel norm % | perp norm % | full | parallel | perp |
|---|---|---|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 1.48 | 99.99 | +0.3785/1.61/n | +0.1285/1.37/n | +0.3785/1.61/n |
| color_red_blue_painted | color | color_red_blue | 0.96 | 100.00 | +0.0069/0.06/n | -0.1215/1.30/n | -0.0069/0.06/n |
| color_red_blue_property | color | color_red_blue | 1.70 | 99.99 | -0.1458/1.33/n | -0.1215/1.30/n | -0.1354/1.73/n |
| color_black_white_direct | color | color_black_white | 0.52 | 100.00 | +0.1319/4.22/Y | +0.0451/0.16/n | +0.1250/2.67/Y |
| color_black_white_painted | color | color_black_white | 1.44 | 99.99 | +0.0139/0.14/n | +0.0451/0.16/n | +0.0035/0.05/n |
| color_black_white_property | color | color_black_white | 2.57 | 99.97 | -0.0347/0.44/n | +0.0451/0.16/n | -0.0347/0.37/n |
| object_desc_apple_banana | object | object_apple_banana | 0.29 | 100.00 | -0.1562/2.37/n | +0.0312/0.11/n | -0.1094/1.40/n |
| object_desc_car_truck | object | object_car_truck | 1.40 | 99.99 | +0.3281/1.17/n | +0.2969/2.59/Y | +0.3281/1.39/n |

## glm4

layer=L26, train_n=8, test_n=6, alphas=[4.0, 8.0, 12.0], attn=sdpa

Cell format: best own Δ / selectivity ratio / pass

| candidate | family | own task | parallel norm % | perp norm % | full | parallel | perp |
|---|---|---|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 31.49 | 94.91 | +6.4218/3.71/Y | +8.3403/7.74/Y | +4.3289/2.92/Y |
| color_red_blue_painted | color | color_red_blue | 30.73 | 95.16 | +6.3704/3.49/Y | +8.3403/7.74/Y | +4.2447/2.56/Y |
| color_red_blue_property | color | color_red_blue | 30.87 | 95.12 | +6.6513/3.51/Y | +8.3403/7.74/Y | +4.5582/2.79/Y |
| color_black_white_direct | color | color_black_white | 29.82 | 95.45 | +4.0415/15.46/Y | +7.0197/16.20/Y | +2.1914/14.29/Y |
| color_black_white_painted | color | color_black_white | 29.15 | 95.66 | +4.5291/4.78/Y | +7.0197/16.20/Y | +2.7682/2.64/Y |
| color_black_white_property | color | color_black_white | 34.96 | 93.69 | +4.7622/12.67/Y | +7.0197/16.20/Y | +2.8772/7.55/Y |
| object_desc_apple_banana | object | object_apple_banana | 5.24 | 99.86 | +1.5569/3.67/Y | +12.7043/53.41/Y | +0.7777/1.78/n |
| object_desc_car_truck | object | object_car_truck | 5.56 | 99.85 | +1.6980/3.61/Y | +7.7021/23.25/Y | +1.2542/2.64/Y |

## deepseek7b

layer=L18, train_n=8, test_n=6, alphas=[4.0, 8.0, 12.0], attn=sdpa

Cell format: best own Δ / selectivity ratio / pass

| candidate | family | own task | parallel norm % | perp norm % | full | parallel | perp |
|---|---|---|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 0.63 | 100.00 | +0.0451/0.77/n | +0.5660/7.41/Y | +0.0312/0.75/n |
| color_red_blue_painted | color | color_red_blue | 1.97 | 99.98 | +0.1597/1.35/n | -0.1667/4.27/n | +0.1667/2.53/Y |
| color_red_blue_property | color | color_red_blue | 0.46 | 100.00 | -0.0104/1.50/n | -0.1667/4.27/n | -0.0174/0.33/n |
| color_black_white_direct | color | color_black_white | 5.78 | 99.83 | +0.2743/1.36/n | +0.8403/9.78/Y | +0.2396/1.19/n |
| color_black_white_painted | color | color_black_white | 1.16 | 99.99 | +0.0104/0.10/n | -0.2326/7.44/n | +0.0625/1.60/n |
| color_black_white_property | color | color_black_white | 3.04 | 99.95 | +0.0486/1.24/n | +0.8403/9.78/Y | +0.0278/1.14/n |
| object_desc_apple_banana | object | object_apple_banana | 1.53 | 99.99 | +0.0156/0.24/n | -0.3125/10.00/n | +0.0234/0.42/n |
| object_desc_car_truck | object | object_car_truck | 1.35 | 99.99 | +0.0938/4.00/Y | +0.8086/8.62/Y | +0.0703/0.55/n |

## Cross-model Compact

| model | passed components | best BW full | best RB full | best object full | best object perp |
|---|---|---|---|---|---|
| qwen3 | color_black_white_direct:full,color_black_white_direct:perp,object_desc_car_truck:parallel | color_black_white_direct +0.132/4.22/Y | color_red_blue_direct +0.378/1.61/n | object_desc_car_truck +0.328/1.17/n | object_desc_car_truck +0.328/1.39/n |
| glm4 | color_red_blue_direct:full,color_red_blue_direct:parallel,color_red_blue_direct:perp,color_red_blue_painted:full,color_red_blue_painted:parallel,color_red_blue_painted:perp,color_red_blue_property:full,color_red_blue_property:parallel,color_red_blue_property:perp,color_black_white_direct:full,color_black_white_direct:parallel,color_black_white_direct:perp,color_black_white_painted:full,color_black_white_painted:parallel,color_black_white_painted:perp,color_black_white_property:full,color_black_white_property:parallel,color_black_white_property:perp,object_desc_apple_banana:full,object_desc_apple_banana:parallel,object_desc_car_truck:full,object_desc_car_truck:parallel,object_desc_car_truck:perp | color_black_white_property +4.762/12.67/Y | color_red_blue_property +6.651/3.51/Y | object_desc_car_truck +1.698/3.61/Y | object_desc_car_truck +1.254/2.64/Y |
| deepseek7b | color_red_blue_direct:parallel,color_red_blue_painted:perp,color_black_white_direct:parallel,color_black_white_property:parallel,object_desc_car_truck:full,object_desc_car_truck:parallel | color_black_white_direct +0.274/1.36/n | color_red_blue_painted +0.160/1.35/n | object_desc_car_truck +0.094/4.00/Y | object_desc_car_truck +0.070/0.55/n |

