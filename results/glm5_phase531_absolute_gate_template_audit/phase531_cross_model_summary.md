# Phase531 Absolute Gate and Template Audit Summary

## qwen3

layer=L12, train_n=8, test_n=6, alphas=[4.0, 8.0, 12.0], min_abs_delta=0.25, attn=sdpa

Cell format: own Δ / ratio / RatioGate AbsoluteGate StrictGate

| candidate | family | own task | par % | full | parallel | perp | random_perp | random_readout |
|---|---|---|---:|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 1.48 | +0.378/1.61/rAn | +0.128/1.37/ran | +0.378/1.61/rAn | +0.108/5.17/Ran | +0.128/1.37/ran |
| color_red_blue_painted | color | color_red_blue | 0.96 | +0.007/0.06/ran | -0.122/1.30/ran | -0.007/0.06/ran | -0.003/0.04/ran | +0.128/1.37/ran |
| color_red_blue_property | color | color_red_blue | 1.70 | -0.146/1.33/ran | -0.122/1.30/ran | -0.135/1.73/ran | +0.031/0.28/ran | +0.128/1.37/ran |
| color_black_white_direct | color | color_black_white | 0.52 | +0.132/4.22/Ran | +0.045/0.16/ran | +0.125/2.67/Ran | +0.073/0.75/ran | +0.045/0.16/ran |
| color_black_white_painted | color | color_black_white | 1.44 | +0.014/0.14/ran | +0.045/0.16/ran | +0.003/0.05/ran | +0.097/3.11/Ran | +0.045/0.16/ran |
| color_black_white_property | color | color_black_white | 2.57 | -0.035/0.44/ran | +0.045/0.16/ran | -0.035/0.37/ran | +0.257/1.49/rAn | +0.045/0.16/ran |
| object_desc_apple_banana | object | object_apple_banana | 0.29 | -0.156/2.37/ran | +0.031/0.11/ran | -0.109/1.40/ran | -0.109/0.64/ran | +0.031/0.11/ran |
| object_desc_car_truck | object | object_car_truck | 1.40 | +0.328/1.17/rAn | +0.297/2.59/RAY | +0.328/1.39/rAn | +0.016/0.10/ran | +0.297/2.59/RAY |

### Template Direction Cosine

#### red_blue

| dir | norm | color_red_blue_direct | color_red_blue_painted | color_red_blue_property |
|---|---:|---:|---:|---:|
| color_red_blue_direct | 6.2336 | +1.0000 | +0.1431 | -0.0335 |
| color_red_blue_painted | 5.8075 | +0.1431 | +1.0000 | +0.0948 |
| color_red_blue_property | 3.2158 | -0.0335 | +0.0948 | +1.0000 |

#### black_white

| dir | norm | color_black_white_direct | color_black_white_painted | color_black_white_property |
|---|---:|---:|---:|---:|
| color_black_white_direct | 8.3741 | +1.0000 | +0.0750 | -0.0114 |
| color_black_white_painted | 6.1987 | +0.0750 | +1.0000 | +0.1335 |
| color_black_white_property | 3.0993 | -0.0114 | +0.1335 | +1.0000 |

## glm4

layer=L26, train_n=8, test_n=6, alphas=[4.0, 8.0, 12.0], min_abs_delta=0.25, attn=sdpa

Cell format: own Δ / ratio / RatioGate AbsoluteGate StrictGate

| candidate | family | own task | par % | full | parallel | perp | random_perp | random_readout |
|---|---|---|---:|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 31.49 | +6.422/3.71/RAY | +8.340/7.74/RAY | +4.329/2.92/RAY | -0.069/0.27/ran | +8.340/7.74/RAY |
| color_red_blue_painted | color | color_red_blue | 30.73 | +6.370/3.49/RAY | +8.340/7.74/RAY | +4.245/2.56/RAY | +0.368/0.72/rAn | +8.340/7.74/RAY |
| color_red_blue_property | color | color_red_blue | 30.87 | +6.651/3.51/RAY | +8.340/7.74/RAY | +4.558/2.79/RAY | -0.079/0.38/ran | +8.340/7.74/RAY |
| color_black_white_direct | color | color_black_white | 29.82 | +4.041/15.46/RAY | +7.020/16.20/RAY | +2.191/14.29/RAY | +0.151/0.32/ran | +7.020/16.20/RAY |
| color_black_white_painted | color | color_black_white | 29.15 | +4.529/4.78/RAY | +7.020/16.20/RAY | +2.768/2.64/RAY | -0.006/0.03/ran | +7.020/16.20/RAY |
| color_black_white_property | color | color_black_white | 34.96 | +4.762/12.67/RAY | +7.020/16.20/RAY | +2.877/7.55/RAY | +0.213/0.30/ran | +7.020/16.20/RAY |
| object_desc_apple_banana | object | object_apple_banana | 5.24 | +1.557/3.67/RAY | +12.704/53.41/RAY | +0.778/1.78/rAn | +0.703/1.26/rAn | +12.704/53.41/RAY |
| object_desc_car_truck | object | object_car_truck | 5.56 | +1.698/3.61/RAY | +7.702/23.25/RAY | +1.254/2.64/RAY | -0.050/0.15/ran | +7.702/23.25/RAY |

### Template Direction Cosine

#### red_blue

| dir | norm | color_red_blue_direct | color_red_blue_painted | color_red_blue_property |
|---|---:|---:|---:|---:|
| color_red_blue_direct | 10.2330 | +1.0000 | +0.7355 | +0.8109 |
| color_red_blue_painted | 5.6266 | +0.7355 | +1.0000 | +0.7249 |
| color_red_blue_property | 12.2799 | +0.8109 | +0.7249 | +1.0000 |

#### black_white

| dir | norm | color_black_white_direct | color_black_white_painted | color_black_white_property |
|---|---:|---:|---:|---:|
| color_black_white_direct | 9.3708 | +1.0000 | +0.5517 | +0.6704 |
| color_black_white_painted | 6.4934 | +0.5517 | +1.0000 | +0.6523 |
| color_black_white_property | 16.7673 | +0.6704 | +0.6523 | +1.0000 |

## deepseek7b

layer=L18, train_n=8, test_n=6, alphas=[4.0, 8.0, 12.0], min_abs_delta=0.25, attn=sdpa

Cell format: own Δ / ratio / RatioGate AbsoluteGate StrictGate

| candidate | family | own task | par % | full | parallel | perp | random_perp | random_readout |
|---|---|---|---:|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 0.63 | +0.045/0.77/ran | +0.566/7.41/RAY | +0.031/0.75/ran | +0.080/0.68/ran | +0.566/7.41/RAY |
| color_red_blue_painted | color | color_red_blue | 1.97 | +0.160/1.35/ran | -0.167/4.27/ran | +0.167/2.53/Ran | +0.073/1.04/ran | +0.566/7.41/RAY |
| color_red_blue_property | color | color_red_blue | 0.46 | -0.010/1.50/ran | -0.167/4.27/ran | -0.017/0.33/ran | +0.024/0.52/ran | +0.566/7.41/RAY |
| color_black_white_direct | color | color_black_white | 5.78 | +0.274/1.36/rAn | +0.840/9.78/RAY | +0.240/1.19/ran | +0.031/0.90/ran | +0.840/9.78/RAY |
| color_black_white_painted | color | color_black_white | 1.16 | +0.010/0.10/ran | -0.233/7.44/ran | +0.062/1.60/ran | -0.028/1.42/ran | +0.840/9.78/RAY |
| color_black_white_property | color | color_black_white | 3.04 | +0.049/1.24/ran | +0.840/9.78/RAY | +0.028/1.14/ran | +0.069/1.43/ran | +0.840/9.78/RAY |
| object_desc_apple_banana | object | object_apple_banana | 1.53 | +0.016/0.24/ran | -0.312/10.00/ran | +0.023/0.42/ran | +0.078/2.50/Ran | +0.930/9.56/RAY |
| object_desc_car_truck | object | object_car_truck | 1.35 | +0.094/4.00/Ran | +0.809/8.62/RAY | +0.070/0.55/ran | -0.023/0.37/ran | +0.809/8.62/RAY |

### Template Direction Cosine

#### red_blue

| dir | norm | color_red_blue_direct | color_red_blue_painted | color_red_blue_property |
|---|---:|---:|---:|---:|
| color_red_blue_direct | 48.9241 | +1.0000 | +0.0410 | +0.1872 |
| color_red_blue_painted | 52.8198 | +0.0410 | +1.0000 | +0.0867 |
| color_red_blue_property | 32.5628 | +0.1872 | +0.0867 | +1.0000 |

#### black_white

| dir | norm | color_black_white_direct | color_black_white_painted | color_black_white_property |
|---|---:|---:|---:|---:|
| color_black_white_direct | 50.4426 | +1.0000 | +0.1072 | +0.1909 |
| color_black_white_painted | 50.2643 | +0.1072 | +1.0000 | +0.1184 |
| color_black_white_property | 43.9552 | +0.1909 | +0.1184 | +1.0000 |

## Cross-model Compact

| model | strict passed learned components | strict passed random controls |
|---|---|---|
| qwen3 | object_desc_car_truck:parallel | object_desc_car_truck:random_readout |
| glm4 | color_red_blue_direct:full,color_red_blue_direct:parallel,color_red_blue_direct:perp,color_red_blue_painted:full,color_red_blue_painted:parallel,color_red_blue_painted:perp,color_red_blue_property:full,color_red_blue_property:parallel,color_red_blue_property:perp,color_black_white_direct:full,color_black_white_direct:parallel,color_black_white_direct:perp,color_black_white_painted:full,color_black_white_painted:parallel,color_black_white_painted:perp,color_black_white_property:full,color_black_white_property:parallel,color_black_white_property:perp,object_desc_apple_banana:full,object_desc_apple_banana:parallel,object_desc_car_truck:full,object_desc_car_truck:parallel,object_desc_car_truck:perp | color_red_blue_direct:random_readout,color_red_blue_painted:random_readout,color_red_blue_property:random_readout,color_black_white_direct:random_readout,color_black_white_painted:random_readout,color_black_white_property:random_readout,object_desc_apple_banana:random_readout,object_desc_car_truck:random_readout |
| deepseek7b | color_red_blue_direct:parallel,color_black_white_direct:parallel,color_black_white_property:parallel,object_desc_car_truck:parallel | color_red_blue_direct:random_readout,color_red_blue_painted:random_readout,color_red_blue_property:random_readout,color_black_white_direct:random_readout,color_black_white_painted:random_readout,color_black_white_property:random_readout,object_desc_apple_banana:random_readout,object_desc_car_truck:random_readout |

