# Phase531 Absolute Gate and Template Audit Summary

## qwen3

layer=L12, train_n=10, test_n=8, alphas=[4.0, 8.0, 12.0], min_abs_delta=0.25, attn=sdpa

Cell format: own Δ / ratio / RatioGate AbsoluteGate StrictGate

| candidate | family | own task | par % | full | parallel | perp | random_perp | random_readout |
|---|---|---|---:|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 2.61 | +0.315/1.34/rAn | +0.164/1.66/ran | +0.315/1.26/rAn | +0.076/0.59/ran | +0.164/1.66/ran |
| color_red_blue_painted | color | color_red_blue | 1.52 | -0.062/0.67/ran | -0.172/1.83/ran | -0.070/0.64/ran | -0.068/0.72/ran | +0.164/1.66/ran |
| color_red_blue_property | color | color_red_blue | 1.60 | -0.206/2.19/ran | -0.172/1.83/ran | -0.206/2.63/ran | +0.005/0.07/ran | +0.164/1.66/ran |
| color_black_white_direct | color | color_black_white | 0.55 | +0.180/1.25/ran | +0.112/0.36/ran | +0.180/2.76/Ran | -0.013/0.21/ran | +0.112/0.36/ran |
| color_black_white_painted | color | color_black_white | 1.60 | -0.076/0.64/ran | +0.112/0.36/ran | -0.034/0.32/ran | +0.021/0.38/ran | +0.112/0.36/ran |
| color_black_white_property | color | color_black_white | 2.69 | -0.083/1.07/ran | +0.112/0.36/ran | -0.104/1.11/ran | +0.214/1.24/ran | +0.112/0.36/ran |
| object_desc_apple_banana | object | object_apple_banana | 0.29 | -0.156/1.82/ran | +0.031/0.09/ran | -0.109/1.11/ran | -0.109/0.82/ran | +0.031/0.09/ran |
| object_desc_car_truck | object | object_car_truck | 1.40 | +0.328/1.02/rAn | +0.297/2.53/RAY | +0.328/1.05/rAn | +0.016/0.08/ran | +0.297/2.53/RAY |

### Template Direction Cosine

#### red_blue

| dir | norm | color_red_blue_direct | color_red_blue_painted | color_red_blue_property |
|---|---:|---:|---:|---:|
| color_red_blue_direct | 6.3515 | +1.0000 | +0.1615 | -0.0686 |
| color_red_blue_painted | 5.6459 | +0.1615 | +1.0000 | +0.0885 |
| color_red_blue_property | 3.4169 | -0.0686 | +0.0885 | +1.0000 |

#### black_white

| dir | norm | color_black_white_direct | color_black_white_painted | color_black_white_property |
|---|---:|---:|---:|---:|
| color_black_white_direct | 7.6872 | +1.0000 | +0.1037 | -0.0286 |
| color_black_white_painted | 6.2121 | +0.1037 | +1.0000 | +0.1581 |
| color_black_white_property | 3.0300 | -0.0286 | +0.1581 | +1.0000 |

## glm4

layer=L26, train_n=10, test_n=8, alphas=[4.0, 8.0, 12.0], min_abs_delta=0.25, attn=sdpa

Cell format: own Δ / ratio / RatioGate AbsoluteGate StrictGate

| candidate | family | own task | par % | full | parallel | perp | random_perp | random_readout |
|---|---|---|---:|---:|---:|---:|---:|---:|
| color_red_blue_direct | color | color_red_blue | 32.06 | +6.759/3.95/RAY | +8.466/7.86/RAY | +4.493/3.05/RAY | -0.077/0.28/ran | +8.466/7.86/RAY |
| color_red_blue_painted | color | color_red_blue | 30.93 | +6.733/3.66/RAY | +8.466/7.86/RAY | +4.534/2.78/RAY | +0.466/0.92/rAn | +8.466/7.86/RAY |
| color_red_blue_property | color | color_red_blue | 30.94 | +7.035/3.80/RAY | +8.466/7.86/RAY | +4.838/2.99/RAY | -0.047/0.23/ran | +8.466/7.86/RAY |
| color_black_white_direct | color | color_black_white | 30.27 | +4.414/12.64/RAY | +7.396/17.07/RAY | +2.419/23.87/RAY | +0.115/0.25/ran | +7.396/17.07/RAY |
| color_black_white_painted | color | color_black_white | 28.42 | +4.732/4.61/RAY | +7.396/17.07/RAY | +2.821/2.54/RAY | -0.001/0.00/ran | +7.396/17.07/RAY |
| color_black_white_property | color | color_black_white | 35.01 | +5.049/12.65/RAY | +7.396/17.07/RAY | +3.023/8.44/RAY | +0.090/0.12/ran | +7.396/17.07/RAY |
| object_desc_apple_banana | object | object_apple_banana | 5.24 | +1.557/3.89/RAY | +12.704/85.31/RAY | +0.778/1.98/rAn | +0.703/1.14/rAn | +12.704/85.31/RAY |
| object_desc_car_truck | object | object_car_truck | 5.56 | +1.698/3.44/RAY | +7.702/23.25/RAY | +1.254/2.50/RAY | -0.050/0.14/ran | +7.702/23.25/RAY |

### Template Direction Cosine

#### red_blue

| dir | norm | color_red_blue_direct | color_red_blue_painted | color_red_blue_property |
|---|---:|---:|---:|---:|
| color_red_blue_direct | 9.6532 | +1.0000 | +0.7440 | +0.8078 |
| color_red_blue_painted | 5.5315 | +0.7440 | +1.0000 | +0.7326 |
| color_red_blue_property | 12.3535 | +0.8078 | +0.7326 | +1.0000 |

#### black_white

| dir | norm | color_black_white_direct | color_black_white_painted | color_black_white_property |
|---|---:|---:|---:|---:|
| color_black_white_direct | 8.8444 | +1.0000 | +0.5442 | +0.6679 |
| color_black_white_painted | 6.1993 | +0.5442 | +1.0000 | +0.6485 |
| color_black_white_property | 16.3814 | +0.6679 | +0.6485 | +1.0000 |

## Cross-model Compact

| model | strict passed learned components | strict passed random controls |
|---|---|---|
| qwen3 | object_desc_car_truck:parallel | object_desc_car_truck:random_readout |
| glm4 | color_red_blue_direct:full,color_red_blue_direct:parallel,color_red_blue_direct:perp,color_red_blue_painted:full,color_red_blue_painted:parallel,color_red_blue_painted:perp,color_red_blue_property:full,color_red_blue_property:parallel,color_red_blue_property:perp,color_black_white_direct:full,color_black_white_direct:parallel,color_black_white_direct:perp,color_black_white_painted:full,color_black_white_painted:parallel,color_black_white_painted:perp,color_black_white_property:full,color_black_white_property:parallel,color_black_white_property:perp,object_desc_apple_banana:full,object_desc_apple_banana:parallel,object_desc_car_truck:full,object_desc_car_truck:parallel,object_desc_car_truck:perp | color_red_blue_direct:random_readout,color_red_blue_painted:random_readout,color_red_blue_property:random_readout,color_black_white_direct:random_readout,color_black_white_painted:random_readout,color_black_white_property:random_readout,object_desc_apple_banana:random_readout,object_desc_car_truck:random_readout |

