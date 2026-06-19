# Phase532 Multi-Seed Controls Summary

## qwen3

layer=L12, train_n=8, test_n=6, alphas=[8.0, 12.0], min_abs_delta=0.25, seeds=[11, 23, 37, 41], attn=sdpa

| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| category_fruit | category | category_fruit | 5.60 | +2.693/5.50/Y | +0.385/0.46/n | +2.651/5.66/Y | +0.589 | 0 | +0.385/0.46/n | nonrandom_perp |
| color_red_blue_direct | color | color_red_blue | 1.48 | +0.385/0.75/n | -0.021/0.09/n | +0.406/0.85/n | +0.354 | 0 | -0.021/0.09/n | fail |
| color_black_white_direct | color | color_black_white | 0.52 | +0.354/5.23/Y | -0.635/0.66/n | +0.344/3.88/Y | +1.115 | 0 | -0.635/0.66/n | perp_not_above_random_max |
| object_desc_car_truck | object | object_car_truck | 1.40 | +0.328/0.35/n | +0.297/0.46/n | +0.328/0.35/n | +0.297 | 0 | +0.297/0.46/n | fail |

## glm4

layer=L26, train_n=8, test_n=6, alphas=[8.0, 12.0], min_abs_delta=0.25, seeds=[11, 23, 37, 41], attn=sdpa

| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| category_fruit | category | category_fruit | 4.96 | +0.400/0.62/n | +7.790/10.24/Y | +0.098/0.15/n | +1.151 | 0 | +7.790/10.24/Y | readout_only |
| color_red_blue_direct | color | color_red_blue | 31.49 | +7.511/12.43/Y | +8.444/10.88/Y | +5.513/6.71/Y | +0.604 | 1 | +8.444/10.88/Y | nonrandom_perp |
| color_black_white_direct | color | color_black_white | 29.82 | +5.359/47.86/Y | +8.230/14.43/Y | +3.250/18.35/Y | +1.115 | 1 | +8.230/14.43/Y | nonrandom_perp |
| object_desc_car_truck | object | object_car_truck | 5.56 | +1.698/3.51/Y | +7.702/12.12/Y | +1.254/2.77/Y | +0.129 | 0 | +7.702/12.12/Y | nonrandom_perp |

## deepseek7b

layer=L18, train_n=8, test_n=6, alphas=[8.0, 12.0], min_abs_delta=0.25, seeds=[11, 23, 37, 41], attn=sdpa

| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| category_fruit | category | category_fruit | 1.06 | +0.031/0.31/n | -0.453/4.83/n | +0.036/0.30/n | +0.104 | 0 | +0.667/32.00/Y | fail |
| color_red_blue_direct | color | color_red_blue | 0.63 | +0.073/2.07/n | +0.656/14.00/Y | +0.031/0.57/n | +0.115 | 0 | +0.656/14.00/Y | readout_only |
| color_black_white_direct | color | color_black_white | 5.78 | +0.250/0.92/n | +0.854/6.31/Y | +0.229/0.81/n | +0.167 | 0 | +0.854/6.31/Y | readout_only |
| object_desc_car_truck | object | object_car_truck | 1.35 | +0.094/0.90/n | +0.809/8.62/Y | +0.070/0.79/n | +0.105 | 0 | +0.809/8.62/Y | readout_only |

## Cross-model Compact

| model | nonrandom learned perp components |
|---|---|
| qwen3 | category_fruit:perp |
| glm4 | color_red_blue_direct:perp,color_black_white_direct:perp,object_desc_car_truck:perp |
| deepseek7b | none |

