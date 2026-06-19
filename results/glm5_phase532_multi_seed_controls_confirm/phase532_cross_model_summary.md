# Phase532 Multi-Seed Controls Summary

## qwen3

layer=L12, train_n=10, test_n=8, alphas=[8.0, 12.0], min_abs_delta=0.25, seeds=[11, 23, 37, 41], attn=sdpa

| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| category_fruit | category | category_fruit | 6.07 | +2.520/13.44/Y | +0.457/0.54/n | +2.480/14.43/Y | +0.605 | 0 | +0.457/0.54/n | nonrandom_perp |
| color_red_blue_direct | color | color_red_blue | 2.61 | +0.430/0.85/n | +0.094/0.22/n | +0.430/0.81/n | +0.422 | 0 | +0.094/0.22/n | fail |
| color_black_white_direct | color | color_black_white | 0.55 | +0.453/2.64/Y | -0.258/0.24/n | +0.477/4.07/Y | +0.773 | 0 | -0.258/0.24/n | perp_not_above_random_max |
| object_desc_car_truck | object | object_car_truck | 1.40 | +0.328/0.32/n | +0.297/0.40/n | +0.328/0.33/n | +0.297 | 0 | +0.297/0.40/n | fail |

## glm4

layer=L26, train_n=10, test_n=8, alphas=[8.0, 12.0], min_abs_delta=0.25, seeds=[11, 23, 37, 41], attn=sdpa

| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| category_fruit | category | category_fruit | 4.78 | -0.198/0.31/n | +7.299/9.78/Y | -0.453/0.67/n | +1.101 | 0 | +7.299/9.78/Y | readout_only |
| color_red_blue_direct | color | color_red_blue | 32.06 | +7.369/12.31/Y | +8.407/8.50/Y | +5.327/5.65/Y | +0.598 | 1 | +8.407/8.50/Y | nonrandom_perp |
| color_black_white_direct | color | color_black_white | 30.27 | +5.322/28.84/Y | +8.135/18.89/Y | +3.199/36.00/Y | +1.062 | 1 | +8.135/18.89/Y | nonrandom_perp |
| object_desc_car_truck | object | object_car_truck | 5.56 | +1.698/3.48/Y | +7.702/12.78/Y | +1.254/2.65/Y | +0.129 | 0 | +7.702/12.78/Y | nonrandom_perp |

## Cross-model Compact

| model | nonrandom learned perp components |
|---|---|
| qwen3 | category_fruit:perp |
| glm4 | color_red_blue_direct:perp,color_black_white_direct:perp,object_desc_car_truck:perp |

