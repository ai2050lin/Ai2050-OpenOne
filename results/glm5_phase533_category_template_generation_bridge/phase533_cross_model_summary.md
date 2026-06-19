# Phase533 Category Template and Generation Bridge Summary

## qwen3

layer=L12, train_n=12, test_n=8, bridge_n=6, max_new_tokens=3, alphas=[8.0, 12.0], seeds=[11, 23, 37, 41, 53, 67, 79, 83], min_abs_delta=0.25, attn=sdpa

Cell format: best own delta / selectivity ratio / strict gate.

| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| category_direct | category | category_direct | 6.10 | +1.832/1.97/n | +0.406/0.57/n | +1.789/2.06/Y | +1.004 | 2 | +0.406/0.57/n | perp_above_random_but_random_passes |
| category_belongs | category | category_belongs | 0.78 | +0.977/1.25/n | -0.430/0.56/n | +1.016/1.38/n | +0.414 | 0 | +0.719/1.70/n | fail |
| category_kind | category | category_kind | 5.34 | +0.242/0.15/n | +0.348/0.48/n | +0.195/0.12/n | +0.023 | 0 | +0.348/0.48/n | fail |
| color_red_blue_direct | color | color_red_blue | 2.11 | +0.602/1.27/n | +0.445/0.90/n | +0.602/1.23/n | +1.086 | 1 | +0.445/0.90/n | fail |
| object_desc_car_truck | object | object_car_truck | 1.94 | +0.242/0.24/n | +0.227/0.32/n | +0.211/0.20/n | +0.234 | 0 | +0.227/0.32/n | fail |

### Category Template Cosines

| dir | category_belongs | category_direct | category_kind |
|---|---:|---:|---:|
| category_belongs | +1.0000 | +0.2751 | +0.2014 |
| category_direct | +0.2751 | +1.0000 | +0.4025 |
| category_kind | +0.2014 | +0.4025 | +1.0000 |

### Category Greedy Generation Bridge

Cell format: any target-token hit rate [per-step hit rates].

| candidate | baseline | perp | random_readout | perp-baseline | random_readout-baseline |
|---|---:|---:|---:|---:|---:|
| category_direct | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | +0.00 | +0.00 |
| category_belongs | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | +0.00 | +0.00 |
| category_kind | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | +0.00 | +0.00 |

## glm4

layer=L26, train_n=12, test_n=8, bridge_n=6, max_new_tokens=3, alphas=[8.0, 12.0], seeds=[11, 23, 37, 41, 53, 67, 79, 83], min_abs_delta=0.25, attn=sdpa

Cell format: best own delta / selectivity ratio / strict gate.

| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| category_direct | category | category_direct | 6.63 | +1.156/0.84/n | +9.520/1.35/n | +0.334/0.24/n | +0.395 | 0 | +9.520/1.35/n | fail |
| category_belongs | category | category_belongs | 4.12 | +1.685/0.53/n | +7.053/0.74/n | +1.342/0.49/n | +0.174 | 0 | +7.053/0.74/n | fail |
| category_kind | category | category_kind | 11.85 | +0.677/0.31/n | +6.444/0.68/n | -0.026/0.04/n | +1.042 | 1 | +6.444/0.68/n | fail |
| color_red_blue_direct | color | color_red_blue | 32.29 | +7.541/6.94/Y | +8.469/16.77/Y | +5.394/4.23/Y | +0.641 | 0 | +8.469/16.77/Y | clean_nonrandom_perp |
| object_desc_car_truck | object | object_car_truck | 5.09 | +1.825/2.38/Y | +8.419/37.81/Y | +1.346/1.75/n | +0.483 | 0 | +8.419/37.81/Y | readout_interface |

### Category Template Cosines

| dir | category_belongs | category_direct | category_kind |
|---|---:|---:|---:|
| category_belongs | +1.0000 | +0.0878 | +0.2499 |
| category_direct | +0.0878 | +1.0000 | +0.2721 |
| category_kind | +0.2499 | +0.2721 | +1.0000 |

### Category Greedy Generation Bridge

Cell format: any target-token hit rate [per-step hit rates].

| candidate | baseline | perp | random_readout | perp-baseline | random_readout-baseline |
|---|---:|---:|---:|---:|---:|
| category_direct | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | +0.00 | +0.00 |
| category_belongs | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | +0.00 | +0.00 |
| category_kind | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | +0.00 | +0.00 |

## deepseek7b

layer=L18, train_n=12, test_n=8, bridge_n=6, max_new_tokens=3, alphas=[8.0, 12.0], seeds=[11, 23, 37, 41, 53, 67, 79, 83], min_abs_delta=0.25, attn=sdpa

Cell format: best own delta / selectivity ratio / strict gate.

| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| category_direct | category | category_direct | 0.06 | +0.109/0.90/n | -0.449/0.89/n | +0.121/1.19/n | +0.043 | 0 | +0.684/0.95/n | fail |
| category_belongs | category | category_belongs | 2.65 | -0.049/0.45/n | -0.504/1.12/n | -0.031/0.44/n | +0.037 | 0 | +0.717/1.05/n | fail |
| category_kind | category | category_kind | 1.44 | +0.133/0.73/n | -0.320/0.64/n | +0.141/0.73/n | +0.133 | 0 | +0.508/0.71/n | fail |
| color_red_blue_direct | color | color_red_blue | 1.17 | +0.102/0.87/n | +0.703/20.00/Y | +0.102/0.93/n | +0.180 | 0 | +0.703/20.00/Y | readout_interface |
| object_desc_car_truck | object | object_car_truck | 0.06 | +0.125/0.73/n | -0.516/5.08/n | +0.113/0.81/n | +0.055 | 0 | +0.750/6.00/Y | fail |

### Category Template Cosines

| dir | category_belongs | category_direct | category_kind |
|---|---:|---:|---:|
| category_belongs | +1.0000 | +0.1116 | +0.1193 |
| category_direct | +0.1116 | +1.0000 | +0.1729 |
| category_kind | +0.1193 | +0.1729 | +1.0000 |

### Category Greedy Generation Bridge

Cell format: any target-token hit rate [per-step hit rates].

| candidate | baseline | perp | random_readout | perp-baseline | random_readout-baseline |
|---|---:|---:|---:|---:|---:|
| category_direct | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | +0.00 | +0.00 |
| category_belongs | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | +0.00 | +0.00 |
| category_kind | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | 0.00 [0.00,0.00,0.00] | +0.00 | +0.00 |

## Cross-model Compact

| model | clean nonrandom learned perp | learned perp above random max but random also passes |
|---|---|---|
| qwen3 | none | category_direct |
| glm4 | color_red_blue_direct | none |
| deepseek7b | none | none |

