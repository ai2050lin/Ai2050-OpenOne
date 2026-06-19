# Phase534 Template-Invariant Gate Summary

## qwen3

layers=[10, 12, 14], primary=L12, train_n=12, test_n=8, bridge_n=6, max_new_tokens=4, alphas=[8.0, 12.0], cumulative_alphas=[2.0, 4.0, 6.0], seeds=[11, 23, 37, 41, 53, 67, 79, 83], attn=sdpa

Own format: own delta / selectivity ratio / strict. Transfer format: min / mean / ratio / pass.

| component | own | category transfer |
|---|---:|---:|
| category_common_perp | +1.738/1.38/n | +0.121/+1.039/0.15/n |
| category_direct_perp | +1.789/2.06/Y | +0.242/+0.966/0.94/n |
| category_belongs_perp | +1.016/1.38/n | -0.066/+0.561/0.33/n |
| category_kind_perp | +0.195/0.12/n | +0.195/+0.887/0.17/n |
| category_direct_residual | +1.277/2.27/Y | +0.008/+0.302/0.02/n |
| category_belongs_residual | +0.055/0.06/n | -0.684/-0.280/1.14/n |
| category_kind_residual | +0.168/0.28/n | +0.141/+0.221/0.24/n |
| color_red_blue_perp | +0.602/1.23/n | - |
| object_car_truck_perp | +0.211/0.20/n | - |

### Common Random And Cumulative

random_common max transfer-min=-0.047, strict transfer pass count=0

cumulative common transfer-min=+0.398, mean=+1.352, ratio=2.04, pass=Y, alpha=6.0

### Category Template Cosines At Primary Layer

| dir | category_belongs | category_direct | category_kind | cos_to_common | residual_norm_pct |
|---|---:|---:|---:|---:|---:|
| category_belongs | +1.0000 | +0.2751 | +0.2014 | +0.6769 | 73.61 |
| category_direct | +0.2751 | +1.0000 | +0.4025 | +0.7691 | 63.92 |
| category_kind | +0.2014 | +0.4025 | +1.0000 | +0.7353 | 67.78 |

### Generation Bridge

| condition | trace |
|---|---|
| baseline | hit=0.00, path=0.67, rank=610.0, m1=+0.266 |
| single_common_perp | hit=0.17, path=0.83, rank=398.2, m1=+1.880 |
| single_direct_perp | hit=0.17, path=0.83, rank=490.0, m1=+2.000 |
| cumulative_common_perp | hit=0.17, path=0.83, rank=317.8, m1=+1.906 |

## glm4

layers=[24, 26, 28], primary=L26, train_n=12, test_n=8, bridge_n=6, max_new_tokens=4, alphas=[8.0, 12.0], cumulative_alphas=[2.0, 4.0, 6.0], seeds=[11, 23, 37, 41, 53, 67, 79, 83], attn=sdpa

Own format: own delta / selectivity ratio / strict. Transfer format: min / mean / ratio / pass.

| component | own | category transfer |
|---|---:|---:|
| category_common_perp | +2.023/0.77/n | +0.535/+0.716/0.37/n |
| category_direct_perp | +0.334/0.24/n | -0.318/-0.096/0.23/n |
| category_belongs_perp | +1.342/0.49/n | +1.325/+1.803/0.72/n |
| category_kind_perp | -0.026/0.04/n | -0.026/+0.275/0.04/n |
| category_direct_residual | -1.570/2.13/n | -1.570/-0.915/15.62/n |
| category_belongs_residual | +0.001/0.00/n | +0.001/+0.533/0.00/n |
| category_kind_residual | -0.256/0.42/n | -0.256/-0.024/0.42/n |
| color_red_blue_perp | +5.394/4.23/Y | - |
| object_car_truck_perp | +1.346/1.75/n | - |

### Common Random And Cumulative

random_common max transfer-min=-0.049, strict transfer pass count=0

cumulative common transfer-min=+0.646, mean=+1.216, ratio=0.24, pass=n, alpha=4.0

### Category Template Cosines At Primary Layer

| dir | category_belongs | category_direct | category_kind | cos_to_common | residual_norm_pct |
|---|---:|---:|---:|---:|---:|
| category_belongs | +1.0000 | +0.0878 | +0.2499 | +0.6512 | 75.89 |
| category_direct | +0.0878 | +1.0000 | +0.2721 | +0.6620 | 74.95 |
| category_kind | +0.2499 | +0.2721 | +1.0000 | +0.7409 | 67.16 |

### Generation Bridge

| condition | trace |
|---|---|
| baseline | hit=0.17, path=0.67, rank=105.3, m1=+1.126 |
| single_common_perp | hit=0.17, path=0.33, rank=38.7, m1=+2.608 |
| single_direct_perp | hit=0.00, path=0.33, rank=57.8, m1=+1.092 |
| cumulative_common_perp | hit=0.17, path=0.33, rank=40.8, m1=+2.709 |

## deepseek7b

layers=[16, 18, 20], primary=L18, train_n=12, test_n=8, bridge_n=6, max_new_tokens=4, alphas=[8.0, 12.0], cumulative_alphas=[2.0, 4.0, 6.0], seeds=[11, 23, 37, 41, 53, 67, 79, 83], attn=sdpa

Own format: own delta / selectivity ratio / strict. Transfer format: min / mean / ratio / pass.

| component | own | category transfer |
|---|---:|---:|
| category_common_perp | +0.121/0.83/n | +0.062/+0.091/1.14/n |
| category_direct_perp | +0.121/1.19/n | +0.000/+0.045/0.00/n |
| category_belongs_perp | -0.031/0.44/n | -0.031/+0.001/0.44/n |
| category_kind_perp | +0.141/0.73/n | +0.070/+0.134/0.78/n |
| category_direct_residual | +0.020/0.59/n | +0.016/+0.023/0.57/n |
| category_belongs_residual | -0.166/1.63/n | -0.166/-0.048/1.63/n |
| category_kind_residual | +0.133/2.12/n | -0.047/+0.048/1.20/n |
| color_red_blue_perp | +0.102/0.93/n | - |
| object_car_truck_perp | +0.113/0.81/n | - |

### Common Random And Cumulative

random_common max transfer-min=+0.055, strict transfer pass count=0

cumulative common transfer-min=+0.086, mean=+0.172, ratio=1.83, pass=n, alpha=6.0

### Category Template Cosines At Primary Layer

| dir | category_belongs | category_direct | category_kind | cos_to_common | residual_norm_pct |
|---|---:|---:|---:|---:|---:|
| category_belongs | +1.0000 | +0.1116 | +0.1193 | +0.6308 | 77.60 |
| category_direct | +0.1116 | +1.0000 | +0.1729 | +0.6583 | 75.28 |
| category_kind | +0.1193 | +0.1729 | +1.0000 | +0.6622 | 74.93 |

### Generation Bridge

| condition | trace |
|---|---|
| baseline | hit=0.00, path=0.00, rank=689.3, m1=-0.906 |
| single_common_perp | hit=0.00, path=0.00, rank=747.8, m1=-0.729 |
| single_direct_perp | hit=0.00, path=0.00, rank=785.7, m1=-0.779 |
| cumulative_common_perp | hit=0.00, path=0.00, rank=781.3, m1=-0.672 |

## Cross-model Compact

| model | verdict |
|---|---|
| qwen3 | multi_layer_common_only |
| glm4 | no_category_common |
| deepseek7b | no_category_common |

