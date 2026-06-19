# Phase536 Pair Quality and Selectivity Summary

## qwen3

windows={'early': [8, 10, 12], 'center': [10, 12, 14]}, train_n=12, test_n=8, alphas=[4.0, 8.0], seeds=[11, 23, 37, 41], attn=sdpa

Transfer format: min / mean / specificity.

| pair | base margin | base rank | top1 | template cos avg | best common | best direct | best shuffled | random max | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fruit_tool | +4.307 | 332.0 | 0.25 | +0.343 | early:+0.750/+1.116/0.71 | +0.586/+1.272/0.45 | -0.094/+0.095/0.46 | -0.055 | strong_but_not_specific |
| animal_tool | +8.130 | 19.0 | 0.12 | +0.310 | center:+0.461/+0.616/0.46 | +0.156/+0.598/0.13 | -0.035/+0.018/0.08 | +0.184 | strong_but_not_specific |
| vehicle_furniture | +2.909 | 189.6 | 0.21 | +0.352 | center:+0.875/+1.259/1.29 | +0.609/+1.102/0.72 | +0.285/+0.512/0.40 | +0.156 | candidate_common_pair |
| clothing_tool | +1.836 | 141.1 | 0.17 | +0.304 | center:-0.094/+0.221/0.12 | -0.125/+0.172/0.13 | -0.336/-0.029/0.63 | +0.211 | weak |
| fruit_vegetable | +3.484 | 332.0 | 0.25 | +0.170 | early:-0.188/-0.039/0.43 | -0.043/+0.061/0.09 | -0.336/-0.216/0.63 | -0.016 | weak |

## glm4

windows={'early': [22, 24, 26], 'center': [24, 26, 28]}, train_n=12, test_n=8, alphas=[4.0, 8.0], seeds=[11, 23, 37, 41], attn=sdpa

Transfer format: min / mean / specificity.

| pair | base margin | base rank | top1 | template cos avg | best common | best direct | best shuffled | random max | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fruit_tool | +4.185 | 543.8 | 0.04 | +0.194 | early:+0.380/+1.063/0.14 | -0.365/+0.035/0.26 | +0.797/+1.822/0.24 | +0.031 | strong_but_not_specific |
| animal_tool | +4.896 | 92.2 | 0.00 | +0.261 | early:-0.247/+0.862/0.21 | -0.596/+0.046/0.78 | -0.332/+0.539/0.15 | -0.100 | baseline_not_ideal |
| vehicle_furniture | +1.824 | 996.8 | 0.25 | +0.217 | early:+1.190/+1.822/0.52 | +1.621/+2.163/1.05 | +0.284/+1.307/0.09 | +0.205 | strong_but_not_specific |
| clothing_tool | +1.591 | 2103.2 | 0.00 | +0.159 | early:+0.647/+1.124/0.17 | -0.523/-0.188/0.51 | -0.105/+0.847/0.10 | +0.496 | strong_but_not_specific |
| fruit_vegetable | +3.289 | 543.8 | 0.04 | +0.070 | early:+1.912/+2.813/0.68 | +0.385/+0.651/0.23 | +1.021/+1.404/0.48 | +0.211 | strong_but_not_specific |

## deepseek7b

windows={'early': [14, 16, 18], 'center': [16, 18, 20]}, train_n=12, test_n=8, alphas=[4.0, 8.0], seeds=[11, 23, 37, 41], attn=sdpa

Transfer format: min / mean / specificity.

| pair | base margin | base rank | top1 | template cos avg | best common | best direct | best shuffled | random max | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fruit_tool | +2.513 | 1289.2 | 0.25 | +0.223 | center:+0.221/+0.249/0.53 | +0.130/+0.179/0.38 | +0.000/+0.023/0.00 | +0.047 | baseline_not_ideal |
| animal_tool | +4.035 | 533.8 | 0.29 | +0.265 | center:+0.374/+0.453/0.80 | +0.072/+0.255/0.26 | +0.074/+0.127/0.20 | -0.004 | strong_but_not_specific |
| vehicle_furniture | -2.127 | 36657.2 | 0.04 | +0.263 | center:-0.023/+0.183/0.16 | +0.094/+0.277/0.42 | -0.093/+0.051/0.53 | -0.016 | baseline_not_ideal |
| clothing_tool | -0.980 | 8231.6 | 0.04 | +0.255 | center:-0.018/+0.097/0.07 | +0.016/+0.145/0.04 | -0.059/+0.004/0.40 | -0.041 | baseline_not_ideal |
| fruit_vegetable | +2.199 | 1289.2 | 0.25 | +0.176 | center:+0.040/+0.073/0.22 | +0.084/+0.099/0.24 | -0.050/-0.028/0.25 | +0.015 | baseline_not_ideal |

## Cross-model Compact

| model | candidate common pairs |
|---|---|
| qwen3 | vehicle_furniture |
| glm4 | none |
| deepseek7b | none |

