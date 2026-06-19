# Phase538 Interface Response Matrix Summary

## qwen3

pairs=['vehicle_furniture', 'clothing_tool', 'furniture_clothing', 'vehicle_tool', 'vehicle_clothing', 'furniture_tool'], windows={'center': [10, 12, 14], 'late': [12, 14, 16], 'extended': [10, 12, 14, 16]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0, 8.0], seeds=4, attn=sdpa

Cell format: self min / self mean / off max abs / specificity / top off-pair.

| source pair | best win | common | direct | shuffled | random self max | random pass-like | strict pass |
|---|---|---:|---:|---:|---:|---:|---|
| vehicle_furniture | extended | +1.195/+1.533/1.887/0.63/vehicle_clothing | +0.820/+1.371/1.703/0.48/vehicle_clothing | +0.250/+0.461/0.570/0.44/vehicle_clothing | +0.039 | 0 | False |
| clothing_tool | late | +0.172/+0.305/0.750/0.23/vehicle_furniture | +0.023/+0.122/0.500/0.05/vehicle_furniture | -0.055/+0.018/0.227/0.24/furniture_tool | +0.164 | 0 | False |
| furniture_clothing | center | +0.414/+0.475/0.926/0.45/clothing_tool | +0.148/+0.216/0.453/0.33/clothing_tool | -0.102/+0.005/0.148/0.68/vehicle_tool | +0.008 | 0 | False |
| vehicle_tool | extended | +1.547/+1.612/1.734/0.89/vehicle_furniture | +0.781/+0.974/1.242/0.63/vehicle_furniture | +0.633/+0.922/1.090/0.58/vehicle_furniture | -0.016 | 0 | False |
| vehicle_clothing | extended | +1.148/+1.389/2.055/0.56/vehicle_furniture | +0.914/+1.461/2.062/0.44/vehicle_furniture | +0.234/+0.398/0.711/0.33/vehicle_furniture | -0.016 | 0 | False |
| furniture_tool | extended | +0.727/+0.917/1.184/0.61/vehicle_furniture | +0.023/+0.267/0.449/0.05/furniture_clothing | -0.207/+0.061/0.320/0.65/furniture_clothing | +0.051 | 0 | False |

### Common Mean Response Matrices

Each matrix uses the best common alpha/window for that source. Values are mean delta over templates.

| source \ target | vehicle_furniture | clothing_tool | furniture_clothing | vehicle_tool | vehicle_clothing | furniture_tool |
|---|---:|---:|---:|---:|---:|---:|
| vehicle_furniture | +1.533 | -0.255 | -0.428 | +1.340 | +1.406 | -0.421 |
| clothing_tool | -0.406 | +0.305 | -0.030 | -0.172 | -0.419 | +0.163 |
| furniture_clothing | -0.284 | -0.754 | +0.475 | -0.328 | +0.016 | +0.298 |
| vehicle_tool | +1.513 | -0.064 | +0.112 | +1.612 | +1.456 | +0.203 |
| vehicle_clothing | +1.469 | -0.701 | -0.224 | +0.995 | +1.389 | -0.477 |
| furniture_tool | -0.889 | +0.190 | +0.805 | -0.332 | -0.616 | +0.917 |

### Top Leakage Edges

| source | best win | top1 target/max/mean | top2 target/max/mean | top3 target/max/mean |
|---|---|---:|---:|---:|
| vehicle_furniture | extended | vehicle_clothing/1.887/+1.406 | vehicle_tool/1.602/+1.340 | clothing_tool/1.281/-0.255 |
| clothing_tool | late | vehicle_furniture/0.750/-0.406 | vehicle_tool/0.750/-0.172 | vehicle_clothing/0.750/-0.419 |
| furniture_clothing | center | clothing_tool/0.926/-0.754 | vehicle_tool/0.758/-0.328 | vehicle_furniture/0.523/-0.284 |
| vehicle_tool | extended | vehicle_furniture/1.734/+1.513 | vehicle_clothing/1.625/+1.456 | clothing_tool/1.066/-0.064 |
| vehicle_clothing | extended | vehicle_furniture/2.055/+1.469 | clothing_tool/1.805/-0.701 | vehicle_tool/1.250/+0.995 |
| furniture_tool | extended | vehicle_furniture/1.184/-0.889 | vehicle_tool/1.184/-0.332 | vehicle_clothing/1.184/-0.616 |

### Vehicle/Furniture -> Clothing/Tool

- center: mean -0.094, max_abs 0.680, source_min +0.875, spec 0.57
- late: mean -0.215, max_abs 1.238, source_min +1.152, spec 0.69
- extended: mean -0.255, max_abs 1.281, source_min +1.195, spec 0.63

## glm4

pairs=['vehicle_furniture', 'clothing_tool', 'furniture_clothing', 'vehicle_tool', 'vehicle_clothing', 'furniture_tool'], windows={'center': [24, 26, 28], 'late': [26, 28, 30], 'extended': [24, 26, 28, 30]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0, 8.0], seeds=4, attn=sdpa

Cell format: self min / self mean / off max abs / specificity / top off-pair.

| source pair | best win | common | direct | shuffled | random self max | random pass-like | strict pass |
|---|---|---:|---:|---:|---:|---:|---|
| vehicle_furniture | extended | +1.428/+2.028/3.916/0.36/vehicle_tool | +1.695/+2.283/3.551/0.48/vehicle_tool | -0.181/+0.655/1.207/0.15/vehicle_tool | +0.291 | 0 | False |
| clothing_tool | center | +0.086/+0.530/2.457/0.03/vehicle_clothing | -0.259/-0.076/0.756/0.34/furniture_clothing | +0.098/+0.549/2.027/0.05/vehicle_clothing | +0.496 | 0 | False |
| furniture_clothing | late | +0.000/+0.668/0.884/0.00/clothing_tool | -0.018/+0.418/0.533/0.03/clothing_tool | +0.031/+0.453/0.686/0.05/clothing_tool | -0.145 | 0 | False |
| vehicle_tool | center | +2.059/+2.187/2.367/0.87/vehicle_furniture | +0.925/+1.021/1.664/0.56/furniture_tool | +1.711/+1.940/2.672/0.64/vehicle_furniture | -0.080 | 0 | False |
| vehicle_clothing | late | +1.168/+2.268/4.808/0.24/clothing_tool | +1.107/+2.268/3.129/0.35/vehicle_tool | +1.356/+1.682/4.346/0.31/clothing_tool | -0.125 | 0 | False |
| furniture_tool | center | +0.104/+0.234/1.059/0.10/vehicle_furniture | -0.212/+0.009/0.645/0.33/vehicle_furniture | -0.041/+0.020/0.719/0.06/vehicle_tool | -0.165 | 0 | False |

### Common Mean Response Matrices

Each matrix uses the best common alpha/window for that source. Values are mean delta over templates.

| source \ target | vehicle_furniture | clothing_tool | furniture_clothing | vehicle_tool | vehicle_clothing | furniture_tool |
|---|---:|---:|---:|---:|---:|---:|
| vehicle_furniture | +2.028 | -1.392 | -1.031 | +3.350 | +2.691 | -1.328 |
| clothing_tool | -0.367 | +0.530 | -1.136 | -0.222 | -1.401 | -0.426 |
| furniture_clothing | -0.407 | -0.856 | +0.668 | +0.061 | +0.032 | +0.367 |
| vehicle_tool | +1.526 | -0.187 | -0.496 | +2.187 | +1.515 | -0.700 |
| vehicle_clothing | +2.092 | -4.070 | -0.251 | +2.658 | +2.268 | -0.652 |
| furniture_tool | -0.652 | -0.097 | +0.240 | +0.014 | -0.393 | +0.234 |

### Top Leakage Edges

| source | best win | top1 target/max/mean | top2 target/max/mean | top3 target/max/mean |
|---|---|---:|---:|---:|
| vehicle_furniture | extended | vehicle_tool/3.916/+3.350 | vehicle_clothing/3.084/+2.691 | clothing_tool/2.165/-1.392 |
| clothing_tool | center | vehicle_clothing/2.457/-1.401 | furniture_clothing/1.946/-1.136 | vehicle_tool/0.829/-0.222 |
| furniture_clothing | late | clothing_tool/0.884/-0.856 | furniture_tool/0.698/+0.367 | vehicle_furniture/0.633/-0.407 |
| vehicle_tool | center | vehicle_furniture/2.367/+1.526 | vehicle_clothing/2.344/+1.515 | furniture_tool/1.284/-0.700 |
| vehicle_clothing | late | clothing_tool/4.808/-4.070 | vehicle_furniture/3.062/+2.092 | vehicle_tool/3.025/+2.658 |
| furniture_tool | center | vehicle_furniture/1.059/-0.652 | vehicle_tool/0.617/+0.014 | vehicle_clothing/0.602/-0.393 |

### Vehicle/Furniture -> Clothing/Tool

- center: mean -0.907, max_abs 1.579, source_min +1.028, spec 0.27
- late: mean -0.332, max_abs 0.729, source_min +1.061, spec 0.34
- extended: mean -1.392, max_abs 2.165, source_min +1.428, spec 0.36

## deepseek7b

pairs=['vehicle_furniture', 'clothing_tool', 'furniture_clothing', 'vehicle_tool', 'vehicle_clothing', 'furniture_tool'], windows={'center': [16, 18, 20], 'late': [18, 20, 22], 'extended': [16, 18, 20, 22]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0, 8.0], seeds=4, attn=sdpa

Cell format: self min / self mean / off max abs / specificity / top off-pair.

| source pair | best win | common | direct | shuffled | random self max | random pass-like | strict pass |
|---|---|---:|---:|---:|---:|---:|---|
| vehicle_furniture | late | +0.034/+0.264/0.680/0.05/vehicle_clothing | +0.126/+0.272/0.471/0.27/vehicle_clothing | -0.044/-0.002/0.070/0.62/vehicle_clothing | -0.004 | 0 | False |
| clothing_tool | center | -0.006/+0.042/0.070/0.08/furniture_tool | +0.006/+0.141/0.176/0.03/vehicle_tool | -0.025/+0.002/0.115/0.22/vehicle_tool | -0.023 | 0 | False |
| furniture_clothing | late | +0.008/+0.020/0.072/0.11/vehicle_furniture | +0.037/+0.068/0.123/0.30/clothing_tool | -0.008/-0.003/0.207/0.04/vehicle_furniture | +0.006 | 0 | False |
| vehicle_tool | extended | +0.430/+0.590/0.867/0.50/vehicle_clothing | +0.356/+0.512/0.655/0.54/vehicle_clothing | +0.008/+0.024/0.125/0.06/furniture_tool | +0.004 | 0 | False |
| vehicle_clothing | extended | +0.434/+0.587/0.734/0.59/vehicle_furniture | +0.290/+0.440/0.557/0.52/vehicle_tool | +0.246/+0.341/0.496/0.50/vehicle_furniture | -0.030 | 0 | False |
| furniture_tool | late | -0.004/+0.012/0.080/0.05/vehicle_clothing | +0.082/+0.100/0.297/0.28/clothing_tool | -0.062/-0.029/0.070/0.89/furniture_clothing | +0.027 | 0 | False |

### Common Mean Response Matrices

Each matrix uses the best common alpha/window for that source. Values are mean delta over templates.

| source \ target | vehicle_furniture | clothing_tool | furniture_clothing | vehicle_tool | vehicle_clothing | furniture_tool |
|---|---:|---:|---:|---:|---:|---:|
| vehicle_furniture | +0.264 | +0.106 | +0.219 | +0.425 | +0.429 | +0.156 |
| clothing_tool | -0.004 | +0.042 | -0.016 | +0.016 | -0.012 | -0.007 |
| furniture_clothing | +0.033 | -0.048 | +0.020 | -0.011 | +0.000 | +0.008 |
| vehicle_tool | +0.373 | +0.222 | +0.164 | +0.590 | +0.504 | +0.207 |
| vehicle_clothing | +0.402 | +0.076 | +0.293 | +0.537 | +0.587 | +0.198 |
| furniture_tool | +0.020 | +0.005 | -0.011 | -0.006 | -0.027 | +0.012 |

### Top Leakage Edges

| source | best win | top1 target/max/mean | top2 target/max/mean | top3 target/max/mean |
|---|---|---:|---:|---:|
| vehicle_furniture | late | vehicle_clothing/0.680/+0.429 | vehicle_tool/0.664/+0.425 | furniture_clothing/0.262/+0.219 |
| clothing_tool | center | furniture_tool/0.070/-0.007 | vehicle_clothing/0.061/-0.012 | vehicle_furniture/0.057/-0.004 |
| furniture_clothing | late | vehicle_furniture/0.072/+0.033 | clothing_tool/0.065/-0.048 | vehicle_clothing/0.033/+0.000 |
| vehicle_tool | extended | vehicle_clothing/0.867/+0.504 | vehicle_furniture/0.836/+0.373 | clothing_tool/0.363/+0.222 |
| vehicle_clothing | extended | vehicle_furniture/0.734/+0.402 | vehicle_tool/0.703/+0.537 | furniture_clothing/0.306/+0.293 |
| furniture_tool | late | vehicle_clothing/0.080/-0.027 | vehicle_tool/0.076/-0.006 | clothing_tool/0.047/+0.005 |

### Vehicle/Furniture -> Clothing/Tool

- center: mean +0.074, max_abs 0.125, source_min -0.018, spec 0.12
- late: mean +0.106, max_abs 0.162, source_min +0.034, spec 0.05
- extended: mean +0.162, max_abs 0.205, source_min +0.024, spec 0.03

## Cross-model Compact

| model | source | best win | self min | specificity | top off-pair | top off abs | strict pass |
|---|---|---|---:|---:|---|---:|---|
| qwen3 | vehicle_furniture | extended | +1.195 | 0.63 | vehicle_clothing | 1.887 | False |
| qwen3 | clothing_tool | late | +0.172 | 0.23 | vehicle_furniture | 0.750 | False |
| qwen3 | furniture_clothing | center | +0.414 | 0.45 | clothing_tool | 0.926 | False |
| qwen3 | vehicle_tool | extended | +1.547 | 0.89 | vehicle_furniture | 1.734 | False |
| qwen3 | vehicle_clothing | extended | +1.148 | 0.56 | vehicle_furniture | 2.055 | False |
| qwen3 | furniture_tool | extended | +0.727 | 0.61 | vehicle_furniture | 1.184 | False |
| glm4 | vehicle_furniture | extended | +1.428 | 0.36 | vehicle_tool | 3.916 | False |
| glm4 | clothing_tool | center | +0.086 | 0.03 | vehicle_clothing | 2.457 | False |
| glm4 | furniture_clothing | late | +0.000 | 0.00 | clothing_tool | 0.884 | False |
| glm4 | vehicle_tool | center | +2.059 | 0.87 | vehicle_furniture | 2.367 | False |
| glm4 | vehicle_clothing | late | +1.168 | 0.24 | clothing_tool | 4.808 | False |
| glm4 | furniture_tool | center | +0.104 | 0.10 | vehicle_furniture | 1.059 | False |
| deepseek7b | vehicle_furniture | late | +0.034 | 0.05 | vehicle_clothing | 0.680 | False |
| deepseek7b | clothing_tool | center | -0.006 | 0.08 | furniture_tool | 0.070 | False |
| deepseek7b | furniture_clothing | late | +0.008 | 0.11 | vehicle_furniture | 0.072 | False |
| deepseek7b | vehicle_tool | extended | +0.430 | 0.50 | vehicle_clothing | 0.867 | False |
| deepseek7b | vehicle_clothing | extended | +0.434 | 0.59 | vehicle_furniture | 0.734 | False |
| deepseek7b | furniture_tool | late | -0.004 | 0.05 | vehicle_clothing | 0.080 | False |

