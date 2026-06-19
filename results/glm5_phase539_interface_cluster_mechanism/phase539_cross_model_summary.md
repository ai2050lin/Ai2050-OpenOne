# Phase539 Interface Cluster Mechanism Summary

## qwen3

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], targets=10, windows={'center': [10, 12, 14], 'extended': [10, 12, 14, 16]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0], seeds=2, attn=sdpa

Cell format: self min / self mean / off max abs / specificity / top off-pair.

| source | condition | best win | response |
|---|---|---|---:|
| vehicle_furniture | residual_full | extended | +1.016/+1.398/1.711/0.59/vehicle_clothing |
| vehicle_furniture | residual_perp | extended | +1.000/+1.319/1.594/0.63/vehicle_clothing |
| vehicle_furniture | residual_parallel | extended | +0.484/+1.276/1.805/0.27/vehicle_tool |
| vehicle_furniture | attention_perp | extended | +1.000/+1.319/1.492/0.67/vehicle_clothing |
| vehicle_furniture | mlp_perp | extended | +1.000/+1.331/1.602/0.62/vehicle_clothing |
| vehicle_tool | residual_full | extended | +1.379/+1.436/1.469/0.94/vehicle_furniture |
| vehicle_tool | residual_perp | extended | +1.281/+1.322/1.367/0.94/vehicle_furniture |
| vehicle_tool | residual_parallel | extended | +0.719/+1.260/1.742/0.41/vehicle_furniture |
| vehicle_tool | attention_perp | extended | +1.191/+1.298/1.320/0.90/vehicle_furniture |
| vehicle_tool | mlp_perp | extended | +1.273/+1.328/1.367/0.93/vehicle_furniture |
| vehicle_clothing | residual_full | extended | +0.992/+1.312/1.852/0.54/vehicle_furniture |
| vehicle_clothing | residual_perp | extended | +0.930/+1.208/1.711/0.54/vehicle_furniture |
| vehicle_clothing | residual_parallel | extended | +0.547/+1.353/1.902/0.29/vehicle_furniture |
| vehicle_clothing | attention_perp | extended | +0.977/+1.250/1.609/0.61/vehicle_furniture |
| vehicle_clothing | mlp_perp | extended | +0.938/+1.206/1.711/0.55/vehicle_furniture |

### Core Cosines

#### center

Common-perp cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.657 | +0.750 |
| vehicle_tool | +0.657 | +1.000 | +0.617 |
| vehicle_clothing | +0.750 | +0.617 | +1.000 |

Readout cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.795 | +0.831 |
| vehicle_tool | +0.795 | +1.000 | +0.755 |
| vehicle_clothing | +0.831 | +0.755 | +1.000 |

#### extended

Common-perp cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.657 | +0.755 |
| vehicle_tool | +0.657 | +1.000 | +0.608 |
| vehicle_clothing | +0.755 | +0.608 | +1.000 |

Readout cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.795 | +0.831 |
| vehicle_tool | +0.795 | +1.000 | +0.755 |
| vehicle_clothing | +0.831 | +0.755 | +1.000 |

### Key Edge Snapshots

| source | strongest condition | win | vehicle_furniture | vehicle_tool | vehicle_clothing | clothing_tool | fruit_tool | animal_tool |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vehicle_furniture | residual_full | extended | +1.398/1.961 | +1.224/1.445 | +1.284/1.711 | -0.115/0.891 | +0.227/0.578 | -0.030/0.320 |
| vehicle_tool | residual_full | extended | +1.339/1.469 | +1.436/1.469 | +1.289/1.469 | +0.034/0.688 | +0.116/0.664 | +0.229/0.488 |
| vehicle_clothing | residual_full | extended | +1.365/1.852 | +0.979/1.203 | +1.312/1.742 | -0.546/1.434 | -0.237/0.828 | -0.533/0.758 |

## glm4

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], targets=10, windows={'center': [24, 26, 28], 'extended': [24, 26, 28, 30]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0], seeds=2, attn=sdpa

Cell format: self min / self mean / off max abs / specificity / top off-pair.

| source | condition | best win | response |
|---|---|---|---:|
| vehicle_furniture | residual_full | extended | +2.301/+3.042/4.571/0.50/vehicle_tool |
| vehicle_furniture | residual_perp | extended | +0.989/+1.934/3.576/0.28/vehicle_tool |
| vehicle_furniture | residual_parallel | center | +2.934/+4.629/6.452/0.45/vehicle_clothing |
| vehicle_furniture | attention_perp | extended | +2.065/+2.590/4.217/0.49/vehicle_tool |
| vehicle_furniture | mlp_perp | extended | +0.991/+1.946/3.568/0.28/vehicle_tool |
| vehicle_tool | residual_full | center | +2.898/+3.253/3.553/0.82/vehicle_clothing |
| vehicle_tool | residual_perp | center | +2.059/+2.187/2.367/0.87/vehicle_furniture |
| vehicle_tool | residual_parallel | extended | +6.687/+7.363/6.032/1.11/vehicle_clothing |
| vehicle_tool | attention_perp | center | +2.189/+2.384/3.023/0.72/vehicle_furniture |
| vehicle_tool | mlp_perp | center | +2.061/+2.181/2.355/0.87/vehicle_furniture |
| vehicle_clothing | residual_full | extended | +1.982/+3.496/5.858/0.34/clothing_tool |
| vehicle_clothing | residual_perp | extended | +1.135/+2.314/5.429/0.21/clothing_tool |
| vehicle_clothing | residual_parallel | extended | +4.361/+6.265/5.497/0.79/clothing_tool |
| vehicle_clothing | attention_perp | extended | +1.451/+2.508/6.086/0.24/clothing_tool |
| vehicle_clothing | mlp_perp | extended | +1.115/+2.302/5.434/0.21/clothing_tool |

### Core Cosines

#### center

Common-perp cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.432 | +0.562 |
| vehicle_tool | +0.432 | +1.000 | +0.430 |
| vehicle_clothing | +0.562 | +0.430 | +1.000 |

Readout cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.766 | +0.801 |
| vehicle_tool | +0.766 | +1.000 | +0.729 |
| vehicle_clothing | +0.801 | +0.729 | +1.000 |

#### extended

Common-perp cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.425 | +0.553 |
| vehicle_tool | +0.425 | +1.000 | +0.413 |
| vehicle_clothing | +0.553 | +0.413 | +1.000 |

Readout cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.766 | +0.801 |
| vehicle_tool | +0.766 | +1.000 | +0.729 |
| vehicle_clothing | +0.801 | +0.729 | +1.000 |

### Key Edge Snapshots

| source | strongest condition | win | vehicle_furniture | vehicle_tool | vehicle_clothing | clothing_tool | fruit_tool | animal_tool |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vehicle_furniture | residual_parallel | center | +4.629/5.677 | +3.371/4.392 | +5.430/6.452 | +0.742/1.301 | -0.348/1.163 | +0.014/0.418 |
| vehicle_tool | residual_parallel | extended | +3.492/4.426 | +7.363/8.196 | +5.252/6.032 | +3.319/4.524 | +3.693/5.166 | +2.836/4.921 |
| vehicle_clothing | residual_parallel | extended | +2.667/3.508 | +2.873/3.356 | +6.265/7.496 | -4.343/5.497 | -0.237/0.815 | -0.661/0.970 |

## deepseek7b

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], targets=10, windows={'center': [16, 18, 20], 'extended': [16, 18, 20, 22]}, train_n=12, test_n=8, alphas=[2.0, 4.0, 6.0], seeds=2, attn=sdpa

Cell format: self min / self mean / off max abs / specificity / top off-pair.

| source | condition | best win | response |
|---|---|---|---:|
| vehicle_furniture | residual_full | extended | +0.016/+0.257/0.688/0.02/vehicle_clothing |
| vehicle_furniture | residual_perp | extended | +0.016/+0.251/0.684/0.02/vehicle_clothing |
| vehicle_furniture | residual_parallel | extended | +0.713/+0.783/0.797/0.89/vehicle_clothing |
| vehicle_furniture | attention_perp | extended | +0.030/+0.281/0.691/0.04/vehicle_clothing |
| vehicle_furniture | mlp_perp | extended | +0.007/+0.237/0.660/0.01/vehicle_clothing |
| vehicle_tool | residual_full | extended | +0.352/+0.468/0.688/0.51/vehicle_clothing |
| vehicle_tool | residual_perp | extended | +0.311/+0.421/0.633/0.49/vehicle_clothing |
| vehicle_tool | residual_parallel | extended | +1.321/+1.419/1.473/0.90/vehicle_clothing |
| vehicle_tool | attention_perp | extended | +0.370/+0.461/0.609/0.61/vehicle_clothing |
| vehicle_tool | mlp_perp | extended | +0.316/+0.430/0.656/0.48/vehicle_clothing |
| vehicle_clothing | residual_full | extended | +0.326/+0.471/0.602/0.54/vehicle_furniture |
| vehicle_clothing | residual_perp | extended | +0.305/+0.449/0.594/0.51/vehicle_furniture |
| vehicle_clothing | residual_parallel | extended | +0.715/+0.848/0.876/0.82/vehicle_tool |
| vehicle_clothing | attention_perp | extended | +0.351/+0.495/0.605/0.58/vehicle_furniture |
| vehicle_clothing | mlp_perp | extended | +0.318/+0.452/0.594/0.54/vehicle_furniture |

### Core Cosines

#### center

Common-perp cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.722 | +0.747 |
| vehicle_tool | +0.722 | +1.000 | +0.651 |
| vehicle_clothing | +0.747 | +0.651 | +1.000 |

Readout cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.742 | +0.771 |
| vehicle_tool | +0.742 | +1.000 | +0.705 |
| vehicle_clothing | +0.771 | +0.705 | +1.000 |

#### extended

Common-perp cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.716 | +0.744 |
| vehicle_tool | +0.716 | +1.000 | +0.645 |
| vehicle_clothing | +0.744 | +0.645 | +1.000 |

Readout cosine:
| pair | vehicle_furniture | vehicle_tool | vehicle_clothing |
|---|---:|---:|---:|
| vehicle_furniture | +1.000 | +0.742 | +0.771 |
| vehicle_tool | +0.742 | +1.000 | +0.705 |
| vehicle_clothing | +0.771 | +0.705 | +1.000 |

### Key Edge Snapshots

| source | strongest condition | win | vehicle_furniture | vehicle_tool | vehicle_clothing | clothing_tool | fruit_tool | animal_tool |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vehicle_furniture | residual_parallel | extended | +0.783/0.846 | +0.632/0.766 | +0.662/0.797 | +0.142/0.180 | +0.155/0.176 | +0.106/0.211 |
| vehicle_tool | residual_parallel | extended | +1.146/1.426 | +1.419/1.473 | +1.198/1.473 | +0.427/0.492 | +0.413/0.502 | +0.623/0.751 |
| vehicle_clothing | residual_parallel | extended | +0.731/0.867 | +0.752/0.876 | +0.848/0.915 | -0.078/0.156 | +0.205/0.320 | +0.123/0.203 |

## Cross-model Compact

| model | source | strongest condition | win | self min | off max | specificity | top off |
|---|---|---|---|---:|---:|---:|---|
| qwen3 | vehicle_furniture | residual_full | extended | +1.016 | 1.711 | 0.59 | vehicle_clothing |
| qwen3 | vehicle_tool | residual_full | extended | +1.379 | 1.469 | 0.94 | vehicle_furniture |
| qwen3 | vehicle_clothing | residual_full | extended | +0.992 | 1.852 | 0.54 | vehicle_furniture |
| glm4 | vehicle_furniture | residual_parallel | center | +2.934 | 6.452 | 0.45 | vehicle_clothing |
| glm4 | vehicle_tool | residual_parallel | extended | +6.687 | 6.032 | 1.11 | vehicle_clothing |
| glm4 | vehicle_clothing | residual_parallel | extended | +4.361 | 5.497 | 0.79 | clothing_tool |
| deepseek7b | vehicle_furniture | residual_parallel | extended | +0.713 | 0.797 | 0.89 | vehicle_clothing |
| deepseek7b | vehicle_tool | residual_parallel | extended | +1.321 | 1.473 | 0.90 | vehicle_clothing |
| deepseek7b | vehicle_clothing | residual_parallel | extended | +0.715 | 0.876 | 0.82 | vehicle_tool |

