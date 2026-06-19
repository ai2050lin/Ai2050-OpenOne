# Phase542 Generation Closure Audit Summary

## qwen3

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], windows={'center': [10, 12, 14], 'extended': [10, 12, 14, 16]}, train_n=12, test_n=8, max_new_tokens=5, alpha=6.0, attn=sdpa

| source | condition | best win | generation metrics | class |
|---|---|---|---|---|
| vehicle_furniture | baseline | center | hit target 0.46, comp 0.00, cluster 0.04, off 0.00, first target 0.21, first comp 0.00, rankT 187.8 | baseline |
| vehicle_furniture | residual_perp | center | hit target 0.50, comp 0.00, cluster 0.04, off 0.00, first target 0.21, first comp 0.00, rankT 142.8 | no_closure |
| vehicle_furniture | residual_parallel | center | hit target 0.54, comp 0.00, cluster 0.04, off 0.00, first target 0.21, first comp 0.00, rankT 91.6 | no_closure |
| vehicle_furniture | residual_full | extended | hit target 0.54, comp 0.00, cluster 0.00, off 0.00, first target 0.25, first comp 0.00, rankT 129.8 | first_step_only |
| vehicle_tool | baseline | center | hit target 0.46, comp 0.04, cluster 0.00, off 0.00, first target 0.21, first comp 0.00, rankT 187.8 | baseline |
| vehicle_tool | residual_perp | center | hit target 0.38, comp 0.04, cluster 0.00, off 0.00, first target 0.21, first comp 0.00, rankT 138.4 | no_closure |
| vehicle_tool | residual_parallel | extended | hit target 0.50, comp 0.04, cluster 0.00, off 0.00, first target 0.21, first comp 0.00, rankT 84.9 | no_closure |
| vehicle_tool | residual_full | extended | hit target 0.42, comp 0.00, cluster 0.00, off 0.00, first target 0.21, first comp 0.00, rankT 112.0 | no_closure |
| vehicle_clothing | baseline | center | hit target 0.46, comp 0.00, cluster 0.04, off 0.00, first target 0.21, first comp 0.00, rankT 187.8 | baseline |
| vehicle_clothing | residual_perp | center | hit target 0.50, comp 0.00, cluster 0.04, off 0.00, first target 0.21, first comp 0.00, rankT 145.0 | no_closure |
| vehicle_clothing | residual_parallel | extended | hit target 0.58, comp 0.00, cluster 0.00, off 0.00, first target 0.25, first comp 0.00, rankT 80.2 | no_closure |
| vehicle_clothing | residual_full | center | hit target 0.50, comp 0.00, cluster 0.04, off 0.00, first target 0.21, first comp 0.00, rankT 136.6 | no_closure |

## glm4

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], windows={'center': [24, 26, 28], 'extended': [24, 26, 28, 30]}, train_n=12, test_n=8, max_new_tokens=5, alpha=6.0, attn=sdpa

| source | condition | best win | generation metrics | class |
|---|---|---|---|---|
| vehicle_furniture | baseline | center | hit target 0.33, comp 0.00, cluster 0.00, off 0.04, first target 0.25, first comp 0.00, rankT 982.3 | baseline |
| vehicle_furniture | residual_perp | extended | hit target 0.46, comp 0.00, cluster 0.00, off 0.00, first target 0.25, first comp 0.00, rankT 145.1 | no_closure |
| vehicle_furniture | residual_parallel | extended | hit target 0.75, comp 0.00, cluster 0.00, off 0.00, first target 0.33, first comp 0.00, rankT 79.5 | generation_closure_positive |
| vehicle_furniture | residual_full | extended | hit target 0.58, comp 0.00, cluster 0.00, off 0.00, first target 0.29, first comp 0.00, rankT 71.2 | generation_closure_positive |
| vehicle_tool | baseline | center | hit target 0.33, comp 0.00, cluster 0.00, off 0.04, first target 0.25, first comp 0.00, rankT 982.3 | baseline |
| vehicle_tool | residual_perp | center | hit target 0.21, comp 0.00, cluster 0.00, off 0.00, first target 0.17, first comp 0.00, rankT 151.8 | no_closure |
| vehicle_tool | residual_parallel | center | hit target 0.88, comp 0.00, cluster 0.00, off 0.00, first target 0.33, first comp 0.00, rankT 20.0 | generation_closure_positive |
| vehicle_tool | residual_full | center | hit target 0.29, comp 0.00, cluster 0.00, off 0.00, first target 0.21, first comp 0.00, rankT 77.8 | no_closure |
| vehicle_clothing | baseline | center | hit target 0.33, comp 0.00, cluster 0.00, off 0.04, first target 0.25, first comp 0.00, rankT 982.3 | baseline |
| vehicle_clothing | residual_perp | center | hit target 0.46, comp 0.00, cluster 0.00, off 0.00, first target 0.21, first comp 0.00, rankT 230.5 | no_closure |
| vehicle_clothing | residual_parallel | extended | hit target 0.71, comp 0.00, cluster 0.00, off 0.00, first target 0.25, first comp 0.00, rankT 72.5 | generation_closure_positive |
| vehicle_clothing | residual_full | extended | hit target 0.50, comp 0.00, cluster 0.00, off 0.00, first target 0.29, first comp 0.00, rankT 134.0 | no_closure |

## deepseek7b

core=['vehicle_furniture', 'vehicle_tool', 'vehicle_clothing'], windows={'center': [16, 18, 20], 'extended': [16, 18, 20, 22]}, train_n=12, test_n=8, max_new_tokens=5, alpha=6.0, attn=sdpa

| source | condition | best win | generation metrics | class |
|---|---|---|---|---|
| vehicle_furniture | baseline | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 36004.1 | baseline |
| vehicle_furniture | residual_perp | center | hit target 0.04, comp 0.00, cluster 0.00, off 0.04, first target 0.04, first comp 0.00, rankT 34142.2 | no_closure |
| vehicle_furniture | residual_parallel | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 32317.8 | no_closure |
| vehicle_furniture | residual_full | center | hit target 0.04, comp 0.00, cluster 0.00, off 0.04, first target 0.04, first comp 0.00, rankT 34245.8 | no_closure |
| vehicle_tool | baseline | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 36004.1 | baseline |
| vehicle_tool | residual_perp | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 33972.5 | no_closure |
| vehicle_tool | residual_parallel | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 26012.6 | no_closure |
| vehicle_tool | residual_full | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 33747.0 | no_closure |
| vehicle_clothing | baseline | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 36004.1 | baseline |
| vehicle_clothing | residual_perp | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 33205.7 | no_closure |
| vehicle_clothing | residual_parallel | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 31161.9 | no_closure |
| vehicle_clothing | residual_full | center | hit target 0.08, comp 0.00, cluster 0.00, off 0.04, first target 0.08, first comp 0.00, rankT 33121.5 | no_closure |

## Intervention Compact

| model | source | condition | win | base target hit | target hit | gain | competitor hit | first target | class |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| qwen3 | vehicle_furniture | residual_perp | center | 0.46 | 0.50 | +0.04 | 0.00 | 0.21 | no_closure |
| qwen3 | vehicle_furniture | residual_parallel | center | 0.46 | 0.54 | +0.08 | 0.00 | 0.21 | no_closure |
| qwen3 | vehicle_furniture | residual_full | extended | 0.46 | 0.54 | +0.08 | 0.00 | 0.25 | first_step_only |
| qwen3 | vehicle_tool | residual_perp | center | 0.46 | 0.38 | -0.08 | 0.04 | 0.21 | no_closure |
| qwen3 | vehicle_tool | residual_parallel | extended | 0.46 | 0.50 | +0.04 | 0.04 | 0.21 | no_closure |
| qwen3 | vehicle_tool | residual_full | extended | 0.46 | 0.42 | -0.04 | 0.00 | 0.21 | no_closure |
| qwen3 | vehicle_clothing | residual_perp | center | 0.46 | 0.50 | +0.04 | 0.00 | 0.21 | no_closure |
| qwen3 | vehicle_clothing | residual_parallel | extended | 0.46 | 0.58 | +0.13 | 0.00 | 0.25 | no_closure |
| qwen3 | vehicle_clothing | residual_full | center | 0.46 | 0.50 | +0.04 | 0.00 | 0.21 | no_closure |
| glm4 | vehicle_furniture | residual_perp | extended | 0.33 | 0.46 | +0.12 | 0.00 | 0.25 | no_closure |
| glm4 | vehicle_furniture | residual_parallel | extended | 0.33 | 0.75 | +0.42 | 0.00 | 0.33 | generation_closure_positive |
| glm4 | vehicle_furniture | residual_full | extended | 0.33 | 0.58 | +0.25 | 0.00 | 0.29 | generation_closure_positive |
| glm4 | vehicle_tool | residual_perp | center | 0.33 | 0.21 | -0.12 | 0.00 | 0.17 | no_closure |
| glm4 | vehicle_tool | residual_parallel | center | 0.33 | 0.88 | +0.54 | 0.00 | 0.33 | generation_closure_positive |
| glm4 | vehicle_tool | residual_full | center | 0.33 | 0.29 | -0.04 | 0.00 | 0.21 | no_closure |
| glm4 | vehicle_clothing | residual_perp | center | 0.33 | 0.46 | +0.12 | 0.00 | 0.21 | no_closure |
| glm4 | vehicle_clothing | residual_parallel | extended | 0.33 | 0.71 | +0.38 | 0.00 | 0.25 | generation_closure_positive |
| glm4 | vehicle_clothing | residual_full | extended | 0.33 | 0.50 | +0.17 | 0.00 | 0.29 | no_closure |
| deepseek7b | vehicle_furniture | residual_perp | center | 0.08 | 0.04 | -0.04 | 0.00 | 0.04 | no_closure |
| deepseek7b | vehicle_furniture | residual_parallel | center | 0.08 | 0.08 | +0.00 | 0.00 | 0.08 | no_closure |
| deepseek7b | vehicle_furniture | residual_full | center | 0.08 | 0.04 | -0.04 | 0.00 | 0.04 | no_closure |
| deepseek7b | vehicle_tool | residual_perp | center | 0.08 | 0.08 | +0.00 | 0.00 | 0.08 | no_closure |
| deepseek7b | vehicle_tool | residual_parallel | center | 0.08 | 0.08 | +0.00 | 0.00 | 0.08 | no_closure |
| deepseek7b | vehicle_tool | residual_full | center | 0.08 | 0.08 | +0.00 | 0.00 | 0.08 | no_closure |
| deepseek7b | vehicle_clothing | residual_perp | center | 0.08 | 0.08 | +0.00 | 0.00 | 0.08 | no_closure |
| deepseek7b | vehicle_clothing | residual_parallel | center | 0.08 | 0.08 | +0.00 | 0.00 | 0.08 | no_closure |
| deepseek7b | vehicle_clothing | residual_full | center | 0.08 | 0.08 | +0.00 | 0.00 | 0.08 | no_closure |

